from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.routes.backtests import router as backtests_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.learning import router as learning_router
from app.api.routes.lifecycle import router as lifecycle_router
from app.api.routes.signals import router as signals_router
from app.api.routes.trades import router as trades_router
from app.backtesting.models import Trade, TradeDirection, TradeStatus
from app.database.repositories import (
    BacktestRepository,
    MLModelRepository,
    PredictionRepository,
    SignalRepository,
    TradeRepository,
)
from app.learning.service import LearningService
from app.patterns.service import PatternService

from ..conftest import requires_postgres

from datetime import datetime, timezone
from uuid import uuid4


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(signals_router, prefix="/signals")
    app.include_router(trades_router, prefix="/trades")
    app.include_router(lifecycle_router, prefix="/lifecycle")
    app.include_router(dashboard_router, prefix="/dashboard")
    app.include_router(learning_router, prefix="/learning")
    app.include_router(backtests_router, prefix="/backtests")

    app.state.signal_repository = SignalRepository()
    app.state.trade_repository = TradeRepository()
    app.state.backtest_repository = BacktestRepository()
    app.state.ml_model_repository = MLModelRepository()
    app.state.prediction_repository = PredictionRepository()

    learning = LearningService(min_samples=10000)
    service = PatternService(learning_service=learning)
    app.state.learning = learning
    app.state.pattern_service = service
    app.state.strategy_manager = service.strategy_manager
    return app


async def _client(pg_db) -> AsyncClient:
    app = _build_app()
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@requires_postgres
async def test_backtests_run_persists_and_can_be_read(pg_db):
    async with await _client(pg_db) as client:
        run = await client.post("/backtests/runs", json={"name": "api-integration"})
        assert run.status_code == 200
        payload = run.json()
        backtest_id = payload["id"]
        assert payload["trades_count"] > 0

        detail = await client.get(f"/backtests/{backtest_id}")
        assert detail.status_code == 200
        body = detail.json()
        assert body["name"] == "api-integration"
        assert body["metrics"]["total_trades"] == payload["trades_count"]
        assert body["equity_curve"]

        listing = await client.get("/backtests/")
        assert any(bt["id"] == backtest_id for bt in listing.json()["backtests"])


@requires_postgres
async def test_learning_record_via_api_persists(pg_db):
    async with await _client(pg_db) as client:
        resp = await client.post(
            "/learning/record",
            json={
                "trade": {
                    "symbol": "BTCUSDT",
                    "timeframe": "1h",
                    "pattern_name": "double_top",
                    "direction": "SHORT",
                    "pnl": 88.0,
                    "entry_price": 50000.0,
                    "stop_loss": 51000.0,
                    "take_profit": 49000.0,
                    "score": 82.0,
                },
                "indicators": {"rsi": 31.0},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["recorded"] is True

        entries = await client.get("/learning/entries", params={"instrument": "BTCUSDT"})
        assert entries.status_code == 200
        assert entries.json()["total"] == 1

        stats = await client.get("/learning/stats")
        assert stats.status_code == 200
        assert stats.json()["total_entries"] == 1
        assert stats.json()["wins"] == 1


@requires_postgres
async def test_trades_listed_from_db(pg_db):
    trade = Trade(
        id=str(uuid4()),
        symbol="EURUSD",
        timeframe="H1",
        direction=TradeDirection.LONG,
        entry_price=1.1,
        entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        stop_loss=1.09,
        take_profit=1.12,
        size=0.5,
        pnl=25.0,
        status=TradeStatus.OPEN,
        pattern_name="double_bottom",
    )
    await TradeRepository().add(trade)

    async with await _client(pg_db) as client:
        listing = await client.get("/trades/", params={"status": "OPEN"})
        assert listing.status_code == 200
        ids = [t["id"] for t in listing.json()["trades"]]
        assert trade.id in ids


@requires_postgres
async def test_dashboard_reflects_persisted_trades(pg_db):
    trade = Trade(
        id=str(uuid4()),
        symbol="GBPUSD",
        timeframe="H1",
        direction=TradeDirection.SHORT,
        entry_price=1.25,
        entry_time=datetime(2026, 2, 1, tzinfo=timezone.utc),
        stop_loss=1.26,
        take_profit=1.24,
        size=1.0,
        pnl=-10.0,
        status=TradeStatus.OPEN,
        pattern_name="double_top",
    )
    await TradeRepository().add(trade)

    async with await _client(pg_db) as client:
        overview = await client.get("/dashboard/overview")
        assert overview.status_code == 200
        assert overview.json()["open_trades"] == 1
