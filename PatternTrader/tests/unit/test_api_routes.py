from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.backtests import router as backtests_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.learning import router as learning_router
from app.api.routes.lifecycle import router as lifecycle_router
from app.api.routes.signals import router as signals_router
from app.api.routes.trades import router as trades_router
from app.backtesting.models import Trade, TradeDirection, TradeStatus
from app.database.repositories import BacktestRepository, SignalRepository, TradeRepository
from app.lifecycle.engine import LifecycleEngine
from app.patterns.base_pattern import PatternResult, PatternType
from app.signals.models import Signal, SignalPriority


def make_pattern(symbol: str = "BTCUSDT") -> PatternResult:
    return PatternResult(
        pattern_name="double_top",
        pattern_type=PatternType.REVERSAL,
        symbol=symbol,
        timeframe="1h",
        confidence=0.8,
    )


class FakePipeline:
    def __init__(self) -> None:
        self.lifecycle = LifecycleEngine()

    def stats(self) -> dict[str, int]:
        return {
            "tracked": len(self.lifecycle.get_all()),
            "active": len(self.lifecycle.get_active()),
            "expired": 0,
            "confirmed": 0,
            "signals_sent": 0,
        }


class FakeService:
    def __init__(self) -> None:
        self.pipeline = FakePipeline()


@pytest.fixture
def sync_api_app(sync_db):
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
    service = FakeService()
    app.state.pattern_service = service
    app.state.learning = FakeLearningService()
    app.state.prediction_repository = None
    return TestClient(app)


class FakeLearningService:
    mode = type("M", (), {"value": "OFFLINE"})()

    async def entries(self, **kwargs: Any):
        return []

    async def stats(self) -> dict:
        return {"total": 0}

    async def record_trade(self, trade, **kwargs: Any):
        return type("E", (), {"model_dump": lambda self, mode="json": {"pattern": "double_top"}})()

    async def train_offline(self, n_splits: int = 5) -> dict:
        return {"trained": True, "n_splits": n_splits}

    def predict(self, **kwargs: Any) -> type:
        class P:
            def model_dump(self, mode="json"):
                return {"model_name": "knowledge", "probability": 0.6}

        return P()

    def set_mode(self, mode) -> None:
        FakeLearningService.mode = mode


@pytest.fixture
async def seeded_signal(sync_api_app):
    repo = SignalRepository()
    signal = Signal(
        symbol="BTCUSDT",
        timeframe="1h",
        pattern_name="double_top",
        direction="SHORT",
        priority=SignalPriority.CRITICAL,
        entry_price=50000.0,
        stop_loss=51000.0,
        take_profit=48000.0,
        risk_reward_ratio=2.0,
        score=96.0,
        health=90.0,
        ml_probability=0.8,
    )
    await repo.add(signal)
    return signal


@pytest.fixture
async def seeded_trade(sync_api_app):
    repo = TradeRepository()
    trade = Trade(
        id=str(uuid4()),
        symbol="BTCUSDT",
        timeframe="1h",
        direction=TradeDirection.LONG,
        entry_price=50000.0,
        entry_time=datetime(2026, 8, 7, tzinfo=timezone.utc),
        stop_loss=49500.0,
        take_profit=51000.0,
        size=1.0,
        status=TradeStatus.OPEN,
        pattern_name="double_bottom",
    )
    await repo.add(trade)
    return trade


def test_signals_list_empty(sync_api_app):
    resp = sync_api_app.get("/signals/")
    assert resp.status_code == 200
    assert resp.json()["signals"] == []


@pytest.mark.asyncio
async def test_signals_list_and_get(sync_api_app, seeded_signal):
    resp = sync_api_app.get("/signals/")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["signals"]) == 1
    assert body["signals"][0]["pattern"] == "double_top"

    detail = sync_api_app.get(f"/signals/{seeded_signal.id}")
    assert detail.status_code == 200
    assert detail.json()["symbol"] == "BTCUSDT"


def test_signals_get_unknown_404(sync_api_app):
    assert sync_api_app.get(f"/signals/{uuid4()}").status_code == 404


def test_trades_list_empty(sync_api_app):
    resp = sync_api_app.get("/trades/")
    assert resp.status_code == 200
    assert resp.json()["trades"] == []


@pytest.mark.asyncio
async def test_trades_list_and_get(sync_api_app, seeded_trade):
    resp = sync_api_app.get("/trades/")
    assert resp.status_code == 200
    assert len(resp.json()["trades"]) == 1

    detail = sync_api_app.get(f"/trades/{seeded_trade.id}")
    assert detail.status_code == 200
    assert detail.json()["symbol"] == "BTCUSDT"


def test_trades_get_unknown_404(sync_api_app):
    assert sync_api_app.get(f"/trades/{uuid4()}").status_code == 404


@pytest.mark.asyncio
async def test_lifecycle_statistics_and_list(sync_api_app):
    await sync_api_app.app.state.pattern_service.pipeline.lifecycle.register(make_pattern())
    stats = sync_api_app.get("/lifecycle/statistics")
    assert stats.status_code == 200
    assert stats.json()["total"] == 1

    listing = sync_api_app.get("/lifecycle/")
    assert listing.status_code == 200
    assert len(listing.json()["lifecycles"]) == 1


@pytest.mark.asyncio
async def test_lifecycle_unknown_404(sync_api_app):
    assert sync_api_app.get(f"/lifecycle/{uuid4()}").status_code == 404


@pytest.mark.asyncio
async def test_dashboard_overview_and_active(sync_api_app):
    await sync_api_app.app.state.pattern_service.pipeline.lifecycle.register(make_pattern())
    overview = sync_api_app.get("/dashboard/overview")
    assert overview.status_code == 200
    body = overview.json()
    assert body["active_patterns"] == 1
    assert body["total_lifecycles"] == 1

    active = sync_api_app.get("/dashboard/active")
    assert active.status_code == 200
    assert len(active.json()["patterns"]) == 1


def test_dashboard_by_state_invalid(sync_api_app):
    resp = sync_api_app.get("/dashboard/by-state/NOPE")
    assert resp.status_code == 200
    assert "error" in resp.json()


def test_learning_entries_and_mode(sync_api_app):
    entries = sync_api_app.get("/learning/entries")
    assert entries.status_code == 200
    assert entries.json()["total"] == 0

    mode = sync_api_app.get("/learning/mode")
    assert mode.status_code == 200
    assert mode.json()["mode"] == "OFFLINE"

    changed = sync_api_app.post("/learning/mode", params={"mode": "online"})
    assert changed.status_code == 200
    assert changed.json()["mode"] == "online"


def test_backtests_run_list_get(sync_api_app):
    run = sync_api_app.post("/backtests/runs", json={})
    assert run.status_code == 200
    backtest_id = run.json()["id"]

    listing = sync_api_app.get("/backtests/")
    assert listing.status_code == 200
    assert any(bt["id"] == backtest_id for bt in listing.json()["backtests"])

    detail = sync_api_app.get(f"/backtests/{backtest_id}")
    assert detail.status_code == 200
    assert "metrics" in detail.json()

    trades = sync_api_app.get(f"/backtests/{backtest_id}/trades")
    assert trades.status_code == 200
    assert "trades" in trades.json()


def test_backtests_get_unknown_404(sync_api_app):
    assert sync_api_app.get("/backtests/9999").status_code == 404


def test_backtests_walk_forward(sync_api_app):
    resp = sync_api_app.post("/backtests/walk-forward", json={"train_size": 100, "test_size": 40})
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "walk_forward"
    assert len(body["folds"]) > 0


def test_backtests_monte_carlo(sync_api_app):
    resp = sync_api_app.post(
        "/backtests/monte-carlo",
        json={"simulations": 50, "config": {"initial_capital": 5000}},
    )
    assert resp.status_code == 200
    assert resp.json()["simulations"] == 50


def test_backtests_optimize(sync_api_app):
    resp = sync_api_app.post(
        "/backtests/optimize",
        json={"param_grid": {"risk_per_trade": [0.01, 0.02]}},
    )
    assert resp.status_code == 200
    assert "best_params" in resp.json()
