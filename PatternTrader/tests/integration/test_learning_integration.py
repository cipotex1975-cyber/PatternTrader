from datetime import datetime, timezone
from uuid import uuid4

from app.database.repositories import MLModelRepository
from app.learning.service import LearningService

from ..conftest import requires_postgres


def _trade(pnl: float, symbol: str = "BTCUSDT", pattern: str = "double_top") -> dict:
    return {
        "symbol": symbol,
        "timeframe": "1h",
        "pattern_name": pattern,
        "direction": "SHORT",
        "pnl": pnl,
        "entry_price": 50000.0,
        "stop_loss": 51000.0,
        "take_profit": 49000.0,
        "score": 80.0,
    }


@requires_postgres
async def test_record_trade_persists_entry(pg_db):
    service = LearningService(min_samples=10000)

    entry = await service.record_trade(
        _trade(pnl=120.5),
        indicators={"rsi": 35.0},
        variables={"trend": "down"},
    )
    assert entry.outcome.value == "WIN"

    entries = await service.entries(instrument="BTCUSDT", pattern="double_top")
    assert len(entries) == 1
    assert entries[0].id == entry.id
    assert entries[0].pnl == 120.5
    assert entries[0].indicators.get("rsi") == 35.0
    assert entries[0].entry_time.tzinfo is not None


@requires_postgres
async def test_stats_aggregates(pg_db):
    service = LearningService(min_samples=10000)
    await service.record_trade(_trade(pnl=100.0))
    await service.record_trade(_trade(pnl=-50.0))
    await service.record_trade(_trade(pnl=0.0))

    stats = await service.stats()
    assert stats["total_entries"] == 3
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["by_pattern"]["double_top"]["count"] == 3
    assert stats["by_pattern"]["double_top"]["pnl"] == 50.0


@requires_postgres
async def test_ml_model_upsert_roundtrip(pg_db):
    repo = MLModelRepository()

    await repo.upsert(
        name="knowledge_model",
        model_type="classification",
        version="0.2.0",
        path="models/knowledge_model.joblib",
        metrics={"accuracy": 0.72, "samples": 40},
        is_active=True,
        trained_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    item = await repo.get("knowledge_model")
    assert item is not None
    assert item["model_type"] == "classification"
    assert item["version"] == "0.2.0"
    assert item["metrics"]["accuracy"] == 0.72
    assert item["is_active"] is True

    await repo.upsert(
        name="knowledge_model",
        version="0.3.0",
        metrics={"accuracy": 0.75},
    )
    updated = await repo.get("knowledge_model")
    assert updated is not None
    assert updated["version"] == "0.3.0"
    assert updated["metrics"]["accuracy"] == 0.75


@requires_postgres
async def test_ml_model_trained_at_stored_aware(pg_db):
    repo = MLModelRepository()
    trained_at = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    await repo.upsert(name="integration_model", trained_at=trained_at)

    item = await repo.get("integration_model")
    assert item is not None
    assert item["trained_at"] == trained_at.isoformat()


@requires_postgres
async def test_register_in_db_activates_per_symbol(pg_db):
    from train_and_compare import register_in_db

    winner = {
        "model": "random_forest",
        "metrics": {"accuracy": 0.8, "roc_auc": 0.75},
    }
    await register_in_db(
        "USDCAD", "H1", winner, "/tmp/random_forest_USDCAD.pkl", "roc_auc", promote=True
    )

    active = await MLModelRepository().get_active()
    assert any(m["name"] == "random_forest_USDCAD" and m["is_active"] for m in active)

    winner_new = {
        "model": "xgboost",
        "metrics": {"accuracy": 0.85, "roc_auc": 0.8},
    }
    await register_in_db(
        "USDCAD", "H1", winner_new, "/tmp/xgboost_USDCAD.json", "roc_auc", promote=True
    )

    active = await MLModelRepository().get_active()
    names = {m["name"] for m in active}
    assert "xgboost_USDCAD" in names
    assert "random_forest_USDCAD" not in names


@requires_postgres
async def test_register_in_db_without_promote_stays_inactive(pg_db):
    from train_and_compare import register_in_db

    winner = {
        "model": "random_forest",
        "metrics": {"accuracy": 0.8, "roc_auc": 0.75},
    }
    await register_in_db(
        "USDCAD", "H1", winner, "/tmp/random_forest_USDCAD.pkl", "roc_auc", promote=False
    )

    item = await MLModelRepository().get("random_forest_USDCAD")
    assert item is not None
    assert item["is_active"] is False

    winner_new = {
        "model": "xgboost",
        "metrics": {"accuracy": 0.85, "roc_auc": 0.8},
    }
    await register_in_db(
        "USDCAD", "H1", winner_new, "/tmp/xgboost_USDCAD.json", "roc_auc", promote=True
    )

    active = await MLModelRepository().get_active()
    names = {m["name"] for m in active}
    assert "xgboost_USDCAD" in names
    assert "random_forest_USDCAD" not in names


@requires_postgres
async def test_signal_repository_roundtrip(pg_db):
    from app.database.repositories import SignalRepository
    from app.signals.models import Signal, SignalPriority

    signal = Signal(
        symbol="BTCUSDT",
        timeframe="1h",
        pattern_name="double_top",
        direction="SHORT",
        priority=SignalPriority.HIGH,
        entry_price=49000.0,
        stop_loss=51000.0,
        take_profit=47000.0,
        risk_reward_ratio=2.0,
        score=90.0,
        health=95.0,
        ml_probability=0.8,
    )
    repo = SignalRepository()
    await repo.add(signal)

    stored = await repo.get(str(signal.id))
    assert stored is not None
    assert stored.symbol == "BTCUSDT"
    assert stored.priority == SignalPriority.HIGH
    assert stored.entry_price == 49000.0

    listed = await repo.list(symbol="BTCUSDT")
    assert any(s.id == signal.id for s in listed)


@requires_postgres
async def test_trade_repository_roundtrip(pg_db):
    from app.backtesting.models import Trade, TradeDirection, TradeStatus
    from app.database.repositories import TradeRepository

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
    repo = TradeRepository()
    await repo.add(trade)

    stored = await repo.get(trade.id)
    assert stored is not None
    assert stored.symbol == "EURUSD"
    assert stored.status == TradeStatus.OPEN
    assert stored.entry_time == trade.entry_time

    open_trades = await repo.list(status=TradeStatus.OPEN)
    assert any(t.id == trade.id for t in open_trades)
