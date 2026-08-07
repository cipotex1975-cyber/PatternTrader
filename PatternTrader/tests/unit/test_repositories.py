from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest

from app.backtesting.models import (
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    Trade,
    TradeDirection,
    TradeStatus,
)
from app.database import base
from app.database import models as orm_models  # noqa: F401
from app.database.repositories import (
    AssetRepository,
    BacktestRepository,
    LifecycleRepository,
    LogRepository,
    MetricRepository,
    MLModelRepository,
    PredictionRepository,
    SignalRepository,
    TradeRepository,
)
from app.lifecycle.models import LifecycleEvent, LifecycleState, LifecycleTransition
from app.ml.base import MLPrediction
from app.patterns.base_pattern import PatternResult, PatternType
from app.signals.models import Signal, SignalPriority


@pytest.mark.asyncio
async def test_asset_get_or_create_is_idempotent(sync_db):
    repo = AssetRepository()
    first = await repo.get_or_create("BTCUSDT")
    second = await repo.get_or_create("BTCUSDT")
    assert first == second
    asset = await repo.get("BTCUSDT")
    assert asset is not None
    assert asset.symbol == "BTCUSDT"


def make_trade(**overrides: Any) -> Trade:
    data: dict[str, Any] = dict(
        id=str(uuid4()),
        symbol="BTCUSDT",
        timeframe="1h",
        direction=TradeDirection.LONG,
        entry_price=50000.0,
        entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        stop_loss=49500.0,
        take_profit=51000.0,
        size=1.0,
        status=TradeStatus.OPEN,
        pattern_name="double_top",
        score=80.0,
        metadata={"strategy": "trend_follow"},
    )
    data.update(overrides)
    return Trade(**data)


def make_signal(**overrides: Any) -> Signal:
    data: dict[str, Any] = dict(
        id=uuid4(),
        symbol="BTCUSDT",
        timeframe="1h",
        pattern_name="double_top",
        direction="SHORT",
        priority=SignalPriority.HIGH,
        entry_price=50000.0,
        stop_loss=51000.0,
        take_profit=48000.0,
        risk_reward_ratio=2.0,
        score=90.0,
        health=85.0,
        ml_probability=0.75,
        reasons=["Pattern detected"],
    )
    data.update(overrides)
    return Signal(**data)


def make_pattern() -> PatternResult:
    return PatternResult(
        pattern_name="double_top",
        pattern_type=PatternType.REVERSAL,
        symbol="ETHUSDT",
        timeframe="1h",
        confidence=0.85,
        health=88.0,
        score=70.0,
        entry_price=2000.0,
        stop_loss=2100.0,
        take_profit=1850.0,
        key_levels={"neckline": 1950.0},
        metadata={"source": "test"},
    )


def make_lifecycle(pattern: PatternResult) -> LifecycleEvent:
    return LifecycleEvent(
        pattern_id=pattern.id,
        symbol=pattern.symbol,
        timeframe=pattern.timeframe,
        pattern_name=pattern.pattern_name,
        current_state=LifecycleState.DETECTED,
        transitions=[
            LifecycleTransition(
                from_state=LifecycleState.DETECTED,
                to_state=LifecycleState.DETECTED,
                reason="created",
            )
        ],
    )


@pytest.mark.asyncio
async def test_trade_roundtrip_and_update_closed(sync_db):
    repo = TradeRepository()
    trade = make_trade()
    await repo.add(trade)

    loaded = await repo.get(trade.id)
    assert loaded is not None
    assert loaded.symbol == "BTCUSDT"
    assert loaded.direction == TradeDirection.LONG
    assert loaded.status == TradeStatus.OPEN
    assert loaded.metadata["strategy"] == "trend_follow"

    trade.status = TradeStatus.CLOSED
    trade.exit_price = 51000.0
    trade.exit_time = datetime(2024, 1, 1, 2, tzinfo=timezone.utc)
    trade.pnl = 500.0
    await repo.update_closed(trade)

    closed = await repo.get(trade.id)
    assert closed is not None
    assert closed.status == TradeStatus.CLOSED
    assert closed.exit_price == 51000.0
    assert closed.pnl == 500.0


@pytest.mark.asyncio
async def test_trade_list_filters(sync_db):
    repo = TradeRepository()
    await repo.add(make_trade(direction=TradeDirection.LONG, symbol="BTCUSDT"))
    await repo.add(make_trade(direction=TradeDirection.SHORT, symbol="ETHUSDT"))

    shorts = await repo.list(status=TradeStatus.OPEN, symbol="ETHUSDT")
    assert len(shorts) == 1
    assert shorts[0].direction == TradeDirection.SHORT
    assert len(await repo.list(symbol="BTCUSDT")) == 1
    assert len(await repo.list()) == 2


@pytest.mark.asyncio
async def test_signal_roundtrip_and_update_status(sync_db):
    repo = SignalRepository()
    signal = make_signal()
    await repo.add(signal)

    loaded = await repo.get(str(signal.id))
    assert loaded is not None
    assert loaded.priority == SignalPriority.HIGH
    assert loaded.score == 90.0
    assert loaded.reasons == ["Pattern detected"]

    await repo.update_status(signal.id, signal.status, sent_at=datetime.utcnow())
    updated = await repo.get(str(signal.id))
    assert updated is not None
    assert updated.sent_at is not None


@pytest.mark.asyncio
async def test_signal_list_filters(sync_db):
    repo = SignalRepository()
    await repo.add(make_signal(priority=SignalPriority.HIGH, symbol="BTCUSDT"))
    await repo.add(make_signal(priority=SignalPriority.LOW, symbol="ETHUSDT"))

    assert len(await repo.list(priority=SignalPriority.HIGH)) == 1
    assert len(await repo.list(symbol="ETHUSDT")) == 1
    assert len(await repo.list()) == 2


@pytest.mark.asyncio
async def test_lifecycle_register_and_update_transition(sync_db):
    repo = LifecycleRepository()
    pattern = make_pattern()
    lifecycle = make_lifecycle(pattern)
    await repo.register_pattern(pattern, lifecycle)

    lifecycle.add_transition(LifecycleState.FORMING, reason="structure ok")
    await repo.update_transition(lifecycle)

    pattern2 = make_pattern()
    lifecycle2 = make_lifecycle(pattern2)
    await repo.update_transition(lifecycle2)
    assert lifecycle2.current_state == LifecycleState.DETECTED


@pytest.mark.asyncio
async def test_backtest_roundtrip(sync_db):
    repo = BacktestRepository()
    result = BacktestResult(
        config=BacktestConfig(initial_capital=100000.0),
        metrics=BacktestMetrics(total_trades=5, win_rate=0.6, total_pnl=2500.0),
        trades=[make_trade()],
        equity_curve=[{"x": 0, "y": 100000.0}],
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 2, 1, tzinfo=timezone.utc),
        initial_capital=100000.0,
        final_capital=102500.0,
        metadata={"strategy": "trend_follow"},
    )
    backtest_id = await repo.add(result, name="test-run")
    assert isinstance(backtest_id, int)

    loaded = await repo.get(backtest_id)
    assert loaded is not None
    assert loaded["name"] == "test-run"
    assert loaded["result"].metrics.win_rate == 0.6
    assert loaded["result"].total_return == pytest.approx(0.025)

    listed = await repo.list()
    assert len(listed) == 1
    assert listed[0]["id"] == backtest_id


@pytest.mark.asyncio
async def test_ml_model_upsert_and_list(sync_db):
    repo = MLModelRepository()
    await repo.upsert(
        name="random_forest",
        model_type="classification",
        version="1.0",
        path="/tmp/model.pkl",
        metrics={"accuracy": 0.8},
        is_active=True,
        metadata={"trained": True},
    )
    await repo.upsert(
        name="random_forest",
        version="2.0",
        metrics={"accuracy": 0.9},
        is_active=True,
    )

    record = await repo.get("random_forest")
    assert record is not None
    assert record["version"] == "2.0"
    assert record["metrics"]["accuracy"] == 0.9
    assert record["is_active"] is True
    assert record["metadata"]["trained"] is True
    assert len(await repo.list()) == 1
    assert await repo.get("missing") is None


@pytest.mark.asyncio
async def test_prediction_add_persists(sync_db):
    repo = PredictionRepository()
    await repo.add(
        MLPrediction(
            model_name="random_forest",
            symbol="BTCUSDT",
            timeframe="1h",
            pattern_name="double_top",
            probability=0.85,
            confidence=0.7,
            features_used=["rsi", "atr"],
            metadata={"source": "test"},
        )
    )

    async with base.get_async_session() as session:
        from sqlalchemy import select

        result = await session.execute(select(orm_models.Prediction))
        orm = result.scalars().first()
    assert orm is not None
    assert orm.model_name == "random_forest"
    assert orm.probability == 0.85
    assert orm.features_used == ["rsi", "atr"]


@pytest.mark.asyncio
async def test_log_record_and_list(sync_db):
    repo = LogRepository()
    await repo.record("ERROR", "boom", source="engine", metadata={"code": 500})
    await repo.record("WARNING", "careful")
    await repo.record("INFO", "all good")

    errors = await repo.list(level="ERROR")
    assert len(errors) == 1
    assert errors[0].message == "boom"
    assert errors[0].metadata_json == {"code": 500}
    assert len(await repo.list()) == 3


@pytest.mark.asyncio
async def test_metric_record_and_list(sync_db):
    repo = MetricRepository()
    await repo.record("trades_per_second", 12.5, tags={"exchange": "binance"})
    await repo.record("latency_ms", 3.2)

    trades = await repo.list(name="trades_per_second")
    assert len(trades) == 1
    assert trades[0].value == 12.5
    assert trades[0].tags == {"exchange": "binance"}
    assert len(await repo.list()) == 2
