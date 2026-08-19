import asyncio
import time
from uuid import uuid4

import pytest

from app.backtesting.models import TradeDirection, TradeStatus
from app.core.events.bus import EventBus
from app.core.events.models import Event, EventType
from app.execution.engine import ExecutionEngine
from app.execution.models import ExitReason
from app.learning.models import LearningMode
from app.learning.repository import MemoryKnowledgeRepository
from app.learning.service import LearningService
from app.lifecycle.engine import LifecycleEngine
from app.lifecycle.models import LifecycleState
from app.patterns.base_pattern import PatternResult, PatternType

FEATURES = {
    "rsi": 70.0,
    "macd_line": 100.0,
    "macd_signal": 80.0,
    "macd_histogram": 20.0,
    "ema_21": 50500.0,
    "ema_50": 50000.0,
    "atr": 300.0,
    "volume_ratio": 1.5,
    "price_change": 0.01,
    "high_low_range": 0.02,
    "close_position": 0.6,
    "trend_strength": 0.01,
}


def signal_data(**overrides) -> dict:
    data = {
        "signal_id": f"sig-{uuid4()}",
        "pattern_id": str(uuid4()),
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "pattern_name": "double_top",
        "direction": "SHORT",
        "score": 96.0,
        "entry_price": 50000.0,
        "stop_loss": 51000.0,
        "take_profit": 48000.0,
        "risk_reward_ratio": 2.0,
        "strategy": "trend_follow",
        "size": 1.0,
        "indicators": dict(FEATURES),
    }
    data.update(overrides)
    return data


async def wait_until(condition, timeout: float = 2.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if condition():
            return True
        await asyncio.sleep(0.01)
    return False


@pytest.mark.asyncio
async def test_open_trade_publishes_event_and_tracks():
    bus = EventBus()
    collected = []

    async def handler(event: Event) -> None:
        collected.append(event)

    bus.subscribe(EventType.TRADE_OPENED, handler)
    await bus.start()
    try:
        engine = ExecutionEngine()
        engine._bus = bus

        trade = await engine.open_trade(signal_data())

        assert trade is not None
        assert trade.status == TradeStatus.OPEN
        assert trade.direction == TradeDirection.SHORT
        assert trade.pattern_name == "double_top"
        assert len(engine.get_open_trades()) == 1
        assert await wait_until(lambda: len(collected) == 1)
        assert collected[0].type == EventType.TRADE_OPENED
    finally:
        await bus.stop()
        bus.unsubscribe(EventType.TRADE_OPENED, handler)


@pytest.mark.asyncio
async def test_long_trade_closes_on_stop_loss():
    bus = EventBus()
    closed = []

    async def handler(event: Event) -> None:
        closed.append(event)

    bus.subscribe(EventType.TRADE_CLOSED, handler)
    await bus.start()
    try:
        engine = ExecutionEngine()
        engine._bus = bus

        trade = await engine.open_trade(
            signal_data(
                direction="LONG",
                entry_price=50000.0,
                stop_loss=49500.0,
                take_profit=51000.0,
            )
        )
        assert trade is not None

        await engine._on_candle(
            Event(
                type=EventType.CANDLE_UPDATED,
                source="test",
                data={
                    "symbol": "BTCUSDT",
                    "timeframe": "1h",
                    "high": 50100.0,
                    "low": 49400.0,
                    "close": 49450.0,
                },
            )
        )

        assert len(engine.get_open_trades()) == 0
        closed_trades = engine.get_closed_trades()
        assert len(closed_trades) == 1
        assert closed_trades[0].status == TradeStatus.CLOSED
        assert closed_trades[0].metadata["exit_reason"] == ExitReason.STOP_LOSS.value
        assert closed_trades[0].pnl < 0
        assert await wait_until(lambda: len(closed) == 1)
        assert closed[0].type == EventType.TRADE_CLOSED
    finally:
        await bus.stop()
        bus.unsubscribe(EventType.TRADE_CLOSED, handler)


@pytest.mark.asyncio
async def test_short_trade_closes_on_take_profit():
    engine = ExecutionEngine()
    await engine.open_trade(signal_data())

    await engine._on_candle(
        Event(
            type=EventType.CANDLE_UPDATED,
            source="test",
            data={
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "high": 50500.0,
                "low": 47900.0,
                "close": 48000.0,
            },
        )
    )

    assert len(engine.get_open_trades()) == 0
    closed = engine.get_closed_trades()[0]
    assert closed.metadata["exit_reason"] == ExitReason.TAKE_PROFIT.value
    assert closed.pnl > 0
    assert closed.exit_price == 48000.0


@pytest.mark.asyncio
async def test_close_trade_feeds_lifecycle():
    lifecycle = LifecycleEngine()
    pattern = PatternResult(
        pattern_name="double_top",
        pattern_type=PatternType.REVERSAL,
        symbol="BTCUSDT",
        timeframe="1h",
        confidence=0.85,
    )
    lifecycle_event = await lifecycle.register(pattern)
    engine = ExecutionEngine(lifecycle=lifecycle)

    trade = await engine.open_trade(signal_data(pattern_id=str(pattern.id)))

    assert lifecycle_event.current_state == LifecycleState.OPEN

    await engine.close_trade(trade.id, 51000.0, ExitReason.STOP_LOSS)

    assert lifecycle_event.current_state == LifecycleState.CLOSED
    states = [t.to_state for t in lifecycle_event.transitions]
    assert LifecycleState.OPEN in states
    assert LifecycleState.SL_HIT in states
    assert LifecycleState.CLOSED in states


@pytest.mark.asyncio
async def test_signal_sent_opens_and_dedups():
    bus = EventBus()
    opened = []

    async def handler(event: Event) -> None:
        opened.append(event)

    bus.subscribe(EventType.TRADE_OPENED, handler)
    await bus.start()
    try:
        engine = ExecutionEngine()
        engine._bus = bus
        data = signal_data()

        await engine._on_signal_sent(Event(type=EventType.SIGNAL_SENT, source="test", data=data))
        await engine._on_signal_sent(Event(type=EventType.SIGNAL_SENT, source="test", data=data))

        assert len(engine.get_open_trades()) == 1
        assert await wait_until(lambda: len(opened) == 1)
    finally:
        await bus.stop()
        bus.unsubscribe(EventType.TRADE_OPENED, handler)


@pytest.mark.asyncio
async def test_invalid_signal_does_not_open_and_cancels_lifecycle():
    lifecycle = LifecycleEngine()
    pattern = PatternResult(
        pattern_name="double_top",
        pattern_type=PatternType.REVERSAL,
        symbol="BTCUSDT",
        timeframe="1h",
        confidence=0.85,
    )
    lifecycle_event = await lifecycle.register(pattern)
    engine = ExecutionEngine(lifecycle=lifecycle)

    data = signal_data(
        pattern_id=str(pattern.id),
        entry_price=0,
        stop_loss=51000.0,
        take_profit=48000.0,
    )
    await engine._on_signal_sent(Event(type=EventType.SIGNAL_SENT, source="test", data=data))

    assert len(engine.get_open_trades()) == 0
    assert lifecycle_event.current_state == LifecycleState.CANCELLED


@pytest.mark.asyncio
async def test_learning_service_receives_closed_trade():
    bus = EventBus()
    svc = LearningService(
        repository=MemoryKnowledgeRepository(),
        mode=LearningMode.OFFLINE,
        min_samples=100,
    )
    svc._bus = bus
    await svc.start()

    engine = ExecutionEngine()
    engine._bus = bus

    await bus.start()
    try:
        trade = await engine.open_trade(signal_data())
        assert trade is not None
        await engine.close_trade(trade.id, 51000.0, ExitReason.STOP_LOSS)

        assert await wait_until(lambda: len(svc.repository._entries) >= 1)
        entries = await svc.entries()
        assert any(e.pattern == "double_top" for e in entries)
    finally:
        await bus.stop()
        await svc.stop()
