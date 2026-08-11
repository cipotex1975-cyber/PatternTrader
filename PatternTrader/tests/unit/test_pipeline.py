from datetime import datetime, timedelta, timezone

import asyncio

import pytest

from app.core.events.bus import EventBus
from app.core.events.models import EventType
from app.lifecycle.models import LifecycleState
from app.market.candles.models import Candle, CandleData
from app.ml.features import TECHNICAL_FEATURE_NAMES
from app.patterns.base_pattern import PatternResult, PatternType, TradeDirection
from app.patterns import pipeline as pipeline_module
from app.patterns.pipeline import PatternPipeline
from app.risk.models import PositionSize, RiskAssessment
from app.signals.models import Signal, SignalPriority


def build_candles(closes, volumes=None):
    volumes = volumes or [1000] * len(closes)
    candles = []
    prev = closes[0]
    base_ts = datetime.now(timezone.utc)
    for i, (close, volume) in enumerate(zip(closes, volumes)):
        candles.append(
            Candle(
                symbol="BTCUSDT",
                timeframe="1h",
                data=CandleData(
                    timestamp=base_ts + timedelta(hours=i),
                    open=prev,
                    high=max(prev, close) * 1.002,
                    low=min(prev, close) * 0.998,
                    close=close,
                    volume=volume,
                ),
            )
        )
        prev = close
    return candles


def double_top_closes():
    closes = [47000 + i * 300 for i in range(10)]
    closes += [49400, 49100, 48800]
    closes += [49100, 49400, 49700, 49950, 49700, 49400]
    closes += [48900, 48700, 48500, 48300, 48100]
    return closes


def build_double_top_candles():
    closes = double_top_closes()
    volumes = [1000] * len(closes)
    for i in range(-5, 0):
        volumes[i] = 2000
    return build_candles(closes, volumes)


@pytest.mark.asyncio
async def test_pipeline_detects_and_tracks():
    candles = build_double_top_candles()
    pipeline = PatternPipeline(data_source=lambda symbol, timeframe: candles)

    stats = await pipeline.process_symbol("BTCUSDT", "1h")

    assert stats["tracked"] == 1
    tracked = list(pipeline.tracked.values())[0]
    assert tracked.result.pattern_name == "double_top"
    assert tracked.result.direction.value == "SHORT"


@pytest.mark.asyncio
async def test_pipeline_applies_health():
    candles = build_double_top_candles()
    pipeline = PatternPipeline(data_source=lambda symbol, timeframe: candles)

    await pipeline.process_symbol("BTCUSDT", "1h")

    tracked = list(pipeline.tracked.values())[0]
    assert 0 <= tracked.result.health <= 100
    assert "health_report" in tracked.result.metadata


@pytest.mark.asyncio
async def test_pipeline_expires_without_breakout():
    holder = {"candles": build_double_top_candles()}
    pipeline = PatternPipeline(data_source=lambda symbol, timeframe: holder["candles"])

    await pipeline.process_symbol("BTCUSDT", "1h")

    holder["candles"] = build_candles([51000] * 30)

    for _ in range(25):
        await pipeline.process_symbol("BTCUSDT", "1h")

    stats = pipeline.stats()
    assert stats["expired"] >= 1
    assert stats["active"] == 0


@pytest.mark.asyncio
async def test_pipeline_confirms_and_signals():
    candles = build_double_top_candles()
    pipeline = PatternPipeline(data_source=lambda symbol, timeframe: candles)

    for _ in range(4):
        await pipeline.process_symbol("BTCUSDT", "1h")

    stats = pipeline.stats()
    assert stats["confirmed"] >= 1

    tracked = list(pipeline.tracked.values())[0]
    assert tracked.result.risk_reward_ratio is not None
    assert tracked.result.risk_reward_ratio >= 2.0
    assert "strategy_decisions" in tracked.result.metadata


@pytest.mark.asyncio
async def test_pipeline_sends_telegram_for_critical():
    candles = build_double_top_candles()
    pipeline = PatternPipeline(
        data_source=lambda symbol, timeframe: candles,
        strategy_params={"breakout": {"rsi_min": 25.0, "rsi_max": 80.0}},
    )

    sent = []

    async def fake_send(signal, candles=None, pattern=None):
        sent.append(signal)
        return True

    async def fake_create(pattern, score_result, ml_probability=0.0, strategy_signal=None):
        signal = Signal(
            symbol=pattern.symbol,
            timeframe=pattern.timeframe,
            pattern_name=pattern.pattern_name,
            direction=pattern.direction.value,
            priority=SignalPriority.CRITICAL,
            entry_price=pattern.entry_price or 0,
            stop_loss=pattern.stop_loss or 0,
            take_profit=pattern.take_profit or 0,
            risk_reward_ratio=2.0,
            score=96.0,
            health=pattern.health,
            ml_probability=0.95,
        )
        pipeline._signal_engine._signals[signal.id] = signal
        return signal

    pipeline._telegram.send_signal = fake_send
    pipeline._signal_engine.create_signal = fake_create

    for _ in range(4):
        await pipeline.process_symbol("BTCUSDT", "1h")

    assert len(sent) == 1
    assert sent[0].priority == SignalPriority.CRITICAL


async def _await_events(collected, timeout: float = 2.0) -> None:
    start = asyncio.get_event_loop().time()
    while len(collected) == 0:
        if asyncio.get_event_loop().time() - start > timeout:
            return
        await asyncio.sleep(0.01)


def _register_pipeline_handlers(event_type):
    original_get_event_bus = pipeline_module.get_event_bus
    bus = EventBus()
    pipeline_module.get_event_bus = lambda: bus
    collected = []

    async def handler(event):
        collected.append(event)

    bus.subscribe(event_type, handler)

    def restore():
        pipeline_module.get_event_bus = original_get_event_bus

    return bus, collected, restore


@pytest.mark.asyncio
async def test_pipeline_publishes_candle_update():
    bus, collected, restore = _register_pipeline_handlers(EventType.CANDLE_UPDATED)
    await bus.start()
    try:
        candles = build_double_top_candles()
        pipeline = PatternPipeline(data_source=lambda symbol, timeframe: candles)
        await pipeline.process_symbol("BTCUSDT", "1h")

        await _await_events(collected)
        assert len(collected) == 1
        data = collected[0].data
        assert data["symbol"] == "BTCUSDT"
        assert data["timeframe"] == "1h"
        for key in ("timestamp", "open", "high", "low", "close", "volume"):
            assert key in data
    finally:
        await bus.stop()
        restore()


@pytest.mark.asyncio
async def test_pipeline_signal_sent_event_is_enriched():
    bus, collected, restore = _register_pipeline_handlers(EventType.SIGNAL_SENT)
    await bus.start()
    try:
        candles = build_double_top_candles()
        pipeline = PatternPipeline(
            data_source=lambda symbol, timeframe: candles,
            strategy_params={"breakout": {"rsi_min": 25.0, "rsi_max": 80.0}},
        )

        async def fake_send(signal, candles=None, pattern=None):
            return True

        async def fake_create(pattern, score_result, ml_probability=0.0, strategy_signal=None):
            signal = Signal(
                symbol=pattern.symbol,
                timeframe=pattern.timeframe,
                pattern_name=pattern.pattern_name,
                direction=pattern.direction.value,
                priority=SignalPriority.CRITICAL,
                entry_price=pattern.entry_price or 0,
                stop_loss=pattern.stop_loss or 0,
                take_profit=pattern.take_profit or 0,
                risk_reward_ratio=2.0,
                score=96.0,
                health=pattern.health,
                ml_probability=0.95,
                metadata={"strategy": "breakout", "strategy_size": 0.5},
            )
            pipeline._signal_engine._signals[signal.id] = signal
            return signal

        pipeline._telegram.send_signal = fake_send
        pipeline._signal_engine.create_signal = fake_create

        for _ in range(4):
            await pipeline.process_symbol("BTCUSDT", "1h")

        await _await_events(collected)
        assert len(collected) == 1
        data = collected[0].data
        assert data["signal_id"]
        assert data["pattern_id"]
        assert data["direction"] == "SHORT"
        assert data["strategy"] == "breakout"
        assert data["size"] == 0.5
        assert data["entry_price"] > 0
        assert data["stop_loss"] > 0
        assert data["take_profit"] > 0
        indicators = data["indicators"]
        assert set(TECHNICAL_FEATURE_NAMES).issubset(indicators.keys())
    finally:
        await bus.stop()
        restore()


@pytest.mark.asyncio
async def test_pipeline_rejects_signal_when_risk_unacceptable():
    candles = build_double_top_candles()
    pipeline = PatternPipeline(
        data_source=lambda symbol, timeframe: candles,
        strategy_params={"breakout": {"rsi_min": 25.0, "rsi_max": 80.0}},
    )

    def rejected_assess(pattern, entry, stop, tp):
        return RiskAssessment(
            symbol=pattern.symbol,
            timeframe=pattern.timeframe,
            pattern_name=pattern.pattern_name,
            entry_price=entry,
            stop_loss=stop,
            take_profit=tp,
            position_size=PositionSize(
                symbol=pattern.symbol,
                direction=pattern.direction.value,
                entry_price=entry,
                stop_loss=stop,
                take_profit=tp,
                size=0.0,
                risk_amount=0.0,
                risk_pct=50.0,
                potential_reward=0.0,
                risk_reward_ratio=2.0,
                max_loss=0.0,
            ),
            is_acceptable=False,
            risk_score=80.0,
            warnings=["max_risk_per_trade exceeded"],
        )

    pipeline._risk.assess = rejected_assess

    async def fake_send(signal, candles=None, pattern=None):
        return True

    async def fake_create(pattern, score_result, ml_probability=0.0, strategy_signal=None):
        return Signal(
            symbol=pattern.symbol,
            timeframe=pattern.timeframe,
            pattern_name=pattern.pattern_name,
            direction=pattern.direction.value,
            priority=SignalPriority.CRITICAL,
            entry_price=pattern.entry_price or 0,
            stop_loss=pattern.stop_loss or 0,
            take_profit=pattern.take_profit or 0,
            risk_reward_ratio=2.0,
            score=96.0,
            health=pattern.health,
            ml_probability=0.95,
        )

    pipeline._telegram.send_signal = fake_send
    pipeline._signal_engine.create_signal = fake_create

    for _ in range(4):
        await pipeline.process_symbol("BTCUSDT", "1h")

    rejected = pipeline.lifecycle.get_by_state(LifecycleState.REJECTED)
    assert len(rejected) >= 1


@pytest.mark.asyncio
async def test_pipeline_cancels_pending_signal_on_deformation():
    holder = {"candles": build_double_top_candles()}
    pipeline = PatternPipeline(
        data_source=lambda symbol, timeframe: holder["candles"],
        strategy_params={"breakout": {"rsi_min": 25.0, "rsi_max": 80.0}},
    )

    async def fake_send(signal, candles=None, pattern=None):
        return True

    async def fake_create(pattern, score_result, ml_probability=0.0, strategy_signal=None):
        return Signal(
            symbol=pattern.symbol,
            timeframe=pattern.timeframe,
            pattern_name=pattern.pattern_name,
            direction=pattern.direction.value,
            priority=SignalPriority.CRITICAL,
            entry_price=pattern.entry_price or 0,
            stop_loss=pattern.stop_loss or 0,
            take_profit=pattern.take_profit or 0,
            risk_reward_ratio=2.0,
            score=96.0,
            health=pattern.health,
            ml_probability=0.95,
        )

    pipeline._telegram.send_signal = fake_send
    pipeline._signal_engine.create_signal = fake_create

    for _ in range(4):
        await pipeline.process_symbol("BTCUSDT", "1h")

    assert len(pipeline.lifecycle.get_by_state(LifecycleState.SIGNAL_SENT)) >= 1

    holder["candles"] = build_candles([45000] * 30)
    await pipeline.process_symbol("BTCUSDT", "1h")

    cancelled = pipeline.lifecycle.get_by_state(LifecycleState.CANCELLED)
    assert len(cancelled) >= 1


def test_prepare_price_levels_long_continuation():
    pattern = PatternResult(
        pattern_name="bull_flag",
        pattern_type=PatternType.CONTINUATION,
        symbol="BTCUSDT",
        timeframe="1h",
        direction=TradeDirection.LONG,
        confidence=0.8,
        key_levels={"pole_high": 50000, "flag_low": 49000, "target": 52000},
    )
    pipeline = PatternPipeline(data_source=lambda symbol, timeframe: [])

    pipeline._prepare_price_levels(pattern)

    assert pattern.entry_price == 50000
    assert pattern.stop_loss is not None and pattern.stop_loss < 50000
    assert pattern.take_profit is not None and pattern.take_profit > 50000
    assert pattern.risk_reward_ratio is not None


def test_prepare_price_levels_short_continuation():
    pattern = PatternResult(
        pattern_name="bear_flag",
        pattern_type=PatternType.CONTINUATION,
        symbol="BTCUSDT",
        timeframe="1h",
        direction=TradeDirection.SHORT,
        confidence=0.8,
        key_levels={"pole_low": 50000, "flag_high": 51000, "target": 48000},
    )
    pipeline = PatternPipeline(data_source=lambda symbol, timeframe: [])

    pipeline._prepare_price_levels(pattern)

    assert pattern.entry_price == 50000
    assert pattern.stop_loss is not None and pattern.stop_loss > 50000
    assert pattern.take_profit is not None and pattern.take_profit < 50000
    assert pattern.risk_reward_ratio is not None


def test_prepare_price_levels_bull_pennant():
    pattern = PatternResult(
        pattern_name="bull_pennant",
        pattern_type=PatternType.CONTINUATION,
        symbol="BTCUSDT",
        timeframe="1h",
        direction=TradeDirection.LONG,
        confidence=0.8,
        key_levels={"pole_high": 50000, "pennant_low": 49000, "target": 52000},
    )
    pipeline = PatternPipeline(data_source=lambda symbol, timeframe: [])

    pipeline._prepare_price_levels(pattern)

    assert pattern.entry_price == 50000
    assert pattern.stop_loss is not None and pattern.stop_loss < 50000


def test_prepare_price_levels_bear_pennant():
    pattern = PatternResult(
        pattern_name="bear_pennant",
        pattern_type=PatternType.CONTINUATION,
        symbol="BTCUSDT",
        timeframe="1h",
        direction=TradeDirection.SHORT,
        confidence=0.8,
        key_levels={"pole_low": 50000, "pennant_high": 51000, "target": 48000},
    )
    pipeline = PatternPipeline(data_source=lambda symbol, timeframe: [])

    pipeline._prepare_price_levels(pattern)

    assert pattern.entry_price == 50000
    assert pattern.stop_loss is not None and pattern.stop_loss > 50000


@pytest.mark.asyncio
async def test_pipeline_sends_when_priority_meets_min_priority():
    from app.core.config.settings import get_settings

    settings = get_settings()
    original = settings.telegram.min_priority
    settings.telegram.min_priority = "LOW"
    try:
        candles = build_double_top_candles()
        pipeline = PatternPipeline(
            data_source=lambda symbol, timeframe: candles,
            strategy_params={"breakout": {"rsi_min": 25.0, "rsi_max": 80.0}},
        )
        sent = []

        async def fake_send(signal, candles=None, pattern=None):
            sent.append(signal)
            return True

        async def fake_create(pattern, score_result, ml_probability=0.0, strategy_signal=None):
            signal = Signal(
                symbol=pattern.symbol,
                timeframe=pattern.timeframe,
                pattern_name=pattern.pattern_name,
                direction=pattern.direction.value,
                priority=SignalPriority.HIGH,
                entry_price=pattern.entry_price or 0,
                stop_loss=pattern.stop_loss or 0,
                take_profit=pattern.take_profit or 0,
                risk_reward_ratio=2.0,
                score=90.0,
                health=pattern.health,
                ml_probability=0.8,
            )
            pipeline._signal_engine._signals[signal.id] = signal
            return signal

        pipeline._telegram.send_signal = fake_send
        pipeline._signal_engine.create_signal = fake_create

        for _ in range(4):
            await pipeline.process_symbol("BTCUSDT", "1h")

        assert len(sent) == 1
        assert sent[0].priority == SignalPriority.HIGH
    finally:
        settings.telegram.min_priority = original
