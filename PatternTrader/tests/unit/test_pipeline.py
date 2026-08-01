from datetime import datetime, timedelta, timezone

import pytest

from app.lifecycle.models import LifecycleState
from app.market.candles.models import Candle, CandleData
from app.patterns.pipeline import PatternPipeline
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
    assert stats["signals_sent"] >= 1
    assert len(pipeline.lifecycle.get_by_state(LifecycleState.SIGNAL_SENT)) >= 1

    tracked = list(pipeline.tracked.values())[0]
    assert tracked.result.risk_reward_ratio is not None
    assert tracked.result.risk_reward_ratio >= 2.0


@pytest.mark.asyncio
async def test_pipeline_sends_telegram_for_critical():
    candles = build_double_top_candles()
    pipeline = PatternPipeline(data_source=lambda symbol, timeframe: candles)

    sent = []

    async def fake_send(signal):
        sent.append(signal)
        return True

    async def fake_create(pattern, score_result, ml_probability=0.0):
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

    assert len(sent) == 1
    assert sent[0].priority == SignalPriority.CRITICAL
