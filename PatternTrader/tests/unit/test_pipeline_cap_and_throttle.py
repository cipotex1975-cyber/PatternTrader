from datetime import datetime, timedelta, timezone

import pytest

from app.core.config.settings import Settings
from app.market.candles.models import Candle, CandleData
from app.patterns import pipeline as pipeline_module
from app.patterns.base_pattern import PatternResult, PatternType, TradeDirection
from app.patterns.pipeline import PatternPipeline, TrackedPattern


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


def double_top_candles():
    closes = [47000 + i * 300 for i in range(10)]
    closes += [49400, 49100, 48800]
    closes += [49100, 49400, 49700, 49950, 49700, 49400]
    closes += [48900, 48700, 48500, 48300, 48100]
    volumes = [1000] * len(closes)
    for i in range(-5, 0):
        volumes[i] = 2000
    return build_candles(closes, volumes)


def settings_with(max_patterns: int, health_interval: int) -> Settings:
    settings = Settings()
    settings.patterns.lifecycle.max_patterns_per_symbol = max_patterns
    settings.patterns.health.recalculate_interval_seconds = health_interval
    return settings


class FakeDetector:
    def __init__(self) -> None:
        self._calls = 0
        self.name = "fake_detector"
        self.max_confirmation_candles = 10

    def detect(self, candles, symbol, timeframe):
        self._calls += 1
        return PatternResult(
            pattern_name=f"fake_pattern_{self._calls}",
            pattern_type=PatternType.CONTINUATION,
            symbol=symbol,
            timeframe=timeframe,
            direction=TradeDirection.LONG,
            confidence=0.8,
        )

    def validate(self, pattern, candles):
        return True

    def update(self, pattern, candles):
        return pattern


def _inject_active(pipeline: PatternPipeline, symbol: str, n: int) -> None:
    for i in range(n):
        result = PatternResult(
            pattern_name=f"injected_{i}",
            pattern_type=PatternType.REVERSAL,
            symbol=symbol,
            timeframe="1h",
            confidence=0.7,
        )
        tracked = TrackedPattern(
            detector=FakeDetector(),
            result=result,
            lifecycle_id=result.id,
        )
        pipeline._tracked[result.id] = tracked
        pipeline._active_keys.add((symbol, "1h", result.pattern_name))


@pytest.mark.asyncio
async def test_pipeline_respects_max_patterns_cap(monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "get_settings", lambda: settings_with(max_patterns=1, health_interval=0)
    )
    pipeline = PatternPipeline()
    _inject_active(pipeline, "BTCUSDT", 2)
    detector = FakeDetector()
    pipeline._detectors = [detector]

    await pipeline._detect_new([], "BTCUSDT", "1h", {})

    assert detector._calls >= 1
    assert len(pipeline.tracked) == 2


@pytest.mark.asyncio
async def test_pipeline_allows_new_pattern_below_cap(monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "get_settings", lambda: settings_with(max_patterns=5, health_interval=0)
    )
    pipeline = PatternPipeline()
    _inject_active(pipeline, "BTCUSDT", 2)
    detector = FakeDetector()
    pipeline._detectors = [detector]

    await pipeline._detect_new([], "BTCUSDT", "1h", {})

    assert len(pipeline.tracked) == 3
    assert any(t.result.pattern_name.startswith("fake_pattern") for t in pipeline.tracked.values())


@pytest.mark.asyncio
async def test_pipeline_health_recalculation_is_throttled(monkeypatch):
    monkeypatch.setattr(
        pipeline_module,
        "get_settings",
        lambda: settings_with(max_patterns=50, health_interval=3600),
    )
    pipeline = PatternPipeline()

    calls = []
    real_calculate = pipeline._health.calculate

    async def spy_calculate(result, detector, candles, latest_indicators):
        calls.append(1)
        return await real_calculate(result, detector, candles, latest_indicators)

    pipeline._health.calculate = spy_calculate

    await pipeline.process_symbol("BTCUSDT", "1h", candles=double_top_candles())
    await pipeline.process_symbol("BTCUSDT", "1h", candles=double_top_candles())

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_pipeline_health_recalculates_when_interval_elapsed(monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "get_settings", lambda: settings_with(max_patterns=50, health_interval=0)
    )
    pipeline = PatternPipeline()

    calls = []
    real_calculate = pipeline._health.calculate

    async def spy_calculate(result, detector, candles, latest_indicators):
        calls.append(1)
        return await real_calculate(result, detector, candles, latest_indicators)

    pipeline._health.calculate = spy_calculate

    await pipeline.process_symbol("BTCUSDT", "1h", candles=double_top_candles())
    await pipeline.process_symbol("BTCUSDT", "1h", candles=double_top_candles())

    assert len(calls) == 2
