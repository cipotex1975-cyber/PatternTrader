from datetime import datetime, timezone

import pytest

from app.health.engine import HealthEngine
from app.market.candles.models import Candle, CandleData
from app.patterns.base_pattern import PatternResult, PatternType, TradeDirection
from app.patterns.reversal.double_top import DoubleTopPattern


def create_test_pattern(count=0, max_candles=20, direction=TradeDirection.SHORT):
    return PatternResult(
        pattern_name="double_top",
        pattern_type=PatternType.REVERSAL,
        symbol="BTCUSDT",
        timeframe="1h",
        direction=direction,
        confidence=0.8,
        key_levels={
            "peak1": 50000,
            "peak2": 49900,
            "neckline": 48800,
            "target": 47800,
        },
        max_confirmation_candles=max_candles,
        current_candle_count=count,
    )


def create_test_candles(n=30, close=48500, volume=1000):
    candles = []
    for _ in range(n):
        candles.append(
            Candle(
                symbol="BTCUSDT",
                timeframe="1h",
                data=CandleData(
                    timestamp=datetime.now(timezone.utc),
                    open=close,
                    high=close * 1.002,
                    low=close * 0.998,
                    close=close,
                    volume=volume,
                ),
            )
        )
    return candles


def test_health_weights_sum_to_one():
    assert abs(sum(HealthEngine.FACTOR_WEIGHTS.values()) - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_health_report_shape():
    engine = HealthEngine()
    report = await engine.calculate(
        create_test_pattern(), DoubleTopPattern(), create_test_candles(), {}
    )
    assert 0 <= report.health <= 100
    assert len(report.factors) == 8
    names = {f.name for f in report.factors}
    assert names == {
        "time_decay",
        "deformation",
        "volume",
        "trend",
        "atr",
        "slope",
        "false_breakouts",
        "volatility",
    }


@pytest.mark.asyncio
async def test_health_time_decay():
    engine = HealthEngine()
    pattern = create_test_pattern(count=19)
    report = await engine.calculate(pattern, DoubleTopPattern(), create_test_candles(), {})
    factor = next(f for f in report.factors if f.name == "time_decay")
    assert factor.score == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_health_false_breakouts_reduce_score():
    engine = HealthEngine()
    pattern = create_test_pattern()
    candles = create_test_candles()
    for i in range(1, 4):
        candle = candles[-i]
        data = candle.data
        candles[-i] = Candle(
            symbol="BTCUSDT",
            timeframe="1h",
            data=CandleData(
                timestamp=data.timestamp,
                open=data.open,
                high=data.high,
                low=48700,
                close=48850,
                volume=data.volume,
            ),
        )
    report = await engine.calculate(pattern, DoubleTopPattern(), candles, {})
    factor = next(f for f in report.factors if f.name == "false_breakouts")
    assert factor.score < 100
