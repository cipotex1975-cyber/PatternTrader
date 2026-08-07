from datetime import datetime, timedelta, timezone

import pytest

from app.confirmation.engine import ConfirmationEngine
from app.confirmation.models import ConfirmationStatus
from app.market.candles.models import Candle, CandleData
from app.patterns.base_pattern import PatternResult, PatternType, TradeDirection


def build_candle(close: float, index: int = 0) -> Candle:
    return Candle(
        symbol="BTCUSDT",
        timeframe="1h",
        data=CandleData(
            timestamp=datetime.now(timezone.utc) + timedelta(hours=index),
            open=close * 0.999,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=1000.0,
        ),
    )


def build_pattern(
    name: str,
    pattern_type: PatternType,
    direction: TradeDirection,
    key_levels: dict[str, float],
) -> PatternResult:
    return PatternResult(
        pattern_name=name,
        pattern_type=pattern_type,
        symbol="BTCUSDT",
        timeframe="1h",
        direction=direction,
        confidence=0.8,
        key_levels=key_levels,
    )


@pytest.fixture
def engine() -> ConfirmationEngine:
    return ConfirmationEngine()


def test_breakout_reversal_long_above_neckline(engine: ConfirmationEngine) -> None:
    pattern = build_pattern(
        "double_bottom",
        PatternType.REVERSAL,
        TradeDirection.LONG,
        {"neckline": 50000, "trough1": 48000, "trough2": 48100, "target": 52000},
    )
    candles = [build_candle(50500, 0)]

    check = engine._check_breakout(pattern, candles)

    assert check.status == ConfirmationStatus.PASSED
    assert check.threshold == 50000


def test_breakout_reversal_short_below_neckline(engine: ConfirmationEngine) -> None:
    pattern = build_pattern(
        "double_top",
        PatternType.REVERSAL,
        TradeDirection.SHORT,
        {"neckline": 50000, "peak1": 52000, "peak2": 51900, "target": 48000},
    )
    candles = [build_candle(49500, 0)]

    check = engine._check_breakout(pattern, candles)

    assert check.status == ConfirmationStatus.PASSED
    assert check.threshold == 50000


def test_breakout_continuation_long_above_pole_high(engine: ConfirmationEngine) -> None:
    pattern = build_pattern(
        "bull_flag",
        PatternType.CONTINUATION,
        TradeDirection.LONG,
        {"pole_high": 50000, "flag_low": 49000, "target": 52000},
    )
    candles = [build_candle(50500, 0)]

    check = engine._check_breakout(pattern, candles)

    assert check.status == ConfirmationStatus.PASSED
    assert check.threshold == 50000


def test_breakout_continuation_short_below_pole_low(engine: ConfirmationEngine) -> None:
    pattern = build_pattern(
        "bear_flag",
        PatternType.CONTINUATION,
        TradeDirection.SHORT,
        {"pole_low": 50000, "flag_high": 51000, "target": 48000},
    )
    candles = [build_candle(49500, 0)]

    check = engine._check_breakout(pattern, candles)

    assert check.status == ConfirmationStatus.PASSED
    assert check.threshold == 50000


def test_breakout_continuation_long_no_breakout_yet(engine: ConfirmationEngine) -> None:
    pattern = build_pattern(
        "bull_pennant",
        PatternType.CONTINUATION,
        TradeDirection.LONG,
        {"pole_high": 50000, "pennant_low": 49000, "target": 52000},
    )
    candles = [build_candle(49800, 0)]

    check = engine._check_breakout(pattern, candles)

    assert check.status == ConfirmationStatus.FAILED


def test_breakout_continuation_short_no_breakout_yet(engine: ConfirmationEngine) -> None:
    pattern = build_pattern(
        "bear_pennant",
        PatternType.CONTINUATION,
        TradeDirection.SHORT,
        {"pole_low": 50000, "pennant_high": 51000, "target": 48000},
    )
    candles = [build_candle(50200, 0)]

    check = engine._check_breakout(pattern, candles)

    assert check.status == ConfirmationStatus.FAILED


def test_breakout_missing_levels(engine: ConfirmationEngine) -> None:
    pattern = build_pattern(
        "double_top",
        PatternType.REVERSAL,
        TradeDirection.SHORT,
        {},
    )
    candles = [build_candle(49500, 0)]

    check = engine._check_breakout(pattern, candles)

    assert check.status == ConfirmationStatus.FAILED


def test_trend_alignment_continuation_long(engine: ConfirmationEngine) -> None:
    pattern = build_pattern(
        "bull_flag",
        PatternType.CONTINUATION,
        TradeDirection.LONG,
        {"pole_high": 50000},
    )
    check = engine._check_trend_alignment({"ema_21": 51000, "ema_50": 50000}, pattern)
    assert check.status == ConfirmationStatus.PASSED


def test_trend_alignment_continuation_short(engine: ConfirmationEngine) -> None:
    pattern = build_pattern(
        "bear_flag",
        PatternType.CONTINUATION,
        TradeDirection.SHORT,
        {"pole_low": 50000},
    )
    check = engine._check_trend_alignment({"ema_21": 49000, "ema_50": 50000}, pattern)
    assert check.status == ConfirmationStatus.PASSED


def test_trend_alignment_continuation_short_fails_in_uptrend(engine: ConfirmationEngine) -> None:
    pattern = build_pattern(
        "bear_flag",
        PatternType.CONTINUATION,
        TradeDirection.SHORT,
        {"pole_low": 50000},
    )
    check = engine._check_trend_alignment({"ema_21": 51000, "ema_50": 50000}, pattern)
    assert check.status == ConfirmationStatus.FAILED
