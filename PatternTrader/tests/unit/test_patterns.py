from datetime import datetime, timezone

import numpy as np

from app.market.candles.models import Candle, CandleData
from app.patterns.continuation.bull_flag import BullFlagPattern
from app.patterns.neutral.ascending_triangle import AscendingTrianglePattern
from app.patterns.neutral.broadening import BroadeningPattern
from app.patterns.neutral.channel import ChannelPattern
from app.patterns.neutral.cup_and_handle import CupAndHandlePattern
from app.patterns.neutral.descending_triangle import DescendingTrianglePattern
from app.patterns.neutral.diamond import DiamondPattern
from app.patterns.neutral.falling_wedge import FallingWedgePattern
from app.patterns.neutral.rectangle import RectanglePattern
from app.patterns.neutral.rising_wedge import RisingWedgePattern
from app.patterns.neutral.rounded_bottom import RoundedBottomPattern
from app.patterns.neutral.symmetric_triangle import SymmetricTrianglePattern
from app.patterns.neutral.triple_bottom import TripleBottomPattern
from app.patterns.neutral.triple_top import TripleTopPattern
from app.patterns.reversal.double_bottom import DoubleBottomPattern
from app.patterns.reversal.double_top import DoubleTopPattern


def create_candle(open_price, high, low, close, volume=1000):
    return Candle(
        symbol="BTCUSDT",
        timeframe="1h",
        data=CandleData(
            timestamp=datetime.now(timezone.utc),
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
        ),
    )


def create_candles_from_bands(highs, lows):
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = (highs + lows) / 2
    return [
        create_candle(float(c), float(h), float(l), float(c))
        for h, l, c in zip(highs, lows, closes)
    ]


def test_double_top_pattern():
    pattern = DoubleTopPattern()
    assert pattern.name == "double_top"
    assert pattern.pattern_type.value == "reversal"
    assert pattern.max_confirmation_candles == 20


def test_double_bottom_pattern():
    pattern = DoubleBottomPattern()
    assert pattern.name == "double_bottom"
    assert pattern.pattern_type.value == "reversal"
    assert pattern.max_confirmation_candles == 20


def test_bull_flag_pattern():
    pattern = BullFlagPattern()
    assert pattern.name == "bull_flag"
    assert pattern.pattern_type.value == "continuation"
    assert pattern.max_confirmation_candles == 12


def test_pattern_registry():
    from app.patterns.registry import PatternRegistry

    patterns = PatternRegistry.get_all()
    assert "double_top" in patterns
    assert "double_bottom" in patterns
    assert "bull_flag" in patterns


NEW_PATTERNS = {
    "ascending_triangle": (AscendingTrianglePattern, "neutral"),
    "descending_triangle": (DescendingTrianglePattern, "neutral"),
    "symmetric_triangle": (SymmetricTrianglePattern, "neutral"),
    "rising_wedge": (RisingWedgePattern, "neutral"),
    "falling_wedge": (FallingWedgePattern, "neutral"),
    "rectangle": (RectanglePattern, "neutral"),
    "channel": (ChannelPattern, "neutral"),
    "cup_and_handle": (CupAndHandlePattern, "continuation"),
    "rounded_bottom": (RoundedBottomPattern, "reversal"),
    "diamond": (DiamondPattern, "neutral"),
    "broadening": (BroadeningPattern, "neutral"),
    "triple_top": (TripleTopPattern, "reversal"),
    "triple_bottom": (TripleBottomPattern, "reversal"),
}


def test_new_patterns_registered_and_basic_metadata():
    from app.patterns.registry import PatternRegistry

    patterns = PatternRegistry.get_all()
    for name, (cls, pattern_type) in NEW_PATTERNS.items():
        assert name in patterns, f"{name} no está registrado"
        instance = cls()
        assert instance.name == name
        assert instance.pattern_type.value == pattern_type


PATTERN_INTERFACE_METHODS = [
    "detect",
    "validate",
    "update",
    "invalidate",
    "statistics",
    "plot",
]


def test_all_patterns_implement_required_interface():
    from app.patterns.registry import PatternRegistry

    patterns = PatternRegistry.get_all()
    assert patterns, "no hay patrones registrados"
    for name, cls in patterns.items():
        for method in PATTERN_INTERFACE_METHODS:
            assert callable(getattr(cls, method)), f"{name} no implementa {method}()"


def test_pattern_plot_returns_figure():
    pattern = DoubleTopPattern()
    candles = [
        create_candle(100 + i * 10, 105 + i * 10, 95 + i * 10, 102 + i * 10) for i in range(10)
    ]
    fig = pattern.plot(candles)
    assert fig is not None
    assert len(fig.data) == 2  # candlestick + volume


def test_ascending_triangle_detects():
    pattern = AscendingTrianglePattern()
    candles = create_candles_from_bands(
        np.full(40, 101.0),
        np.linspace(88, 99, 40),
    )
    result = pattern.detect(candles, "BTCUSDT", "1h")
    assert result is not None
    assert result.pattern_name == "ascending_triangle"
    assert result.direction.value == "LONG"
    assert "neckline" in result.key_levels


def test_descending_triangle_detects():
    pattern = DescendingTrianglePattern()
    candles = create_candles_from_bands(
        np.linspace(110, 101, 40),
        np.full(40, 99.0),
    )
    result = pattern.detect(candles, "BTCUSDT", "1h")
    assert result is not None
    assert result.direction.value == "SHORT"


def test_symmetric_triangle_detects():
    pattern = SymmetricTrianglePattern()
    candles = create_candles_from_bands(
        np.linspace(112, 103.5, 40),
        np.linspace(90, 101, 40),
    )
    result = pattern.detect(candles, "BTCUSDT", "1h")
    assert result is not None


def test_rising_wedge_detects():
    pattern = RisingWedgePattern()
    candles = create_candles_from_bands(
        np.linspace(98, 112, 40),
        np.linspace(90, 108, 40),
    )
    result = pattern.detect(candles, "BTCUSDT", "1h")
    assert result is not None
    assert result.direction.value == "SHORT"


def test_falling_wedge_detects():
    pattern = FallingWedgePattern()
    candles = create_candles_from_bands(
        np.linspace(100, 88, 40),
        np.linspace(88, 80, 40),
    )
    result = pattern.detect(candles, "BTCUSDT", "1h")
    assert result is not None
    assert result.direction.value == "LONG"


def test_rectangle_detects():
    pattern = RectanglePattern()
    candles = create_candles_from_bands(
        np.full(40, 105.0),
        np.full(40, 95.0),
    )
    result = pattern.detect(candles, "BTCUSDT", "1h")
    assert result is not None


def test_channel_detects():
    pattern = ChannelPattern()
    x = np.arange(40)
    candles = create_candles_from_bands(100 + 0.3 * x, 90 + 0.3 * x)
    result = pattern.detect(candles, "BTCUSDT", "1h")
    assert result is not None


def test_cup_and_handle_detects():
    pattern = CupAndHandlePattern()
    cup_low = np.concatenate(
        [np.linspace(100, 90, 12), np.linspace(90, 100, 12), np.linspace(100, 97, 8)]
    )
    candles = create_candles_from_bands(np.full(32, 101.0), cup_low)
    result = pattern.detect(candles, "BTCUSDT", "1h")
    assert result is not None
    assert result.direction.value == "LONG"


def test_rounded_bottom_detects():
    pattern = RoundedBottomPattern()
    lows = 90 + 10 * (np.linspace(-1, 1, 40) ** 2)
    candles = create_candles_from_bands(lows + 5, lows)
    result = pattern.detect(candles, "BTCUSDT", "1h")
    assert result is not None
    assert result.direction.value == "LONG"


def test_diamond_detects():
    pattern = DiamondPattern()
    highs = np.concatenate([np.linspace(112, 105, 20), np.linspace(105, 115, 20)])
    lows = np.concatenate([np.linspace(92, 99, 20), np.linspace(99, 90, 20)])
    candles = create_candles_from_bands(highs, lows)
    result = pattern.detect(candles, "BTCUSDT", "1h")
    assert result is not None


def test_broadening_detects():
    pattern = BroadeningPattern()
    candles = create_candles_from_bands(
        np.linspace(106, 118, 40),
        np.linspace(94, 82, 40),
    )
    result = pattern.detect(candles, "BTCUSDT", "1h")
    assert result is not None


def _zigzag_updown(up, down, cycles):
    series = np.array([])
    for _ in range(cycles):
        series = np.concatenate([series, np.linspace(100, 110, up), np.linspace(110, 100, down)])
    return series


def _zigzag_downup(up, down, cycles):
    series = np.array([])
    for _ in range(cycles):
        series = np.concatenate([series, np.linspace(110, 100, up), np.linspace(100, 110, down)])
    return series


def test_triple_top_detects():
    pattern = TripleTopPattern()
    mid = _zigzag_updown(7, 7, 3)
    candles = create_candles_from_bands(mid * 1.004, mid * 0.996)
    result = pattern.detect(candles, "BTCUSDT", "1h")
    assert result is not None
    assert result.direction.value == "SHORT"
    assert {"peak1", "peak2", "peak3"} <= set(result.key_levels)


def test_triple_bottom_detects():
    pattern = TripleBottomPattern()
    mid = _zigzag_downup(7, 7, 3)
    candles = create_candles_from_bands(mid * 1.004, mid * 0.996)
    result = pattern.detect(candles, "BTCUSDT", "1h")
    assert result is not None
    assert result.direction.value == "LONG"
    assert {"trough1", "trough2", "trough3"} <= set(result.key_levels)
