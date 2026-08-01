from datetime import datetime, timezone

from app.market.candles.models import Candle, CandleData
from app.patterns.continuation.bull_flag import BullFlagPattern
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


PATTERN_INTERFACE_METHODS = [
    "detect",
    "validate",
    "score",
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
