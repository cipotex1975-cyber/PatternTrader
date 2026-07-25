import pytest
from datetime import datetime, timezone
from app.market.candles.models import Candle, CandleData
from app.patterns.reversal.double_top import DoubleTopPattern
from app.patterns.reversal.double_bottom import DoubleBottomPattern
from app.patterns.continuation.bull_flag import BullFlagPattern


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
