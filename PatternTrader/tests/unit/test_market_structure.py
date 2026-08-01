from datetime import datetime, timedelta, timezone

from app.market.candles.models import Candle, CandleData
from app.market.engine import MarketEngine
from app.market.fractals.detector import FractalDetector
from app.market.fractals.models import FractalType
from app.market.pivots.detector import PivotDetector
from app.market.pivots.models import Pivot, PivotType
from app.market.trendlines.channel import ChannelDetector
from app.market.trendlines.detector import TrendlineDetector
from app.market.trendlines.models import TrendDirection, Trendline, TrendlineType
from app.market.trendlines.trend import TrendAnalyzer
from app.market.zigzag.detector import ZigZagDetector
from app.market.zigzag.models import ZigZagType


def create_candle(open_price, high, low, close, volume=1000, index=0):
    return Candle(
        symbol="BTCUSDT",
        timeframe="1h",
        data=CandleData(
            timestamp=datetime.now(timezone.utc) + timedelta(hours=index),
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
        ),
    )


def build_series(prices, high_offset=0.0, low_offset=0.0, volume=1000):
    return [
        create_candle(p, p + high_offset, p - low_offset, p, volume=volume, index=i)
        for i, p in enumerate(prices)
    ]


class TestPivotDetector:
    def test_detects_swing_highs(self):
        prices = [10, 12, 15, 13, 11, 14, 16, 12]
        candles = build_series(prices)
        detector = PivotDetector(lookback=1)
        pivots = detector.find_pivots(candles)

        assert [(p.index, p.type, p.price) for p in pivots] == [
            (2, PivotType.SWING_HIGH, 15.0),
            (4, PivotType.SWING_LOW, 11.0),
            (6, PivotType.SWING_HIGH, 16.0),
        ]

    def test_insufficient_candles(self):
        detector = PivotDetector(lookback=2)
        assert detector.find_pivots(build_series([1, 2, 3])) == []


class TestFractalDetector:
    def test_detects_bill_williams_fractals(self):
        prices = [10, 12, 15, 13, 11, 14, 16, 12]
        candles = build_series(prices)
        detector = FractalDetector(window=1)
        fractals = detector.detect(candles)

        assert [(f.index, f.type) for f in fractals] == [
            (2, FractalType.UP),
            (4, FractalType.DOWN),
            (6, FractalType.UP),
        ]

    def test_detect_up_and_down_filters(self):
        prices = [10, 12, 15, 13, 11, 14, 16, 12]
        candles = build_series(prices)
        detector = FractalDetector(window=1)
        assert detector.detect_up(candles)[0].type == FractalType.UP
        assert detector.detect_down(candles)[0].type == FractalType.DOWN


class TestZigZagDetector:
    def test_single_up_leg(self):
        prices = [100, 110, 130, 120, 125, 105, 108, 150]
        candles = build_series(prices)
        points = ZigZagDetector(threshold=0.2).detect(candles)

        assert [(p.type, round(p.price, 2)) for p in points] == [
            (ZigZagType.LOW, 100.0),
            (ZigZagType.HIGH, 150.0),
        ]

    def test_alternating_highs_and_lows(self):
        prices = [100, 120, 140, 130, 118, 100, 130]
        candles = build_series(prices)
        points = ZigZagDetector(threshold=0.2).detect(candles)

        types = [p.type for p in points]
        assert types == [
            ZigZagType.LOW,
            ZigZagType.HIGH,
            ZigZagType.LOW,
            ZigZagType.HIGH,
        ]

    def test_atr_threshold_widening(self):
        prices = [100, 101, 102, 101, 100, 101, 100]
        candles = build_series(prices)
        points = ZigZagDetector(threshold=0.2, atr_multiplier=1.5).detect(candles, atr=10.0)
        assert points == []


class TestTrendlineDetector:
    def test_detect_from_pivots(self):
        candles = [
            create_candle(10, 12, 7, 10, index=0),
            create_candle(9, 10, 8, 9, index=1),
            create_candle(19, 20, 18, 19, index=2),
            create_candle(21, 22, 17, 21, index=3),
        ]
        pivots = [
            Pivot(
                index=1, timestamp=candles[1].data.timestamp, price=10.0, type=PivotType.SWING_HIGH
            ),
            Pivot(
                index=2, timestamp=candles[2].data.timestamp, price=20.0, type=PivotType.SWING_HIGH
            ),
            Pivot(
                index=1, timestamp=candles[1].data.timestamp, price=8.0, type=PivotType.SWING_LOW
            ),
            Pivot(
                index=2, timestamp=candles[2].data.timestamp, price=18.0, type=PivotType.SWING_LOW
            ),
        ]

        detector = TrendlineDetector(min_pivots=2)
        trendlines = detector.detect_from_pivots(candles, pivots)

        assert len(trendlines) == 2
        resistance = next(t for t in trendlines if t.type == TrendlineType.RESISTANCE)
        support = next(t for t in trendlines if t.type == TrendlineType.SUPPORT)

        assert resistance.slope == 10.0
        assert support.slope == 10.0
        assert resistance.touches == 2
        assert support.touches == 2

    def test_too_few_pivots(self):
        detector = TrendlineDetector(min_pivots=3)
        trendlines = detector.detect_from_pivots(build_series([1, 2, 3]), [])
        assert trendlines == []


class TestTrendAnalyzer:
    def test_uptrend_with_higher_highs_and_lows(self):
        candles = build_series([10, 11, 12, 13, 14, 15])
        pivots = [
            Pivot(
                index=0, timestamp=candles[0].data.timestamp, price=10.0, type=PivotType.SWING_HIGH
            ),
            Pivot(
                index=1, timestamp=candles[1].data.timestamp, price=15.0, type=PivotType.SWING_HIGH
            ),
            Pivot(
                index=2, timestamp=candles[2].data.timestamp, price=20.0, type=PivotType.SWING_HIGH
            ),
            Pivot(
                index=3, timestamp=candles[3].data.timestamp, price=8.0, type=PivotType.SWING_LOW
            ),
            Pivot(
                index=4, timestamp=candles[4].data.timestamp, price=13.0, type=PivotType.SWING_LOW
            ),
            Pivot(
                index=5, timestamp=candles[5].data.timestamp, price=18.0, type=PivotType.SWING_LOW
            ),
        ]

        trend = TrendAnalyzer(lookback=3).analyze(candles, pivots)

        assert trend.direction == TrendDirection.UPTREND
        assert trend.higher_highs == 2
        assert trend.higher_lows == 2
        assert trend.strength == 100.0
        assert trend.slope > 0

    def test_sideways_when_no_pivots(self):
        candles = build_series([10, 11, 12, 13, 14, 15])
        trend = TrendAnalyzer(lookback=3).analyze(candles, [])
        assert trend.direction == TrendDirection.SIDEWAYS


class TestChannelDetector:
    def test_detects_ascending_channel(self):
        candles = [
            create_candle(10, 12, 7, 10, index=0),
            create_candle(9, 10, 8, 9, index=1),
            create_candle(19, 20, 18, 19, index=2),
            create_candle(21, 22, 17, 21, index=3),
        ]
        resistance = Trendline(
            type=TrendlineType.RESISTANCE,
            start_index=1,
            end_index=2,
            start_price=10.0,
            end_price=20.0,
            start_timestamp=candles[1].data.timestamp,
            end_timestamp=candles[2].data.timestamp,
            slope=10.0,
            angle_degrees=84.29,
            touches=2,
        )
        support = Trendline(
            type=TrendlineType.SUPPORT,
            start_index=1,
            end_index=2,
            start_price=8.0,
            end_price=18.0,
            start_timestamp=candles[1].data.timestamp,
            end_timestamp=candles[2].data.timestamp,
            slope=10.0,
            angle_degrees=84.29,
            touches=2,
        )

        channels = ChannelDetector(slope_tolerance=0.15).detect(candles, [resistance, support])

        assert len(channels) == 1
        assert channels[0].direction == TrendDirection.UPTREND
        assert channels[0].width == 2.0

    def test_non_parallel_lines_do_not_form_channel(self):
        candles = build_series([10, 11, 12, 13, 14, 15])
        resistance = Trendline(
            type=TrendlineType.RESISTANCE,
            start_index=0,
            end_index=3,
            start_price=10.0,
            end_price=13.0,
            start_timestamp=candles[0].data.timestamp,
            end_timestamp=candles[3].data.timestamp,
            slope=1.0,
            angle_degrees=45.0,
            touches=2,
        )
        support = Trendline(
            type=TrendlineType.SUPPORT,
            start_index=0,
            end_index=3,
            start_price=8.0,
            end_price=9.0,
            start_timestamp=candles[0].data.timestamp,
            end_timestamp=candles[3].data.timestamp,
            slope=0.33,
            angle_degrees=18.43,
            touches=2,
        )

        channels = ChannelDetector(slope_tolerance=0.15).detect(candles, [resistance, support])
        assert channels == []


class TestMarketEngine:
    def test_analyze_builds_full_structure(self):
        prices = [
            100,
            104,
            108,
            112,
            116,
            120,
            124,
            128,
            124,
            120,
            116,
            120,
            124,
            128,
            132,
            136,
            140,
            144,
            148,
            144,
            140,
            136,
            140,
            144,
            148,
            152,
            156,
            160,
            164,
            168,
        ]
        candles = build_series(prices, high_offset=1.0, low_offset=1.0)

        structure = MarketEngine().analyze(candles, symbol="BTCUSDT", timeframe="1h")

        assert structure.symbol == "BTCUSDT"
        assert structure.timeframe == "1h"
        assert structure.candle_count == 30
        assert structure.trend.direction == TrendDirection.UPTREND
        assert "rsi" in structure.latest_indicators
        assert "atr" in structure.latest_indicators
        assert "momentum" in structure.latest_indicators
        assert structure.latest_indicators["momentum"] > 0

    def test_analyze_empty(self):
        structure = MarketEngine().analyze([], symbol="BTCUSDT", timeframe="1h")
        assert structure.candle_count == 0
        assert structure.trend.direction == TrendDirection.SIDEWAYS
