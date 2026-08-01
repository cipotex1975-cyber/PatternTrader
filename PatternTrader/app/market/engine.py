from __future__ import annotations

from app.market.candles.models import Candle
from app.market.fractals.detector import FractalDetector
from app.market.indicators.calculator import IndicatorCalculator
from app.market.models import MarketStructure
from app.market.pivots.detector import PivotDetector
from app.market.trendlines.channel import ChannelDetector
from app.market.trendlines.detector import TrendlineDetector
from app.market.trendlines.trend import TrendAnalyzer
from app.market.zigzag.detector import ZigZagDetector


class MarketEngine:
    """Builds the complete price structure of a market.

    Combines technical indicators, swing pivots, fractals, ZigZag, trendlines,
    channels and trend analysis into a single ``MarketStructure`` object.
    """

    def __init__(self) -> None:
        self._indicators = IndicatorCalculator()
        self._pivots = PivotDetector()
        self._fractals = FractalDetector()
        self._zigzag = ZigZagDetector()
        self._trendlines = TrendlineDetector()
        self._channels = ChannelDetector()
        self._trend = TrendAnalyzer()

    def analyze(
        self, candles: list[Candle], symbol: str = "", timeframe: str = ""
    ) -> MarketStructure:
        if not candles:
            return MarketStructure(symbol=symbol, timeframe=timeframe)

        indicators = self._indicators.calculate_all(candles)
        latest_indicators = self._indicators.get_latest_indicators(candles)

        pivots = self._pivots.find_pivots(candles)
        fractals = self._fractals.detect(candles)
        zigzag = self._zigzag.detect(candles, atr=latest_indicators.get("atr"))
        trendlines = self._trendlines.detect_from_pivots(candles, pivots)
        channels = self._channels.detect(candles, trendlines)
        trend = self._trend.analyze(candles, pivots)

        return MarketStructure(
            symbol=symbol,
            timeframe=timeframe,
            candle_count=len(candles),
            indicators=indicators,
            latest_indicators=latest_indicators,
            pivots=pivots,
            fractals=fractals,
            zigzag=zigzag,
            trendlines=trendlines,
            channels=channels,
            trend=trend,
        )

    @property
    def pivot_detector(self) -> PivotDetector:
        return self._pivots

    @property
    def fractal_detector(self) -> FractalDetector:
        return self._fractals

    @property
    def zigzag_detector(self) -> ZigZagDetector:
        return self._zigzag

    @property
    def trendline_detector(self) -> TrendlineDetector:
        return self._trendlines

    @property
    def channel_detector(self) -> ChannelDetector:
        return self._channels

    @property
    def trend_analyzer(self) -> TrendAnalyzer:
        return self._trend
