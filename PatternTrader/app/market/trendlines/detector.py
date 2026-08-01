from __future__ import annotations

import math

from app.core.config.settings import get_settings
from app.market.candles.models import Candle
from app.market.pivots.detector import PivotDetector
from app.market.pivots.models import Pivot, PivotType
from app.market.trendlines.models import Trendline, TrendlineType


class TrendlineDetector:
    """Builds support and resistance trendlines from swing pivots.

    Consecutive swing highs are connected to build resistance trendlines and
    consecutive swing lows to build support trendlines.
    """

    def __init__(
        self,
        lookback: int | None = None,
        min_pivots: int | None = None,
        touch_tolerance: float = 0.01,
    ) -> None:
        settings = get_settings()
        structure = settings.market.structure
        if lookback is None:
            lookback = structure.pivot_lookback
        if min_pivots is None:
            min_pivots = structure.trend_min_pivots
        self._lookback = max(1, lookback)
        self._min_pivots = max(2, min_pivots)
        self._touch_tolerance = touch_tolerance

    @property
    def min_pivots(self) -> int:
        return self._min_pivots

    def detect(self, candles: list[Candle]) -> list[Trendline]:
        detector = PivotDetector(self._lookback)
        pivots = detector.find_pivots(candles)
        return self.detect_from_pivots(candles, pivots)

    def detect_from_pivots(self, candles: list[Candle], pivots: list[Pivot]) -> list[Trendline]:
        highs = [p for p in pivots if p.type == PivotType.SWING_HIGH]
        lows = [p for p in pivots if p.type == PivotType.SWING_LOW]

        trendlines: list[Trendline] = []
        trendlines.extend(self._build_series(candles, highs, TrendlineType.RESISTANCE))
        trendlines.extend(self._build_series(candles, lows, TrendlineType.SUPPORT))
        return trendlines

    def _build_series(
        self,
        candles: list[Candle],
        pivots: list[Pivot],
        type_: TrendlineType,
    ) -> list[Trendline]:
        if len(pivots) < self._min_pivots:
            return []

        trendlines: list[Trendline] = []
        use_high = type_ == TrendlineType.RESISTANCE
        for i in range(len(pivots) - 1):
            a = pivots[i]
            b = pivots[i + 1]
            if b.index <= a.index:
                continue

            slope = (b.price - a.price) / (b.index - a.index)
            angle = math.degrees(math.atan(slope))
            touches = self._count_touches(
                candles,
                a.index,
                b.index,
                a.price,
                slope,
                use_high=use_high,
            )

            trendlines.append(
                Trendline(
                    type=type_,
                    start_index=a.index,
                    end_index=b.index,
                    start_price=round(a.price, 8),
                    end_price=round(b.price, 8),
                    start_timestamp=a.timestamp,
                    end_timestamp=b.timestamp,
                    slope=slope,
                    angle_degrees=angle,
                    touches=touches,
                )
            )

        return trendlines

    def _count_touches(
        self,
        candles: list[Candle],
        start_index: int,
        end_index: int,
        start_price: float,
        slope: float,
        use_high: bool,
    ) -> int:
        touches = 0
        for k in range(start_index, end_index + 1):
            line_value = start_price + slope * (k - start_index)
            if line_value <= 0:
                continue
            price = candles[k].data.high if use_high else candles[k].data.low
            if abs(price - line_value) / line_value <= self._touch_tolerance:
                touches += 1
        return touches

    def line_value_at(self, trendline: Trendline, index: int) -> float:
        """Value of a trendline at a given candle index."""
        return trendline.start_price + trendline.slope * (index - trendline.start_index)
