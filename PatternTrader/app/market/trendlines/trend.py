from __future__ import annotations

import numpy as np

from app.core.config.settings import get_settings
from app.market.candles.models import Candle
from app.market.pivots.models import Pivot, PivotType
from app.market.trendlines.models import Trend, TrendDirection


class TrendAnalyzer:
    """Classifies the current market trend and measures its strength."""

    def __init__(self, lookback: int | None = None) -> None:
        settings = get_settings()
        if lookback is None:
            lookback = settings.market.structure.trend_strength_lookback
        self._lookback = max(2, lookback)

    @property
    def lookback(self) -> int:
        return self._lookback

    def analyze(self, candles: list[Candle], pivots: list[Pivot] | None = None) -> Trend:
        highs = [p for p in (pivots or []) if p.type == PivotType.SWING_HIGH]
        lows = [p for p in (pivots or []) if p.type == PivotType.SWING_LOW]

        higher_highs = self._count_sequence(highs, rising=True)
        lower_highs = max(0, len(highs) - 1 - higher_highs)
        higher_lows = self._count_sequence(lows, rising=True)
        lower_lows = max(0, len(lows) - 1 - higher_lows)

        slope = self._fit_slope(candles)

        up_score = higher_highs + higher_lows
        down_score = lower_highs + lower_lows
        total = up_score + down_score

        if up_score > down_score:
            direction = TrendDirection.UPTREND
            consistent = higher_highs + higher_lows
        elif down_score > up_score:
            direction = TrendDirection.DOWNTREND
            consistent = lower_highs + lower_lows
        else:
            direction = TrendDirection.SIDEWAYS
            consistent = total

        strength = min(100.0, (consistent / max(total, 1)) * 100.0)

        return Trend(
            direction=direction,
            slope=slope,
            strength=round(strength, 2),
            higher_highs=higher_highs,
            higher_lows=higher_lows,
            lower_highs=lower_highs,
            lower_lows=lower_lows,
        )

    @staticmethod
    def _count_sequence(pivots: list[Pivot], rising: bool) -> int:
        count = 0
        for i in range(1, len(pivots)):
            prev_price = pivots[i - 1].price
            curr_price = pivots[i].price
            if (rising and curr_price > prev_price) or (not rising and curr_price < prev_price):
                count += 1
        return count

    @staticmethod
    def _fit_slope(candles: list[Candle]) -> float:
        if len(candles) < 2:
            return 0.0
        x = np.arange(len(candles))
        y = np.array([c.data.close for c in candles], dtype=float)
        try:
            slope = np.polyfit(x, y, 1)[0]
        except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
            return 0.0
        return float(slope)
