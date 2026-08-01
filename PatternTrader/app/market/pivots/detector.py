from __future__ import annotations

from app.core.config.settings import get_settings
from app.market.candles.models import Candle
from app.market.pivots.models import Pivot, PivotType


class PivotDetector:
    """Detects swing highs and swing lows in a candle series.

    A swing point is a local extreme over a window of ``lookback`` bars on
    each side (classic pivot definition, also known as ZigZag-style pivots).
    """

    def __init__(self, lookback: int | None = None) -> None:
        if lookback is None:
            settings = get_settings()
            lookback = settings.market.structure.pivot_lookback
        self._lookback = max(1, lookback)

    @property
    def lookback(self) -> int:
        return self._lookback

    def detect_swing_highs(self, candles: list[Candle]) -> list[Pivot]:
        return [
            Pivot(
                index=i,
                timestamp=candles[i].data.timestamp,
                price=float(candles[i].data.high),
                type=PivotType.SWING_HIGH,
            )
            for i in self._extreme_indices(candles, use_high=True)
        ]

    def detect_swing_lows(self, candles: list[Candle]) -> list[Pivot]:
        return [
            Pivot(
                index=i,
                timestamp=candles[i].data.timestamp,
                price=float(candles[i].data.low),
                type=PivotType.SWING_LOW,
            )
            for i in self._extreme_indices(candles, use_high=False)
        ]

    def find_pivots(self, candles: list[Candle]) -> list[Pivot]:
        """Return all pivots (highs and lows) sorted by index."""
        pivots = self.detect_swing_highs(candles) + self.detect_swing_lows(candles)
        return sorted(pivots, key=lambda p: p.index)

    def _extreme_indices(self, candles: list[Candle], use_high: bool) -> list[int]:
        n = len(candles)
        lookback = self._lookback
        if n < (2 * lookback) + 1:
            return []

        indices: list[int] = []
        for i in range(lookback, n - lookback):
            current = candles[i].data.high if use_high else candles[i].data.low
            values = [
                candles[j].data.high if use_high else candles[j].data.low
                for j in range(i - lookback, i + lookback + 1)
            ]

            if use_high:
                is_extreme = current >= max(values) and values.count(current) == 1
            else:
                is_extreme = current <= min(values) and values.count(current) == 1

            if is_extreme:
                indices.append(i)

        return indices
