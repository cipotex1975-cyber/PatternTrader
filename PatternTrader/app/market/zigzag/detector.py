from __future__ import annotations

from app.core.config.settings import get_settings
from app.market.candles.models import Candle
from app.market.zigzag.models import ZigZagPoint, ZigZagType


class ZigZagDetector:
    """Detects ZigZag pivots using a percentage (or ATR-based) threshold.

    A reversal is only confirmed when the price retraces more than the
    threshold from the running extreme, producing the classic ZigZag
    alternating high/low structure.
    """

    def __init__(self, threshold: float | None = None, atr_multiplier: float | None = None) -> None:
        settings = get_settings()
        structure = settings.market.structure
        if threshold is None:
            threshold = structure.zigzag_threshold
        if atr_multiplier is None:
            atr_multiplier = structure.zigzag_atr_multiplier
        self._threshold = threshold
        self._atr_multiplier = atr_multiplier

    @property
    def threshold(self) -> float:
        return self._threshold

    def detect(self, candles: list[Candle], atr: float | None = None) -> list[ZigZagPoint]:
        if len(candles) < 2:
            return []

        threshold = self._threshold
        if atr is not None and atr > 0:
            threshold = max(self._threshold, self._atr_multiplier * atr / candles[0].data.close)

        points: list[ZigZagPoint] = []
        direction: str | None = None

        pivot_idx = 0
        pivot_price = candles[0].data.low

        extreme_idx = 0
        extreme_price = candles[0].data.close
        extreme_type = ZigZagType.HIGH

        for i in range(1, len(candles)):
            high = candles[i].data.high
            low = candles[i].data.low

            if direction is None:
                if high >= pivot_price * (1 + threshold):
                    direction = "up"
                    points.append(self._build(candles, pivot_idx, pivot_price, ZigZagType.LOW))
                    extreme_idx, extreme_price, extreme_type = i, high, ZigZagType.HIGH
                elif low <= pivot_price * (1 - threshold):
                    direction = "down"
                    points.append(self._build(candles, pivot_idx, pivot_price, ZigZagType.HIGH))
                    extreme_idx, extreme_price, extreme_type = i, low, ZigZagType.LOW
                continue

            if direction == "up":
                if high > extreme_price:
                    extreme_idx, extreme_price = i, high
                elif low <= extreme_price * (1 - threshold):
                    points.append(self._build(candles, extreme_idx, extreme_price, ZigZagType.HIGH))
                    direction = "down"
                    extreme_idx, extreme_price, extreme_type = i, low, ZigZagType.LOW
            else:  # direction == "down"
                if low < extreme_price:
                    extreme_idx, extreme_price = i, low
                elif high >= extreme_price * (1 + threshold):
                    points.append(self._build(candles, extreme_idx, extreme_price, ZigZagType.LOW))
                    direction = "up"
                    extreme_idx, extreme_price, extreme_type = i, high, ZigZagType.HIGH

        if points:
            points.append(self._build(candles, extreme_idx, extreme_price, extreme_type))

        return points

    @staticmethod
    def _build(candles: list[Candle], index: int, price: float, type_: ZigZagType) -> ZigZagPoint:
        return ZigZagPoint(
            index=index,
            timestamp=candles[index].data.timestamp,
            price=float(price),
            type=type_,
        )
