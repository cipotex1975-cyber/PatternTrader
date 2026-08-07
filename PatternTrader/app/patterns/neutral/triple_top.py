from __future__ import annotations

from typing import Optional

import numpy as np

from app.core.logger import get_logger
from app.market.candles.models import Candle
from app.patterns.base_pattern import (
    BasePattern,
    PatternResult,
    PatternType,
    TradeDirection,
)
from app.patterns.neutral.geometry import find_peaks
from app.patterns.registry import register_pattern

logger = get_logger("TripleTopPattern")


@register_pattern
class TripleTopPattern(BasePattern):
    @property
    def name(self) -> str:
        return "triple_top"

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.REVERSAL

    @property
    def max_confirmation_candles(self) -> int:
        return 25

    def detect(
        self,
        candles: list[Candle],
        symbol: str,
        timeframe: str,
    ) -> Optional[PatternResult]:
        if len(candles) < 20:
            return None

        highs = np.array([c.data.high for c in candles], dtype=float)
        lows = np.array([c.data.low for c in candles], dtype=float)

        peaks = find_peaks(highs, distance=3)
        if len(peaks) < 3:
            return None

        for i in range(len(peaks) - 2):
            p1, p2, p3 = peaks[i], peaks[i + 1], peaks[i + 2]
            prices = [highs[p1], highs[p2], highs[p3]]

            if max(prices) - min(prices) > max(prices) * 0.02:
                continue

            valley1 = lows[p1 : p2 + 1].min()
            valley2 = lows[p2 : p3 + 1].min()
            neckline = min(valley1, valley2)

            peak_price = max(prices)
            if peak_price <= neckline:
                continue

            height = peak_price - neckline
            if height / peak_price < 0.01:
                continue

            return PatternResult(
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                direction=TradeDirection.SHORT,
                confidence=self._calculate_confidence(prices, neckline),
                key_levels={
                    "peak1": float(highs[p1]),
                    "peak2": float(highs[p2]),
                    "peak3": float(highs[p3]),
                    "neckline": float(neckline),
                    "target": float(neckline - height),
                },
                max_confirmation_candles=self.max_confirmation_candles,
            )

        return None

    def validate(self, pattern: PatternResult, candles: list[Candle]) -> bool:
        if not pattern.key_levels:
            return False

        neckline = pattern.key_levels.get("neckline", 0)
        if not candles:
            return False

        latest_close = candles[-1].data.close
        return latest_close > neckline * 0.98

    def _calculate_confidence(self, prices: list[float], neckline: float) -> float:
        spread = (max(prices) - min(prices)) / max(prices)
        height = max(prices) - neckline
        ratio = min(1.0, height / max(prices) * 20)

        confidence = 0.5
        confidence += (1 - spread) * 0.25
        confidence += ratio * 0.25

        return min(1.0, confidence)
