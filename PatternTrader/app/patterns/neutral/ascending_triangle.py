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
from app.patterns.neutral.geometry import fit_line, line_at
from app.patterns.registry import register_pattern

logger = get_logger("AscendingTrianglePattern")


@register_pattern
class AscendingTrianglePattern(BasePattern):
    @property
    def name(self) -> str:
        return "ascending_triangle"

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.NEUTRAL

    @property
    def max_confirmation_candles(self) -> int:
        return 15

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

        res_slope, res_intercept = fit_line(highs)
        sup_slope, sup_intercept = fit_line(lows)

        n = len(candles)
        res_end = line_at(res_slope, res_intercept, n - 1)
        sup_end = line_at(sup_slope, sup_intercept, n - 1)

        if abs(res_slope) > 0.001 * res_end:
            return None
        if sup_slope <= 0.0001 * sup_end:
            return None
        if sup_end >= res_end:
            return None

        height = res_end - lows.min()
        if height / res_end < 0.02:
            return None

        return PatternResult(
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            symbol=symbol,
            timeframe=timeframe,
            direction=TradeDirection.LONG,
            confidence=self._calculate_confidence(res_slope, sup_slope, height, res_end),
            key_levels={
                "neckline": res_end,
                "support": line_at(sup_slope, sup_intercept, 0),
                "valley": float(lows.min()),
                "apex": self._apex(res_slope, res_intercept, sup_slope, sup_intercept),
                "target": res_end + height,
            },
            max_confirmation_candles=self.max_confirmation_candles,
        )

    def validate(self, pattern: PatternResult, candles: list[Candle]) -> bool:
        if not pattern.key_levels:
            return False

        neckline = pattern.key_levels.get("neckline", 0)
        if not candles:
            return False

        latest_close = candles[-1].data.close
        return latest_close > neckline * 0.98

    def _apex(
        self,
        res_slope: float,
        res_intercept: float,
        sup_slope: float,
        sup_intercept: float,
    ) -> float:
        if sup_slope == res_slope:
            return float("inf")
        x = (res_intercept - sup_intercept) / (sup_slope - res_slope)
        return line_at(sup_slope, sup_intercept, x)

    def _calculate_confidence(
        self, res_slope: float, sup_slope: float, height: float, res_end: float
    ) -> float:
        flatness = max(0.0, 1.0 - abs(res_slope) / (res_end * 0.01))
        convergence = min(1.0, max(0.0, sup_slope / (res_end * 0.005)))
        ratio = min(1.0, height / res_end * 20)

        confidence = 0.4
        confidence += flatness * 0.2
        confidence += convergence * 0.2
        confidence += ratio * 0.2

        return min(1.0, confidence)
