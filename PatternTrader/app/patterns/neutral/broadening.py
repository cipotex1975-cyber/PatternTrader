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

logger = get_logger("BroadeningPattern")


@register_pattern
class BroadeningPattern(BasePattern):
    @property
    def name(self) -> str:
        return "broadening"

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
        if len(candles) < 25:
            return None

        highs = np.array([c.data.high for c in candles], dtype=float)
        lows = np.array([c.data.low for c in candles], dtype=float)
        n = len(candles)

        res_slope, res_intercept = fit_line(highs)
        sup_slope, sup_intercept = fit_line(lows)

        if res_slope <= 0.0001 * line_at(res_slope, res_intercept, n - 1):
            return None
        if sup_slope >= -0.0001 * line_at(sup_slope, sup_intercept, n - 1):
            return None

        start_range = line_at(res_slope, res_intercept, 0) - line_at(sup_slope, sup_intercept, 0)
        end_range = line_at(res_slope, res_intercept, n - 1) - line_at(
            sup_slope, sup_intercept, n - 1
        )

        if start_range <= 0 or end_range <= start_range:
            return None

        resistance = line_at(res_slope, res_intercept, n - 1)
        support = line_at(sup_slope, sup_intercept, n - 1)
        if support >= resistance:
            return None

        height = resistance - support
        if height / resistance < 0.03:
            return None

        return PatternResult(
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            symbol=symbol,
            timeframe=timeframe,
            direction=TradeDirection.LONG,
            confidence=self._calculate_confidence(start_range, end_range, res_slope, sup_slope),
            key_levels={
                "neckline": resistance,
                "resistance": resistance,
                "support": support,
                "valley": float(lows.min()),
                "target": resistance + height,
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

    def _calculate_confidence(
        self, start_range: float, end_range: float, res_slope: float, sup_slope: float
    ) -> float:
        widening = min(1.0, max(0.0, (end_range / start_range) - 1))
        divergence = min(1.0, abs(res_slope - sup_slope) / (end_range * 0.01))

        confidence = 0.5
        confidence += widening * 0.25
        confidence += divergence * 0.25

        return min(1.0, confidence)
