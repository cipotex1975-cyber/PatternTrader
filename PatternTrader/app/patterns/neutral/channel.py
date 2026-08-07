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

logger = get_logger("ChannelPattern")


@register_pattern
class ChannelPattern(BasePattern):
    @property
    def name(self) -> str:
        return "channel"

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

        res_slope, res_intercept = fit_line(highs)
        sup_slope, sup_intercept = fit_line(lows)

        n = len(candles)
        res_end = line_at(res_slope, res_intercept, n - 1)
        sup_end = line_at(sup_slope, sup_intercept, n - 1)

        parallel_threshold = (abs(res_slope) + abs(sup_slope)) / 2
        if abs(res_slope - sup_slope) > parallel_threshold * 0.5 + 1e-9:
            return None
        if abs(res_slope) < 0.0005 * res_end:
            return None
        if sup_end >= res_end:
            return None

        height = res_end - sup_end
        if height / res_end < 0.02:
            return None

        return PatternResult(
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            symbol=symbol,
            timeframe=timeframe,
            direction=TradeDirection.LONG,
            confidence=self._calculate_confidence(
                res_slope, sup_slope, height, res_end
            ),
            key_levels={
                "neckline": res_end,
                "resistance": line_at(res_slope, res_intercept, 0),
                "support": sup_end,
                "valley": float(lows.min()),
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

    def _calculate_confidence(
        self, res_slope: float, sup_slope: float, height: float, res_end: float
    ) -> float:
        slope_mag = min(1.0, abs(res_slope) / (res_end * 0.01))
        parallel = min(
            1.0,
            1.0 - abs(res_slope - sup_slope) / (res_end * 0.005),
        )
        ratio = min(1.0, height / res_end * 20)

        confidence = 0.4
        confidence += slope_mag * 0.2
        confidence += parallel * 0.2
        confidence += ratio * 0.2

        return min(1.0, confidence)
