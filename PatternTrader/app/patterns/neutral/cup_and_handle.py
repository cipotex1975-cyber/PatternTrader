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
from app.patterns.registry import register_pattern

logger = get_logger("CupAndHandlePattern")


@register_pattern
class CupAndHandlePattern(BasePattern):
    @property
    def name(self) -> str:
        return "cup_and_handle"

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.CONTINUATION

    @property
    def max_confirmation_candles(self) -> int:
        return 20

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

        cup_start = highs[: n // 2].mean()
        cup_end = highs[n // 2 :].mean()
        if abs(cup_start - cup_end) / cup_start > 0.03:
            return None

        valley_idx = int(np.argmin(lows))
        valley = lows[valley_idx]
        neckline = (cup_start + cup_end) / 2

        if valley <= 0 or neckline <= valley:
            return None

        depth = neckline - valley
        if depth / neckline < 0.05:
            return None

        left_rise = highs[0] - valley
        if left_rise / depth < 0.3:
            return None

        handle = lows[valley_idx:]
        if len(handle) < 5:
            return None

        handle_peak = np.max(highs[valley_idx:])
        handle_low = np.min(handle)
        if handle_peak > neckline * 1.05:
            return None

        return PatternResult(
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            symbol=symbol,
            timeframe=timeframe,
            direction=TradeDirection.LONG,
            confidence=self._calculate_confidence(depth, neckline, handle_low, handle_peak),
            key_levels={
                "neckline": neckline,
                "valley": float(valley),
                "handle_low": float(handle_low),
                "handle_high": float(handle_peak),
                "target": neckline + depth,
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
        self, depth: float, neckline: float, handle_low: float, handle_peak: float
    ) -> float:
        depth_ratio = min(1.0, depth / neckline * 20)
        handle_drawdown = min(1.0, (neckline - handle_low) / depth)
        handle_contained = max(0.0, 1.0 - max(0.0, handle_peak - neckline) / (neckline * 0.05))

        confidence = 0.4
        confidence += depth_ratio * 0.2
        confidence += handle_drawdown * 0.2
        confidence += handle_contained * 0.2

        return min(1.0, confidence)
