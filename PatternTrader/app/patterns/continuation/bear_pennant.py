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

logger = get_logger("BearPennantPattern")


@register_pattern
class BearPennantPattern(BasePattern):
    @property
    def name(self) -> str:
        return "bear_pennant"

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.CONTINUATION

    @property
    def max_confirmation_candles(self) -> int:
        return 12

    def detect(
        self,
        candles: list[Candle],
        symbol: str,
        timeframe: str,
    ) -> Optional[PatternResult]:
        if len(candles) < 15:
            return None

        highs = np.array([c.data.high for c in candles])
        lows = np.array([c.data.low for c in candles])

        pole_end = self._find_pole_end(highs, lows)
        if pole_end is None or pole_end < 5:
            return None

        pennant_highs = highs[pole_end:]
        pennant_lows = lows[pole_end:]

        if len(pennant_highs) < 3:
            return None

        high_slope = self._calculate_slope(pennant_highs)
        low_slope = self._calculate_slope(pennant_lows)

        if high_slope <= 0 or low_slope >= 0:
            return None

        if abs(low_slope) < abs(high_slope) * 0.5:
            return None

        pole_height = highs[0] - lows[pole_end]
        if pole_height <= 0:
            return None

        pennant_height = np.max(pennant_highs) - np.min(pennant_lows)
        if pennant_height > pole_height * 0.4:
            return None

        pattern_high = np.max(pennant_highs)
        pattern_low = lows[pole_end]

        return PatternResult(
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            symbol=symbol,
            timeframe=timeframe,
            direction=TradeDirection.SHORT,
            confidence=self._calculate_confidence(
                pole_height, pennant_height, high_slope, low_slope
            ),
            key_levels={
                "pole_low": pattern_low,
                "pennant_high": pattern_high,
                "pole_height": pole_height,
                "target": pattern_low - pole_height,
            },
            max_confirmation_candles=self.max_confirmation_candles,
        )

    def validate(self, pattern: PatternResult, candles: list[Candle]) -> bool:
        if not pattern.key_levels:
            return False

        pennant_high = pattern.key_levels.get("pennant_high", float("inf"))
        if not candles:
            return False

        latest_close = candles[-1].data.close
        return latest_close < pennant_high * 1.02

    def _find_pole_end(self, highs: np.ndarray, lows: np.ndarray) -> Optional[int]:
        min_idx = np.argmin(lows[: len(lows) // 2])
        return min_idx if min_idx > 0 else None

    def _calculate_slope(self, values: np.ndarray) -> float:
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]

    def _calculate_confidence(
        self, pole_height: float, pennant_height: float, high_slope: float, low_slope: float
    ) -> float:
        ratio = pennant_height / pole_height if pole_height > 0 else 0
        confidence = 0.5
        confidence += min(pole_height / 100, 0.2)
        confidence += (1 - ratio) * 0.15
        confidence += min(abs(high_slope - low_slope) * 100, 0.15)
        return min(1.0, confidence)
