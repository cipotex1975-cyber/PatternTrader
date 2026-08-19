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

logger = get_logger("RoundedBottomPattern")


@register_pattern
class RoundedBottomPattern(BasePattern):
    @property
    def name(self) -> str:
        return "rounded_bottom"

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
        if len(candles) < 30:
            return None

        lows = np.array([c.data.low for c in candles], dtype=float)

        x = np.arange(len(lows), dtype=float)
        coeffs = np.polyfit(x, lows, 2)
        a, b, c = coeffs

        if a <= 0:
            return None

        vertex_x = -b / (2 * a)
        if not (0 < vertex_x < len(lows) - 1):
            return None

        vertex_y = a * vertex_x**2 + b * vertex_x + c
        left_y = a * 0**2 + b * 0 + c
        right_y = a * (len(lows) - 1) ** 2 + b * (len(lows) - 1) + c

        if left_y <= vertex_y or right_y <= vertex_y:
            return None

        height = min(left_y, right_y) - vertex_y
        if height / left_y < 0.05:
            return None

        neckline = min(left_y, right_y)
        if lows.min() > vertex_y * 1.001:
            return None

        return PatternResult(
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            symbol=symbol,
            timeframe=timeframe,
            direction=TradeDirection.LONG,
            confidence=self._calculate_confidence(vertex_x, len(lows), height, left_y),
            key_levels={
                "neckline": float(neckline),
                "valley": float(lows.min()),
                "target": float(neckline + height),
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

    def _calculate_confidence(self, vertex_x: float, n: int, height: float, left_y: float) -> float:
        normalized = vertex_x / max(1, n - 1)
        centered = 1.0 - min(1.0, abs(normalized - 0.5) * 2)
        depth = min(1.0, height / left_y * 20)

        confidence = 0.4
        confidence += centered * 0.3
        confidence += depth * 0.3

        return min(1.0, confidence)
