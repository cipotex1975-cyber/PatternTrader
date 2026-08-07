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

logger = get_logger("DiamondPattern")


@register_pattern
class DiamondPattern(BasePattern):
    @property
    def name(self) -> str:
        return "diamond"

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

        mid = n // 2

        contract_highs = highs[:mid]
        contract_lows = lows[:mid]
        expand_highs = highs[mid:]
        expand_lows = lows[mid:]

        if len(contract_highs) < 5 or len(expand_highs) < 5:
            return None

        start_range = contract_highs.max() - contract_lows.min()
        end_range = expand_highs.max() - expand_lows.min()

        if start_range <= 0 or end_range <= start_range:
            return None

        min_gap = contract_highs[-1] - contract_lows[-1]
        if min_gap <= 0 or min_gap > start_range * 0.6:
            return None

        resistance = float(highs.max())
        support = float(lows.min())
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
            confidence=self._calculate_confidence(start_range, end_range, min_gap),
            key_levels={
                "neckline": resistance,
                "resistance": resistance,
                "support": support,
                "valley": support,
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

    def _calculate_confidence(self, start_range: float, end_range: float, min_gap: float) -> float:
        contraction = min(1.0, 1.0 - min_gap / start_range)
        expansion = min(1.0, (end_range / start_range) - 1)

        confidence = 0.5
        confidence += contraction * 0.25
        confidence += expansion * 0.25

        return min(1.0, confidence)
