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
from app.patterns.neutral.geometry import find_troughs
from app.patterns.registry import register_pattern

logger = get_logger("TripleBottomPattern")


@register_pattern
class TripleBottomPattern(BasePattern):
    @property
    def name(self) -> str:
        return "triple_bottom"

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

        troughs = find_troughs(lows, distance=3)
        if len(troughs) < 3:
            return None

        for i in range(len(troughs) - 2):
            t1, t2, t3 = troughs[i], troughs[i + 1], troughs[i + 2]
            prices = [lows[t1], lows[t2], lows[t3]]

            if max(prices) - min(prices) > max(prices) * 0.02:
                continue

            peak1 = highs[t1 : t2 + 1].max()
            peak2 = highs[t2 : t3 + 1].max()
            neckline = min(peak1, peak2)

            trough_price = min(prices)
            if trough_price >= neckline:
                continue

            height = neckline - trough_price
            if height / neckline < 0.01:
                continue

            return PatternResult(
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                direction=TradeDirection.LONG,
                confidence=self._calculate_confidence(prices, neckline),
                key_levels={
                    "trough1": float(lows[t1]),
                    "trough2": float(lows[t2]),
                    "trough3": float(lows[t3]),
                    "neckline": float(neckline),
                    "target": float(neckline + height),
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
        return latest_close < neckline * 1.02

    def _calculate_confidence(self, prices: list[float], neckline: float) -> float:
        spread = (max(prices) - min(prices)) / max(prices)
        height = neckline - min(prices)
        ratio = min(1.0, height / neckline * 20)

        confidence = 0.5
        confidence += (1 - spread) * 0.25
        confidence += ratio * 0.25

        return min(1.0, confidence)
