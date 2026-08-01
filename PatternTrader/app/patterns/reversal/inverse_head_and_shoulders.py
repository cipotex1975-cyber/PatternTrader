from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import find_peaks as _scipy_find_peaks

from app.core.logger import get_logger
from app.market.candles.models import Candle
from app.patterns.base_pattern import (
    BasePattern,
    PatternResult,
    PatternType,
    TradeDirection,
)
from app.patterns.registry import register_pattern

logger = get_logger("InverseHeadAndShouldersPattern")


@register_pattern
class InverseHeadAndShouldersPattern(BasePattern):
    @property
    def name(self) -> str:
        return "inverse_head_and_shoulders"

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
        if len(candles) < 15:
            return None

        highs = np.array([c.data.high for c in candles])
        lows = np.array([c.data.low for c in candles])

        troughs = self._find_troughs(lows, distance=3)
        if len(troughs) < 3:
            return None

        for i in range(len(troughs) - 2):
            left_idx = troughs[i]
            head_idx = troughs[i + 1]
            right_idx = troughs[i + 2]

            left_price = lows[left_idx]
            head_price = lows[head_idx]
            right_price = lows[right_idx]

            if head_price >= left_price or head_price >= right_price:
                continue

            if abs(left_price - right_price) / head_price > 0.03:
                continue

            neckline1_idx = np.argmax(highs[left_idx:head_idx]) + left_idx
            neckline2_idx = np.argmax(highs[head_idx:right_idx]) + head_idx
            neckline = (highs[neckline1_idx] + highs[neckline2_idx]) / 2

            if head_price >= neckline:
                continue

            pattern_height = neckline - head_price
            if pattern_height / neckline < 0.01:
                continue

            return PatternResult(
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                direction=TradeDirection.LONG,
                confidence=self._calculate_confidence(
                    left_price, head_price, right_price, neckline
                ),
                key_levels={
                    "left_shoulder": left_price,
                    "head": head_price,
                    "right_shoulder": right_price,
                    "neckline": neckline,
                    "target": neckline + pattern_height,
                },
                max_confirmation_candles=self.max_confirmation_candles,
            )

        return None

    def validate(self, pattern: PatternResult, candles: list[Candle]) -> bool:
        if not pattern.key_levels:
            return False

        neckline = pattern.key_levels.get("neckline", float("inf"))
        if not candles:
            return False

        latest_close = candles[-1].data.close
        return latest_close < neckline * 1.02

    def score(self, pattern: PatternResult, indicators: dict[str, float]) -> float:
        score = 55.0

        rsi = indicators.get("rsi", 50)
        if rsi < 30:
            score += 15
        elif rsi < 40:
            score += 10

        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        if macd > macd_signal:
            score += 10

        volume = indicators.get("volume", 0)
        if volume > 0:
            score += 5

        return min(100.0, score)

    def _find_troughs(self, data: np.ndarray, distance: int = 3) -> list[int]:
        idx, _ = _scipy_find_peaks(-data, distance=distance)
        return idx.tolist()

    def _calculate_confidence(
        self, left: float, head: float, right: float, neckline: float
    ) -> float:
        shoulder_diff = abs(left - right) / max(left, right)
        height = neckline - head
        ratio = height / neckline

        confidence = 0.5
        confidence += (1 - shoulder_diff) * 0.2
        confidence += min(ratio * 10, 0.3)

        return min(1.0, confidence)
