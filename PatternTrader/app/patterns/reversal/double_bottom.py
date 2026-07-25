from __future__ import annotations

from typing import Optional

import numpy as np

from app.core.logger import get_logger
from app.market.candles.models import Candle
from app.patterns.base_pattern import BasePattern, PatternResult, PatternType, PatternStatus
from app.patterns.registry import register_pattern

logger = get_logger("DoubleBottomPattern")


@register_pattern
class DoubleBottomPattern(BasePattern):
    @property
    def name(self) -> str:
        return "double_bottom"

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.REVERSAL

    @property
    def max_confirmation_candles(self) -> int:
        return 20

    def detect(
        self,
        candles: list[Candle],
        symbol: str,
        timeframe: str,
    ) -> Optional[PatternResult]:
        if len(candles) < 10:
            return None

        highs = np.array([c.data.high for c in candles])
        lows = np.array([c.data.low for c in candles])
        closes = np.array([c.data.close for c in candles])

        troughs = self._find_troughs(lows, distance=3)
        if len(troughs) < 2:
            return None

        for i in range(len(troughs) - 1):
            trough1_idx = troughs[i]
            trough2_idx = troughs[i + 1]

            trough1_price = lows[trough1_idx]
            trough2_price = lows[trough2_idx]

            if abs(trough1_price - trough2_price) / trough1_price > 0.02:
                continue

            peak_idx = np.argmax(highs[trough1_idx:trough2_idx]) + trough1_idx
            neckline = highs[peak_idx]

            if trough1_price >= neckline:
                continue

            pattern_height = neckline - trough1_price
            if pattern_height / neckline < 0.01:
                continue

            return PatternResult(
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                confidence=self._calculate_confidence(trough1_price, trough2_price, neckline),
                key_levels={
                    "trough1": trough1_price,
                    "trough2": trough2_price,
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
        score = 50.0

        rsi = indicators.get("rsi", 50)
        if rsi < 30:
            score += 15
        elif rsi < 40:
            score += 10

        volume = indicators.get("volume", 0)
        if volume > 0:
            score += 5

        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        if macd > macd_signal:
            score += 10

        bb_lower = indicators.get("bb_lower", 0)
        if bb_lower and pattern.key_levels.get("trough1", float("inf")) <= bb_lower:
            score += 10

        return min(100.0, score)

    def _find_troughs(self, data: np.ndarray, distance: int = 3) -> list[int]:
        troughs = []
        for i in range(distance, len(data) - distance):
            if all(data[i] <= data[i - j] for j in range(1, distance + 1)) and all(
                data[i] <= data[i + j] for j in range(1, distance + 1)
            ):
                troughs.append(i)
        return troughs

    def _calculate_confidence(self, trough1: float, trough2: float, neckline: float) -> float:
        trough_diff = abs(trough1 - trough2) / max(trough1, trough2)
        height = neckline - trough1
        ratio = height / neckline

        confidence = 0.5
        confidence += (1 - trough_diff) * 0.2
        confidence += min(ratio * 10, 0.3)

        return min(1.0, confidence)
