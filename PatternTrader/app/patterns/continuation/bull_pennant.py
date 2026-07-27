from __future__ import annotations

from typing import Optional

import numpy as np

from app.core.logger import get_logger
from app.market.candles.models import Candle
from app.patterns.base_pattern import BasePattern, PatternResult, PatternType, PatternStatus, TradeDirection
from app.patterns.registry import register_pattern

logger = get_logger("BullPennantPattern")


@register_pattern
class BullPennantPattern(BasePattern):
    @property
    def name(self) -> str:
        return "bull_pennant"

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

        if high_slope >= 0 or low_slope <= 0:
            return None

        if abs(high_slope) < abs(low_slope) * 0.5:
            return None

        pole_height = highs[pole_end] - lows[0]
        if pole_height <= 0:
            return None

        pennant_height = np.max(pennant_highs) - np.min(pennant_lows)
        if pennant_height > pole_height * 0.4:
            return None

        pattern_high = highs[pole_end]
        pattern_low = np.min(pennant_lows)

        return PatternResult(
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            symbol=symbol,
            timeframe=timeframe,
            direction=TradeDirection.LONG,
            confidence=self._calculate_confidence(pole_height, pennant_height, high_slope, low_slope),
            key_levels={
                "pole_high": pattern_high,
                "pennant_low": pattern_low,
                "pole_height": pole_height,
                "target": pattern_high + pole_height,
            },
            max_confirmation_candles=self.max_confirmation_candles,
        )

    def validate(self, pattern: PatternResult, candles: list[Candle]) -> bool:
        if not pattern.key_levels:
            return False

        pennant_low = pattern.key_levels.get("pennant_low", 0)
        if not candles:
            return False

        latest_close = candles[-1].data.close
        return latest_close > pennant_low * 0.98

    def score(self, pattern: PatternResult, indicators: dict[str, float]) -> float:
        score = 55.0

        ema_21 = indicators.get("ema_21", 0)
        ema_50 = indicators.get("ema_50", 0)
        if ema_21 > ema_50:
            score += 15

        rsi = indicators.get("rsi", 50)
        if 40 < rsi < 70:
            score += 10

        volume = indicators.get("volume", 0)
        if volume > 0:
            score += 5

        return min(100.0, score)

    def _find_pole_end(self, highs: np.ndarray, lows: np.ndarray) -> Optional[int]:
        max_idx = np.argmax(highs[: len(highs) // 2])
        return max_idx if max_idx > 0 else None

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
