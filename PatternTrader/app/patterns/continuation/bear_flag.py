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

logger = get_logger("BearFlagPattern")


@register_pattern
class BearFlagPattern(BasePattern):
    @property
    def name(self) -> str:
        return "bear_flag"

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

        flag_highs = highs[pole_end:]
        flag_lows = lows[pole_end:]

        if len(flag_highs) < 5:
            return None

        flag_slope = self._calculate_slope(flag_lows)
        if flag_slope < -0.001:
            return None

        pole_height = highs[0] - lows[pole_end]
        if pole_height <= 0:
            return None

        flag_height = np.max(flag_highs) - np.min(flag_lows)
        if flag_height > pole_height * 0.6:
            return None

        pattern_high = np.max(flag_highs)
        pattern_low = lows[pole_end]

        return PatternResult(
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            symbol=symbol,
            timeframe=timeframe,
            direction=TradeDirection.SHORT,
            confidence=self._calculate_confidence(pole_height, flag_height, flag_slope),
            key_levels={
                "pole_low": pattern_low,
                "flag_high": pattern_high,
                "pole_height": pole_height,
                "target": pattern_low - pole_height,
            },
            max_confirmation_candles=self.max_confirmation_candles,
        )

    def validate(self, pattern: PatternResult, candles: list[Candle]) -> bool:
        if not pattern.key_levels:
            return False

        flag_high = pattern.key_levels.get("flag_high", float("inf"))
        if not candles:
            return False

        latest_close = candles[-1].data.close
        return latest_close < flag_high * 1.02

    def score(self, pattern: PatternResult, indicators: dict[str, float]) -> float:
        score = 55.0

        ema_21 = indicators.get("ema_21", 0)
        ema_50 = indicators.get("ema_50", 0)
        if ema_21 < ema_50:
            score += 15

        rsi = indicators.get("rsi", 50)
        if 30 < rsi < 60:
            score += 10

        volume = indicators.get("volume", 0)
        if volume > 0:
            score += 5

        return min(100.0, score)

    def _find_pole_end(self, highs: np.ndarray, lows: np.ndarray) -> Optional[int]:
        search_end = max(len(lows) // 3, 10)
        min_idx = np.argmin(lows[:search_end])
        return min_idx if min_idx > 0 else None

    def _calculate_slope(self, values: np.ndarray) -> float:
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]

    def _calculate_confidence(self, pole_height: float, flag_height: float, slope: float) -> float:
        ratio = flag_height / pole_height if pole_height > 0 else 0
        confidence = 0.5
        confidence += min(pole_height / 100, 0.2)
        confidence += (1 - ratio) * 0.15
        confidence += min(abs(slope) * 100, 0.15)
        return min(1.0, confidence)
