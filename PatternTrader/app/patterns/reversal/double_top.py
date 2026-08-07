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

logger = get_logger("DoubleTopPattern")


@register_pattern
class DoubleTopPattern(BasePattern):
    @property
    def name(self) -> str:
        return "double_top"

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

        peaks = self._find_peaks(highs, distance=3)
        if len(peaks) < 2:
            return None

        for i in range(len(peaks) - 1):
            peak1_idx = peaks[i]
            peak2_idx = peaks[i + 1]

            peak1_price = highs[peak1_idx]
            peak2_price = highs[peak2_idx]

            if abs(peak1_price - peak2_price) / peak1_price > 0.02:
                continue

            valley_idx = np.argmin(lows[peak1_idx:peak2_idx]) + peak1_idx
            neckline = lows[valley_idx]

            if peak1_price <= neckline:
                continue

            pattern_height = peak1_price - neckline
            if pattern_height / peak1_price < 0.01:
                continue

            if closes[-1] >= neckline:
                continue

            return PatternResult(
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                direction=TradeDirection.SHORT,
                confidence=self._calculate_confidence(peak1_price, peak2_price, neckline),
                key_levels={
                    "peak1": peak1_price,
                    "peak2": peak2_price,
                    "neckline": neckline,
                    "target": neckline - pattern_height,
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
        return latest_close > neckline * 0.98

    def _find_peaks(self, data: np.ndarray, distance: int = 3) -> list[int]:
        idx, _ = _scipy_find_peaks(data, distance=distance)
        return idx.tolist()

    def _calculate_confidence(self, peak1: float, peak2: float, neckline: float) -> float:
        peak_diff = abs(peak1 - peak2) / max(peak1, peak2)
        height = peak1 - neckline
        ratio = height / peak1

        confidence = 0.5
        confidence += (1 - peak_diff) * 0.2
        confidence += min(ratio * 10, 0.3)

        return min(1.0, confidence)
