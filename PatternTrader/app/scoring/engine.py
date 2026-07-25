from __future__ import annotations

from typing import Optional

from app.core.config.settings import get_settings
from app.core.logger import get_logger
from app.market.candles.models import Candle
from app.patterns.base_pattern import PatternResult
from app.scoring.models import ScoreComponent, ScoreResult

logger = get_logger("ScoringEngine")


class ScoringEngine:
    def __init__(self) -> None:
        settings = get_settings()
        self._weights = settings.scoring.weights

    def calculate_score(
        self,
        pattern: PatternResult,
        indicators: dict[str, float],
        candles: list[Candle] | None = None,
    ) -> ScoreResult:
        components: list[ScoreComponent] = []

        pattern_score = self._score_pattern_structure(pattern)
        components.append(
            ScoreComponent(
                name="pattern_structure",
                weight=self._weights.pattern_structure,
                value=pattern.confidence * 100,
                score=pattern_score,
                reason=f"Pattern confidence: {pattern.confidence:.2%}",
            )
        )

        volume_score = self._score_volume(indicators, candles)
        components.append(
            ScoreComponent(
                name="volume",
                weight=self._weights.volume,
                value=indicators.get("volume", 0),
                score=volume_score,
                reason="Volume analysis",
            )
        )

        momentum_score = self._score_momentum(indicators)
        components.append(
            ScoreComponent(
                name="momentum",
                weight=self._weights.momentum,
                value=indicators.get("rsi", 50),
                score=momentum_score,
                reason="Momentum indicators",
            )
        )

        atr_score = self._score_atr(indicators, candles)
        components.append(
            ScoreComponent(
                name="atr",
                weight=self._weights.atr,
                value=indicators.get("atr", 0),
                score=atr_score,
                reason="ATR volatility",
            )
        )

        rsi_score = self._score_rsi(indicators)
        components.append(
            ScoreComponent(
                name="rsi",
                weight=self._weights.rsi,
                value=indicators.get("rsi", 50),
                score=rsi_score,
                reason="RSI overbought/oversold",
            )
        )

        macd_score = self._score_macd(indicators)
        components.append(
            ScoreComponent(
                name="macd",
                weight=self._weights.macd,
                value=indicators.get("macd", 0),
                score=macd_score,
                reason="MACD signal",
            )
        )

        ema_score = self._score_ema(indicators)
        components.append(
            ScoreComponent(
                name="ema",
                weight=self._weights.ema,
                value=indicators.get("ema_21", 0),
                score=ema_score,
                reason="EMA trend alignment",
            )
        )

        ml_score = 50.0
        components.append(
            ScoreComponent(
                name="ml_history",
                weight=self._weights.ml_history,
                value=ml_score,
                score=ml_score,
                reason="ML model prediction (placeholder)",
            )
        )

        total_score = sum(c.weight * c.score for c in components)
        total_score = max(0.0, min(100.0, total_score))

        grade = self._calculate_grade(total_score)
        confidence = self._calculate_confidence(components)

        return ScoreResult(
            total_score=total_score,
            components=components,
            grade=grade,
            confidence=confidence,
            metadata={
                "pattern_name": pattern.pattern_name,
                "symbol": pattern.symbol,
                "timeframe": pattern.timeframe,
            },
        )

    def _score_pattern_structure(self, pattern: PatternResult) -> float:
        return pattern.confidence * 100

    def _score_volume(self, indicators: dict[str, float], candles: list[Candle] | None) -> float:
        if not candles or len(candles) < 20:
            return 50.0

        recent_volume = sum(c.data.volume for c in candles[-5:]) / 5
        avg_volume = sum(c.data.volume for c in candles[-20:]) / 20

        if avg_volume == 0:
            return 50.0

        ratio = recent_volume / avg_volume
        if ratio > 1.5:
            return 90.0
        elif ratio > 1.2:
            return 75.0
        elif ratio > 0.8:
            return 60.0
        else:
            return 40.0

    def _score_momentum(self, indicators: dict[str, float]) -> float:
        rsi = indicators.get("rsi", 50)
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)

        score = 50.0
        if 40 < rsi < 60:
            score += 10
        elif rsi > 70 or rsi < 30:
            score -= 10

        if macd > macd_signal:
            score += 10
        else:
            score -= 10

        return max(0.0, min(100.0, score))

    def _score_atr(self, indicators: dict[str, float], candles: list[Candle] | None) -> float:
        if not candles or len(candles) < 20:
            return 50.0

        current_atr = indicators.get("atr", 0)
        if current_atr == 0:
            return 50.0

        avg_price = sum(c.data.close for c in candles[-20:]) / 20
        if avg_price == 0:
            return 50.0

        atr_pct = (current_atr / avg_price) * 100

        if 0.5 < atr_pct < 2.0:
            return 80.0
        elif 0.3 < atr_pct < 3.0:
            return 60.0
        else:
            return 40.0

    def _score_rsi(self, indicators: dict[str, float]) -> float:
        rsi = indicators.get("rsi", 50)

        if 40 < rsi < 60:
            return 70.0
        elif 30 < rsi < 70:
            return 60.0
        elif rsi > 80 or rsi < 20:
            return 40.0
        else:
            return 50.0

    def _score_macd(self, indicators: dict[str, float]) -> float:
        macd = indicators.get("macd", 0)
        signal = indicators.get("macd_signal", 0)
        histogram = indicators.get("macd_histogram", 0)

        score = 50.0
        if macd > signal:
            score += 15
        if histogram > 0:
            score += 10

        return max(0.0, min(100.0, score))

    def _score_ema(self, indicators: dict[str, float]) -> float:
        ema_21 = indicators.get("ema_21", 0)
        ema_50 = indicators.get("ema_50", 0)
        ema_200 = indicators.get("ema_200", 0)

        score = 50.0
        if ema_21 > ema_50:
            score += 15
        if ema_50 > ema_200:
            score += 10

        return max(0.0, min(100.0, score))

    def _calculate_grade(self, score: float) -> str:
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 55:
            return "C"
        elif score >= 50:
            return "C-"
        else:
            return "D"

    def _calculate_confidence(self, components: list[ScoreComponent]) -> float:
        if not components:
            return 0.0

        weights_sum = sum(c.weight for c in components)
        if weights_sum == 0:
            return 0.0

        avg_score = sum(c.score * c.weight for c in components) / weights_sum
        return avg_score / 100.0
