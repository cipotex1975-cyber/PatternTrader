from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.config.settings import get_settings
from app.core.logger import get_logger
from app.market.candles.models import Candle
from app.ml.models.random_forest import RandomForestModel
from app.patterns.base_pattern import PatternResult
from app.scoring.models import ScoreComponent, ScoreResult

logger = get_logger("ScoringEngine")


class ScoringEngine:
    def __init__(self) -> None:
        settings = get_settings()
        self._weights = settings.scoring.weights
        self._ml_model: RandomForestModel | None = None
        self._load_ml_model(settings.ml.model_path)

    def _load_ml_model(self, model_path: str) -> None:
        """Load trained ML model if available."""
        model_dir = Path(model_path)
        if not model_dir.exists():
            logger.warning(f"ML model directory not found: {model_dir}")
            return

        for model_file in model_dir.glob("*.pkl"):
            try:
                self._ml_model = RandomForestModel()
                self._ml_model.load(str(model_file))
                logger.info(f"Loaded ML model: {model_file.name}")
                break
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")

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

        ml_features = self._extract_ml_features(indicators, candles)
        ml_score = self._get_ml_score(ml_features)
        components.append(
            ScoreComponent(
                name="ml_history",
                weight=self._weights.ml_history,
                value=ml_score,
                score=ml_score,
                reason=f"ML prediction: {ml_score:.1f}%",
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

    def _extract_ml_features(
        self,
        indicators: dict[str, float],
        candles: list[Candle] | None,
    ) -> np.ndarray | None:
        """Extract features for ML model prediction."""
        if not candles or len(candles) < 20:
            return None

        try:
            closes = np.array([c.data.close for c in candles])
            highs = np.array([c.data.high for c in candles])
            lows = np.array([c.data.low for c in candles])
            volumes = np.array([c.data.volume for c in candles])

            def ema(series: np.ndarray, period: int) -> float:
                alpha = 2 / (period + 1)
                result = series[0]
                for val in series[1:]:
                    result = alpha * val + (1 - alpha) * result
                return result

            def rsi(series: np.ndarray, period: int = 14) -> float:
                deltas = np.diff(series)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains[-period:])
                avg_loss = np.mean(losses[-period:])
                if avg_loss == 0:
                    return 100.0
                rs = avg_gain / avg_loss
                return 100 - (100 / (1 + rs))

            def macd_calc(series: np.ndarray) -> tuple[float, float, float]:
                ema12 = ema(series, 12)
                ema26 = ema(series, 26)
                macd_line = ema12 - ema26
                signal_line = ema(np.array([macd_line]), 9)
                histogram = macd_line - signal_line
                return macd_line, signal_line, histogram

            def atr_calc(
                highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
            ) -> float:
                tr1 = highs[-period:] - lows[-period:]
                tr2 = np.abs(highs[-period:] - closes[-period - 1 : -1])
                tr3 = np.abs(lows[-period:] - closes[-period - 1 : -1])
                tr = np.maximum(np.maximum(tr1, tr2), tr3)
                return np.mean(tr)

            ema_21 = ema(closes, 21)
            ema_50 = ema(closes, 50)
            rsi_val = rsi(closes)
            macd_line, signal_line, histogram = macd_calc(closes)
            atr_val = atr_calc(highs, lows, closes)

            vol_avg = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
            volume_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1.0

            price_change = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] != 0 else 0
            high_low_range = (highs[-1] - lows[-1]) / closes[-1] if closes[-1] != 0 else 0
            close_position = (
                (closes[-1] - lows[-1]) / (highs[-1] - lows[-1])
                if (highs[-1] - lows[-1]) != 0
                else 0.5
            )
            trend_strength = (ema_21 - ema_50) / ema_50 if ema_50 != 0 else 0

            features = np.array(
                [
                    rsi_val,
                    macd_line,
                    signal_line,
                    histogram,
                    ema_21,
                    ema_50,
                    atr_val,
                    volume_ratio,
                    price_change,
                    high_low_range,
                    close_position,
                    trend_strength,
                ]
            )

            return features

        except Exception as e:
            logger.error(f"Failed to extract ML features: {e}")
            return None

    def _get_ml_score(self, features: np.ndarray | None) -> float:
        """Get ML model prediction score."""
        if features is None or self._ml_model is None or not self._ml_model.is_trained:
            return 50.0

        try:
            proba = self._ml_model.predict_proba(features.reshape(1, -1))
            return float(proba[0][1]) * 100
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return 50.0
