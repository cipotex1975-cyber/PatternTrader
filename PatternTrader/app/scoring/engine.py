from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config.settings import get_settings
from app.core.logger import get_logger
from app.market.candles.models import Candle
from app.ml.base import BaseMLModel
from app.ml.factory import MLModelFactory
from app.ml.features import extract_technical_features, features_to_dict
from app.ml.models.random_forest import RandomForestModel
from app.patterns.base_pattern import PatternResult
from app.scoring.models import ScoreComponent, ScoreResult

logger = get_logger("ScoringEngine")


class ScoringEngine:
    def __init__(self) -> None:
        settings = get_settings()
        self._weights = settings.scoring.weights
        self._ml_model: BaseMLModel | None = None
        self._symbol_models: dict[str, BaseMLModel] = {}
        self._knowledge: Any = None
        self._model_path = settings.ml.model_path
        self._load_ml_model(self._model_path)

    def attach_knowledge(self, learning_service: Any) -> None:
        """Conecta el modelo de aprendizaje continuo al scoring.

        Cuando el modelo de conocimiento está entrenado, el componente
        ``ml_history`` usa su predicción en lugar del RandomForest estático.
        """
        self._knowledge = learning_service

    def _load_ml_model(self, model_path: str) -> None:
        """Load trained ML model if available.

        Carga un modelo genérico (fallback) desde los artefactos ``*.pkl`` que no
        tengan sidecar por par. Los modelos específicos de símbolo se cargan bajo
        demanda en ``_load_ml_model_for_symbol``.
        """
        model_dir = Path(model_path)
        if not model_dir.exists():
            logger.warning(f"ML model directory not found: {model_dir}")
            return

        per_symbol: set[str] = set()
        for meta_file in model_dir.glob("*.meta.json"):
            try:
                meta = json.loads(meta_file.read_text())
                model_name = meta.get("model_name")
                symbol = meta.get("symbol")
                ext = meta.get("extension", ".pkl")
                if model_name and symbol:
                    per_symbol.add(f"{model_name}_{symbol}{ext}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Sidecar ilegible {meta_file}: {e}")

        for model_file in model_dir.glob("*.pkl"):
            if model_file.name in per_symbol:
                continue
            try:
                self._ml_model = RandomForestModel()
                self._ml_model.load(str(model_file))
                logger.info(f"Loaded ML model: {model_file.name}")
                break
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")

    def _load_ml_model_for_symbol(self, symbol: str) -> BaseMLModel | None:
        """Carga (y cachea) el modelo específico de un par usando su sidecar.

        El sidecar ``{modelo}_{symbol}.meta.json`` identifica la clase del
        artefacto para rehidratarlo correctamente sin depender de la DB.
        """
        if symbol in self._symbol_models:
            return self._symbol_models[symbol]

        model_dir = Path(self._model_path)
        if not model_dir.exists():
            return None

        meta_files = sorted(model_dir.glob(f"*.{symbol}.meta.json"))
        if not meta_files:
            return None

        try:
            meta = json.loads(meta_files[0].read_text())
            model_name = meta["model_name"]
            ext = meta.get("extension", ".pkl")
            artifact = model_dir / f"{model_name}_{symbol}{ext}"
            model = MLModelFactory.create_new(model_name)
            model.load(str(artifact))
            self._symbol_models[symbol] = model
            logger.info(f"Loaded per-symbol ML model: {model_name} for {symbol}")
            return model
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to load per-symbol model for {symbol}: {e}")
            return None

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

        ml_features = self._extract_ml_features(candles)
        ml_score = self._get_ml_score(ml_features, pattern)
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
        candles: list[Candle] | None,
    ) -> np.ndarray | None:
        """Extract features for ML model prediction."""
        if not candles or len(candles) < 20:
            return None
        return extract_technical_features(candles)

    def _get_ml_score(
        self,
        features: np.ndarray | None,
        pattern: PatternResult | None = None,
    ) -> float:
        """Get ML model prediction score.

        Prefiere el modelo de aprendizaje continuo (alimentado con operaciones
        reales cerradas) cuando está entrenado; si no, usa el modelo específico
        del símbolo del patrón (entrenado por par con `train_and_compare.py`)
        y, como último recurso, el modelo genérico estático. Sin modelo
        disponible devuelve un score neutro de 50.
        """
        if (
            self._knowledge is not None
            and getattr(self._knowledge, "is_trained", False)
            and features is not None
        ):
            indicators = features_to_dict(features)
            prediction = self._knowledge.predict(
                indicators=indicators,
                variables={},
                instrument=pattern.symbol if pattern else "",
                timeframe=pattern.timeframe if pattern else "",
                pattern=pattern.pattern_name if pattern else "",
            )
            return max(0.0, min(100.0, prediction.probability * 100))

        if features is None:
            return 50.0

        model: BaseMLModel | None = None
        if pattern is not None and getattr(pattern, "symbol", ""):
            model = self._load_ml_model_for_symbol(pattern.symbol)

        if model is None:
            model = self._ml_model

        if model is None or not model.is_trained:
            return 50.0

        try:
            prediction = model.get_prediction(
                features,
                symbol=pattern.symbol if pattern else "",
                timeframe=pattern.timeframe if pattern else "",
                pattern_name=pattern.pattern_name if pattern else "",
            )
            return max(0.0, min(100.0, prediction.probability * 100))
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return 50.0
