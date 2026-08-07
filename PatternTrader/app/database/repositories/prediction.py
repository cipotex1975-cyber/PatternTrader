# mypy: ignore-errors
from __future__ import annotations

from app.database.base import get_async_session
from app.database.models import Prediction as PredictionORM
from app.ml.base import MLPrediction


class PredictionRepository:
    """Persistencia de predicciones ML."""

    async def add(self, prediction: MLPrediction) -> None:
        async with get_async_session() as session:
            session.add(
                PredictionORM(
                    model_name=prediction.model_name,
                    symbol=prediction.symbol,
                    timeframe=prediction.timeframe,
                    pattern_name=prediction.pattern_name,
                    probability=prediction.probability,
                    confidence=prediction.confidence,
                    features_used=prediction.features_used,
                    created_at=prediction.timestamp,
                    metadata_json=prediction.metadata or {},
                )
            )
            await session.flush()
