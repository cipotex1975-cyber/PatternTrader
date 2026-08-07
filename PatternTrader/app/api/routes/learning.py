from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import (
    get_learning_service,
    get_prediction_repository,
)
from app.database.repositories import PredictionRepository
from app.learning.models import LearningMode, TradeOutcome
from app.learning.service import LearningService

router = APIRouter()


@router.get("/entries")
async def list_entries(
    instrument: Optional[str] = None,
    timeframe: Optional[str] = None,
    pattern: Optional[str] = None,
    outcome: Optional[TradeOutcome] = None,
    limit: int = 100,
    service: LearningService = Depends(get_learning_service),
):
    entries = await service.entries(
        instrument=instrument,
        timeframe=timeframe,
        pattern=pattern,
        outcome=outcome,
        limit=limit,
    )
    return {
        "total": len(entries),
        "entries": [e.model_dump(mode="json") for e in entries],
    }


@router.get("/stats")
async def get_stats(
    service: LearningService = Depends(get_learning_service),
):
    return await service.stats()


@router.post("/record")
async def record_trade(
    payload: dict[str, Any],
    service: LearningService = Depends(get_learning_service),
):
    try:
        entry = await service.record_trade(
            payload.get("trade") or payload,
            indicators=payload.get("indicators"),
            variables=payload.get("variables"),
            image_path=payload.get("image_path", ""),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"No se pudo registrar la operación: {e}"
        ) from e
    return {"recorded": True, "entry": entry.model_dump(mode="json")}


@router.post("/train")
async def train_offline(
    n_splits: int = 5,
    service: LearningService = Depends(get_learning_service),
):
    report = await service.train_offline(n_splits=n_splits)
    return report


@router.post("/predict")
async def predict(
    payload: dict[str, Any],
    service: LearningService = Depends(get_learning_service),
    predictions_repo: PredictionRepository = Depends(get_prediction_repository),
):
    prediction = service.predict(
        indicators=payload.get("indicators", {}),
        variables=payload.get("variables"),
        instrument=payload.get("instrument", ""),
        timeframe=payload.get("timeframe", ""),
        pattern=payload.get("pattern", ""),
    )
    await predictions_repo.add(prediction)
    return prediction.model_dump(mode="json")


@router.get("/mode")
async def get_mode(
    service: LearningService = Depends(get_learning_service),
):
    return {"mode": service.mode.value}


@router.post("/mode")
async def set_mode(
    mode: LearningMode,
    service: LearningService = Depends(get_learning_service),
):
    service.set_mode(mode)
    return {"mode": mode.value}
