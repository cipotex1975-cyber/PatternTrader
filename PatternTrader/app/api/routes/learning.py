from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from app.learning.models import KnowledgeEntry, LearningMode, TradeOutcome
from app.learning.service import LearningService
from app.learning.repository import MemoryKnowledgeRepository

router = APIRouter()

_learning_service = LearningService(repository=MemoryKnowledgeRepository())


@router.get("/entries")
async def list_entries(
    instrument: Optional[str] = None,
    timeframe: Optional[str] = None,
    pattern: Optional[str] = None,
    outcome: Optional[TradeOutcome] = None,
    limit: int = 100,
):
    entries = await _learning_service.entries(
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
async def get_stats():
    return await _learning_service.stats()


@router.post("/record")
async def record_trade(payload: dict[str, Any]):
    try:
        entry = await _learning_service.record_trade(
            payload.get("trade") or payload,
            indicators=payload.get("indicators"),
            variables=payload.get("variables"),
            image_path=payload.get("image_path", ""),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"No se pudo registrar la operación: {e}") from e
    return {"recorded": True, "entry": entry.model_dump(mode="json")}


@router.post("/train")
async def train_offline(n_splits: int = 5):
    report = await _learning_service.train_offline(n_splits=n_splits)
    return report


@router.post("/predict")
async def predict(payload: dict[str, Any]):
    prediction = _learning_service.predict(
        indicators=payload.get("indicators", {}),
        variables=payload.get("variables"),
        instrument=payload.get("instrument", ""),
        timeframe=payload.get("timeframe", ""),
        pattern=payload.get("pattern", ""),
    )
    return prediction.model_dump(mode="json")


@router.get("/mode")
async def get_mode():
    return {"mode": _learning_service.mode.value}


@router.post("/mode")
async def set_mode(mode: LearningMode):
    _learning_service.set_mode(mode)
    return {"mode": mode.value}
