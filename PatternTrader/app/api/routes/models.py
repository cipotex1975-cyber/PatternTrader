from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_ml_model_repository, get_prediction_repository
from app.core.exceptions import ConfigurationError
from app.database.repositories import MLModelRepository, PredictionRepository
from app.ml.factory import MLModelFactory

router = APIRouter()


@router.get("/")
async def list_models(
    repo: MLModelRepository = Depends(get_ml_model_repository),
):
    registered = MLModelFactory.get_all()
    loaded = MLModelFactory.get_loaded()
    db_models = await repo.list()
    return {
        "registered": [
            {"name": name, "type": cls().model_type, "loaded": name in loaded}
            for name, cls in registered.items()
        ],
        "models": db_models,
    }


@router.get("/{name}")
async def get_model(
    name: str,
    repo: MLModelRepository = Depends(get_ml_model_repository),
):
    registered = MLModelFactory.get_all()
    record = await repo.get(name)

    if name in registered:
        instance = MLModelFactory.create(name)
        return {
            "name": name,
            "type": instance.model_type,
            "registered": True,
            "trained": instance.is_trained,
            "feature_importance": instance.get_feature_importance(),
            "record": record,
        }
    if record is not None:
        return {"name": name, "registered": False, "record": record}
    raise HTTPException(status_code=404, detail=f"Model {name} not found")


@router.post("/{name}/predict")
async def predict_model(
    name: str,
    payload: dict[str, Any],
    repo: PredictionRepository = Depends(get_prediction_repository),
):
    try:
        model = MLModelFactory.create(name)
    except ConfigurationError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    if not model.is_trained:
        raise HTTPException(status_code=400, detail=f"Model {name} is not trained")

    features = payload.get("features")
    if not features:
        raise HTTPException(status_code=400, detail="Missing 'features' list")

    prediction = model.get_prediction(
        features=np.asarray(features, dtype=float).reshape(1, -1),
        symbol=payload.get("symbol", "UNKNOWN"),
        timeframe=payload.get("timeframe", ""),
        pattern_name=payload.get("pattern_name", ""),
    )
    await repo.add(prediction)
    return prediction.model_dump(mode="json")
