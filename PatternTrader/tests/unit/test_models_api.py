import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.routes.models import router
from app.database import base
from app.database import models as orm_models
from app.database.repositories import MLModelRepository, PredictionRepository
from app.ml.factory import MLModelFactory
from app.ml.models.random_forest import RandomForestModel  # noqa: F401  (registra el modelo)


@pytest.fixture
async def client(sync_db):
    app = FastAPI()
    app.include_router(router, prefix="/models")
    app.state.ml_model_repository = MLModelRepository()
    app.state.prediction_repository = PredictionRepository()
    return TestClient(app)


@pytest.fixture
def trained_rf():
    MLModelFactory.clear()
    model = MLModelFactory.create("random_forest")
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 4))
    y = np.array([int(x[0] + x[1] > 0) for x in X])
    model.train(X, y, feature_names=["rsi", "atr", "volume", "trend"])
    return model


@pytest.mark.asyncio
async def test_list_models(client):
    resp = client.get("/models/")
    assert resp.status_code == 200
    body = resp.json()
    names = {m["name"] for m in body["registered"]}
    assert "random_forest" in names
    rf = next(m for m in body["registered"] if m["name"] == "random_forest")
    assert rf["type"] == "classification"
    assert rf["loaded"] is False
    assert body["models"] == []


@pytest.mark.asyncio
async def test_get_registered_model(client):
    resp = client.get("/models/random_forest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["registered"] is True
    assert body["trained"] is False
    assert body["feature_importance"] == {}
    assert body["record"] is None


@pytest.mark.asyncio
async def test_get_unknown_model_returns_404(client):
    resp = client.get("/models/nope")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_predict_untrained_model_returns_400(client):
    resp = client.post("/models/random_forest/predict", json={"features": [1.0, 2.0, 3.0, 4.0]})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_predict_missing_features_returns_400(client, trained_rf):
    resp = client.post("/models/random_forest/predict", json={"symbol": "BTCUSDT"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_predict_unknown_model_returns_404(client):
    resp = client.post("/models/nope/predict", json={"features": [1.0]})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_predict_trained_model_and_persists(client, trained_rf):
    resp = client.post(
        "/models/random_forest/predict",
        json={
            "features": [1.5, 0.2, 3.0, 2.0],
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "pattern_name": "double_top",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "random_forest"
    assert body["symbol"] == "BTCUSDT"
    assert 0.0 <= body["probability"] <= 1.0

    async with base.get_async_session() as session:
        result = await session.execute(select(orm_models.Prediction))
        rows = result.scalars().all()
    assert len(rows) == 1
    assert rows[0].model_name == "random_forest"


@pytest.mark.asyncio
async def test_get_model_after_training_has_importance(client, trained_rf):
    resp = client.get("/models/random_forest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["trained"] is True
    assert len(body["feature_importance"]) == 4
