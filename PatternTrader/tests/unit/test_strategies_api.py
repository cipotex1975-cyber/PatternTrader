import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.strategies import router
from app.strategy.manager import StrategyManager


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router, prefix="/strategies")
    app.state.strategy_manager = StrategyManager()
    return TestClient(app)


def test_list_strategies(client):
    resp = client.get("/strategies/")
    assert resp.status_code == 200
    names = {s["name"] for s in resp.json()["strategies"]}
    assert {"trend_follow", "breakout", "contrarian"} <= names


def test_get_strategy(client):
    resp = client.get("/strategies/trend_follow")
    assert resp.status_code == 200
    strategy = resp.json()["strategy"]
    assert strategy["enabled"] is True
    assert strategy["name"] == "trend_follow"


def test_get_unknown_strategy_returns_404(client):
    assert client.get("/strategies/nope").status_code == 404


def test_patch_disable_strategy(client):
    resp = client.patch("/strategies/trend_follow", json={"enabled": False})
    assert resp.status_code == 200
    assert resp.json()["strategy"]["enabled"] is False


def test_patch_enable_strategy(client):
    client.patch("/strategies/breakout", json={"enabled": False})
    resp = client.patch("/strategies/breakout", json={"enabled": True})
    assert resp.status_code == 200
    assert resp.json()["strategy"]["enabled"] is True


def test_patch_set_parameters(client):
    resp = client.patch(
        "/strategies/trend_follow",
        json={"parameters": {"default_size": 3.0}},
    )
    assert resp.status_code == 200
    assert resp.json()["strategy"]["parameters"]["default_size"] == 3.0


def test_patch_empty_body_returns_422(client):
    resp = client.patch("/strategies/trend_follow", json={})
    assert resp.status_code == 422


def test_patch_unknown_strategy_returns_404(client):
    resp = client.patch("/strategies/nope", json={"enabled": False})
    assert resp.status_code == 404
