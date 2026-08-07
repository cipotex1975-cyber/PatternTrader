import numpy as np
import pytest

from app.ml.factory import MLModelFactory
from app.ml.models.autoencoder import AutoEncoderModel
from app.ml.models.catboost_model import CatBoostModel
from app.ml.models.cnn_model import CNNModel
from app.ml.models.isolation_forest import IsolationForestModel
from app.ml.models.lightgbm_model import LightGBMModel
from app.ml.models.lstm_model import LSTMModel
from app.ml.models.random_forest import RandomForestModel
from app.ml.models.transformer_model import TransformerModel
from app.ml.models.xgboost_model import XGBoostModel

EXPECTED_MODELS = [
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "lstm",
    "transformer",
    "cnn",
    "isolation_forest",
    "autoencoder",
]


def test_all_models_registered_in_factory():
    registered = MLModelFactory.get_all()
    for name in EXPECTED_MODELS:
        assert name in registered, f"{name} no está registrado"


def _make_tabular_data():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (120, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _make_sequence_data():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (80, 30, 2))
    y = (X[:, :, 0].mean(axis=1) + X[:, :, 1].mean(axis=1) > 0).astype(int)
    return X, y


@pytest.mark.parametrize(
    "model_cls",
    [RandomForestModel, XGBoostModel, LightGBMModel, CatBoostModel],
)
def test_tabular_models_train_predict_evaluate(model_cls):
    X, y = _make_tabular_data()
    kwargs = (
        {"iterations": 20, "depth": 4}
        if model_cls is CatBoostModel
        else {"n_estimators": 20, "max_depth": 4}
    )
    model = model_cls(**kwargs)
    result = model.train(X, y, feature_names=[f"f{i}" for i in range(8)])

    assert model.is_trained
    assert "train_accuracy" in result

    preds = model.predict(X)
    assert preds.shape == y.shape

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)

    metrics = model.evaluate(X, y)
    assert 0 <= metrics["accuracy"] <= 1
    assert "roc_auc" in metrics

    importance = model.get_feature_importance()
    assert len(importance) == 8


@pytest.mark.parametrize(
    "model_cls",
    [RandomForestModel, XGBoostModel, LightGBMModel, CatBoostModel],
)
def test_tabular_models_save_load_roundtrip(model_cls, tmp_path):
    X, y = _make_tabular_data()
    kwargs = (
        {"iterations": 20, "depth": 4}
        if model_cls is CatBoostModel
        else {"n_estimators": 20, "max_depth": 4}
    )
    model = model_cls(**kwargs)
    model.train(X, y)

    path = tmp_path / f"{model.name}.bin"
    model.save(str(path))

    restored = model_cls()
    assert not restored.is_trained
    restored.load(str(path))
    assert restored.is_trained

    assert np.array_equal(model.predict(X), restored.predict(X))


def test_get_prediction_requires_trained():
    model = XGBoostModel()
    with pytest.raises(ValueError):
        model.get_prediction(np.zeros((1, 8)), "BTCUSDT", "1h", "double_top")


@pytest.mark.parametrize(
    "model_cls",
    [LSTMModel, TransformerModel, CNNModel],
)
def test_sequence_models_train_predict_evaluate(model_cls):
    X, y = _make_sequence_data()
    model = model_cls(
        sequence_length=30,
        feature_dim=2,
        hidden_dim=8,
        epochs=3,
        batch_size=16,
    )
    result = model.train(X, y)

    assert model.is_trained
    assert "loss" in result

    preds = model.predict(X)
    assert preds.shape == y.shape

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)

    metrics = model.evaluate(X, y)
    assert 0 <= metrics["accuracy"] <= 1


@pytest.mark.parametrize(
    "model_cls",
    [LSTMModel, TransformerModel, CNNModel],
)
def test_sequence_models_get_prediction_and_roundtrip(model_cls, tmp_path):
    X, y = _make_sequence_data()
    model = model_cls(
        sequence_length=30,
        feature_dim=2,
        hidden_dim=8,
        epochs=3,
        batch_size=16,
    )
    model.train(X, y)

    prediction = model.get_prediction(X[0], "BTCUSDT", "1h", "double_top")
    assert prediction.model_name == model.name
    assert 0 <= prediction.probability <= 1
    assert 0 <= prediction.confidence <= 1

    prediction_flat = model.get_prediction(X[0].reshape(-1), "BTCUSDT", "1h", "double_top")
    assert prediction_flat.probability == pytest.approx(prediction.probability, abs=1e-6)

    path = tmp_path / f"{model.name}.pt"
    model.save(str(path))

    restored = model_cls()
    assert not restored.is_trained
    restored.load(str(path))
    assert restored.is_trained


def test_isolation_forest_train_predict():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (100, 6))
    model = IsolationForestModel(n_estimators=50, contamination=0.1)
    result = model.train(X, np.zeros(100))

    assert model.is_trained
    assert result["samples"] == 100

    preds = model.predict(X)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})

    proba = model.predict_proba(X)
    assert proba.shape == (100, 2)


def test_autoencoder_detects_anomalies(tmp_path):
    rng = np.random.default_rng(2)
    normal = rng.normal(0, 1, (120, 30))
    anomalies = rng.normal(8, 1, (10, 30))
    X = np.vstack([normal, anomalies])
    y = np.concatenate([np.zeros(120), np.ones(10)])

    model = AutoEncoderModel(input_dim=30, hidden_dim=16, latent_dim=4, epochs=5)
    model.train(normal, np.zeros(120))

    assert model.is_trained
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert proba[120:, 1].mean() > proba[:120, 1].mean()

    metrics = model.evaluate(X, y)
    assert metrics["accuracy"] > 0.9

    path = tmp_path / "autoencoder.pt"
    model.save(str(path))
    restored = AutoEncoderModel()
    restored.load(str(path))
    assert restored.is_trained
