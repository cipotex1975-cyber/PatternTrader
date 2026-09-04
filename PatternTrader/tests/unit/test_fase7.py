"""Pruebas de FASE 7 — evaluación y semántica de scores.

Verifica:
- AUC (ROC/PR) usa score continuo (predict_proba[:, 1]), nunca predict().
- F1/accuracy/precision/recall dependen del classification threshold.
- El threshold NO afecta a ROC/PR-AUC.
- La orientación del anomaly score es correcta (1 = positivo/anómalo).
- Los anomaly detectors se excluyen del ranking por defecto.
- Cada fila del summary lleva model_family.
- La optimización de threshold sobre VALIDATION encuentra el mejor y nunca usa TEST.
"""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from app.ml.factory import MLModelFactory
from app.ml.training.compare import (
    ANOMALY_MODELS,
    SUPERVISED_MODELS,
    classify_with_threshold,
    evaluate_model,
    metrics_at_threshold,
    model_family_label,
    optimize_classification_threshold,
    run_comparison,
)


class _StubModel:
    """Modelo stub: predict() usa 0.6 como umbral, predict_proba() devuelve scores."""

    name = "stub"
    model_type = "classification"

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.6).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([1.0 - self._score, self._score])

    def evaluate(self, X, y):  # pragma: no cover - no usado
        return {}


def _make_scores(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    probs = np.clip(rng.uniform(0, 1, n) + y * 0.25, 0.0, 1.0)
    return y, probs


def test_auc_uses_continuous_score_not_predict():
    y, probs = _make_scores(400, seed=3)
    stub = _StubModel()
    stub._score = probs  # type: ignore[attr-defined]

    metrics = evaluate_model(stub, np.zeros((len(y), 1)), y)
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y, probs))
    assert metrics["pr_auc"] == pytest.approx(average_precision_score(y, probs))


def test_f1_depends_on_classification_threshold():
    y, probs = _make_scores(500, seed=4)
    stub = _StubModel()
    stub._score = probs  # type: ignore[attr-defined]
    X = np.zeros((len(y), 1))

    m30 = evaluate_model(stub, X, y, classification_threshold=0.30)
    m70 = evaluate_model(stub, X, y, classification_threshold=0.70)
    assert m30["f1"] != m70["f1"] or m30["recall"] != m70["recall"]
    # AUC/PR-AUC no cambian con el threshold.
    assert m30["roc_auc"] == pytest.approx(m70["roc_auc"])
    assert m30["pr_auc"] == pytest.approx(m70["pr_auc"])


def test_threshold_does_not_affect_auc():
    y, probs = _make_scores(400, seed=5)
    m_low = metrics_at_threshold(y, probs, 0.25)
    m_high = metrics_at_threshold(y, probs, 0.75)
    assert m_low["roc_auc"] == pytest.approx(m_high["roc_auc"])
    assert m_low["pr_auc"] == pytest.approx(m_high["pr_auc"])


def test_classify_threshold_boundary():
    preds = classify_with_threshold([0.49, 0.50, 0.51], 0.50)
    np.testing.assert_array_equal(preds, [0, 1, 1])


def test_model_family_labels():
    assert model_family_label("classification") == "supervised classification"
    assert model_family_label("anomaly") == "anomaly detection"
    assert model_family_label("unknown_type") == "unknown_type"


def test_anomaly_models_excluded_by_default(small_dataset):
    summary, _ = run_comparison(
        small_dataset["X_train"],
        small_dataset["y_train"],
        small_dataset["X_val"],
        small_dataset["y_val"],
        model_names=["all"],
        exclude_anomaly=True,
    )
    assert "isolation_forest" not in set(summary["model"])
    assert "autoencoder" not in set(summary["model"])
    # Los supervisados mínimos sí están.
    assert "xgboost" in set(summary["model"])


def test_anomaly_models_included_when_explicit(small_dataset):
    summary, _ = run_comparison(
        small_dataset["X_train"],
        small_dataset["y_train"],
        small_dataset["X_val"],
        small_dataset["y_val"],
        model_names=["isolation_forest", "xgboost"],
        exclude_anomaly=True,  # petición explícita por nombre respeta la selección
    )
    models = set(summary["model"])
    assert "isolation_forest" in models
    assert "xgboost" in models


def test_model_family_in_summary(small_dataset):
    summary, _ = run_comparison(
        small_dataset["X_train"],
        small_dataset["y_train"],
        small_dataset["X_val"],
        small_dataset["y_val"],
        model_names=["xgboost", "isolation_forest"],
        exclude_anomaly=False,
    )
    row_sup = summary[summary["model"] == "xgboost"].iloc[0]
    row_anom = summary[summary["model"] == "isolation_forest"].iloc[0]
    assert row_sup["model_family"] == "supervised classification"
    assert row_anom["model_family"] == "anomaly detection"


def test_anomaly_family_sets_are_disjoint_and_cover_registered():
    registered = set(MLModelFactory.get_all())
    assert not (SUPERVISED_MODELS & ANOMALY_MODELS)
    assert SUPERVISED_MODELS | ANOMALY_MODELS == registered


def test_optimize_threshold_finds_best():
    y, probs = _make_scores(800, seed=6)
    best_threshold, table = optimize_classification_threshold(y, probs, selection_metric="f1")
    assert len(table) == len(
        [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    )
    assert 0.20 <= best_threshold <= 0.80
    # El threshold devuelto debe maximizar F1 sobre la tabla.
    best_f1 = max(r["f1"] for r in table)
    best_row = next(r for r in table if r["threshold"] == best_threshold)
    assert best_row["f1"] == pytest.approx(best_f1)


def test_optimize_threshold_rejects_non_threshold_metric():
    y, probs = _make_scores(100, seed=7)
    with pytest.raises(ValueError):
        optimize_classification_threshold(y, probs, selection_metric="roc_auc")


def test_anomaly_orientation_isolation_forest():
    from app.ml.models.isolation_forest import IsolationForestModel

    rng = np.random.default_rng(8)
    normal = rng.normal(0, 1, (150, 8))
    anomalies = rng.normal(10, 1, (15, 8))
    X = np.vstack([normal, anomalies])

    model = IsolationForestModel(n_estimators=100, contamination=0.1)
    model.train(X, np.zeros(len(X)))

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    # 1 = positivo = anómalo: las anomalías (cols 150+) deben tener mayor
    # probabilidad de clase 1 que las normales.
    assert proba[150:, 1].mean() > proba[:150, 1].mean()
    assert np.all((proba >= 0) & (proba <= 1))


def test_anomaly_orientation_autoencoder():
    from app.ml.models.autoencoder import AutoEncoderModel

    rng = np.random.default_rng(9)
    normal = rng.normal(0, 1, (120, 30))
    anomalies = rng.normal(9, 1, (12, 30))
    X = np.vstack([normal, anomalies])

    model = AutoEncoderModel(input_dim=30, hidden_dim=16, latent_dim=4, epochs=5)
    model.train(normal, np.zeros(len(normal)))

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert proba[120:, 1].mean() > proba[:120, 1].mean()
    assert np.all((proba >= 0) & (proba <= 1))


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(42)
    n_train, n_val = 120, 40
    X_train = rng.normal(0, 1, (n_train, 6))
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_val = rng.normal(0, 1, (n_val, 6))
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(int)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }
