import numpy as np
import pytest

from app.ml.factory import MLModelFactory
from app.ml.models.sequence_base import _early_stopping_decision
from app.ml.training.data import build_sequences


def _make_matrix(n: int = 120, features: int = 12, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _seq_split(
    n_train: int = 50, n_val: int = 16, features: int = 6, sequence_length: int = 8, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = _make_matrix(n=n_train + n_val, features=features, seed=seed)
    seq_tr, lab_tr = build_sequences(X[:n_train], y[:n_train], sequence_length=sequence_length)
    seq_val, lab_val = build_sequences(
        X[n_train : n_train + n_val], y[n_train : n_train + n_val], sequence_length=sequence_length
    )
    return seq_tr, lab_tr, seq_val, lab_val


class TestEarlyStoppingDecision:
    """FASE 6, sección 2: la patience detiene el entrenamiento (regla pura)."""

    def test_best_epoch_is_index_of_min(self):
        history = [1.0, 0.9, 0.8, 0.81, 0.82, 0.83, 0.84]
        best_epoch, trained = _early_stopping_decision(history, patience=3)
        assert best_epoch == 3
        assert trained == 6

    def test_patience_stops_after_n_failures(self):
        # Sin mejora nunca: best=epoch 1, para tras 1+patience epochs.
        history = [0.4, 0.41, 0.42, 0.43]
        best_epoch, trained = _early_stopping_decision(history, patience=2)
        assert best_epoch == 1
        assert trained == 3

    def test_full_run_when_improving_until_end(self):
        history = [0.5, 0.4, 0.3, 0.2, 0.1]
        best_epoch, trained = _early_stopping_decision(history, patience=5)
        assert best_epoch == 5
        assert trained == 5

    def test_empty_history(self):
        assert _early_stopping_decision([], patience=5) == (0, 0)


class TestSequenceEarlyStopping:
    """FASE 6, secciones 1-3: validation en entrenamiento, checkpoint, sin TEST."""

    def _model(self, epochs: int = 3, patience: int = 2, seed: int = 0):
        return MLModelFactory.create_new(
            "cnn",
            sequence_length=8,
            feature_dim=6,
            hidden_dim=8,
            kernel_size=3,
            epochs=epochs,
            patience=patience,
            batch_size=8,
            random_state=seed,
        )

    def test_train_uses_validation_not_test(self):
        seq_tr, lab_tr, seq_val, lab_val = _seq_split(seed=11)
        metrics = self._model().train(seq_tr, lab_tr, X_val=seq_val, y_val=lab_val)

        assert "validation_loss" in metrics and np.isfinite(metrics["validation_loss"])
        assert "validation_accuracy" in metrics and 0 <= metrics["validation_accuracy"] <= 1
        assert "train_accuracy" in metrics and np.isfinite(metrics["train_accuracy"])
        assert "best_epoch" in metrics and 1 <= metrics["best_epoch"] <= 3
        assert "best_validation_loss" in metrics and np.isfinite(metrics["best_validation_loss"])
        # El TEST FINAL jamás entra al train(): no hay parámetro de test.
        assert "test" not in metrics

    def test_early_stopping_respects_patience_rule(self):
        seq_tr, lab_tr, seq_val, lab_val = _seq_split(seed=3)
        epochs, patience = 10, 2
        metrics = self._model(epochs=epochs, patience=patience).train(
            seq_tr, lab_tr, X_val=seq_val, y_val=lab_val
        )
        assert metrics["epochs"] <= epochs
        if metrics["early_stopping"]:
            # Detenido en best_epoch + patience (regla de _early_stopping_decision).
            assert metrics["epochs"] == metrics["best_epoch"] + patience
        else:
            assert metrics["epochs"] == epochs

    def test_sequences_still_work_without_validation(self):
        seq_tr, lab_tr, _, _ = _seq_split(seed=5)
        metrics = self._model(epochs=2).train(seq_tr, lab_tr)
        assert metrics["best_epoch"] == 0
        assert np.isfinite(metrics["train_loss"])
        assert not metrics["early_stopping"]

    def test_checkpoint_persists_best_validation_state(self, tmp_path):
        seq_tr, lab_tr, seq_val, lab_val = _seq_split(seed=9)
        model = self._model(epochs=5, patience=1)
        metrics = model.train(seq_tr, lab_tr, X_val=seq_val, y_val=lab_val)

        artifact = str(tmp_path / "cnn_checkpoint.pt")
        model.save(artifact)

        loaded = self._model()
        loaded.load(artifact)
        # Checkpoint = mejor validation (no el último epoch).
        assert loaded._best_epoch == metrics["best_epoch"]
        assert loaded._best_validation_loss == pytest.approx(metrics["best_validation_loss"])


class TestTreeEarlyStopping:
    """FASE 6, sección 5: eval_set + early stopping nativo en los árboles."""

    @pytest.mark.parametrize(
        ("name", "params"),
        [
            ("xgboost", {"n_estimators": 30, "max_depth": 3, "early_stopping_rounds": 3}),
            ("lightgbm", {"n_estimators": 30, "max_depth": 3, "early_stopping_rounds": 3}),
            ("catboost", {"iterations": 30, "depth": 3, "early_stopping_rounds": 3}),
        ],
    )
    def test_eval_set_and_early_stopping_active(self, name, params):
        X, y = _make_matrix(n=120, features=8, seed=2)
        X_tr, y_tr = X[:90], y[:90]
        X_val, y_val = X[90:], y[90:]

        model = MLModelFactory.create_new(name, **params)
        metrics = model.train(X_tr, y_tr, X_val=X_val, y_val=y_val)

        assert "validation_accuracy" in metrics
        assert 0 <= metrics["validation_accuracy"] <= 1
        assert "best_iteration" in metrics and metrics["best_iteration"] >= 0
        assert metrics["early_stopping"] is True

    @pytest.mark.parametrize("name", ["xgboost", "lightgbm", "catboost"])
    def test_tree_without_validation_still_trains(self, name):
        X, y = _make_matrix(n=80, features=8, seed=4)
        params = {"n_estimators": 10} if name != "catboost" else {"iterations": 10}
        params["early_stopping_rounds"] = 0
        model = MLModelFactory.create_new(name, **params)
        metrics = model.train(X, y)
        assert "train_accuracy" in metrics
        assert metrics["early_stopping"] is False
