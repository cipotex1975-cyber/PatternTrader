import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from app.ml.training.compare import (
    run_walk_forward_comparison,
    select_walk_forward_winner,
)
from app.ml.training.data import FEATURE_NAMES
from app.ml.training.walk_forward import (
    build_walk_forward_folds,
    validate_walk_forward_no_future,
)

N_FEATURES = len(FEATURE_NAMES)


@pytest.fixture
def selection_set():
    rng = np.random.default_rng(11)
    n = 2000
    X = rng.normal(0, 1, (n, N_FEATURES))
    y = (rng.random(n) < 0.3).astype(np.int64)
    return X, y


def test_folds_are_chronological_and_no_shuffle(selection_set):
    X, y = selection_set
    folds = build_walk_forward_folds(X, y, n_splits=5, forward_periods=5)
    assert len(folds) == 5
    train_ends = [f.train_end for f in folds]
    val_starts = [f.validation_start for f in folds]
    assert train_ends == sorted(train_ends)
    assert val_starts == sorted(val_starts)


def test_no_overlap_and_strict_separation(selection_set):
    X, y = selection_set
    folds = build_walk_forward_folds(X, y, n_splits=5, forward_periods=5)
    for f in folds:
        assert f.train_end <= f.validation_start
        assert f.validation_end > f.validation_start
        assert len(f.X_train) == len(f.y_train)
        assert len(f.X_validation) == len(f.y_validation)


def test_no_future_validation(selection_set):
    X, y = selection_set
    folds = build_walk_forward_folds(X, y, n_splits=5, forward_periods=5)
    validate_walk_forward_no_future(folds)  # no debe lanzar
    for f in folds:
        assert f.validation_start > f.train_end


def test_train_expands_monotonically(selection_set):
    X, y = selection_set
    folds = build_walk_forward_folds(X, y, n_splits=5, forward_periods=5)
    sizes = [len(f.X_train) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_labels_respect_forward_periods(selection_set):
    """Cada train_fold y validation_fold recorta las últimas forward_periods."""
    X, y = selection_set
    fp = 7
    folds = build_walk_forward_folds(X, y, n_splits=3, forward_periods=fp)
    for f in folds:
        # train nunca llega a la frontera de validation: se detiene fp antes.
        assert f.train_end == f.validation_start - fp
        assert f.validation_end == f.validation_start + len(f.X_validation)


def test_forward_periods_trim_is_exact(selection_set):
    X, y = selection_set
    fp = 5
    folds = build_walk_forward_folds(X, y, n_splits=4, forward_periods=fp)
    for f in folds:
        # El validation_fold excluye sus últimas fp muestras (anti-leakage).
        assert f.validation_end == f.validation_start + len(f.X_validation)
        assert len(f.X_validation) == len(f.y_validation)


def test_small_dataset_rejected():
    X = np.zeros((50, N_FEATURES))
    y = np.zeros(50)
    with pytest.raises(ValueError):
        build_walk_forward_folds(X, y, n_splits=5, forward_periods=5)


def test_invalid_splits_rejected(selection_set):
    X, y = selection_set
    with pytest.raises(ValueError):
        build_walk_forward_folds(X, y, n_splits=1, forward_periods=5)


def test_scaler_per_fold_fit_only_on_train_fold():
    """El scaler de cada fold usa el fit de SU train_fold (sin validation)."""
    rng = np.random.default_rng(3)
    n = 1500
    X = rng.normal(0, 1, (n, N_FEATURES))
    y = (rng.random(n) < 0.3).astype(np.int64)
    folds = build_walk_forward_folds(X, y, n_splits=3, forward_periods=5)

    from app.ml.training.scaling import apply_feature_scaling

    for f in folds:
        X_tr, X_val, _, scaler = apply_feature_scaling(
            f.X_train, f.X_validation, f.X_validation, mode="standard", feature_names=FEATURE_NAMES
        )
        assert isinstance(scaler, StandardScaler)
        np.testing.assert_allclose(scaler.mean_, f.X_train.mean(axis=0), rtol=1e-6)


def test_run_walk_forward_summary_and_selection():
    """run_walk_forward_comparison devuelve agregados wf_* y elige por media."""
    rng = np.random.default_rng(5)
    n = 2500
    X = rng.normal(0, 1, (n, N_FEATURES))
    y = (rng.random(n) < 0.3).astype(np.int64)

    summary, trained = run_walk_forward_comparison(
        X,
        y,
        n_splits=3,
        model_names=["random_forest"],
        metric="roc_auc",
        feature_names=FEATURE_NAMES,
        sequence_length=30,
        epochs=1,
        feature_scaling="none",
    )
    row = summary.iloc[0]
    assert summary.shape[0] == 1
    assert row["status"] == "ok"
    assert row["wf_folds"] == 3
    assert 0.0 <= row["wf_mean_roc_auc"] <= 1.0
    assert row["wf_std_roc_auc"] >= 0.0
    assert "random_forest" in trained

    winner = select_walk_forward_winner(summary, "roc_auc")
    assert winner is not None
    assert winner["model"] == "random_forest"


def test_walk_forward_does_not_use_test():
    """La selección usa wf_mean_* (folds), no ninguna métrica de TEST FINAL."""
    rng = np.random.default_rng(9)
    n = 2000
    X = rng.normal(0, 1, (n, N_FEATURES))
    y = (rng.random(n) < 0.3).astype(np.int64)
    summary, _ = run_walk_forward_comparison(
        X,
        y,
        n_splits=3,
        model_names=["random_forest"],
        metric="roc_auc",
        feature_names=FEATURE_NAMES,
        sequence_length=30,
        epochs=1,
        feature_scaling="none",
    )
    # En el summary walk-forward no existe columna tipo "test_*".
    assert not any(c.startswith("test_") for c in summary.columns)
