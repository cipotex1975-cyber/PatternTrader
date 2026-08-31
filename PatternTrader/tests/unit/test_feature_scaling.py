import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from app.ml.factory import MLModelFactory
from app.ml.training.data import FEATURE_NAMES
from app.ml.training.scaling import (
    apply_feature_scaling,
    load_scaler_sidecar,
    save_scaler_sidecar,
    scaler_from_artifact,
    scaler_to_artifact,
)


@pytest.fixture
def matrices():
    rng = np.random.default_rng(7)
    X_train = rng.normal(10, 2, (200, len(FEATURE_NAMES)))
    X_val = rng.normal(12, 2, (60, len(FEATURE_NAMES)))
    X_test = rng.normal(8, 3, (60, len(FEATURE_NAMES)))
    return X_train, X_val, X_test


def test_mode_none_returns_untouched(matrices):
    X_train, X_val, X_test = matrices
    t, v, te, scaler = apply_feature_scaling(X_train, X_val, X_test, mode="none")
    assert scaler is None
    np.testing.assert_array_equal(t, X_train)
    np.testing.assert_array_equal(v, X_val)
    np.testing.assert_array_equal(te, X_test)


def test_invalid_mode_rejected(matrices):
    X_train, X_val, X_test = matrices
    with pytest.raises(ValueError):
        apply_feature_scaling(X_train, X_val, X_test, mode="bogus")


def test_scaler_fit_only_on_train(matrices):
    X_train, X_val, X_test = matrices
    X_tr, X_val_t, X_test_t, scaler = apply_feature_scaling(
        X_train, X_val, X_test, mode="standard", feature_names=FEATURE_NAMES
    )
    assert isinstance(scaler, StandardScaler)
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(scaler.scale_, X_train.std(axis=0), rtol=1e-6)


def test_validation_and_test_are_only_transformed(matrices):
    """validation y test NO pueden refitear el scaler: deben coincidir con
    scaler.transform() usando stats de TRAIN."""
    X_train, X_val, X_test = matrices
    _, X_val_t, X_test_t, scaler = apply_feature_scaling(
        X_train, X_val, X_test, mode="standard", feature_names=FEATURE_NAMES
    )
    np.testing.assert_allclose(X_val_t, scaler.transform(X_val), rtol=1e-9)
    np.testing.assert_allclose(X_test_t, scaler.transform(X_test), rtol=1e-9)


def test_test_never_used_for_fitting(matrices):
    """Si TEST entrara en el fit, mean_ y scale_ cambiarían. Se verifica que el
    scaler devuelto coincide con un fit EXCLUSIVO de TRAIN."""
    X_train, X_val, X_test = matrices
    _, _, _, scaler = apply_feature_scaling(
        X_train, X_val, X_test, mode="standard", feature_names=FEATURE_NAMES
    )
    ref = StandardScaler().fit(X_train)
    np.testing.assert_allclose(scaler.mean_, ref.mean_, rtol=1e-9)
    np.testing.assert_allclose(scaler.scale_, ref.scale_, rtol=1e-9)


def test_feature_order_mismatch_rejected():
    X_train = np.zeros((10, 3))
    with pytest.raises(ValueError):
        apply_feature_scaling(
            X_train, X_train, X_train, mode="standard", feature_names=FEATURE_NAMES
        )


def test_scaler_roundtrip_artifact(matrices):
    X_train, _, _, scaler = apply_feature_scaling(
        *matrices, mode="standard", feature_names=FEATURE_NAMES
    )
    artifact = scaler_to_artifact(scaler, FEATURE_NAMES)
    assert artifact["type"] == "StandardScaler"
    assert artifact["fitted_on"] == "TRAIN_ONLY"
    assert artifact["features"] == FEATURE_NAMES
    assert len(artifact["mean_"]) == len(FEATURE_NAMES)

    restored = scaler_from_artifact(artifact)
    np.testing.assert_allclose(restored.transform(X_train), scaler.transform(X_train), rtol=1e-9)


def test_scaler_sidecar_file_roundtrip(matrices, tmp_path):
    _, _, _, scaler = apply_feature_scaling(*matrices, mode="standard", feature_names=FEATURE_NAMES)
    path = tmp_path / "scaler.json"
    save_scaler_sidecar(scaler, str(path), FEATURE_NAMES)
    loaded = load_scaler_sidecar(str(path))
    assert isinstance(loaded, StandardScaler)
    np.testing.assert_allclose(loaded.mean_, scaler.mean_, rtol=1e-12)
    np.testing.assert_allclose(loaded.scale_, scaler.scale_, rtol=1e-12)


def test_scaler_artifact_matches_sidecar_json(matrices, tmp_path):
    """El bloque preprocessing del sidecar (meta.json) y el artefacto .scaler.json
    deben serializar el mismo scaler (feature order coincide)."""
    _, _, _, scaler = apply_feature_scaling(*matrices, mode="standard", feature_names=FEATURE_NAMES)
    scaler_path = tmp_path / "scaler.json"
    save_scaler_sidecar(scaler, str(scaler_path), FEATURE_NAMES)
    on_disk = load_scaler_sidecar(str(scaler_path))
    meta_block = scaler_to_artifact(scaler, FEATURE_NAMES)
    json.loads(json.dumps(meta_block))  # JSON-serializable
    np.testing.assert_allclose(on_disk.mean_, meta_block["mean_"])


def _build_raw_sequence(seq_len=30, features=12, seed=3):
    """Features crudas por vela: shape (seq_len, features)."""
    rng = np.random.default_rng(seed)
    return rng.normal(5, 1, (seq_len, features)).astype(np.float32)


def test_sequence_model_serving_matches_training_scaling():
    """El serving (get_prediction con features crudas 2D) debe escalar con el scaler
    fit en TRAIN: raw → scaler → sequence → model. Predicción sobre crudo con
    scaler adjunto == predicción sobre datos ya escalados sin volver a escalar."""
    seq_len, features = 8, 4
    rng = np.random.default_rng(11)
    X_tr = rng.normal(10, 2, (120, features))
    y_tr = (X_tr[:, 0] > 0).astype(int)
    from app.ml.training.data import build_sequences

    _, _, _, scaler = apply_feature_scaling(
        X_tr, X_tr, X_tr, mode="standard", feature_names=[f"f{i}" for i in range(features)]
    )

    model = MLModelFactory.create_new(
        "lstm",
        sequence_length=seq_len,
        feature_dim=features,
        hidden_dim=16,
        epochs=2,
    )
    X_seq, y_seq = build_sequences(scaler.transform(X_tr).astype(np.float32), y_tr, seq_len)
    model.train(X_seq, y_seq, feature_names=[f"f{i}" for i in range(features)])

    raw = _build_raw_sequence(seq_len, features)

    model._scaler = scaler
    pred_raw = model.get_prediction(raw.copy(), symbol="X", timeframe="H1", pattern_name="p")

    model._scaler = None
    pred_pre = model.get_prediction(
        scaler.transform(raw).astype(np.float32).copy(),
        symbol="X",
        timeframe="H1",
        pattern_name="p",
    )

    np.testing.assert_allclose(pred_raw.probability, pred_pre.probability, atol=1e-6)


def test_sequence_model_prepare_scales_only_when_scaler_present():
    """_prepare sobre input 2D (seq_len×features) aplica el scaler sólo si está
    presente; sin scaler el input pasa sin modificar."""
    seq_len, features = 8, 4
    X_tr = _build_raw_sequence(10, features, seed=4)
    _, _, _, scaler = apply_feature_scaling(
        X_tr,
        X_tr,
        X_tr,
        mode="standard",
        feature_names=[f"f{i}" for i in range(features)],
    )

    model = MLModelFactory.create_new(
        "cnn", sequence_length=seq_len, feature_dim=features, hidden_dim=16, epochs=1
    )
    model._scaler = scaler
    raw = _build_raw_sequence(seq_len, features)

    scaled = model._prepare(raw.copy())
    assert scaled.shape == (seq_len, features, 1)
    np.testing.assert_allclose(scaled[:, :, 0], scaler.transform(raw).astype(np.float32), rtol=1e-5)

    model._scaler = None
    unscaled = model._prepare(raw.copy())
    assert unscaled.shape == (seq_len, features, 1)
    np.testing.assert_allclose(unscaled[:, :, 0], raw.astype(np.float32), rtol=1e-6)


def test_run_comparison_attaches_scaler_and_sidecar_writes_preprocessing(tmp_path):
    """FASE 4 end-to-end: con mode=standard, el ganador lleva el scaler y su
    sidecar (meta.json) registra el bloque preprocessing + artefacto .scaler.json."""
    from app.ml.training.compare import run_comparison, save_winner, select_winner

    rng = np.random.default_rng(21)
    n, features = 260, len(FEATURE_NAMES)
    X = rng.normal(0, 1, (n, features))
    y = ((X[:, 0] + X[:, 1] + X[:, 2]) > 0).astype(int)

    n_tr, n_val = 180, 40
    X_tr, X_val = X[:n_tr], X[n_tr : n_tr + n_val]
    y_tr, y_val = y[:n_tr], y[n_tr : n_tr + n_val]

    summary, trained = run_comparison(
        X_tr,
        y_tr,
        X_val,
        y_val,
        model_names=["cnn"],
        metric="roc_auc",
        feature_names=FEATURE_NAMES,
        sequence_length=10,
        epochs=1,
        feature_scaling="standard",
    )
    winner = select_winner(summary, "roc_auc")
    assert winner is not None
    model = trained[winner["model"]]
    scaler = getattr(model, "_scaler", None)
    assert scaler is not None

    artifact, sidecar = save_winner(trained, winner, str(tmp_path), "TESTX", metric="roc_auc")
    meta = json.loads(Path(sidecar).read_text())
    recorded = meta.get("preprocessing")
    assert recorded is not None
    assert recorded["type"] == "StandardScaler"
    assert recorded["fitted_on"] == "TRAIN_ONLY"
    assert recorded["features"] == FEATURE_NAMES
    scaler_path = Path(tmp_path) / f"{winner['model']}_TESTX.scaler.json"
    assert scaler_path.exists()
    loaded = load_scaler_sidecar(str(scaler_path))
    assert isinstance(loaded, StandardScaler)
