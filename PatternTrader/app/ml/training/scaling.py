from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

SUPPORTED_SCALING_MODES = ("none", "standard")
SCALER_SIDECAR_STEM = "scaler"


class ScalerNotFittedError(RuntimeError):
    """El scaler solicitado no está ajustado (mode=standard sin fit previo)."""


def apply_feature_scaling(
    X_train: np.ndarray,
    X_validation: np.ndarray,
    X_test: np.ndarray,
    mode: str = "none",
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler | None]:
    """Aplica scaling reproducible sobre los tres segmentos cronológicos.

    La única fuente de ``scaler.fit`` es ``X_train`` (TRAIN_ONLY, sin leakage).
    ``X_validation`` y ``X_test`` usan exclusivamente ``transform``. Con
    ``mode="none"`` se devuelven las matrices sin modificar y ``scaler=None``.

    Returns:
        (X_train_t, X_validation_t, X_test_t, scaler)
    """
    if mode not in SUPPORTED_SCALING_MODES:
        raise ValueError(f"mode inválido: {mode}. Válidos: {', '.join(SUPPORTED_SCALING_MODES)}")
    if mode == "none":
        return X_train, X_validation, X_test, None

    X_tr = np.asarray(X_train, dtype=np.float64)
    X_val = np.asarray(X_validation, dtype=np.float64)
    X_te = np.asarray(X_test, dtype=np.float64)

    if X_tr.ndim != 2 or X_tr.shape[1] == 0:
        raise ValueError(f"X_train debe ser 2D con features; shape {X_tr.shape}")

    if feature_names is not None:
        expected = len(feature_names)
        if X_tr.shape[1] != expected:
            raise ValueError(
                f"Feature order mismatch: X_train tiene {X_tr.shape[1]} columnas "
                f"pero se declaran {expected} features"
            )

    scaler = StandardScaler()
    scaler.fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_val), scaler.transform(X_te), scaler


def scaler_to_artifact(scaler: StandardScaler, feature_names: list[str]) -> dict[str, Any]:
    """Serializa los parámetros del scaler en un dict JSON-serializable.

    Rehidratable con ``scaler_from_artifact`` para servir el MISMO scaler
    que se usó en training (FASE 4, sección 5: serving replica el scaler).
    """
    means: list[float] = [float(x) for x in scaler.mean_] if scaler.mean_ is not None else []
    scales: list[float] = [float(x) for x in scaler.scale_] if scaler.scale_ is not None else []
    var: list[float] = [float(x) for x in scaler.var_] if scaler.var_ is not None else []
    return {
        "type": "StandardScaler",
        "features": list(feature_names),
        "fitted_on": "TRAIN_ONLY",
        "mean_": means,
        "scale_": scales,
        "var_": var,
    }


def scaler_from_artifact(artifact: dict[str, Any]) -> StandardScaler:
    """Rehidrata un ``StandardScaler`` desde el dict generado por ``scaler_to_artifact``."""
    if artifact.get("type") != "StandardScaler":
        raise ValueError(f"Tipo de scaler no soportado: {artifact.get('type')}")
    scaler = StandardScaler()
    scaler.mean_ = np.array(artifact["mean_"], dtype=np.float64)
    scaler.scale_ = np.array(artifact["scale_"], dtype=np.float64)
    scaler.var_ = np.array(artifact["var_"], dtype=np.float64)
    scaler.n_features_in_ = len(artifact["mean_"])
    scaler.n_samples_seen_ = 1
    return scaler


def save_scaler_sidecar(scaler: StandardScaler, path: str, feature_names: list[str]) -> None:
    """Escribe el artefacto JSON del scaler y su metadata ``preprocessing``."""
    artifact = scaler_to_artifact(scaler, feature_names)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)


def load_scaler_sidecar(path: str) -> StandardScaler:
    """Carga un ``StandardScaler`` desde el artefacto JSON del scaler."""
    with open(path) as f:
        artifact = json.load(f)
    return scaler_from_artifact(artifact)
