from __future__ import annotations

import hashlib
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logger import get_logger
from app.ml.training.compare import _model_kwargs
from app.ml.training.data import FEATURE_NAMES, FEATURE_VERSION

logger = get_logger("Reproducibility")


def seed_all(seed: int | None = None) -> None:
    """Fija las semillas de Python/NumPy/PyTorch para reproducibilidad (FASE 9).

    Si ``seed`` es ``None`` no fija nada (permite aleatoriedad cuando se desea).
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch siempre presente en el repo
        pass


def hash_file_sha256(path: str | Path) -> str:
    """SHA-256 del contenido crudo del archivo (FASE 9, sección 3).

    Detecta "mismo nombre de archivo pero contenido diferente". Si el archivo no
    existe devuelve ``"unknown"``.
    """
    p = Path(path)
    if not p.is_file():
        return "unknown"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_sha() -> str | None:
    """Commit SHA de la cabecera actual del repo (FASE 9, ``git.commit_sha``)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        sha = out.stdout.strip()
        return sha or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def get_software_versions() -> dict[str, str]:
    """Versiones de software clave para reproducibilidad (FASE 9, ``software``)."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
    }
    try:
        from sklearn import __version__ as v  # noqa: PLC0415

        versions["sklearn"] = v
    except ImportError:  # pragma: no cover
        pass
    try:
        import numpy as _np  # noqa: PLC0415

        versions["numpy"] = _np.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import pandas as _pd  # noqa: PLC0415

        versions["pandas"] = _pd.__version__
    except ImportError:  # pragma: no cover
        pass
    for lib in ("torch", "xgboost", "lightgbm", "catboost"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:  # pragma: no cover
            pass
    return versions


def _dataset_block(
    data_path: str,
    ranges: dict[str, dict[str, Any]],
    samples_total: int,
) -> dict[str, Any]:
    """Bloque ``dataset`` del sidecar: path, hash, inicio/fin y muestras."""
    start = end = None
    for key in ("train", "validation", "test"):
        r = ranges.get(key) or {}
        if r.get("start"):
            start = start or r["start"]
        if r.get("end"):
            end = r["end"]
    return {
        "path": Path(data_path).name,
        "hash": hash_file_sha256(data_path),
        "start_datetime": start,
        "end_datetime": end,
        "samples": int(samples_total),
    }


def _training_block(
    model_name: str,
    feature_names: list[str] | None,
    sequence_length: int,
    epochs: int,
    settings: Any,
    hyperparams: dict[str, dict[str, Any]] | None,
    patience: int,
    early_stop_rounds: int,
) -> dict[str, Any]:
    """Bloque ``training`` con hiperparámetros efectivos del modelo ganador."""
    kwargs = _model_kwargs(
        model_name,
        feature_names=feature_names,
        sequence_length=sequence_length,
        epochs=epochs,
        settings=settings,
        hyperparams=hyperparams,
        patience=patience,
        early_stop_rounds=early_stop_rounds,
    )
    block: dict[str, Any] = {
        "epochs": epochs,
        "hyperparameters": kwargs,
    }
    # Campos de entrenamiento explícitos solo cuando son relevantes.
    if "learning_rate" in kwargs:
        block["learning_rate"] = kwargs["learning_rate"]
    if "batch_size" in kwargs:
        block["batch_size"] = kwargs["batch_size"]
    return block


def build_model_sidecar_context(
    *,
    model_name: str,
    data_path: str,
    ranges: dict[str, dict[str, Any]],
    samples_total: int,
    feature_names: list[str] | None = None,
    forward_periods: int,
    threshold: float,
    min_up_moves: int,
    preprocessing_type: str,
    sequence_length: int,
    epochs: int,
    settings: Any,
    hyperparams: dict[str, dict[str, Any]] | None,
    patience: int,
    early_stop_rounds: int,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
    selection_metric: str,
    selection_dataset: str,
    random_seed: int | None,
) -> dict[str, Any]:
    """Construye la metadata completa del sidecar según la FASE 9.

    Devuelve un dict de bloques que ``save_winner`` fusiona en el sidecar
    ``.meta.json``. No requiere cambios en los modelos.
    """
    features = feature_names or list(FEATURE_NAMES)
    meta: dict[str, Any] = {
        "timeframe": None,
        "dataset": _dataset_block(data_path, ranges, samples_total),
        "features": {
            "names": features,
            "count": len(features),
            "version": FEATURE_VERSION,
        },
        "label": {
            "forward_periods": int(forward_periods),
            "threshold": float(threshold),
            "min_up_moves": int(min_up_moves),
        },
        "preprocessing": {"type": preprocessing_type},
        "sequence": {"length": int(sequence_length)},
        "training": _training_block(
            model_name,
            feature_names=features,
            sequence_length=sequence_length,
            epochs=epochs,
            settings=settings,
            hyperparams=hyperparams,
            patience=patience,
            early_stop_rounds=early_stop_rounds,
        ),
        "validation": {"metrics": val_metrics_block(validation_metrics)},
        "selection": {
            "metric": selection_metric,
            "dataset": selection_dataset,
        },
        "software": get_software_versions(),
        "git": {"commit_sha": get_git_commit_sha()},
        "random_seed": random_seed,
    }
    if test_metrics:
        meta["test"] = {"metrics": val_metrics_block(test_metrics)}
    return meta


def val_metrics_block(metrics: dict[str, Any]) -> dict[str, float]:
    """Filtra métricas a valores float/JSON-serializables."""
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[str(k)] = float(v)
    return out
