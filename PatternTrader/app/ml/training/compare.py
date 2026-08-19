from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.core.config.settings import Settings
from app.core.logger import get_logger
from app.ml.base import BaseMLModel
from app.ml.factory import MLModelFactory
from app.ml.training.data import format_for_model

logger = get_logger("TrainAndCompare")

# Métricas comparables entre todos los modelos (0-1).
AVAILABLE_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")

# Extensión del artefacto por familia de modelo (formato nativo de guardado).
MODEL_EXTENSIONS: dict[str, str] = {
    "random_forest": ".pkl",
    "lightgbm": ".pkl",
    "isolation_forest": ".pkl",
    "xgboost": ".json",
    "catboost": ".cbm",
    "lstm": ".pt",
    "transformer": ".pt",
    "cnn": ".pt",
    "autoencoder": ".pt",
}


def _model_kwargs(
    name: str,
    feature_names: list[str] | None = None,
    sequence_length: int = 30,
    epochs: int = 10,
    settings: Settings | None = None,
    hyperparams: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Hiperparámetros por modelo: defaults → config YAML → override explícito."""
    n_features = len(feature_names) if feature_names else 1

    defaults: dict[str, Any] = {
        "random_forest": {"n_estimators": 100, "max_depth": 10},
        "xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
        "lightgbm": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
        "catboost": {"iterations": 100, "depth": 6, "learning_rate": 0.1},
        "lstm": {"num_layers": 2, "hidden_dim": 64},
        "transformer": {"nhead": 4, "num_layers": 2, "hidden_dim": 64},
        "cnn": {"kernel_size": 3, "hidden_dim": 64},
        "isolation_forest": {"n_estimators": 200, "contamination": 0.05},
        "autoencoder": {"hidden_dim": 32, "latent_dim": 8},
    }

    kwargs = dict(defaults.get(name, {}))

    if settings is not None:
        cfg = settings.ml.models
        if name == "random_forest":
            kwargs.update(
                {
                    "n_estimators": cfg.random_forest.n_estimators,
                    "max_depth": cfg.random_forest.max_depth,
                }
            )
        elif name == "xgboost":
            kwargs.update(
                {
                    "n_estimators": cfg.xgboost.n_estimators,
                    "max_depth": cfg.xgboost.max_depth,
                    "learning_rate": cfg.xgboost.learning_rate,
                }
            )
        elif name == "lightgbm":
            kwargs.update(
                {
                    "n_estimators": cfg.lightgbm.n_estimators,
                    "max_depth": cfg.lightgbm.max_depth,
                    "learning_rate": cfg.lightgbm.learning_rate,
                }
            )
        elif name == "lstm":
            kwargs.update(
                {
                    "sequence_length": cfg.lstm.sequence_length,
                    "hidden_dim": cfg.lstm.hidden_size,
                    "num_layers": cfg.lstm.num_layers,
                }
            )

    if name in ("lstm", "transformer", "cnn"):
        kwargs["sequence_length"] = sequence_length
        kwargs["feature_dim"] = n_features
        kwargs["epochs"] = epochs
        kwargs.setdefault("batch_size", 16)
        kwargs.setdefault("hidden_dim", 64)

    if name == "autoencoder":
        kwargs["input_dim"] = n_features
        kwargs["epochs"] = epochs

    if hyperparams and name in hyperparams:
        kwargs.update(hyperparams[name])

    return kwargs


def evaluate_model(model: BaseMLModel, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Métricas unificadas (0-1) para cualquier modelo de la plataforma."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    labels = np.asarray(y)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predictions)),
    }
    if len(np.unique(labels)) > 1:
        metrics["precision"] = float(precision_score(labels, predictions, zero_division=0))
        metrics["recall"] = float(recall_score(labels, predictions, zero_division=0))
        metrics["f1"] = float(f1_score(labels, predictions, zero_division=0))
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        metrics["pr_auc"] = float(average_precision_score(labels, probabilities))
    return metrics


def run_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_names: list[str] | None = None,
    metric: str = "roc_auc",
    feature_names: list[str] | None = None,
    sequence_length: int = 30,
    epochs: int = 10,
    settings: Settings | None = None,
    hyperparams: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, BaseMLModel]]:
    """Entrena todos los modelos solicitados sobre el mismo split y compara métricas.

    Retorna (summary, trained) donde ``summary`` es un DataFrame con una fila por
    modelo y ``trained`` mapea nombre → instancia entrenada.
    """
    if metric not in AVAILABLE_METRICS:
        raise ValueError(f"Metric desconocida: {metric}. Válidas: {', '.join(AVAILABLE_METRICS)}")

    registered = set(MLModelFactory.get_all())
    if not model_names or "all" in model_names:
        selected = sorted(registered)
    else:
        selected = [name for name in model_names if name in registered]

    if not selected:
        raise ValueError(
            f"No hay modelos válidos entre {model_names}. "
            f"Registrados: {', '.join(sorted(registered))}"
        )

    rows: list[dict[str, Any]] = []
    trained: dict[str, BaseMLModel] = {}

    for name in selected:
        try:
            X_tr, y_tr = format_for_model(name, X_train, y_train, sequence_length)
            X_te, y_te = format_for_model(name, X_test, y_test, sequence_length)
            if y_tr is None or y_te is None:
                raise ValueError(f"{name}: no se pudieron formatear las etiquetas")

            kwargs = _model_kwargs(
                name,
                feature_names=feature_names,
                sequence_length=sequence_length,
                epochs=epochs,
                settings=settings,
                hyperparams=hyperparams,
            )
            model = MLModelFactory.create_new(name, **kwargs)
            train_metrics = model.train(X_tr, y_tr, feature_names=feature_names)
            eval_metrics = evaluate_model(model, X_te, y_te)

            train_acc = train_metrics.get("train_accuracy")
            train_loss = train_metrics.get("loss")

            row: dict[str, Any] = {
                "model": name,
                "status": "ok",
                "train_accuracy": (
                    float(train_acc)
                    if train_acc is not None
                    else float(train_loss) if train_loss is not None else float("nan")
                ),
                "samples_train": int(X_tr.shape[0]),
                "samples_test": int(X_te.shape[0]),
            }
            row.update(eval_metrics)
            rows.append(row)
            trained[name] = model
            logger.info(
                f"{name}: train={row['train_accuracy']:.4f} {metric}={row.get(metric, 'n/a')}"
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"{name}: falló el entrenamiento: {e}")
            rows.append(
                {
                    "model": name,
                    "status": f"error: {e}",
                    "train_accuracy": float("nan"),
                    "samples_train": 0,
                    "samples_test": 0,
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("Ningún modelo se entrenó correctamente")
    return summary, trained


def select_winner(summary: pd.DataFrame, metric: str = "roc_auc") -> dict[str, Any] | None:
    """Elige la fila con mejor métrica objetivo entre los modelos exitosos."""
    valid = summary[summary["status"] == "ok"].copy()
    if metric not in valid.columns:
        return None
    valid = valid.dropna(subset=[metric])
    if valid.empty:
        return None
    best = valid.loc[valid[metric].idxmax()]
    return {"model": best["model"], "metrics": best.to_dict()}


def _version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_winner(
    trained: dict[str, BaseMLModel],
    winner: dict[str, Any],
    save_dir: str,
    symbol: str,
    metric: str = "roc_auc",
) -> tuple[str, str]:
    """Persiste el artefacto ganador con nomenclatura por par y su sidecar.

    Nombres: ``{modelo}_{symbol}{ext}`` + ``{modelo}_{symbol}.meta.json``
    (el sidecar permite al ScoringEngine rehidratar el modelo del par sin DB).
    """
    model_name = winner["model"]
    model = trained.get(model_name)
    if model is None:
        raise ValueError(f"No hay instancia entrenada para {model_name}")

    ext = MODEL_EXTENSIONS[model_name]
    artifact = Path(save_dir) / f"{model_name}_{symbol}{ext}"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(artifact))

    metrics = {
        k: v
        for k, v in winner["metrics"].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    meta = {
        "model_name": model_name,
        "symbol": symbol,
        "extension": ext,
        "metric": metric,
        "version": _version(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    sidecar = Path(save_dir) / f"{model_name}_{symbol}.meta.json"
    sidecar.write_text(json.dumps(meta, indent=2, default=str))
    logger.info(f"Modelo ganador guardado: {artifact} (metric={metric})")

    return str(artifact), str(sidecar)


def save_summary(summary: pd.DataFrame, save_dir: str, symbol: str) -> str:
    """Guarda la tabla comparativa completa en JSON para trazabilidad."""
    path = Path(save_dir) / f"{symbol}_comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            summary.where(pd.notnull(summary), None).to_dict(orient="records"),
            indent=2,
            default=str,
        )
    )
    logger.info(f"Tabla comparativa guardada: {path}")
    return str(path)


def format_summary_table(summary: pd.DataFrame, metric: str = "roc_auc") -> str:
    """Renderiza la tabla comparativa en consola."""
    columns = [
        "model",
        "train_accuracy",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]
    header = (
        f"{'MODELO':<18} {'TRAIN':>7} {'ACC':>7} {'PREC':>7} {'REC':>7} "
        f"{'F1':>7} {'AUC':>7} {'PR_AUC':>7}  ESTADO"
    )
    lines = [header, "-" * len(header)]
    for _, row in summary.iterrows():
        cells = []
        for col in columns[1:]:
            val = row.get(col)
            cells.append(f"{val:>7.4f}" if isinstance(val, (int, float)) else f"{'-':>7}")
        lines.append(f"{str(row.get('model', '')):<18} {' '.join(cells)}  {row.get('status', '')}")
    if metric in summary.columns:
        lines.append("")
        lines.append(f"Mejor según '{metric}': {select_winner(summary, metric)}")
    return "\n".join(lines)
