from __future__ import annotations

import json
import math
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

from app.core.config import Settings
from app.core.logger import get_logger
from app.ml.base import BaseMLModel
from app.ml.factory import MLModelFactory
from app.ml.training.data import SEQUENCE_MODELS, format_for_model
from app.ml.training.scaling import (
    SCALER_SIDECAR_STEM,
    apply_feature_scaling,
    scaler_to_artifact,
)

logger = get_logger("TrainAndCompare")

# Métricas comparables entre todos los modelos (0-1).
AVAILABLE_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")

# Familias de modelos (FASE 7). Los anomaly detectors NO se comparan con modelos
# supervisados en el ranking por defecto (se excluyen salvo `--include-anomaly-models`).
SUPERVISED_MODELS = {
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "lstm",
    "transformer",
    "cnn",
}
ANOMALY_MODELS = {"isolation_forest", "autoencoder"}

# Thresholds candidatos para la optimización de classification threshold sobre VALIDATION.
DEFAULT_THRESHOLD_CANDIDATES = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]


def model_family_label(model_type: str) -> str:
    """Etiqueta de familia de modelo para output (FASE 7, sección 6)."""
    if model_type == "classification":
        return "supervised classification"
    if model_type == "anomaly":
        return "anomaly detection"
    return model_type


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
    patience: int = 5,
    early_stop_rounds: int = 0,
) -> dict[str, Any]:
    """Hiperparámetros por modelo: defaults → config YAML → override explícito.

    FASE 6: ``patience`` va a los modelos secuenciales y ``early_stop_rounds`` a
    los árboles (XGBoost/LightGBM/CatBoost) para activar early stopping real.
    """
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
        kwargs["patience"] = patience

    if name == "autoencoder":
        kwargs["input_dim"] = n_features
        kwargs["epochs"] = epochs

    if name in ("xgboost", "lightgbm", "catboost"):
        kwargs["early_stopping_rounds"] = early_stop_rounds

    if hyperparams and name in hyperparams:
        kwargs.update(hyperparams[name])

    return kwargs


def classify_with_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Convierte probabilidades de la clase positiva en clases binarias.

    comportamiento:
        P >= threshold → 1
        P <  threshold → 0

    Ejemplo con threshold=0.50:
        P=[0.20, 0.42, 0.50, 0.61, 0.83] → [0, 0, 1, 1, 1]

    Valida que el threshold esté en [0, 1]. Esta función es un mecanismo
    de DECISIÓN: cambiar el threshold no altera el modelo ni las métricas
    basadas en probabilidades (ROC-AUC, PR-AUC).
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold debe estar en [0, 1]; recibido {threshold}")
    probs = np.asarray(probabilities, dtype=np.float64)
    return (probs >= threshold).astype(np.int64)


def metrics_at_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Métricas dependientes del threshold para una probabilidad dada.

    ROC-AUC y PR-AUC se calculan con las probabilidades (independientes del
    threshold); accuracy, precision, recall y f1 con la clasificación binaria
    inducida por ``classify_with_threshold``. Usa ``zero_division=0`` para
    thresholds degenerados (sin positivos ni negativos predichos).
    """
    labels = np.asarray(y_true)
    predictions = classify_with_threshold(probabilities, threshold)
    n = len(labels)
    positive_predictions = int(predictions.sum())
    predicted_positive_rate = positive_predictions / n if n > 0 else 0.0

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predictions)),
    }
    if len(np.unique(labels)) > 1:
        metrics["precision"] = float(precision_score(labels, predictions, zero_division=0))
        metrics["recall"] = float(recall_score(labels, predictions, zero_division=0))
        metrics["f1"] = float(f1_score(labels, predictions, zero_division=0))
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        metrics["pr_auc"] = float(average_precision_score(labels, probabilities))
    metrics["positive_predictions"] = float(positive_predictions)
    metrics["predicted_positive_rate"] = float(predicted_positive_rate)
    return metrics


def evaluate_model(
    model: BaseMLModel,
    X: np.ndarray,
    y: np.ndarray,
    classification_threshold: float = 0.50,
) -> dict[str, float]:
    """Métricas unificadas (0-1) para cualquier modelo de la plataforma.

    FASE 7: la DECISIÓN binaria (accuracy/precision/recall/f1) se deriva SIEMPRE de
    ``classify_with_threshold(predict_proba[:, 1], classification_threshold)`` en lugar de
    ``model.predict()`` (que usaba el threshold interno de cada modelo). ROC-AUC/PR-AUC se
    calculan con el score continuo (``predict_proba[:, 1]``) y son independientes del
    threshold.
    """
    probabilities = model.predict_proba(X)[:, 1]
    predictions = classify_with_threshold(probabilities, classification_threshold)
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


def optimize_classification_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    selection_metric: str = "f1",
    candidate_thresholds: list[float] | None = None,
) -> tuple[float, list[dict[str, float]]]:
    """Optimiza el classification threshold sobre VALIDATION (FASE 7, sección 7).

    Barre ``candidate_thresholds`` y, para cada uno, computa ``metrics_at_threshold``.
    Devuelve ``(best_threshold, table)`` donde ``table`` es la lista completa de filas
    (threshold + métricas) y ``best_threshold`` es el que maximiza ``selection_metric``.

    NUNCA debe usarse el TEST FINAL aquí: esto sólo escala sobre el bloque de selección.
    """
    thresholds = candidate_thresholds or list(DEFAULT_THRESHOLD_CANDIDATES)
    if selection_metric not in {"accuracy", "precision", "recall", "f1"}:
        raise ValueError(
            "selection_metric para threshold optimization debe ser una métrica "
            "dependiente del threshold (accuracy/precision/recall/f1)."
        )

    table: list[dict[str, float]] = []
    for t in thresholds:
        row = metrics_at_threshold(y_true, probabilities, t)
        row = {"threshold": float(t), **row}
        table.append(row)

    best = max(table, key=lambda r: float(r.get(selection_metric, float("-inf"))))
    return float(best["threshold"]), table


def _as_float(value: Any) -> float:
    """Convierte un valor de métrica a float; ``None``/inválido → NaN (FASE 6)."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_eval_sequences(
    context_X: np.ndarray,
    target_X: np.ndarray,
    y_target: np.ndarray,
    sequence_length: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Construye ventanas secuenciales para evaluar ``target`` con contexto previo.

    Las últimas ``sequence_length - 1`` filas de ``context_X`` (el segmento
    cronológicamente anterior) se usan como contexto histórico para construir
    las primeras ventanas de ``target_X``. Este contexto es anterior o
    contemporáneo al instante de predicción, por lo que NO es leakage
    (FASE 2, sección 9).

    Las etiquetas devueltas pertenecen exclusivamente a ``y_target``.
    """
    context = sequence_length - 1

    if len(context_X) < context:
        raise ValueError(
            f"Se necesitan al menos {context} muestras de contexto "
            f"para construir las secuencias."
        )

    X_context = np.concatenate(
        [context_X[-context:], target_X],
        axis=0,
    )

    windows = np.stack(
        [
            X_context[i - sequence_length + 1 : i + 1]
            for i in range(sequence_length - 1, len(X_context))
        ]
    )

    return windows, y_target


# Alias de compatibilidad: el antiguo nombre describía solo el caso test.
build_test_sequences = build_eval_sequences


def run_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    model_names: list[str] | None = None,
    metric: str = "roc_auc",
    feature_names: list[str] | None = None,
    sequence_length: int = 30,
    epochs: int = 10,
    settings: Settings | None = None,
    hyperparams: dict[str, dict[str, Any]] | None = None,
    feature_scaling: str = "none",
    early_stop_rounds: int = 20,
    patience: int = 5,
    walk_forward_splits: int = 5,
    classification_threshold: float = 0.50,
    exclude_anomaly: bool = True,
) -> tuple[pd.DataFrame, dict[str, BaseMLModel]]:
    """Entrena todos los modelos solicitados sobre el mismo split y compara métricas.

    FASE 2: la comparación/selección usa EXCLUSIVAMENTE VALIDATION. El TEST
    FINAL no entra en esta función; se evalúa una sola vez después, con
    ``evaluate_winner_on_test()``.

    FASE 6: ``X_val``/``y_val`` (el bloque VALIDATION ya construido) se pasa al
    ``model.train()`` para early stopping real / eval_set; el TEST FINAL nunca
    participa. ``patience`` aplica a secuenciales, ``early_stop_rounds`` a
    árboles.

    FASE 7: ``classification_threshold`` se aplica a accuracy/precision/recall/f1
    via ``classify_with_threshold``; ROC-AUC/PR-AUC usan score continuo. ``exclude_anomaly``
    quita del ranking a los anomaly detectors (IsolationForest/AutoEncoder) salvo que se
    pidan explícitamente por nombre. Cada fila del summary lleva ``model_family``.

    Para modelos secuenciales (LSTM, CNN, Transformer), las primeras ventanas
    de la evaluación utilizan las últimas `sequence_length - 1` muestras del
    train como contexto histórico (causal, no leakage).

    Retorna (summary, trained) donde ``summary`` es un DataFrame con una fila por
    modelo y ``trained`` mapea nombre → instancia entrenada.
    """
    if metric not in AVAILABLE_METRICS:
        raise ValueError(
            f"Metric desconocida: {metric}. " f"Válidas: {', '.join(AVAILABLE_METRICS)}"
        )

    registered = set(MLModelFactory.get_all())

    if not model_names or "all" in model_names:
        selected = sorted(registered)
        if exclude_anomaly:
            selected = [name for name in selected if name not in ANOMALY_MODELS]
            logger.info(
                "Anomaly detectors excluidos del ranking por defecto "
                "(usa --include-anomaly-models para incluirlos)."
            )
    else:
        selected = [name for name in model_names if name in registered]

    if not selected:
        raise ValueError(
            f"No hay modelos válidos entre {model_names}. "
            f"Registrados: {', '.join(sorted(registered))}"
        )

    rows: list[dict[str, Any]] = []
    trained: dict[str, BaseMLModel] = {}

    # FASE 4 — Preprocessing reproducible. El scaler (si ``feature_scaling`` es
    # ``standard``) se ajusta SOLO con TRAIN; VALIDATION y TEST usan ``transform``.
    # En ``none`` (default) se devuelven las matrices sin tocar y scaler=None.
    X_tr_scaled, X_val_scaled, _, scaler = apply_feature_scaling(
        X_train,
        X_validation,
        X_validation,
        mode=feature_scaling,
        feature_names=feature_names,
    )

    for name in selected:
        try:
            # ---------------------------------------------------------
            # TRAIN
            # ---------------------------------------------------------
            X_tr, y_tr = format_for_model(
                name,
                X_tr_scaled,
                y_train,
                sequence_length,
            )

            # ---------------------------------------------------------
            # VALIDATION (dataset de comparación/selección)
            # ---------------------------------------------------------
            if name in SEQUENCE_MODELS:
                # Para modelos secuenciales, utilizamos las últimas
                # sequence_length - 1 muestras del train como contexto
                # histórico para las primeras muestras de la validación.
                X_ev, y_ev = build_eval_sequences(
                    X_tr_scaled,
                    X_val_scaled,
                    y_validation,
                    sequence_length,
                )
            else:
                # Modelos tabulares/anomaly detection:
                # mantienen el comportamiento original.
                X_ev, y_ev = format_for_model(  # type: ignore[assignment]
                    name,
                    X_val_scaled,
                    y_validation,
                    sequence_length,
                )

            if y_tr is None or y_ev is None:
                raise ValueError(f"{name}: no se pudieron formatear las etiquetas")

            # ---------------------------------------------------------
            # CREAR MODELO
            # ---------------------------------------------------------
            kwargs = _model_kwargs(
                name,
                feature_names=feature_names,
                sequence_length=sequence_length,
                epochs=epochs,
                settings=settings,
                hyperparams=hyperparams,
                patience=patience,
                early_stop_rounds=early_stop_rounds,
            )

            model = MLModelFactory.create_new(name, **kwargs)
            if scaler is not None and hasattr(model, "_scaler"):
                model._scaler = scaler

            # ---------------------------------------------------------
            # TRAIN (FASE 6: early stopping real sobre VALIDATION)
            # ---------------------------------------------------------
            train_metrics = model.train(
                X_tr,
                y_tr,
                feature_names=feature_names,
                X_val=X_ev,
                y_val=y_ev,
            )

            # ---------------------------------------------------------
            # EVALUACIÓN SOBRE VALIDATION
            # ---------------------------------------------------------
            eval_metrics = evaluate_model(
                model,
                X_ev,
                y_ev,
                classification_threshold=classification_threshold,
            )

            # FASE 7 — Optimización de classification threshold sobre VALIDATION.
            # NUNCA se usa el TEST FINAL aquí; es sólo reporte de selección.
            try:
                val_probas = model.predict_proba(X_ev)[:, 1]
                opt_threshold, opt_table = optimize_classification_threshold(
                    y_ev, val_probas, selection_metric="f1"
                )
                opt_table_json: list[dict[str, float]] = [
                    {k: float(v) for k, v in r.items()} for r in opt_table
                ]
            except Exception as oe:  # noqa: BLE001
                logger.warning(f"{name}: no se pudo optimizar threshold: {oe}")
                opt_threshold, opt_table_json = float("nan"), []

            train_acc = train_metrics.get("train_accuracy")
            train_loss = train_metrics.get("loss", train_metrics.get("train_loss"))

            row: dict[str, Any] = {
                "model": name,
                "model_family": model_family_label(model.model_type),
                "status": "ok",
                "train_accuracy": float(train_acc) if train_acc is not None else float("nan"),
                "train_loss": float(train_loss) if train_loss is not None else float("nan"),
                "validation_accuracy": _as_float(train_metrics.get("validation_accuracy")),
                "validation_loss": _as_float(train_metrics.get("validation_loss")),
                "best_epoch": _as_float(train_metrics.get("best_epoch")),
                "best_validation_loss": _as_float(train_metrics.get("best_validation_loss")),
                "best_iteration": _as_float(train_metrics.get("best_iteration")),
                "early_stopping": bool(train_metrics.get("early_stopping", False)),
                "opt_best_threshold": float(opt_threshold),
                "threshold_table": opt_table_json,
                "samples_train": int(X_tr.shape[0]),
                "samples_validation": int(X_ev.shape[0]),
            }

            row.update(eval_metrics)
            rows.append(row)
            trained[name] = model

            if not math.isnan(row["train_accuracy"]):
                train_repr = f"train_acc={row['train_accuracy']:.4f}"
            elif not math.isnan(row["train_loss"]):
                train_repr = f"train_loss={row['train_loss']:.4f}"
            else:
                train_repr = "train=n/a"

            es_repr = ""
            if row.get("early_stopping"):
                es_repr = (
                    f" ES(best_epoch={row['best_epoch']:.0f} "
                    f"val_loss={row['best_validation_loss']:.4f})"
                )
            elif not math.isnan(row.get("best_epoch", float("nan"))):
                es_repr = f" ES(val_loss={row['best_validation_loss']:.4f})"

            logger.info(f"{name}: {train_repr} {metric}={row.get(metric, 'n/a')}{es_repr}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"{name}: falló el entrenamiento: {e}")

            rows.append(
                {
                    "model": name,
                    "status": f"error: {e}",
                    "train_accuracy": float("nan"),
                    "samples_train": 0,
                    "samples_validation": 0,
                }
            )

    summary = pd.DataFrame(rows)

    if summary.empty:
        raise RuntimeError("Ningún modelo se entrenó correctamente")

    return summary, trained


def run_walk_forward_comparison(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    model_names: list[str] | None = None,
    metric: str = "roc_auc",
    feature_names: list[str] | None = None,
    sequence_length: int = 30,
    epochs: int = 10,
    settings: Settings | None = None,
    hyperparams: dict[str, dict[str, Any]] | None = None,
    feature_scaling: str = "none",
    forward_periods: int = 5,
    min_train_size: int = 100,
    early_stop_rounds: int = 20,
    patience: int = 5,
    classification_threshold: float = 0.50,
    exclude_anomaly: bool = True,
) -> tuple[pd.DataFrame, dict[str, BaseMLModel]]:
    """Validación walk-forward (FASE 5): reentrena cada modelo en N folds expanding.

    Recibe el conjunto de selección (TRAIN+VALIDATION concatenados en orden
    cronológico) y genera ``n_splits`` folds con ``build_walk_forward_folds``.
    Para cada modelo y fold:

    - Se aplica el scaler (Fase 4) con ``fit`` SOLO en el train del fold.
    - Los secuenciales reconstruyen secuencias con el contexto del train del fold.
    - Se registran las métricas por fold (``fold_<metric>``).

    Selección (fase5.md, sección 7): el ganador se elige por la MEDIA de la
    métrica objetivo sobre los folds. ``trained`` conserva la instancia del
    fold más grande (el último), que servirá para evaluar el TEST FINAL aislado.

    Retorna (summary, trained). ``summary`` es un DataFrame con una fila por
    modelo, las columnas de Fase 2 (model/status/samples...) y los agregados
    ``wf_*`` (mean/std/min/max por métrica) calculados sobre los folds.
    """
    from app.ml.training.walk_forward import (
        build_walk_forward_folds,
        validate_walk_forward_no_future,
    )

    if metric not in AVAILABLE_METRICS:
        raise ValueError(
            f"Metric desconocida: {metric}. " f"Válidas: {', '.join(AVAILABLE_METRICS)}"
        )

    registered = set(MLModelFactory.get_all())
    if not model_names or "all" in model_names:
        selected = sorted(registered)
        if exclude_anomaly:
            selected = [name for name in selected if name not in ANOMALY_MODELS]
            logger.info(
                "Anomaly detectors excluidos del ranking por defecto "
                "(usa --include-anomaly-models para incluirlos)."
            )
    else:
        selected = [name for name in model_names if name in registered]
    if not selected:
        raise ValueError(
            f"No hay modelos válidos entre {model_names}. "
            f"Registrados: {', '.join(sorted(registered))}"
        )

    folds = build_walk_forward_folds(
        X,
        y,
        n_splits=n_splits,
        forward_periods=forward_periods,
        min_train_size=min_train_size,
        seq_context=sequence_length,
    )
    validate_walk_forward_no_future(folds)

    per_fold: dict[str, list[dict[str, float]]] = {name: [] for name in selected}
    trained: dict[str, BaseMLModel] = {}
    opt_by_model: dict[str, dict[str, Any]] = {}

    for name in selected:
        for fold in folds:
            X_tr_raw, y_tr_raw = fold.X_train, fold.y_train
            X_val_raw, y_val_raw = fold.X_validation, fold.y_validation

            X_tr_scaled, X_val_scaled, _, scaler = apply_feature_scaling(
                X_tr_raw,
                X_val_raw,
                X_val_raw,
                mode=feature_scaling,
                feature_names=feature_names,
            )

            kwargs = _model_kwargs(
                name,
                feature_names=feature_names,
                sequence_length=sequence_length,
                epochs=epochs,
                settings=settings,
                hyperparams=hyperparams,
                patience=patience,
                early_stop_rounds=early_stop_rounds,
            )
            model = MLModelFactory.create_new(name, **kwargs)
            if scaler is not None and hasattr(model, "_scaler"):
                model._scaler = scaler

            X_tr, y_tr = format_for_model(name, X_tr_scaled, y_tr_raw, sequence_length)
            if y_tr is None:
                raise ValueError(f"{name}: no se pudieron formatear etiquetas de train")

            if name in SEQUENCE_MODELS:
                X_ev, y_ev = build_eval_sequences(
                    X_tr_scaled,
                    X_val_scaled,
                    y_val_raw,
                    sequence_length,
                )
            else:
                X_ev, y_ev = format_for_model(  # type: ignore[assignment]
                    name, X_val_scaled, y_val_raw, sequence_length
                )

            if y_ev is None:
                raise ValueError(f"{name}: no se pudieron formatear etiquetas de validation")

            model.train(
                X_tr,
                y_tr,
                feature_names=feature_names,
                X_val=X_ev,
                y_val=y_ev,
            )
            fold_metrics = evaluate_model(
                model,
                X_ev,
                y_ev,
                classification_threshold=classification_threshold,
            )
            per_fold[name].append(fold_metrics)

            # FASE 7 — Optimización de threshold sobre el ÚLTIMO fold (el más grande).
            if fold.fold_index == len(folds) - 1:
                opt_threshold_, opt_table_ = optimize_classification_threshold(
                    y_ev, model.predict_proba(X_ev)[:, 1], selection_metric="f1"
                )
                opt_by_model[name] = {
                    "opt_best_threshold": float(opt_threshold_),
                    "threshold_table": [{k: float(v) for k, v in r.items()} for r in opt_table_],
                }

            # Conservar la instancia entrenada en el fold más grande (el último).
            if fold.fold_index == len(folds) - 1:
                trained[name] = model

    # --- Agregados walk-forward por modelo ---
    wf_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]
    rows: list[dict[str, Any]] = []
    for name in selected:
        fold_list = per_fold[name]
        trained_model = trained.get(name)
        row: dict[str, Any] = {
            "model": name,
            "model_family": (
                model_family_label(trained_model.model_type) if trained_model is not None else ""
            ),
            "status": "ok" if fold_list else "error: sin folds evaluados",
            "train_accuracy": float("nan"),
            "train_loss": float("nan"),
            "samples_train": 0,
            "samples_validation": 0,
            "wf_folds": len(fold_list),
            "opt_best_threshold": opt_by_model.get(name, {}).get(
                "opt_best_threshold", float("nan")
            ),
            "threshold_table": opt_by_model.get(name, {}).get("threshold_table", []),
        }
        for m in wf_metrics:
            values = [f[m] for f in fold_list if m in f]
            if values:
                arr = np.asarray(values, dtype=np.float64)
                row[f"wf_mean_{m}"] = float(np.mean(arr))
                row[f"wf_std_{m}"] = float(np.std(arr))
                row[f"wf_min_{m}"] = float(np.min(arr))
                row[f"wf_max_{m}"] = float(np.max(arr))
            else:
                row[f"wf_mean_{m}"] = float("nan")
                row[f"wf_std_{m}"] = float("nan")
                row[f"wf_min_{m}"] = float("nan")
                row[f"wf_max_{m}"] = float("nan")
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("Ningún modelo se evaluó correctamente en walk-forward")

    return summary, trained


def select_walk_forward_winner(
    summary: pd.DataFrame, metric: str = "roc_auc"
) -> dict[str, Any] | None:
    """Selecciona al ganador por la MEDIA de la métrica sobre los folds (FASE 5).

    La media ``wf_mean_<metric>`` se computa con las validaciones walk-forward
    y NO utiliza el TEST FINAL. Sirve para ``save_winner``/``register_in_db``.
    """
    mean_col = f"wf_mean_{metric}"
    if mean_col not in summary.columns:
        return None
    valid = summary[summary["status"] == "ok"].copy()
    valid = valid.dropna(subset=[mean_col])
    if valid.empty:
        return None
    best = valid.loc[valid[mean_col].idxmax()]
    return {"model": best["model"], "metrics": best.to_dict()}


def select_winner(summary: pd.DataFrame, metric: str = "roc_auc") -> dict[str, Any] | None:
    """Elige la fila con mejor métrica objetivo entre los modelos exitosos.

    FASE 2: ``summary`` contiene métricas calculadas sobre VALIDATION, por lo
    que la selección nunca usa el TEST FINAL.
    """
    valid = summary[summary["status"] == "ok"].copy()
    if metric not in valid.columns:
        return None
    valid = valid.dropna(subset=[metric])
    if valid.empty:
        return None
    best = valid.loc[valid[metric].idxmax()]
    return {"model": best["model"], "metrics": best.to_dict()}


def evaluate_winner_on_test(
    trained: dict[str, BaseMLModel],
    winner_name: str,
    X_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sequence_length: int = 30,
    classification_threshold: float = 0.50,
) -> dict[str, float]:
    """Evalúa al ganador UNA sola vez sobre el TEST FINAL (FASE 2, sección 8).

    El resultado NO debe volver a entrar en ``select_winner()`` ni en ninguna
    decisión de selección/threshold/tuning: es sólo reporte final.

    FASE 7: ``classification_threshold`` se aplica a accuracy/precision/recall/f1;
    ROC-AUC/PR-AUC usan score continuo e independiente del threshold.
    """
    model = trained.get(winner_name)
    if model is None:
        raise ValueError(f"No hay instancia entrenada para {winner_name}")

    # FASE 4: aplicar el MISMO scaler (fit con TRAIN) a validation y test
    # antes de construir las secuencias de evaluación final.
    X_val = X_validation
    X_te = X_test
    scaler = getattr(model, "_scaler", None)
    if scaler is not None:
        X_val = scaler.transform(X_validation)
        X_te = scaler.transform(X_test)

    X_final: np.ndarray
    y_final: np.ndarray | None
    if winner_name in SEQUENCE_MODELS:
        # Contexto causal: cola de validation para las primeras ventanas de test.
        X_final, y_final = build_eval_sequences(
            X_val,
            X_te,
            y_test,
            sequence_length,
        )
    else:
        X_final, y_final = format_for_model(winner_name, X_te, y_test, sequence_length)

    if y_final is None:
        raise ValueError(f"{winner_name}: no se pudieron formatear las etiquetas de test")

    logger.info(f"Evaluación final del ganador {winner_name} sobre TEST FINAL")
    return evaluate_model(
        model, X_final, y_final, classification_threshold=classification_threshold
    )


def _version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_winner(
    trained: dict[str, BaseMLModel],
    winner: dict[str, Any],
    save_dir: str,
    symbol: str,
    metric: str = "roc_auc",
    final_test_metrics: dict[str, float] | None = None,
    sidecar_context: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Persiste el artefacto ganador con nomenclatura por par y su sidecar.

    Nombres: ``{modelo}_{symbol}{ext}`` + ``{modelo}_{symbol}.meta.json``
    (el sidecar permite al ScoringEngine rehidratar el modelo del par sin DB).
    Si se pasan ``final_test_metrics`` (evaluación única sobre TEST FINAL,
    FASE 2) quedan registrados en el sidecar con fines de trazabilidad.

    El parámetro opcional ``sidecar_context`` (FASE 9) es un dict de bloques
    ricos de reproducibilidad (dataset, features, label, training, software,
    git, random_seed, etc.) producido por ``build_model_sidecar_context``; se
    fusiona en el ``.meta.json``. Un modelo existente con scaler puede
    sobreescribir el bloque ``preprocessing`` con el artefacto reproducible del
    FASE 4.
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
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
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
    if final_test_metrics is not None:
        meta["final_test_metrics"] = {
            k: float(v)
            for k, v in final_test_metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
        }
    # FASE 9 — Fusionar la metadata rica de reproducibilidad (dataset/features/
    # label/training/software/git/random_seed). Si viene, sobreescribe las claves
    # base que coincidan con el bloque de contexto.
    if sidecar_context:
        meta.update(sidecar_context)
    sidecar = Path(save_dir) / f"{model_name}_{symbol}.meta.json"

    # FASE 4 — Persistir el preprocessing reproducible. Si el ganador lleva un
    # scaler (mode=standard, fit TRAIN_ONLY) se guarda su artefacto JSON y se
    # registra el bloque ``preprocessing`` en el sidecar para rehidratarlo en
    # serving (raw → scaler → sequence → model).
    sidecar_extra = ""
    scaler = getattr(model, "_scaler", None)
    feature_names = list(getattr(model, "_feature_names", []) or [])
    if scaler is not None and feature_names:
        scaler_path = Path(save_dir) / f"{model_name}_{symbol}.{SCALER_SIDECAR_STEM}.json"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        scaler_path.write_text(
            json.dumps(scaler_to_artifact(scaler, feature_names), indent=2, default=str)
        )
        meta["preprocessing"] = scaler_to_artifact(scaler, feature_names)
        sidecar_extra = str(scaler_path)

    sidecar.write_text(json.dumps(meta, indent=2, default=str))
    logger.info(f"Modelo ganador guardado: {artifact} (metric={metric})")
    if sidecar_extra:
        logger.info(f"Scaler de preprocesamiento guardado: {sidecar_extra}")

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
        "train_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]
    header = (
        f"{'MODELO':<18} {'T.ACC':>7} {'LOSS':>7} {'ACC':>7} {'PREC':>7} {'REC':>7} "
        f"{'F1':>7} {'AUC':>7} {'PR_AUC':>7}  FAMILIA"
    )
    lines = [header, "-" * len(header)]
    for _, row in summary.iterrows():
        cells = []
        for col in columns[1:]:
            val = row.get(col)
            cells.append(f"{val:>7.4f}" if isinstance(val, (int, float)) else f"{'-':>7}")
        family = str(row.get("model_family", "")) if "model_family" in summary else ""
        lines.append(f"{str(row.get('model', '')):<18} {' '.join(cells)}  {family}")
    if metric in summary.columns:
        lines.append("")
        lines.append(f"Mejor según '{metric}': {select_winner(summary, metric)}")
    return "\n".join(lines)


def format_walk_forward_table(summary: pd.DataFrame, metric: str = "roc_auc") -> str:
    """Renderiza la tabla de VALIDATION walk-forward (FASE 10, sección VALIDATION).

    Muestra por modelo su media de ROC-AUC, su desviación estándar y la media de
    PR-AUC a partir de los agregados ``wf_*`` del summary. NA se renderiza como
    ``-``. La tabla se ordena por la media de la métrica objetivo descendente
    para resaltar el mejor.
    """
    mean_col = f"wf_mean_{metric}"
    std_col = f"wf_std_{metric}"
    header = f"{'MODEL':<12} {'MEAN_AUC':>10} {'STD_AUC':>9} {'MEAN_PR_AUC':>12}"
    lines = [header, "-" * len(header)]

    sort_col = mean_col if mean_col in summary.columns else "model"
    view = summary.sort_values(sort_col, ascending=False) if sort_col != "model" else summary

    for _, row in view.iterrows():
        mean_auc = row.get(mean_col)
        std_auc = row.get(std_col)
        pr_auc = row.get("wf_mean_pr_auc")

        def fmt(v: Any) -> str:
            return f"{float(v):.4f}" if isinstance(v, (int, float)) else "-"

        lines.append(
            f"{str(row.get('model', '')):<12} {fmt(mean_auc):>10} "
            f"{fmt(std_auc):>9} {fmt(pr_auc):>12}"
        )
    return "\n".join(lines)


def classify_signal(
    wf_mean_auc: float | None,
    wf_std_auc: float | None,
    wf_mean_pr_auc: float | None,
    final_oos_auc: float | None,
    positive_ratio: float | None = None,
    *,
    threshold_strong: float = 0.60,
    threshold_possible: float = 0.55,
    max_std_robust: float = 0.03,
    oos_tolerance: float = 0.05,
) -> str:
    """Clasifica la señal según la sección CONCLUSIÓN de la FASE 10.

    Reglas (umbrales fijos con defaults sensatos, config radio simple):
    - ROBUST SIGNAL: media AUC alta (>= threshold_strong), std bajo
      (<= max_std_robust), PR-AUC razonable (>= 1.5x positive_ratio si se conoce,
      sino >= 0.10) y final OOS consistente (dentro de ``oos_tolerance`` de la
      media, y >= threshold_possible).
    - POSSIBLE SIGNAL: media AUC moderada (>= threshold_possible) o final OOS
      dentro de tolerancia sin llegar a robusto.
    - WEAK SIGNAL: hay pico/evidencia parcial pero inestable o colapso OOS.
    - NO EVIDENCE: media AUC baja (< threshold_possible).

    Devuelve solo la etiqueta (sin afirmar rentabilidad): el detalle se deja al
    bloque de la comparación histórica.
    """
    if wf_mean_auc is None or not (isinstance(wf_mean_auc, (int, float))):
        return "NO EVIDENCE"
    if wf_mean_auc < threshold_possible:
        return "NO EVIDENCE"

    std = wf_std_auc if isinstance(wf_std_auc, (int, float)) else float("nan")
    pr = wf_mean_pr_auc if isinstance(wf_mean_pr_auc, (int, float)) else float("nan")

    def _nan(v: float) -> bool:
        return v != v  # NaN

    # Baseline PR-AUC: el de un clasificador aleatorio depende del positivos.
    if isinstance(positive_ratio, (int, float)) and 0 <= positive_ratio <= 1:
        pr_baseline = positive_ratio
    else:
        pr_baseline = 0.15

    # Condiciones del caso robusto.
    std_ok = not _nan(std) and std <= max_std_robust + 1e-9
    pr_ok = (not _nan(pr)) and pr >= max(pr_baseline * 1.5, 0.10) - 1e-9
    oos_ok = False
    if isinstance(final_oos_auc, (int, float)):
        oos_ok = (
            abs(final_oos_auc - wf_mean_auc) <= oos_tolerance
            and final_oos_auc >= threshold_possible
        )

    if wf_mean_auc >= threshold_strong and std_ok and pr_ok and oos_ok:
        return "ROBUST SIGNAL"

    if wf_mean_auc >= threshold_possible and oos_ok:
        return "POSSIBLE SIGNAL"

    if wf_mean_auc >= threshold_possible:
        # Hay media razonable pero inestable o colapso OOS.
        return "WEAK SIGNAL"

    return "NO EVIDENCE"
