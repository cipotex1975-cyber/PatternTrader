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
    # TODO(fase 3+): cablear early_stop_rounds/patience en el entrenamiento real y
    # implementar validación walk-forward con walk_forward_splits. Hoy son placeholders.
    early_stop_rounds: int = 20,
    patience: int = 5,
    walk_forward_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, BaseMLModel]]:
    """Entrena todos los modelos solicitados sobre el mismo split y compara métricas.

    FASE 2: la comparación/selección usa EXCLUSIVAMENTE VALIDATION. El TEST
    FINAL no entra en esta función; se evalúa una sola vez después, con
    ``evaluate_winner_on_test()``.

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
            )

            model = MLModelFactory.create_new(name, **kwargs)
            if scaler is not None and hasattr(model, "_scaler"):
                model._scaler = scaler

            # ---------------------------------------------------------
            # TRAIN
            # ---------------------------------------------------------
            train_metrics = model.train(
                X_tr,
                y_tr,
                feature_names=feature_names,
            )

            # ---------------------------------------------------------
            # EVALUACIÓN SOBRE VALIDATION
            # ---------------------------------------------------------
            eval_metrics = evaluate_model(
                model,
                X_ev,
                y_ev,
            )

            train_acc = train_metrics.get("train_accuracy")
            train_loss = train_metrics.get("loss")

            row: dict[str, Any] = {
                "model": name,
                "status": "ok",
                "train_accuracy": float(train_acc) if train_acc is not None else float("nan"),
                "train_loss": float(train_loss) if train_loss is not None else float("nan"),
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

            logger.info(f"{name}: " f"{train_repr} " f"{metric}={row.get(metric, 'n/a')}")

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

            model.train(X_tr, y_tr, feature_names=feature_names)
            fold_metrics = evaluate_model(model, X_ev, y_ev)
            per_fold[name].append(fold_metrics)

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
        row: dict[str, Any] = {
            "model": name,
            "status": "ok" if fold_list else "error: sin folds evaluados",
            "train_accuracy": float("nan"),
            "train_loss": float("nan"),
            "samples_train": 0,
            "samples_validation": 0,
            "wf_folds": len(fold_list),
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
) -> dict[str, float]:
    """Evalúa al ganador UNA sola vez sobre el TEST FINAL (FASE 2, sección 8).

    El resultado NO debe volver a entrar en ``select_winner()`` ni en ninguna
    decisión de selección/threshold/tuning: es sólo reporte final.
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
    return evaluate_model(model, X_final, y_final)


def _version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_winner(
    trained: dict[str, BaseMLModel],
    winner: dict[str, Any],
    save_dir: str,
    symbol: str,
    metric: str = "roc_auc",
    final_test_metrics: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Persiste el artefacto ganador con nomenclatura por par y su sidecar.

    Nombres: ``{modelo}_{symbol}{ext}`` + ``{modelo}_{symbol}.meta.json``
    (el sidecar permite al ScoringEngine rehidratar el modelo del par sin DB).
    Si se pasan ``final_test_metrics`` (evaluación única sobre TEST FINAL,
    FASE 2) quedan registrados en el sidecar con fines de trazabilidad.
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
