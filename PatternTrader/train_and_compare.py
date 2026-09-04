from __future__ import annotations

import argparse
import asyncio
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config.settings import get_settings  # noqa: E402
from app.core.logger import get_logger  # noqa: E402
from app.ml.training import (  # noqa: E402
    FEATURE_NAMES,
    assess_robustness,
    build_model_sidecar_context,
    build_walk_forward_folds,
    classify_signal,
    create_features,
    create_labels,
    evaluate_winner_on_test,
    format_label_sweep_table,
    format_summary_table,
    format_walk_forward_table,
    load_data,
    model_family_label,
    run_comparison,
    run_label_sweep,
    run_walk_forward_comparison,
    save_summary,
    save_winner,
    seed_all,
    select_walk_forward_winner,
    select_winner,
    split_chronological,
    validate_walk_forward_no_future,
)
from app.ml.training.compare import AVAILABLE_METRICS  # noqa: E402

logger = get_logger("TrainAndCompareCLI")

METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def _winner_family(winner: dict) -> str:
    """Devuelve la familia del ganador (FASE 7) desde la fila del summary."""
    metrics = winner.get("metrics", {})
    if isinstance(metrics, dict) and metrics.get("model_family"):
        return str(metrics["model_family"])
    return "unknown"


def format_early_stopping_block(winner: dict, patience: int) -> str:
    """Bloque de output para early stopping del ganador (FASE 6, sección 6)."""
    metrics = winner.get("metrics", {})
    enabled = bool(metrics.get("early_stopping", False))
    lines = ["Early stopping:", f"  enabled: {str(enabled).lower()}"]
    if enabled:
        lines.append(f"  patience: {patience}")
        best_epoch = metrics.get("best_epoch")
        if best_epoch is not None and not (
            isinstance(best_epoch, float) and math.isnan(best_epoch)
        ):
            lines.append(f"  best_epoch: {int(best_epoch)}")
        best_val_loss = metrics.get("best_validation_loss")
        if best_val_loss is not None and not (
            isinstance(best_val_loss, float) and math.isnan(best_val_loss)
        ):
            lines.append(f"  best_val_loss: {float(best_val_loss):.4f}")
        best_iteration = metrics.get("best_iteration")
        if best_iteration is not None and not (
            isinstance(best_iteration, float) and math.isnan(best_iteration)
        ):
            lines.append(f"  best_iteration: {int(best_iteration)}")
    return "\n".join(lines)


def format_threshold_optimization_table(
    table: list[dict[str, float]],
    best_threshold: float,
    selection_metric: str,
) -> str:
    """Renderiza la tabla de optimización de threshold (FASE 7, sección 7)."""
    header = f"{'threshold':>9} {'precision':>9} {'recall':>9} {'F1':>9}"
    lines = [
        "Threshold Optimization (VALIDATION)",
        "==========================================",
        header,
        "-" * len(header),
    ]
    for row in table:
        lines.append(
            f"{row['threshold']:>9.2f} "
            f"{row.get('precision', float('nan')):>9.4f} "
            f"{row.get('recall', float('nan')):>9.4f} "
            f"{row.get('f1', float('nan')):>9.4f}"
        )
    metric_value = next(
        (r.get(selection_metric) for r in table if r["threshold"] == best_threshold),
        float("nan"),
    )
    lines.append("")
    lines.append(
        f"Best threshold: {best_threshold:.2f} " f"({selection_metric} = {metric_value:.4f})"
    )
    return "\n".join(lines)


def validate_split_sizes(train_size: float, validation_size: float, test_size: float) -> None:
    """Valida que train+validation+test sea 1 con tolerancia de punto flotante."""
    total = train_size + validation_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train + validation + test debe ser 1 (tolerancia 1e-6); recibido {total}"
        )
    if min(train_size, validation_size, test_size) <= 0:
        raise ValueError("train/validation/test deben ser positivos")


def derive_symbol(file_path: str) -> str:
    """Deriva el símbolo del nombre de archivo (p.ej. USDCAD_H1_... → USDCAD)."""
    return Path(file_path).stem.split("_")[0]


def derive_timeframe(file_path: str) -> str:
    """Deriva el timeframe del nombre de archivo (H1, 1h, 4h, 1d...)."""
    parts = Path(file_path).stem.split("_")
    if len(parts) >= 2:
        candidate = parts[1]
        lowered = candidate.lower()
        if lowered[0] in ("h", "m", "d", "w") or lowered[-1] in ("h", "m", "d", "w"):
            return candidate
    return "H1"


async def register_in_db(
    symbol: str,
    timeframe: str,
    winner: dict,
    artifact_path: str,
    metric: str,
    *,
    promote: bool = False,
) -> bool:
    """Registra el modelo ganador por par en `ml_models`.

    FASE 1.1: un fallo de conexión/SQLAlchemy no propaga la excepción ni
    invalida el entrenamiento; se registra el error y devuelve False.

    FASE 11: por defecto el modelo se registra como INACTIVO (no toca el activo
    previo). Solo con ``promote=True`` se ejecuta la promoción explícita y
    atómica (desactiva el activo previo del símbolo y activa el nuevo).
    """
    from sqlalchemy.exc import SQLAlchemyError

    from app.database.repositories import MLModelRepository

    model_name = winner["model"]
    metrics = {
        k: v
        for k, v in winner["metrics"].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
    }
    name = f"{model_name}_{symbol}"
    repo = MLModelRepository()
    try:
        if promote:
            await repo.promote(
                name=name,
                symbol=symbol,
                model_type="classification",
                version=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
                path=artifact_path,
                metrics=metrics,
                trained_at=datetime.now(timezone.utc),
                metadata={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "metric": metric,
                    "model_name": model_name,
                },
            )
        else:
            await repo.upsert(
                name=name,
                model_type="classification",
                version=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
                path=artifact_path,
                metrics=metrics,
                is_active=False,
                trained_at=datetime.now(timezone.utc),
                metadata={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "metric": metric,
                    "model_name": model_name,
                },
            )
    except (SQLAlchemyError, ConnectionError, OSError) as exc:
        logger.opt(exception=True).warning(f"Database registration failed: {exc}")
        print("\nDatabase registration:\n  FAILED\n")
        print(f"Reason:\n  {exc}\n")
        print("INFO: Training artifacts remain valid.")
        return False
    logger.info(
        f"Modelo registrado en DB: {model_name}_{symbol} ({'activo' if promote else 'inactivo'})"
    )
    return True


def _print_fase10_report(
    summary: pd.DataFrame,
    winner: dict[str, Any],
    final_metrics: dict[str, float],
    metric: str,
    label: str,
    positive_ratio: float | None,
) -> None:
    """Imprime el bloque COMPARACIÓN HISTÓRICA y CONCLUSIÓN de la FASE 10.

    Muestra las tres cifras de referencia:
    - Original experiment: LSTM TEST ROC-AUC = 0.6513 (de fase1.1), con la
      aclaración de que NO era OOS puro porque el TEST también participó en la
      selección del ganador.
    - Nuevo validation: media ROC-AUC del ganador sobre los folds walk-forward.
    - Nuevo final OOS: evaluación única del ganador sobre el TEST FINAL.

    NO busca maximizar el resultado para que coincida con 0.6513 y NO afirma
    rentabilidad.
    """
    mean_auc = winner["metrics"].get(f"wf_mean_{metric}")
    std_auc = winner["metrics"].get(f"wf_std_{metric}")
    mean_pr = winner["metrics"].get("wf_mean_pr_auc")
    final_oos_auc = final_metrics.get(metric)

    print("\nCOMPARACIÓN HISTÓRICA")
    print("=====================")
    print("\nOriginal experiment:")
    print("  LSTM TEST ROC-AUC = 0.6513")
    print(
        "  Nota: 0.6513 NO era una evaluación OOS pura porque el TEST FINAL "
        "también se utilizó para seleccionar el ganador en la fase original."
    )
    print("\nNuevo validation (media de folds walk-forward):")
    if isinstance(mean_auc, (int, float)):
        print(f"  {winner['model']} mean {label} = {float(mean_auc):.4f}")
    else:
        print(f"  {winner['model']} mean {label} = N/D")
    print("\nNuevo final OOS (evaluación única sobre TEST FINAL):")
    if isinstance(final_oos_auc, (int, float)):
        print(f"  {winner['model']} {label} = {float(final_oos_auc):.4f}")
    else:
        print(f"  {winner['model']} {label} = N/D")
    print(
        "\n  IMPORTANTE: este reporte NO intenta maximizar el resultado para "
        "coincidir con 0.6513."
    )

    verdict = classify_signal(
        float(mean_auc) if isinstance(mean_auc, (int, float)) else None,
        float(std_auc) if isinstance(std_auc, (int, float)) else None,
        float(mean_pr) if isinstance(mean_pr, (int, float)) else None,
        float(final_oos_auc) if isinstance(final_oos_auc, (int, float)) else None,
        positive_ratio=positive_ratio,
    )
    print("\nCONCLUSIÓN")
    print("==========")
    print(f"\nClasificación de la señal: {verdict}")
    print("  (Solo diagnóstico de la señal; NO se afirma rentabilidad todavía.)")


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena y compara los modelos ML de la plataforma sobre un par, "
            "elige el mejor y lo persiste con nomenclatura por símbolo."
        )
    )
    parser.add_argument("data_file", type=str, help="Archivo OHLCV tab-delimited (MT4/MT5)")
    parser.add_argument("--symbol", type=str, default=None, help="Símbolo (derivado del archivo)")
    parser.add_argument(
        "--timeframe", type=str, default=None, help="Timeframe (derivado del archivo)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modelo(s) a entrenar, separados por coma o repetible (default: all)",
        action="append",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="roc_auc",
        choices=list(AVAILABLE_METRICS),
        help="Métrica objetivo para elegir el ganador (default: roc_auc)",
    )
    parser.add_argument("--save-dir", type=str, default=None, help="Directorio de modelos")
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.70,
        help="Proporción cronológica de TRAIN (default: 0.70)",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.15,
        help="Proporción cronológica de VALIDATION (default: 0.15)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Proporción cronológica de TEST FINAL (default: 0.15)",
    )
    parser.add_argument(
        "--forward-periods", type=int, default=5, help="Velas hacia adelante para labels"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.001, help="Umbral de retorno para label positivo"
    )
    parser.add_argument(
        "--sequence-length", type=int, default=30, help="Longitud de ventana (modelos secuenciales)"
    )
    parser.add_argument(
        "--feature-scaling",
        type=str,
        default="none",
        choices=["none", "standard"],
        help=(
            "Preprocessing sobre features: none (sin escalar) o standard "
            "(StandardScaler fit SOLO con TRAIN). Default: none"
        ),
    )
    parser.add_argument("--epochs", type=int, default=10, help="Épocas (modelos secuenciales)")
    parser.add_argument(
        "--min-up-moves",
        type=int,
        default=2,
        help=(
            "Número mínimo de forward periods que deben superar el umbral "
            "para etiquetar como positivo"
        ),
    )
    parser.add_argument(
        "--walk-forward-splits",
        type=int,
        default=1,
        help=(
            "Número de folds para validación walk-forward (FASE 5). Default 1 = "
            "desactivado (selección por VALIDATION de Fase 2). Con N>1 se "
            "selecciona por la media de la métrica sobre N folds expanding; "
            "el TEST FINAL queda aislado."
        ),
    )
    parser.add_argument(
        "--early-stop-rounds",
        type=int,
        default=20,
        help=(
            "Early-stopping rounds para modelos de árbol (XGBoost/LightGBM/CatBoost). "
            "Usa eval_set sobre VALIDATION + early stopping nativo; "
            "0 = sin early stopping (entrena n_estimators/iterations completos)."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help=(
            "Patience para early-stopping en modelos secuenciales (LSTM/CNN/Transformer): "
            "epochs consecutivos sin mejora de VALIDATION loss antes de detener y "
            "restaurar el mejor estado."
        ),
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.50,
        help=(
            "Umbral para accuracy/precision/recall/F1 en la comparación y el TEST FINAL "
            "(FASE 7). ROC-AUC/PR-AUC usan score continuo y NO dependen del threshold. "
            "Default: 0.50"
        ),
    )
    parser.add_argument(
        "--include-anomaly-models",
        action="store_true",
        help=(
            "Incluir anomaly detectors (IsolationForest, AutoEncoder) en la comparación/ranking. "
            "Por defecto NO se comparan con modelos supervisados."
        ),
    )
    parser.add_argument(
        "--label-sweep",
        action="store_true",
        help=(
            "Modo FASE 8: barrido de robustez de la definición del label (threshold x "
            "min_up_moves) sobre TRAIN/VALIDATION + walk-forward. NO selecciona ganador, "
            "NO evalúa TEST FINAL y NO persiste artefactos ni registra en DB."
        ),
    )
    parser.add_argument(
        "--sweep-model",
        type=str,
        default=None,
        help=(
            "Modelo(s) usados en el --label-sweep (separados por coma; default: random_forest). "
            "Solo modelos tabulares supervisados."
        ),
    )
    parser.add_argument(
        "--sweep-metric",
        type=str,
        default="roc_auc",
        choices=list(AVAILABLE_METRICS),
        help="Métrica de referencia para el diagnóstico del --label-sweep (default: roc_auc)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Semilla para reproducibilidad (FASE 9): fija random/NumPy/PyTorch "
            "antes de entrenar y la registra en el sidecar .meta.json. "
            "Default: None (sin fijar, aleatorio)."
        ),
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Registrar el ganador en MLModelRepository (Postgres/SQLite) como INACTIVO",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promoción explícita: activa el modelo y desactiva el anterior (solo con --db)",
    )
    parser.add_argument("--no-save", action="store_true", help="No persistir el artefacto ganador")
    args = parser.parse_args(argv)

    try:
        validate_split_sizes(args.train_size, args.validation_size, args.test_size)
    except ValueError as exc:
        parser.error(str(exc))

    if not 0.0 <= args.classification_threshold <= 1.0:
        parser.error("--classification-threshold debe estar en [0, 1]")

    # FASE 9 — Fijar semillas globales para reproducibilidad del entrenamiento.
    if args.seed is not None:
        seed_all(args.seed)
        logger.info(f"Semillas fijadas: Python/NumPy/PyTorch (seed={args.seed})")

    settings = get_settings()
    save_dir = args.save_dir or settings.ml.model_path
    symbol = args.symbol or derive_symbol(args.data_file)
    timeframe = args.timeframe or derive_timeframe(args.data_file)

    if args.walk_forward_splits > 1:
        logger.info(
            f"Modo walk-forward activo: {args.walk_forward_splits} splits "
            "(selección por media de folds; TEST FINAL aislado)"
        )

    print(f"\nCargando datos: {args.data_file}")
    df = load_data(args.data_file)
    print(f"Candles cargadas: {len(df)} | símbolo: {symbol} | timeframe: {timeframe}")

    if args.label_sweep:
        print("\n=== LABEL ROBUSTNESS SWEEP (FASE 8) ===")
        print("Evaluación SOLO sobre TRAIN/VALIDATION + walk-forward.")
        print("El TEST FINAL NO participa del barrido.")
        sweep_model: list[str] = []
        if args.sweep_model:
            for group in args.sweep_model.split(","):
                sweep_model.extend(n.strip() for n in group.split(",") if n.strip())
        if not sweep_model:
            sweep_model = ["random_forest"]
        splits = max(2, args.walk_forward_splits if args.walk_forward_splits > 1 else 3)
        print(
            f"Modelos: {', '.join(sweep_model)} | "
            f"walk-forward splits: {splits} | métrica diagnóstico: {args.sweep_metric}"
        )
        sweep_df = run_label_sweep(
            df,
            model_names=sweep_model,
            metric=args.sweep_metric,
            walk_forward_splits=splits,
            settings=settings,
        )
        print("\n" + format_label_sweep_table(sweep_df))
        print("\n" + assess_robustness(sweep_df, metric=args.sweep_metric))
        print("\nNOTA: el sweep solo reporta robustez y NO selecciona configuración por TEST.")
        return

    print("\nCalculando indicadores y labels...")
    df = create_features(df)
    df["label"] = create_labels(
        df,
        forward_periods=args.forward_periods,
        threshold=args.threshold,
        min_up_moves=args.min_up_moves,
    )
    df = df.dropna(subset=FEATURE_NAMES + ["label"])
    print(f"Muestras tras limpiar NaN: {len(df)}")
    print(
        f"Distribución labels: positivos {int(df['label'].sum())} "
        f"({df['label'].mean():.2%}) / negativos {int((df['label'] == 0).sum())}"
    )

    split = split_chronological(
        df,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        forward_periods=args.forward_periods,
    )

    label = metric_label(args.metric)
    print("\nChronological split")
    print("==================")
    for name, key in (("TRAIN", "train"), ("VALIDATION", "validation"), ("TEST FINAL", "test")):
        info = split.ranges[key]
        print(f"\n{name}")
        print(f"  samples: {info['samples']}")
        if info["start"]:
            print(f"  start: {info['start']}")
            print(f"  end: {info['end']}")

    model_names: list[str] = []
    for group in args.model or ["all"]:
        model_names.extend(name.strip() for name in group.split(",") if name.strip())
    print(f"\nEntrenando modelos: {', '.join(model_names)} (métrica objetivo: {args.metric})")

    walk_forward = args.walk_forward_splits > 1

    if walk_forward:
        # FASE 5 — Conjunto de selección = TRAIN + VALIDATION (cronológico).
        X_selection = np.concatenate([split.X_train, split.X_validation], axis=0)
        y_selection = np.concatenate([split.y_train, split.y_validation], axis=0)
        wf_folds = build_walk_forward_folds(
            X_selection,
            y_selection,
            n_splits=args.walk_forward_splits,
            forward_periods=args.forward_periods,
            min_train_size=100,
            seq_context=args.sequence_length,
        )
        validate_walk_forward_no_future(wf_folds)
        print(f"\nWalk-forward validation: {len(wf_folds)} folds expanding window")

        summary, trained = run_walk_forward_comparison(
            X_selection,
            y_selection,
            n_splits=args.walk_forward_splits,
            model_names=model_names,
            metric=args.metric,
            feature_names=FEATURE_NAMES,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            settings=settings,
            feature_scaling=args.feature_scaling,
            forward_periods=args.forward_periods,
            early_stop_rounds=args.early_stop_rounds,
            patience=args.patience,
            classification_threshold=args.classification_threshold,
            exclude_anomaly=not args.include_anomaly_models,
        )
        winner = select_walk_forward_winner(summary, args.metric)
        winner_metric = winner["metrics"].get(f"wf_mean_{args.metric}") if winner else None
        eval_context = X_selection
    else:
        summary, trained = run_comparison(
            split.X_train,
            split.y_train,
            split.X_validation,
            split.y_validation,
            model_names=model_names,
            metric=args.metric,
            feature_names=FEATURE_NAMES,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            settings=settings,
            feature_scaling=args.feature_scaling,
            early_stop_rounds=args.early_stop_rounds,
            patience=args.patience,
            classification_threshold=args.classification_threshold,
            exclude_anomaly=not args.include_anomaly_models,
        )
        winner = select_winner(summary, args.metric)
        winner_metric = winner["metrics"].get(args.metric) if winner else None
        eval_context = split.X_validation

    if walk_forward:
        # FASE 10 — Tabla de VALIDATION walk-forward (MODEL | MEAN_AUC | STD_AUC | MEAN_PR_AUC).
        print("\nVALIDATION (walk-forward folds)\n================================")
        print("\n" + format_walk_forward_table(summary, args.metric))
        print(
            "\nNota: selección SOLO por media de la métrica sobre los folds; "
            "el TEST FINAL no participa."
        )
    else:
        print("\n" + format_summary_table(summary, args.metric))

    if winner is None:
        print("\nNo hubo modelos exitosos. Revisa los errores arriba.")
        return

    if walk_forward:
        print("\nModel selection")
        print("===============")
        print("\nSelection dataset : WALK-FORWARD VALIDATION (media de folds)")
        print(f"Selection metric  : mean {label}")
        print("\nWinner:")
        print(f"  {winner['model']}")
        print(f"  Model family : {model_family_label(_winner_family(winner))}")
        if winner_metric is not None:
            print(f"  mean {label} (folds) = {winner_metric:.4f}")
    else:
        print("\nModel selection")
        print("===============")
        print("\nSelection dataset : VALIDATION")
        print(f"Selection metric  : {label}")
        print("\nWinner:")
        print(f"  {winner['model']}")
        print(f"  Model family : {model_family_label(_winner_family(winner))}")
        if winner_metric is not None:
            print(f"  validation {label} = {winner_metric:.4f}")

    es_block = format_early_stopping_block(winner, args.patience)
    if "  enabled: true" in es_block:
        print(f"\n{es_block}")

    # FASE 7 — Reporte de optimización de threshold sobre VALIDATION (no TEST).
    winner_metrics = winner.get("metrics", {}) or {}
    threshold_table = winner_metrics.get("threshold_table")
    if (
        isinstance(threshold_table, list)
        and threshold_table
        and isinstance(threshold_table[0], dict)
    ):
        table_rows: list[dict[str, float]] = [
            {k: float(v) for k, v in r.items()} for r in threshold_table
        ]
        opt_threshold = winner_metrics.get("opt_best_threshold")
        best_t = float(opt_threshold) if isinstance(opt_threshold, (int, float)) else float("nan")
        print("\n" + format_threshold_optimization_table(table_rows, best_t, "f1"))

    # Evaluación única del ganador sobre TEST FINAL (nunca vuelve a selección).
    final_metrics: dict[str, float] = {}
    try:
        final_metrics = evaluate_winner_on_test(
            trained,
            str(winner["model"]),
            eval_context,
            split.X_test,
            split.y_test,
            sequence_length=args.sequence_length,
            classification_threshold=args.classification_threshold,
        )
    except ValueError as exc:
        logger.warning(f"FINAL OUT-OF-SAMPLE TEST no disponible: {exc}")

    print("\nFINAL OUT-OF-SAMPLE TEST")
    print("========================")
    print(f"\nModel: {winner['model']}")
    print(f"Model family : {model_family_label(_winner_family(winner))}")
    if final_metrics:
        for key in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"):
            if key in final_metrics:
                print(f"{metric_label(key)}: {final_metrics[key]:.4f}")
    else:
        print("  SKIPPED (dataset de test insuficiente para este modelo)")

    if walk_forward:
        _print_fase10_report(
            summary,
            winner,
            final_metrics,
            metric=args.metric,
            label=label,
            positive_ratio=float((df["label"] == 1).mean()) if "label" in df.columns else None,
        )

    if not args.no_save:
        sidecar_context = build_model_sidecar_context(
            model_name=str(winner["model"]),
            data_path=args.data_file,
            ranges=split.ranges,
            samples_total=len(df),
            feature_names=list(FEATURE_NAMES),
            forward_periods=args.forward_periods,
            threshold=args.threshold,
            min_up_moves=args.min_up_moves,
            preprocessing_type="standard" if args.feature_scaling == "standard" else "none",
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            settings=settings,
            hyperparams=None,
            patience=args.patience,
            early_stop_rounds=args.early_stop_rounds,
            validation_metrics=winner.get("metrics", {}) or {},
            test_metrics=final_metrics or None,
            selection_metric=args.metric,
            selection_dataset=("walk-forward" if walk_forward else "validation"),
            random_seed=args.seed,
        )
        artifact_path, _ = save_winner(
            trained,
            winner,
            save_dir,
            symbol,
            metric=args.metric,
            final_test_metrics=final_metrics or None,
            sidecar_context=sidecar_context,
        )
        print(f"\nArtefacto guardado: {artifact_path}")
        summary_path = save_summary(summary, save_dir, symbol)
        print(f"Comparativa guardada: {summary_path}")
    else:
        artifact_path = ""
        summary_path = ""

    logger.info(f"Winner: {winner['model']} | {label}={winner_metric:.4f}")
    logger.info("Training completed successfully.")

    db_registered: bool | None = None
    if args.db:
        db_registered = await register_in_db(
            symbol,
            timeframe,
            winner,
            artifact_path,
            args.metric,
            promote=args.promote,
        )
    if db_registered is None:
        db_status = "SKIPPED"
    else:
        db_status = "SUCCESS" if db_registered else "FAILED"

    print("\n========== RESULTADO ==========")
    print("Training:")
    print("  SUCCESS")
    if args.seed is not None:
        print(f"  random seed = {args.seed}")
    print("Preprocessing:")
    print(f"  mode = {args.feature_scaling}")
    if args.feature_scaling == "standard":
        print("  scaler = StandardScaler")
        print("  fit dataset = TRAIN ONLY")
        print("  features = 12")
    print("Selection:")
    if walk_forward:
        print(f"  dataset = WALK-FORWARD VALIDATION ({args.walk_forward_splits} folds)")
        print(f"  metric = mean {label}")
    else:
        print("  dataset = VALIDATION")
        print(f"  metric = {label}")
    print("Final evaluation:")
    print("  dataset = TEST FINAL")
    print("Winner:")
    print(f"  {winner['model']}")
    print(f"  model family = {model_family_label(_winner_family(winner))}")
    print(f"  classification threshold = {args.classification_threshold:.2f}")
    print(f"  {label} = {winner_metric:.4f}")
    if final_metrics and "roc_auc" in final_metrics:
        print(f"  test {args.metric}={final_metrics['roc_auc']:.4f}")
    if not args.no_save:
        print("Artifact:")
        print(f"  {artifact_path}")
        print("  SUCCESS")
        print("Summary:")
        print(f"  {summary_path}")
        print("  SUCCESS")
    print("Database registration:")
    print(f"  {db_status}")
    print(f"DB_REGISTRATION_STATUS={db_status}")

    if args.db and args.promote and db_registered:
        print("DB promotion:")
        print("  SUCCESS")
        print("Previous active model:")
        print("  deactivated")
        print("New model:")
        print("  active")
    elif args.db and not args.promote:
        print("DB promotion:")
        print("  SKIPPED (registered as inactive; use --promote to activate)")


if __name__ == "__main__":
    asyncio.run(main())
