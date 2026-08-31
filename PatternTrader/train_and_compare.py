from __future__ import annotations

import argparse
import asyncio
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config.settings import get_settings  # noqa: E402
from app.core.logger import get_logger  # noqa: E402
from app.ml.training import (  # noqa: E402
    FEATURE_NAMES,
    build_walk_forward_folds,
    create_features,
    create_labels,
    evaluate_winner_on_test,
    format_summary_table,
    load_data,
    run_comparison,
    run_walk_forward_comparison,
    save_summary,
    save_winner,
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
) -> bool:
    """Registra el modelo ganador por par en `ml_models` (is_active=True).

    FASE 1.1: un fallo de conexión/SQLAlchemy no propaga la excepción ni
    invalida el entrenamiento; se registra el error y devuelve False.
    """
    from sqlalchemy.exc import SQLAlchemyError

    from app.database.repositories import MLModelRepository

    model_name = winner["model"]
    metrics = {
        k: v
        for k, v in winner["metrics"].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
    }
    repo = MLModelRepository()
    try:
        await repo.deactivate_by_symbol(symbol)
        await repo.upsert(
            name=f"{model_name}_{symbol}",
            model_type="classification",
            version=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            path=artifact_path,
            metrics=metrics,
            is_active=True,
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
    logger.info(f"Modelo registrado en DB: {model_name}_{symbol} (activo)")
    return True


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
        help="Early‑stopping rounds para modelos de árbol (RESERVADO: aún no afecta)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience para early‑stopping en modelos secuenciales (RESERVADO: aún no afecta)",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Registrar el ganador en MLModelRepository (Postgres/SQLite)",
    )
    parser.add_argument("--no-save", action="store_true", help="No persistir el artefacto ganador")
    args = parser.parse_args(argv)

    try:
        validate_split_sizes(args.train_size, args.validation_size, args.test_size)
    except ValueError as exc:
        parser.error(str(exc))

    settings = get_settings()
    save_dir = args.save_dir or settings.ml.model_path
    symbol = args.symbol or derive_symbol(args.data_file)
    timeframe = args.timeframe or derive_timeframe(args.data_file)

    if args.walk_forward_splits > 1:
        logger.info(
            f"Modo walk-forward activo: {args.walk_forward_splits} splits "
            "(selección por media de folds; TEST FINAL aislado)"
        )
    if args.early_stop_rounds != 20:
        logger.warning(
            "--early-stop-rounds todavía no tiene efecto (early-stopping de árboles pendiente)"
        )
    if args.patience != 5:
        logger.warning("--patience todavía no tiene efecto (early-stopping secuencial pendiente)")

    print(f"\nCargando datos: {args.data_file}")
    df = load_data(args.data_file)
    print(f"Candles cargadas: {len(df)} | símbolo: {symbol} | timeframe: {timeframe}")

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
        )
        winner = select_winner(summary, args.metric)
        winner_metric = winner["metrics"].get(args.metric) if winner else None
        eval_context = split.X_validation

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
        if winner_metric is not None:
            print(f"  mean {label} (folds) = {winner_metric:.4f}")
    else:
        print("\nModel selection")
        print("===============")
        print("\nSelection dataset : VALIDATION")
        print(f"Selection metric  : {label}")
        print("\nWinner:")
        print(f"  {winner['model']}")
        if winner_metric is not None:
            print(f"  validation {label} = {winner_metric:.4f}")

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
        )
    except ValueError as exc:
        logger.warning(f"FINAL OUT-OF-SAMPLE TEST no disponible: {exc}")

    print("\nFINAL OUT-OF-SAMPLE TEST")
    print("========================")
    print(f"\nModel: {winner['model']}")
    if final_metrics:
        for key in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"):
            if key in final_metrics:
                print(f"{metric_label(key)}: {final_metrics[key]:.4f}")
    else:
        print("  SKIPPED (dataset de test insuficiente para este modelo)")

    if not args.no_save:
        artifact_path, _ = save_winner(
            trained,
            winner,
            save_dir,
            symbol,
            metric=args.metric,
            final_test_metrics=final_metrics or None,
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
        db_registered = await register_in_db(symbol, timeframe, winner, artifact_path, args.metric)
    if db_registered is None:
        db_status = "SKIPPED"
    else:
        db_status = "SUCCESS" if db_registered else "FAILED"

    print("\n========== RESULTADO ==========")
    print("Training:")
    print("  SUCCESS")
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


if __name__ == "__main__":
    asyncio.run(main())
