from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config.settings import get_settings  # noqa: E402
from app.core.logger import get_logger  # noqa: E402
from app.ml.training import (  # noqa: E402
    FEATURE_NAMES,
    create_features,
    create_labels,
    format_summary_table,
    load_data,
    run_comparison,
    save_summary,
    save_winner,
    select_winner,
)
from app.ml.training.compare import AVAILABLE_METRICS  # noqa: E402

logger = get_logger("TrainAndCompareCLI")


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
) -> None:
    """Registra el modelo ganador por par en `ml_models` (is_active=True)."""
    from app.database.repositories import MLModelRepository

    model_name = winner["model"]
    metrics = {
        k: v
        for k, v in winner["metrics"].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    repo = MLModelRepository()
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
    logger.info(f"Modelo registrado en DB: {model_name}_{symbol} (activo)")


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
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción de test")
    parser.add_argument(
        "--forward-periods", type=int, default=5, help="Velas hacia adelante para labels"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.001, help="Umbral de retorno para label positivo"
    )
    parser.add_argument(
        "--sequence-length", type=int, default=30, help="Longitud de ventana (modelos secuenciales)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Épocas (modelos secuenciales)")
    parser.add_argument("--min-up-moves", type=int, default=2, help="Número mínimo de forward periods que deben superar el umbral para etiquetar como positivo")
    parser.add_argument("--use-smoten", action="store_true", help="Aplicar SMOTEN para balancear clases en los modelos tabulares")
    parser.add_argument("--walk-forward-splits", type=int, default=5, help="Número de folds para validación walk‑forward")
    parser.add_argument("--early-stop-rounds", type=int, default=20, help="Early‑stopping rounds para modelos de árbol")
    parser.add_argument("--patience", type=int, default=5, help="Patience para early‑stopping en modelos secuenciales")
    parser.add_argument(
        "--db",
        action="store_true",
        help="Registrar el ganador en MLModelRepository (Postgres/SQLite)",
    )
    parser.add_argument("--no-save", action="store_true", help="No persistir el artefacto ganador")
    args = parser.parse_args(argv)

    settings = get_settings()
    save_dir = args.save_dir or settings.ml.model_path
    symbol = args.symbol or derive_symbol(args.data_file)
    timeframe = args.timeframe or derive_timeframe(args.data_file)

    print(f"\nCargando datos: {args.data_file}")
    df = load_data(args.data_file)
    print(f"Candles cargadas: {len(df)} | símbolo: {symbol} | timeframe: {timeframe}")

    print("\nCalculando indicadores y labels...")
    df = create_features(df)
    df["label"] = create_labels(df, forward_periods=args.forward_periods, threshold=args.threshold)
    df = df.dropna(subset=FEATURE_NAMES + ["label"])
    print(f"Muestras tras limpiar NaN: {len(df)}")
    print(
        f"Distribución labels: positivos {int(df['label'].sum())} "
        f"({df['label'].mean():.2%}) / negativos {int((df['label'] == 0).sum())}"
    )

    X = df[FEATURE_NAMES].values
    y = df["label"].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, shuffle=False
    )
    print(f"\nSplit cronológico (sin shuffle): train={len(X_train)} test={len(X_test)}")

    model_names: list[str] = []
    for group in args.model or ["all"]:
        model_names.extend(name.strip() for name in group.split(",") if name.strip())
    print(f"\nEntrenando modelos: {', '.join(model_names)} (métrica objetivo: {args.metric})")
    summary, trained = run_comparison(
        X_train,
        y_train,
        X_test,
        y_test,
        model_names=model_names,
        metric=args.metric,
        feature_names=FEATURE_NAMES,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        settings=settings,
    )

    print("\n" + format_summary_table(summary, args.metric))

    winner = select_winner(summary, args.metric)
    if winner is None:
        print("\nNo hubo modelos exitosos. Revisa los errores arriba.")
        return

    print(
        f"\nGanador: {winner['model']} | {args.metric}=" f"{winner['metrics'].get(args.metric):.4f}"
    )

    if not args.no_save:
        artifact_path, _ = save_winner(trained, winner, save_dir, symbol, metric=args.metric)
        print(f"Artefacto guardado: {artifact_path}")
        summary_path = save_summary(summary, save_dir, symbol)
        print(f"Comparativa guardada: {summary_path}")
    else:
        artifact_path = ""

    if args.db:
        await register_in_db(symbol, timeframe, winner, artifact_path, args.metric)


if __name__ == "__main__":
    asyncio.run(main())
