"""FASE 5 — Walk-Forward Validation (expanding window).

Auditoría reproducible sobre el dataset real. Ejecuta la validación
walk-forward implementada en FASE 5:

- folds expanding cronológicos (sin shuffle, sin futuro, sin overlap).
- antileakage de labels (recorte de forward_periods).
- scaler por fold (fit solo con el train del fold).
- selección del ganador por la MEDIA de la métrica sobre los folds.
- TEST FINAL aislado: se evalúa una sola vez al final.

Por defecto usa ``--model lstm`` (rápido) y ``n_splits=3`` para pruebas.
Este script NO cambia el modelo de producción ni los defaults del pipeline:
la activación es opt-in con ``--walk-forward-splits N>1``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.ml.training.compare import (  # noqa: E402
    evaluate_winner_on_test,
    run_walk_forward_comparison,
    select_walk_forward_winner,
)
from app.ml.training.data import (  # noqa: E402
    FEATURE_NAMES,
    create_features,
    create_labels,
    load_data,
    split_chronological,
)
from app.ml.training.walk_forward import (  # noqa: E402
    build_walk_forward_folds,
    validate_walk_forward_no_future,
)

DATA_FILE = "app/datos_test/USDCAD_H1_201005311000_202606010000.txt"
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
FORWARD_PERIODS = 5
THRESHOLD = 0.001
MIN_UP_MOVES = 2
SEQUENCE_LENGTH = 30
EPOCHS = 1
N_SPLITS = 2
MODEL = "lstm"
METRIC = "roc_auc"


def main() -> None:
    print()
    print("FASE 5 — WALK-FORWARD VALIDATION")
    print("=" * 70)
    print()

    df = load_data(DATA_FILE)
    df = create_features(df)
    df["label"] = create_labels(
        df,
        forward_periods=FORWARD_PERIODS,
        threshold=THRESHOLD,
        min_up_moves=MIN_UP_MOVES,
    )
    df = df.dropna(subset=FEATURE_NAMES + ["label"])

    split = split_chronological(
        df,
        train_size=TRAIN_SIZE,
        validation_size=VALIDATION_SIZE,
        forward_periods=FORWARD_PERIODS,
    )

    print("## Split (FASE 2)")
    print(f"  TRAIN      : {len(split.X_train)} muestras")
    print(f"  VALIDATION : {len(split.X_validation)} muestras")
    print(f"  TEST FINAL : {len(split.X_test)} muestras")
    print(f"  Features   : {len(FEATURE_NAMES)}")
    print()

    # Conjunto de selección para walk-forward = TRAIN + VALIDATION (cronológico).
    X_selection = np.concatenate([split.X_train, split.X_validation], axis=0)
    y_selection = np.concatenate([split.y_train, split.y_validation], axis=0)

    folds = build_walk_forward_folds(
        X_selection,
        y_selection,
        n_splits=N_SPLITS,
        forward_periods=FORWARD_PERIODS,
        seq_context=SEQUENCE_LENGTH,
    )
    validate_walk_forward_no_future(folds)

    print("## Folds (expanding window, sin shuffle/futuro/overlap)")
    header = f"{'Fold':<6} {'TRAIN':>8} {'VALID t0':>9} {'VALID t1':>9}"
    print(header)
    print("-" * len(header))
    for f in folds:
        print(
            f"  {f.fold_index:<4} {len(f.X_train):>8} {f.validation_start:>9} {f.validation_end:>9}"
        )
    print()

    print(f"## Walk-Forward sobre todos los folds (modelo={MODEL}, metric={METRIC})")
    summary, trained = run_walk_forward_comparison(
        X_selection,
        y_selection,
        n_splits=N_SPLITS,
        model_names=[MODEL],
        metric=METRIC,
        feature_names=FEATURE_NAMES,
        sequence_length=SEQUENCE_LENGTH,
        epochs=EPOCHS,
        feature_scaling="none",
    )

    row = summary.iloc[0]
    print(f"  Folds evaluados : {int(row['wf_folds'])}")
    print("  ROC-AUC")
    print(f"    mean = {row['wf_mean_roc_auc']:.4f}")
    print(f"    std  = {row['wf_std_roc_auc']:.4f}")
    print(f"    min  = {row['wf_min_roc_auc']:.4f}")
    print(f"    max  = {row['wf_max_roc_auc']:.4f}")
    print("  PR-AUC")
    print(f"    mean = {row['wf_mean_pr_auc']:.4f}")
    print(f"    std  = {row['wf_std_pr_auc']:.4f}")
    print()

    winner = select_walk_forward_winner(summary, METRIC)
    if winner is None:
        print("  No hubo ganador (revisar errores arriba)")
        return
    print("## Selección")
    print("  dataset : WALK-FORWARD VALIDATION (media de folds)")
    print(f"  metric  : mean {METRIC}")
    print(f"  winner  : {winner['model']}")
    print(f"  mean {METRIC} = {winner['metrics'][f'wf_mean_{METRIC}']:.4f}")
    print()

    final_metrics = evaluate_winner_on_test(
        trained,
        str(winner["model"]),
        X_selection,
        split.X_test,
        split.y_test,
        sequence_length=SEQUENCE_LENGTH,
    )
    print("## FINAL OUT-OF-SAMPLE TEST (aislado)")
    print(f"  model : {winner['model']}")
    for key in ("roc_auc", "pr_auc", "accuracy", "f1"):
        if key in final_metrics:
            print(f"  {key:>10} = {final_metrics[key]:.4f}")
    print()


if __name__ == "__main__":
    main()
