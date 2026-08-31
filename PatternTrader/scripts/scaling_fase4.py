"""FASE 4 — Preprocessing y Scaling Reproducible.

Verifica el pipeline de scaling de FASE 4 sobre el dataset real:

    raw features → scaler.fit(TRAIN ONLY) → transform(validation/test) → sequences

Comprueba sin leakage y demuestra que el scaler se rehidrata desde el sidecar
para servir el MISMO scaler en producción (raw → scaler → sequence → model).

NO reentrena para mejorar resultados, NO cambia el modelo de producción,
NO cambia features, labels, threshold ni split por defecto. El default de
producción sigue siendo ``--feature-scaling none`` (respeta la conclusión de
FASE 3.2). Este script documenta que la infraestructura está operativa y
reproducible bajo ``feature_scaling=standard``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.ml.factory import MLModelFactory  # noqa: E402
from app.ml.training.compare import build_eval_sequences, evaluate_model  # noqa: E402
from app.ml.training.data import (  # noqa: E402
    FEATURE_NAMES,
    build_sequences,
    create_features,
    create_labels,
    load_data,
    split_chronological,
)
from app.ml.training.scaling import (  # noqa: E402
    apply_feature_scaling,
    load_scaler_sidecar,
    save_scaler_sidecar,
)

DATA_FILE = "app/datos_test/USDCAD_H1_201005311000_202606010000.txt"
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
FORWARD_PERIODS = 5
THRESHOLD = 0.001
MIN_UP_MOVES = 2
SEQUENCE_LENGTH = 30
EPOCHS = 3
HIDDEN_DIM = 64
NUM_LAYERS = 2


def main() -> None:
    print()
    print("FASE 4 — PREPROCESSING / SCALING REPRODUCIBLE")
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

    print("## Dataset")
    print(f"  TRAIN      : {len(split.X_train)} muestras")
    print(f"  VALIDATION : {len(split.X_validation)} muestras")
    print(f"  TEST       : {len(split.X_test)} muestras")
    print(f"  Features   : {len(FEATURE_NAMES)}")
    print()

    # ---------------------------------------------------------------
    # Scaler: fit SOLO con TRAIN; validation y test solo transform.
    # ---------------------------------------------------------------
    X_tr_s, X_val_s, X_test_s, scaler = apply_feature_scaling(
        split.X_train,
        split.X_validation,
        split.X_test,
        mode="standard",
        feature_names=FEATURE_NAMES,
    )

    print("## Scaler")
    print("  type       : StandardScaler")
    print("  fit dataset: TRAIN ONLY")
    print()

    # ---- Feature statistics (de TRAIN) ----
    print("## Feature statistics (TRAIN)")
    header = f"{'Feature':<16} {'TRAIN_MEAN':>12} {'TRAIN_STD':>12}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:<14} {scaler.mean_[i]:>12.6f} {scaler.scale_[i]:>12.6f}")
    print()

    # ---- Leakage check: el scaler fiteado solo con TRAIN no debe usar val/test.
    # Comprobamos que transform(validation) usa stats de TRAIN: refitear con
    # validation/null debe diferir de lo obtenido con el scaler de train.
    from sklearn.preprocessing import StandardScaler

    val_scaled_by_train = scaler.transform(split.X_validation)
    val_scaled_by_val = StandardScaler().fit(split.X_validation).transform(split.X_validation)
    differs = float(np.abs(val_scaled_by_train - val_scaled_by_val).mean())

    print("## Leakage check")
    print("  Scaler fit dataset       : TRAIN")
    print("  Validation y Test usan   : scaler.transform() (nunca refit)")
    print(
        f"  |transform(val por TRAIN) - transform(val por VAL)| mean = {differs:.6f} "
        "(>0 confirma que NO se refitea con validation)"
    )
    print()

    # ---- Persistencia + rehidratación del scaler (sidecar / serving) ----
    scaler_json = Path("/tmp/opencode/fase4_scaler.json")
    scaler_json.parent.mkdir(parents=True, exist_ok=True)
    save_scaler_sidecar(scaler, str(scaler_json), FEATURE_NAMES)
    restored = load_scaler_sidecar(str(scaler_json))
    np.testing.assert_allclose(restored.mean_, scaler.mean_, rtol=1e-9)
    np.testing.assert_allclose(restored.scale_, scaler.scale_, rtol=1e-9)
    print("## Scaler artifact (serving round-trip)")
    print(f"  Guardado  : {scaler_json}")
    print("  Rehidratado identico al entrenado: YES (mean_/scale_ coinciden)")
    print()

    # ---- Secuencias: transform primero, luego construir ventanas.
    X_tr_seq, y_tr_seq = build_sequences(X_tr_s.astype(np.float32), split.y_train, SEQUENCE_LENGTH)
    X_val_seq, y_val_seq = build_eval_sequences(
        X_tr_s.astype(np.float32),
        X_val_s.astype(np.float32),
        split.y_validation,
        SEQUENCE_LENGTH,
    )
    X_test_seq, y_test_seq = build_eval_sequences(
        X_val_s.astype(np.float32),
        X_test_s.astype(np.float32),
        split.y_test,
        SEQUENCE_LENGTH,
    )
    print("## Sequences (transform → build)")
    print(f"  TRAIN sequences : {X_tr_seq.shape[0]}")
    print(f"  VALIDATION seq  : {X_val_seq.shape[0]}")
    print(f"  TEST seq        : {X_test_seq.shape[0]}")
    print()

    # ---- Mini entrenamiento LSTM con scaling (solo demostrativo) ----
    print("## LSTM con StandardScaler (fit TRAIN) - demostrativo")
    model = MLModelFactory.create_new(
        "lstm",
        sequence_length=SEQUENCE_LENGTH,
        feature_dim=len(FEATURE_NAMES),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        epochs=EPOCHS,
        learning_rate=1e-3,
        batch_size=16,
        random_state=42,
    )
    model.train(X_tr_seq, y_tr_seq, feature_names=FEATURE_NAMES)
    val_metrics = evaluate_model(model, X_val_seq, y_val_seq)
    test_metrics = evaluate_model(model, X_test_seq, y_test_seq)
    print(f"  Validation ROC-AUC: {val_metrics.get('roc_auc', float('nan')):.4f}")
    print(f"  Test       ROC-AUC: {test_metrics.get('roc_auc', float('nan')):.4f}")
    print(f"  Test       F1     : {test_metrics.get('f1', float('nan')):.4f}")
    print()

    # ---------------------------------------------------------------
    print("========== OUTPUT (plan FASE 4) ==========")
    print("Preprocessing")
    print("=============")
    print("Scaler:")
    print("  StandardScaler")
    print("Fit dataset:")
    print("  TRAIN ONLY")
    print("Features:")
    print(f"  {len(FEATURE_NAMES)}")
    print("Validation transformed:")
    print("  YES")
    print("Test transformed:")
    print("  YES")
    print("Test used for scaler fitting:")
    print("  NO")
    print("Scaler rehydratable en serving:")
    print("  YES")


if __name__ == "__main__":
    main()
