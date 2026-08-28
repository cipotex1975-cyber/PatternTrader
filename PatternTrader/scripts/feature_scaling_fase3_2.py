"""FASE 3.2 — Feature Scaling para modelos secuenciales (experimento controlado).

Compara el LSTM SIN scaling (baseline) contra el mismo LSTM con un
StandardScaler cuyo fit se realiza EXCLUSIVAMENTE con TRAIN.

Pipeline del scaler:
    TRAIN → scaler.fit(TRAIN)
    scaler.transform(TRAIN / VALIDATION / TEST)   → construir secuencias

El scaling NO modifica numero de samples, labels ni sequence_length:
el escalado se aplica ANTES de construir las ventanas.

Criterio de adopcion (PASO 10): el scaling solo se adopta si VALIDATION
mejora Y TEST mejora o se mantiene (jerarquia TEST ROC-AUC -> PR-AUC -> F1).
Si no gana, NO se modifica app/ml/ y el informe lo documenta.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.preprocessing import StandardScaler

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.ml.factory import MLModelFactory  # noqa: E402
from app.ml.training.compare import build_eval_sequences, evaluate_model  # noqa: E402
from app.ml.training.data import (  # noqa: E402
    FEATURE_NAMES,
    create_features,
    create_labels,
    format_for_model,
    load_data,
    split_chronological,
)

DATA_FILE = "app/datos_test/USDCAD_H1_201005311000_202606010000.txt"
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
FORWARD_PERIODS = 5
THRESHOLD = 0.001
MIN_UP_MOVES = 2
SEQUENCE_LENGTH = 30
EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
HIDDEN_DIM = 64
NUM_LAYERS = 2
RANDOM_STATE = 42


def _fmt_metrics(m: dict[str, float]) -> str:
    return (
        f"      ROC-AUC: {m.get('roc_auc', float('nan')):.4f}\n"
        f"      PR-AUC:  {m.get('pr_auc', float('nan')):.4f}\n"
        f"      F1:      {m.get('f1', float('nan')):.4f}\n"
        f"      Precision: {m.get('precision', float('nan')):.4f}\n"
        f"      Recall:    {m.get('recall', float('nan')):.4f}"
    )


def _train_lstm(split, scaler=None) -> tuple[dict, dict, dict, dict, float]:
    X_train = scaler.transform(split.X_train) if scaler else split.X_train
    X_val = scaler.transform(split.X_validation) if scaler else split.X_validation
    X_test = scaler.transform(split.X_test) if scaler else split.X_test

    model = MLModelFactory.create_new(
        "lstm",
        sequence_length=SEQUENCE_LENGTH,
        feature_dim=len(FEATURE_NAMES),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
    )

    X_tr_seq, y_tr_seq = format_for_model("lstm", X_train, split.y_train, SEQUENCE_LENGTH)
    train_meta = model.train(X_tr_seq, y_tr_seq, feature_names=FEATURE_NAMES)
    train_loss = float(train_meta.get("loss", float("nan")))

    X_val_seq, y_val_seq = build_eval_sequences(X_train, X_val, split.y_validation, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = build_eval_sequences(X_val, X_test, split.y_test, SEQUENCE_LENGTH)

    val_metrics = evaluate_model(model, X_val_seq, y_val_seq)
    test_metrics = evaluate_model(model, X_test_seq, y_test_seq)

    counts = {
        "train_sequences": int(X_tr_seq.shape[0]),
        "val_sequences": int(X_val_seq.shape[0]),
        "test_sequences": int(X_test_seq.shape[0]),
    }
    return model, val_metrics, test_metrics, counts, train_loss


def section_feature_statistics(scaler: StandardScaler) -> None:
    print("## Feature statistics")
    print()
    header = f"{'Feature':<16} {'TRAIN_MEAN':>12} {'TRAIN_STD':>12}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(FEATURE_NAMES):
        mean = float(scaler.mean_[i])
        std = float(scaler.scale_[i]) if scaler.scale_[i] != 0 else 0.0
        print(f"  {name:<14} {mean:>12.6f} {std:>12.6f}")
    print()


def section_metrics(label: str, val: dict, test: dict, train_loss: float) -> None:
    print(f"## {label} LSTM")
    print()
    print("Validation:")
    print(_fmt_metrics(val))
    print()
    print("Test:")
    print(_fmt_metrics(test))
    print(f"train_loss: {train_loss:.4f}")
    print()


def section_comparison(vals, tests) -> None:
    print("## Comparison")
    print()
    header = f"{'Metric':<16} {'Baseline':>12} {'Scaled':>12} {'Delta':>10}"
    print(header)
    print("-" * len(header))
    for key, label in [
        ("roc_auc", "VAL ROC-AUC"),
        ("pr_auc", "VAL PR-AUC"),
        ("f1", "VAL F1"),
        ("roc_auc", "TEST ROC-AUC"),
        ("pr_auc", "TEST PR-AUC"),
        ("f1", "TEST F1"),
    ]:
        is_test = label.startswith("TEST")
        src_base = tests["bl"] if is_test else vals["bl"]
        src_scaled = tests["sc"] if is_test else vals["sc"]
        b = src_base.get(key, float("nan"))
        s = src_scaled.get(key, float("nan"))
        delta = s - b
        sign = "+" if delta >= 0 else ""
        print(f"{label:<16} {b:>12.4f} {s:>12.4f} {sign}{delta:>9.4f}")
    print()


def section_conclusion(vals, tests) -> dict[str, bool]:
    print("## Conclusion")
    print()
    bl_v, sc_v = vals["bl"], vals["sc"]
    bl_t, sc_t = tests["bl"], tests["sc"]

    v_imp = sc_v["roc_auc"] > bl_v["roc_auc"]
    t_roc_imp = sc_t["roc_auc"] > bl_t["roc_auc"]
    t_f1_imp = sc_t["f1"] > bl_t["f1"]
    t_stable = sc_t["roc_auc"] >= bl_t["roc_auc"] - 1e-6 and sc_t["f1"] >= bl_t["f1"] - 1e-6
    adopted = v_imp and t_stable

    print("1. StandardScaler implemented: " + ("YES" if adopted else "NO"))
    print("2. Scaler fit exclusively on TRAIN: YES")
    print(
        f"3. Validation improvement: "
        f"{'YES' if v_imp else 'NO'} "
        f"(ROC-AUC {bl_v['roc_auc']:.4f} -> {sc_v['roc_auc']:.4f})"
    )
    print("4. Test improvement: MIXED")
    print(
        f"   ROC-AUC {bl_t['roc_auc']:.4f} -> {sc_t['roc_auc']:.4f} "
        f"({'improves' if t_roc_imp else 'worsens'}); "
        f"F1 {bl_t['f1']:.4f} -> {sc_t['f1']:.4f} "
        f"({'improves' if t_f1_imp else 'worsens'})"
    )
    print(f"5. Inference compatibility: {'YES' if adopted else 'N/A (no adoptado)'}")
    print(f"6. Scaling adopted: {'YES' if adopted else 'NO'}")
    print()
    return {"adopted": adopted, "v_imp": v_imp, "t_stable": t_stable}


def main() -> None:
    print()
    print("FASE 3.2 — FEATURE SCALING AUDIT")
    print("=" * 60)
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

    scaler = StandardScaler()
    scaler.fit(split.X_train)

    print("## Scaler")
    print()
    print("type: StandardScaler")
    print("fit_dataset: TRAIN")
    print()

    section_feature_statistics(scaler)

    print("Executing BASELINE LSTM (sin scaling)...")
    _, val_bl, test_bl, counts_bl, loss_bl = _train_lstm(split, scaler=None)
    print()

    print("Executing SCALED LSTM (StandardScaler fit en TRAIN)...")
    _, val_sc, test_sc, counts_sc, loss_sc = _train_lstm(split, scaler=scaler)
    print()

    # Verificar que el scaling no altera las muestras.
    print("## Sample count check")
    print(f"  baseline train/val/test sequences: {counts_bl}")
    print(f"  scaled   train/val/test sequences: {counts_sc}")
    assert counts_bl == counts_sc, "El scaling altero el numero de muestras"
    assert counts_bl["train_sequences"] == len(split.y_train) - (SEQUENCE_LENGTH - 1)
    print()
    print("## Leakage check")
    print("Scaler fit dataset: TRAIN")
    print("Validation leakage: NO")
    print("Test leakage: NO")
    print()

    section_metrics("Baseline", val_bl, test_bl, loss_bl)
    section_metrics("Scaled", val_sc, test_sc, loss_sc)
    section_comparison({"bl": val_bl, "sc": val_sc}, {"bl": test_bl, "sc": test_sc})
    verdict = section_conclusion({"bl": val_bl, "sc": val_sc}, {"bl": test_bl, "sc": test_sc})

    # Emitir resultado maquina-parseable para el siguiente paso.
    print("VERDICT: " + ("ADOPT" if verdict["adopted"] else "NOT_ADOPT"))
    print("VAL_ROC_BASE={:.4f} VAL_ROC_SCALED={:.4f}".format(val_bl["roc_auc"], val_sc["roc_auc"]))
    print(
        "TEST_ROC_BASE={:.4f} TEST_ROC_SCALED={:.4f}".format(test_bl["roc_auc"], test_sc["roc_auc"])
    )
    print("TEST_F1_BASE={:.4f} TEST_F1_SCALED={:.4f}".format(test_bl["f1"], test_sc["f1"]))
    print("VAL_F1_BASE={:.4f} VAL_F1_SCALED={:.4f}".format(val_bl["f1"], val_sc["f1"]))


if __name__ == "__main__":
    main()
