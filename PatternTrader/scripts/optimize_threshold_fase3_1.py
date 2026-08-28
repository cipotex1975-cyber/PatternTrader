"""FASE 3.1 — Optimización del threshold de clasificación.

Reutiliza las mismas funciones del pipeline real (create_features,
create_labels, split_chronological, build_eval_sequences, MLModelFactory).

Principio: el threshold es un hiperparámetro de DECISIÓN. Se selecciona
EXCLUSIVAMENTE con VALIDATION y se aplica UNA sola vez sobre TEST FINAL.
TEST nunca participa en la selección.

Esta fase NO cambia features, arquitectura, epochs, sequence_length,
optimizer, learning rate, labels, forward_periods ni el split temporal.
ROC-AUC y PR-AUC siguen calculándose con probabilidades (independientes
del threshold).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.ml.factory import MLModelFactory  # noqa: E402
from app.ml.training.compare import (  # noqa: E402
    build_eval_sequences,
    metrics_at_threshold,
)
from app.ml.training.data import (  # noqa: E402
    FEATURE_NAMES,
    create_features,
    create_labels,
    format_for_model,
    load_data,
    split_chronological,
)

# ── Parámetros idénticos a train_and_compare.py (CLI defaults) ──────────────
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

# Búsqueda de threshold sobre VALIDATION.
SWEEP_START = 0.20
SWEEP_STOP = 0.801
SWEEP_STEP = 0.01

# Sidecar donde se persiste el threshold seleccionado (sin sobrescribir).
MODEL_NAME = "lstm"
SYMBOL = "USDCAD"
META_PATH = Path("models") / f"{MODEL_NAME}_{SYMBOL}.meta.json"


def _delta(val: float, ref: float) -> str:
    d = val - ref
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def section_1_code_inspection() -> None:
    print("=" * 60)
    print("PASO 1 — Current pipeline inspection")
    print("=" * 60)
    print()
    print("Current probability source:")
    print("  predict_proba() → softmax(dim=1); P(class_1) = [:, 1]")
    print("  (app/ml/models/sequence_base.py)")
    print()
    print("Current classification method:")
    print("  predict() → probs.argmax(axis=1)  (two-class softmax)")
    print()
    print("Current threshold: 0.50 (argmax equivale a threshold 0.50)")
    print()
    print("Current metric calculation path:")
    print("  sequence_base.evaluate / compare.evaluate_model:")
    print("    accuracy/precision/recall/f1 → predict() (clases binarias)")
    print("    roc_auc  → roc_auc_score(y_true, probabilities)")
    print("    pr_auc   → average_precision_score(y_true, probabilities)")
    print()
    print("  La conversión clase se realiza en:")
    print("    - sequence_base.predict()  (argmax)")
    print("    - sequence_base.evaluate() (usa predict)")
    print("    - compare.evaluate_model() (usa model.predict)")
    print()


def section_2_baseline(val_probs, val_y, test_probs, test_y) -> None:
    print("=" * 60)
    print("BASELINE  (threshold = 0.50)")
    print("=" * 60)
    print()
    print("classification_method: argmax")
    print("baseline_threshold: 0.50")
    print()

    val_bl = metrics_at_threshold(val_y, val_probs, 0.50)
    test_bl = metrics_at_threshold(test_y, test_probs, 0.50)

    print("VALIDATION:")
    print(f"  ROC-AUC: {val_bl['roc_auc']:.4f}")
    print(f"  PR-AUC:  {val_bl['pr_auc']:.4f}")
    print(f"  F1:      {val_bl['f1']:.4f}")
    print(f"  Precision: {val_bl['precision']:.4f}")
    print(f"  Recall:    {val_bl['recall']:.4f}")
    print()
    print("TEST FINAL:")
    print(f"  ROC-AUC: {test_bl['roc_auc']:.4f}")
    print(f"  PR-AUC:  {test_bl['pr_auc']:.4f}")
    print(f"  F1:      {test_bl['f1']:.4f}")
    print(f"  Precision: {test_bl['precision']:.4f}")
    print(f"  Recall:    {test_bl['recall']:.4f}")
    print()
    return val_bl, test_bl


def section_4_sweep(val_y, val_probs) -> dict:
    print("=" * 60)
    print("PASO 4/6 — Threshold sweep on VALIDATION (MAX F1)")
    print("=" * 60)
    print()
    thresholds = np.arange(SWEEP_START, SWEEP_STOP, SWEEP_STEP)
    rows = []
    for t in thresholds:
        m = metrics_at_threshold(val_y, val_probs, float(t))
        rows.append(
            {
                "threshold": round(float(t), 2),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "predicted_positive_rate": m["predicted_positive_rate"],
            }
        )

    header = (
        f"{'Threshold':<10} {'Precision':>10} {'Recall':>8} " f"{'F1':>8} {'Pred_Pos_Rate':>15}"
    )
    print(header)
    print("-" * len(header))

    shown = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
    for row in rows:
        if row["threshold"] in shown:
            print(
                f"{row['threshold']:<10.2f} {row['precision']:>10.4f} "
                f"{row['recall']:>8.4f} {row['f1']:>8.4f} "
                f"{row['predicted_positive_rate']*100:>14.2f}%"
            )
    print()

    best = max(rows, key=lambda r: r["f1"])
    print(f"Best validation threshold: {best['threshold']:.2f}")
    print("  metric: F1")
    print(f"  validation F1: {best['f1']:.4f}")
    print(f"  precision: {best['precision']:.4f}")
    print(f"  recall: {best['recall']:.4f}")
    print(f"  predicted positive rate: {best['predicted_positive_rate']*100:.2f}%")
    print()

    if best["predicted_positive_rate"] < 0.005:
        print(
            "WARNING: el mejor threshold produce una tasa de positivos "
            "predichos extremadamente baja (< 0.5%). Reportando el valor real "
            "en lugar de ocultarlo."
        )
        print()
    return best


def section_8_apply_test(test_y, test_probs, best, test_bl) -> None:
    print("=" * 60)
    print("PASO 8 — Apply optimized threshold to TEST FINAL (una sola vez)")
    print("=" * 60)
    print()
    t = best["threshold"]
    test_opt = metrics_at_threshold(test_y, test_probs, t)

    print("Baseline threshold = 0.50")
    print(f"  F1: {test_bl['f1']:.4f}")
    print(f"  Precision: {test_bl['precision']:.4f}")
    print(f"  Recall: {test_bl['recall']:.4f}")
    print()
    print(f"Optimized threshold = {t:.2f}")
    print(f"  F1: {test_opt['f1']:.4f}")
    print(f"  Precision: {test_opt['precision']:.4f}")
    print(f"  Recall: {test_opt['recall']:.4f}")
    print()
    return test_opt


def section_9_final_comparison(
    val_y, val_probs, test_y, test_probs, best, val_bl, test_bl, test_opt
) -> None:
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print()
    t = best["threshold"]

    # VALIDATION: baseline vs mejor threshold
    val_opt = metrics_at_threshold(val_y, val_probs, t)

    print("FINAL TEST")
    print()
    header = f"{'Metric':<10} {'thr=0.50':>12} {'optimized':>12} {'Delta':>10}"
    print(header)
    print("-" * len(header))
    for key, label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
    ]:
        base = test_bl[key]
        opt = test_opt[key]
        print(f"{label:<10} {base:>12.4f} {opt:>12.4f} {_delta(opt, base):>10}")
    print()

    # Robustez: mejora validada en VALIDATION que sobrevive en TEST.
    val_f1_delta = val_opt["f1"] - val_bl["f1"]
    test_f1_delta = test_opt["f1"] - test_bl["f1"]
    roc_change = abs(test_opt["roc_auc"] - test_bl["roc_auc"])
    pr_change = abs(test_opt["pr_auc"] - test_bl["pr_auc"])

    print("Conclusion")
    print("----------")
    print(f"1. Threshold optimization on VALIDATION: " f"{'SUCCESS' if val_f1_delta > 0 else 'NO'}")
    print(f"2. Validation F1 improvement: {val_f1_delta:+.4f}")
    print(f"3. Test F1 improvement: {test_f1_delta:+.4f}")
    print(f"4. Precision impact: {_delta(test_opt['precision'], test_bl['precision'])}")
    print(f"5. Recall impact: {_delta(test_opt['recall'], test_bl['recall'])}")
    print(
        f"6. ROC-AUC changed: {'YES' if roc_change > 1e-6 else 'NO'}"
        f" (delta={_delta(test_opt['roc_auc'], test_bl['roc_auc'])})"
    )
    print(
        f"7. PR-AUC changed: {'YES' if pr_change > 1e-6 else 'NO'}"
        f" (delta={_delta(test_opt['pr_auc'], test_bl['pr_auc'])})"
    )
    print(
        f"8. Threshold appears robust: "
        f"{'YES' if val_f1_delta > 0 and test_f1_delta >= 0 else 'NO'}"
    )
    print()


def section_10_persist(best: dict) -> None:
    print("=" * 60)
    print("PASO 10 — Persistence (decision_threshold en meta.json)")
    print("=" * 60)
    print()
    import json

    if not META_PATH.exists():
        print(f"  META_PATH no existe ({META_PATH}); se omite persistencias.")
        print()
        return

    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    t = float(best["threshold"])
    meta["decision_threshold"] = t
    meta["threshold_selection"] = {
        "dataset": "validation",
        "metric": "f1",
        "range": [SWEEP_START, 0.80],
        "step": SWEEP_STEP,
    }

    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"  {META_PATH}")
    print(f"  decision_threshold: {t}")
    print(f"  threshold_selection: {meta['threshold_selection']}")
    print("  Información existente preservada (solo se añadieron claves).")
    print()


def main() -> None:
    print()
    print("FASE 3.1 — THRESHOLD OPTIMIZATION")
    print("=" * 60)
    print()

    section_1_code_inspection()

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

    X_tr_seq, y_tr_seq = format_for_model("lstm", split.X_train, split.y_train, SEQUENCE_LENGTH)
    model.train(X_tr_seq, y_tr_seq, feature_names=FEATURE_NAMES)

    X_val_seq, y_val_seq = build_eval_sequences(
        split.X_train, split.X_validation, split.y_validation, SEQUENCE_LENGTH
    )
    X_test_seq, y_test_seq = build_eval_sequences(
        split.X_validation, split.X_test, split.y_test, SEQUENCE_LENGTH
    )

    val_probs = model.predict_proba(X_val_seq)[:, 1]
    test_probs = model.predict_proba(X_test_seq)[:, 1]

    val_bl, test_bl = section_2_baseline(val_probs, y_val_seq, test_probs, y_test_seq)
    best = section_4_sweep(y_val_seq, val_probs)
    test_opt = section_8_apply_test(y_test_seq, test_probs, best, test_bl)
    section_9_final_comparison(
        y_val_seq, val_probs, y_test_seq, test_probs, best, val_bl, test_bl, test_opt
    )
    section_10_persist(best)


if __name__ == "__main__":
    main()
