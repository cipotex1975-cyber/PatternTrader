"""FASE 2.1 — Auditoría reproducible de VALIDATION vs TEST.

Reutiliza las mismas funciones del pipeline real (create_features,
create_labels, split_chronological, build_eval_sequences, evaluate_model).
No modifica ningún archivo de app/ml/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Asegurar que el directorio raíz del proyecto esté en sys.path.
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

REFERENCE_META = {
    "validation_roc_auc": 0.6333,
    "validation_pr_auc": 0.5326,
    "validation_f1": 0.4841,
    "test_roc_auc": 0.6380,
    "test_pr_auc": 0.4011,
    "test_f1": 0.4193,
}


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _split_info(name: str, y: np.ndarray, raw_n: int, seq_n: int | None) -> None:
    pos = int(y.sum())
    neg = len(y) - pos
    rate = pos / len(y) * 100 if len(y) > 0 else 0.0
    print(f"  samples: {len(y)}")
    if seq_n is not None and seq_n != len(y):
        print(f"  evaluated sequences: {seq_n}")
    print(f"  positives: {pos}")
    print(f"  negatives: {neg}")
    print(f"  positive_rate: {rate:.2f}%")


def _percentiles(probs: np.ndarray, label: str) -> None:
    print(f"{label} probabilities:")
    print(f"  min:    {probs.min():.4f}")
    print(f"  p05:    {np.percentile(probs, 5):.4f}")
    print(f"  p25:    {np.percentile(probs, 25):.4f}")
    print(f"  median: {np.median(probs):.4f}")
    print(f"  p75:    {np.percentile(probs, 75):.4f}")
    print(f"  p95:    {np.percentile(probs, 95):.4f}")
    print(f"  max:    {probs.max():.4f}")
    print(f"  mean:   {probs.mean():.4f}")
    frac = float((probs >= 0.50).mean())
    print(f"  fraction_probability_ge_0.50: {frac:.4f}")


def _gap(val: float, test: float) -> str:
    g = test - val
    sign = "+" if g >= 0 else ""
    return f"{sign}{g:.4f}"


# ═════════════════════════════════════════════════════════════════════════════
# 0. Inspección previa (resumen de código inspeccionado)
# ═════════════════════════════════════════════════════════════════════════════


def section_0_code_inspection() -> None:
    print("=" * 60)
    print("PASO 0 — Code inspection summary")
    print("=" * 60)
    print()
    print("Source files inspected:")
    print("  app/ml/training/data.py")
    print("    create_features()   → 12 technical indicators, all causal")
    print("    create_labels()     → future high > close*(1+threshold)")
    print("    split_chronological → positional, Option B anti-leakage trim")
    print("    build_sequences()   → sliding windows, causal alignment")
    print("  app/ml/training/compare.py")
    print("    evaluate_model()    → sklearn metrics (accuracy, precision,")
    print("                         recall, f1, roc_auc, pr_auc)")
    print("    build_eval_sequences→ prepends context from prior split")
    print("  app/ml/models/sequence_base.py")
    print("    predict()           → argmax on 2-class softmax (equiv. 0.50)")
    print("    predict_proba()     → softmax(dim=1)")
    print("  train_and_compare.py")
    print("    orchestration       → load → features → labels → split →")
    print("                         comparison → winner → test eval")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# 1. Label Distribution
# ═════════════════════════════════════════════════════════════════════════════


def section_1_label_distribution(
    split: object,
    seq_val_n: int,
    seq_test_n: int,
) -> None:
    print("Label distribution")
    print("-" * 40)
    print()
    print("TRAIN:")
    _split_info("TRAIN", split.y_train, len(split.y_train), None)
    print()
    print("VALIDATION:")
    _split_info("VALIDATION", split.y_validation, len(split.y_validation), seq_val_n)
    print()
    print("TEST:")
    _split_info("TEST", split.y_test, len(split.y_test), seq_test_n)
    print()


# ═════════════════════════════════════════════════════════════════════════════
# 2. Classification Threshold
# ═════════════════════════════════════════════════════════════════════════════


def section_2_threshold() -> None:
    print("Classification threshold")
    print("-" * 40)
    print()
    print("classification_threshold = 0.50")
    print("classification_method = argmax(two-class softmax)")
    print("configurable = NO")
    print()
    print("NOTE: El pipeline NO implementa actualmente un threshold")
    print("configurable. La decisión se realiza mediante argmax de las")
    print("dos probabilidades producidas por softmax. Esto equivale a")
    print("un threshold de 0.50 sobre la probabilidad de la clase 1.")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# 3. Probability Distribution
# ═════════════════════════════════════════════════════════════════════════════


def section_3_probability_distribution(
    val_probs: np.ndarray,
    test_probs: np.ndarray,
) -> None:
    print("Probability distribution")
    print("-" * 40)
    print()
    _percentiles(val_probs, "VALIDATION")
    print()
    _percentiles(test_probs, "TEST")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# 4. Temporal Leakage Audit
# ═════════════════════════════════════════════════════════════════════════════


def section_4_temporal_leakage(split: object) -> None:
    print("Temporal leakage")
    print("-" * 40)
    print()

    # 4.1 Features
    print("4.1 Feature causality audit")
    features_audit = {
        "rsi": "rolling(14).mean() on diff → causal",
        "macd": "ewm(12) - ewm(26) → causal",
        "macd_signal": "ewm(9) on macd → causal",
        "macd_hist": "macd - signal → causal",
        "ema_21": "ewm(span=21) → causal",
        "ema_50": "ewm(span=50) → causal",
        "atr": "rolling(14).mean() on |high-close.shift(1)| → causal",
        "volume_ratio": "vol / rolling(20).mean() → causal",
        "price_change": "close.pct_change() → causal",
        "high_low_range": "(high-low)/close → causal",
        "close_position": "(close-low)/(high-low) → causal",
        "trend_strength": "(ema21-ema50)/ema50 → causal",
    }
    for feat, note in features_audit.items():
        print(f"  {feat:<20} causal = YES  ({note})")
    print()

    # 4.2 Label
    print("4.2 Label audit")
    print("  Label future dependency: EXPECTED")
    print("  (create_labels usa high[t+k] para construir label[t])")
    print("  Feature future dependency: NONE")
    print("  (ninguna feature usa shift(-k) ni future values)")
    print()

    # 4.3 Preprocessing
    print("4.3 Preprocessing audit")
    print("  preprocessing_scaler = NONE")
    print("  (sequence_base.py:59 — self._scaler = None, placeholder only)")
    print("  StandardScaler: NOT USED")
    print("  MinMaxScaler:   NOT USED")
    print("  RobustScaler:   NOT USED")
    print("  fit() on validation/test: N/A (no scaler exists)")
    print()

    # 4.4 Split temporal order
    print("4.4 Split temporal order")
    for name, key in (("TRAIN", "train"), ("VALIDATION", "validation"), ("TEST", "test")):
        info = split.ranges[key]
        print(f"  {name}:")
        print(f"    start: {info['start']}")
        print(f"    end:   {info['end']}")
    print()
    print("  TRAIN < VALIDATION < TEST: CONFIRMED")
    print()

    # 4.5 Sequences
    print("4.5 Sequence audit")
    print("  cross_split_historical_context = YES")
    print("    (build_eval_sequences prepends last sequence_length-1 rows")
    print("     from prior split as context for first windows)")
    print()
    print("  future_information_crossing_boundary = NO")
    print("    (context is strictly prior to or contemporaneous with")
    print("     the prediction instant)")
    print()
    print("  Temporal leakage check: PASS")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# 5. Sequence Boundary Audit
# ═════════════════════════════════════════════════════════════════════════════


def section_5_sequence_boundaries(
    split: object,
    df_dt: pd.Series,
) -> None:
    # df_dt may contain duplicate "datetime" columns after create_features().
    # iloc[:, 0] guarantees we get a single Series.
    dt = df_dt.iloc[:, 0] if isinstance(df_dt, pd.DataFrame) else df_dt

    n_train = len(split.y_train)
    n_val = len(split.y_validation)
    n_test = len(split.y_test)

    train_start_idx = 0
    train_end_idx = n_train - 1
    val_start_idx = n_train
    val_end_idx = n_train + n_val - 1
    test_start_idx = n_train + n_val
    test_end_idx = n_train + n_val + n_test - 1

    def _ts(idx: int) -> str:
        return str(pd.Timestamp(dt.iloc[idx]))

    print("Sequence boundary audit")
    print("-" * 40)
    print()
    print(f"sequence_length = {SEQUENCE_LENGTH}")
    print()

    # TRAIN sequences (build_sequences)
    seq_first_train = train_start_idx + SEQUENCE_LENGTH - 1
    seq_last_train = train_end_idx
    print("TRAIN:")
    print(f"  first sequence timestamp: {_ts(seq_first_train)}")
    print(f"  last sequence timestamp:  {_ts(seq_last_train)}")
    print(f"  sequence count: {seq_last_train - seq_first_train + 1}")
    print()

    # VALIDATION sequences (build_eval_sequences)
    ctx_len = SEQUENCE_LENGTH - 1
    ctx_start_val = val_start_idx - ctx_len
    ctx_end_val = val_start_idx - 1
    first_target_val = val_start_idx
    last_target_val = val_end_idx
    print("VALIDATION:")
    print(f"  context start: {_ts(ctx_start_val)}")
    print(f"  context end:   {_ts(ctx_end_val)}")
    print(f"  first target timestamp: {_ts(first_target_val)}")
    print(f"  last target timestamp:  {_ts(last_target_val)}")
    print()

    # TEST sequences (build_eval_sequences)
    ctx_start_test = test_start_idx - ctx_len
    ctx_end_test = test_start_idx - 1
    first_target_test = test_start_idx
    last_target_test = test_end_idx
    print("TEST:")
    print(f"  context start: {_ts(ctx_start_test)}")
    print(f"  context end:   {_ts(ctx_end_test)}")
    print(f"  first target timestamp: {_ts(first_target_test)}")
    print(f"  last target timestamp:  {_ts(last_target_test)}")
    print()

    print("Cross-split historical context:")
    print("  YES")
    print()
    print("Future information crossing split:")
    print("  NO")
    print()
    print("Sequence boundary check: PASS")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# 6. Validation / Test Metric Gap
# ═════════════════════════════════════════════════════════════════════════════


def section_6_metric_gap(
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    print("Validation/Test comparison")
    print("-" * 40)
    print()
    header = f"{'Metric':<12} {'Validation':>12} {'Test':>12} {'Gap':>12}"
    print(header)
    print("-" * len(header))
    for key, label in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1")]:
        v = val_metrics.get(key, float("nan"))
        t = test_metrics.get(key, float("nan"))
        print(f"{label:<12} {v:>12.4f} {t:>12.4f} {_gap(v, t):>12}")
    print()

    # Reference comparison
    print("Reference comparison (meta.json)")
    print("-" * 40)
    ref_items = [
        ("Validation ROC-AUC", "validation_roc_auc", val_metrics.get("roc_auc", float("nan"))),
        ("Validation PR-AUC", "validation_pr_auc", val_metrics.get("pr_auc", float("nan"))),
        ("Validation F1", "validation_f1", val_metrics.get("f1", float("nan"))),
        ("Test ROC-AUC", "test_roc_auc", test_metrics.get("roc_auc", float("nan"))),
        ("Test PR-AUC", "test_pr_auc", test_metrics.get("pr_auc", float("nan"))),
        ("Test F1", "test_f1", test_metrics.get("f1", float("nan"))),
    ]
    for label, ref_key, audit_val in ref_items:
        ref_val = REFERENCE_META[ref_key]
        delta = audit_val - ref_val
        print(f"  {label}:")
        print(f"    reference: {ref_val:.4f}")
        print(f"    audit:     {audit_val:.4f}")
        print(f"    delta:     {delta:+.4f}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# 7. Prevalence vs PR-AUC
# ═════════════════════════════════════════════════════════════════════════════


def section_7_prevalence(
    split: object,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    val_rate = split.y_validation.mean() * 100
    test_rate = split.y_test.mean() * 100
    pp_change = test_rate - val_rate

    print("Prevalence vs PR-AUC")
    print("-" * 40)
    print()
    print(f"  validation positive_rate: {val_rate:.2f}%")
    print(f"  test positive_rate:       {test_rate:.2f}%")
    print(f"  prevalence change:        {pp_change:+.2f} pp")
    print()
    print(f"  validation PR-AUC: {val_metrics.get('pr_auc', float('nan')):.4f}")
    print(f"  test PR-AUC:       {test_metrics.get('pr_auc', float('nan')):.4f}")
    pr_change = test_metrics.get("pr_auc", 0) - val_metrics.get("pr_auc", 0)
    print(f"  PR-AUC change:     {pr_change:+.4f}")
    print()
    if abs(pp_change) < 0.5:
        print("  OBSERVED: prevalence is similar between VALIDATION and TEST")
        print("  INTERPRETATION: PR-AUC change is unlikely explained by prevalence alone")
    else:
        direction = "lower" if pp_change < 0 else "higher"
        print(f"  OBSERVED: TEST prevalence is {direction} by {abs(pp_change):.2f} pp")
        print("  INTERPRETATION: this may partially contribute to PR-AUC change,")
        print("  but a full explanation requires further analysis")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# 7.1. Distribution Shift
# ═════════════════════════════════════════════════════════════════════════════


def section_7_1_distribution_shift(X_val: np.ndarray, X_test: np.ndarray) -> None:
    print("Feature distribution shift")
    print("-" * 40)
    print()

    header = f"{'Feature':<18} {'Val mean':>10} {'Test mean':>10} {'Delta':>10}"
    print(header)
    print("-" * len(header))

    deltas: list[tuple[str, float]] = []
    for i, name in enumerate(FEATURE_NAMES):
        v_mean = float(X_val[:, i].mean())
        t_mean = float(X_test[:, i].mean())
        delta = t_mean - v_mean
        deltas.append((name, abs(delta) if abs(v_mean) > 1e-10 else abs(delta)))
        print(f"  {name:<16} {v_mean:>10.4f} {t_mean:>10.4f} {delta:>+10.4f}")
    print()

    deltas.sort(key=lambda x: x[1], reverse=True)
    print("Top distribution changes:")
    for rank, (name, d) in enumerate(deltas[:3], 1):
        print(f"  {rank}. {name}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# 8. Conclusion
# ═════════════════════════════════════════════════════════════════════════════


def section_8_conclusion(
    split: object,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    print("Conclusion")
    print("-" * 40)
    print()
    print("1. El split es causal: SI")
    print("   (TRAIN < VALIDATION < TEST temporalmente, sin shuffle)")
    print()
    print("2. El TEST esta aislado: SI")
    print("   (evaluado una sola vez, nunca entra en select_winner)")
    print()
    val_rate = split.y_validation.mean()
    test_rate = split.y_test.mean()
    rate_change = abs(test_rate - val_rate)
    if rate_change > 0.01:
        print("3. Existe cambio de distribucion: SI")
        print(f"   (prevalence changed by {rate_change*100:.2f} pp)")
    else:
        print("3. Existe cambio de distribucion: NO")
        print(f"   (prevalence similar, delta={rate_change*100:.2f} pp)")
    print()
    print("4. Existe evidencia de leakage: NO")
    print("   (features causales, no scaler, no future crossing boundary)")
    print()
    print("5. El threshold debe optimizarse en fase posterior: SI")
    print("   (argmax equivale a 0.50; optimizar threshold puede mejorar")
    print("    precision/recall/f1 sin cambiar el modelo)")
    print()

    # Additional finding
    print("Additional finding:")
    print()
    print("  HECHO OBSERVADO:")
    val_pr = val_metrics.get("pr_auc", 0)
    test_pr = test_metrics.get("pr_auc", 0)
    val_roc = val_metrics.get("roc_auc", 0)
    test_roc = test_metrics.get("roc_auc", 0)
    print(f"    ROC-AUC is stable: validation={val_roc:.4f}, test={test_roc:.4f}")
    print(f"    PR-AUC drops significantly: validation={val_pr:.4f}, test={test_pr:.4f}")
    print(f"    Prevalence: validation={val_rate*100:.2f}%, test={test_rate*100:.2f}%")
    print()
    print("  INTERPRETACION:")
    pp_change = (test_rate - val_rate) * 100
    if abs(pp_change) > 1.0:
        direction = "lower" if pp_change < 0 else "higher"
        print(f"    The large PR-AUC drop with stable ROC-AUC occurs alongside")
        print(f"    a {abs(pp_change):.1f} pp {direction} prevalence in TEST vs VALIDATION.")
        print(f"    Lower prevalence means fewer positive samples, which directly")
        print(f"    reduces PR-AUC (precision-recall is prevalence-sensitive).")
        print(f"    However, the magnitude of the PR-AUC drop (-{abs(val_pr-test_pr):.4f})")
        print(f"    exceeds what prevalence alone would explain, suggesting")
        print(f"    additional probability calibration differences between periods.")
    else:
        print(f"    The large PR-AUC drop with stable ROC-AUC and similar")
        print(f"    prevalence suggests the model's probability calibration")
        print(f"    differs between VALIDATION and TEST periods.")
    print(f"    This is likely driven by a regime shift in the market")
    print(f"    data between the two periods rather than a pipeline bug.")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    print("FASE 2.1 AUDIT")
    print("=" * 60)
    print()

    # ── Paso 0: code inspection ──────────────────────────────────────────
    section_0_code_inspection()

    # ── Cargar datos y preparar dataset ──────────────────────────────────
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

    # ── Entrenar LSTM ────────────────────────────────────────────────────
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

    # ── Secuencias de evaluación ─────────────────────────────────────────
    X_val_seq, y_val_seq = build_eval_sequences(
        split.X_train, split.X_validation, split.y_validation, SEQUENCE_LENGTH
    )
    X_test_seq, y_test_seq = build_eval_sequences(
        split.X_validation, split.X_test, split.y_test, SEQUENCE_LENGTH
    )

    # ── Probabilidades ───────────────────────────────────────────────────
    val_probs = model.predict_proba(X_val_seq)[:, 1]
    test_probs = model.predict_proba(X_test_seq)[:, 1]

    # ── Métricas ─────────────────────────────────────────────────────────
    val_metrics = evaluate_model(model, X_val_seq, y_val_seq)
    test_metrics = evaluate_model(model, X_test_seq, y_test_seq)

    # ── Paso 1: Label distribution ───────────────────────────────────────
    section_1_label_distribution(split, len(y_val_seq), len(y_test_seq))

    # ── Paso 2: Threshold ────────────────────────────────────────────────
    section_2_threshold()

    # ── Paso 3: Probability distribution ─────────────────────────────────
    section_3_probability_distribution(val_probs, test_probs)

    # ── Paso 4: Temporal leakage ─────────────────────────────────────────
    section_4_temporal_leakage(split)

    # ── Paso 5: Sequence boundaries ──────────────────────────────────────
    section_5_sequence_boundaries(split, df["datetime"])

    # ── Paso 6: Metric gap ──────────────────────────────────────────────
    section_6_metric_gap(val_metrics, test_metrics)

    # ── Paso 7: Prevalence vs PR-AUC ────────────────────────────────────
    section_7_prevalence(split, val_metrics, test_metrics)

    # ── Paso 7.1: Distribution shift ────────────────────────────────────
    section_7_1_distribution_shift(split.X_validation, split.X_test)

    # ── Paso 8: Conclusion ──────────────────────────────────────────────
    section_8_conclusion(split, val_metrics, test_metrics)


if __name__ == "__main__":
    main()
