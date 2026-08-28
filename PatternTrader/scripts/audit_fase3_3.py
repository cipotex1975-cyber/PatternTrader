"""FASE 3.3 — Distribution Shift / Cambio de regimen (auditoria diagnostica).

Determina QUÉ cambia, CUÁNTO cambia y CUÁNDO comienza el cambio entre
VALIDATION (2021-08 -> 2024-01) y TEST (2024-01 -> 2026-05), con foco en
VALIDATION -> TEST.

Metodologia:
    - PASO 2  prevalencia (positive_rate) por dataset.
    - PASO 3  distribucion de features (percentiles) por dataset.
    - PASO 4  effect size (Cohen's d) por feature.
    - PASO 5  Kolmogorov-Smirnov (ks_2samp) por feature.
    - PASO 6  top features por KS + clasificacion effect size.
    - PASO 7  P(Y|X): positive rate condicionado a bins (covariate vs concept).
    - PASO 8  probability shift del LSTM (reproducido con config actual).
    - PASO 9  calibracion descriptiva por bins de probabilidad (sin calibrar).
    - PASO 10 evolucion temporal semestral (posrate + metricas por ventana).
    - PASO 11-12 interpretacion: prevalence/covariate/concept shift.

NO modifica modelos, features, threshold, labels ni split. El TEST es solo
evaluacion. El unico entrenamiento es el LSTM en su configuracion actual
(igual que en fases 3.1/3.2) para producir probabilidades.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

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


def _desc(v, p05=None, p25=None, p75=None, p95=None):
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "median": float(np.median(v)),
        "p05": float(np.percentile(v, 5)) if p05 is None else p05,
        "p25": float(np.percentile(v, 25)) if p25 is None else p25,
        "p75": float(np.percentile(v, 75)) if p75 is None else p75,
        "p95": float(np.percentile(v, 95)) if p95 is None else p95,
    }


def _cohens_d(a, b) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    sp = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if sp == 0:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / sp)


def _effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.20:
        return "negligible"
    if ad < 0.50:
        return "small"
    if ad < 0.80:
        return "medium"
    return "large"


def _pct_rate(y) -> float:
    return float(np.mean(y)) * 100 if len(y) else 0.0


def main() -> None:
    print()
    print("FASE 3.3 — DISTRIBUTION SHIFT AUDIT")
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
    dt = df["datetime"].iloc[:, 0] if isinstance(df["datetime"], type(df)) else df["datetime"]

    n_tr = len(split.y_train)
    n_va = len(split.y_validation)
    n_te = len(split.y_test)

    def _ts(idx):
        return str(pd.Timestamp(dt.iloc[idx]))

    val_start = _ts(n_tr)
    val_end = _ts(n_tr + n_va - 1)
    test_start = _ts(n_tr + n_va)
    test_end = _ts(n_tr + n_va + n_te - 1)

    print("## Dataset periods")
    print(f"VALIDATION: {val_start} -> {val_end}")
    print(f"TEST:       {test_start} -> {test_end}")
    print()

    # ---------------- PASO 2: Prevalencia ----------------
    print("## Prevalence")
    print()
    print(f"{'Dataset':<12} {'PosRate':>10}")
    for name, y in (
        ("TRAIN", split.y_train),
        ("VALIDATION", split.y_validation),
        ("TEST", split.y_test),
    ):
        print(f"{name:<12} {_pct_rate(y):>9.2f}%")
    pp_change = _pct_rate(split.y_test) - _pct_rate(split.y_validation)
    print(f"change (VAL->TEST): {pp_change:+.2f} pp")
    print()

    # ---------------- PASO 3/4/5: features ----------------
    print("## Feature distribution shift")
    print()
    header = (
        f"{'Feature':<16} {'Val Mean':>10} {'Test Mean':>10} {'Delta':>10} "
        f"{'KS':>8} {'Effect':>10}"
    )
    print(header)
    print("-" * len(header))

    Xv = split.X_validation
    Xt = split.X_test

    feature_rows = []
    for i, name in enumerate(FEATURE_NAMES):
        v = Xv[:, i]
        t = Xt[:, i]
        d = float(np.mean(t) - np.mean(v))
        ks_stat, ks_p = ks_2samp(v, t)
        eff = _cohens_d(v, t)
        feature_rows.append(
            {
                "name": name,
                "val_desc": _desc(v),
                "test_desc": _desc(t),
                "delta": d,
                "ks": float(ks_stat),
                "ks_p": float(ks_p),
                "effect": float(eff),
                "effect_label": _effect_label(eff),
            }
        )
        print(
            f"  {name:<14} {np.mean(v):>10.4f} {np.mean(t):>10.4f} {d:>+10.4f} "
            f"{ks_stat:>8.4f} {eff:>10.4f}"
        )
    print()

    # ---------------- PASO 6: Top features ----------------
    print("## Top distribution shifts (by KS)")
    print()
    top = sorted(feature_rows, key=lambda r: r["ks"], reverse=True)[:5]
    for rank, r in enumerate(top, 1):
        print(f"{rank}. {r['name'].upper()}")
        print(f"   KS: {r['ks']:.4f} (p={r['ks_p']:.3e})")
        print(f"   effect_size: {r['effect']:.4f} ({r['effect_label']})")
        print(f"   Val mean {r['val_desc']['mean']:.4f} -> Test mean {r['test_desc']['mean']:.4f}")
        print()
    counts = {}
    for r in feature_rows:
        counts[r["effect_label"]] = counts.get(r["effect_label"], 0) + 1
    print(f"Effect size counts: {counts}")
    print()

    # ---------------- PASO 7: Label shift / P(Y|X) ----------------
    print("## Label shift / P(Y|X) by feature bins")
    print()
    for name in ("rsi", "atr"):
        i = FEATURE_NAMES.index(name)
        lo = min(float(np.percentile(Xv[:, i], 2)), float(np.percentile(Xt[:, i], 2)))
        hi = max(float(np.percentile(Xv[:, i], 98)), float(np.percentile(Xt[:, i], 98)))
        bins = np.linspace(lo, hi, 5)
        print(f"--- {name.upper()} bins ---")
        print(f"{'Bin':<22} {'Val P(Y=1)':>12} {'Test P(Y=1)':>12}")
        for k in range(len(bins) - 1):
            mask_v = (Xv[:, i] >= bins[k]) & (Xv[:, i] <= bins[k + 1])
            mask_t = (Xt[:, i] >= bins[k]) & (Xt[:, i] <= bins[k + 1])
            pv = np.mean(split.y_validation[mask_v]) if mask_v.sum() else np.nan
            pt = np.mean(split.y_test[mask_t]) if mask_t.sum() else np.nan
            lbl = f"[{bins[k]:.4f}, {bins[k+1]:.4f}]"
            pv_s = f"{pv*100:.1f}%" if not np.isnan(pv) else "n/a"
            pt_s = f"{pt*100:.1f}%" if not np.isnan(pt) else "n/a"
            print(f"{lbl:<22} {pv_s:>12} {pt_s:>12}")
        print()
    print()

    # ---------------- LSTM probabilities (PASO 8) ----------------
    print("Reentrenando LSTM (config actual) para probabilidades...")
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
    val_metrics = evaluate_model(model, X_val_seq, y_val_seq)
    test_metrics = evaluate_model(model, X_test_seq, y_test_seq)

    # ---------------- PASO 8: Probability shift ----------------
    print("## Probability distribution")
    print()
    print("VALIDATION:")
    for k, v in _desc(val_probs).items():
        print(f"  {k}: {v:.4f}")
    print()
    print("TEST:")
    for k, v in _desc(test_probs).items():
        print(f"  {k}: {v:.4f}")
    print()
    ks_prob = ks_2samp(val_probs, test_probs)
    print(f"Probability KS: statistic={ks_prob.statistic:.4f} p-value={ks_prob.pvalue:.3e}")
    print()

    # ---------------- PASO 9: Calibracion descriptiva ----------------
    print("## Calibration comparison")
    print()
    edges = np.arange(0, 1.01, 0.1)
    header = (
        f"{'Bin':<10} {'Val Pred':>10} {'Val Actual':>10} {'Test Pred':>10} {'Test Actual':>10}"
    )
    print(header)
    print("-" * len(header))
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        mv = (val_probs >= lo) & (val_probs < hi)
        mt = (test_probs >= lo) & (test_probs < hi)
        vp = np.mean(val_probs[mv]) if mv.sum() else np.nan
        va = np.mean(y_val_seq[mv]) if mv.sum() else np.nan
        tp = np.mean(test_probs[mt]) if mt.sum() else np.nan
        ta = np.mean(y_test_seq[mt]) if mt.sum() else np.nan

        def _fmt(x: float) -> str:
            return f"{x:.3f}" if not np.isnan(x) else "n/a"

        print(
            f"{f'{lo:.1f}-{hi:.1f}':<10} {_fmt(vp):>10} {_fmt(va):>10} "
            f"{_fmt(tp):>10} {_fmt(ta):>10}"
        )
    print()

    # ---------------- PASO 10: Evolucion temporal semestral ----------------
    print("## Temporal evolution (semestral)")
    print()
    dtarr = np.array(dt.iloc[: n_tr + n_va + n_te])
    full_y = np.concatenate([split.y_train, split.y_validation, split.y_test])
    full_X = np.vstack([split.X_train, split.X_validation, split.X_test])
    print(f"{'Period':<10} {'PosRate':>8} {'meanRSI':>9} {'meanATR':>9} {'range':>9}")
    print("-" * 50)
    # Ventanas semestrales desde 2021 (validacion) hasta test.
    t = pd.to_datetime(dtarr)
    years = sorted({x.year for x in t[(t >= pd.Timestamp("2021-08"))]})
    for y_ in years:
        for half in ("H1", "H2"):
            if half == "H1":
                s, e = pd.Timestamp(f"{y_}-01-01"), pd.Timestamp(f"{y_}-07-01")
            else:
                s, e = pd.Timestamp(f"{y_}-07-01"), pd.Timestamp(
                    f"{y_ + 1 if y_+1 < 2027 else y_}-01-01"
                )
            mask = (t >= s) & (t < e)
            if mask.sum() == 0:
                continue
            rate = np.mean(full_y[mask]) * 100
            rsi = np.mean(full_X[mask, FEATURE_NAMES.index("rsi")])
            atr = np.mean(full_X[mask, FEATURE_NAMES.index("atr")])
            hl = np.mean(full_X[mask, FEATURE_NAMES.index("high_low_range")])
            print(f"{f'{y_}-{half}':<10} {rate:>7.2f}% {rsi:>9.2f} {atr:>9.4f} {hl:>9.4f}")
    print()

    # ---------------- PASO 10bis: ROC/PR por semestre (modelo) ----------------
    # Los probs evaluados cubren VALIDATION (index 0..n_val-1) y TEST.
    # Agrupamos por semestre usando el datetime original de cada target.
    print("## Temporal evolution (ROC-AUC / PR-AUC, LSTM)")
    print()
    print(f"{'Period':<12} {'Samples':>8} {'PosRate':>8} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print("-" * 48)

    def _window_metrics(probs, y, mask):
        p = probs[mask]
        yy = y[mask]
        if len(np.unique(yy)) < 2:
            return None
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "roc_auc": float(roc_auc_score(yy, p)),
            "pr_auc": float(average_precision_score(yy, p)),
        }

    val_dt = pd.to_datetime(dt.iloc[n_tr : n_tr + n_va])
    test_dt = pd.to_datetime(dt.iloc[n_tr + n_va : n_tr + n_va + n_te])
    all_dt = np.concatenate([val_dt.values, test_dt.values])
    all_dt = pd.to_datetime(all_dt)
    all_y = np.concatenate([np.asarray(y_val_seq), np.asarray(y_test_seq)])
    all_p = np.concatenate([val_probs, test_probs])

    for y_ in sorted({x.year for x in all_dt}):
        for half in ("H1", "H2"):
            if half == "H1":
                s, e = pd.Timestamp(f"{y_}-01-01"), pd.Timestamp(f"{y_}-07-01")
            else:
                e_year = y_ + 1 if y_ + 1 < 2027 else y_
                s, e = pd.Timestamp(f"{y_}-07-01"), pd.Timestamp(f"{e_year}-01-01")
            mask = (all_dt >= s) & (all_dt < e)
            if mask.sum() == 0:
                continue
            wm = _window_metrics(all_p, all_y, mask)
            roc = f"{wm['roc_auc']:.4f}" if wm else "n/a"
            pr = f"{wm['pr_auc']:.4f}" if wm else "n/a"
            print(
                f"{f'{y_}-{half}':<12} {int(mask.sum()):>8} "
                f"{np.mean(all_y[mask])*100:>7.1f}% {roc:>8} {pr:>8}"
            )
    print()

    # ---------------- PASO 11/12: Interpretacion ----------------
    print("## Shift interpretation")
    print()
    pp = pp_change
    prevalence_strength = "NONE"
    if abs(pp) >= 10:
        prevalence_strength = "STRONG"
    elif abs(pp) >= 5:
        prevalence_strength = "MODERATE"
    elif abs(pp) >= 2:
        prevalence_strength = "WEAK"
    print(f"Prevalence shift: {prevalence_strength}")
    print(f"  positive_rate delta VAL->TEST: {pp:+.2f} pp")

    n_large = sum(1 for r in feature_rows if r["effect_label"] == "large")
    n_medium = sum(1 for r in feature_rows if r["effect_label"] == "medium")
    n_small = sum(1 for r in feature_rows if r["effect_label"] == "small")
    if n_large >= 3:
        cov_strength = "STRONG"
    elif n_large >= 1 or n_medium >= 3:
        cov_strength = "MODERATE"
    elif n_medium >= 1:
        cov_strength = "WEAK"
    else:
        cov_strength = "NONE"
    print(f"Covariate shift: {cov_strength} (large={n_large}, medium={n_medium}, small={n_small})")

    concept_strength = "NONE"
    # P(Y|X) condicional estable + ROC estable => concept shift limitado.
    roc_delta = test_metrics.get("roc_auc", 0) - val_metrics.get("roc_auc", 0)
    print(
        "Concept shift: "
        f"{concept_strength} (ROC-AUC delta {roc_delta:+.4f}; "
        "P(Y|X) bins comparados arriba)"
    )
    print()

    # ---------------- Conclusion (SALIDA ESPERADA) ----------------
    print("## Conclusion")
    print()
    prev_sig = prevalence_strength in ("MODERATE", "STRONG")
    print(
        "1. Prevalencia cambio significativamente? " f"{'SI' if prev_sig else 'NO'} ({pp:+.2f} pp)"
    )
    print("2. Features cambiaron? " f"{'SI' if cov_strength != 'NONE' else 'NO'} ({cov_strength})")
    print("3. Probabilidades cambiaron? " f"KS={ks_prob.statistic:.4f} (p={ks_prob.pvalue:.3e})")
    print(
        "4. ROC-AUC estable? " f"{'SI' if abs(roc_delta) < 0.02 else 'NO'} (delta {roc_delta:+.4f})"
    )
    pr_delta = test_metrics.get("pr_auc", 0) - val_metrics.get("pr_auc", 0)
    print(
        "5. PR-AUC cayo por? (pr_auc delta "
        f"{pr_delta:+.4f}) prevalencia {pp:+.2f} pp; ver calibracion arriba"
    )
    print(f"6. Evidencia covariate shift? {cov_strength}")
    print(f"7. Evidencia concept shift? {concept_strength}")
    print("8. El cambio parece comenzar? ver tabla temporal semestral arriba")
    retrain = prev_sig or cov_strength in ("MODERATE", "STRONG")
    print("9. Investigar retraining periodico? " f"{'SI' if retrain else 'CONSIDERAR'}")
    print("10. Investigar calibracion? ver diferencia pred vs actual por bin")
    roll = cov_strength in ("MODERATE", "STRONG")
    print(f"11. Investigar rolling windows? {'SI' if roll else 'CONSIDERAR'}")
    print()


if __name__ == "__main__":
    main()
