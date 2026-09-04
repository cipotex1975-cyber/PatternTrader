from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.core.logger import get_logger
from app.ml.training.compare import run_walk_forward_comparison
from app.ml.training.data import FEATURE_NAMES, create_features, create_labels, split_chronological

logger = get_logger("LabelSweep")

# Grilla oficial de la FASE 8 (docs/mejoras/fase8): 4 thresholds x 3 min_up_moves,
# con forward_periods fijo en 5. Se evalúa SOLO sobre TRAIN/VALIDATION o walk-forward;
# el TEST FINAL nunca participa.
LABEL_GRID: list[dict[str, Any]] = (
    [{"threshold": 0.0005, "min_up_moves": m, "forward_periods": 5} for m in (1, 2, 3)]
    + [{"threshold": 0.0010, "min_up_moves": m, "forward_periods": 5} for m in (1, 2, 3)]
    + [{"threshold": 0.0015, "min_up_moves": m, "forward_periods": 5} for m in (1, 2, 3)]
    + [{"threshold": 0.0020, "min_up_moves": m, "forward_periods": 5} for m in (1, 2, 3)]
)

SWEEP_COLUMNS = [
    "threshold",
    "min_moves",
    "forward_periods",
    "positive_ratio",
    "mean_validation_auc",
    "std_validation_auc",
    "mean_validation_pr_auc",
]


def _positive_ratio_from_df(df: pd.DataFrame) -> float:
    """Fracción de muestras positivas tras dropna de features+label."""
    if df.empty:
        return float("nan")
    return float(df["label"].mean())


def run_label_sweep(
    df_raw: pd.DataFrame,
    model_names: list[str] | None = None,
    metric: str = "roc_auc",
    walk_forward_splits: int = 3,
    settings: Any = None,
    seed: int | None = None,
    hyperparams: dict[str, dict[str, Any]] | None = None,
    configs: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Barrido de robustez de la definición del label (FASE 8, sección 1-2).

    Para cada configuración de ``LABEL_GRID`` (threshold x min_up_moves,
    forward_periods=5) evalúa la señal mediante walk-forward sobre el conjunto
    de selección (TRAIN+VALIDATION cronológico). El TEST FINAL NO participa.

    Las features son causales (independientes del label), por lo que se calculan
    UNA sola vez sobre la serie completa y se reutilizan en cada configuración.

    Retorna un ``pd.DataFrame`` con una fila por configuración y las columnas:
    threshold, min_moves, forward_periods, positive_ratio,
    mean_validation_auc, std_validation_auc, mean_validation_pr_auc.
    """
    if walk_forward_splits < 2:
        raise ValueError("walk_forward_splits debe ser >= 2 para el label sweep (FASE 8)")

    best = model_names or ["random_forest"]
    names = [n for n in best if n in {"random_forest", "xgboost", "lightgbm", "catboost"}]
    if not names:
        # Fallback: cualquier supervisado disponible que no necesite secuencias.
        names = ["random_forest"]

    # Features causales: se calculan una sola vez (independientes del label).
    df_feats = create_features(df_raw)

    grid = configs if configs is not None else LABEL_GRID
    rows: list[dict[str, Any]] = []
    for cfg in grid:
        threshold = float(cfg["threshold"])
        min_moves = int(cfg["min_up_moves"])
        forward_periods = int(cfg["forward_periods"])

        try:
            df = df_feats.copy()
            df["label"] = create_labels(
                df,
                forward_periods=forward_periods,
                threshold=threshold,
                min_up_moves=min_moves,
            )
            df = df.dropna(subset=FEATURE_NAMES + ["label"])

            positive_ratio = _positive_ratio_from_df(df)

            split = split_chronological(
                df,
                train_size=0.70,
                validation_size=0.15,
                forward_periods=forward_periods,
            )

            X_selection = np.concatenate([split.X_train, split.X_validation], axis=0)
            y_selection = np.concatenate([split.y_train, split.y_validation], axis=0)

            summary, _ = run_walk_forward_comparison(
                X_selection,
                y_selection,
                n_splits=walk_forward_splits,
                model_names=names,
                metric=metric,
                forward_periods=forward_periods,
                min_train_size=100,
                feature_names=None,
                settings=settings,
                hyperparams=hyperparams,
            )

            row = summary.iloc[0]
            mean_auc = _as_float(row.get(f"wf_mean_{metric}"))
            std_auc = _as_float(row.get(f"wf_std_{metric}"))
            mean_pr = _as_float(row.get("wf_mean_pr_auc"))

            rows.append(
                {
                    "threshold": threshold,
                    "min_moves": min_moves,
                    "forward_periods": forward_periods,
                    "positive_ratio": positive_ratio,
                    "mean_validation_auc": mean_auc,
                    "std_validation_auc": std_auc,
                    "mean_validation_pr_auc": mean_pr,
                }
            )
            logger.info(
                f"threshold={threshold:.4f} min_moves={min_moves} "
                f"pos={positive_ratio:.3f} mean_{metric}={mean_auc:.4f} "
                f"std={std_auc:.4f} pr_auc={mean_pr:.4f}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"threshold={threshold:.4f} min_moves={min_moves}: "
                f"configuración no evaluable ({exc})"
            )
            rows.append(
                {
                    "threshold": threshold,
                    "min_moves": min_moves,
                    "forward_periods": forward_periods,
                    "positive_ratio": float("nan"),
                    "mean_validation_auc": float("nan"),
                    "std_validation_auc": float("nan"),
                    "mean_validation_pr_auc": float("nan"),
                }
            )

    out = pd.DataFrame(rows, columns=SWEEP_COLUMNS)
    if out.empty:
        raise RuntimeError("Ninguna configuración de label pudo evaluarse")
    return out.sort_values("mean_validation_auc", ascending=False).reset_index(drop=True)


def format_label_sweep_table(df: pd.DataFrame) -> str:
    """Renderiza la tabla de robustez del label (FASE 8, sección 3)."""
    header = (
        f"{'threshold':>9} {'min_moves':>9} {'positive_%':>10} {'mean_AUC':>9} "
        f"{'std_AUC':>9} {'PR_AUC':>9}"
    )
    lines = [
        "Label robustness sweep (TRAIN/VALIDATION + walk-forward)",
        "==========================================================",
        header,
        "-" * len(header),
    ]
    for _, row in df.iterrows():

        def fmt(v: Any) -> str:
            return (
                f"{float(v):>9.4f}"
                if isinstance(v, (int, float)) and not math.isnan(v)
                else f"{'-':>9}"
            )

        pct = (
            f"{float(row['positive_ratio']) * 100:>9.2f}%"
            if isinstance(row["positive_ratio"], (int, float))
            and not math.isnan(row["positive_ratio"])
            else f"{'-':>9}"
        )
        lines.append(
            f"{row['threshold']:>9.4f} {int(row['min_moves']):>9d} {pct} "
            f"{fmt(row['mean_validation_auc'])} {fmt(row['std_validation_auc'])} "
            f"{fmt(row['mean_validation_pr_auc'])}"
        )
    return "\n".join(lines)


def assess_robustness(
    df: pd.DataFrame, metric: str = "roc_auc", stability_threshold: float = 0.58
) -> str:
    """Diagnóstico cualitativo de robustez (FASE 8, sección 4).

    Siguiendo el criterio del documento: si el AUC se mantiene en un rango
    alto (>= ``stability_threshold``) en varias configuraciones, la señal es
    robusta; si colapsa al variar los parámetros (p. ej. un pico aislado), es
    frágil. No es una selección: solo reporte. La columna de referencia fija es
    ``mean_validation_auc`` (el sweep guarda ahí la media de la métrica
    elegida); el parámetro ``metric`` solo afecta al texto del veredicto.
    """
    col = "mean_validation_auc"
    if col not in df.columns:
        return (
            f"No hay columna '{col}' en la tabla del sweep. "
            "No se puede emitir un diagnóstico de robustez."
        )
    valid = df.dropna(subset=[col])
    if valid.empty:
        return (
            f"No hay configuraciones evaluables para '{metric}'. "
            "No se puede emitir un diagnóstico de robustez."
        )

    vals = valid[col].astype(float)
    stable = [v for v in vals if v >= stability_threshold]
    spread = float(vals.max() - vals.min())
    n_stable = len(stable)
    n_total = len(vals)

    if n_stable >= max(1, n_total // 2):
        verdict = (
            f"Señal relativamente ROBUSTA: {n_stable}/{n_total} configuraciones "
            f"mantienen {metric} >= {stability_threshold:.2f} "
            f"(rango [{vals.min():.3f}, {vals.max():.3f}], spread {spread:.3f})."
        )
    elif vals.max() >= stability_threshold:
        verdict = (
            f"Posiblemente FRÁGIL: solo {n_stable}/{n_total} configuraciones alcanzan "
            f"{metric} >= {stability_threshold:.2f}. El pico ({vals.max():.3f}) es aislado; "
            "el resultado depende sensiblemente de la definición del label."
        )
    else:
        verdict = (
            f"Señal DÉBIL/NO EVIDENTE: ninguna configuración alcanza "
            f"{metric} >= {stability_threshold:.2f} (máx {vals.max():.3f})."
        )
    return verdict


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v if not math.isnan(v) else float("nan")
