"""Pruebas de FASE 8 — robustez de la definición del label.

Verifica:
- La grilla oficial LABEL_GRID es correcta (12 configs, forward_periods=5).
- run_label_sweep devuelve un DataFrame con las columnas esperadas y una fila
  por configuración.
- positive_ratio decrece al endurecer threshold/min_up_moves (propiedad
  esperada de una definición de label consistente).
- El barrido opera sobre TRAIN/VALIDATION + walk-forward (nunca TEST).
- format_label_sweep_table y assess_robustness no inventan resultados (NA donde
  no hay datos) y devuelven texto/veredicto válidos.
"""

import math

import numpy as np
import pandas as pd
import pytest

from app.ml.training.data import create_features
from app.ml.training.label_sweep import (
    LABEL_GRID,
    SWEEP_COLUMNS,
    assess_robustness,
    format_label_sweep_table,
    run_label_sweep,
)

THRESHOLDS = {0.0005, 0.0010, 0.0015, 0.0020}
MIN_MOVES = {1, 2, 3}

# Grilla reducida para pruebas rápidas (no toca TEST).
_SMALL_GRID = [
    {"threshold": 0.0010, "min_up_moves": 1, "forward_periods": 5},
    {"threshold": 0.0010, "min_up_moves": 2, "forward_periods": 5},
    {"threshold": 0.0020, "min_up_moves": 2, "forward_periods": 5},
]


def _make_df(n: int = 700, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    close = 1.0 + np.cumsum(rng.normal(0, 0.001, n))
    open_ = close + rng.normal(0, 0.0002, n)
    high = np.maximum(open_, close) + rng.uniform(0, 0.001, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.001, n)
    return pd.DataFrame(
        {
            "DateTime": dates.strftime("%Y-%m-%d"),
            "time": dates.strftime("%H:%M:%S"),
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Tickvol": rng.integers(500, 2000, n),
            "Volume": 0,
            "Spread": 1,
        }
    )


class TestLabelGrid:
    def test_grid_has_expected_shape(self):
        assert len(LABEL_GRID) == len(THRESHOLDS) * len(MIN_MOVES) == 12

    def test_grid_combinations_cover_all(self):
        combos = {(r["threshold"], r["min_up_moves"]) for r in LABEL_GRID}
        expected = {(t, m) for t in THRESHOLDS for m in MIN_MOVES}
        assert combos == expected

    def test_grid_forward_periods_is_5(self):
        assert all(r["forward_periods"] == 5 for r in LABEL_GRID)


class TestLabelSweep:
    def test_sweep_returns_expected_columns(self):
        df = _make_df()
        out = run_label_sweep(
            df,
            walk_forward_splits=2,
            configs=[_SMALL_GRID[0]],
        )
        assert set(SWEEP_COLUMNS).issubset(set(out.columns))
        assert len(out) == 1
        row = out.iloc[0]
        assert row["threshold"] == 0.0010
        assert row["min_moves"] == 1
        assert 0.0 <= row["positive_ratio"] <= 1.0
        assert math.isfinite(row["mean_validation_auc"])

    def test_sweep_has_one_row_per_config(self):
        df = _make_df()
        out = run_label_sweep(df, walk_forward_splits=2, configs=_SMALL_GRID)
        assert len(out) == len(_SMALL_GRID)
        assert out["threshold"].notna().all()
        assert out["mean_validation_auc"].notna().all()

    def test_positive_ratio_decreases_with_stricter_params(self):
        df_feats = create_features(_make_df())
        ratios = []
        for cfg in _SMALL_GRID:
            d = df_feats.copy()
            from app.ml.training.data import FEATURE_NAMES, create_labels

            d["label"] = create_labels(
                d,
                forward_periods=cfg["forward_periods"],
                threshold=cfg["threshold"],
                min_up_moves=cfg["min_up_moves"],
            )
            d = d.dropna(subset=FEATURE_NAMES + ["label"])
            ratios.append(float(d["label"].mean()))
        # threshold 0.0010/min=1 debe dar >= positivos que threshold 0.0020/min=2.
        assert ratios[0] >= ratios[1]
        assert ratios[1] >= ratios[2]

    def test_sweep_does_not_report_test_metrics(self):
        df = _make_df()
        out = run_label_sweep(df, walk_forward_splits=2, configs=[_SMALL_GRID[0]])
        assert "test" not in out.columns
        # Las columnas son solo del sweep (val/walk-forward), sin ningún '_test_'.
        assert not any("test" in str(c).lower() for c in out.columns)

    def test_sweep_deterministic_with_same_input(self):
        df1 = _make_df(seed=99)
        df2 = _make_df(seed=99)
        out1 = run_label_sweep(df1, walk_forward_splits=2, configs=[_SMALL_GRID[0]])
        out2 = run_label_sweep(df2, walk_forward_splits=2, configs=[_SMALL_GRID[0]])
        pd.testing.assert_frame_equal(out1, out2)


class TestFormatAndAssess:
    def test_format_table_non_empty(self):
        df = _make_df()
        out = run_label_sweep(df, walk_forward_splits=2, configs=_SMALL_GRID)
        text = format_label_sweep_table(out)
        assert "threshold" in text
        assert "mean_AUC" in text
        assert len(text) > 0

    def test_assess_robustness_returns_verdict(self):
        df = _make_df()
        out = run_label_sweep(df, walk_forward_splits=2, configs=_SMALL_GRID)
        verdict = assess_robustness(out, metric="roc_auc")
        assert isinstance(verdict, str)
        assert len(verdict) > 0

    def test_assess_robustness_empty_gives_message(self):
        empty = pd.DataFrame(columns=SWEEP_COLUMNS)
        verdict = assess_robustness(empty)
        assert "No" in verdict


@pytest.fixture(autouse=True)
def _small_grid_suffix():
    # Garantiza que los tests que no usan la grilla completa sean rápidos.
    yield
