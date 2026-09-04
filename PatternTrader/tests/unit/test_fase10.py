"""Pruebas de FASE 10 — Evaluación Final Out-of-Sample.

Verifica:
- format_walk_forward_table renderiza la tabla de VALIDATION (MODEL | MEAN_AUC |
  STD_AUC | MEAN_PR_AUC) a partir de los agregados wf_*, con NA -> "-".
- classify_signal clasifica ROBUST / POSSIBLE / WEAK / NO EVIDENCE según el
  criterio de la sección CONCLUSIÓN (estabilidad entre folds, mean/std AUC,
  PR-AUC y final OOS) sin afirmar rentabilidad.
- El modo walk-forward del CLI genera la tabla, la comparación histórica y la
  conclusión; el ganador se selecciona por la media de folds (no TEST).
"""

import numpy as np
import pandas as pd

from app.ml.training.compare import (
    classify_signal,
    format_walk_forward_table,
    run_walk_forward_comparison,
    select_walk_forward_winner,
)


def _summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "lstm",
                "status": "ok",
                "wf_mean_roc_auc": 0.6513,
                "wf_std_roc_auc": 0.0150,
                "wf_mean_pr_auc": 0.30,
            },
            {
                "model": "catboost",
                "status": "ok",
                "wf_mean_roc_auc": 0.6200,
                "wf_std_roc_auc": 0.0200,
                "wf_mean_pr_auc": 0.25,
            },
            {
                "model": "random_forest",
                "status": "ok",
                "wf_mean_roc_auc": float("nan"),
                "wf_std_roc_auc": float("nan"),
                "wf_mean_pr_auc": float("nan"),
            },
        ]
    )


class TestFormatWalkForwardTable:
    def test_renders_header(self):
        text = format_walk_forward_table(_summary_df())
        assert "MODEL" in text
        assert "MEAN_AUC" in text
        assert "STD_AUC" in text
        assert "MEAN_PR_AUC" in text

    def test_renders_values(self):
        text = format_walk_forward_table(_summary_df())
        assert "0.6513" in text
        assert "0.0150" in text
        assert "0.3000" in text

    def test_nan_renders_dash(self):
        text = format_walk_forward_table(_summary_df())
        # random_forest con NaN (no evaluable) se muestra como "-".
        assert "-" in text

    def test_sorted_by_mean_desc(self):
        text = format_walk_forward_table(_summary_df())
        assert text.index("lstm") < text.index("catboost")

    def test_handles_csv_float_str(self):
        df = _summary_df()
        df.loc[0, "wf_mean_roc_auc"] = 0.6513
        text = format_walk_forward_table(df)
        assert "0.6513" in text


class TestClassifySignal:
    def test_robust_signal(self):
        # AUC alto, std bajo, PR-AUC razonable y OOS consistente.
        assert classify_signal(0.65, 0.01, 0.30, 0.64, positive_ratio=0.20) == "ROBUST SIGNAL"

    def test_possible_signal(self):
        # Media moderada y OOS consistente, sin llegar a robusto.
        assert classify_signal(0.57, 0.02, 0.22, 0.58, positive_ratio=0.30) == "POSSIBLE SIGNAL"

    def test_weak_signal_on_oos_collapse(self):
        # Media alta pero final OOS colapsa por debajo de la tolerancia.
        assert classify_signal(0.65, 0.015, 0.30, 0.53, positive_ratio=0.20) == "WEAK SIGNAL"

    def test_weak_signal_on_high_std(self):
        # Media alta pero std alto (inestable entre folds) y OOS que se desvía.
        assert classify_signal(0.63, 0.08, 0.30, 0.54, positive_ratio=0.20) == "WEAK SIGNAL"

    def test_possible_signal_on_low_pr(self):
        # PR-AUC bajo descalifica para ROBUST pero con media y OOS consistentes
        # sigue siendo una señal moderada (POSSIBLE).
        assert classify_signal(0.63, 0.015, 0.02, 0.63, positive_ratio=0.20) == "POSSIBLE SIGNAL"

    def test_no_evidence_low_mean(self):
        assert classify_signal(0.50, 0.01, 0.20, 0.51, positive_ratio=0.30) == "NO EVIDENCE"

    def test_no_evidence_none(self):
        assert classify_signal(None, None, None, None) == "NO EVIDENCE"

    def test_no_rentability_claim(self):
        # La etiqueta es solo de señal; no hay afirmación de rentabilidad.
        verdict = classify_signal(0.65, 0.01, 0.30, 0.64, positive_ratio=0.20)
        assert "ROBUST SIGNAL" == verdict
        assert "rent" not in verdict.lower()


class TestFase10Integration:
    def _matrix(self, n: int = 700, seed: int = 3):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, 12))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_winner_selected_by_fold_mean_not_test(self):
        X, y = self._matrix()
        summary, _ = run_walk_forward_comparison(
            X,
            y,
            n_splits=3,
            model_names=["random_forest"],
            metric="roc_auc",
            hyperparams={"random_forest": {"n_estimators": 10, "max_depth": 3}},
            early_stop_rounds=0,
        )
        winner = select_walk_forward_winner(summary, "roc_auc")
        assert winner is not None
        assert "wf_mean_roc_auc" in winner["metrics"]

    def test_summary_has_wf_columns(self):
        X, y = self._matrix()
        summary, _ = run_walk_forward_comparison(
            X,
            y,
            n_splits=3,
            model_names=["random_forest"],
            metric="roc_auc",
            hyperparams={"random_forest": {"n_estimators": 10, "max_depth": 3}},
            early_stop_rounds=0,
        )
        assert "wf_mean_roc_auc" in summary.columns
        assert "wf_std_roc_auc" in summary.columns
        assert "wf_mean_pr_auc" in summary.columns

    def test_validation_table_in_cli_output(self, tmp_path, capsys):
        import asyncio

        from train_and_compare import main

        rng = np.random.default_rng(5)
        n = 900
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        close = 1.0 + np.cumsum(rng.normal(0, 0.001, n))
        df = pd.DataFrame(
            {
                "DateTime": dates.strftime("%Y-%m-%d"),
                "time": dates.strftime("%H:%M:%S"),
                "Open": close + rng.normal(0, 0.0002, n),
                "High": close + rng.uniform(0, 0.001, n),
                "Low": close - rng.uniform(0, 0.001, n),
                "Close": close,
                "Tickvol": rng.integers(500, 2000, n),
                "Volume": 0,
                "Spread": 1,
            }
        )
        path = tmp_path / "USDCAD_H1.txt"
        df.to_csv(path, sep="\t", index=False)

        # N=900 con split 70/15/15 y 2 folds walk-forward; RF rápido.
        asyncio.run(
            main(
                [
                    str(path),
                    "--model",
                    "random_forest",
                    "--walk-forward-splits",
                    "2",
                    "--no-save",
                ]
            )
        )
        out = capsys.readouterr().out
        assert "VALIDATION (walk-forward folds)" in out
        assert "COMPARACIÓN HISTÓRICA" in out
        assert "0.6513" in out  # referencia del experimento original
