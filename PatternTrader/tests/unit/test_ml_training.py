import json

import numpy as np
import pandas as pd
import pytest

from app.ml.factory import MLModelFactory
from app.ml.training.compare import (
    AVAILABLE_METRICS,
    evaluate_model,
    run_comparison,
    save_summary,
    save_winner,
    select_winner,
)
from app.ml.training.data import (
    FEATURE_NAMES,
    build_sequences,
    create_features,
    create_labels,
    format_for_model,
    load_data,
    prepare_dataset,
)


def _make_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    open_ = np.linspace(1.0, 2.0, n) + rng.normal(0, 0.0001, n)
    close = open_ + rng.uniform(0.001, 0.005, n)
    high = close + rng.uniform(0.001, 0.005, n)
    low = open_ - rng.uniform(0, 0.001, n)
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


def _make_matrix(n: int = 120, features: int = 12, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


class TestLoadData:
    def test_sorts_and_renames(self, tmp_path):
        df = _make_df()
        path = tmp_path / "USDCAD_H1.txt"
        df.to_csv(path, sep="\t", index=False)

        loaded = load_data(str(path))
        assert "datetime" in loaded.columns
        assert "open" in loaded.columns and "close" in loaded.columns
        assert loaded["datetime"].is_monotonic_increasing
        assert not loaded["open"].isna().any()


class TestFeaturesAndLabels:
    def test_create_features_has_expected_columns(self):
        feats = create_features(_make_df())
        for col in FEATURE_NAMES:
            assert col in feats.columns, col

    def test_prepare_dataset_returns_clean_matrices(self):
        X, y, df = prepare_dataset(_make_df(n=300), forward_periods=5, threshold=0.0)
        assert X.shape[1] == len(FEATURE_NAMES)
        assert len(X) == len(y) == len(df)
        assert set(np.unique(y)).issubset({0, 1})

    def test_create_labels_respects_forward_periods(self):
        df = _make_df(n=200, seed=7)
        labels_short = create_labels(df, forward_periods=2, threshold=0.0)
        labels_long = create_labels(df, forward_periods=20, threshold=0.0)
        assert len(labels_short) == len(labels_long) == 200
        assert set(np.unique(labels_short)).issubset({0, 1})
        assert set(np.unique(labels_long)).issubset({0, 1})


class TestSequences:
    def test_build_sequences_shape(self):
        X, y = _make_matrix(n=120, features=12)
        Xs, ys = build_sequences(X, y, sequence_length=30)
        assert Xs.shape == (120 - 30 + 1, 30, 12)
        assert ys.shape == (120 - 30 + 1,)
        assert np.array_equal(Xs[-1], X[-30:])

    def test_build_sequences_requires_enough_samples(self):
        X, y = _make_matrix(n=10)
        with pytest.raises(ValueError):
            build_sequences(X, y, sequence_length=30)

    def test_format_for_model_tabular_vs_sequence(self):
        X, y = _make_matrix(n=120)
        X_tab, y_tab = format_for_model("random_forest", X, y)
        assert X_tab.ndim == 2
        assert np.array_equal(y_tab, y)

        X_seq, y_seq = format_for_model("lstm", X, y, sequence_length=30)
        assert X_seq.ndim == 3
        assert X_seq.shape[2] == 12
        assert y_seq.shape == (91,)


class TestComparison:
    def test_run_comparison_and_winner(self):
        X, y = _make_matrix(n=200)
        split = int(len(X) * 0.8)
        summary, trained = run_comparison(
            X[:split],
            y[:split],
            X[split:],
            y[split:],
            model_names=["random_forest", "xgboost"],
            feature_names=[f"f{i}" for i in range(12)],
            hyperparams={
                "random_forest": {"n_estimators": 20, "max_depth": 4},
                "xgboost": {"n_estimators": 20, "max_depth": 4},
            },
        )
        assert set(summary["model"]) == {"random_forest", "xgboost"}
        assert (summary["status"] == "ok").all()
        assert "random_forest" in trained

        winner = select_winner(summary, "roc_auc")
        assert winner is not None
        assert winner["model"] in {"random_forest", "xgboost"}
        assert 0 <= winner["metrics"]["roc_auc"] <= 1

    def test_run_comparison_uses_factory(self):
        X, y = _make_matrix(n=100)
        split = 80
        summary, _ = run_comparison(
            X[:split], y[:split], X[split:], y[split:],
            model_names=["random_forest"],
            hyperparams={"random_forest": {"n_estimators": 10, "max_depth": 3}},
        )
        assert summary.iloc[0]["status"] == "ok"
        # La factory singleton no debe quedar contaminada por el script.
        assert not MLModelFactory.get_loaded()

    def test_run_comparison_unknown_metric(self):
        X, y = _make_matrix(n=50)
        with pytest.raises(ValueError):
            run_comparison(X, y, X, y, metric="nope")

    def test_sequence_models_can_run(self):
        X, y = _make_matrix(n=80, features=12)
        summary, trained = run_comparison(
            X[:60], y[:60], X[60:], y[60:],
            model_names=["cnn"],
            feature_names=[f"f{i}" for i in range(12)],
            sequence_length=10,
            epochs=2,
            hyperparams={"cnn": {"hidden_dim": 4, "kernel_size": 3}},
        )
        assert summary.iloc[0]["status"] == "ok"
        assert "cnn" in trained


class TestEvaluateAndSave:
    def test_evaluate_model_metrics(self):
        X, y = _make_matrix(n=50)
        model = MLModelFactory.create_new(
            "random_forest", n_estimators=10, max_depth=3
        )
        model.train(X, y)
        metrics = evaluate_model(model, X, y)
        for key in AVAILABLE_METRICS:
            assert key in metrics
            assert 0 <= metrics[key] <= 1

    def test_save_winner_writes_artifact_and_sidecar(self, tmp_path):
        X, y = _make_matrix(n=120)
        summary, trained = run_comparison(
            X[:90], y[:90], X[90:], y[90:],
            model_names=["random_forest"],
            hyperparams={"random_forest": {"n_estimators": 10, "max_depth": 3}},
        )
        winner = select_winner(summary, "roc_auc")
        artifact, sidecar = save_winner(
            trained, winner, str(tmp_path), "USDCAD", metric="roc_auc"
        )

        assert artifact == str(tmp_path / "random_forest_USDCAD.pkl")
        assert sidecar == str(tmp_path / "random_forest_USDCAD.meta.json")
        assert (tmp_path / "random_forest_USDCAD.pkl").exists()
        assert (tmp_path / "random_forest_USDCAD.meta.json").exists()

        meta = json.loads((tmp_path / "random_forest_USDCAD.meta.json").read_text())
        assert meta["model_name"] == "random_forest"
        assert meta["symbol"] == "USDCAD"
        assert meta["extension"] == ".pkl"

    def test_save_summary_json(self, tmp_path):
        summary = pd.DataFrame(
            [{"model": "random_forest", "status": "ok", "roc_auc": 0.7}]
        )
        path = save_summary(summary, str(tmp_path), "EURUSD")
        data = json.loads(Path(path).read_text())
        assert data[0]["model"] == "random_forest"


from pathlib import Path  # noqa: E402


class TestSelectWinner:
    def test_skips_failed_and_na_models(self):
        summary = pd.DataFrame(
            [
                {"model": "a", "status": "ok", "roc_auc": None},
                {"model": "b", "status": "error", "roc_auc": 0.9},
                {"model": "c", "status": "ok", "roc_auc": 0.8},
            ]
        )
        winner = select_winner(summary, "roc_auc")
        assert winner["model"] == "c"
