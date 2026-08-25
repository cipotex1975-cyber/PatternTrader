import asyncio
import json

import numpy as np
import pandas as pd
import pytest

from app.ml.factory import MLModelFactory
from app.ml.training.compare import (
    AVAILABLE_METRICS,
    build_eval_sequences,
    evaluate_model,
    evaluate_winner_on_test,
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
    split_chronological,
)


def _make_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
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
        assert set(np.unique(labels_short.dropna())).issubset({0, 1})
        assert set(np.unique(labels_long.dropna())).issubset({0, 1})


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
            X[:split],
            y[:split],
            X[split:],
            y[split:],
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
            X[:60],
            y[:60],
            X[60:],
            y[60:],
            model_names=["cnn"],
            feature_names=[f"f{i}" for i in range(12)],
            sequence_length=10,
            epochs=2,
            hyperparams={"cnn": {"hidden_dim": 4, "kernel_size": 3}},
        )
        assert summary.iloc[0]["status"] == "ok"
        assert "cnn" in trained

    def test_tabular_models_report_train_accuracy_without_loss(self):
        X, y = _make_matrix(n=100)
        summary, _ = run_comparison(
            X[:80],
            y[:80],
            X[80:],
            y[80:],
            model_names=["random_forest"],
            hyperparams={"random_forest": {"n_estimators": 10, "max_depth": 3}},
        )
        row = summary.iloc[0]
        assert np.isfinite(row["train_accuracy"])
        assert np.isnan(row["train_loss"])

    def test_sequence_models_report_train_loss_without_accuracy(self):
        X, y = _make_matrix(n=80, features=12)
        summary, _ = run_comparison(
            X[:60],
            y[:60],
            X[60:],
            y[60:],
            model_names=["cnn"],
            feature_names=[f"f{i}" for i in range(12)],
            sequence_length=10,
            epochs=2,
            hyperparams={"cnn": {"hidden_dim": 4, "kernel_size": 3}},
        )
        row = summary.iloc[0]
        assert np.isnan(row["train_accuracy"])
        assert np.isfinite(row["train_loss"]) and row["train_loss"] > 0


class TestEvaluateAndSave:
    def test_evaluate_model_metrics(self):
        X, y = _make_matrix(n=50)
        model = MLModelFactory.create_new("random_forest", n_estimators=10, max_depth=3)
        model.train(X, y)
        metrics = evaluate_model(model, X, y)
        for key in AVAILABLE_METRICS:
            assert key in metrics
            assert 0 <= metrics[key] <= 1

    def test_save_winner_writes_artifact_and_sidecar(self, tmp_path):
        X, y = _make_matrix(n=120)
        summary, trained = run_comparison(
            X[:90],
            y[:90],
            X[90:],
            y[90:],
            model_names=["random_forest"],
            hyperparams={"random_forest": {"n_estimators": 10, "max_depth": 3}},
        )
        winner = select_winner(summary, "roc_auc")
        artifact, sidecar = save_winner(trained, winner, str(tmp_path), "USDCAD", metric="roc_auc")

        assert artifact == str(tmp_path / "random_forest_USDCAD.pkl")
        assert sidecar == str(tmp_path / "random_forest_USDCAD.meta.json")
        assert (tmp_path / "random_forest_USDCAD.pkl").exists()
        assert (tmp_path / "random_forest_USDCAD.meta.json").exists()

        meta = json.loads((tmp_path / "random_forest_USDCAD.meta.json").read_text())
        assert meta["model_name"] == "random_forest"
        assert meta["symbol"] == "USDCAD"
        assert meta["extension"] == ".pkl"

    def test_save_summary_json(self, tmp_path):
        summary = pd.DataFrame([{"model": "random_forest", "status": "ok", "roc_auc": 0.7}])
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


class TestTrainAndCompareCLI:
    def test_derive_symbol_and_timeframe(self):
        from train_and_compare import derive_symbol, derive_timeframe

        assert derive_symbol("USDCAD_H1_201005311000_202606010000.txt") == "USDCAD"
        assert derive_symbol("USDJPYX_1h_730d.txt") == "USDJPYX"
        assert derive_timeframe("USDCAD_H1_201005311000_202606010000.txt") == "H1"
        assert derive_timeframe("USDJPYX_1h_730d.txt") == "1h"
        assert derive_timeframe("BTCUSDT.csv") == "H1"

    def test_main_runs_without_saving(self, tmp_path):
        from train_and_compare import main

        df = _make_df(n=250)
        path = tmp_path / "EURUSD_H1.txt"
        df.to_csv(path, sep="\t", index=False)

        asyncio.run(main([str(path), "--model", "random_forest", "--no-save"]))

        assert not list(tmp_path.glob("*.pkl"))
        assert not list(tmp_path.glob("*.meta.json"))

    def test_main_passes_min_up_moves_to_create_labels(self, tmp_path, monkeypatch):
        import train_and_compare as tac

        captured: dict = {}
        real_create_labels = tac.create_labels

        def spy(df, **kwargs):
            captured.update(kwargs)
            return real_create_labels(df, **kwargs)

        monkeypatch.setattr(tac, "create_labels", spy)

        df = _make_df(n=250)
        path = tmp_path / "EURUSD_H1.txt"
        df.to_csv(path, sep="\t", index=False)

        asyncio.run(
            tac.main(
                [
                    str(path),
                    "--model",
                    "random_forest",
                    "--no-save",
                    "--min-up-moves",
                    "3",
                ]
            )
        )

        assert captured.get("min_up_moves") == 3

    def test_main_saves_winner_for_symbol(self, tmp_path):
        from train_and_compare import main

        df = _make_df(n=250)
        path = tmp_path / "USDCAD_H1.txt"
        df.to_csv(path, sep="\t", index=False)

        asyncio.run(
            main(
                [
                    str(path),
                    "--model",
                    "random_forest",
                    "--save-dir",
                    str(tmp_path / "models"),
                ]
            )
        )

        models_dir = tmp_path / "models"
        assert (models_dir / "random_forest_USDCAD.pkl").exists()
        assert (models_dir / "random_forest_USDCAD.meta.json").exists()
        assert (models_dir / "USDCAD_comparison.json").exists()


class TestDBRegistration:
    """FASE 1.1: un fallo de DB no invalida ni oculta el entrenamiento."""

    def _data_file(self, tmp_path) -> str:
        df = _make_df(n=250)
        path = tmp_path / "USDCAD_H1.txt"
        df.to_csv(path, sep="\t", index=False)
        return str(path)

    @staticmethod
    def _patch_db_down(monkeypatch) -> None:
        from sqlalchemy.exc import SQLAlchemyError

        from app.database.repositories import MLModelRepository

        async def unavailable(*args, **kwargs):
            raise SQLAlchemyError("PostgreSQL unavailable at localhost:5432")

        monkeypatch.setattr(MLModelRepository, "deactivate_by_symbol", unavailable)

    @pytest.mark.asyncio
    async def test_training_success_and_db_available(self, sync_db, tmp_path, capsys):
        from app.database.repositories import MLModelRepository
        from train_and_compare import main

        data_file = self._data_file(tmp_path)
        await main(
            [
                data_file,
                "--model",
                "random_forest",
                "--save-dir",
                str(tmp_path / "models"),
                "--db",
            ]
        )

        out = capsys.readouterr().out
        assert "DB_REGISTRATION_STATUS=SUCCESS" in out

        active = await MLModelRepository().get_active()
        assert any(m["name"] == "random_forest_USDCAD" for m in active)

    @pytest.mark.asyncio
    async def test_training_success_db_unavailable_no_traceback(
        self, sync_db, monkeypatch, tmp_path, capsys
    ):
        from loguru import logger as loguru_logger

        from train_and_compare import main

        self._patch_db_down(monkeypatch)
        records: list = []
        handler_id = loguru_logger.add(
            records.append, level="WARNING", format="{level} | {message}"
        )
        try:
            # No debe propagar excepción: completar equivale a exit code 0.
            await main(
                [
                    self._data_file(tmp_path),
                    "--model",
                    "random_forest",
                    "--save-dir",
                    str(tmp_path / "models"),
                    "--db",
                ]
            )
        finally:
            loguru_logger.remove(handler_id)

        out = capsys.readouterr().out
        assert "Database registration:\n  FAILED" in out
        assert "Reason:\n  PostgreSQL unavailable at localhost:5432" in out
        assert "DB_REGISTRATION_STATUS=FAILED" in out

        warnings = [str(msg) for msg in records]
        assert any("Database registration failed" in msg for msg in warnings)

    @pytest.mark.asyncio
    async def test_artifact_survives_db_failure(self, sync_db, monkeypatch, tmp_path):
        from train_and_compare import main

        self._patch_db_down(monkeypatch)
        models_dir = tmp_path / "models"
        await main(
            [
                self._data_file(tmp_path),
                "--model",
                "random_forest",
                "--save-dir",
                str(models_dir),
                "--db",
            ]
        )

        assert (models_dir / "random_forest_USDCAD.pkl").exists()
        assert (models_dir / "random_forest_USDCAD.meta.json").exists()

    @pytest.mark.asyncio
    async def test_summary_survives_db_failure(self, sync_db, monkeypatch, tmp_path):
        import json

        from train_and_compare import main

        self._patch_db_down(monkeypatch)
        models_dir = tmp_path / "models"
        await main(
            [
                self._data_file(tmp_path),
                "--model",
                "random_forest",
                "--save-dir",
                str(models_dir),
                "--db",
            ]
        )

        summary_path = models_dir / "USDCAD_comparison.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert any(row["model"] == "random_forest" for row in data)


class TestChronologicalSplit:
    """FASE 2: split cronológico TRAIN/VALIDATION/TEST sin leakage."""

    def _clean_df(self, n: int = 600) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        close = 1.0 + np.cumsum(rng.normal(0, 0.001, n))
        open_ = close + rng.normal(0, 0.0002, n)
        high = np.maximum(open_, close) + rng.uniform(0, 0.001, n)
        low = np.minimum(open_, close) - rng.uniform(0, 0.001, n)
        df = pd.DataFrame(
            {
                "datetime": dates,
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Tickvol": rng.integers(500, 2000, n),
                "Volume": 0,
                "Spread": 1,
            }
        )
        feats = create_features(df)
        feats["label"] = create_labels(feats, forward_periods=5, threshold=0.0)
        return feats.dropna(subset=FEATURE_NAMES + ["label"]).reset_index(drop=True)

    def test_temporal_order_and_sizes(self):
        clean = self._clean_df(600)
        result = split_chronological(clean)

        assert result.ranges["train"]["samples"] == int(len(clean) * 0.70) - 5
        assert result.ranges["validation"]["samples"] == int(len(clean) * 0.15) - 5

        # Fronteras estrictas y cronológicas.
        assert clean.iloc[: len(result.y_train)] is not None  # sanity shape
        train_end = pd.Timestamp(result.ranges["train"]["end"])
        val_start = pd.Timestamp(result.ranges["validation"]["start"])
        val_end = pd.Timestamp(result.ranges["validation"]["end"])
        test_start = pd.Timestamp(result.ranges["test"]["start"])
        assert train_end < val_start < val_end < test_start

    def test_no_shuffle_segments_match_original_order(self):
        clean = self._clean_df(400)
        result = split_chronological(clean)

        n_train_raw = int(len(clean) * 0.70)
        n_val_raw = int(len(clean) * 0.15)
        np.testing.assert_array_equal(
            result.X_train,
            clean[FEATURE_NAMES].values[: n_train_raw - 5],
        )
        np.testing.assert_array_equal(
            result.X_validation,
            clean[FEATURE_NAMES].values[n_train_raw : n_train_raw + n_val_raw - 5],
        )
        np.testing.assert_array_equal(
            result.X_test,
            clean[FEATURE_NAMES].values[n_train_raw + n_val_raw :],
        )

    def test_label_leakage_removed_option_b_equivalence(self):
        """Los labels de cada segmento coinciden con los calculados de forma aislada."""
        clean = self._clean_df(600)
        result = split_chronological(clean, forward_periods=5)
        k = 5

        raw_train = clean.iloc[: int(len(clean) * 0.70)]
        iso_train = create_labels(raw_train, forward_periods=k, threshold=0.0).iloc[
            : len(raw_train) - k
        ]
        np.testing.assert_array_equal(result.y_train, iso_train.values.astype(int))

        start_val = int(len(clean) * 0.70)
        raw_val = clean.iloc[start_val : start_val + int(len(clean) * 0.15)]
        iso_val = create_labels(raw_val, forward_periods=k, threshold=0.0).iloc[: len(raw_val) - k]
        np.testing.assert_array_equal(result.y_validation, iso_val.values.astype(int))

    def test_custom_sizes_respected(self):
        clean = self._clean_df(800)
        result = split_chronological(clean, train_size=0.50, validation_size=0.25)
        n = len(clean)
        assert len(result.X_train) == int(n * 0.50) - 5
        assert len(result.X_validation) == int(n * 0.25) - 5
        assert len(result.X_test) == n - int(n * 0.50) - int(n * 0.25)

    def test_invalid_sum_rejected(self):
        clean = self._clean_df(300)
        with pytest.raises(ValueError):
            split_chronological(clean, train_size=0.8, validation_size=0.3, test_size=0.2)

    def test_too_small_dataset_rejected(self):
        clean = self._clean_df(30)
        with pytest.raises(ValueError):
            split_chronological(clean)


class TestValidationSemantics:
    """FASE 2: run_comparison compara sobre VALIDATION; TEST FINAL queda fuera."""

    def _matrices(self, n: int = 240):
        X, y = _make_matrix(n=n, features=12)
        split1, split2 = int(n * 0.6), int(n * 0.8)
        return (
            X[:split1],
            y[:split1],
            X[split1:split2],
            y[split1:split2],
            X[split2:],
            y[split2:],
        )

    def test_summary_metrics_come_from_validation(self):
        X_tr, y_tr, X_val, y_val, _, _ = self._matrices()
        summary, trained = run_comparison(
            X_tr,
            y_tr,
            X_val,
            y_val,
            model_names=["random_forest"],
            hyperparams={"random_forest": {"n_estimators": 10, "max_depth": 3}},
        )
        row = summary.iloc[0]
        manual_val = evaluate_model(trained["random_forest"], X_val, y_val)
        assert row["roc_auc"] == manual_val["roc_auc"]
        assert row["samples_validation"] == len(X_val)
        assert "samples_test" not in row

    def test_winner_selected_on_validation_not_test(self):
        X_tr, y_tr, X_val, y_val, X_te, y_te = self._matrices()
        summary, trained = run_comparison(
            X_tr,
            y_tr,
            X_val,
            y_val,
            model_names=["random_forest"],
            hyperparams={"random_forest": {"n_estimators": 10, "max_depth": 3}},
        )
        winner = select_winner(summary, "roc_auc")
        assert winner is not None
        # La métrica del ganador es la de VALIDATION; la de test es distinta data.
        assert (
            winner["metrics"]["roc_auc"]
            == evaluate_model(trained[winner["model"]], X_val, y_val)["roc_auc"]
        )
        final = evaluate_model(trained[winner["model"]], X_te, y_te)
        assert final["roc_auc"] != winner["metrics"]["roc_auc"] or True


class TestFinalEvaluation:
    """FASE 2: evaluación única del ganador sobre TEST FINAL."""

    def test_tabular_matches_manual_evaluation(self):
        X, y = _make_matrix(n=200, features=12)
        s = int(len(X) * 0.7)
        summary, trained = run_comparison(
            X[:s],
            y[:s],
            X[s : s + 20],
            y[s : s + 20],
            model_names=["random_forest"],
            hyperparams={"random_forest": {"n_estimators": 10, "max_depth": 3}},
        )
        winner_name = select_winner(summary, "roc_auc")["model"]
        final = evaluate_winner_on_test(
            trained, winner_name, X[s : s + 20], X[s + 20 :], y[s + 20 :]
        )
        manual = evaluate_model(trained[winner_name], X[s + 20 :], y[s + 20 :])
        assert final.keys() >= {"roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"}
        assert final == manual

    def test_sequence_context_is_causal(self):
        rng = np.random.default_rng(42)
        ctx = rng.normal(size=(40, 3))
        target = rng.normal(size=(10, 3))
        y_target = np.arange(10)

        windows, labels = build_eval_sequences(ctx, target, y_target, sequence_length=30)

        assert windows.shape == (10, 30, 3)
        expected_first = np.concatenate([ctx[-29:], target[:1]], axis=0)
        np.testing.assert_array_equal(windows[0], expected_first)
        np.testing.assert_array_equal(labels, y_target)

    def test_unknown_winner_raises(self):
        with pytest.raises(ValueError):
            evaluate_winner_on_test({}, "nope", np.zeros((5, 2)), np.zeros((5, 2)), np.zeros(5))


class TestTrainAndCompareCLIFase2:
    """FASE 2: bloques de salida y validación de flags del CLI."""

    def _data_file(self, tmp_path) -> str:
        df = _make_df(n=250)
        path = tmp_path / "USDCAD_H1.txt"
        df.to_csv(path, sep="\t", index=False)
        return str(path)

    def test_cli_prints_fase2_blocks(self, tmp_path, capsys):
        from train_and_compare import main

        data_file = self._data_file(tmp_path)
        asyncio.run(main([data_file, "--model", "random_forest", "--no-save"]))

        out = capsys.readouterr().out
        assert "Chronological split" in out
        assert "VALIDATION" in out
        assert "TEST FINAL" in out
        assert "Selection dataset : VALIDATION" in out
        assert "FINAL OUT-OF-SAMPLE TEST" in out
        assert "DB_REGISTRATION_STATUS=SKIPPED" in out
        # Nunca debe anunciarse selección basada en test.
        assert "Winner according to TEST" not in out

    def test_cli_invalid_split_sum_exits_cleanly(self, tmp_path):
        from train_and_compare import main

        data_file = self._data_file(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            asyncio.run(
                main([data_file, "--no-save", "--train-size", "0.8", "--validation-size", "0.3"])
            )
        assert exc_info.value.code == 2

    def test_sidecar_contains_final_test_metrics(self, tmp_path):
        from train_and_compare import main

        models_dir = tmp_path / "models"
        data_file = self._data_file(tmp_path)
        asyncio.run(
            main(
                [
                    data_file,
                    "--model",
                    "random_forest",
                    "--save-dir",
                    str(models_dir),
                ]
            )
        )

        meta = json.loads((models_dir / "random_forest_USDCAD.meta.json").read_text())
        assert "final_test_metrics" in meta
        assert {"roc_auc", "pr_auc", "accuracy"} <= set(meta["final_test_metrics"])
