"""Pruebas de FASE 9 — Reproducibilidad y Model Metadata.

Verifica:
- seed_all fija semillas de Python/NumPy/PyTorch (o es no-op con None).
- hash_file_sha256 es determinista y detecta cambios de contenido.
- get_git_commit_sha devuelve un SHA válido en repo git (o None fuera).
- get_software_versions incluye al menos python.
- build_model_sidecar_context produce todos los bloques de metadata de FASE 9.
- save_winner integra el sidecar_context en el .meta.json persistido.
"""

import hashlib
import json
import random

import numpy as np

from app.ml.training.compare import run_comparison, save_winner, select_winner
from app.ml.training.data import FEATURE_NAMES
from app.ml.training.reproducibility import (
    build_model_sidecar_context,
    get_git_commit_sha,
    get_software_versions,
    hash_file_sha256,
    seed_all,
)


class TestSeedAll:
    def test_seed_all_is_deterministic(self):
        # Con la misma semilla, la secuencia de random y numpy es idéntica.
        seed_all(42)
        py_a = [random.random() for _ in range(5)]
        np_a = np.random.random(5).tolist()

        seed_all(42)
        py_b = [random.random() for _ in range(5)]
        np_b = np.random.random(5).tolist()

        assert py_a == py_b
        assert np_a == np_b

    def test_seed_all_none_is_noop(self):
        # seed_all(None) no debe resettar las semillas.
        random.seed(123)
        np.random.seed(123)
        state_py = random.getstate()
        state_np = np.random.get_state()
        seed_all(None)
        assert random.getstate() == state_py
        # Comparar el estado numpy campo a campo (contiene arrays).
        nps = np.random.get_state()
        assert nps[0] == state_np[0]
        assert np.array_equal(nps[1], state_np[1])
        assert nps[2:] == state_np[2:]


class TestHashFileSha256:
    def test_deterministic(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text("contenido de ejemplo")
        assert hash_file_sha256(p) == hash_file_sha256(p)

    def test_detects_content_change(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text("contenido A")
        h1 = hash_file_sha256(p)
        p.write_text("contenido B")
        h2 = hash_file_sha256(p)
        assert h1 != h2

    def test_missing_file_returns_unknown(self, tmp_path):
        assert hash_file_sha256(tmp_path / "no_existe.txt") == "unknown"

    def test_matches_manual_sha256(self, tmp_path):
        p = tmp_path / "data.txt"
        content = b"bytes crudos para el hash"
        p.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert hash_file_sha256(p) == expected


class TestGitSha:
    def test_returns_hex_string(self):
        sha = get_git_commit_sha()
        # En un repo git devuelve un SHA hex de 40 caracteres (o None si no es git).
        if sha is not None:
            assert len(sha) == 40
            assert all(c in "0123456789abcdef" for c in sha)


class TestSoftwareVersions:
    def test_contains_python(self):
        versions = get_software_versions()
        assert "python" in versions
        assert isinstance(versions["python"], str)
        assert len(versions["python"]) > 0


class TestBuildModelSidecarContext:
    def _context(self, **overrides):
        base = dict(
            model_name="random_forest",
            data_path="/tmp/ejemplo.txt",
            ranges={
                "train": {"start": "2020-01-01", "end": "2021-01-01"},
                "validation": {"start": "2021-01-01", "end": "2021-06-01"},
                "test": {"start": "2021-06-01", "end": "2022-01-01"},
            },
            samples_total=1000,
            feature_names=list(FEATURE_NAMES),
            forward_periods=5,
            threshold=0.001,
            min_up_moves=2,
            preprocessing_type="none",
            sequence_length=30,
            epochs=10,
            settings=None,
            hyperparams=None,
            patience=5,
            early_stop_rounds=20,
            validation_metrics={"roc_auc": 0.65, "pr_auc": 0.3},
            test_metrics={"roc_auc": 0.62},
            selection_metric="roc_auc",
            selection_dataset="validation",
            random_seed=42,
        )
        base.update(overrides)
        return build_model_sidecar_context(**base)

    def test_contains_all_phase9_keys(self):
        ctx = self._context()
        expected = {
            "dataset",
            "features",
            "label",
            "preprocessing",
            "sequence",
            "training",
            "validation",
            "selection",
            "software",
            "git",
            "random_seed",
        }
        assert expected.issubset(set(ctx.keys()))

    def test_dataset_has_path_and_hash(self):
        ctx = self._context()
        ds = ctx["dataset"]
        assert "path" in ds
        assert "hash" in ds
        assert "start_datetime" in ds
        assert "end_datetime" in ds
        assert ds["samples"] == 1000
        # Archivo inexistente -> hash "unknown".
        assert ds["hash"] == "unknown"

    def test_label_block(self):
        ctx = self._context()
        assert ctx["label"] == {
            "forward_periods": 5,
            "threshold": 0.001,
            "min_up_moves": 2,
        }

    def test_training_contains_hyperparameters(self):
        ctx = self._context()
        assert "hyperparameters" in ctx["training"]
        assert ctx["training"]["epochs"] == 10

    def test_test_block_included_when_metrics(self):
        ctx = self._context(test_metrics={"roc_auc": 0.62})
        assert "test" in ctx
        assert ctx["test"]["metrics"]["roc_auc"] == 0.62

    def test_random_seed(self):
        assert self._context()["random_seed"] == 42


class TestSaveWinnerSidecarContext:
    def test_save_winner_includes_sidecar_context(self, tmp_path):
        X = np.random.default_rng(1).normal(0, 1, (120, 12))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        summary, trained = run_comparison(
            X[:90],
            y[:90],
            X[90:],
            y[90:],
            model_names=["random_forest"],
            hyperparams={"random_forest": {"n_estimators": 10, "max_depth": 3}},
        )
        winner = select_winner(summary, "roc_auc")
        sidecar_context = build_model_sidecar_context(
            model_name=str(winner["model"]),
            data_path=str(tmp_path / "data.txt"),
            ranges={
                "train": {"start": None, "end": None},
                "validation": {"start": None, "end": None},
                "test": {"start": None, "end": None},
            },
            samples_total=120,
            feature_names=list(FEATURE_NAMES),
            forward_periods=5,
            threshold=0.001,
            min_up_moves=2,
            preprocessing_type="none",
            sequence_length=30,
            epochs=10,
            settings=None,
            hyperparams=None,
            patience=5,
            early_stop_rounds=20,
            validation_metrics=winner.get("metrics", {}) or {},
            test_metrics=None,
            selection_metric="roc_auc",
            selection_dataset="validation",
            random_seed=42,
        )
        _, sidecar = save_winner(
            trained,
            winner,
            str(tmp_path),
            "USDCAD",
            metric="roc_auc",
            sidecar_context=sidecar_context,
        )

        meta = json.loads(open(sidecar, encoding="utf-8").read())
        assert "dataset" in meta
        assert "features" in meta
        assert "label" in meta
        assert "training" in meta
        assert "software" in meta
        assert "git" in meta
        assert meta["random_seed"] == 42
        # El contexto NO pisa el model_name/symbol base de save_winner.
        assert meta["model_name"] == "random_forest"
        assert meta["symbol"] == "USDCAD"

    def test_save_winner_without_context_unchanged(self, tmp_path):
        X = np.random.default_rng(1).normal(0, 1, (120, 12))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        summary, trained = run_comparison(
            X[:90],
            y[:90],
            X[90:],
            y[90:],
            model_names=["random_forest"],
            hyperparams={"random_forest": {"n_estimators": 10, "max_depth": 3}},
        )
        winner = select_winner(summary, "roc_auc")
        _, sidecar = save_winner(trained, winner, str(tmp_path), "USDCAD", metric="roc_auc")
        meta = json.loads(open(sidecar, encoding="utf-8").read())
        assert "dataset" not in meta
        assert "features" not in meta
