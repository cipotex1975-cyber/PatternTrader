import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from app.market.candles.models import Candle, CandleData
from app.ml.training.compare import run_comparison, save_winner, select_winner
from app.patterns.base_pattern import PatternResult, PatternType
from app.scoring.engine import ScoringEngine


def create_test_pattern():
    return PatternResult(
        pattern_name="double_top",
        pattern_type=PatternType.REVERSAL,
        symbol="BTCUSDT",
        timeframe="1h",
        confidence=0.85,
        health=90.0,
        key_levels={"neckline": 50000, "peak1": 52000, "peak2": 51800},
    )


def create_test_candles():
    candles = []
    for i in range(30):
        candles.append(
            Candle(
                symbol="BTCUSDT",
                timeframe="1h",
                data=CandleData(
                    timestamp=datetime.now(timezone.utc),
                    open=50000 + i * 10,
                    high=50100 + i * 10,
                    low=49900 + i * 10,
                    close=50050 + i * 10,
                    volume=1000 + i * 100,
                ),
            )
        )
    return candles


def test_scoring_engine_initialization():
    engine = ScoringEngine()
    assert engine is not None


def test_scoring_calculates_score():
    engine = ScoringEngine()
    pattern = create_test_pattern()
    candles = create_test_candles()
    indicators = {
        "rsi": 65,
        "macd": 100,
        "macd_signal": 90,
        "ema_21": 50100,
        "ema_50": 50000,
        "atr": 200,
        "volume": 1500,
    }

    result = engine.calculate_score(pattern, indicators, candles)
    assert 0 <= result.total_score <= 100
    assert result.grade is not None
    assert 0 <= result.confidence <= 1


def test_scoring_components():
    engine = ScoringEngine()
    pattern = create_test_pattern()
    indicators = {"rsi": 50, "macd": 0, "macd_signal": 0}

    result = engine.calculate_score(pattern, indicators)
    assert len(result.components) > 0
    assert all(c.weight > 0 for c in result.components)


@pytest.mark.asyncio
async def test_scoring_uses_knowledge_model_when_attached():
    from app.learning.models import LearningMode
    from app.learning.repository import MemoryKnowledgeRepository
    from app.learning.service import LearningService

    svc = LearningService(repository=MemoryKnowledgeRepository(), mode=LearningMode.ONLINE)
    for i in range(12):
        win = i % 2 == 0
        ind = {"rsi": 80.0 if win else 20.0, "atr": 100.0}
        await svc.record_trade(
            {
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "pattern_name": "double_top",
                "direction": "SHORT",
                "entry_price": 50000,
                "exit_price": 49500 if win else 50500,
                "pnl": 500 if win else -500,
            },
            indicators=ind,
        )
    assert svc.is_trained

    engine = ScoringEngine()
    engine.attach_knowledge(svc)

    pattern = create_test_pattern()
    candles = create_test_candles()
    result = engine.calculate_score(pattern, {"rsi": 50}, candles)

    ml = next(c for c in result.components if c.name == "ml_history")
    assert 0 <= ml.score <= 100
    assert 0 <= result.confidence <= 1


def _make_matrix(n: int = 120, features: int = 12, seed: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _train_and_save_winner(save_dir, symbol, model_name="random_forest"):
    X, y = _make_matrix(n=120)
    split = 90
    summary, trained = run_comparison(
        X[:split],
        y[:split],
        X[split:],
        y[split:],
        model_names=[model_name],
        hyperparams={model_name: {"n_estimators": 20, "max_depth": 4}},
    )
    winner = select_winner(summary, "roc_auc")
    return save_winner(trained, winner, str(save_dir), symbol, metric="roc_auc")


class TestScoringPerSymbolModel:
    def test_loads_per_symbol_winner(self, tmp_path):
        artifact, sidecar = _train_and_save_winner(tmp_path, "USDCAD")

        assert Path(artifact).exists()
        assert Path(sidecar).exists()

        engine = ScoringEngine(model_path=str(tmp_path))
        assert engine._ml_model is None  # sin fallback genérico en este directorio

        model = engine._load_ml_model_for_symbol("USDCAD")
        assert model is not None
        assert model.is_trained
        assert model.name == "random_forest"

        # Caché: la segunda llamada devuelve la misma instancia.
        assert engine._load_ml_model_for_symbol("USDCAD") is model

    def test_unknown_symbol_returns_none(self, tmp_path):
        _train_and_save_winner(tmp_path, "USDCAD")
        engine = ScoringEngine(model_path=str(tmp_path))
        assert engine._load_ml_model_for_symbol("EURUSD") is None

    def test_per_symbol_chooses_newest_sidecar(self, tmp_path):
        _, sidecar_rf = _train_and_save_winner(tmp_path, "USDCAD", "random_forest")
        _, sidecar_xgb = _train_and_save_winner(tmp_path, "USDCAD", "xgboost")

        meta_rf = json.loads(Path(sidecar_rf).read_text())
        meta_rf["trained_at"] = "2020-01-01T00:00:00+00:00"
        Path(sidecar_rf).write_text(json.dumps(meta_rf))

        meta_xgb = json.loads(Path(sidecar_xgb).read_text())
        meta_xgb["trained_at"] = "2026-08-01T00:00:00+00:00"
        Path(sidecar_xgb).write_text(json.dumps(meta_xgb))

        engine = ScoringEngine(model_path=str(tmp_path))
        model = engine._load_ml_model_for_symbol("USDCAD")
        assert model is not None
        assert model.name == "xgboost"

    def test_falls_back_when_newest_artifact_missing(self, tmp_path):
        _, sidecar_rf = _train_and_save_winner(tmp_path, "USDCAD", "random_forest")
        _, sidecar_xgb = _train_and_save_winner(tmp_path, "USDCAD", "xgboost")

        meta_xgb = json.loads(Path(sidecar_xgb).read_text())
        meta_xgb["trained_at"] = "2026-08-01T00:00:00+00:00"
        Path(sidecar_xgb).write_text(json.dumps(meta_xgb))

        meta_rf = json.loads(Path(sidecar_rf).read_text())
        meta_rf["trained_at"] = "2020-01-01T00:00:00+00:00"
        Path(sidecar_rf).write_text(json.dumps(meta_rf))

        # El artefacto más reciente apunta a un archivo inexistente → degradar.
        Path(sidecar_xgb).parent.joinpath(
            f"{meta_xgb['model_name']}_USDCAD{meta_xgb['extension']}"
        ).unlink()

        engine = ScoringEngine(model_path=str(tmp_path))
        model = engine._load_ml_model_for_symbol("USDCAD")
        assert model is not None
        assert model.name == "random_forest"

    def test_per_symbol_model_used_in_score(self, tmp_path):
        _train_and_save_winner(tmp_path, "USDCAD")
        engine = ScoringEngine(model_path=str(tmp_path))

        pattern = PatternResult(
            pattern_name="double_top",
            pattern_type=PatternType.REVERSAL,
            symbol="USDCAD",
            timeframe="1h",
            confidence=0.85,
            key_levels={"neckline": 1.05, "peak1": 1.06, "peak2": 1.058},
        )
        result = engine.calculate_score(pattern, {"rsi": 50}, create_test_candles())

        ml = next(c for c in result.components if c.name == "ml_history")
        assert 0 <= ml.score <= 100
        assert engine._symbol_models.get("USDCAD") is not None
