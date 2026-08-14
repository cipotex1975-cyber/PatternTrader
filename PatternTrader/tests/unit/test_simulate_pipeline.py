from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.market.candles.models import Candle
from simulate_pipeline import load_candles, run_simulation
from tests.unit.test_pipeline import double_top_closes


def _write_ohlcv_csv(path, closes, *, delimiter=",", prefix=""):
    rows = [f"{prefix}DateTime,Open,High,Low,Close,Tickvol,Volume"]
    start = datetime(2020, 1, 1, 0, 0, 0)
    prev = closes[0]
    for i, close in enumerate(closes):
        volume = 2000 if i >= len(closes) - 5 else 1000
        rows.append(
            f"{start + timedelta(hours=i)}.{i:06d},{prev},"
            f"{max(prev, close) * 1.002:.5f},{min(prev, close) * 0.998:.5f},"
            f"{close},{volume},{volume}"
        )
        prev = close
    path.write_text("\n".join(rows).replace(",", delimiter))
    return path


def _double_top_csv(tmp_path, *, delimiter=",", prefix=""):
    return _write_ohlcv_csv(
        tmp_path / f"{prefix}USDCAD_H1_test.txt", double_top_closes(), delimiter=delimiter
    )


class TestLoadCandles:
    def test_parses_tab_delimited_mt4_format(self, tmp_path):
        path = _double_top_csv(tmp_path, delimiter="\t")
        candles = load_candles(str(path), "USDCAD", "H1")

        assert len(candles) == len(double_top_closes())
        assert all(isinstance(c, Candle) for c in candles)
        assert candles[0].data.timestamp < candles[-1].data.timestamp
        assert candles[-1].data.close == double_top_closes()[-1]

    def test_parses_comma_separated_csv(self, tmp_path):
        path = _double_top_csv(tmp_path, delimiter=",")
        candles = load_candles(str(path), "USDCAD", "H1")

        assert len(candles) == len(double_top_closes())
        assert candles[-1].data.close == double_top_closes()[-1]
        assert candles[-1].data.volume == 2000

    @pytest.mark.asyncio
    async def test_raises_when_warmup_exceeds_candles(self, tmp_path):
        path = _double_top_csv(tmp_path)
        with pytest.raises(ValueError, match="--warmup"):
            await run_simulation(str(path), warmup=500, step=10, use_db=False)


@pytest.mark.asyncio
async def test_memory_simulation_replays_and_detects(tmp_path):
    path = _double_top_csv(tmp_path)
    report = await run_simulation(
        str(path),
        warmup=20,
        step=8,
        use_db=False,
        force_telegram=False,
    )

    assert report["use_db"] is False
    assert report["total_candles"] == len(double_top_closes())
    assert report["ticks"] >= 2
    assert report["events"]["PATTERN_DETECTED"] >= 1
    assert "double_top" in report["patterns_by_type"]
    assert report["symbol"] == "USDCAD"
    assert report["timeframe"] == "H1"


@pytest.mark.asyncio
async def test_per_symbol_ml_model_loaded(tmp_path):
    from tests.unit.test_scoring import _train_and_save_winner

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _train_and_save_winner(model_dir, "USDCAD")

    path = _double_top_csv(tmp_path)
    report = await run_simulation(
        str(path),
        warmup=20,
        step=8,
        use_db=False,
        model_dir=str(model_dir),
    )

    assert report["ml_models"] == {"USDCAD": "random_forest"}
