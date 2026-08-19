from pathlib import Path

from app.patterns.base_pattern import PatternResult, PatternType
from run_backtest import (
    _derive_sl_tp,
    get_pair_config,
    load_candles,
    main,
    resolve_data_path,
    resolve_excluded_patterns,
)

DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "datos_test"
    / "USDCAD_H1_201005311000_202606010000.txt"
)


def _pattern(name: str, key_levels: dict) -> PatternResult:
    return PatternResult(
        id="test",
        symbol="USDCAD",
        timeframe="H1",
        pattern_name=name,
        pattern_type=PatternType.REVERSAL,
        confidence=0.8,
        health=80.0,
        score=0.9,
        entry_price=1.0,
        stop_loss=0.0,
        take_profit=0.0,
        risk_reward_ratio=2.0,
        key_levels=key_levels,
        status="detected",
    )


class TestLoadCandles:
    def test_loads_sorted_timezone_aware_candles(self):
        candles = load_candles(str(DATA_FILE), "USDCAD", "H1", max_candles=500)

        assert len(candles) == 500
        assert all(c.symbol == "USDCAD" for c in candles)
        assert all(c.timeframe == "H1" for c in candles)
        timestamps = [c.data.timestamp for c in candles]
        assert timestamps == sorted(timestamps)
        assert all(ts.tzinfo is not None for ts in timestamps)
        assert all(c.data.open > 0 and c.data.close > 0 for c in candles)

    def test_max_candles_takes_tail(self):
        full = load_candles(str(DATA_FILE), "USDCAD", "H1")
        tail = load_candles(str(DATA_FILE), "USDCAD", "H1", max_candles=100)

        assert len(full) > len(tail)
        assert tail[0].data.timestamp == full[-100].data.timestamp
        assert tail[-1].data.timestamp == full[-1].data.timestamp


class TestPairConfig:
    def test_merges_pair_with_defaults(self):
        cfg = get_pair_config("USDCAD")

        assert cfg["window"] >= 1
        assert cfg["step"] >= 1
        assert cfg["max_patterns"] >= 1
        assert "sl_tp" in cfg and isinstance(cfg["sl_tp"], dict)

    def test_unknown_pair_uses_defaults(self):
        assert get_pair_config("NOPE") == get_pair_config(None)


class TestDeriveSlTp:
    def test_double_top_default_uses_peak(self):
        sl, tp = _derive_sl_tp(
            _pattern("double_top", {"peak1": 1.4, "peak2": 1.38, "target": 1.1}),
            {},
        )
        assert sl == 1.4 * 1.002
        assert tp == 1.1

    def test_double_top_neckline_method(self):
        sl, tp = _derive_sl_tp(
            _pattern(
                "double_top",
                {"peak1": 1.4, "peak2": 1.38, "neckline": 1.2, "target": 1.1},
            ),
            {"double_top": {"sl_method": "neckline"}},
        )
        assert sl == 1.2 * 1.002

    def test_double_bottom_uses_trough(self):
        sl, tp = _derive_sl_tp(
            _pattern(
                "double_bottom",
                {"trough1": 1.2, "trough2": 1.18, "neckline": 1.4, "target": 1.6},
            ),
            {},
        )
        assert sl == 1.18 * 0.998
        assert tp == 1.6

    def test_bull_flag(self):
        sl, tp = _derive_sl_tp(
            _pattern("bull_flag", {"flag_low": 1.3, "target": 1.6}),
            {},
        )
        assert sl == 1.3 * 0.998
        assert tp == 1.6

    def test_bear_flag(self):
        sl, tp = _derive_sl_tp(
            _pattern("bear_flag", {"flag_high": 1.5, "target": 1.2}),
            {},
        )
        assert sl == 1.5 * 1.002
        assert tp == 1.2

    def test_head_and_shoulders_default_uses_head(self):
        sl, tp = _derive_sl_tp(
            _pattern(
                "head_and_shoulders",
                {"head": 1.6, "neckline": 1.3, "target": 1.0},
            ),
            {},
        )
        assert sl == 1.6 * 1.002

    def test_inverse_head_and_shoulders_uses_neckline(self):
        sl, tp = _derive_sl_tp(
            _pattern(
                "inverse_head_and_shoulders",
                {"neckline": 1.4, "target": 1.7},
            ),
            {},
        )
        assert sl == 1.4 * 0.998

    def test_unknown_pattern_returns_none(self):
        assert _derive_sl_tp(_pattern("nope", {"a": 1}), {}) == (None, None)

    def test_missing_key_levels_returns_none(self):
        assert _derive_sl_tp(_pattern("double_top", {}), {}) == (None, None)


class TestResolveDataPath:
    def test_auto_discovers_file(self):
        path = resolve_data_path("USDCAD", "H1")
        assert path is not None
        assert path.exists()

    def test_explicit_path_wins(self):
        assert resolve_data_path("USDCAD", "H1", str(DATA_FILE)) == DATA_FILE

    def test_missing_symbol_returns_none(self):
        assert resolve_data_path("NOPE", "H1") is None


class TestResolveExcluded:
    def test_cli_arg_takes_precedence(self):
        assert resolve_excluded_patterns("bear_flag,double_top", {"exclude": ["x"]}) == {
            "bear_flag",
            "double_top",
        }

    def test_config_used_when_no_arg(self):
        assert resolve_excluded_patterns(None, {"exclude": ["x"]}) == {"x"}

    def test_empty_without_arg_and_config(self):
        assert resolve_excluded_patterns(None, {}) == set()


class TestMainSmoke:
    def test_runs_backtest_with_capped_inputs(self, capsys):
        main(["--max-candles", "500", "--max-patterns", "5"])

        out = capsys.readouterr().out
        assert "RESULTADOS DEL BACKTEST" in out
        assert "METRICAS DE TRADING" in out
        assert "Velas cargadas: 500" in out
