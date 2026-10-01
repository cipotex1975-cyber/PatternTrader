from app.core.config.settings import get_settings
from app.core.constants.market import Timeframes, normalize_timeframe


def test_settings_loads():
    settings = get_settings()
    assert settings.application.name == "PatternTrader"
    assert settings.application.version == "0.1.0"


def test_database_settings():
    settings = get_settings()
    assert settings.database.host == "localhost"
    assert settings.database.port == 5432


def test_market_settings():
    settings = get_settings()
    assert (
        "BTCUSDT" in settings.market.default_timeframes
        or "USDCAD" in settings.market.default_symbols
    )
    assert len(settings.market.default_timeframes) > 0


def test_market_default_timeframes_are_canonical():
    """market.default_timeframes es la fuente única del pipeline: debe estar canónica.

    Si un timeframe se dejara en forma legacy ("H1"), el modelo ML entrenado
    para ese timeframe no casaría nunca con una señal live.
    """
    settings = get_settings()
    for tf in settings.market.default_timeframes:
        assert normalize_timeframe(tf) == tf, f"timeframe no canónico: {tf!r}"


def test_normalize_timeframe_letter_prefix():
    """Convención letra-primero de los ficheros OHLCV y de config/pairs.yaml."""
    assert normalize_timeframe("H1") == "1h"
    assert normalize_timeframe("M15") == "15m"
    assert normalize_timeframe("M5") == "5m"
    assert normalize_timeframe("D1") == "1d"
    assert normalize_timeframe("W1") == "1w"
    assert normalize_timeframe("H4") == "4h"


def test_normalize_timeframe_digit_prefix_and_case():
    assert normalize_timeframe("1h") == "1h"
    assert normalize_timeframe("15m") == "15m"
    assert normalize_timeframe("1H") == "1h"
    assert normalize_timeframe("1D") == "1d"
    assert normalize_timeframe("  1h  ") == "1h"


def test_normalize_timeframe_month_is_not_minute():
    """ "1M" es mes y "1m" es minuto: no deben colapsar en el mismo valor."""
    assert normalize_timeframe("MN1") == "1M"
    assert normalize_timeframe("1M") == "1M"
    assert normalize_timeframe("1m") == "1m"
    assert normalize_timeframe("M1") == "1m"
    assert normalize_timeframe("1M") != normalize_timeframe("1m")


def test_normalize_timeframe_is_idempotent():
    for raw in ("H1", "M15", "1h", "1M", "MN1", "4h", "basura", ""):
        once = normalize_timeframe(raw)
        assert normalize_timeframe(once) == once


def test_normalize_timeframe_preserves_enum_values():
    for tf in Timeframes:
        assert normalize_timeframe(tf.value) == tf.value


def test_normalize_timeframe_unknown_is_lowercased_not_defaulted():
    """Un valor irreconocible se hace visible en el llamador, no se sustituye."""
    assert normalize_timeframe("basura") == "basura"
    assert normalize_timeframe("XX") == "xx"


def test_timeframes_to_minutes_accepts_both_conventions():
    assert Timeframes.to_minutes("H1") == 60
    assert Timeframes.to_minutes("1h") == 60
    assert Timeframes.to_minutes("M15") == 15
    assert Timeframes.to_minutes("D1") == 1440
    assert Timeframes.to_minutes("MN1") == 43200
    # Antes reventaba con ValueError; ahora degrada a 1h sin propagar la excepción.
    assert Timeframes.to_minutes("basura") == 60


def test_scoring_weights():
    settings = get_settings()
    weights = settings.scoring.weights
    assert weights.pattern_structure == 0.35
    assert weights.volume == 0.20
    total = sum(
        [
            weights.pattern_structure,
            weights.volume,
            weights.momentum,
            weights.atr,
            weights.rsi,
            weights.macd,
            weights.ema,
            weights.ml_history,
        ]
    )
    assert abs(total - 1.0) < 0.01


def test_risk_settings():
    settings = get_settings()
    assert settings.risk.max_risk_per_trade == 0.02
    assert settings.risk.max_daily_risk == 0.06
