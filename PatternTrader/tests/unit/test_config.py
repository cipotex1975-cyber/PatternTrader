from app.core.config.settings import get_settings


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
    assert "BTCUSDT" in settings.market.default_symbols
    assert len(settings.market.default_timeframes) > 0


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
