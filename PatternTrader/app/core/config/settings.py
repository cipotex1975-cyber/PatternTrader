from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    model_config = {"env_prefix": "DB_"}

    host: str = "localhost"
    port: int = 5432
    name: str = "pattern_trader"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 10
    echo: bool = False

    @property
    def url(self) -> str:
        override = os.environ.get("DATABASE_URL")
        if override:
            return override
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        )


class TelegramSettings(BaseSettings):
    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = False
    cooldown_minutes: int = 5
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    timeout_seconds: float = 10.0
    dedup_store_path: str = "./data/state/telegram_dedup.json"
    send_image: bool = True
    min_priority: str = "CRITICAL"


class BinanceSettings(BaseSettings):
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True


class BybitSettings(BaseSettings):
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True


class YahooSettings(BaseSettings):
    enabled: bool = True


class PolygonSettings(BaseSettings):
    api_key: str = ""
    enabled: bool = True


class AlphaVantageSettings(BaseSettings):
    api_key: str = ""
    enabled: bool = True


class MetaTraderSettings(BaseSettings):
    enabled: bool = False
    login: int = 0
    password: str = ""
    server: str = ""
    path: str = ""


class InteractiveBrokersSettings(BaseSettings):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1


class DataProviderSettings(BaseSettings):
    default: str = "binance"
    binance: BinanceSettings = Field(default_factory=BinanceSettings)
    bybit: BybitSettings = Field(default_factory=BybitSettings)
    yahoo: YahooSettings = Field(default_factory=YahooSettings)
    polygon: PolygonSettings = Field(default_factory=PolygonSettings)
    alphavantage: AlphaVantageSettings = Field(default_factory=AlphaVantageSettings)
    metatrader: MetaTraderSettings = Field(default_factory=MetaTraderSettings)
    interactive_brokers: InteractiveBrokersSettings = Field(
        default_factory=InteractiveBrokersSettings
    )


class IndicatorSettings(BaseSettings):
    ema_periods: list[int] = [9, 21, 50, 100, 200]
    sma_periods: list[int] = [20, 50, 100, 200]
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    vwap_enabled: bool = True
    momentum_period: int = 10


class MarketStructureSettings(BaseSettings):
    pivot_lookback: int = 2
    fractal_window: int = 2
    zigzag_threshold: float = 0.03
    zigzag_atr_multiplier: float = 1.5
    trend_min_pivots: int = 2
    trend_strength_lookback: int = 5
    channel_slope_tolerance: float = 0.15


class MarketSettings(BaseSettings):
    default_timeframes: list[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
    default_symbols: list[str] = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    indicators: IndicatorSettings = Field(default_factory=IndicatorSettings)
    structure: MarketStructureSettings = Field(default_factory=MarketStructureSettings)


class ScoringWeights(BaseSettings):
    pattern_structure: float = 0.35
    volume: float = 0.20
    momentum: float = 0.10
    atr: float = 0.10
    rsi: float = 0.10
    macd: float = 0.05
    ema: float = 0.05
    ml_history: float = 0.05


class ScoringSettings(BaseSettings):
    weights: ScoringWeights = Field(default_factory=ScoringWeights)


class PatternScoringSettings(BaseSettings):
    min_score_to_observe: float = 60
    min_score_to_prepare: float = 75
    min_score_to_alert: float = 85
    min_score_to_send: float = 95
    cooldown_minutes: int = 5
    signal_ttl_hours: int = 24


class PatternLifecycleSettings(BaseSettings):
    enabled: bool = True
    check_interval_seconds: int = 5
    max_patterns_per_symbol: int = 50
    timeframes: list[str] = ["15m", "1h", "4h"]
    candle_limit: int = 500


class PatternHealthSettings(BaseSettings):
    recalculate_interval_seconds: int = 10


class PatternDetectionSettings(BaseSettings):
    peak_tolerance: float = 0.02
    flag_slope_tolerance: float = 0.001


class PatternConfirmationSettings(BaseSettings):
    max_spread_ratio: float = 0.001


class PatternSettings(BaseSettings):
    scoring: PatternScoringSettings = Field(default_factory=PatternScoringSettings)
    lifecycle: PatternLifecycleSettings = Field(default_factory=PatternLifecycleSettings)
    health: PatternHealthSettings = Field(default_factory=PatternHealthSettings)
    detection: PatternDetectionSettings = Field(default_factory=PatternDetectionSettings)
    confirmation: PatternConfirmationSettings = Field(default_factory=PatternConfirmationSettings)


class StrategySettings(BaseSettings):
    enabled: list[str] = ["trend_follow", "breakout", "contrarian"]
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)


class RiskSettings(BaseSettings):
    max_risk_per_trade: float = 0.02
    max_daily_risk: float = 0.06
    max_exposure_per_asset: float = 0.10
    max_correlated_exposure: float = 0.15
    default_rr_ratio: float = 2.0
    trailing_stop_enabled: bool = False
    symbol_sectors: dict[str, str] = Field(default_factory=dict)
    correlations: dict[str, dict[str, float]] = Field(default_factory=dict)


class BacktestingSettings(BaseSettings):
    default_initial_capital: float = 100000
    default_commission: float = 0.001
    default_slippage: float = 0.0005
    default_max_positions: int = 10
    walk_forward_splits: int = 5
    monte_carlo_simulations: int = 1000


class RandomForestModelSettings(BaseSettings):
    n_estimators: int = 100
    max_depth: int = 10


class XGBoostModelSettings(BaseSettings):
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1


class LightGBMModelSettings(BaseSettings):
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1


class LSTMModelSettings(BaseSettings):
    sequence_length: int = 60
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2


class MLModelsSettings(BaseSettings):
    random_forest: RandomForestModelSettings = Field(default_factory=RandomForestModelSettings)
    xgboost: XGBoostModelSettings = Field(default_factory=XGBoostModelSettings)
    lightgbm: LightGBMModelSettings = Field(default_factory=LightGBMModelSettings)
    lstm: LSTMModelSettings = Field(default_factory=LSTMModelSettings)


class MLSettings(BaseSettings):
    model_path: str = "./models/"
    training_data_path: str = "./data/training/"
    models: MLModelsSettings = Field(default_factory=MLModelsSettings)


class LoggingSettings(BaseSettings):
    level: str = "INFO"
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    rotation: str = "100 MB"
    retention: str = "30 days"
    compression: str = "gz"


class ServerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = True


class ApplicationSettings(BaseSettings):
    name: str = "PatternTrader"
    version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"


class Settings(BaseSettings):
    application: ApplicationSettings = Field(default_factory=ApplicationSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    data_providers: DataProviderSettings = Field(default_factory=DataProviderSettings)
    market: MarketSettings = Field(default_factory=MarketSettings)
    patterns: PatternSettings = Field(default_factory=PatternSettings)
    strategies: StrategySettings = Field(default_factory=StrategySettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    backtesting: BacktestingSettings = Field(default_factory=BacktestingSettings)
    ml: MLSettings = Field(default_factory=MLSettings)

    model_config = {
        "env_prefix": "",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_env_vars(value: str) -> str:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.environ.get(env_var, "")
    return value


def _resolve_nested_env_vars(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _resolve_nested_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_resolve_nested_env_vars(item) for item in data]
    elif isinstance(data, str):
        return _resolve_env_vars(data)
    return data


@lru_cache()
def get_settings() -> Settings:
    config_dir = Path(__file__).parent.parent.parent.parent / "config"
    config_data = _load_yaml_config(config_dir / "settings.yaml")
    config_data = _resolve_nested_env_vars(config_data)
    return Settings(**config_data)
