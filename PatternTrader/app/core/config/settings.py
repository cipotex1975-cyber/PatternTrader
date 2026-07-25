from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
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
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class TelegramSettings(BaseSettings):
    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = False


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


class DataProviderSettings(BaseSettings):
    default: str = "binance"
    binance: BinanceSettings = Field(default_factory=BinanceSettings)
    bybit: BybitSettings = Field(default_factory=BybitSettings)
    yahoo: YahooSettings = Field(default_factory=YahooSettings)


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


class MarketSettings(BaseSettings):
    default_timeframes: list[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
    default_symbols: list[str] = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    indicators: IndicatorSettings = Field(default_factory=IndicatorSettings)


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


class PatternLifecycleSettings(BaseSettings):
    check_interval_seconds: int = 5
    max_patterns_per_symbol: int = 50


class PatternHealthSettings(BaseSettings):
    recalculate_interval_seconds: int = 10


class PatternSettings(BaseSettings):
    scoring: PatternScoringSettings = Field(default_factory=PatternScoringSettings)
    lifecycle: PatternLifecycleSettings = Field(default_factory=PatternLifecycleSettings)
    health: PatternHealthSettings = Field(default_factory=PatternHealthSettings)


class RiskSettings(BaseSettings):
    max_risk_per_trade: float = 0.02
    max_daily_risk: float = 0.06
    max_exposure_per_asset: float = 0.10
    max_correlated_exposure: float = 0.15
    default_rr_ratio: float = 2.0
    trailing_stop_enabled: bool = False


class BacktestingSettings(BaseSettings):
    default_initial_capital: float = 100000
    default_commission: float = 0.001
    default_slippage: float = 0.0005
    walk_forward_splits: int = 5
    monte_carlo_simulations: int = 1000


class MLSettings(BaseSettings):
    model_path: str = "./models/"
    training_data_path: str = "./data/training/"


class LoggingSettings(BaseSettings):
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
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
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    backtesting: BacktestingSettings = Field(default_factory=BacktestingSettings)
    ml: MLSettings = Field(default_factory=MLSettings)

    model_config = {"env_prefix": "", "env_file": ".env", "env_file_encoding": "utf-8"}


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
