from __future__ import annotations

from typing import Any


class PatternTraderError(Exception):
    """Base exception for PatternTrader."""

    def __init__(self, message: str = "", details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(PatternTraderError):
    """Raised when there's a configuration error."""
    pass


class DataProviderError(PatternTraderError):
    """Raised when there's an error with a data provider."""
    pass


class PatternError(PatternTraderError):
    """Raised when there's an error with pattern detection."""
    pass


class MarketDataError(PatternTraderError):
    """Raised when there's an error with market data."""
    pass


class BacktestingError(PatternTraderError):
    """Raised when there's an error during backtesting."""
    pass


class MLModelError(PatternTraderError):
    """Raised when there's an error with ML models."""
    pass


class RiskError(PatternTraderError):
    """Raised when there's an error with risk management."""
    pass


class SignalError(PatternTraderError):
    """Raised when there's an error with signal generation."""
    pass


class TelegramError(PatternTraderError):
    """Raised when there's an error with Telegram integration."""
    pass


class DatabaseError(PatternTraderError):
    """Raised when there's a database error."""
    pass
