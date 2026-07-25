from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TradeDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class Trade(BaseModel):
    id: str
    symbol: str
    timeframe: str
    direction: TradeDirection
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    size: float = 1.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    pattern_name: str = ""
    score: float = 0.0
    metadata: dict = Field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        if self.exit_time and self.entry_time:
            return (self.exit_time - self.entry_time).total_seconds()
        return None

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        if self.stop_loss and self.take_profit and self.entry_price:
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
            if risk > 0:
                return reward / risk
        return None


class BacktestConfig(BaseModel):
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    max_positions: int = 10
    risk_per_trade: float = 0.02
    max_daily_loss: float = 0.06
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.02


class BacktestMetrics(BaseModel):
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    expectancy: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0


class BacktestResult(BaseModel):
    config: BacktestConfig
    metrics: BacktestMetrics
    trades: list[Trade] = Field(default_factory=list)
    equity_curve: list[dict] = Field(default_factory=list)
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    metadata: dict = Field(default_factory=dict)

    @property
    def total_return(self) -> float:
        return (self.final_capital - self.initial_capital) / self.initial_capital

    @property
    def annualized_return(self) -> float:
        days = (self.end_date - self.start_date).days
        if days <= 0:
            return 0.0
        return ((self.final_capital / self.initial_capital) ** (365 / days)) - 1
