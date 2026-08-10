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
    fees: float = 0.0
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
    use_atr_stops: bool = False
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.0


class BacktestMetrics(BaseModel):
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    ulcer_index: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    average_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0
    average_win: float = 0.0
    average_loss: float = 0.0
    average_trade: float = 0.0
    payoff_ratio: float = 0.0
    expectancy: float = 0.0
    expectancy_r: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    annual_return: float = 0.0
    annualized_volatility: float = 0.0
    volatility: float = 0.0
    total_fees: float = 0.0


class ClassificationMetrics(BaseModel):
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    confusion_matrix: list[list[int]] = Field(default_factory=lambda: [[0, 0], [0, 0]])


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


class BacktestCase(BaseModel):
    name: str = "backtest"
    candles_count: int = 0
    patterns_count: int = 0
    metadata: dict = Field(default_factory=dict)


class ValidationFold(BaseModel):
    index: int
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None
    train_size: int = 0
    test_size: int = 0
    metrics: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class ValidationResult(BaseModel):
    method: str
    folds: list[ValidationFold] = Field(default_factory=list)
    aggregate: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class MonteCarloResult(BaseModel):
    simulations: int = 0
    initial_capital: float = 0.0
    final_equities: list[float] = Field(default_factory=list)
    max_drawdowns: list[float] = Field(default_factory=list)
    percentiles: dict = Field(default_factory=dict)
    probability_of_profit: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    median_final_equity: float = 0.0
    mean_final_equity: float = 0.0
    best_final_equity: float = 0.0
    worst_final_equity: float = 0.0
