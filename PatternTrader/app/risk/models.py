from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RiskLimits(BaseModel):
    max_risk_per_trade: float = 0.02
    max_daily_risk: float = 0.06
    max_exposure_per_asset: float = 0.10
    max_correlated_exposure: float = 0.15
    max_open_positions: int = 10
    max_leverage: float = 1.0


class PositionSize(BaseModel):
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    risk_amount: float
    risk_pct: float
    potential_reward: float
    risk_reward_ratio: float
    max_loss: float


class RiskAssessment(BaseModel):
    symbol: str
    timeframe: str
    pattern_name: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: PositionSize
    is_acceptable: bool
    risk_score: float = Field(ge=0.0, le=100.0)
    warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)

    @property
    def risk_reward_ratio(self) -> float:
        return self.position_size.risk_reward_ratio

    @property
    def risk_pct(self) -> float:
        return self.position_size.risk_pct
