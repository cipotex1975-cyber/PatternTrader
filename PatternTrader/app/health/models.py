from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthFactor(BaseModel):
    name: str
    value: float
    score: float = Field(ge=0.0, le=100.0)
    weight: float = Field(ge=0.0)
    reason: str = ""


class HealthReport(BaseModel):
    health: float = Field(ge=0.0, le=100.0)
    factors: list[HealthFactor] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)

    @property
    def strongest_factor(self) -> HealthFactor | None:
        if not self.factors:
            return None
        return max(self.factors, key=lambda f: f.score)

    @property
    def weakest_factor(self) -> HealthFactor | None:
        if not self.factors:
            return None
        return min(self.factors, key=lambda f: f.score)
