from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ScoreComponent(BaseModel):
    name: str
    weight: float
    value: float
    score: float
    reason: str = ""


class ScoreResult(BaseModel):
    total_score: float = Field(ge=0.0, le=100.0)
    components: list[ScoreComponent] = Field(default_factory=list)
    grade: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.total_score >= 75

    @property
    def should_alert(self) -> bool:
        return self.total_score >= 85

    @property
    def should_send(self) -> bool:
        return self.total_score >= 95
