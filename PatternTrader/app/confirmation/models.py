from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ConfirmationStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    PENDING = "PENDING"
    SKIPPED = "SKIPPED"


class ConfirmationRule(BaseModel):
    name: str
    description: str
    required: bool = True
    weight: float = 1.0


class ConfirmationCheck(BaseModel):
    rule: ConfirmationRule
    status: ConfirmationStatus
    value: float | None = None
    threshold: float | None = None
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConfirmationResult(BaseModel):
    is_confirmed: bool
    score: float = Field(ge=0.0, le=100.0)
    checks: list[ConfirmationCheck] = Field(default_factory=list)
    passed_checks: int = 0
    failed_checks: int = 0
    total_required: int = 0
    passed_required: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        total = len(self.checks)
        if total == 0:
            return 0.0
        return self.passed_checks / total

    @property
    def required_pass_rate(self) -> float:
        if self.total_required == 0:
            return 1.0
        return self.passed_required / self.total_required
