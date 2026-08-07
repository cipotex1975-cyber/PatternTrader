from __future__ import annotations

from fastapi import Request

from app.database.repositories import (
    BacktestRepository,
    MLModelRepository,
    PredictionRepository,
    SignalRepository,
    TradeRepository,
)
from app.learning.service import LearningService
from app.patterns.service import PatternService


def get_learning_service(request: Request) -> LearningService:
    return request.app.state.learning  # type: ignore[no-any-return]


def get_pattern_service(request: Request) -> PatternService:
    return request.app.state.pattern_service  # type: ignore[no-any-return]


def get_signal_repository(request: Request) -> SignalRepository:
    return request.app.state.signal_repository  # type: ignore[no-any-return]


def get_trade_repository(request: Request) -> TradeRepository:
    return request.app.state.trade_repository  # type: ignore[no-any-return]


def get_backtest_repository(request: Request) -> BacktestRepository:
    return request.app.state.backtest_repository  # type: ignore[no-any-return]


def get_ml_model_repository(request: Request) -> MLModelRepository:
    return request.app.state.ml_model_repository  # type: ignore[no-any-return]


def get_prediction_repository(request: Request) -> PredictionRepository:
    return request.app.state.prediction_repository  # type: ignore[no-any-return]
