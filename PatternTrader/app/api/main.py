from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    backtests,
    dashboard,
    health,
    learning,
    lifecycle,
    models,
    patterns,
    signals,
    strategies,
    trades,
)
from app.core.config.settings import get_settings
from app.core.events.bus import get_event_bus
from app.core.logger import get_logger, setup_logger
from app.database.base import init_db
from app.database.repositories import (
    BacktestRepository,
    LifecycleRepository,
    MLModelRepository,
    PredictionRepository,
    SignalRepository,
    TradeRepository,
)
from app.learning.repository import KnowledgeRepository
from app.learning.service import LearningService
from app.patterns.service import PatternService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logger()
    logger = get_logger("API")
    await init_db()
    await get_event_bus().start()

    lifecycle_repo = LifecycleRepository()
    signal_repo = SignalRepository()
    trade_repo = TradeRepository()
    ml_model_repo = MLModelRepository()
    prediction_repo = PredictionRepository()

    learning = LearningService(
        repository=KnowledgeRepository(),
        ml_model_repository=ml_model_repo,
    )
    await learning.start()

    service = PatternService(
        learning_service=learning,
        lifecycle_repository=lifecycle_repo,
        signal_repository=signal_repo,
        trade_repository=trade_repo,
    )
    await service.start()

    app.state.learning = learning
    app.state.pattern_service = service
    app.state.strategy_manager = service.strategy_manager
    app.state.signal_repository = signal_repo
    app.state.trade_repository = trade_repo
    app.state.backtest_repository = BacktestRepository()
    app.state.ml_model_repository = ml_model_repo
    app.state.prediction_repository = prediction_repo
    try:
        yield
    finally:
        await service.stop()
        await learning.stop()
        await get_event_bus().stop()
        logger.info("API shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.application.name,
        version=settings.application.version,
        description="Professional chart pattern detection and trading platform",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(patterns.router, prefix="/api/v1/patterns", tags=["Patterns"])
    app.include_router(signals.router, prefix="/api/v1/signals", tags=["Signals"])
    app.include_router(trades.router, prefix="/api/v1/trades", tags=["Trades"])
    app.include_router(backtests.router, prefix="/api/v1/backtests", tags=["Backtests"])
    app.include_router(learning.router, prefix="/api/v1/learning", tags=["Learning"])
    app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
    app.include_router(lifecycle.router, prefix="/api/v1/lifecycle", tags=["Lifecycle"])
    app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
    app.include_router(strategies.router, prefix="/api/v1/strategies", tags=["Strategies"])

    return app
