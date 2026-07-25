from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config.settings import get_settings
from app.core.logger import setup_logger
from app.api.routes import patterns, signals, backtests, health, dashboard


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logger()
    yield


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
    app.include_router(backtests.router, prefix="/api/v1/backtests", tags=["Backtests"])
    app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])

    return app
