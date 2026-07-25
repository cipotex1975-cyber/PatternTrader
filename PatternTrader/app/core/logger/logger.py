from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from app.core.config.settings import get_settings


def setup_logger() -> None:
    settings = get_settings()
    logger.remove()

    log_format = settings.logging.format

    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.logging.level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        format=log_format,
        level=settings.logging.level,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        compression=settings.logging.compression,
        backtrace=True,
        diagnose=True,
    )

    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        format=log_format,
        level="ERROR",
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        compression=settings.logging.compression,
        backtrace=True,
        diagnose=True,
    )

    logger.add(
        log_dir / "trades_{time:YYYY-MM-DD}.log",
        format=log_format,
        level="INFO",
        rotation="1 day",
        retention="90 days",
        compression="gz",
    )


def get_logger(name: str | None = None) -> Any:
    if name:
        return logger.bind(name=name)
    return logger
