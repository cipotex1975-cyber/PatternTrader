from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from app.core.config.settings import get_settings

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOG_DIR = _REPO_ROOT / "logs"
_CONFIGURED = False


def setup_logger() -> None:
    global _CONFIGURED
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

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.add(
        _LOG_DIR / "app_{time:YYYY-MM-DD}.log",
        format=log_format,
        level=settings.logging.level,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        compression=settings.logging.compression,
        backtrace=True,
        diagnose=True,
    )

    logger.add(
        _LOG_DIR / "errors_{time:YYYY-MM-DD}.log",
        format=log_format,
        level="ERROR",
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        compression=settings.logging.compression,
        backtrace=True,
        diagnose=True,
    )

    logger.add(
        _LOG_DIR / "trades_{time:YYYY-MM-DD}.log",
        format=log_format,
        level="INFO",
        rotation=settings.logging.trades_rotation,
        retention=settings.logging.trades_retention,
        compression=settings.logging.compression,
    )

    _CONFIGURED = True


def _ensure_configured() -> None:
    if not _CONFIGURED:
        setup_logger()


def get_logger(name: str | None = None) -> Any:
    _ensure_configured()
    if name:
        return logger.bind(name=name)
    return logger
