from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from app.core.logger import get_logger

logger = get_logger("SystemMonitor")


class SystemMonitor:
    def __init__(self) -> None:
        self._start_time = time.time()
        self._metrics: dict[str, Any] = {}
        self._counters: dict[str, int] = {}

    def record_metric(self, name: str, value: float) -> None:
        self._metrics[name] = {
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def increment_counter(self, name: str, count: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + count

    def get_uptime(self) -> float:
        return time.time() - self._start_time

    def get_metrics(self) -> dict[str, Any]:
        return {
            "uptime_seconds": self.get_uptime(),
            "metrics": self._metrics.copy(),
            "counters": self._counters.copy(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_health_status(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "uptime": self.get_uptime(),
            "metrics_count": len(self._metrics),
            "counters_count": len(self._counters),
        }

    def reset(self) -> None:
        self._metrics.clear()
        self._counters.clear()
        self._start_time = time.time()
