from __future__ import annotations

import time
from typing import Any, Optional

from app.core.logger import get_logger

logger = get_logger("MemoryCache")


class MemoryCache:
    def __init__(self, default_ttl: int = 300) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            entry = self._cache[key]
            if time.time() < entry["expires_at"]:
                return entry["value"]
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ttl = ttl or self._default_ttl
        self._cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
            "created_at": time.time(),
        }

    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def size(self) -> int:
        self._cleanup_expired()
        return len(self._cache)

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [k for k, v in self._cache.items() if now >= v["expires_at"]]
        for k in expired:
            del self._cache[k]

    def get_all_keys(self) -> list[str]:
        self._cleanup_expired()
        return list(self._cache.keys())
