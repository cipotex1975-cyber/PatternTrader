from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_to_datetime(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> float:
    return dt.timestamp()
