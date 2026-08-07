from __future__ import annotations

from enum import Enum


class ExitReason(str, Enum):
    """Razón por la que una posición se cierra."""

    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    MANUAL = "manual"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"
