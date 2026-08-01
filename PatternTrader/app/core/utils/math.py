from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal


def round_to_tick(price: float, tick_size: float) -> float:
    if tick_size <= 0:
        return price
    decimal_price = Decimal(str(price))
    decimal_tick = Decimal(str(tick_size))
    return float(decimal_price.quantize(decimal_tick, rounding=ROUND_HALF_UP))


def calculate_percentage(value: float, total: float) -> float:
    if total == 0:
        return 0.0
    return (value / total) * 100


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)
