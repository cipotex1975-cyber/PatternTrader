from __future__ import annotations

from enum import Enum


class Timeframes(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"

    @classmethod
    def to_minutes(cls, timeframe: str) -> int:
        multipliers = {
            "m": 1,
            "h": 60,
            "d": 1440,
            "w": 10080,
            "M": 43200,
        }
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        return value * multipliers.get(unit, 1)


class Patterns(str, Enum):
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    BULL_PENNANT = "bull_pennant"
    BEAR_PENNANT = "bear_pennant"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    RECTANGLE = "rectangle"
    CHANNEL = "channel"
    CUP_AND_HANDLE = "cup_and_handle"
    ROUNDED_BOTTOM = "rounded_bottom"
    DIAMOND = "diamond"
    BROADENING_FORMATION = "broadening_formation"


class Indicators(str, Enum):
    EMA = "ema"
    SMA = "sma"
    RSI = "rsi"
    MACD = "macd"
    ATR = "atr"
    BB = "bollinger_bands"
    VWAP = "vwap"
    STOCH = "stochastic"
    ADX = "adx"
    CCI = "cci"
    WILLIAMS_R = "williams_r"
    MOMENTUM = "momentum"
    OBV = "obv"
    MFI = "mfi"
    ICHIMOKU = "ichimoku"
