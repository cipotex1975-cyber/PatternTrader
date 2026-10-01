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
        canonical = normalize_timeframe(timeframe)
        if len(canonical) < 2 or not canonical[:-1].isdigit():
            return 60
        unit = canonical[-1]
        value = int(canonical[:-1])
        return value * multipliers.get(unit, 1)


_UNIT_ALIASES = {"m": "m", "h": "h", "d": "d", "w": "w", "M": "M"}

# Valores canónicos de ``Timeframes`` que designan meses, no minutos. Sin este
# conjunto, "1M" (mes) se normalizaría a "1m" (minuto) por el alias de mes.
_MONTH_CANONICAL = frozenset({"1M", "3M", "6M", "12M"})


def normalize_timeframe(timeframe: str) -> str:
    """Normaliza un timeframe a su forma canónica en minúsculas (``1h``, ``15m``).

    Acepta las convenciones que circulan por el proyecto, que no son homogéneas:
    el pipeline live y ``market.default_timeframes`` usan dígitos primero
    (``15m``, ``1h``, ``1d``), mientras que la nomenclatura de los ficheros OHLCV
    y ``config/pairs.yaml`` usan la letra primero y en mayúsculas (``H1``,
    ``M15``, ``D1``). Sin canonicalizar, un modelo entrenado desde
    ``USDCAD_H1_*.txt`` se etiquetaría ``H1`` y nunca casaría con una señal
    live de ``1h``.

    Es idempotente: normalizar un valor ya canónico lo devuelve intacto. El mes
    (``1M``, ``MN1``) se preserva en mayúscula y no se confunde con el minuto.

    Los valores irreconocibles se devuelven en minúsculas tal cual, en lugar de
    forzar un default, para que el error sea visible en el llamador.
    """
    raw = str(timeframe).strip()
    if not raw:
        return raw

    if raw in _MONTH_CANONICAL:
        return raw

    lowered = raw.lower()

    # Prefijo de mes: MN1 → 1M (debe resolverse antes que el alias de minuto).
    if lowered.startswith("mn") and lowered[2:].isdigit():
        return f"{int(lowered[2:])}M"

    # Letra primero: H1, M15, D1, W1.
    prefix_letter, rest = lowered[0], lowered[1:]
    if prefix_letter in _UNIT_ALIASES and rest.isdigit():
        return f"{int(rest)}{_UNIT_ALIASES[prefix_letter]}"

    # Dígito primero: 1h, 15m, 1M, 4h.
    digit_part, unit_part = lowered[:-1], lowered[-1]
    if digit_part.isdigit() and unit_part in _UNIT_ALIASES:
        unit = _UNIT_ALIASES[unit_part]
        value = int(digit_part)
        # "1M" es mes; "1m" es minuto. Se distingue por la caja del original.
        if unit == "m" and value == 1 and raw[-1].isupper():
            return "1M"
        return f"{value}{unit}"

    return lowered


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
