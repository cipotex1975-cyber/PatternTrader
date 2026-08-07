from __future__ import annotations

from typing import Optional

import numpy as np

from app.core.logger import get_logger
from app.market.candles.models import Candle

logger = get_logger("MarketFeatures")

# Vector canónico de features técnicas usado tanto por el ScoringEngine como
# por el aprendizaje continuo (unificación de modelos). Orden fijo: no cambiar.
TECHNICAL_FEATURE_NAMES = [
    "rsi",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "ema_21",
    "ema_50",
    "atr",
    "volume_ratio",
    "price_change",
    "high_low_range",
    "close_position",
    "trend_strength",
]


def _ema(series: np.ndarray, period: int) -> float:
    alpha = 2 / (period + 1)
    result = series[0]
    for val in series[1:]:
        result = alpha * val + (1 - alpha) * result
    return float(result)


def _rsi(series: np.ndarray, period: int = 14) -> float:
    deltas = np.diff(series)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(series: np.ndarray) -> tuple[float, float, float]:
    ema12 = _ema(series, 12)
    ema26 = _ema(series, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(np.array([macd_line]), 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> float:
    tr1 = highs[-period:] - lows[-period:]
    tr2 = np.abs(highs[-period:] - closes[-period - 1 : -1])
    tr3 = np.abs(lows[-period:] - closes[-period - 1 : -1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return float(np.mean(tr))


def extract_technical_features(candles: list[Candle]) -> Optional[np.ndarray]:
    """Extrae el vector canónico de features técnicas a partir de velas.

    Requiere al menos 20 velas; devuelve ``None`` si no hay datos suficientes.
    """
    if not candles or len(candles) < 20:
        return None

    try:
        closes = np.array([c.data.close for c in candles])
        highs = np.array([c.data.high for c in candles])
        lows = np.array([c.data.low for c in candles])
        volumes = np.array([c.data.volume for c in candles])

        ema_21 = _ema(closes, 21)
        ema_50 = _ema(closes, 50)
        rsi_val = _rsi(closes)
        macd_line, signal_line, histogram = _macd(closes)
        atr_val = _atr(highs, lows, closes)

        vol_avg = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        volume_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1.0

        price_change = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] != 0 else 0
        high_low_range = (highs[-1] - lows[-1]) / closes[-1] if closes[-1] != 0 else 0
        close_position = (
            (closes[-1] - lows[-1]) / (highs[-1] - lows[-1])
            if (highs[-1] - lows[-1]) != 0
            else 0.5
        )
        trend_strength = (ema_21 - ema_50) / ema_50 if ema_50 != 0 else 0

        return np.array(
            [
                rsi_val,
                macd_line,
                signal_line,
                histogram,
                ema_21,
                ema_50,
                atr_val,
                volume_ratio,
                price_change,
                high_low_range,
                close_position,
                trend_strength,
            ]
        )
    except Exception as e:
        logger.error(f"Failed to extract market features: {e}")
        return None


def features_to_dict(features: Optional[np.ndarray]) -> dict[str, float]:
    """Convierte el vector canónico en un dict para los consumidores que reciben
    ``indicators`` (FeatureBuilder, LearningService.predict, etc.)."""
    if features is None:
        return {}
    return {
        name: float(value) for name, value in zip(TECHNICAL_FEATURE_NAMES, features)
    }
