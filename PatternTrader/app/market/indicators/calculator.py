from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.config.settings import get_settings
from app.core.logger import get_logger
from app.market.candles.models import Candle

logger = get_logger("IndicatorCalculator")


class IndicatorCalculator:
    def __init__(self) -> None:
        settings = get_settings()
        self._config = settings.market.indicators

    def calculate_all(self, candles: list[Candle]) -> dict[str, dict[str, float]]:
        if not candles:
            return {}

        df = self._candles_to_dataframe(candles)
        indicators: dict[str, dict[str, float]] = {}

        for period in self._config.ema_periods:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        for period in self._config.sma_periods:
            df[f"sma_{period}"] = df["close"].rolling(window=period).mean()

        df["rsi"] = self._calculate_rsi(df["close"], self._config.rsi_period)

        macd = self._calculate_macd(
            df["close"],
            self._config.macd_fast,
            self._config.macd_slow,
            self._config.macd_signal,
        )
        df["macd"] = macd["macd"]
        df["macd_signal"] = macd["signal"]
        df["macd_histogram"] = macd["histogram"]

        df["atr"] = self._calculate_atr(df, self._config.atr_period)

        df["momentum"] = df["close"].pct_change(periods=self._config.momentum_period) * 100.0

        bb = self._calculate_bollinger_bands(
            df["close"], self._config.bb_period, self._config.bb_std
        )
        df["bb_upper"] = bb["upper"]
        df["bb_middle"] = bb["middle"]
        df["bb_lower"] = bb["lower"]

        if self._config.vwap_enabled:
            df["vwap"] = self._calculate_vwap(df)

        for i, candle in enumerate(candles):
            timestamp = candle.data.timestamp.isoformat()
            indicators[timestamp] = {}
            for col in df.columns:
                if col not in ["timestamp", "open", "high", "low", "close", "volume"]:
                    value = df[col].iloc[i]
                    if pd.notna(value):
                        indicators[timestamp][col] = float(value)

        return indicators

    def _candles_to_dataframe(self, candles: list[Candle]) -> pd.DataFrame:
        data = {
            "timestamp": [c.data.timestamp for c in candles],
            "open": [c.data.open for c in candles],
            "high": [c.data.high for c in candles],
            "low": [c.data.low for c in candles],
            "close": [c.data.close for c in candles],
            "volume": [c.data.volume for c in candles],
        }
        return pd.DataFrame(data)

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self, series: pd.Series, fast: int, slow: int, signal: int
    ) -> dict[str, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=period).mean()

    def _calculate_bollinger_bands(
        self, series: pd.Series, period: int, std_dev: float
    ) -> dict[str, pd.Series]:
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return {"upper": upper, "middle": middle, "lower": lower}

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        return vwap

    def get_latest_indicators(self, candles: list[Candle]) -> dict[str, float]:
        indicators = self.calculate_all(candles)
        if not indicators:
            return {}
        latest_timestamp = list(indicators.keys())[-1]
        return indicators[latest_timestamp]
