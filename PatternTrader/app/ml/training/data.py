from __future__ import annotations

import numpy as np
import pandas as pd

# Vector de features técnicas usado por el entrenamiento multi-modelo.
# El mismo orden lo comparte `app/ml/features.py` para el scoring en vivo.
FEATURE_NAMES = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "ema_21",
    "ema_50",
    "atr",
    "volume_ratio",
    "price_change",
    "high_low_range",
    "close_position",
    "trend_strength",
]

# Modelos que consumen datos secuenciales 3D (samples, timesteps, features).
SEQUENCE_MODELS = {"lstm", "transformer", "cnn"}


def load_data(file_path: str) -> pd.DataFrame:
    """Carga un archivo OHLCV tab-delimitado (formato MT4/MT5) ordenado por fecha."""
    df = pd.read_csv(file_path, sep="\t")
    df.columns = [c.strip() for c in df.columns]

    if "DateTime" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["DateTime"] + " " + df["time"])
    elif "DateTime" in df.columns:
        df["datetime"] = pd.to_datetime(df["DateTime"])
    else:
        first_col = df.columns[0]
        df["datetime"] = pd.to_datetime(df[first_col])

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Tickvol": "tickvol",
            "Volume": "volume",
            "Spread": "spread",
        }
    )

    for col in ["open", "high", "low", "close", "tickvol", "volume", "spread"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = calculate_ema(series, 12)
    ema26 = calculate_ema(series, 26)
    macd = ema12 - ema26
    signal = calculate_ema(macd, 9)
    histogram = macd - signal
    return macd, signal, histogram


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula los indicadores técnicos que forman el vector canónico."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    df["rsi"] = calculate_rsi(df["close"], 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = calculate_macd(df["close"])
    df["ema_21"] = calculate_ema(df["close"], 21)
    df["ema_50"] = calculate_ema(df["close"], 50)
    df["atr"] = calculate_atr(df, 14)

    vol_col = "tickvol" if "tickvol" in df.columns else "volume"
    if vol_col not in df.columns:
        df[vol_col] = 0
    df["volume_ratio"] = df[vol_col] / df[vol_col].rolling(20, min_periods=1).mean()
    df["price_change"] = df["close"].pct_change()
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["trend_strength"] = (df["ema_21"] - df["ema_50"]) / df["ema_50"]

    return df


def create_labels(
    df: pd.DataFrame,
    forward_periods: int = 5,
    threshold: float = 0.003,
    min_up_moves: int = 2,
) -> pd.Series:
    """Create binary labels for the dataset.

    A candle is labelled **1** (positive) if **at least** ``min_up_moves`` of the next
    ``forward_periods`` candles have a ``high`` price that exceeds the current ``close``
    by ``threshold`` (expressed as a proportion, e.g. 0.003 → 0.3 %).

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame with at least ``close`` and ``high`` columns.
    forward_periods: int, default 5
        Number of future candles to look ahead.
    threshold: float, default 0.003
        Minimum relative price increase required for a forward candle to be counted.
    min_up_moves: int, default 2
        Minimum number of forward candles that must satisfy the threshold to label
        the current candle as positive.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Compute a boolean matrix where each column k indicates whether the high at t+k
    # exceeds the current close by the threshold.
    up_moves = []
    for k in range(1, forward_periods + 1):
        future_high = df["high"].shift(-k)
        condition = (future_high / df["close"] - 1) > threshold
        up_moves.append(condition.astype(int))
    # Sum across the forward window.
    up_sum = sum(up_moves)
    # Positive label if the count meets or exceeds the required moves.
    labels = (up_sum >= min_up_moves).astype(int)
    # Ensure we have no NaNs at the tail where the shift produced missing values.
    labels.iloc[-forward_periods:] = np.nan
    return labels


def prepare_dataset(
    df: pd.DataFrame, forward_periods: int = 5, threshold: float = 0.001
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Aplica features + labels y devuelve (X, y, df_limpio)."""
    df = create_features(df)
    df["label"] = create_labels(df, forward_periods=forward_periods, threshold=threshold)
    df = df.dropna(subset=FEATURE_NAMES + ["label"])
    X = df[FEATURE_NAMES].values
    y = df["label"].values.astype(int)
    return X, y, df


def build_sequences(
    X: np.ndarray, y: np.ndarray, sequence_length: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Convierte matrices 2D en ventanas secuenciales 3D para LSTM/Transformer/CNN.

    Cada ventana termina en el instante ``i`` y se alinea con ``y[i]`` (el label
    describe el futuro a partir de ``i``), respetando la causalidad temporal.
    """
    if len(X) <= sequence_length:
        raise ValueError(
            f"Se necesitan más de {sequence_length} muestras para construir "
            f"secuencias; se tienen {len(X)}"
        )
    windows = np.stack(
        [X[i - sequence_length + 1 : i + 1] for i in range(sequence_length - 1, len(X))]
    )
    labels = y[sequence_length - 1 :]
    return windows, labels


def format_for_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray | None = None,
    sequence_length: int = 30,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Formatea los datos según la familia del modelo.

    - Tabular / anomalías (RF, XGB, LightGBM, CatBoost, IsolationForest,
      AutoEncoder): mantiene la matriz 2D.
    - Secuenciales (LSTM, Transformer, CNN): construye ventanas 3D.
    """
    if model_name in SEQUENCE_MODELS:
        if y is None:
            raise ValueError(f"{model_name} requiere etiquetas para formatear secuencias")
        windows, labels = build_sequences(X, y, sequence_length)
        return windows, labels
    return np.asarray(X, dtype=np.float32), (None if y is None else np.asarray(y, dtype=np.int64))
