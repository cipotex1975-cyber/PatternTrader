from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class SplitResult:
    """Split cronológico TRAIN/VALIDATION/TEST sin leakage de labels (FASE 2).

    ``ranges`` contiene las fechas reales (columna ``datetime``) de inicio y
    fin de cada segmento tras el recorte anti-leakage.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    ranges: dict[str, dict[str, Any]] = field(default_factory=dict)


def split_chronological(
    df_features: pd.DataFrame,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float | None = None,
    forward_periods: int = 5,
) -> SplitResult:
    """Divide cronológicamente en TRAIN/VALIDATION/TEST evitando leakage de labels.

    Decisión documentada (FASE 2, sección 2): **OPCIÓN B** — las features se
    calculan una sola vez sobre la serie completa (todos los indicadores son
    causales: rolling/ewm/shift(1), nunca miran el futuro); los labels se
    calculan también globalmente y después del corte posicional se RECORTAN
    explícitamente las últimas ``forward_periods`` filas de TRAIN y de
    VALIDATION. Así ninguna muestra consume ``high`` de un segmento posterior.
    El resultado es idéntico a la Opción A (labels por segmento) pero auditable.

    Si se pasa ``test_size`` se valida que train+validation+test sea 1 con
    tolerancia de punto flotante (1e-6); si no, test es el resto.
    Nunca se hace shuffle; los tres segmentos son contiguos en el tiempo.
    """
    if not 0 < train_size < 1 or not 0 < validation_size < 1:
        raise ValueError("train_size y validation_size deben estar en (0, 1)")
    if test_size is not None:
        total = train_size + validation_size + test_size
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train + validation + test debe ser 1 (tolerancia 1e-6); recibido {total}"
            )
    test_fraction = 1.0 - train_size - validation_size
    if test_fraction <= 0:
        raise ValueError(f"La fracción de test resultante debe ser positiva: {test_fraction}")

    n = len(df_features)
    n_train = int(n * train_size)
    n_validation = int(n * validation_size)
    if n_train == 0 or n_validation == 0 or n_train + n_validation >= n:
        raise ValueError(
            f"Dataset demasiado pequeño ({n} muestras) para el split "
            f"{train_size}/{validation_size}/{test_fraction:.2f}"
        )

    # Corte por posición sobre el dataframe ya limpio (features+labels).
    train_df = df_features.iloc[:n_train]
    validation_df = df_features.iloc[n_train : n_train + n_validation]
    test_df = df_features.iloc[n_train + n_validation :]

    # OPCIÓN B: recorte explícito para que ningún label consuma futuro del
    # segmento siguiente (create_labels mira t+1..t+forward_periods).
    train_df = train_df.iloc[: len(train_df) - forward_periods]
    validation_df = validation_df.iloc[: len(validation_df) - forward_periods]

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError("Algún segmento quedó vacío tras el split/recorte")

    def _range(segment: pd.DataFrame) -> dict[str, Any]:
        positions = [i for i, c in enumerate(segment.columns) if c == "datetime"]
        if not positions or segment.empty:
            return {"samples": int(len(segment)), "start": None, "end": None}
        dt = segment.iloc[:, positions[0]]
        return {
            "samples": int(len(segment)),
            "start": pd.to_datetime(dt.iloc[0]).isoformat(),
            "end": pd.to_datetime(dt.iloc[-1]).isoformat(),
        }

    ranges = {
        "train": _range(train_df),
        "validation": _range(validation_df),
        "test": _range(test_df),
    }

    return SplitResult(
        X_train=train_df[FEATURE_NAMES].values,
        y_train=train_df["label"].values.astype(int),
        X_validation=validation_df[FEATURE_NAMES].values,
        y_validation=validation_df["label"].values.astype(int),
        X_test=test_df[FEATURE_NAMES].values,
        y_test=test_df["label"].values.astype(int),
        ranges=ranges,
    )


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
