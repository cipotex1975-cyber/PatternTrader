from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from app.ml.models.random_forest import RandomForestModel


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


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t")
    df.columns = [c.strip() for c in df.columns]

    if "DateTime" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["DateTime"] + " " + df["time"])
    elif "DateTime" in df.columns:
        df["datetime"] = pd.to_datetime(df["DateTime"])
    else:
        first_col = df.columns[0]
        df["datetime"] = pd.to_datetime(df[first_col])

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Tickvol": "tickvol",
        "Volume": "volume",
        "Spread": "spread",
    })

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
    df = df.copy()

    df["rsi"] = calculate_rsi(df["close"], 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = calculate_macd(df["close"])
    df["ema_21"] = calculate_ema(df["close"], 21)
    df["ema_50"] = calculate_ema(df["close"], 50)
    df["atr"] = calculate_atr(df, 14)

    df["volume_ratio"] = df["tickvol"] / df["tickvol"].rolling(20, min_periods=1).mean()
    df["price_change"] = df["close"].pct_change()
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["trend_strength"] = (df["ema_21"] - df["ema_50"]) / df["ema_50"]

    return df


def create_labels(
    df: pd.DataFrame, forward_periods: int = 5, threshold: float = 0.001
) -> pd.Series:
    future_high = (
        df["high"]
        .shift(-1)
        .rolling(5)
        .max()
    )
    # ¿Dentro de 5 horas el cierre estará al menos un 0.1% por encima del precio actual?
    # future_return = df["close"].shift(-forward_periods) / df["close"] - 1

    # ¿En algún momento de las próximas 5 horas el precio llegó a subir un 0.1%?
    future_return = future_high / df["close"] - 1
    labels = (future_return > threshold).astype(int)
    return labels


def main():
    parser = argparse.ArgumentParser(description="Train ML model with OHLCV data")
    parser.add_argument("data_file", type=str, help="Path to the data file")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size (default: 0.2)")
    parser.add_argument(
        "--n-estimators", type=int, default=500, help="Number of trees (default: 100)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=15, help="Maximum tree depth (default: 15)"
    )
    parser.add_argument(
        "--forward-periods",
        type=int,
        default=5,
        help="Forward periods for labels (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="Threshold for positive label (default: 0.001)",
    )
    parser.add_argument("--save-model", type=str, default=None, help="Path to save trained model")
    args = parser.parse_args()

    print(f"\nLoading data from: {args.data_file}")
    df = load_data(args.data_file)
    print(f"Loaded {len(df)} candles")

    print("\nCalculating technical indicators...")
    df = create_features(df)

    print("Creating labels...")
    df["label"] = create_labels(df, forward_periods=args.forward_periods, threshold=args.threshold)

    df = df.dropna(subset=FEATURE_NAMES + ["label"])
    print(f"Samples after dropping NaN: {len(df)}")

    print("\nLabel distribution:")
    print(f"  Positive (1): {int(df['label'].sum())} ({df['label'].mean():.2%})")
    print(f"  Negative (0): {int((df['label'] == 0).sum())} ({1 - df['label'].mean():.2%})")

    X = df[FEATURE_NAMES].values
    y = df["label"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, shuffle=False
    )

    print("\nSplit:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")

    print(
        f"\nTraining Random Forest (n_estimators={args.n_estimators}, "
        f"max_depth={args.max_depth})..."
    )
    model = RandomForestModel(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )
    train_metrics = model.train(X_train, y_train, feature_names=FEATURE_NAMES)
    print(f"  Train accuracy: {train_metrics['train_accuracy']:.4f}")

    print("\nEvaluating on test set...")
    eval_metrics = model.evaluate(X_test, y_test)
    print(f"Accuracy : {eval_metrics['accuracy']:.4f}")
    print(f"Precision: {eval_metrics['precision']:.4f}")
    print(f"Recall   : {eval_metrics['recall']:.4f}")
    print(f"F1 Score : {eval_metrics['f1']:.4f}")

    proba = model.predict_proba(X_test)[:, 1]

    try:
        roc = roc_auc_score(y_test, proba)
        print(f"ROC AUC  : {roc:.4f}")
    except ValueError:
        print("ROC AUC  : N/A (single class)")

    print("\nProbability statistics")
    print(f"Min : {proba.min():.4f}")
    print(f"Max: {proba.max():.4f}")
    print(f"Mean: {proba.mean():.4f}")

    print("\nProbability percentiles")
    for p in [50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(proba, p):.4f}")

    print("\n==============================")
    print("Threshold analysis")
    print("==============================")

    for thr in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
        pred = (proba >= thr).astype(int)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        print(
            f"  {thr:.2f} | "
            f"Pred={pred.sum():5d} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f}"
        )

    print("\nFeature importance:")
    importance = model.get_feature_importance()
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, score in sorted_imp:
        print(f"  {feat:20s} {score:.4f}")

    if args.save_model:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        print(f"\nModel saved to: {save_path}")
    else:
        print("\nUse --save-model <path> to save the trained model")


if __name__ == "__main__":
    main()
