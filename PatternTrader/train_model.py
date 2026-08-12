from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from app.ml.models.random_forest import RandomForestModel
from app.ml.training import (
    FEATURE_NAMES,
    create_features,
    create_labels,
    load_data,
)


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
