"""
train.py
--------
End-to-end training pipeline for the NVDA stock price predictor.

Usage::

    python train.py [--start YYYY-MM-DD] [--model-type rf|gbm] [--no-fetch]

Steps
-----
1. (Optionally) fetch fresh NVDA data from Yahoo Finance.
2. Build feature set via ``features.build_features``.
3. Chronological train/test split (80 / 20).
4. Train model.
5. Evaluate on held-out test set.
6. Persist model to ``models/nvda_rf_model.joblib``.
"""

import argparse
import sys

from data_fetcher import fetch_nvda_data, load_nvda_data, TICKER
from features import build_features, FEATURE_COLS
from model import (
    train_model,
    evaluate_model,
    save_model,
    cross_validate_model,
    DEFAULT_MODEL_PATH,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train NVDA stock price predictor")
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="Start date for historical data (YYYY-MM-DD). Default: 2015-01-01",
    )
    parser.add_argument(
        "--model-type",
        choices=["rf", "gbm"],
        default="rf",
        help="Model type: 'rf' (Random Forest) or 'gbm' (Gradient Boosting). Default: rf",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip downloading data and use previously cached CSV instead.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_MODEL_PATH,
        help=f"Where to save the trained model. Default: {DEFAULT_MODEL_PATH}",
    )
    return parser.parse_args(argv)


def run(args=None):
    args = parse_args(args)

    # 1. Data acquisition
    if args.no_fetch:
        print("Loading cached data …")
        raw = load_nvda_data(TICKER)
    else:
        raw = fetch_nvda_data(start=args.start)

    print(f"Raw data shape: {raw.shape}  ({raw.index[0].date()} → {raw.index[-1].date()})")

    # 2. Feature engineering
    df = build_features(raw)
    print(f"Feature data shape after engineering: {df.shape}")

    # 3. Train / test split (chronological)
    split = int(len(df) * 0.80)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X_train = train_df[available_features]
    y_train = train_df["target"]
    X_test = test_df[available_features]
    y_test = test_df["target"]

    print(f"Train size: {len(train_df)}  |  Test size: {len(test_df)}")

    # 4. Cross-validation
    print("\nRunning time-series cross-validation …")
    cv_results = cross_validate_model(
        df[available_features], df["target"], model_type=args.model_type
    )
    print(f"CV RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")

    # 5. Train on full training set
    print(f"\nTraining {args.model_type.upper()} model …")
    model = train_model(X_train, y_train, model_type=args.model_type)

    # 6. Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    print(
        f"\nTest-set metrics:"
        f"\n  MAE  : ${metrics['mae']:.4f}"
        f"\n  RMSE : ${metrics['rmse']:.4f}"
        f"\n  R²   : {metrics['r2']:.4f}"
    )

    # 7. Feature importances (RF / GBM expose them the same way)
    regressor = model.named_steps["regressor"]
    if hasattr(regressor, "feature_importances_"):
        importances = sorted(
            zip(available_features, regressor.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        print("\nTop-10 feature importances:")
        for feat, imp in importances[:10]:
            print(f"  {feat:<20} {imp:.4f}")

    # 8. Persist model
    save_model(model, args.output)
    print("\nTraining complete.")
    return model, metrics


if __name__ == "__main__":
    run()
