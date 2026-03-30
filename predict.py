"""
predict.py
----------
Load the trained NVDA model and predict the next-day close price using the
most recent available market data.

Usage::

    python predict.py [--days N] [--no-fetch]

The script prints:
- The last N actual closing prices.
- The model's predicted next-day price for each of the last N rows.
- A simple directional signal (BUY / SELL / HOLD based on predicted vs actual).
"""

import argparse
from datetime import date

from data_fetcher import fetch_nvda_data, load_nvda_data, TICKER
from features import build_features, FEATURE_COLS
from model import load_model, DEFAULT_MODEL_PATH


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict NVDA next-day close price")
    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Number of most-recent trading days to predict. Default: 10",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Use cached data instead of downloading fresh data.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the saved model file. Default: {DEFAULT_MODEL_PATH}",
    )
    return parser.parse_args(argv)


def signal(actual: float, predicted: float, threshold: float = 0.005) -> str:
    """Simple directional signal.

    Parameters
    ----------
    actual:
        Current closing price.
    predicted:
        Predicted next-day closing price.
    threshold:
        Minimum relative change to trigger BUY/SELL.
    """
    change = (predicted - actual) / actual
    if change > threshold:
        return "🟢 BUY"
    if change < -threshold:
        return "🔴 SELL"
    return "🟡 HOLD"


def run(args=None):
    args = parse_args(args)

    # Load data
    if args.no_fetch:
        raw = load_nvda_data(TICKER)
    else:
        raw = fetch_nvda_data()

    # Build features
    df = build_features(raw)
    available_features = [c for c in FEATURE_COLS if c in df.columns]

    # Load model
    model = load_model(args.model)

    # Predict on the last N rows
    tail = df.tail(args.days)
    X = tail[available_features]
    predictions = model.predict(X)

    print(f"\n{'Date':<12} {'Actual Close':>14} {'Predicted Next':>15} {'Signal':>10}")
    print("-" * 55)
    for dt, actual, pred in zip(tail.index, tail["Close"], predictions):
        sig = signal(actual, pred)
        print(f"{str(dt.date()):<12} {actual:>14.4f} {pred:>15.4f} {sig:>10}")

    # Latest prediction
    latest_date = tail.index[-1].date()
    latest_actual = tail["Close"].iloc[-1]
    latest_pred = predictions[-1]
    print(f"\n📊 Latest ({latest_date}) | Close: ${latest_actual:.2f} | "
          f"Predicted next-day: ${latest_pred:.2f} | {signal(latest_actual, latest_pred)}")
    return predictions


if __name__ == "__main__":
    run()
