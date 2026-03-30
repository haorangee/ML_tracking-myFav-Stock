"""
visualize.py
------------
Visualize actual vs predicted NVDA close prices on the test set and save
the plot to ``data/nvda_predictions.png``.

Usage::

    python visualize.py [--no-fetch] [--output PATH]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_fetcher import fetch_nvda_data, load_nvda_data, TICKER
from features import build_features, FEATURE_COLS
from model import load_model, DEFAULT_MODEL_PATH

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_OUTPUT = os.path.join(OUTPUT_DIR, "nvda_predictions.png")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Visualize NVDA predictions")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Use cached data instead of downloading.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Where to save the plot PNG.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="Path to the saved model.")
    return parser.parse_args(argv)


def plot_predictions(
    dates,
    actual,
    predicted,
    output_path: str = DEFAULT_OUTPUT,
) -> None:
    """Render and save a two-panel chart.

    Panel 1: Actual vs Predicted closing price.
    Panel 2: Prediction error (actual − predicted).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    error = actual - predicted

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("NVDA Stock Price — Actual vs Predicted (Test Set)", fontsize=14)

    # Panel 1: prices
    ax1 = axes[0]
    ax1.plot(dates, actual, label="Actual Close", color="#1f77b4", linewidth=1.5)
    ax1.plot(dates, predicted, label="Predicted Next-Day Close",
             color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: error
    ax2 = axes[1]
    ax2.bar(dates, error, color=["#2ca02c" if e >= 0 else "#d62728" for e in error],
            alpha=0.7, width=1.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Error (USD)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def run(args=None):
    args = parse_args(args)

    if args.no_fetch:
        raw = load_nvda_data(TICKER)
    else:
        raw = fetch_nvda_data()

    df = build_features(raw)
    available_features = [c for c in FEATURE_COLS if c in df.columns]

    model = load_model(args.model)

    # Use the same 80/20 split as train.py
    split = int(len(df) * 0.80)
    test_df = df.iloc[split:]
    X_test = test_df[available_features]

    predictions = model.predict(X_test)
    actual = test_df["Close"].values

    plot_predictions(test_df.index, actual, predictions, output_path=args.output)
    return args.output


if __name__ == "__main__":
    run()
