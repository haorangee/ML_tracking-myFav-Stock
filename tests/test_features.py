"""
tests/test_features.py
----------------------
Unit tests for the feature-engineering module (features.py).

These tests do NOT require network access or a trained model — they operate
entirely on small synthetic DataFrames.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Make sure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features import (
    add_moving_averages,
    add_macd,
    add_bollinger_bands,
    add_rsi,
    add_return_and_volatility,
    add_lagged_prices,
    add_volume_features,
    add_target,
    build_features,
    FEATURE_COLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

class TestMovingAverages:
    def test_columns_created(self):
        df = add_moving_averages(make_ohlcv())
        for col in ("ma5", "ma10", "ma20", "ma50", "ema12", "ema26"):
            assert col in df.columns, f"Missing column: {col}"

    def test_ma5_value(self):
        df = make_ohlcv()
        df = add_moving_averages(df)
        # After 4 NaN rows, ma5 should equal rolling mean of last 5 closes
        idx = 50
        expected = df["Close"].iloc[idx - 4 : idx + 1].mean()
        assert abs(df["ma5"].iloc[idx] - expected) < 1e-9

    def test_ema_length_preserved(self):
        df = make_ohlcv(80)
        result = add_moving_averages(df)
        assert len(result) == 80


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:
    def setup_method(self):
        self.df = add_macd(add_moving_averages(make_ohlcv()))

    def test_columns_exist(self):
        for col in ("macd", "macd_signal", "macd_hist"):
            assert col in self.df.columns

    def test_macd_equals_ema_diff(self):
        df = self.df.dropna()
        np.testing.assert_allclose(df["macd"], df["ema12"] - df["ema26"])

    def test_histogram_equals_macd_minus_signal(self):
        df = self.df.dropna()
        np.testing.assert_allclose(df["macd_hist"], df["macd"] - df["macd_signal"])


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def setup_method(self):
        self.df = add_bollinger_bands(make_ohlcv())

    def test_columns_exist(self):
        for col in ("bb_mid", "bb_upper", "bb_lower", "bb_width"):
            assert col in self.df.columns

    def test_upper_greater_than_lower(self):
        df = self.df.dropna()
        assert (df["bb_upper"] > df["bb_lower"]).all()

    def test_bb_width_positive(self):
        df = self.df.dropna()
        assert (df["bb_width"] > 0).all()


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def setup_method(self):
        self.df = add_rsi(make_ohlcv())

    def test_rsi_column_exists(self):
        assert "rsi" in self.df.columns

    def test_rsi_bounds(self):
        rsi = self.df["rsi"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI must be in [0, 100]"


# ---------------------------------------------------------------------------
# Return & Volatility
# ---------------------------------------------------------------------------

class TestReturnAndVolatility:
    def setup_method(self):
        self.df = add_return_and_volatility(make_ohlcv())

    def test_columns_exist(self):
        assert "daily_return" in self.df.columns
        assert "volatility_5d" in self.df.columns

    def test_daily_return_first_is_nan(self):
        assert pd.isna(self.df["daily_return"].iloc[0])

    def test_volatility_positive(self):
        vol = self.df["volatility_5d"].dropna()
        assert (vol >= 0).all()


# ---------------------------------------------------------------------------
# Lagged prices
# ---------------------------------------------------------------------------

class TestLaggedPrices:
    def test_lag_columns_created(self):
        df = add_lagged_prices(make_ohlcv(), lags=5)
        for lag in range(1, 6):
            assert f"lag{lag}" in df.columns

    def test_lag1_is_previous_close(self):
        df = add_lagged_prices(make_ohlcv())
        assert df["lag1"].iloc[5] == df["Close"].iloc[4]

    def test_custom_lags(self):
        df = add_lagged_prices(make_ohlcv(), lags=3)
        assert "lag3" in df.columns
        assert "lag4" not in df.columns


# ---------------------------------------------------------------------------
# Volume features
# ---------------------------------------------------------------------------

class TestVolumeFeatures:
    def setup_method(self):
        self.df = add_volume_features(make_ohlcv())

    def test_columns_exist(self):
        for col in ("vol_ma5", "vol_ma20", "rel_volume"):
            assert col in self.df.columns

    def test_relative_volume_positive(self):
        rel = self.df["rel_volume"].dropna()
        assert (rel > 0).all()


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

class TestTarget:
    def test_target_is_next_close(self):
        df = add_target(make_ohlcv())
        # target at row i should equal Close at row i+1
        assert df["target"].iloc[0] == df["Close"].iloc[1]

    def test_last_target_is_nan(self):
        df = add_target(make_ohlcv())
        assert pd.isna(df["target"].iloc[-1])


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def setup_method(self):
        self.raw = make_ohlcv(200)
        self.df = build_features(self.raw)

    def test_no_nan_after_build(self):
        assert not self.df.isnull().any().any(), "build_features should drop all NaN rows"

    def test_feature_cols_present(self):
        missing = [c for c in FEATURE_COLS if c not in self.df.columns]
        assert not missing, f"Missing feature columns: {missing}"

    def test_target_present(self):
        assert "target" in self.df.columns

    def test_row_count_reduced(self):
        assert len(self.df) < len(self.raw), "Rows with NaN should have been removed"

    def test_monotone_index(self):
        assert self.df.index.is_monotonic_increasing
