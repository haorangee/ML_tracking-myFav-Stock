"""
features.py
-----------
Feature engineering for stock price prediction.

Derived features:
- Moving averages (MA5, MA10, MA20, MA50)
- Exponential moving averages (EMA12, EMA26)
- MACD and signal line
- Bollinger Bands (upper / lower / width)
- Relative Strength Index (RSI-14)
- Daily return and volatility (5-day rolling std of return)
- Lagged close prices (lag1 … lag5)
- Target: next-day close price (``target``)
"""

import numpy as np
import pandas as pd


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple and exponential moving averages."""
    for window in (5, 10, 20, 50):
        df[f"ma{window}"] = df["Close"].rolling(window).mean()
    df["ema12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["Close"].ewm(span=26, adjust=False).mean()
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD line and signal line (9-day EMA of MACD)."""
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Upper and lower Bollinger Bands and band width."""
    rolling = df["Close"].rolling(window)
    df["bb_mid"] = rolling.mean()
    std = rolling.std()
    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)
    return df


def add_return_and_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Daily log-return and 5-day rolling volatility."""
    df["daily_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility_5d"] = df["daily_return"].rolling(5).std()
    return df


def add_lagged_prices(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Lagged close prices (lag1 = yesterday's close, etc.)."""
    for lag in range(1, lags + 1):
        df[f"lag{lag}"] = df["Close"].shift(lag)
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume moving averages and relative volume."""
    df["vol_ma5"] = df["Volume"].rolling(5).mean()
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["rel_volume"] = df["Volume"] / df["vol_ma20"]
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Next-day close price as the prediction target."""
    df["target"] = df["Close"].shift(-1)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature-engineering steps and drop rows with NaN values.

    Parameters
    ----------
    df:
        Raw OHLCV data frame (must contain ``Close`` and ``Volume`` columns).

    Returns
    -------
    pd.DataFrame
        Feature-enriched data frame with ``target`` column, NaN rows removed.
    """
    df = df.copy()
    df = add_moving_averages(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_rsi(df)
    df = add_return_and_volatility(df)
    df = add_lagged_prices(df)
    df = add_volume_features(df)
    df = add_target(df)
    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "ma5", "ma10", "ma20", "ma50",
    "ema12", "ema26",
    "macd", "macd_signal", "macd_hist",
    "bb_mid", "bb_upper", "bb_lower", "bb_width",
    "rsi",
    "daily_return", "volatility_5d",
    "lag1", "lag2", "lag3", "lag4", "lag5",
    "vol_ma5", "vol_ma20", "rel_volume",
]
