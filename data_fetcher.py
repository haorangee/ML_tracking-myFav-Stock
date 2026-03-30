"""
data_fetcher.py
---------------
Downloads historical NVDA (Nvidia) stock price data using yfinance and saves it
to a CSV file under the `data/` directory.
"""

import os
import yfinance as yf
import pandas as pd

TICKER = "NVDA"
DEFAULT_START = "2015-01-01"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def fetch_nvda_data(
    ticker: str = TICKER,
    start: str = DEFAULT_START,
    end: str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Download historical daily OHLCV data for *ticker* from Yahoo Finance.

    Parameters
    ----------
    ticker:
        Stock symbol to download.  Defaults to ``"NVDA"``.
    start:
        Start date in ``YYYY-MM-DD`` format.
    end:
        End date in ``YYYY-MM-DD`` format.  ``None`` means today.
    save:
        When ``True`` the data frame is saved to ``data/<ticker>.csv``.

    Returns
    -------
    pd.DataFrame
        Daily OHLCV data with a ``DatetimeIndex``.
    """
    print(f"Fetching {ticker} data from {start} …")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol and date range.")

    # Flatten MultiIndex columns produced by some yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        df.to_csv(path)
        print(f"Data saved to {path}  ({len(df)} rows)")

    return df


def load_nvda_data(ticker: str = TICKER) -> pd.DataFrame:
    """Load previously saved data from ``data/<ticker>.csv``.

    Parameters
    ----------
    ticker:
        Stock symbol to load.

    Returns
    -------
    pd.DataFrame
        Daily OHLCV data with a ``DatetimeIndex``.
    """
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file '{path}' not found. Run fetch_nvda_data() first."
        )
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


if __name__ == "__main__":
    data = fetch_nvda_data()
    print(data.tail())
