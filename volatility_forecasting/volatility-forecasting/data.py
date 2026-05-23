"""
data.py — Market data loading and return computation.
"""

import yfinance as yf
import pandas as pd
import numpy as np


def fetch_returns(ticker: str = "^GSPC", start: str = "2010-01-01", end: str = "2024-12-31") -> pd.Series:
    """
    Download daily adjusted close prices and compute log returns.

    Log returns are used (not simple) because they are:
    - time-additive
    - approximately normally distributed for short horizons
    - standard in risk modelling
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.name = ticker
    return log_returns


def split_train_test(series: pd.Series, train_ratio: float = 0.8):
    """Split a time series into in-sample (train) and out-of-sample (test) windows."""
    n = len(series)
    cutoff = int(n * train_ratio)
    return series.iloc[:cutoff], series.iloc[cutoff:]
