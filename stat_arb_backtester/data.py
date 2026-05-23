"""
data.py — Price data loading for pairs trading.
"""

import yfinance as yf
import pandas as pd
import numpy as np


def fetch_prices(tickers: list, start: str = "2018-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Download daily adjusted close prices for a list of tickers.
    Returns a DataFrame with tickers as columns; inner join on dates.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])
    prices = raw.dropna()
    return prices


def compute_log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices)


def split_train_test(prices: pd.DataFrame, train_ratio: float = 0.60) -> tuple:
    """
    60/40 split: cointegration test and hedge ratio estimated in-sample;
    strategy traded fully out-of-sample.
    """
    n = len(prices)
    cutoff = int(n * train_ratio)
    return prices.iloc[:cutoff], prices.iloc[cutoff:]
