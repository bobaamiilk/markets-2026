"""
data.py — Load VOO price data from a locally-downloaded CSV.

Why this exists:
    yfinance frequently rate-limits (YFRateLimitError) on repeated calls.
    Instead of hitting the API every run, we load a one-time manual CSV export.

Expected CSV format (this is exactly what yfinance produces when you do
`df.to_csv("voo_prices.csv")` on a multi-index download):

    Price,Adj Close,Close,High,Low,Open,Volume
    Ticker,VOO,VOO,VOO,VOO,VOO,VOO
    Date,,,,,,
    2010-09-09,68.5,68.5,69.0,68.0,68.2,500000
    2010-09-10,68.8,68.8,69.1,68.5,68.6,510000
    ...

Rows 1 and 2 (the "Ticker" and "Date" label rows) are metadata artefacts of
yfinance's MultiIndex columns and must be skipped. The first column header
is literally the string "Price" but its values are dates.
"""

import pandas as pd
import os

DATA_PATH = "voo_prices.csv"


def fetch_returns(path: str = DATA_PATH) -> pd.Series:
    """
    Load VOO daily close prices from CSV and compute log returns.

    Returns:
        pd.Series of log returns, indexed by date, named "VOO".
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing dataset: {path}\n"
            f"Download it once with:\n"
            f'  python3 -c "import yfinance as yf; '
            f"yf.download('VOO', start='2010-01-01').to_csv('{path}')\""
        )

    # Skip the 'Ticker' and 'Date' metadata rows (rows 1 and 2, 0-indexed)
    df = pd.read_csv(path, skiprows=[1, 2])

    # The first column holds dates but is labelled "Price" (yfinance quirk).
    # Rename it explicitly rather than relying on position.
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    # Ensure numeric (CSV round-trip can leave strings)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Pick price column: prefer Adj Close, fall back to Close
    if "Adj Close" in df.columns and df["Adj Close"].notna().any():
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        raise ValueError(f"No usable price column found. Columns: {df.columns.tolist()}")

    prices = df[price_col].dropna()

    if prices.empty:
        raise RuntimeError(
            f"Loaded {path} but found 0 valid price rows after parsing. "
            f"Check the CSV has real data rows below the header."
        )

    # Use log returns (time-additive, ~normal) — consistent with models.py
    import numpy as np
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.name = "VOO"

    return log_returns


def split_train_test(returns: pd.Series, train_ratio: float = 0.8) -> tuple:
    """80/20 chronological split — no shuffling, this is a time series."""
    if len(returns) < 10:
        raise ValueError(f"Not enough data to split: only {len(returns)} observations.")

    split_idx = int(len(returns) * train_ratio)
    train = returns.iloc[:split_idx]
    test = returns.iloc[split_idx:]
    return train, test
