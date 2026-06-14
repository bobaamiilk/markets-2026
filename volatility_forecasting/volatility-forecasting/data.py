import pandas as pd
import os


DATA_PATH = "data/raw/voo_prices.csv"

import pandas as pd

import pandas as pd

def fetch_returns():
    df = pd.read_csv("voo_prices.csv")

    df.columns = df.columns.str.strip()

    # standardise column names
    df = df.rename(columns={
        "Vol.": "Volume",
        "Change %": "ChangePct"
    })

    # detect price column
    if "Price" in df.columns:
        price_col = "Price"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        raise ValueError(f"No usable price column found. Columns: {df.columns.tolist()}")

    # clean numeric noise (important)
    df[price_col] = (
        df[price_col]
        .astype(str)
        .str.replace(",", "")
        .str.replace("%", "")
    )

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])

    # if Date missing, assume already sorted time series
    returns = df[price_col].pct_change().dropna()
    returns.name = "VOO"

    return returns

def split_train_test(returns: pd.Series, train_ratio=0.8):
    split_idx = int(len(returns) * train_ratio)

    train = returns.iloc[:split_idx]
    test = returns.iloc[split_idx:]

    return train, test