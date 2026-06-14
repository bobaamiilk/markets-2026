import pandas as pd


def build_from_raw():
    df = pd.read_csv("data/raw/voo_prices.csv", parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")

    returns = df["Adj Close"].pct_change().dropna()
    returns.to_parquet("data/processed/voo_returns.parquet")

    print("Processed dataset saved.")


if __name__ == "__main__":
    build_from_raw()