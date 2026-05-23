"""
main.py — Statistical Arbitrage Backtester

Pipeline:
  1. Download prices for chosen pairs (GLD/SLV and XOM/CVX)
  2. Split 60/40 train/test
  3. Test cointegration on training data
  4. Estimate hedge ratio (OLS on log prices)
  5. Compute out-of-sample spread and rolling z-score
  6. Run backtest with z-score entry/exit rules
  7. Report Sharpe, drawdown, Calmar, win rate
  8. Generate all plots
"""

import warnings
import pandas as pd
warnings.filterwarnings("ignore")

from data import fetch_prices, compute_log_prices, split_train_test
from models import test_cointegration, compute_spread, compute_zscore
from backtest import run_backtest
from plots import (
    plot_price_series,
    plot_zscore,
    plot_cumulative_pnl,
    plot_positions,
    plot_return_distribution,
)

# ── Pairs to test ────────────────────────────────────────────────────────────
# GLD/SLV: Gold and Silver ETFs — commodity co-movement
# XOM/CVX: ExxonMobil and Chevron — integrated oil majors
# KO/PEP:  Coca-Cola and PepsiCo — consumer staples
PAIRS = [
    ("GLD", "SLV"),
    ("XOM", "CVX"),
    ("KO",  "PEP"),
]

ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 4.0
ZSCORE_WINDOW = 30    # rolling window for z-score normalisation
TRAIN_RATIO = 0.60


def run_pair(ticker_a: str, ticker_b: str):
    print(f"\n{'='*55}")
    print(f"  Pair: {ticker_a} / {ticker_b}")
    print(f"{'='*55}")

    # ── 1. Data ──────────────────────────────────────────────
    prices = fetch_prices([ticker_a, ticker_b], start="2015-01-01", end="2024-12-31")
    log_prices = compute_log_prices(prices)

    train_prices, test_prices = split_train_test(prices, TRAIN_RATIO)
    train_log, test_log = split_train_test(log_prices, TRAIN_RATIO)

    print(f"  Train: {train_log.index[0].date()} – {train_log.index[-1].date()} ({len(train_log)} obs)")
    print(f"  Test:  {test_log.index[0].date()} – {test_log.index[-1].date()} ({len(test_log)} obs)")

    # ── 2. Cointegration test ────────────────────────────────
    cr = test_cointegration(train_log, ticker_a, ticker_b)
    print(f"\n  Cointegration test:")
    print(f"    Hedge ratio (beta):  {cr.hedge_ratio:.4f}")
    print(f"    ADF p-value:         {cr.adf_pvalue:.4f}")
    print(f"    Coint p-value:       {cr.coint_pvalue:.4f}")
    print(f"    Cointegrated (5%):   {cr.is_cointegrated}")

    if not cr.is_cointegrated:
        print(f"  [WARN] Pair not cointegrated at 5% — backtest proceeds but signal may be spurious.")

    # ── 3. Out-of-sample spread and z-score ──────────────────
    # Apply in-sample hedge ratio to OOS data (no re-estimation)
    spread_oos = compute_spread(test_log, cr)
    zscore_oos = compute_zscore(spread_oos, window=ZSCORE_WINDOW)

    # ── 4. Backtest ──────────────────────────────────────────
    result = run_backtest(
        log_prices_oos=test_log,
        prices_oos=test_prices,
        zscore_oos=zscore_oos,
        hedge_ratio=cr.hedge_ratio,
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        stop_z=STOP_Z,
    )

    print(f"\n  Performance (out-of-sample):")
    print(f"    Sharpe ratio:        {result.sharpe:.3f}")
    print(f"    Annualised return:   {result.annualised_return:.2%}")
    print(f"    Max drawdown:        {result.max_drawdown:.4f}")
    print(f"    Calmar ratio:        {result.calmar:.3f}")
    print(f"    Win rate:            {result.win_rate:.2%}" if result.total_trades > 0 else "    Win rate:            n/a")
    print(f"    Total trades:        {result.total_trades}")

    # ── 5. Plots ─────────────────────────────────────────────
    plot_price_series(test_prices, ticker_a, ticker_b, label="(OOS)")
    plot_zscore(zscore_oos, ENTRY_Z, EXIT_Z)
    plot_cumulative_pnl(result.cum_pnl, result.daily_pnl, result.sharpe, result.max_drawdown)
    plot_positions(result.positions, zscore_oos)
    plot_return_distribution(result.daily_pnl)

    return result


def main():
    print("=" * 55)
    print("STATISTICAL ARBITRAGE BACKTESTER")
    print("=" * 55)

    summary = []
    for a, b in PAIRS:
        try:
            res = run_pair(a, b)
            summary.append({
                "pair": f"{a}/{b}",
                "sharpe": res.sharpe,
                "ann_return": f"{res.annualised_return:.2%}",
                "max_drawdown": res.max_drawdown,
                "calmar": res.calmar,
                "trades": res.total_trades,
            })
        except Exception as e:
            print(f"  [ERROR] {a}/{b}: {e}")

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    summary_df = pd.DataFrame(summary).set_index("pair")
    print(summary_df.to_string())
    print(f"\nOutputs saved to ./outputs/")


if __name__ == "__main__":
    main()
