import pandas as pd
from data import fetch_returns, split_train_test
from models import fit_garch, fit_egarch
from backtest import compute_var, backtest_var, compare_models
from plots import *


def main():
    print("=" * 60)
    print("VOLATILITY FORECASTING + VaR BACKTESTING (VOO OFFLINE)")
    print("=" * 60)

    # ─────────────────────────────
    # 1. DATA (offline)
    # ─────────────────────────────
    returns = fetch_returns()

    if returns is None or returns.empty:
        raise RuntimeError("Empty dataset loaded.")

    train, test = split_train_test(returns)

    if len(train) == 0 or len(test) == 0:
        raise RuntimeError("Train/test split failed.")

    print(f"\nLoaded {len(returns)} observations (VOO)")
    print(f"Train: {train.index[0].date()} → {train.index[-1].date()}")
    print(f"Test:  {test.index[0].date()} → {test.index[-1].date()}")

    # ─────────────────────────────
    # 2. MODELS
    # ─────────────────────────────
    print("\nFitting models...")

    garch = fit_garch(train)
    egarch = fit_egarch(train)

    print("\nModel comparison:")
    print(compare_models({
        "GARCH": garch,
        "EGARCH": egarch
    }))

    # ─────────────────────────────
    # 3. ROLLING VOL + VaR
    # ─────────────────────────────
    from rolling import rolling_var_estimates

    garch_vol = rolling_var_estimates(returns, len(train), "garch")
    egarch_vol = rolling_var_estimates(returns, len(train), "egarch")

    garch_var = compute_var(garch_vol)
    egarch_var = compute_var(egarch_vol)

    # ─────────────────────────────
    # 4. BACKTEST
    # ─────────────────────────────
    print("\nBacktesting VaR...")

    print("\nGARCH:")
    print(backtest_var(test, garch_var).to_string(index=False))

    print("\nEGARCH:")
    print(backtest_var(test, egarch_var).to_string(index=False))

    # ─────────────────────────────
    # 5. PLOTS
    # ─────────────────────────────
    print("\nGenerating plots...")

    plot_returns_and_vol(returns, garch.conditional_vol, egarch.conditional_vol)
    plot_var_backtest(test, garch_var, "GARCH")
    plot_var_backtest(test, egarch_var, "EGARCH")
    plot_vol_comparison(garch.conditional_vol, egarch.conditional_vol)
    plot_rolling_violation_rate(test, garch_var, "GARCH")
    plot_rolling_violation_rate(test, egarch_var, "EGARCH")

    print("\nDone (offline VOO pipeline).")


if __name__ == "__main__":
    main()