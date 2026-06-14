"""
main.py — Volatility Forecasting + VaR Backtesting (VOO, offline CSV)

Pipeline:
  1. Load VOO log returns from local CSV (no network calls)
  2. Split 80/20 train/test
  3. Fit GARCH(1,1) and EGARCH(1,1) on training data
  4. Roll forward through test period, refitting monthly (no look-ahead)
  5. Compute parametric VaR at 95% and 99%
  6. Backtest VaR violations with Kupiec POF test
  7. Compare models on AIC/BIC/log-likelihood
  8. Generate all plots
"""

import warnings
warnings.filterwarnings("ignore")

from data import fetch_returns, split_train_test
from models import fit_garch, fit_egarch
from backtest import compute_var, backtest_var, compare_models
from rolling import rolling_var_estimates
from plots import (
    plot_returns_and_vol,
    plot_var_backtest,
    plot_vol_comparison,
    plot_rolling_violation_rate,
)


def main():
    print("=" * 60)
    print("VOLATILITY FORECASTING + VaR BACKTESTING (VOO, OFFLINE)")
    print("=" * 60)

    # ── 1. Data ──────────────────────────────────────────────────
    print("\n[1/6] Loading VOO returns from voo_prices.csv...")
    returns = fetch_returns()

    if returns is None or returns.empty:
        raise RuntimeError("Empty dataset loaded.")

    train, test = split_train_test(returns, train_ratio=0.80)

    if len(train) == 0 or len(test) == 0:
        raise RuntimeError("Train/test split failed — check data length.")

    print(f"      Loaded {len(returns)} observations (VOO)")
    print(f"      Train: {train.index[0].date()} – {train.index[-1].date()} ({len(train)} obs)")
    print(f"      Test:  {test.index[0].date()} – {test.index[-1].date()} ({len(test)} obs)")

    # ── 2. In-sample fit ─────────────────────────────────────────
    print("\n[2/6] Fitting GARCH(1,1) and EGARCH(1,1) on training data...")
    garch_fit = fit_garch(train)
    egarch_fit = fit_egarch(train)
    print(f"      GARCH  | AIC={garch_fit.aic:.1f} | BIC={garch_fit.bic:.1f} | LogL={garch_fit.log_likelihood:.1f}")
    print(f"      EGARCH | AIC={egarch_fit.aic:.1f} | BIC={egarch_fit.bic:.1f} | LogL={egarch_fit.log_likelihood:.1f}")

    # ── 3. Model comparison ──────────────────────────────────────
    print("\n[3/6] Model comparison (information criteria):")
    comparison = compare_models({"GARCH(1,1)": garch_fit, "EGARCH(1,1)": egarch_fit})
    print(comparison.to_string())

    # ── 4. Rolling out-of-sample vol forecasts ───────────────────
    print("\n[4/6] Generating rolling out-of-sample VaR estimates (this takes ~30-60s)...")
    garch_oos_vol = rolling_var_estimates(returns, len(train), "garch")
    egarch_oos_vol = rolling_var_estimates(returns, len(train), "egarch")
    print(f"      GARCH rolling forecasts:  {len(garch_oos_vol)}")
    print(f"      EGARCH rolling forecasts: {len(egarch_oos_vol)}")

    garch_var = compute_var(garch_oos_vol)
    egarch_var = compute_var(egarch_oos_vol)

    # ── 5. Backtest ──────────────────────────────────────────────
    print("\n[5/6] VaR Backtesting (Kupiec POF Test):")
    print("\n  GARCH(1,1):")
    garch_bt = backtest_var(test, garch_var)
    print(garch_bt.to_string(index=False))

    print("\n  EGARCH(1,1):")
    egarch_bt = backtest_var(test, egarch_var)
    print(egarch_bt.to_string(index=False))

    # ── 6. Plots ─────────────────────────────────────────────────
    print("\n[6/6] Generating plots...")
    plot_returns_and_vol(returns, garch_fit.conditional_vol, egarch_fit.conditional_vol)
    plot_var_backtest(test, garch_var, "GARCH11")
    plot_var_backtest(test, egarch_var, "EGARCH11")
    plot_vol_comparison(garch_fit.conditional_vol, egarch_fit.conditional_vol)
    plot_rolling_violation_rate(test, garch_var, "GARCH11")
    plot_rolling_violation_rate(test, egarch_var, "EGARCH11")

    print("\n" + "=" * 60)
    print("Done. Outputs saved to ./outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
