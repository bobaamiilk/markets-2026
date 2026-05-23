"""
main.py — Volatility Forecasting + VaR Backtesting Pipeline

Pipeline:
  1. Download S&P 500 daily log returns
  2. Split 80/20 train/test
  3. Fit GARCH(1,1) and EGARCH(1,1) on training data
  4. Compute conditional volatility in-sample and out-of-sample via rolling re-estimation
  5. Compute parametric VaR at 95% and 99%
  6. Run Kupiec POF backtest on out-of-sample VaR violations
  7. Compare models on AIC / BIC / log-likelihood
  8. Generate all plots
"""

import warnings
import pandas as pd
warnings.filterwarnings("ignore")

from data import fetch_returns, split_train_test
from models import fit_garch, fit_egarch
from backtest import compute_var, backtest_var, compare_models
from plots import (
    plot_returns_and_vol,
    plot_var_backtest,
    plot_vol_comparison,
    plot_rolling_violation_rate,
)


def rolling_var_estimates(returns_full: pd.Series, train_end_idx: int, model_fn, scale=100.0):
    """
    Expanding window re-estimation: refit model at each step on all data up to t,
    record next-day conditional vol. This avoids look-ahead bias.

    For computational speed we use a large step size rather than daily re-fitting
    (daily re-fitting would be production-correct but slow for demo purposes).
    """
    from arch import arch_model
    import numpy as np

    step = 21  # refit monthly
    vols = []
    dates = []

    for end in range(train_end_idx, len(returns_full), step):
        window = returns_full.iloc[:end] * scale
        try:
            if model_fn == "garch":
                am = arch_model(window, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            else:
                am = arch_model(window, vol="EGARCH", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=1, reindex=False)
            forecasted_vol = float(np.sqrt(fc.variance.values[-1, 0])) / scale
            forecast_date = returns_full.index[min(end, len(returns_full)-1)]
            vols.append(forecasted_vol)
            dates.append(forecast_date)
        except Exception:
            continue

    return pd.Series(vols, index=dates)


def main():
    print("=" * 60)
    print("VOLATILITY FORECASTING + VaR BACKTESTING")
    print("=" * 60)

    # ── 1. Data ──────────────────────────────────────────────────
    print("\n[1/6] Downloading S&P 500 data (2010–2024)...")
    returns = fetch_returns("^GSPC", start="2010-01-01", end="2024-12-31")
    print(f"      {len(returns)} trading days loaded.")

    train, test = split_train_test(returns, train_ratio=0.80)
    train_end_idx = len(train)
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
    print("\n[4/6] Generating rolling out-of-sample VaR estimates (this takes ~60s)...")
    garch_oos_vol = rolling_var_estimates(returns, train_end_idx, "garch")
    egarch_oos_vol = rolling_var_estimates(returns, train_end_idx, "egarch")

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
