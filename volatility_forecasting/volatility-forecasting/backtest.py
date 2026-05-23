"""
backtest.py — Value-at-Risk computation and violation backtesting.

VaR Definition:
    VaR_alpha(t) = -sigma_t * z_alpha
    where z_alpha is the alpha-quantile of N(0,1).
    i.e. with probability alpha, the loss will NOT exceed VaR_alpha.

Violation (exceedance):
    r_t < -VaR_alpha(t)  =>  violation at time t

Kupiec POF Test (Proportion Of Failures):
    Tests whether the empirical violation rate matches the nominal level.
    Under H0: violations ~ Binomial(T, 1-alpha)
    LR_pof = -2 * ln[ p0^n0 * (1-p0)^n1 / (p_hat^n0 * (1-p_hat)^n1) ]
    LR_pof ~ chi-squared(1) under H0
"""

import numpy as np
import pandas as pd
from scipy import stats
from models import FitResult


def compute_var(cond_vol: pd.Series, confidence_levels=(0.95, 0.99)) -> pd.DataFrame:
    """
    Parametric VaR using GARCH-estimated conditional volatility.
    Returns daily VaR (positive number = potential loss magnitude).
    """
    results = {}
    for alpha in confidence_levels:
        z = stats.norm.ppf(1 - alpha)   # negative quantile, e.g. -1.645 at 95%
        results[f"VaR_{int(alpha*100)}"] = -cond_vol * z
    return pd.DataFrame(results, index=cond_vol.index)


def backtest_var(returns: pd.Series, var_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count VaR violations and compute empirical violation rates.
    Also runs Kupiec POF likelihood ratio test.

    Returns a summary DataFrame with:
    - expected_rate: nominal level (e.g. 0.05 for 95% VaR)
    - actual_rate: empirical violation rate
    - violations: count
    - kupiec_pvalue: p-value under H0 (correct model)
    """
    # Align returns and VaR estimates on common dates
    common = returns.index.intersection(var_df.index)
    r = returns.loc[common]
    var = var_df.loc[common]

    rows = []
    T = len(r)

    for col in var.columns:
        level_str = col.split("_")[1]   # "95" or "99"
        alpha = int(level_str) / 100    # 0.95 or 0.99
        expected_rate = 1 - alpha

        violations = (r < -var[col]).sum()
        actual_rate = violations / T

        # Kupiec test
        p0 = expected_rate
        p1 = actual_rate if actual_rate > 0 else 1e-10
        n1 = violations
        n0 = T - violations

        lr = -2 * (
            n1 * np.log(p0) + n0 * np.log(1 - p0)
            - n1 * np.log(p1) - n0 * np.log(1 - p1)
        )
        p_value = 1 - stats.chi2.cdf(lr, df=1)

        rows.append({
            "confidence": f"{level_str}%",
            "expected_rate": expected_rate,
            "actual_rate": round(actual_rate, 4),
            "violations": int(violations),
            "total_obs": T,
            "kupiec_lr": round(lr, 4),
            "kupiec_pvalue": round(p_value, 4),
            "model_rejected": p_value < 0.05,
        })

    return pd.DataFrame(rows)


def compare_models(results: dict) -> pd.DataFrame:
    """Compare GARCH and EGARCH on information criteria and log-likelihood."""
    rows = []
    for name, fit in results.items():
        rows.append({
            "model": name,
            "log_likelihood": round(fit.log_likelihood, 2),
            "AIC": round(fit.aic, 2),
            "BIC": round(fit.bic, 2),
        })
    df = pd.DataFrame(rows).set_index("model")
    df["best_AIC"] = df["AIC"] == df["AIC"].min()
    df["best_BIC"] = df["BIC"] == df["BIC"].min()
    return df
