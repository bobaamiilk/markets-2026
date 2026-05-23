"""
models.py — Cointegration testing and spread / z-score computation.

Pairs Trading Logic:
  1. Test for cointegration using Engle-Granger two-step:
       - Regress log(P_A) on log(P_B) to estimate hedge ratio beta
       - Test residuals (the spread) for stationarity via ADF
  2. If spread is I(0), it mean-reverts -> tradeable signal
  3. Normalise spread to z-score using rolling mean/std:
       z_t = (spread_t - mu_rolling) / sigma_rolling
  4. Trade on z-score crossing entry/exit thresholds

Hedge Ratio (OLS):
  log(P_A) = alpha + beta * log(P_B) + eps
  beta is the number of units of B to short for each unit of A held long.
  This ratio is estimated in-sample and held fixed throughout the backtest.

Note: We use log prices throughout because:
  - Regression on log prices gives a returns-based hedge ratio
  - Spread in log-price space is a log return spread
  - Cointegration is more robust on log-price series for financial data
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint
from dataclasses import dataclass


@dataclass
class CointResult:
    ticker_a: str
    ticker_b: str
    hedge_ratio: float      # beta: units of B per unit of A
    intercept: float
    adf_statistic: float
    adf_pvalue: float
    coint_pvalue: float     # Engle-Granger cointegration test p-value
    is_cointegrated: bool   # at 5% level


def test_cointegration(log_prices: pd.DataFrame, ticker_a: str, ticker_b: str) -> CointResult:
    """
    Estimate hedge ratio via OLS and test residual stationarity.
    Also runs statsmodels coint() as a cross-check.
    """
    y = log_prices[ticker_a].values
    x = log_prices[ticker_b].values

    X = add_constant(x)
    res = OLS(y, X).fit()
    intercept = res.params[0]
    hedge_ratio = res.params[1]

    spread = y - (intercept + hedge_ratio * x)

    adf = adfuller(spread, autolag="AIC")
    adf_stat = adf[0]
    adf_pval = adf[1]

    _, coint_pval, _ = coint(log_prices[ticker_a], log_prices[ticker_b])

    return CointResult(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        adf_statistic=adf_stat,
        adf_pvalue=adf_pval,
        coint_pvalue=coint_pval,
        is_cointegrated=(adf_pval < 0.05),
    )


def compute_spread(log_prices: pd.DataFrame, cr: CointResult) -> pd.Series:
    """Compute the spread (residual) using in-sample estimated hedge ratio."""
    y = log_prices[cr.ticker_a]
    x = log_prices[cr.ticker_b]
    spread = y - (cr.intercept + cr.hedge_ratio * x)
    spread.name = f"spread_{cr.ticker_a}_{cr.ticker_b}"
    return spread


def compute_zscore(spread: pd.Series, window: int = 30) -> pd.Series:
    """
    Rolling z-score normalisation.
    mu and sigma are estimated on a rolling `window`-day lookback.
    Using rolling (not expanding) parameters keeps the signal responsive to regime changes.
    """
    mu = spread.rolling(window).mean()
    sigma = spread.rolling(window).std()
    z = (spread - mu) / sigma
    z.name = "zscore"
    return z
