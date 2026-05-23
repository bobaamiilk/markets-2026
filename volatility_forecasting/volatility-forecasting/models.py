"""
models.py — GARCH(1,1) and EGARCH(1,1) volatility model fitting.

GARCH(1,1):
    sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}

EGARCH(1,1) (Nelson 1991):
    log(sigma^2_t) = omega + alpha * (|z_{t-1}| - E|z|) + gamma * z_{t-1} + beta * log(sigma^2_{t-1})
    - Captures asymmetric leverage effect: negative shocks increase vol more than positive ones.
    - Log specification guarantees sigma^2_t > 0 without parameter constraints.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from dataclasses import dataclass
from typing import Any


@dataclass
class FitResult:
    name: str
    model_fit: Any          # ARCHModelResult
    conditional_vol: pd.Series
    aic: float
    bic: float
    log_likelihood: float


def fit_garch(returns: pd.Series, scale: float = 100.0) -> FitResult:
    """
    Fit GARCH(1,1) with Normal innovations.
    Scaling returns by 100 improves numerical stability of the optimiser.
    """
    r = returns * scale
    am = arch_model(r, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)

    cond_vol = res.conditional_volatility / scale  # back to raw scale

    return FitResult(
        name="GARCH(1,1)",
        model_fit=res,
        conditional_vol=cond_vol,
        aic=res.aic,
        bic=res.bic,
        log_likelihood=res.loglikelihood,
    )


def fit_egarch(returns: pd.Series, scale: float = 100.0) -> FitResult:
    """
    Fit EGARCH(1,1) with Normal innovations.
    gamma < 0 confirms leverage effect (bad news raises vol more than good news).
    """
    r = returns * scale
    am = arch_model(r, vol="EGARCH", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)

    cond_vol = res.conditional_volatility / scale

    return FitResult(
        name="EGARCH(1,1)",
        model_fit=res,
        conditional_vol=cond_vol,
        aic=res.aic,
        bic=res.bic,
        log_likelihood=res.loglikelihood,
    )


def forecast_variance(fit: FitResult, horizon: int = 10) -> np.ndarray:
    """One-step-ahead variance forecasts for the next `horizon` periods."""
    fc = fit.model_fit.forecast(horizon=horizon, reindex=False)
    return np.sqrt(fc.variance.values[-1]) / 100  # annualise-ready sigma
