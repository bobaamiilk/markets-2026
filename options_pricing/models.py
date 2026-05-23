"""
models.py — Black-Scholes closed-form option pricing.

Black-Scholes Assumptions:
  - Underlying follows geometric Brownian motion: dS = mu*S*dt + sigma*S*dW
  - No dividends, frictionless markets, constant risk-free rate and volatility
  - European exercise only

Formulae:
  d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
  d2 = d1 - sigma * sqrt(T)

  Call price = S * N(d1) - K * exp(-rT) * N(d2)
  Put price  = K * exp(-rT) * N(-d2) - S * N(-d1)

  where N(.) is the standard normal CDF.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


@dataclass
class BSResult:
    price: float
    d1: float
    d2: float
    option_type: str


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    """Compute d1 and d2 for Black-Scholes."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive.")
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> BSResult:
    """
    Closed-form Black-Scholes price.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiry in years
        r: Continuously compounded risk-free rate
        sigma: Annualised implied volatility
        option_type: 'call' or 'put'
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return BSResult(price=price, d1=d1, d2=d2, option_type=option_type)


def bs_price_surface(
    S: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> np.ndarray:
    """
    Compute BS price over a grid of strikes and maturities.
    Returns matrix of shape (len(maturities), len(strikes)).
    """
    surface = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            if T > 0:
                res = bs_price(S, K, T, r, sigma, option_type)
                surface[i, j] = res.price
    return surface
