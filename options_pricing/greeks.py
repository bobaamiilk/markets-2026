"""
greeks.py — Analytical Black-Scholes Greeks.

Greeks measure sensitivity of option price to model parameters:

  Delta (Δ) = dV/dS          — price sensitivity to spot
  Gamma (Γ) = d²V/dS²        — delta sensitivity to spot (convexity)
  Vega  (ν) = dV/d(sigma)    — price sensitivity to volatility
  Theta (Θ) = dV/dT          — price decay per calendar day
  Rho   (ρ) = dV/dr          — price sensitivity to interest rate

Key intuitions for trading:
  - Delta: directional exposure; delta-hedging eliminates first-order price risk
  - Gamma: cost of delta-hedging; high gamma = large rebalancing requirement
  - Vega:  vol exposure; long options = long vega (benefit from rising vol)
  - Theta: time decay; short options = long theta (collect premium daily)
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float    # per calendar day
    rho: float
    option_type: str


def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> Greeks:
    """Analytical Black-Scholes Greeks."""
    if T <= 0:
        T = 1e-6  # near-expiry

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    npdf_d1 = norm.pdf(d1)   # standard normal PDF at d1

    # ── Delta ────────────────────────────────────────────────
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1   # = N(d1) - 1

    # ── Gamma (same for call and put by put-call parity) ─────
    gamma = npdf_d1 / (S * sigma * np.sqrt(T))

    # ── Vega (same for call and put) ─────────────────────────
    # dV/d(sigma); divide by 100 to express per 1% vol move
    vega = S * npdf_d1 * np.sqrt(T) / 100

    # ── Theta (per calendar day) ─────────────────────────────
    common_theta = -(S * npdf_d1 * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        theta = (common_theta - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (common_theta + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # ── Rho (per 1% rate move) ───────────────────────────────
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return Greeks(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        option_type=option_type,
    )


def greeks_surface(
    S: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float,
    sigma: float,
    option_type: str = "call",
    greek: str = "delta",
) -> np.ndarray:
    """
    Compute a single Greek over a (maturity, strike) grid.
    Returns shape (len(maturities), len(strikes)).
    """
    surface = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            if T > 0:
                g = compute_greeks(S, K, T, r, sigma, option_type)
                surface[i, j] = getattr(g, greek)
    return surface
