"""
monte_carlo.py — Monte Carlo option pricing via Geometric Brownian Motion simulation.

GBM Dynamics (risk-neutral measure):
  dS = r * S * dt + sigma * S * dW_t

Discretised with Euler-Maruyama:
  S_{t+dt} = S_t * exp((r - sigma^2/2) * dt + sigma * sqrt(dt) * Z)
  where Z ~ N(0,1)

The exp() form (log-normal step) is used rather than the raw Euler scheme
because it exactly preserves the log-normal distribution and avoids S going negative.

Variance Reduction:
  Antithetic variates: for each standard normal draw Z, also simulate -Z.
  This halves MC variance for roughly the same computational cost.
  The true price is the average of the correlated pair of payoffs.

Implied Volatility (IV) Solver:
  IV is the sigma that equates BS price to the observed market price.
  Solved numerically via Brentq root-finding (guaranteed convergence on [lower, upper]).
"""

import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass
from models import bs_price


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int
    n_steps: int
    ci_lower: float     # 95% confidence interval
    ci_upper: float


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int = 42,
    antithetic: bool = True,
) -> MCResult:
    """
    Price a European option via Monte Carlo simulation of GBM paths.

    Args:
        n_paths: Number of simulation paths (more = lower MC error)
        n_steps: Number of time steps per path (daily for annual T)
        antithetic: Use antithetic variates for variance reduction
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    if antithetic:
        # Generate half the paths; mirror with -Z
        half = n_paths // 2
        Z = rng.standard_normal((n_steps, half))
        Z_full = np.concatenate([Z, -Z], axis=1)   # shape (n_steps, n_paths)
    else:
        Z_full = rng.standard_normal((n_steps, n_paths))

    # Vectorised path simulation (all paths simultaneously)
    log_returns = drift + diffusion * Z_full     # shape (n_steps, n_paths)
    log_price_increments = np.cumsum(log_returns, axis=0)
    S_T = S * np.exp(log_price_increments[-1])   # terminal prices

    # Payoffs
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    # Discount to present value
    discounted = np.exp(-r * T) * payoffs

    price = discounted.mean()
    se = discounted.std() / np.sqrt(n_paths)

    return MCResult(
        price=price,
        std_error=se,
        n_paths=n_paths,
        n_steps=n_steps,
        ci_lower=price - 1.96 * se,
        ci_upper=price + 1.96 * se,
    )


def mc_path_sample(
    S: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 20,
    n_steps: int = 252,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate a small sample of full GBM paths for visualisation.
    Returns shape (n_steps+1, n_paths) — includes initial price.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_steps, n_paths))
    log_rets = drift + diffusion * Z
    log_cum = np.cumsum(log_rets, axis=0)

    paths = S * np.exp(log_cum)
    paths = np.vstack([np.full((1, n_paths), S), paths])
    return paths


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
) -> float:
    """
    Implied volatility via Brentq root-finding on the BS price equation.

    Brentq guarantees convergence when the function changes sign on [lo, hi].
    IV must lie in (0, 5) = (0%, 500%) for any reasonable equity option.

    Returns NaN if the market price is outside the arbitrage-free bounds
    (below intrinsic value or above theoretical max).
    """
    intrinsic = max(S - K * np.exp(-r * T), 0) if option_type == "call" else max(K * np.exp(-r * T) - S, 0)
    if market_price < intrinsic:
        return float("nan")

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type).price - market_price

    try:
        iv = brentq(objective, 1e-6, 5.0, xtol=tol, maxiter=200)
        return iv
    except ValueError:
        return float("nan")


def iv_surface(
    market_prices: np.ndarray,
    S: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float,
    option_type: str = "call",
) -> np.ndarray:
    """
    Compute implied volatility for a grid of (maturity, strike) market prices.
    market_prices: shape (len(maturities), len(strikes))
    """
    iv_grid = np.zeros_like(market_prices)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            if T > 0:
                iv_grid[i, j] = implied_vol(market_prices[i, j], S, K, T, r, option_type)
    return iv_grid
