"""
main.py — Options Pricing Engine

Pipeline:
  1. Compute Black-Scholes prices for a range of strikes and maturities
  2. Simulate GBM paths via Monte Carlo; compare MC price to BS
  3. Compute all Greeks analytically; plot vs spot
  4. Generate 3D pricing surface (call price vs strike x maturity)
  5. Solve for implied volatility and plot IV smile
  6. Report numerical accuracy of MC estimator
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from models import bs_price, bs_price_surface
from greeks import compute_greeks, greeks_surface
from monte_carlo import mc_price, mc_path_sample, implied_vol, iv_surface
from plots import (
    plot_gbm_paths,
    plot_price_surface,
    plot_greeks_vs_spot,
    plot_bs_vs_mc,
    plot_iv_smile,
)

# ── Base parameters ───────────────────────────────────────────────────────────
S     = 500.0   # Current spot price (e.g. S&P 500 / 2)
K_atm = 500.0   # At-the-money strike
T     = 1.0     # 1 year to expiry
r     = 0.05    # Risk-free rate (5%)
sigma = 0.20    # 20% annualised vol (typical for equity index)
OPT   = "call"


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    print("=" * 60)
    print("OPTIONS PRICING ENGINE")
    print("=" * 60)

    # ── 1. Single option price and Greeks ────────────────────────
    section("1. Black-Scholes ATM Call")
    res = bs_price(S, K_atm, T, r, sigma, OPT)
    print(f"  Price:  {res.price:.4f}")
    print(f"  d1:     {res.d1:.4f}")
    print(f"  d2:     {res.d2:.4f}")

    g = compute_greeks(S, K_atm, T, r, sigma, OPT)
    print(f"\n  Greeks (ATM, T=1y, σ=20%):")
    print(f"    Delta:  {g.delta:+.4f}  (≈ prob ITM at expiry)")
    print(f"    Gamma:  {g.gamma:.6f} (sensitivity of delta to spot)")
    print(f"    Vega:   {g.vega:.4f}  (price change per 1% vol move)")
    print(f"    Theta:  {g.theta:.4f}  (daily time decay)")
    print(f"    Rho:    {g.rho:.4f}  (price change per 1% rate move)")

    # ── 2. Monte Carlo price comparison ──────────────────────────
    section("2. Monte Carlo vs Black-Scholes")
    mc_res = mc_price(S, K_atm, T, r, sigma, OPT, n_paths=200_000, antithetic=True)
    print(f"  MC price:     {mc_res.price:.4f}  ±{mc_res.std_error:.4f} (1σ SE)")
    print(f"  BS price:     {res.price:.4f}")
    print(f"  Abs error:    {abs(mc_res.price - res.price):.6f}")
    print(f"  95% CI:       [{mc_res.ci_lower:.4f}, {mc_res.ci_upper:.4f}]")
    print(f"  BS in CI:     {mc_res.ci_lower <= res.price <= mc_res.ci_upper}")

    # Strike range for comparison
    strikes_range = np.linspace(350, 650, 25)
    bs_prices_range = np.array([bs_price(S, K, T, r, sigma, OPT).price for K in strikes_range])
    mc_results = [mc_price(S, K, T, r, sigma, OPT, n_paths=50_000) for K in strikes_range]
    mc_prices_range = np.array([m.price for m in mc_results])
    mc_errors_range = np.array([m.std_error for m in mc_results])

    plot_bs_vs_mc(strikes_range, bs_prices_range, mc_prices_range, mc_errors_range, S, T, sigma, OPT)

    # ── 3. GBM path visualisation ─────────────────────────────────
    section("3. GBM Path Simulation")
    paths = mc_path_sample(S, T, r, sigma, n_paths=30, n_steps=252)
    print(f"  Simulated 30 paths over T={T}y with {252} steps")
    print(f"  Terminal price range: [{paths[-1].min():.2f}, {paths[-1].max():.2f}]")
    plot_gbm_paths(paths, K_atm, S, T)

    # ── 4. Pricing surface ────────────────────────────────────────
    section("4. Pricing Surface (Strike × Maturity)")
    strikes_surf = np.linspace(350, 650, 30)
    maturities_surf = np.linspace(0.05, 2.0, 20)
    price_surf = bs_price_surface(S, strikes_surf, maturities_surf, r, sigma, OPT)
    print(f"  Surface computed: {len(maturities_surf)} maturities × {len(strikes_surf)} strikes")
    plot_price_surface(strikes_surf, maturities_surf, price_surf,
                       title="Call Price Surface (BS)", zlabel="Call Price", fname="price_surface.png")

    # ── 5. Greeks surface and vs-spot ─────────────────────────────
    section("5. Greeks")
    spot_range = np.linspace(300, 700, 200)
    greeks_dict = {
        "delta": np.array([compute_greeks(s, K_atm, T, r, sigma, OPT).delta for s in spot_range]),
        "gamma": np.array([compute_greeks(s, K_atm, T, r, sigma, OPT).gamma for s in spot_range]),
        "vega":  np.array([compute_greeks(s, K_atm, T, r, sigma, OPT).vega  for s in spot_range]),
        "theta": np.array([compute_greeks(s, K_atm, T, r, sigma, OPT).theta for s in spot_range]),
    }
    plot_greeks_vs_spot(spot_range, greeks_dict, K_atm, T, sigma, OPT)

    # Delta surface
    delta_surf = greeks_surface(S, strikes_surf, maturities_surf, r, sigma, OPT, "delta")
    plot_price_surface(strikes_surf, maturities_surf, delta_surf,
                       title="Delta Surface (Call)", zlabel="Delta", fname="delta_surface.png")

    gamma_surf = greeks_surface(S, strikes_surf, maturities_surf, r, sigma, OPT, "gamma")
    plot_price_surface(strikes_surf, maturities_surf, gamma_surf,
                       title="Gamma Surface (Call)", zlabel="Gamma", fname="gamma_surface.png")

    # ── 6. Implied Volatility smile ───────────────────────────────
    section("6. Implied Volatility Smile")
    # Simulate a market with a vol skew: OTM puts more expensive (realistic)
    # We generate synthetic market prices using a known sigma_skew function
    strikes_smile = np.linspace(380, 620, 20)
    sigma_skew = 0.20 + 0.15 * np.exp(-0.5 * ((strikes_smile - S) / 80)**2) * (strikes_smile < S).astype(float)
    sigma_skew += 0.05 * ((strikes_smile - S) / S)**2   # quadratic smile

    market_prices = np.array([
        bs_price(S, K, T, r, sk, OPT).price
        for K, sk in zip(strikes_smile, sigma_skew)
    ])

    # Now recover IV from these "market prices"
    ivs = np.array([
        implied_vol(mp, S, K, T, r, OPT)
        for mp, K in zip(market_prices, strikes_smile)
    ])

    print(f"  IV range: [{ivs[~np.isnan(ivs)].min():.1%}, {ivs[~np.isnan(ivs)].max():.1%}]")
    print(f"  ATM IV:   {ivs[np.argmin(np.abs(strikes_smile - S))]:.1%}")
    plot_iv_smile(strikes_smile, ivs, S, label=f"(S={S}, T={T}y)")

    print("\n" + "=" * 60)
    print("Done. All outputs saved to ./outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
