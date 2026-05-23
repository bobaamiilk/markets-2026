"""
plots.py — Visualisations for the options pricing engine.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import os

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(STYLE)


def plot_gbm_paths(paths: np.ndarray, K: float, S: float, T: float):
    """Plot a sample of GBM simulation paths with strike line."""
    time_axis = np.linspace(0, T, paths.shape[0])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Simulated GBM Price Paths (Sample)", fontsize=13, fontweight="bold")

    for i in range(paths.shape[1]):
        ax.plot(time_axis, paths[:, i], linewidth=0.6, alpha=0.55)

    ax.axhline(K, color="black", linewidth=1.5, linestyle="--", label=f"Strike K={K:.0f}")
    ax.axhline(S, color="gray", linewidth=0.8, linestyle=":", label=f"Spot S={S:.0f}")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Price")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/gbm_paths.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] Saved gbm_paths.png")


def plot_price_surface(
    strikes: np.ndarray,
    maturities: np.ndarray,
    surface: np.ndarray,
    title: str,
    zlabel: str,
    fname: str,
):
    """3D surface plot over (strike, maturity) grid."""
    K_grid, T_grid = np.meshgrid(strikes, maturities)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    surf = ax.plot_surface(K_grid, T_grid, surface, cmap="viridis", alpha=0.90, linewidth=0)
    fig.colorbar(surf, ax=ax, shrink=0.5, label=zlabel)

    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity T (yrs)")
    ax.set_zlabel(zlabel)
    ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {fname}")


def plot_greeks_vs_spot(
    spot_range: np.ndarray,
    greeks_dict: dict,
    K: float,
    T: float,
    sigma: float,
    option_type: str,
):
    """Plot each Greek as a function of spot price."""
    greek_names = list(greeks_dict.keys())
    n = len(greek_names)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"Greeks vs Spot — {option_type.capitalize()} K={K:.0f}, T={T:.2f}y, σ={sigma:.0%}",
        fontsize=13, fontweight="bold"
    )

    colors = ["#2980b9", "#e74c3c", "#27ae60", "#f39c12", "#8e44ad"]

    for ax, name, color in zip(axes, greek_names, colors):
        ax.plot(spot_range, greeks_dict[name], color=color, linewidth=1.5)
        ax.axvline(K, color="gray", linewidth=0.8, linestyle="--", alpha=0.7, label=f"K={K:.0f}")
        ax.set_ylabel(name.capitalize())
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", linewidth=0.4)

    axes[-1].set_xlabel("Spot Price S")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/greeks_vs_spot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] Saved greeks_vs_spot.png")


def plot_bs_vs_mc(
    strikes: np.ndarray,
    bs_prices: np.ndarray,
    mc_prices: np.ndarray,
    mc_errors: np.ndarray,
    S: float,
    T: float,
    sigma: float,
    option_type: str,
):
    """Compare Black-Scholes closed-form vs Monte Carlo prices across strikes."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle(
        f"Black-Scholes vs Monte Carlo — {option_type.capitalize()} S={S}, T={T}y, σ={sigma:.0%}",
        fontsize=13, fontweight="bold"
    )

    axes[0].plot(strikes, bs_prices, color="#2980b9", linewidth=2.0, label="Black-Scholes (closed-form)")
    axes[0].errorbar(strikes, mc_prices, yerr=1.96 * mc_errors,
                     color="#e74c3c", linewidth=1.2, fmt="o", markersize=3,
                     capsize=3, label="Monte Carlo (95% CI)", alpha=0.8)
    axes[0].axvline(S, color="gray", linestyle=":", linewidth=0.8, label=f"Spot S={S}")
    axes[0].set_ylabel("Option Price")
    axes[0].legend(fontsize=9)

    # Absolute error panel
    abs_err = np.abs(bs_prices - mc_prices)
    axes[1].bar(strikes, abs_err, width=(strikes[1] - strikes[0]) * 0.8, color="#e67e22", alpha=0.7)
    axes[1].set_xlabel("Strike K")
    axes[1].set_ylabel("|BS − MC| Error")
    axes[1].set_title("Absolute Pricing Error", fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/bs_vs_mc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] Saved bs_vs_mc.png")


def plot_iv_smile(strikes: np.ndarray, ivs: np.ndarray, S: float, label: str = ""):
    """Plot implied volatility smile / skew across strikes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"Implied Volatility Smile {label}", fontsize=12, fontweight="bold")
    ax.plot(strikes, ivs * 100, color="#2980b9", linewidth=1.8, marker="o", markersize=4)
    ax.axvline(S, color="gray", linestyle="--", linewidth=0.8, label=f"ATM (S={S})")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied Volatility (%)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/iv_smile.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] Saved iv_smile.png")
