"""
plots.py — Visualisations for volatility forecasting and VaR backtesting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

COLORS = {
    "returns": "#2c3e50",
    "garch": "#2980b9",
    "egarch": "#e74c3c",
    "var95": "#f39c12",
    "var99": "#c0392b",
    "violation": "#8e44ad",
}


def plot_returns_and_vol(returns: pd.Series, garch_vol: pd.Series, egarch_vol: pd.Series):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("S&P 500 — Log Returns and Conditional Volatility", fontsize=14, fontweight="bold")

    axes[0].plot(returns, color=COLORS["returns"], linewidth=0.7, alpha=0.85, label="Log returns")
    axes[0].set_ylabel("Log Return")
    axes[0].legend(fontsize=9)

    axes[1].plot(garch_vol, color=COLORS["garch"], linewidth=1.0, label="GARCH(1,1) σ_t")
    axes[1].plot(egarch_vol, color=COLORS["egarch"], linewidth=1.0, alpha=0.85, linestyle="--", label="EGARCH(1,1) σ_t")
    axes[1].set_ylabel("Daily Volatility")
    axes[1].legend(fontsize=9)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/returns_and_volatility.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved returns_and_volatility.png")


def plot_var_backtest(returns: pd.Series, var_df: pd.DataFrame, model_name: str):
    """
    Plot realised returns vs VaR bands; highlight violation days in red.
    """
    common = returns.index.intersection(var_df.index)
    r = returns.loc[common]
    var = var_df.loc[common]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title(f"{model_name} — VaR Backtest (Out-of-Sample)", fontsize=13, fontweight="bold")

    ax.plot(r, color=COLORS["returns"], linewidth=0.6, alpha=0.8, label="Returns", zorder=2)
    ax.plot(-var["VaR_95"], color=COLORS["var95"], linewidth=1.0, linestyle="--", label="95% VaR (negative)", zorder=3)
    ax.plot(-var["VaR_99"], color=COLORS["var99"], linewidth=1.0, linestyle="-.", label="99% VaR (negative)", zorder=3)

    # Mark violations at 99% level
    violations_99 = r[r < -var["VaR_99"]]
    ax.scatter(violations_99.index, violations_99.values, color=COLORS["violation"],
               s=20, zorder=5, label=f"99% violations ({len(violations_99)})", marker="x")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("Daily Log Return")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fname = f"{OUTDIR}/var_backtest_{model_name.lower().replace('(','').replace(')','').replace(',','')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {fname}")


def plot_vol_comparison(garch_vol: pd.Series, egarch_vol: pd.Series):
    """Scatter plot of GARCH vs EGARCH conditional vol to visualise divergence."""
    common = garch_vol.index.intersection(egarch_vol.index)
    g = garch_vol.loc[common]
    e = egarch_vol.loc[common]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(g, e, alpha=0.3, s=5, color=COLORS["garch"])
    lims = [min(g.min(), e.min()), max(g.max(), e.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="45° line")
    ax.set_xlabel("GARCH(1,1) σ_t")
    ax.set_ylabel("EGARCH(1,1) σ_t")
    ax.set_title("GARCH vs EGARCH Conditional Volatility", fontsize=12, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/vol_comparison_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved vol_comparison_scatter.png")


def plot_rolling_violation_rate(returns: pd.Series, var_df: pd.DataFrame, model_name: str, window: int = 252):
    """Rolling 1-year violation rate to detect VaR model degradation over time."""
    common = returns.index.intersection(var_df.index)
    r = returns.loc[common]
    var = var_df.loc[common]

    for col, nominal, color in [("VaR_95", 0.05, COLORS["var95"]), ("VaR_99", 0.01, COLORS["var99"])]:
        violations = (r < -var[col]).astype(int)
        rolling_rate = violations.rolling(window).mean()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(rolling_rate, color=color, linewidth=1.2, label=f"Rolling {window}d violation rate")
        ax.axhline(nominal, color="black", linewidth=1.0, linestyle="--", label=f"Nominal rate ({nominal:.0%})")
        ax.set_title(f"{model_name} — Rolling Violation Rate ({col})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Violation Rate")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.tight_layout()
        fname = f"{OUTDIR}/rolling_violation_{model_name.lower().replace('(','').replace(')','').replace(',','')}_{col}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot] Saved {fname}")
