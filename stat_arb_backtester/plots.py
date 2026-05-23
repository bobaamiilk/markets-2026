"""
plots.py — Visualisations for statistical arbitrage backtester.
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


def plot_price_series(prices: pd.DataFrame, ticker_a: str, ticker_b: str, label: str = ""):
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig.suptitle(f"Normalised Price Series — {ticker_a} vs {ticker_b} {label}", fontsize=13, fontweight="bold")

    norm = prices / prices.iloc[0]
    axes[0].plot(norm[ticker_a], label=ticker_a, color="#2980b9")
    axes[0].plot(norm[ticker_b], label=ticker_b, color="#e74c3c", alpha=0.85)
    axes[0].set_ylabel("Normalised Price (rebased to 1)")
    axes[0].legend()

    spread_log = np.log(prices[ticker_a]) - np.log(prices[ticker_b])
    axes[1].plot(spread_log, color="#27ae60", linewidth=0.9)
    axes[1].set_ylabel("Log Price Spread")
    axes[1].axhline(spread_log.mean(), color="black", linewidth=0.8, linestyle="--", label="Mean")
    axes[1].legend()
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fname = f"{OUTDIR}/price_series_{ticker_a}_{ticker_b}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {fname}")


def plot_zscore(zscore: pd.Series, entry_z: float = 2.0, exit_z: float = 0.5):
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_title("Z-Score of Spread — Mean Reversion Signal", fontsize=13, fontweight="bold")

    ax.plot(zscore, color="#2c3e50", linewidth=0.8, label="Z-score")
    ax.axhline(entry_z, color="#e74c3c", linestyle="--", linewidth=1.0, label=f"+{entry_z} entry")
    ax.axhline(-entry_z, color="#2980b9", linestyle="--", linewidth=1.0, label=f"-{entry_z} entry")
    ax.axhline(exit_z, color="#95a5a6", linestyle=":", linewidth=0.8, label=f"±{exit_z} exit")
    ax.axhline(-exit_z, color="#95a5a6", linestyle=":", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.fill_between(zscore.index, entry_z, zscore.values, where=(zscore > entry_z), alpha=0.15, color="#e74c3c")
    ax.fill_between(zscore.index, -entry_z, zscore.values, where=(zscore < -entry_z), alpha=0.15, color="#2980b9")
    ax.set_ylabel("Z-score")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/zscore.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] Saved zscore.png")


def plot_cumulative_pnl(cum_pnl: pd.Series, daily_pnl: pd.Series, sharpe: float, max_dd: float):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(
        f"Strategy P&L — Sharpe: {sharpe:.2f} | Max Drawdown: {max_dd:.4f}",
        fontsize=13, fontweight="bold"
    )

    axes[0].plot(cum_pnl, color="#2980b9", linewidth=1.2, label="Cumulative log-return P&L")
    axes[0].fill_between(cum_pnl.index, 0, cum_pnl, where=(cum_pnl >= 0), alpha=0.2, color="#27ae60")
    axes[0].fill_between(cum_pnl.index, 0, cum_pnl, where=(cum_pnl < 0), alpha=0.2, color="#e74c3c")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("Cumulative P&L (log-return)")
    axes[0].legend()

    # Drawdown
    rolling_max = cum_pnl.cummax()
    drawdown = cum_pnl - rolling_max
    axes[1].fill_between(drawdown.index, drawdown, 0, color="#e74c3c", alpha=0.5, label="Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/cumulative_pnl.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] Saved cumulative_pnl.png")


def plot_positions(positions: pd.DataFrame, zscore: pd.Series):
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig.suptitle("Strategy Positions vs Z-Score", fontsize=13, fontweight="bold")

    common = zscore.index.intersection(positions.index)
    axes[0].plot(zscore.loc[common], color="#2c3e50", linewidth=0.7)
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].set_ylabel("Z-score")

    axes[1].step(positions.index, positions["pos_a"], color="#2980b9", linewidth=0.9, where="post", label="Position A")
    axes[1].set_ylabel("Position (units)")
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].legend()
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/positions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] Saved positions.png")


def plot_return_distribution(daily_pnl: pd.Series):
    from scipy import stats
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Daily P&L Distribution", fontsize=12, fontweight="bold")
    ax.hist(daily_pnl, bins=60, color="#2980b9", alpha=0.7, edgecolor="white", density=True)

    x = np.linspace(daily_pnl.min(), daily_pnl.max(), 200)
    mu, sigma = daily_pnl.mean(), daily_pnl.std()
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r--", linewidth=1.5, label="Normal fit")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Daily P&L")
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/pnl_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] Saved pnl_distribution.png")
