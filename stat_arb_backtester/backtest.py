"""
backtest.py — Pairs trading strategy execution and performance attribution.

Strategy Rules:
  - Enter long spread  (long A, short B) when z < -entry_z
  - Enter short spread (short A, long B) when z >  entry_z
  - Exit position when |z| < exit_z
  - Hard stop-loss when |z| > stop_z (spread diverging dangerously)

Position sizing:
  - Dollar-neutral: $1 long A, $hedge_ratio short B per unit
  - Position size scaled to target 1% daily volatility (optional)

P&L Attribution:
  - P&L on leg A: returns_A * position_A
  - P&L on leg B: returns_B * position_B * (-hedge_ratio)
  - Net P&L = A leg + B leg

Metrics:
  - Sharpe ratio  = mean(daily_pnl) / std(daily_pnl) * sqrt(252)
  - Max drawdown  = max peak-to-trough decline in cumulative P&L
  - Win rate      = fraction of trades with positive net P&L
  - Calmar ratio  = annualised return / |max drawdown|
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class BacktestResult:
    daily_pnl: pd.Series
    cum_pnl: pd.Series
    positions: pd.DataFrame       # columns: pos_a, pos_b
    trade_log: pd.DataFrame
    sharpe: float
    max_drawdown: float
    calmar: float
    win_rate: float
    total_trades: int
    annualised_return: float


def run_backtest(
    log_prices_oos: pd.DataFrame,
    prices_oos: pd.DataFrame,
    zscore_oos: pd.Series,
    hedge_ratio: float,
    ticker_a: str,
    ticker_b: str,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    transaction_cost_bps: float = 5.0,
) -> BacktestResult:
    """
    Vectorised backtest of the z-score mean-reversion strategy.

    `transaction_cost_bps` applies round-trip at entry and exit.
    1 bps = 0.01% per leg.
    """
    tc = transaction_cost_bps / 10_000  # as decimal

    returns_a = log_prices_oos[ticker_a].diff()
    returns_b = log_prices_oos[ticker_b].diff()

    z = zscore_oos.reindex(log_prices_oos.index)

    # Initialise state
    pos_a = np.zeros(len(z))    # +1 = long A, -1 = short A
    pos_b = np.zeros(len(z))    # +hedge_ratio = long B, -hedge_ratio = short B
    position = 0                # current state: -1, 0, +1
    trade_log = []

    for i in range(1, len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):
            continue

        prev_pos = position

        if position == 0:
            if zi < -entry_z:
                position = 1    # long spread
            elif zi > entry_z:
                position = -1   # short spread
        elif position == 1:
            if zi > -exit_z or zi > stop_z:
                position = 0
        elif position == -1:
            if zi < exit_z or zi < -stop_z:
                position = 0

        # Record trade events
        if prev_pos != position:
            trade_log.append({
                "date": z.index[i],
                "action": "entry" if prev_pos == 0 else "exit",
                "direction": position if prev_pos == 0 else prev_pos,
                "zscore": round(zi, 3),
            })

        pos_a[i] = position
        pos_b[i] = -position * hedge_ratio

    pos_a_s = pd.Series(pos_a, index=z.index)
    pos_b_s = pd.Series(pos_b, index=z.index)

    # P&L: position at t-1 * return at t (avoid look-ahead)
    pnl_a = pos_a_s.shift(1) * returns_a
    pnl_b = pos_b_s.shift(1) * returns_b

    # Transaction costs: applied on position changes
    turnover = pos_a_s.diff().abs()
    cost = turnover * tc * 2   # 2 legs
    daily_pnl = (pnl_a + pnl_b - cost).dropna()

    cum_pnl = daily_pnl.cumsum()

    # ── Metrics ─────────────────────────────────────────────────
    trading_days = 252
    ann_return = daily_pnl.mean() * trading_days
    ann_vol = daily_pnl.std() * np.sqrt(trading_days)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    rolling_max = cum_pnl.cummax()
    drawdown = cum_pnl - rolling_max
    max_dd = drawdown.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Win rate per trade (not per day)
    trade_df = pd.DataFrame(trade_log)
    win_rate = float("nan")
    total_trades = 0
    if not trade_df.empty:
        entries = trade_df[trade_df["action"] == "entry"]
        exits = trade_df[trade_df["action"] == "exit"]
        total_trades = min(len(entries), len(exits))
        if total_trades > 0:
            pnl_per_trade = []
            for j in range(total_trades):
                t_in = entries.iloc[j]["date"]
                t_out = exits.iloc[j]["date"]
                trade_pnl = daily_pnl.loc[t_in:t_out].sum()
                pnl_per_trade.append(trade_pnl)
            win_rate = sum(p > 0 for p in pnl_per_trade) / total_trades

    return BacktestResult(
        daily_pnl=daily_pnl,
        cum_pnl=cum_pnl,
        positions=pd.DataFrame({"pos_a": pos_a_s, "pos_b": pos_b_s}),
        trade_log=trade_df,
        sharpe=round(sharpe, 3),
        max_drawdown=round(max_dd, 5),
        calmar=round(calmar, 3),
        win_rate=round(win_rate, 3) if not np.isnan(win_rate) else float("nan"),
        total_trades=total_trades,
        annualised_return=round(ann_return, 4),
    )
