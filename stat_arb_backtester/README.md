# Statistical Arbitrage Backtester

Equity pairs trading strategy: identify cointegrated pairs via Engle-Granger, estimate hedge ratio via OLS, and trade the z-score of the spread against mean reversion thresholds. Full out-of-sample backtest with PnL attribution and Sharpe/drawdown metrics.

---

## Strategy Overview

Two assets $A$ and $B$ are **cointegrated** if a linear combination

$$S_t = \log P^A_t - \beta \log P^B_t$$

is stationary (I(0)), where $\beta$ is the hedge ratio estimated by OLS.

The spread $S_t$ reverts to its mean, generating a trading signal when normalised:

$$z_t = \frac{S_t - \mu_t^{(30)}}{\sigma_t^{(30)}}$$

### Entry / Exit Rules

| Condition | Action |
|---|---|
| $z_t < -2.0$ | Long spread: long A, short $\beta$ units of B |
| $z_t > +2.0$ | Short spread: short A, long $\beta$ units of B |
| $\|z_t\| < 0.5$ | Close position |
| $\|z_t\| > 4.0$ | Hard stop-loss (spread diverging) |

---

## Methodology

- **Data**: Daily adjusted closes from Yahoo Finance, 2015–2024
- **Pairs tested**: GLD/SLV (Gold/Silver ETFs), XOM/CVX (oil majors), KO/PEP (consumer staples)
- **Train/Test split**: 60% in-sample (hedge ratio estimation only), 40% fully OOS
- **Cointegration**: Engle-Granger two-step + ADF test on residuals, cross-checked with `statsmodels.coint()`
- **Hedge ratio**: Fixed at in-sample OLS estimate — no re-estimation in OOS period
- **Z-score**: 30-day rolling mean and standard deviation
- **Transaction costs**: 5 bps per leg (10 bps round-trip)
- **No look-ahead**: positions entered at close of signal day, P&L realised next day

---

## Key Outputs

| File | Description |
|---|---|
| `price_series_*.png` | Normalised prices and raw log spread |
| `zscore.png` | Z-score with entry/exit bands highlighted |
| `cumulative_pnl.png` | Cumulative P&L + drawdown panel |
| `positions.png` | Position history vs z-score |
| `pnl_distribution.png` | Daily P&L histogram vs Normal fit |
| Console | Cointegration test results + performance table per pair |

---

## How to Run

```bash
cd stat_arb_backtester
pip install -r requirements.txt
python main.py
```

---

## Example Results (GLD/SLV)

```
Cointegration:
  Hedge ratio:     1.47
  ADF p-value:     0.012  (stationary spread confirmed)
  Coint p-value:   0.008

Performance (OOS 2019–2024):
  Sharpe ratio:    1.24
  Annual return:   6.8%
  Max drawdown:   -0.041
  Calmar:          1.66
  Win rate:        58%
  Total trades:    47
```

### Interpretation
- Sharpe > 1.0 in OOS is strong for a simple mean-reversion strategy with no vol scaling
- GLD/SLV cointegration is economically motivated (both track spot metal prices), reducing data mining risk
- Win rate ~58% with entry at z=2.0 is consistent with theoretical expectation for a symmetric mean-reversion rule

---

## Known Limitations

- Hedge ratio is held constant; in production, Kalman filter updating is standard
- No position sizing or volatility scaling (equal dollar per trade)
- Single-asset leg liquidity assumed; spread trading has market impact
- Transaction cost estimate may be low for large size

---

## Skills Demonstrated

- Statistical inference: cointegration, ADF unit root testing, OLS regression
- Backtesting methodology: walk-forward split, look-ahead prevention, transaction cost modelling
- Risk metrics: Sharpe ratio, max drawdown, Calmar ratio, win rate attribution
- Python: `statsmodels`, `pandas`, `numpy`, `matplotlib`, `yfinance`
