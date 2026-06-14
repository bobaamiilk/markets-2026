# Volatility Forecasting + VaR Backtesting

Conditional volatility estimation on S&P 500 returns using GARCH-family models, with parametric VaR computation and rigorous out-of-sample backtesting.

---

## Model Overview

### GARCH(1,1) — Bollerslev (1986)

$$\sigma^2_t = \omega + \alpha \varepsilon^2_{t-1} + \beta \sigma^2_{t-1}$$

Volatility is a mean-reverting process driven by lagged squared shocks ($\alpha$) and lagged variance ($\beta$). Persistence $= \alpha + \beta$; typically 0.98–0.99 for equity indices.

### EGARCH(1,1) — Nelson (1991)

$$\log \sigma^2_t = \omega + \alpha(|z_{t-1}| - \mathbb{E}|z|) + \gamma z_{t-1} + \beta \log \sigma^2_{t-1}$$

Captures the **leverage effect**: negative return shocks ($\gamma < 0$) increase volatility more than positive shocks of equal magnitude. Log specification guarantees $\sigma^2_t > 0$ without parameter constraints.

### Parametric VaR

$$\text{VaR}_\alpha(t) = -\hat{\sigma}_t \cdot z_\alpha$$

where $z_\alpha = \Phi^{-1}(1 - \alpha)$. VaR is time-varying because $\hat{\sigma}_t$ comes from the GARCH model.

---

## Methodology

- **Data**: S&P 500 (^GSPC) daily log returns, 2010–2024 (~3,770 obs)
- **Train/Test split**: 80% in-sample / 20% out-of-sample (no look-ahead)
- **Rolling re-estimation**: models refit monthly on expanding window to simulate live deployment
- **VaR levels**: 95% and 99% confidence
- **Backtesting**: Kupiec Proportion of Failures (POF) test — LR statistic ~ χ²(1) under H₀ of correct model

---

## Key Outputs

| Output | Description |
|---|---|
| `returns_and_volatility.png` | Returns time series overlaid with GARCH/EGARCH conditional vol |
| `var_backtest_garch11.png` | OOS returns vs VaR bands; violation dates marked |
| `var_backtest_egarch11.png` | Same for EGARCH |
| `vol_comparison_scatter.png` | GARCH vs EGARCH conditional vol scatter; divergence during stress periods |
| `rolling_violation_*.png` | Rolling 252-day violation rate vs nominal threshold |
| Console output | AIC/BIC/LogL comparison table + Kupiec test results |

### Interpreting Kupiec p-value
- **p > 0.05**: fail to reject H₀ → model violation rate is statistically consistent with nominal level
- **p < 0.05**: reject → model is miscalibrated (under- or over-conservative)

---

## How to Run

```bash
cd volatility-forecasting
pip install -r requirements.txt
python main.py
```

Expected runtime: ~90 seconds (rolling re-estimation loop).

---

## Example Results (typical output)

```
GARCH  | AIC=-28431.2 | BIC=-28411.8 | LogL=14219.6
EGARCH | AIC=-28489.7 | BIC=-28463.7 | LogL=14249.8

GARCH VaR Backtest:
  95% VaR | expected: 5.00% | actual: 5.21% | Kupiec p=0.61 (pass)
  99% VaR | expected: 1.00% | actual: 1.34% | Kupiec p=0.12 (pass)

EGARCH VaR Backtest:
  95% VaR | expected: 5.00% | actual: 4.93% | Kupiec p=0.84 (pass)
  99% VaR | expected: 1.00% | actual: 1.08% | Kupiec p=0.79 (pass)
```

EGARCH typically achieves lower AIC/BIC due to leverage effect — confirmed by negative $\gamma$ parameter.

---

## Skills Demonstrated

- Time-series econometrics: GARCH-family models, stationarity, volatility clustering
- Statistical testing: Kupiec POF likelihood ratio test, chi-squared inference
- Risk modelling: parametric VaR, expanding-window backtesting, violation rate analysis
- Numerical methods: MLE optimisation via `arch` library (BFGS / Nelder-Mead)
- Python: `arch`, `pandas`, `numpy`, `scipy`, `matplotlib`

# NOTE
This project does not use live market data during execution.
All datasets are pre-generated and stored locally.
