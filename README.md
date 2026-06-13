# Quant Projects

Three production-quality quantitative finance implementations covering volatility modelling, statistical arbitrage, and derivatives pricing.

| Project | Domain | Core Methods |
|---|---|---|
| `volatility_forecasting` | Risk / Vol | GARCH(1,1), EGARCH, VaR backtesting |
| `stat_arb_backtester` | Alpha / Execution | Cointegration, z-score mean reversion, PnL attribution |
| `options_pricing_engine` | Derivatives | Black-Scholes, Monte Carlo GBM, Greeks, IV solver |

## Structure

```
quant-projects/
├── volatility-forecasting/
│   ├── data.py
│   ├── models.py
│   ├── backtest.py
│   ├── plots.py
│   ├── main.py
│   ├── requirements.txt
│   └── README.md
├── stat-arb-backtester/
│   ├── data.py
│   ├── models.py
│   ├── backtest.py
│   ├── plots.py
│   ├── main.py
│   ├── requirements.txt
│   └── README.md
└── options-pricing-engine/
    ├── models.py
    ├── greeks.py
    ├── monte_carlo.py
    ├── plots.py
    ├── main.py
    ├── requirements.txt
    └── README.md
```

## Quick Start

Each project is self-contained. Navigate into any subdirectory and follow its README.

```bash
cd volatility_forecasting && pip install -r requirements.txt && python main.py
cd stat_arb_backtester   && pip install -r requirements.txt && python main.py
cd options_pricing_engine && pip install -r requirements.txt && python main.py
```
