# Options Pricing Engine

European option pricing via Black-Scholes closed-form and Monte Carlo GBM simulation, with analytical Greeks computation, 3D surface visualisation, and implied volatility inversion.

---

## Theory

### Black-Scholes Model

Under the risk-neutral measure, the underlying follows GBM:

$$dS_t = r S_t \, dt + \sigma S_t \, dW_t$$

The closed-form European call price is:

$$C = S \cdot N(d_1) - K e^{-rT} \cdot N(d_2)$$

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

Put price follows directly from put-call parity: $P = C - S + K e^{-rT}$.

### Monte Carlo (GBM)

The log-normal terminal price is simulated using the exact discretisation:

$$S_{t+\Delta t} = S_t \cdot \exp\!\left[\left(r - \tfrac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\, Z\right], \quad Z \sim \mathcal{N}(0,1)$$

Antithetic variates (pairing $Z$ and $-Z$) halve Monte Carlo variance at no extra cost. Price = discounted average payoff across $N$ paths.

### Greeks (Analytical)

| Greek | Formula | Trading Meaning |
|---|---|---|
| $\Delta$ | $N(d_1)$ (call) | Directional exposure; hedge ratio |
| $\Gamma$ | $N'(d_1) / (S\sigma\sqrt{T})$ | Rate of delta change; convexity cost |
| $\nu$ (Vega) | $S N'(d_1)\sqrt{T}$ | P&L per 1% vol move |
| $\Theta$ | See code | Daily P&L decay |
| $\rho$ | $KTe^{-rT}N(d_2)$ | P&L per 1% rate move |

### Implied Volatility

IV is the unique $\sigma^*$ satisfying $\text{BS}(S, K, T, r, \sigma^*) = C_{\text{market}}$. Solved numerically via Brent's method (guaranteed convergence, no derivative required).

---

## Methodology

- **Models**: Black-Scholes (closed-form), Monte Carlo GBM with antithetic variates
- **Greeks**: Analytical computation for Delta, Gamma, Vega, Theta, Rho
- **IV solver**: Brentq root-finding on $[10^{-6}, 500\%]$ bracket
- **Surfaces**: Prices and Greeks computed over $(K, T)$ grid; 3D visualised
- **Validation**: MC price vs BS price convergence checked across all strikes; BS inside MC 95% CI confirmed

---

## Key Outputs

| File | Description |
|---|---|
| `gbm_paths.png` | 30 simulated GBM paths with strike line |
| `price_surface.png` | 3D call price over (strike, maturity) |
| `delta_surface.png` | 3D delta surface |
| `gamma_surface.png` | 3D gamma surface |
| `greeks_vs_spot.png` | All Greeks as function of spot price |
| `bs_vs_mc.png` | BS vs MC prices across strikes with error bars |
| `iv_smile.png` | Recovered IV smile (skew-shaped) |

---

## How to Run

```bash
cd options-pricing-engine
pip install -r requirements.txt
python main.py
```

Expected runtime: ~30 seconds (Monte Carlo with 200k paths).

---

## Example Output

```
Black-Scholes ATM Call (S=500, K=500, T=1y, σ=20%, r=5%):
  Price:   52.2529
  Delta:   +0.6368   (≈ probability of expiring ITM)
  Gamma:    0.003752
  Vega:     1.8762   (per 1% vol move)
  Theta:   -0.0446   (daily decay)
  Rho:     +2.3287   (per 1% rate move)

Monte Carlo (200k paths, antithetic):
  MC price:  52.3249 ± 0.1653
  Abs error: 0.0720
  BS in CI:  True
```

---

## Skills Demonstrated

- Derivatives pricing: Black-Scholes PDE, risk-neutral pricing, put-call parity
- Numerical methods: Monte Carlo simulation, variance reduction (antithetic variates), Brent root-finding
- Greeks: analytical derivation and surface visualisation
- Python: `numpy`, `scipy`, `matplotlib` (3D surfaces), vectorised simulation
