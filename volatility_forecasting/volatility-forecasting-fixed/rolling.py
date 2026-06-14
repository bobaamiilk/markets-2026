"""
rolling.py — Expanding-window rolling re-estimation of GARCH/EGARCH for
out-of-sample volatility forecasts.

Why this is needed:
    A model fit once on training data and never updated will drift out of
    date. In practice you periodically refit on all data available up to
    "today" and forecast one step ahead. This module simulates that process
    on the test set, refitting every `step` days (default: monthly).

This avoids look-ahead bias: at each refit point, only data up to that
point in time is used.
"""

import warnings
import numpy as np
import pandas as pd
from arch import arch_model

warnings.filterwarnings("ignore")


def rolling_var_estimates(
    returns_full: pd.Series,
    train_end_idx: int,
    model: str = "garch",
    step: int = 21,
    scale: float = 100.0,
) -> pd.Series:
    """
    Re-fit the model every `step` observations on an expanding window
    (all data from the start up to the current point), and record the
    one-step-ahead conditional volatility forecast.

    Args:
        returns_full: full return series (train + test)
        train_end_idx: index marking end of training data; rolling starts here
        model: "garch" or "egarch"
        step: re-estimation frequency in trading days (21 ≈ monthly)
        scale: scaling factor for numerical stability of the optimiser

    Returns:
        pd.Series of forecasted daily volatility (sigma), indexed by date,
        covering the test period.
    """
    if train_end_idx >= len(returns_full):
        raise ValueError(
            f"train_end_idx ({train_end_idx}) must be < len(returns_full) ({len(returns_full)})"
        )

    vols = []
    dates = []

    for end in range(train_end_idx, len(returns_full), step):
        window = returns_full.iloc[:end] * scale

        if len(window) < 30:
            continue  # not enough data to fit a GARCH model

        try:
            if model == "garch":
                am = arch_model(window, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            elif model == "egarch":
                am = arch_model(window, vol="EGARCH", p=1, q=1, dist="normal", rescale=False)
            else:
                raise ValueError(f"Unknown model: {model!r} (expected 'garch' or 'egarch')")

            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=1, reindex=False)
            forecasted_vol = float(np.sqrt(fc.variance.values[-1, 0])) / scale

            forecast_date = returns_full.index[min(end, len(returns_full) - 1)]
            vols.append(forecasted_vol)
            dates.append(forecast_date)

        except Exception as e:
            print(f"[rolling] Skipping refit at index {end} ({model}): {e}")
            continue

    if not vols:
        raise RuntimeError(
            f"rolling_var_estimates produced 0 forecasts for model={model!r}. "
            f"Check train_end_idx and that returns_full has enough data."
        )

    return pd.Series(vols, index=dates, name=f"{model}_vol")
