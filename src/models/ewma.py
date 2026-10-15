"""Exponentially weighted moving average variance forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_variance(
    returns: pd.Series,
    decay: float = 0.94,
    initial_variance: float | None = None,
) -> pd.Series:
    """Estimate one-step-ahead variance forecasts using EWMA.

    The value at date t is the forecast for the variance of the
    return on date t, based only on returns through date t-1.
    """
    if not 0 < decay < 1:
        raise ValueError("decay must be strictly between 0 and 1.")

    if returns.empty:
        raise ValueError("returns must not be empty.")

    if returns.isna().any():
        raise ValueError("returns must not contain missing values.")

    values = returns.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values.")

    if initial_variance is not None and (
        not np.isfinite(initial_variance) or initial_variance <= 0
    ):
        raise ValueError("initial_variance must be finite and positive.")

    if initial_variance is None:
        initial_variance = float(np.mean(values**2))

    forecasts = np.empty(len(values), dtype=float)
    forecasts[0] = initial_variance

    for index in range(1, len(values)):
        forecasts[index] = (
            decay * forecasts[index - 1]
            + (1 - decay) * values[index - 1] ** 2
        )

    return pd.Series(
        forecasts,
        index=returns.index,
        name="ewma_variance",
    )
