"""Chronological out-of-sample volatility forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ewma_forecasts(
    features: pd.DataFrame,
    test_start: str,
    decay: float = 0.94,
    minimum_training_observations: int = 252,
) -> pd.DataFrame:
    """Generate one- and five-day EWMA forecasts without lookahead.

    Initialize variance from the first minimum_training_observations
    pre-test returns. Process subsequent returns chronologically.
    Each forecast at date t uses returns observed strictly before t.
    """
    required = {"log_return", "target_rv_1", "target_rv_5"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if features.empty:
        raise ValueError("features must not be empty.")

    data = features.copy()
    data.index = pd.DatetimeIndex(data.index).normalize()
    data.index.name = "Date"

    if data.index.has_duplicates:
        raise ValueError("features contains duplicate dates.")
    if not data.index.is_monotonic_increasing:
        raise ValueError("features must be sorted by date.")
    if not 0 < decay < 1:
        raise ValueError("decay must be strictly between 0 and 1.")
    if minimum_training_observations < 2:
        raise ValueError("minimum_training_observations must be at least 2.")

    test_start_date = pd.Timestamp(test_start)
    training_returns = data.loc[
        data.index < test_start_date, "log_return"
    ].dropna()

    if len(training_returns) < minimum_training_observations:
        raise ValueError("Insufficient pre-test observations to initialize EWMA.")

    initial_sample = training_returns.iloc[:minimum_training_observations]
    initial_values = initial_sample.to_numpy(dtype=float)

    if not np.isfinite(initial_values).all():
        raise ValueError("Training returns must be finite.")

    variance = float(np.mean(initial_values**2))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Initial variance must be finite and positive.")

    # Start after the final return used for initialization.
    initialization_date = initial_sample.index[-1]
    start_position = data.index.get_loc(initialization_date) + 1
    forecasts = pd.Series(
        index=data.index[data.index >= test_start_date],
        dtype=float,
        name="ewma_variance_1d",
    )

    for date in data.index[start_position:]:
        # The current return is not observed until after this forecast.
        if date >= test_start_date:
            forecasts.loc[date] = variance

        observed_return = data.at[date, "log_return"]
        if pd.notna(observed_return):
            observed_return = float(observed_return)
            if not np.isfinite(observed_return):
                raise ValueError("Returns must be finite.")
            variance = (
                decay * variance
                + (1 - decay) * observed_return**2
            )

    result = pd.DataFrame(index=forecasts.index)
    result["ewma_variance_1d"] = forecasts
    result["ewma_variance_5d"] = 5 * forecasts
    result["target_rv_1"] = data.loc[result.index, "target_rv_1"]
    result["target_rv_5"] = data.loc[result.index, "target_rv_5"]

    return result
