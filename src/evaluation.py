"""Metrics for out-of-sample variance forecast evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_variance_forecasts(
    forecasts: pd.Series,
    targets: pd.Series,
) -> dict[str, float | int]:
    """Calculate QLIKE on positive targets and MSE on all valid targets."""
    if not forecasts.index.equals(targets.index):
        raise ValueError("Forecasts and targets must have identical indexes.")

    valid = forecasts.notna() & targets.notna()
    forecast_values = forecasts.loc[valid].to_numpy(dtype=float)
    target_values = targets.loc[valid].to_numpy(dtype=float)

    if len(forecast_values) == 0:
        raise ValueError("No valid forecast-target pairs are available.")

    if not np.isfinite(forecast_values).all():
        raise ValueError("Forecasts must be finite.")

    if not np.isfinite(target_values).all():
        raise ValueError("Targets must be finite.")

    if (forecast_values <= 0).any():
        raise ValueError("Variance forecasts must be strictly positive.")

    if (target_values < 0).any():
        raise ValueError("Variance targets must be nonnegative.")

    mse_values = (forecast_values - target_values) ** 2
    positive_target_mask = target_values > 0

    if positive_target_mask.any():
        positive_forecasts = forecast_values[positive_target_mask]
        positive_targets = target_values[positive_target_mask]
        ratio = positive_targets / positive_forecasts
        qlike_values = ratio - np.log(ratio) - 1.0
        qlike = float(np.mean(qlike_values))
    else:
        qlike = float("nan")

    return {
        "n_observations": len(forecast_values),
        "n_mse_observations": len(forecast_values),
        "n_qlike_observations": int(positive_target_mask.sum()),
        "qlike": qlike,
        "mse": float(np.mean(mse_values)),
    }
