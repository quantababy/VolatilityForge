"""Naive VIX-implied variance forecasts for one- and five-day horizons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vix_implied_variance_forecasts(
    features: pd.DataFrame,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """Convert annualized VIX percentages into daily and five-day variance.

    The five-day forecast assumes constant implied variance across five
    trading sessions. VIX is an implied-volatility proxy, not a direct
    forecast of realized variance.
    """
    if "VIX_Close" not in features.columns:
        raise ValueError("Feature data must contain VIX_Close.")

    if features.empty:
        raise ValueError("Feature data must not be empty.")

    if (
        isinstance(trading_days_per_year, bool)
        or not isinstance(trading_days_per_year, int)
        or trading_days_per_year <= 0
    ):
        raise ValueError("trading_days_per_year must be a positive integer.")

    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("Feature index must be a DatetimeIndex.")

    if features.index.has_duplicates:
        raise ValueError("Feature dates must be unique.")

    if not features.index.is_monotonic_increasing:
        raise ValueError("Feature dates must be sorted in ascending order.")

    vix = pd.to_numeric(features["VIX_Close"], errors="coerce")

    if vix.isna().any() or not np.isfinite(vix.to_numpy()).all():
        raise ValueError("VIX_Close must contain finite values.")

    if (vix <= 0).any():
        raise ValueError("VIX_Close values must be positive.")

    daily_variance = (vix / 100.0).pow(2) / trading_days_per_year

    result = pd.DataFrame(
        {
            "vix_implied_variance_1d": daily_variance,
            "vix_implied_variance_5d": 5.0 * daily_variance,
        },
        index=features.index,
    )
    result.index.name = features.index.name or "Date"
    return result
