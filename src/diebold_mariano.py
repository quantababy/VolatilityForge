
"""Diebold-Mariano tests for paired forecast loss differentials."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t


def diebold_mariano_test(
    loss_a: pd.Series,
    loss_b: pd.Series,
    *,
    hac_lags: int = 0,
    forecast_horizon: int = 1,
) -> dict[str, float | int]:
    """Compare paired losses using HAC variance and the HLN correction.

    The differential is loss_a - loss_b:
    positive values favor model B; negative values favor model A.

    The Newey-West HAC estimator uses Bartlett weights. The Harvey-Leybourne-
    Newbold small-sample factor adjusts the DM statistic using the forecast
    horizon, and a Student-t reference distribution with n - 1 degrees of
    freedom is used for the p-value.

    For overlapping h-step forecasts, callers should generally use at least
    h - 1 HAC lags. When HAC lags are used, missing loss pairs are rejected
    because dropping them would change the temporal spacing of observations.
    The correction does not guarantee exact inference under arbitrary
    dependence or model misspecification.
    """
    for name, losses in (("loss_a", loss_a), ("loss_b", loss_b)):
        if losses.index.has_duplicates:
            raise ValueError(f"{name} index must not contain duplicates.")
        if not losses.index.is_monotonic_increasing:
            raise ValueError(f"{name} index must be sorted chronologically.")

    if not loss_a.index.equals(loss_b.index):
        raise ValueError("Loss series must have identical indexes.")

    if (
        not isinstance(hac_lags, int)
        or isinstance(hac_lags, bool)
        or hac_lags < 0
    ):
        raise ValueError("hac_lags must be a non-negative integer.")

    if (
        not isinstance(forecast_horizon, int)
        or isinstance(forecast_horizon, bool)
        or forecast_horizon < 1
    ):
        raise ValueError("forecast_horizon must be a positive integer.")

    valid = loss_a.notna() & loss_b.notna()

    if not valid.all() and hac_lags > 0:
        raise ValueError(
            "Loss series contain missing pairs; HAC lag spacing would be invalid."
        )

    a = loss_a.loc[valid].to_numpy(dtype=float)
    b = loss_b.loc[valid].to_numpy(dtype=float)

    if len(a) < 2:
        raise ValueError("At least two paired loss observations are required.")

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Loss observations must be finite.")

    differential = a - b
    n = len(differential)

    if hac_lags >= n:
        raise ValueError(
            "hac_lags must be smaller than the number of observations."
        )

    if forecast_horizon >= n:
        raise ValueError(
            "forecast_horizon must be smaller than the number of observations."
        )

    mean_differential = float(np.mean(differential))
    centered = differential - mean_differential

    # Newey-West HAC estimate of the long-run variance.
    long_run_variance = float(np.dot(centered, centered) / n)

    for lag in range(1, hac_lags + 1):
        weight = 1.0 - lag / (hac_lags + 1.0)
        autocovariance = float(
            np.dot(centered[lag:], centered[:-lag]) / n
        )
        long_run_variance += 2.0 * weight * autocovariance

    if not np.isfinite(long_run_variance) or long_run_variance <= 0:
        raise ValueError("Loss differential has non-positive HAC variance.")

    standard_error = float(np.sqrt(long_run_variance / n))
    raw_statistic = mean_differential / standard_error

    # Harvey-Leybourne-Newbold small-sample correction.
    h = forecast_horizon
    correction_squared = (
        n + 1 - 2 * h + h * (h - 1) / n
    ) / n

    if correction_squared <= 0:
        raise ValueError("HLN correction is undefined for this sample and horizon.")

    correction_factor = float(np.sqrt(correction_squared))
    statistic = raw_statistic * correction_factor
    p_value = float(2.0 * t.sf(abs(statistic), df=n - 1))

    return {
        "n_observations": n,
        "mean_loss_differential": mean_differential,
        "dm_statistic": float(statistic),
        "raw_dm_statistic": float(raw_statistic),
        "hln_correction_factor": correction_factor,
        "p_value": p_value,
        "hac_lags": hac_lags,
        "forecast_horizon": forecast_horizon,
    }
