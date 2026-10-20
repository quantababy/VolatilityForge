"""Calibration diagnostics for out-of-sample variance forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def mincer_zarnowitz(
    target: pd.Series,
    forecast: pd.Series,
    *,
    hac_lags: int,
) -> dict[str, float | int]:
    """Fit target = intercept + slope * forecast with HAC standard errors."""
    if not target.index.equals(forecast.index):
        raise ValueError("Target and forecast indexes must match exactly.")
    if hac_lags < 0:
        raise ValueError("hac_lags must be nonnegative.")

    frame = pd.DataFrame({"target": target, "forecast": forecast})
    values = frame.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Target and forecast values must be finite.")
    if (frame["target"] < 0).any():
        raise ValueError("Targets must be nonnegative.")
    if (frame["forecast"] <= 0).any():
        raise ValueError("Forecasts must be strictly positive.")
    if len(frame) <= hac_lags + 2:
        raise ValueError("Insufficient observations for requested HAC lags.")
    if frame["forecast"].nunique() < 2:
        raise ValueError("Forecasts must have variation for regression.")

    design = sm.add_constant(frame["forecast"], has_constant="add")
    fit = sm.OLS(frame["target"], design).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags},
    )

    # H0: intercept = 0 and slope = 1.
    joint_test = fit.wald_test(
        (
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([0.0, 1.0]),
        ),
        scalar=True,
    )
    intercept_ci = fit.conf_int().loc["const"]
    slope_ci = fit.conf_int().loc["forecast"]

    return {
        "n_observations": int(fit.nobs),
        "intercept": float(fit.params["const"]),
        "intercept_se_hac": float(fit.bse["const"]),
        "intercept_ci_lower": float(intercept_ci.iloc[0]),
        "intercept_ci_upper": float(intercept_ci.iloc[1]),
        "slope": float(fit.params["forecast"]),
        "slope_se_hac": float(fit.bse["forecast"]),
        "slope_ci_lower": float(slope_ci.iloc[0]),
        "slope_ci_upper": float(slope_ci.iloc[1]),
        "r_squared": float(fit.rsquared),
        "joint_wald_statistic": float(np.asarray(joint_test.statistic).item()),
        "joint_p_value": float(joint_test.pvalue),
        "hac_lags": int(hac_lags),
    }
