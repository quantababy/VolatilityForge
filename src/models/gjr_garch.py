"""GJR-GARCH(1,1) volatility forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


def gjr_garch_variance_forecasts(
    returns: pd.Series,
    minimum_training_observations: int = 252,
    reestimate_every: int = 21,
) -> pd.Series:
    """Generate next-trading-day GJR-GARCH variance forecasts.

    The forecast indexed by date t uses returns observed through t
    and predicts variance for the next trading day.

    Returns are decimal log returns. Forecasts are returned in
    decimal-return variance units.
    """
    if returns.empty:
        raise ValueError("returns must not be empty.")
    if returns.isna().any():
        raise ValueError("returns must not contain missing values.")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns must be sorted by index.")
    if returns.index.has_duplicates:
        raise ValueError("returns must not contain duplicate index values.")
    if minimum_training_observations < 30:
        raise ValueError("minimum_training_observations must be at least 30.")
    if reestimate_every < 1:
        raise ValueError("reestimate_every must be at least 1.")

    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values.")

    forecasts = pd.Series(
        np.nan,
        index=returns.index,
        name="gjr_garch_variance_1d",
        dtype=float,
    )

    fitted_result = None

    for position in range(
        minimum_training_observations - 1,
        len(returns) - 1,
    ):
        observations_available = position + 1

        if (
            fitted_result is None
            or (
                observations_available - minimum_training_observations
            )
            % reestimate_every
            == 0
        ):
            historical_returns = values[:observations_available] * 100
            fitted_model = arch_model(
                historical_returns,
                mean="Zero",
                vol="GARCH",
                p=1,
                o=1,
                q=1,
                dist="normal",
                rescale=False,
            )
            fitted_result = fitted_model.fit(disp="off")

            if fitted_result.convergence_flag != 0:
                raise RuntimeError(
                    "GJR-GARCH estimation did not converge at position "
                    f"{position}."
                )

        forecast = fitted_result.forecast(horizon=1, reindex=False)
        variance_percent_squared = float(forecast.variance.iloc[-1, 0])
        variance_decimal_squared = variance_percent_squared / 10000

        if (
            not np.isfinite(variance_decimal_squared)
            or variance_decimal_squared <= 0
        ):
            raise RuntimeError(
                "GJR-GARCH produced an invalid variance forecast."
            )

        forecasts.iloc[position] = variance_decimal_squared

    return forecasts


def gjr_garch_five_day_variance_forecasts(
    returns: pd.Series,
    minimum_training_observations: int = 252,
    reestimate_every: int = 21,
) -> pd.Series:
    """Forecast cumulative variance over the next five trading sessions."""
    if returns.empty:
        raise ValueError("returns must not be empty.")
    if returns.isna().any():
        raise ValueError("returns must not contain missing values.")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns must be sorted by index.")
    if returns.index.has_duplicates:
        raise ValueError("returns must not contain duplicate index values.")
    if minimum_training_observations < 30:
        raise ValueError("minimum_training_observations must be at least 30.")
    if reestimate_every < 1:
        raise ValueError("reestimate_every must be at least 1.")

    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values.")

    forecasts = pd.Series(
        np.nan,
        index=returns.index,
        name="gjr_garch_variance_5d",
        dtype=float,
    )

    fitted_result = None

    # Keep five future sessions available for target alignment.
    for position in range(
        minimum_training_observations - 1,
        len(returns) - 5,
    ):
        observations_available = position + 1

        if (
            fitted_result is None
            or (
                observations_available - minimum_training_observations
            ) % reestimate_every == 0
        ):
            historical_returns = values[:observations_available] * 100
            fitted_model = arch_model(
                historical_returns,
                mean="Zero",
                vol="GARCH",
                p=1,
                o=1,
                q=1,
                dist="normal",
                rescale=False,
            )
            fitted_result = fitted_model.fit(disp="off")

            if fitted_result.convergence_flag != 0:
                raise RuntimeError(
                    "GJR-GARCH estimation did not converge at position "
                    f"{position}."
                )

        forecast = fitted_result.forecast(horizon=5, reindex=False)
        horizon_variances = forecast.variance.iloc[-1].to_numpy(dtype=float)
        cumulative_variance = float(horizon_variances.sum() / 10000)

        if (
            not np.isfinite(cumulative_variance)
            or cumulative_variance <= 0
        ):
            raise RuntimeError(
                "GJR-GARCH produced an invalid five-day forecast."
            )

        forecasts.iloc[position] = cumulative_variance

    return forecasts
