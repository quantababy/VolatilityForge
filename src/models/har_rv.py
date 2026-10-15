"""HAR-RV volatility forecasting with expanding-window regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def har_rv_variance_forecasts(
    features: pd.DataFrame,
    minimum_training_observations: int = 252,
    reestimate_every: int = 21,
) -> pd.Series:
    """Forecast next-day variance from daily, weekly, and monthly RV.

    Features at date t use information through date t. The target at t
    is the squared return on the next trading day.

    RV predictors are log-transformed. The regression target is the
    log of next-day squared return. Exponentiating the prediction
    yields a positive variance forecast.
    """
    required = {"squared_return", "rv_5", "rv_22", "target_rv_1"}
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

    predictors = pd.DataFrame(index=data.index)
    predictors["daily"] = data["squared_return"]
    predictors["weekly"] = data["rv_5"] / 5
    predictors["monthly"] = data["rv_22"] / 22

    # Log transforms require strictly positive realized-variance proxies.
    predictors = predictors.where(predictors > 0)
    log_predictors = np.log(predictors)
    log_target = np.log(data["target_rv_1"].where(data["target_rv_1"] > 0))

    valid = (
        log_predictors.notna().all(axis=1)
        & log_target.notna()
        & np.isfinite(log_predictors).all(axis=1)
        & np.isfinite(log_target)
    )

    if minimum_training_observations < 30:
        raise ValueError("minimum_training_observations must be at least 30.")
    if reestimate_every < 1:
        raise ValueError("reestimate_every must be at least 1.")

    forecasts = pd.Series(
        np.nan,
        index=data.index,
        name="har_rv_variance_1d",
        dtype=float,
    )

    valid_positions = np.flatnonzero(valid.to_numpy())
    if len(valid_positions) <= minimum_training_observations:
        raise ValueError("Insufficient valid observations for HAR-RV.")

    fitted_model = None
    first_valid_position = int(valid_positions[0])

    for position in range(
        first_valid_position + minimum_training_observations,
        len(data),
    ):
        if not valid.iloc[position]:
            continue

        prior_positions = valid_positions[valid_positions < position]
        if len(prior_positions) < minimum_training_observations:
            continue

        observations_available = len(prior_positions)

        if (
            fitted_model is None
            or (
                observations_available - minimum_training_observations
            ) % reestimate_every == 0
        ):
            train_positions = prior_positions
            x_train = log_predictors.iloc[train_positions].to_numpy()
            y_train = log_target.iloc[train_positions].to_numpy()

            fitted_model = LinearRegression()
            fitted_model.fit(x_train, y_train)

        x_current = log_predictors.iloc[[position]].to_numpy()
        predicted_log_variance = float(fitted_model.predict(x_current)[0])

        # Protect exponentiation from overflow or underflow.
        predicted_log_variance = float(
            np.clip(predicted_log_variance, -700, 700)
        )
        forecast = float(np.exp(predicted_log_variance))

        if not np.isfinite(forecast) or forecast <= 0:
            raise RuntimeError(
                f"HAR-RV produced an invalid forecast at position {position}."
            )

        forecasts.iloc[position] = forecast

    return forecasts
