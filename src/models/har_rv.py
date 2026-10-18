
"""HAR-RV volatility forecasting with expanding-window regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# A small variance floor allows zero squared returns to be log-transformed.
# It affects predictor inputs only; zero targets remain excluded from
# log-target regression because log(0) is undefined.
_PREDICTOR_FLOOR = 1e-12


def _prepare_log_predictors(data: pd.DataFrame) -> pd.DataFrame:
    """Build finite log-RV predictors without discarding zero returns."""
    predictors = pd.DataFrame(
        {
            "daily": data["squared_return"],
            "weekly": data["rv_5"] / 5,
            "monthly": data["rv_22"] / 22,
        },
        index=data.index,
    )

    # Negative or missing variance inputs are invalid; zero is floored.
    predictors = predictors.where(predictors >= 0)
    return np.log(predictors.clip(lower=_PREDICTOR_FLOOR))


def _prepare_data(
    features: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Validate inputs and prepare predictors and log targets."""
    required = {
        "squared_return",
        "rv_5",
        "rv_22",
        target_column,
    }
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

    log_predictors = _prepare_log_predictors(data)
    log_target = np.log(data[target_column].where(data[target_column] > 0))

    valid_predictors = (
        log_predictors.notna().all(axis=1)
        & np.isfinite(log_predictors).all(axis=1)
    )
    valid_target = log_target.notna() & np.isfinite(log_target)

    return data, log_predictors, valid_predictors & valid_target


def _validate_training_settings(
    minimum_training_observations: int,
    reestimate_every: int,
) -> None:
    """Validate expanding-window training settings."""
    if (
        not isinstance(minimum_training_observations, (int, np.integer))
        or isinstance(minimum_training_observations, (bool, np.bool_))
        or minimum_training_observations < 30
    ):
        raise ValueError("minimum_training_observations must be at least 30.")

    if (
        not isinstance(reestimate_every, (int, np.integer))
        or isinstance(reestimate_every, (bool, np.bool_))
        or reestimate_every < 1
    ):
        raise ValueError("reestimate_every must be at least 1.")


def _positive_forecast(predicted_log_variance: float) -> float:
    """Convert a log-variance prediction to a finite positive forecast."""
    forecast = float(np.exp(np.clip(predicted_log_variance, -700, 700)))
    if not np.isfinite(forecast) or forecast <= 0:
        raise RuntimeError("HAR-RV produced an invalid variance forecast.")
    return forecast


def har_rv_variance_forecasts(
    features: pd.DataFrame,
    minimum_training_observations: int = 252,
    reestimate_every: int = 21,
) -> pd.Series:
    """Forecast next-day variance using information available at each close.

    A forecast at t uses predictors through t. Only historical rows with
    valid predictors and strictly positive next-day targets are used for
    log-target training. A smearing correction is estimated from the
    current fitted model's eligible historical training residuals only.
    """
    _validate_training_settings(
        minimum_training_observations, reestimate_every
    )
    data, log_predictors, valid_training_rows = _prepare_data(
        features, "target_rv_1"
    )

    # Forecasts require valid predictors, not a valid target at the origin.
    valid_predictors = (
        log_predictors.notna().all(axis=1)
        & np.isfinite(log_predictors).all(axis=1)
    )
    valid_positions = np.flatnonzero(valid_training_rows.to_numpy())

    if len(valid_positions) < minimum_training_observations:
        raise ValueError("Insufficient valid observations for HAR-RV.")

    forecasts = pd.Series(
        np.nan,
        index=data.index,
        name="har_rv_variance_1d",
        dtype=float,
    )

    fitted_model: LinearRegression | None = None
    smearing_factor = 1.0

    for position in range(len(data) - 1):
        if not valid_predictors.iloc[position]:
            continue

        # Training labels must be strictly earlier than the forecast origin.
        prior_positions = valid_positions[valid_positions < position]
        observations_available = len(prior_positions)

        if observations_available < minimum_training_observations:
            continue

        if (
            fitted_model is None
            or (
                observations_available - minimum_training_observations
            ) % reestimate_every == 0
        ):
            training_x = log_predictors.iloc[prior_positions].to_numpy()
            training_y = np.log(
                data["target_rv_1"].iloc[prior_positions].to_numpy()
            )

            fitted_model = LinearRegression()
            fitted_model.fit(training_x, training_y)

            # Duan smearing factor: estimate retransformation bias using
            # only the labels and predictors eligible at this model refit.
            training_residuals = (
                training_y - fitted_model.predict(training_x)
            )
            smearing_factor = float(np.mean(np.exp(training_residuals)))

            if not np.isfinite(smearing_factor) or smearing_factor <= 0:
                raise RuntimeError("HAR-RV produced an invalid smearing factor.")

        predicted_log_variance = float(
            fitted_model.predict(
                log_predictors.iloc[[position]].to_numpy()
            )[0]
        )

        # Apply the historical smearing correction on the variance scale.
        forecasts.iloc[position] = _positive_forecast(
            predicted_log_variance + np.log(smearing_factor)
        )

    return forecasts


def har_rv_five_day_variance_forecasts(
    features: pd.DataFrame,
    minimum_training_observations: int = 252,
    reestimate_every: int = 21,
) -> pd.Series:
    """Forecast cumulative variance over the next five trading sessions.

    A training label at origin s is eligible at origin t only when s + 5
    is strictly earlier than t's row position, preserving the existing
    conservative label-availability convention.
    """
    _validate_training_settings(
        minimum_training_observations, reestimate_every
    )
    data, log_predictors, valid_training_rows = _prepare_data(
        features, "target_rv_5"
    )

    valid_predictors = (
        log_predictors.notna().all(axis=1)
        & np.isfinite(log_predictors).all(axis=1)
    )
    valid_positions = np.flatnonzero(valid_training_rows.to_numpy())

    forecasts = pd.Series(
        np.nan,
        index=data.index,
        name="har_rv_variance_5d",
        dtype=float,
    )

    fitted_model: LinearRegression | None = None
    last_fit_observations = 0

    # The final five origins have no complete five-session target.
    for position in range(max(0, len(data) - 5)):
        if not valid_predictors.iloc[position]:
            continue

        # Keep the original conservative rule: s + 5 < t.
        eligible = valid_positions[valid_positions <= position - 6]
        observations_available = len(eligible)

        if observations_available < minimum_training_observations:
            continue

        if (
            fitted_model is None
            or observations_available - last_fit_observations
            >= reestimate_every
        ):
            fitted_model = LinearRegression()
            fitted_model.fit(
                log_predictors.iloc[eligible].to_numpy(),
                np.log(data["target_rv_5"].iloc[eligible].to_numpy()),
            )
            last_fit_observations = observations_available

        predicted_log_variance = float(
            fitted_model.predict(
                log_predictors.iloc[[position]].to_numpy()
            )[0]
        )
        forecasts.iloc[position] = _positive_forecast(
            predicted_log_variance
        )

    if forecasts.notna().sum() == 0:
        raise ValueError("Insufficient valid observations for five-day HAR-RV.")

    return forecasts

def har_rv_prospective_variance_forecast(
    features: pd.DataFrame,
    minimum_training_observations: int = 252,
    reestimate_every: int = 21,
) -> float:
    """Forecast next-day variance using labels observable at the latest close."""
    _validate_training_settings(
        minimum_training_observations, reestimate_every
    )

    required = {"squared_return", "rv_5", "rv_22"}
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

    log_predictors = _prepare_log_predictors(data)
    valid_predictors = (
        log_predictors.notna().all(axis=1)
        & np.isfinite(log_predictors).all(axis=1)
    )

    # The target for origin s is squared_return at s+1.
    # At the latest origin t, labels through t-1 are observable.
    target = data["squared_return"].shift(-1)
    log_target = np.log(target.where(target > 0))
    valid_training = (
        valid_predictors
        & log_target.notna()
        & np.isfinite(log_target)
    )

    latest_position = len(data) - 1
    if not valid_predictors.iloc[latest_position]:
        raise ValueError("Latest row has invalid HAR-RV predictors.")

    # A one-day label at s is observable at origin t when s < t.
    eligible_positions = np.flatnonzero(
        valid_training.iloc[:latest_position].to_numpy()
    )

    if len(eligible_positions) < minimum_training_observations:
        raise ValueError("Insufficient valid observations for prospective HAR-RV.")

    model = LinearRegression()
    model.fit(
        log_predictors.iloc[eligible_positions].to_numpy(),
        log_target.iloc[eligible_positions].to_numpy(),
    )

    training_residuals = (
        log_target.iloc[eligible_positions].to_numpy()
        - model.predict(log_predictors.iloc[eligible_positions].to_numpy())
    )
    smearing_factor = float(np.mean(np.exp(training_residuals)))

    if not np.isfinite(smearing_factor) or smearing_factor <= 0:
        raise RuntimeError("HAR-RV produced an invalid smearing factor.")

    predicted_log_variance = float(
        model.predict(log_predictors.iloc[[latest_position]].to_numpy())[0]
    )

    return _positive_forecast(
        predicted_log_variance + np.log(smearing_factor)
    )

def har_rv_five_day_prospective_variance_forecast(
    features: pd.DataFrame,
    minimum_training_observations: int = 252,
) -> float:
    """Forecast cumulative five-session variance at the latest close."""
    _validate_training_settings(minimum_training_observations, 1)

    required = {"squared_return", "rv_5", "rv_22"}
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

    log_predictors = _prepare_log_predictors(data)
    valid_predictors = (
        log_predictors.notna().all(axis=1)
        & np.isfinite(log_predictors).all(axis=1)
    )

    # A label at s is complete only after the five subsequent sessions.
    future_five_day_target = (
        data["squared_return"]
        .rolling(5, min_periods=5)
        .sum()
        .shift(-5)
    )
    log_target = np.log(
        future_five_day_target.where(future_five_day_target > 0)
    )
    valid_training = (
        valid_predictors
        & log_target.notna()
        & np.isfinite(log_target)
    )

    latest_position = len(data) - 1
    if not valid_predictors.iloc[latest_position]:
        raise ValueError("Latest row has invalid HAR-RV predictors.")

    # The label at s covers s+1 through s+5. It must be fully
    # observable before the latest forecast origin.
    eligible_positions = np.flatnonzero(
        valid_training.iloc[: max(0, latest_position - 5)].to_numpy()
    )
    if len(eligible_positions) < minimum_training_observations:
        raise ValueError(
            "Insufficient valid observations for prospective five-day HAR-RV."
        )

    model = LinearRegression()
    model.fit(
        log_predictors.iloc[eligible_positions].to_numpy(),
        log_target.iloc[eligible_positions].to_numpy(),
    )

    predicted_log_variance = float(
        model.predict(log_predictors.iloc[[latest_position]].to_numpy())[0]
    )

    # Preserve the existing five-day model's retransformation treatment:
    # no smearing correction.
    return _positive_forecast(predicted_log_variance)

def _five_day_eligible_training_positions(
    valid_training: pd.Series,
    latest_position: int,
) -> np.ndarray:
    """Return training positions whose five-day labels end before the origin."""
    return np.flatnonzero(
        valid_training.iloc[: max(0, latest_position - 5)].to_numpy()
    )