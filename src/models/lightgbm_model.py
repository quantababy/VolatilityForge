
"""LightGBM next-day variance forecasting with expanding-window training."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

FEATURE_COLUMNS = (
    "log_return",
    "squared_return",
    "rv_5",
    "rv_22",
    "VIX_Close",
)


def lightgbm_variance_forecasts(
    features: pd.DataFrame,
    minimum_training_observations: int = 252,
    reestimate_every: int = 21,
    random_state: int = 42,
) -> pd.Series:
    """Forecast next-day variance without using current or future labels.

    Predictors at date t use information available through the close at t.
    The target at date t is the squared return on the next trading day.
    Training labels are restricted to dates strictly earlier than t.
    """
    required = {*FEATURE_COLUMNS, "target_rv_1"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if features.empty:
        raise ValueError("features must not be empty.")
    if minimum_training_observations < 30:
        raise ValueError("minimum_training_observations must be at least 30.")
    if reestimate_every < 1:
        raise ValueError("reestimate_every must be at least 1.")

    data = features.copy()
    data.index = pd.DatetimeIndex(data.index).normalize()
    data.index.name = "Date"

    if data.index.has_duplicates:
        raise ValueError("features contains duplicate dates.")
    if not data.index.is_monotonic_increasing:
        raise ValueError("features must be sorted by date.")

    predictors = data.loc[:, FEATURE_COLUMNS].astype(float)
    target = data["target_rv_1"].astype(float)

    finite_predictors = np.isfinite(predictors.to_numpy()).all(axis=1)
    valid_predictors = (
        predictors.notna().all(axis=1)
        & pd.Series(finite_predictors, index=data.index)
    )
    valid_targets = target.notna() & np.isfinite(target) & (target >= 0)
    valid = valid_predictors & valid_targets

    valid_positions = np.flatnonzero(valid.to_numpy())
    if len(valid_positions) <= minimum_training_observations:
        raise ValueError("Insufficient valid observations for LightGBM.")

    forecasts = pd.Series(
        np.nan,
        index=data.index,
        name="lightgbm_variance_1d",
        dtype=float,
    )

    fitted_model: LGBMRegressor | None = None
    last_fit_count = 0

    for position in range(len(data)):
        if not valid_predictors.iloc[position]:
            continue

        # Strictly exclude the label at the forecast origin and all later labels.
        prior_positions = valid_positions[valid_positions < position]
        training_count = len(prior_positions)

        if training_count < minimum_training_observations:
            continue

        if (
            fitted_model is None
            or training_count - last_fit_count >= reestimate_every
        ):
            x_train = predictors.iloc[prior_positions]
            y_train = target.iloc[prior_positions]

            fitted_model = LGBMRegressor(
                objective="regression",
                n_estimators=200,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=-1,
                min_child_samples=20,
                reg_lambda=1.0,
                random_state=random_state,
                deterministic=True,
                force_col_wise=True,
                n_jobs=1,
                verbosity=-1,
            )
            fitted_model.fit(x_train, y_train)
            last_fit_count = training_count

        prediction = float(
            fitted_model.predict(predictors.iloc[[position]])[0]
        )

        # QLIKE requires strictly positive variance forecasts.
        training_scale = float(target.iloc[prior_positions].median())
        positive_floor = max(training_scale * 1e-6, 1e-12)
        prediction = max(prediction, positive_floor)

        if not np.isfinite(prediction):
            raise RuntimeError(
                f"LightGBM produced an invalid forecast at {data.index[position]}."
            )

        forecasts.iloc[position] = prediction

    return forecasts

def lightgbm_prospective_variance_forecast(
    features: pd.DataFrame,
    minimum_training_observations: int = 252,
    reestimate_every: int = 21,
    random_state: int = 42,
) -> float:
    """Forecast next-day variance at the latest date using known history only."""
    required = set(FEATURE_COLUMNS)
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if features.empty:
        raise ValueError("features must not be empty.")
    if minimum_training_observations < 30:
        raise ValueError("minimum_training_observations must be at least 30.")
    if reestimate_every < 1:
        raise ValueError("reestimate_every must be at least 1.")

    data = features.copy()
    data.index = pd.DatetimeIndex(data.index).normalize()
    data.index.name = "Date"

    if data.index.has_duplicates:
        raise ValueError("features contains duplicate dates.")
    if not data.index.is_monotonic_increasing:
        raise ValueError("features must be sorted by date.")

    predictors = data.loc[:, FEATURE_COLUMNS].astype(float)
    finite_predictors = np.isfinite(predictors.to_numpy()).all(axis=1)
    valid_predictors = (
        predictors.notna().all(axis=1)
        & pd.Series(finite_predictors, index=data.index)
    )

    if not valid_predictors.iloc[-1]:
        raise ValueError("Latest date has invalid or incomplete predictors.")

    # At origin t, the label at s is usable only if s < t.
    # target_rv_1 at s is the squared return observed at s+1.
    target = data["squared_return"].shift(-1).astype(float)
    valid_targets = target.notna() & np.isfinite(target) & (target >= 0)
    valid_training = valid_predictors & valid_targets

    latest_position = len(data) - 1
    prior_positions = np.flatnonzero(
        valid_training.iloc[:latest_position].to_numpy()
    )

    if len(prior_positions) < minimum_training_observations:
        raise ValueError(
            "Insufficient valid historical observations for prospective "
            "LightGBM forecast."
        )

    x_train = predictors.iloc[prior_positions]
    y_train = target.iloc[prior_positions]

    model = LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=20,
        reg_lambda=1.0,
        random_state=random_state,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)

    prediction = float(model.predict(predictors.iloc[[-1]])[0])

    training_scale = float(y_train.median())
    positive_floor = max(training_scale * 1e-6, 1e-12)
    prediction = max(prediction, positive_floor)

    if not np.isfinite(prediction):
        raise RuntimeError("LightGBM produced an invalid prospective forecast.")

    return prediction


def lightgbm_five_day_variance_forecasts(
    features: pd.DataFrame,
    minimum_training_observations: int = 252,
    reestimate_every: int = 21,
    random_state: int = 42,
) -> pd.Series:
    """Forecast cumulative five-day variance using matured labels only."""
    required = {*FEATURE_COLUMNS, "target_rv_5"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if features.empty:
        raise ValueError("features must not be empty.")
    if minimum_training_observations < 30:
        raise ValueError("minimum_training_observations must be at least 30.")
    if reestimate_every < 1:
        raise ValueError("reestimate_every must be at least 1.")

    data = features.copy()
    data.index = pd.DatetimeIndex(data.index).normalize()
    data.index.name = "Date"

    if data.index.has_duplicates:
        raise ValueError("features contains duplicate dates.")
    if not data.index.is_monotonic_increasing:
        raise ValueError("features must be sorted by date.")

    predictors = data.loc[:, FEATURE_COLUMNS].astype(float)
    target = data["target_rv_5"].astype(float)

    finite_predictors = np.isfinite(predictors.to_numpy()).all(axis=1)
    valid_predictors = (
        predictors.notna().all(axis=1)
        & pd.Series(finite_predictors, index=data.index)
    )
    valid_targets = target.notna() & np.isfinite(target) & (target >= 0)
    valid_training = valid_predictors & valid_targets
    valid_positions = np.flatnonzero(valid_training.to_numpy())

    forecasts = pd.Series(
        np.nan,
        index=data.index,
        name="lightgbm_variance_5d",
        dtype=float,
    )

    fitted_model: LGBMRegressor | None = None
    last_fit_count = 0

    # The final five origins have no complete five-session target.
    for position in range(max(0, len(data) - 5)):
        if not valid_predictors.iloc[position]:
            continue

        # Protocol rule: s + 5 < t, equivalent to s <= t - 6.
        eligible = valid_positions[valid_positions <= position - 6]
        training_count = len(eligible)

        if training_count < minimum_training_observations:
            continue

        if (
            fitted_model is None
            or training_count - last_fit_count >= reestimate_every
        ):
            fitted_model = LGBMRegressor(
                objective="regression",
                n_estimators=200,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=-1,
                min_child_samples=20,
                reg_lambda=1.0,
                random_state=random_state,
                deterministic=True,
                force_col_wise=True,
                n_jobs=1,
                verbosity=-1,
            )
            fitted_model.fit(
                predictors.iloc[eligible],
                target.iloc[eligible],
            )
            last_fit_count = training_count

        prediction = float(
            fitted_model.predict(predictors.iloc[[position]])[0]
        )
        training_scale = float(target.iloc[eligible].median())
        positive_floor = max(training_scale * 1e-6, 1e-12)
        prediction = max(prediction, positive_floor)

        if not np.isfinite(prediction):
            raise RuntimeError(
                "LightGBM produced an invalid five-day forecast at "
                f"{data.index[position]}."
            )

        forecasts.iloc[position] = prediction

    if forecasts.notna().sum() == 0:
        raise ValueError(
            "Insufficient valid observations for five-day LightGBM."
        )

    return forecasts
