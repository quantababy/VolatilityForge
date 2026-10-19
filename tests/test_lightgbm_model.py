
import numpy as np
import pandas as pd
import pytest

from src.models.lightgbm_model import lightgbm_variance_forecasts


def make_features(n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2020-01-01", periods=n)

    returns = rng.normal(0, 0.01, n)
    squared_returns = returns**2

    features = pd.DataFrame(index=index)
    features["log_return"] = returns
    features["squared_return"] = squared_returns
    features["rv_5"] = (
        pd.Series(squared_returns, index=index)
        .rolling(5, min_periods=5)
        .sum()
    )
    features["rv_22"] = (
        pd.Series(squared_returns, index=index)
        .rolling(22, min_periods=22)
        .sum()
    )
    features["VIX_Close"] = 15 + rng.uniform(0, 15, n)
    features["target_rv_1"] = pd.Series(
        squared_returns, index=index
    ).shift(-1)

    return features


def test_forecasts_preserve_index_and_name() -> None:
    features = make_features()
    forecasts = lightgbm_variance_forecasts(
        features,
        minimum_training_observations=40,
        reestimate_every=10,
    )

    pd.testing.assert_index_equal(
        forecasts.index, features.index.rename("Date")
    )
    assert forecasts.name == "lightgbm_variance_1d"


def test_available_forecasts_are_positive_and_finite() -> None:
    forecasts = lightgbm_variance_forecasts(
        make_features(),
        minimum_training_observations=40,
        reestimate_every=10,
    )
    available = forecasts.dropna()

    assert len(available) > 0
    assert np.isfinite(available).all()
    assert (available > 0).all()


def test_forecasts_require_prior_training_history() -> None:
    features = make_features()
    forecasts = lightgbm_variance_forecasts(
        features,
        minimum_training_observations=40,
        reestimate_every=10,
    )

    # The first 22 rows lack a complete monthly RV feature.
    # Forty valid prior labels are then required before forecasting.
    first_forecast = forecasts.first_valid_index()
    assert first_forecast is not None

    first_position = features.index.get_loc(first_forecast)
    assert first_position >= 61
    assert forecasts.iloc[:first_position].isna().all()


def test_future_target_changes_do_not_change_earlier_forecasts() -> None:
    features = make_features()
    changed = features.copy()

    # Alter labels only; leave predictors unchanged.
    changed.iloc[100:, changed.columns.get_loc("target_rv_1")] *= 10

    kwargs = {
        "minimum_training_observations": 40,
        "reestimate_every": 10,
    }
    original_forecasts = lightgbm_variance_forecasts(features, **kwargs)
    changed_forecasts = lightgbm_variance_forecasts(changed, **kwargs)

    # The label at position 100 cannot influence the forecast at position 100.
    pd.testing.assert_series_equal(
        original_forecasts.iloc[:101],
        changed_forecasts.iloc[:101],
    )


def test_future_predictor_changes_do_not_change_earlier_forecasts() -> None:
    features = make_features()
    changed = features.copy()
    cutoff = 100

    # Modify predictors only from the cutoff onward.
    changed.loc[changed.index[cutoff]:, "VIX_Close"] *= 2
    changed.loc[changed.index[cutoff]:, "log_return"] *= 3

    kwargs = {
        "minimum_training_observations": 40,
        "reestimate_every": 10,
    }
    original_forecasts = lightgbm_variance_forecasts(features, **kwargs)
    changed_forecasts = lightgbm_variance_forecasts(changed, **kwargs)

    # Future predictors must not change forecasts made before the cutoff.
    pd.testing.assert_series_equal(
        original_forecasts.iloc[:cutoff],
        changed_forecasts.iloc[:cutoff],
    )

def test_rejects_missing_columns() -> None:
    features = make_features().drop(columns=["VIX_Close"])

    with pytest.raises(ValueError, match="Missing required columns"):
        lightgbm_variance_forecasts(features)


def test_rejects_empty_features() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        lightgbm_variance_forecasts(make_features().iloc[:0])


def test_rejects_duplicate_dates() -> None:
    features = make_features()
    features.index = list(features.index[:-1]) + [features.index[-2]]

    with pytest.raises(ValueError, match="duplicate dates"):
        lightgbm_variance_forecasts(features)


def test_rejects_unsorted_dates() -> None:
    features = make_features().iloc[::-1]

    with pytest.raises(ValueError, match="sorted by date"):
        lightgbm_variance_forecasts(features)


@pytest.mark.parametrize("minimum", [0, 10, 29])
def test_rejects_too_small_training_window(minimum: int) -> None:
    with pytest.raises(ValueError, match="at least 30"):
        lightgbm_variance_forecasts(
            make_features(), minimum_training_observations=minimum
        )


def test_rejects_invalid_reestimation_frequency() -> None:
    with pytest.raises(ValueError, match="reestimate_every"):
        lightgbm_variance_forecasts(
            make_features(),
            minimum_training_observations=40,
            reestimate_every=0,
        )


def test_zero_targets_are_supported() -> None:
    features = make_features()
    features["target_rv_1"] = features["target_rv_1"].clip(lower=0)
    features.iloc[70, features.columns.get_loc("target_rv_1")] = 0.0

    forecasts = lightgbm_variance_forecasts(
        features,
        minimum_training_observations=40,
        reestimate_every=10,
    )

    assert (forecasts.dropna() > 0).all()

def test_prospective_forecast_does_not_require_future_target() -> None:
    from src.models.lightgbm_model import (
        lightgbm_prospective_variance_forecast,
    )

    features = make_features().drop(columns=["target_rv_1"])

    forecast = lightgbm_prospective_variance_forecast(
        features,
        minimum_training_observations=40,
        reestimate_every=10,
    )

    assert np.isfinite(forecast)
    assert forecast > 0


def test_prospective_forecast_uses_only_matured_training_labels(
    monkeypatch,
) -> None:
    import src.models.lightgbm_model as lightgbm_module

    features = make_features(100).drop(columns=["target_rv_1"])
    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            captured["X"] = X.copy()
            captured["y"] = y.copy()
            return self

        def predict(self, X):
            return np.array([1e-4])

    monkeypatch.setattr(
        lightgbm_module,
        "LGBMRegressor",
        FakeModel,
    )

    lightgbm_module.lightgbm_prospective_variance_forecast(
        features,
        minimum_training_observations=30,
    )

    # A target at date s is the squared return at s+1.
    # Only dates strictly before the latest forecast origin may train.
    latest_position = len(features) - 1
    predictors = features.loc[:, lightgbm_module.FEATURE_COLUMNS]
    target = features["squared_return"].shift(-1)

    finite_predictors = np.isfinite(predictors.to_numpy()).all(axis=1)
    valid_predictors = (
        predictors.notna().all(axis=1)
        & pd.Series(finite_predictors, index=features.index)
    )
    valid_targets = target.notna() & np.isfinite(target) & (target >= 0)
    valid_training = valid_predictors & valid_targets

    expected_positions = np.flatnonzero(
        valid_training.iloc[:latest_position].to_numpy()
    )

    assert captured["X"].index.equals(features.index[expected_positions])
    pd.testing.assert_series_equal(
        captured["y"],
        target.iloc[expected_positions],
        check_names=False,
    )
    assert features.index[-1] not in captured["X"].index
