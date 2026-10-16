
import numpy as np
import pandas as pd
import pytest

from src.models.har_rv import (
    har_rv_five_day_variance_forecasts,
    har_rv_variance_forecasts,
)


def make_features(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, n)
    squared_returns = returns**2
    index = pd.bdate_range("2020-01-01", periods=n)

    squared = pd.Series(squared_returns, index=index)
    return pd.DataFrame(
        {
            "squared_return": squared,
            "rv_5": squared.rolling(5, min_periods=5).sum(),
            "rv_22": squared.rolling(22, min_periods=22).sum(),
            "target_rv_1": squared.shift(-1),
        },
        index=index,
    )


def add_five_day_target(features: pd.DataFrame) -> pd.DataFrame:
    """Add the next-five-session sum of squared returns."""
    result = features.copy()
    result["target_rv_5"] = (
        result["squared_return"]
        .rolling(5, min_periods=5)
        .sum()
        .shift(-5)
    )
    return result


def test_forecasts_preserve_index_and_name() -> None:
    features = make_features()
    forecasts = har_rv_variance_forecasts(
        features, minimum_training_observations=40, reestimate_every=10
    )

    pd.testing.assert_index_equal(
        forecasts.index, features.index.rename("Date")
    )
    assert forecasts.name == "har_rv_variance_1d"


def test_forecasts_are_positive_and_finite_when_available() -> None:
    forecasts = har_rv_variance_forecasts(
        make_features(), minimum_training_observations=40, reestimate_every=10
    )
    available = forecasts.dropna()

    assert len(available) > 0
    assert np.isfinite(available).all()
    assert (available > 0).all()


def test_initial_forecasts_require_training_history() -> None:
    features = make_features()
    forecasts = har_rv_variance_forecasts(
        features, minimum_training_observations=40, reestimate_every=10
    )

    # Monthly RV first becomes available at position 21.
    # Forty valid training rows are required before forecasting.
    assert forecasts.iloc[:61].isna().all()
    assert forecasts.iloc[61:-1].notna().all()
    assert forecasts.iloc[-1:].isna().all()


def test_future_observations_do_not_change_earlier_forecasts() -> None:
    features = make_features()
    changed = features.copy()

    changed.iloc[90:, changed.columns.get_loc("squared_return")] *= 3
    changed["rv_5"] = changed["squared_return"].rolling(
        5, min_periods=5
    ).sum()
    changed["rv_22"] = changed["squared_return"].rolling(
        22, min_periods=22
    ).sum()
    changed["target_rv_1"] = changed["squared_return"].shift(-1)

    original_forecasts = har_rv_variance_forecasts(
        features, minimum_training_observations=40, reestimate_every=10
    )
    changed_forecasts = har_rv_variance_forecasts(
        changed, minimum_training_observations=40, reestimate_every=10
    )

    pd.testing.assert_series_equal(
        original_forecasts.iloc[:90],
        changed_forecasts.iloc[:90],
    )


def test_rejects_missing_columns() -> None:
    features = make_features().drop(columns=["rv_22"])

    with pytest.raises(ValueError, match="Missing required columns"):
        har_rv_variance_forecasts(features)


def test_rejects_empty_features() -> None:
    features = make_features().iloc[:0]

    with pytest.raises(ValueError, match="must not be empty"):
        har_rv_variance_forecasts(features)


def test_rejects_duplicate_dates() -> None:
    features = make_features()
    features.index = list(features.index[:-1]) + [features.index[-2]]

    with pytest.raises(ValueError, match="duplicate dates"):
        har_rv_variance_forecasts(features)


def test_rejects_unsorted_dates() -> None:
    features = make_features().iloc[::-1]

    with pytest.raises(ValueError, match="sorted by date"):
        har_rv_variance_forecasts(features)


@pytest.mark.parametrize("minimum", [0, 10, 29])
def test_rejects_too_small_training_window(minimum: int) -> None:
    with pytest.raises(ValueError, match="at least 30"):
        har_rv_variance_forecasts(
            make_features(), minimum_training_observations=minimum
        )


def test_rejects_invalid_reestimation_frequency() -> None:
    with pytest.raises(ValueError, match="reestimate_every"):
        har_rv_variance_forecasts(
            make_features(),
            minimum_training_observations=40,
            reestimate_every=0,
        )


def test_forecast_does_not_use_current_or_future_target_labels() -> None:
    features = make_features()
    changed = features.copy()
    changed.iloc[70:, changed.columns.get_loc("target_rv_1")] *= 10

    original_forecasts = har_rv_variance_forecasts(
        features, minimum_training_observations=40, reestimate_every=10
    )
    changed_forecasts = har_rv_variance_forecasts(
        changed, minimum_training_observations=40, reestimate_every=10
    )

    pd.testing.assert_series_equal(
        original_forecasts.iloc[:71],
        changed_forecasts.iloc[:71],
    )


def test_five_day_forecasts_preserve_index_and_name() -> None:
    features = add_five_day_target(make_features())
    forecasts = har_rv_five_day_variance_forecasts(
        features, minimum_training_observations=40, reestimate_every=10
    )

    pd.testing.assert_index_equal(
        forecasts.index, features.index.rename("Date")
    )
    assert forecasts.name == "har_rv_variance_5d"


def test_five_day_forecasts_are_positive_and_finite() -> None:
    features = add_five_day_target(make_features())
    forecasts = har_rv_five_day_variance_forecasts(
        features, minimum_training_observations=40, reestimate_every=10
    )
    available = forecasts.dropna()

    assert len(available) > 0
    assert np.isfinite(available).all()
    assert (available > 0).all()
    assert forecasts.iloc[-5:].isna().all()


def test_five_day_forecasts_do_not_use_unobservable_training_labels() -> None:
    features = add_five_day_target(make_features())
    changed = features.copy()

    # The label at position 70 covers outcomes through position 75.
    changed.iloc[70, changed.columns.get_loc("target_rv_5")] *= 10

    original_forecasts = har_rv_five_day_variance_forecasts(
        features, minimum_training_observations=40, reestimate_every=10
    )
    changed_forecasts = har_rv_five_day_variance_forecasts(
        changed, minimum_training_observations=40, reestimate_every=10
    )

    pd.testing.assert_series_equal(
        original_forecasts.iloc[:76],
        changed_forecasts.iloc[:76],
    )


def test_five_day_forecasts_reject_missing_target_column() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        har_rv_five_day_variance_forecasts(make_features())
