import numpy as np
import pandas as pd
import pytest

from src.forecasting import generate_ewma_forecasts


def make_features(n=300):
    dates = pd.bdate_range("2020-01-01", periods=n)
    returns = np.linspace(-0.02, 0.02, n)
    return pd.DataFrame(
        {
            "log_return": returns,
            "target_rv_1": np.full(n, 0.0001),
            "target_rv_5": np.full(n, 0.0005),
        },
        index=dates,
    )


def test_forecasts_begin_at_test_start_and_have_both_horizons():
    features = make_features()
    result = generate_ewma_forecasts(
        features,
        test_start="2020-12-01",
        minimum_training_observations=100,
    )

    assert result.index.min() >= pd.Timestamp("2020-12-01")
    assert "ewma_variance_1d" in result
    assert "ewma_variance_5d" in result
    assert (result["ewma_variance_1d"] > 0).all()
    np.testing.assert_allclose(
        result["ewma_variance_5d"],
        5 * result["ewma_variance_1d"],
    )


def test_future_returns_do_not_change_earlier_forecasts():
    features = make_features()
    test_start = pd.Timestamp("2020-12-01")

    original = generate_ewma_forecasts(
        features,
        test_start=str(test_start.date()),
        minimum_training_observations=100,
    )

    changed = features.copy()
    changed.loc[changed.index > test_start, "log_return"] = 0.5

    modified = generate_ewma_forecasts(
        changed,
        test_start=str(test_start.date()),
        minimum_training_observations=100,
    )

    assert original.loc[test_start, "ewma_variance_1d"] == pytest.approx(
        modified.loc[test_start, "ewma_variance_1d"]
    )


def test_rejects_insufficient_training_history():
    features = make_features(n=30)

    with pytest.raises(ValueError, match="Insufficient"):
        generate_ewma_forecasts(
            features,
            test_start="2020-01-20",
            minimum_training_observations=100,
        )

def test_rejects_missing_columns():
    features = make_features().drop(columns=["log_return"])

    with pytest.raises(ValueError, match="Missing required columns"):
        generate_ewma_forecasts(features, test_start="2020-12-01")

def test_initialization_uses_only_initial_training_window():
    features = make_features(n=12)
    features["log_return"] = [
        0.01, -0.02, 0.03, 0.04, -0.01, 0.02,
        -0.03, 0.01, 0.02, -0.01, 0.03, -0.02,
    ]
    test_start = features.index[6]

    result = generate_ewma_forecasts(
        features,
        test_start=str(test_start.date()),
        decay=0.5,
        minimum_training_observations=2,
    )

    variance = np.mean(features["log_return"].iloc[:2] ** 2)
    for observed_return in features["log_return"].iloc[2:7]:
        variance = 0.5 * variance + 0.5 * observed_return**2

    assert result.loc[test_start, "ewma_variance_1d"] == pytest.approx(
        variance
    )

def test_can_forecast_latest_date_without_future_targets():
    features = make_features(n=300).drop(
        columns=["target_rv_1", "target_rv_5"]
    )

    result = generate_ewma_forecasts(
        features,
        test_start=str(features.index[-1].date()),
        minimum_training_observations=100,
    )

    assert result.index.max() == features.index[-1]
    assert pd.notna(result.iloc[-1]["ewma_variance_1d"])
    assert pd.notna(result.iloc[-1]["ewma_variance_5d"])