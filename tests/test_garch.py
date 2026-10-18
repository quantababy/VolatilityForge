"""Tests for GARCH(1,1) variance forecasting."""

import numpy as np
import pandas as pd
import pytest

from src.models.garch import (
    garch_five_day_variance_forecasts,
    garch_prospective_variance_forecast,
    garch_variance_forecasts,
    garch_five_day_prospective_variance_forecast,
)


def make_returns(size: int = 100, seed: int = 42) -> pd.Series:
    """Create reproducible synthetic daily returns."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 0.01, size)
    dates = pd.bdate_range("2020-01-01", periods=size)
    return pd.Series(values, index=dates, name="log_return")


def test_forecasts_preserve_index_and_name() -> None:
    returns = make_returns(80)
    forecasts = garch_variance_forecasts(
        returns,
        minimum_training_observations=30,
        reestimate_every=10,
    )

    assert forecasts.index.equals(returns.index)
    assert forecasts.name == "garch_variance_1d"



def test_initial_training_window_has_no_forecasts() -> None:
    returns = make_returns(80)
    minimum = 30

    forecasts = garch_variance_forecasts(
        returns,
        minimum_training_observations=minimum,
        reestimate_every=10,
    )

    # After observing the 30th return, the next-day forecast is
    # stored at that return's index (position minimum - 1).
    assert forecasts.iloc[: minimum - 1].isna().all()
    assert pd.notna(forecasts.iloc[minimum - 1])
    assert forecasts.iloc[minimum:-1].notna().all()
    assert pd.isna(forecasts.iloc[-1])



def test_forecasts_are_finite_and_positive() -> None:
    returns = make_returns(80)
    forecasts = garch_variance_forecasts(
        returns,
        minimum_training_observations=30,
        reestimate_every=10,
    )

    valid = forecasts.dropna()
    assert np.isfinite(valid.to_numpy()).all()
    assert (valid > 0).all()


def test_rejects_empty_returns() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        garch_variance_forecasts(pd.Series(dtype=float))


def test_rejects_missing_returns() -> None:
    returns = make_returns(80)
    returns.iloc[10] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        garch_variance_forecasts(returns)


def test_rejects_nonfinite_returns() -> None:
    returns = make_returns(80)
    returns.iloc[10] = np.inf

    with pytest.raises(ValueError, match="finite values"):
        garch_variance_forecasts(returns)


def test_rejects_unsorted_dates() -> None:
    returns = make_returns(80).iloc[::-1]

    with pytest.raises(ValueError, match="sorted by index"):
        garch_variance_forecasts(returns)


def test_rejects_duplicate_dates() -> None:
    returns = make_returns(80)
    returns.index = list(returns.index[:-1]) + [returns.index[-2]]

    with pytest.raises(ValueError, match="duplicate index"):
        garch_variance_forecasts(returns)


@pytest.mark.parametrize("minimum", [0, 10, 29])
def test_rejects_too_small_training_window(minimum: int) -> None:
    with pytest.raises(ValueError, match="at least 30"):
        garch_variance_forecasts(
            make_returns(80),
            minimum_training_observations=minimum,
        )


def test_rejects_invalid_reestimation_frequency() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        garch_variance_forecasts(
            make_returns(80),
            minimum_training_observations=30,
            reestimate_every=0,
        )


def test_future_returns_do_not_change_earlier_forecasts() -> None:
    returns = make_returns(90)
    changed = returns.copy()
    changed.iloc[60:] = changed.iloc[60:] * 10

    original_forecasts = garch_variance_forecasts(
        returns,
        minimum_training_observations=30,
        reestimate_every=10,
    )
    changed_forecasts = garch_variance_forecasts(
        changed,
        minimum_training_observations=30,
        reestimate_every=10,
    )

    pd.testing.assert_series_equal(
        original_forecasts.iloc[:60],
        changed_forecasts.iloc[:60],
    )


def test_five_day_forecasts_preserve_index_and_name() -> None:
    returns = make_returns(80)
    forecasts = garch_five_day_variance_forecasts(
        returns,
        minimum_training_observations=30,
        reestimate_every=10,
    )

    assert forecasts.index.equals(returns.index)
    assert forecasts.name == "garch_variance_5d"


def test_five_day_forecasts_respect_training_window_and_horizon() -> None:
    returns = make_returns(80)
    minimum = 30
    forecasts = garch_five_day_variance_forecasts(
        returns,
        minimum_training_observations=minimum,
        reestimate_every=10,
    )

    assert forecasts.iloc[: minimum - 1].isna().all()
    assert pd.notna(forecasts.iloc[minimum - 1])
    assert forecasts.iloc[minimum - 1 : -5].notna().all()
    assert forecasts.iloc[-5:].isna().all()


def test_five_day_forecasts_are_finite_and_positive() -> None:
    forecasts = garch_five_day_variance_forecasts(
        make_returns(80),
        minimum_training_observations=30,
        reestimate_every=10,
    )
    available = forecasts.dropna()

    assert np.isfinite(available).all()
    assert (available > 0).all()


def test_future_returns_do_not_change_earlier_five_day_forecasts() -> None:
    returns = make_returns(90)
    changed = returns.copy()
    changed.iloc[60:] *= 10

    kwargs = {
        "minimum_training_observations": 30,
        "reestimate_every": 10,
    }
    original = garch_five_day_variance_forecasts(returns, **kwargs)
    modified = garch_five_day_variance_forecasts(changed, **kwargs)

    pd.testing.assert_series_equal(
        original.iloc[:60],
        modified.iloc[:60],
    )

def test_prospective_forecast_includes_latest_date():
    returns = make_returns(80)

    forecasts = garch_variance_forecasts(
        returns,
        minimum_training_observations=30,
        reestimate_every=10,
    )

    assert pd.isna(forecasts.iloc[-1])

def test_prospective_forecast_uses_latest_available_returns():
    returns = make_returns(80)

    forecast = garch_prospective_variance_forecast(
        returns,
        minimum_training_observations=30,
    )

    assert np.isfinite(forecast)
    assert forecast > 0

def test_prospective_five_day_forecast_uses_latest_returns() -> None:
    returns = make_returns(80)

    forecast = garch_five_day_prospective_variance_forecast(
        returns,
        minimum_training_observations=30,
    )

    assert np.isfinite(forecast)
    assert forecast > 0