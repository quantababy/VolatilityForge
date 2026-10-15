import numpy as np
import pandas as pd
import pytest

from src.models.gjr_garch import gjr_garch_variance_forecasts


def make_returns(n: int = 90) -> pd.Series:
    rng = np.random.default_rng(42)
    values = rng.normal(0, 0.01, n)
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(values, index=index, name="log_return")


def test_forecasts_preserve_index_and_name() -> None:
    returns = make_returns()
    forecasts = gjr_garch_variance_forecasts(
        returns, minimum_training_observations=40, reestimate_every=10
    )

    pd.testing.assert_index_equal(forecasts.index, returns.index)
    assert forecasts.name == "gjr_garch_variance_1d"


def test_forecasts_respect_initial_training_window() -> None:
    returns = make_returns()
    minimum = 40

    forecasts = gjr_garch_variance_forecasts(
        returns, minimum_training_observations=minimum, reestimate_every=10
    )

    assert forecasts.iloc[: minimum - 1].isna().all()
    assert np.isfinite(forecasts.iloc[minimum - 1])
    assert forecasts.iloc[minimum:-1].notna().all()
    assert pd.isna(forecasts.iloc[-1])


def test_available_forecasts_are_finite_and_positive() -> None:
    forecasts = gjr_garch_variance_forecasts(
        make_returns(),
        minimum_training_observations=40,
        reestimate_every=10,
    )
    available = forecasts.dropna()

    assert np.isfinite(available).all()
    assert (available > 0).all()


def test_rejects_empty_returns() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        gjr_garch_variance_forecasts(pd.Series(dtype=float))


def test_rejects_missing_returns() -> None:
    returns = make_returns()
    returns.iloc[10] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        gjr_garch_variance_forecasts(returns)


def test_rejects_non_finite_returns() -> None:
    returns = make_returns()
    returns.iloc[10] = np.inf

    with pytest.raises(ValueError, match="finite values"):
        gjr_garch_variance_forecasts(returns)


def test_rejects_unsorted_index() -> None:
    returns = make_returns().iloc[::-1]

    with pytest.raises(ValueError, match="sorted by index"):
        gjr_garch_variance_forecasts(returns)


def test_rejects_duplicate_index() -> None:
    returns = make_returns()
    returns.index = list(returns.index[:-1]) + [returns.index[-2]]

    with pytest.raises(ValueError, match="duplicate"):
        gjr_garch_variance_forecasts(returns)


@pytest.mark.parametrize("minimum", [0, 10, 29])
def test_rejects_too_small_training_window(minimum: int) -> None:
    with pytest.raises(ValueError, match="at least 30"):
        gjr_garch_variance_forecasts(
            make_returns(), minimum_training_observations=minimum
        )


def test_rejects_invalid_reestimation_frequency() -> None:
    with pytest.raises(ValueError, match="reestimate_every"):
        gjr_garch_variance_forecasts(
            make_returns(),
            minimum_training_observations=40,
            reestimate_every=0,
        )


def test_future_returns_do_not_change_earlier_forecasts() -> None:
    returns = make_returns(100)
    changed = returns.copy()
    changed.iloc[60:] = changed.iloc[60:] * 3

    original_forecasts = gjr_garch_variance_forecasts(
        returns, minimum_training_observations=40, reestimate_every=10
    )
    changed_forecasts = gjr_garch_variance_forecasts(
        changed, minimum_training_observations=40, reestimate_every=10
    )

    pd.testing.assert_series_equal(
        original_forecasts.iloc[:60],
        changed_forecasts.iloc[:60],
    )
