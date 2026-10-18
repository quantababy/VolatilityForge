import numpy as np
import pandas as pd
import pytest

from src.models.gjr_garch import (
    gjr_garch_five_day_prospective_variance_forecast,
    gjr_garch_five_day_variance_forecasts,
    gjr_garch_prospective_variance_forecast,
    gjr_garch_variance_forecasts,
)


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


def test_five_day_forecasts_preserve_index_and_name() -> None:
    returns = make_returns(80)
    forecasts = gjr_garch_five_day_variance_forecasts(
        returns,
        minimum_training_observations=30,
        reestimate_every=10,
    )

    assert forecasts.index.equals(returns.index)
    assert forecasts.name == "gjr_garch_variance_5d"


def test_five_day_forecasts_respect_training_window_and_horizon() -> None:
    returns = make_returns(80)
    minimum = 30

    forecasts = gjr_garch_five_day_variance_forecasts(
        returns,
        minimum_training_observations=minimum,
        reestimate_every=10,
    )

    assert forecasts.iloc[: minimum - 1].isna().all()
    assert pd.notna(forecasts.iloc[minimum - 1])
    assert forecasts.iloc[minimum - 1 : -5].notna().all()
    assert forecasts.iloc[-5:].isna().all()


def test_five_day_forecasts_are_finite_and_positive() -> None:
    forecasts = gjr_garch_five_day_variance_forecasts(
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

    original = gjr_garch_five_day_variance_forecasts(returns, **kwargs)
    modified = gjr_garch_five_day_variance_forecasts(changed, **kwargs)

    pd.testing.assert_series_equal(
        original.iloc[:60],
        modified.iloc[:60],
    )

def test_prospective_forecast_uses_latest_available_returns():
    returns = make_returns(80)

    forecast = gjr_garch_prospective_variance_forecast(
        returns,
        minimum_training_observations=30,
    )

    assert np.isfinite(forecast)
    assert forecast > 0

def test_prospective_five_day_forecast_uses_latest_returns() -> None:
    returns = make_returns(80)

    forecast = gjr_garch_five_day_prospective_variance_forecast(
        returns,
        minimum_training_observations=30,
    )

    assert np.isfinite(forecast)
    assert forecast > 0


def test_five_day_forecast_sums_horizon_variances_and_scales(
    monkeypatch,
) -> None:
    import src.models.gjr_garch as gjr_module

    class FakeForecast:
        variance = pd.DataFrame(
            [[100.0, 200.0, 300.0, 400.0, 500.0]]
        )

    class FakeResult:
        convergence_flag = 0

        def forecast(self, horizon, reindex):
            assert horizon == 5
            assert reindex is False
            return FakeForecast()

    class FakeModel:
        def fit(self, disp):
            assert disp == "off"
            return FakeResult()

    monkeypatch.setattr(
        gjr_module,
        "arch_model",
        lambda *args, **kwargs: FakeModel(),
    )

    returns = make_returns(35)
    forecasts = gjr_module.gjr_garch_five_day_variance_forecasts(
        returns,
        minimum_training_observations=30,
        reestimate_every=10,
    )

    # (100 + 200 + 300 + 400 + 500) / 10000 = 0.15
    assert forecasts.iloc[29] == pytest.approx(0.15)
