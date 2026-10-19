
import numpy as np
import pandas as pd
import pytest

from src.strategy_backtest import run_volatility_target_backtest


def make_features() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=5)
    return pd.DataFrame(
        {"Close": [100.0, 102.0, 101.0, 104.0, 103.0]},
        index=dates,
    )


def make_forecasts() -> pd.Series:
    dates = make_features().index[:4]
    return pd.Series(
        [0.0004, 0.0001, 0.0004, 0.0001],
        index=dates,
        name="forecast_variance",
    )


def test_forecast_at_t_applies_to_next_session_return() -> None:
    features = make_features()
    forecasts = make_forecasts()

    result = run_volatility_target_backtest(features, forecasts)

    assert result.index.equals(forecasts.index)
    assert result["next_date"].iloc[0] == features.index[1]
    assert result["next_return"].iloc[0] == pytest.approx(0.02)
    assert result["next_return"].iloc[1] == pytest.approx(101 / 102 - 1)
    assert result["next_date"].iloc[-1] == features.index[4]


def test_position_is_capped_and_nonnegative() -> None:
    result = run_volatility_target_backtest(
        make_features(),
        make_forecasts(),
        target_annual_volatility=0.10,
        max_leverage=1.0,
    )

    assert (result["position"] >= 0).all()
    assert (result["position"] <= 1.0).all()


def test_transaction_cost_uses_turnover() -> None:
    result = run_volatility_target_backtest(
        make_features(),
        make_forecasts(),
        transaction_cost_bps=5.0,
    )

    expected_cost = result["turnover"] * 5.0 / 10_000.0
    pd.testing.assert_series_equal(
        result["transaction_cost"],
        expected_cost.rename("transaction_cost"),
    )
    assert result["turnover"].iloc[0] > 0


def test_net_return_equals_gross_return_minus_cost() -> None:
    result = run_volatility_target_backtest(
        make_features(), make_forecasts()
    )

    expected = result["gross_return"] - result["transaction_cost"]
    pd.testing.assert_series_equal(
        result["net_return"],
        expected.rename("net_return"),
    )


def test_last_feature_date_is_not_traded_without_next_return() -> None:
    features = make_features()
    forecasts = pd.Series(
        0.0004,
        index=features.index,
        name="forecast_variance",
    )

    result = run_volatility_target_backtest(features, forecasts)

    assert features.index[-1] not in result.index
    assert len(result) == len(features) - 1


@pytest.mark.parametrize(
    ("forecast_values", "message"),
    [
        ([0.001, 0.0], "strictly positive"),
        ([0.001, -0.001], "strictly positive"),
        ([0.001, np.nan], "finite and nonmissing"),
        ([0.001, np.inf], "finite and nonmissing"),
    ],
)
def test_rejects_invalid_forecasts(forecast_values, message) -> None:
    dates = make_features().index[:2]
    forecasts = pd.Series(forecast_values, index=dates)

    with pytest.raises(ValueError, match=message):
        run_volatility_target_backtest(make_features(), forecasts)


def test_rejects_nonpositive_close_prices() -> None:
    features = make_features()
    features.loc[features.index[1], "Close"] = 0.0

    with pytest.raises(ValueError, match="Close prices must be positive"):
        run_volatility_target_backtest(features, make_forecasts())


def test_rejects_forecast_dates_missing_from_features() -> None:
    forecasts = pd.Series(
        [0.001],
        index=pd.DatetimeIndex(["2030-01-01"]),
    )

    with pytest.raises(ValueError, match="Every forecast date"):
        run_volatility_target_backtest(make_features(), forecasts)

def test_allows_initial_missing_forecasts() -> None:
    features = make_features()
    forecasts = pd.Series(
        [np.nan, np.nan, 0.0004, 0.0001],
        index=features.index[:4],
    )

    result = run_volatility_target_backtest(
        features,
        forecasts.dropna(),
    )

    assert result.index.equals(features.index[2:4])
    assert result["next_date"].iloc[0] == features.index[3]
    assert result["next_date"].iloc[-1] == features.index[4]


def test_rejects_forecast_dates_with_internal_gaps() -> None:
    features = make_features()
    forecasts = pd.Series(
        [0.0004, 0.0001],
        index=features.index[[0, 2]],
    )

    with pytest.raises(ValueError, match="consecutive"):
        run_volatility_target_backtest(features, forecasts)

def test_default_initial_exposure_charges_full_entry_turnover() -> None:
    result = run_volatility_target_backtest(
        make_features(),
        make_forecasts(),
        max_leverage=1.0,
    )

    assert result["turnover"].iloc[0] == pytest.approx(
        result["position"].iloc[0]
    )


def test_initial_exposure_changes_first_turnover_only() -> None:
    features = make_features()
    forecasts = make_forecasts()

    zero_start = run_volatility_target_backtest(
        features, forecasts, initial_exposure=0.0
    )
    invested_start = run_volatility_target_backtest(
        features, forecasts, initial_exposure=0.5
    )

    assert invested_start["turnover"].iloc[0] == pytest.approx(
        abs(invested_start["position"].iloc[0] - 0.5)
    )
    pd.testing.assert_series_equal(
        zero_start["turnover"].iloc[1:],
        invested_start["turnover"].iloc[1:],
        check_names=False,
    )


@pytest.mark.parametrize("initial_exposure", [-0.1, 1.1, np.nan, np.inf])
def test_rejects_invalid_initial_exposure(initial_exposure) -> None:
    with pytest.raises(ValueError, match="initial_exposure"):
        run_volatility_target_backtest(
            make_features(),
            make_forecasts(),
            max_leverage=1.0,
            initial_exposure=initial_exposure,
        )