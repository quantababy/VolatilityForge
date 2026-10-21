
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import build_volatility_features


def make_data(n=35):
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 * np.exp(np.arange(n) * 0.01)
    open_price = np.concatenate(([close[0]], close[:-1]))

    data = pd.DataFrame(
        {
            "Open": open_price,
            "High": np.maximum(open_price, close) * 1.01,
            "Low": np.minimum(open_price, close) * 0.99,
            "Close": close,
            "VIX_Close": np.full(n, 20.0),
        },
        index=dates,
    )
    return data


def test_builds_log_returns_and_squared_returns():
    data = make_data()
    result = build_volatility_features(data)

    expected_return = np.log(data["Close"] / data["Close"].shift(1))
    expected_return.index = pd.DatetimeIndex(expected_return.index).normalize()
    expected_return.index.name = "Date"

    pd.testing.assert_series_equal(
        result["log_return"],
        expected_return.rename("log_return"),
    )
    np.testing.assert_allclose(
        result["squared_return"].dropna(),
        expected_return.pow(2).dropna(),
    )


def test_builds_ohlc_features():
    result = build_volatility_features(make_data())

    expected_columns = [
        "overnight_gap",
        "parkinson_variance",
        "garman_klass_variance",
        "rogers_satchell_variance",
    ]

    for column in expected_columns:
        assert column in result.columns
        assert result[column].notna().any()

    assert result["overnight_gap"].iloc[0] != result["overnight_gap"].iloc[0]


def test_historical_variance_uses_only_past_and_current_returns():
    result = build_volatility_features(make_data())
    squared_returns = result["squared_return"]

    assert result["rv_5"].iloc[:5].isna().all()
    assert result["rv_22"].iloc[:22].isna().all()

    np.testing.assert_allclose(
        result["rv_5"].iloc[5],
        squared_returns.iloc[1:6].sum(),
    )
    np.testing.assert_allclose(
        result["rv_22"].iloc[22],
        squared_returns.iloc[1:23].sum(),
    )


def test_forward_targets_use_future_returns():
    result = build_volatility_features(make_data())
    squared_returns = result["squared_return"]

    np.testing.assert_allclose(
        result["target_rv_1"].iloc[10],
        squared_returns.iloc[11],
    )
    np.testing.assert_allclose(
        result["target_rv_5"].iloc[10],
        squared_returns.iloc[11:16].sum(),
    )


def test_forward_targets_are_missing_at_end():
    result = build_volatility_features(make_data())

    assert result["target_rv_1"].iloc[-1:].isna().all()
    assert result["target_rv_5"].iloc[-5:].isna().all()


def test_rejects_missing_required_columns():
    data = make_data().drop(columns=["VIX_Close"])

    with pytest.raises(ValueError, match="Missing required columns"):
        build_volatility_features(data)


def test_rejects_nonpositive_prices():
    data = make_data()
    data.loc[data.index[5], "Close"] = 0

    with pytest.raises(ValueError, match="positive"):
        build_volatility_features(data)


def test_rejects_duplicate_dates():
    data = make_data()
    data.index = pd.DatetimeIndex(["2024-01-02"] * len(data))

    with pytest.raises(ValueError, match="duplicate dates"):
        build_volatility_features(data)
