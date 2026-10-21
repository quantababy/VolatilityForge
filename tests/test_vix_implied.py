import numpy as np
import pandas as pd
import pytest

from src.models.vix_implied import vix_implied_variance_forecasts


def make_features() -> pd.DataFrame:
    return pd.DataFrame(
        {"VIX_Close": [15.0, 20.0, 30.0]},
        index=pd.bdate_range("2024-01-01", periods=3, name="Date"),
    )


def test_converts_vix_to_daily_and_five_day_variance() -> None:
    forecasts = vix_implied_variance_forecasts(make_features())

    expected_daily = (make_features()["VIX_Close"] / 100.0) ** 2 / 252

    pd.testing.assert_series_equal(
        forecasts["vix_implied_variance_1d"],
        expected_daily.rename("vix_implied_variance_1d"),
    )
    pd.testing.assert_series_equal(
        forecasts["vix_implied_variance_5d"],
        (5 * expected_daily).rename("vix_implied_variance_5d"),
    )


def test_preserves_dates_and_produces_positive_finite_forecasts() -> None:
    features = make_features()
    forecasts = vix_implied_variance_forecasts(features)

    pd.testing.assert_index_equal(forecasts.index, features.index)
    assert np.isfinite(forecasts.to_numpy()).all()
    assert (forecasts > 0).to_numpy().all()


def test_rejects_missing_vix() -> None:
    features = pd.DataFrame(
        index=make_features().index,
    )

    with pytest.raises(ValueError, match="VIX_Close"):
        vix_implied_variance_forecasts(features)


def test_rejects_missing_or_nonfinite_vix_values() -> None:
    features = make_features()
    features.iloc[1, 0] = np.nan

    with pytest.raises(ValueError, match="finite values"):
        vix_implied_variance_forecasts(features)


def test_rejects_nonpositive_vix() -> None:
    features = make_features()
    features.iloc[0, 0] = 0.0

    with pytest.raises(ValueError, match="positive"):
        vix_implied_variance_forecasts(features)


def test_rejects_duplicate_dates() -> None:
    features = make_features()
    features.index = [
        features.index[0],
        features.index[1],
        features.index[1],
    ]

    with pytest.raises(ValueError, match="unique"):
        vix_implied_variance_forecasts(features)


def test_rejects_invalid_annualization() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        vix_implied_variance_forecasts(make_features(), 0)
