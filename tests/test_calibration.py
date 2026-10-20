import numpy as np
import pandas as pd
import pytest

from src.calibration import mincer_zarnowitz


def test_calibration_recovers_linear_relationship():
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    forecast = pd.Series(np.linspace(0.01, 0.10, 80), index=dates)
    target = 0.002 + 1.2 * forecast

    result = mincer_zarnowitz(target, forecast, hac_lags=4)

    assert result["n_observations"] == 80
    assert result["intercept"] == pytest.approx(0.002)
    assert result["slope"] == pytest.approx(1.2)
    assert result["r_squared"] == pytest.approx(1.0)
    assert result["hac_lags"] == 4


def test_calibration_rejects_misaligned_indexes():
    target = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.date_range("2024-01-01", periods=3),
    )
    forecast = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.date_range("2024-01-02", periods=3),
    )

    with pytest.raises(ValueError, match="indexes must match"):
        mincer_zarnowitz(target, forecast, hac_lags=0)


@pytest.mark.parametrize("bad_forecast", [0.0, -0.1])
def test_calibration_rejects_nonpositive_forecasts(bad_forecast):
    dates = pd.date_range("2024-01-01", periods=10)
    target = pd.Series(np.linspace(0.01, 0.10, 10), index=dates)
    forecast = pd.Series(np.linspace(0.02, 0.11, 10), index=dates)
    forecast.iloc[0] = bad_forecast

    with pytest.raises(ValueError, match="strictly positive"):
        mincer_zarnowitz(target, forecast, hac_lags=2)


def test_calibration_rejects_nonfinite_values():
    dates = pd.date_range("2024-01-01", periods=10)
    target = pd.Series(np.linspace(0.01, 0.10, 10), index=dates)
    forecast = pd.Series(np.linspace(0.02, 0.11, 10), index=dates)
    target.iloc[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        mincer_zarnowitz(target, forecast, hac_lags=2)


def test_calibration_rejects_insufficient_observations():
    dates = pd.date_range("2024-01-01", periods=5)
    target = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=dates)
    forecast = pd.Series([0.02, 0.03, 0.04, 0.05, 0.06], index=dates)

    with pytest.raises(ValueError, match="Insufficient observations"):
        mincer_zarnowitz(target, forecast, hac_lags=3)
