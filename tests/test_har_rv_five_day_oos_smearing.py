import numpy as np
import pandas as pd
import pytest

from src.experiment_har_rv_five_day_oos_smearing import (
    eligible_oos_residual_positions,
    estimate_oos_smearing_factor,
)


@pytest.mark.parametrize(
    ("forecast_position", "expected"),
    [
        (0, []),
        (5, []),
        (6, [0]),
        (7, [0, 1]),
        (10, [0, 1, 2, 3, 4]),
    ],
)

def test_eligible_residual_positions_enforce_maturity_cutoff(
    forecast_position: int,
    expected: list[int],
) -> None:
    positions = eligible_oos_residual_positions(
        forecast_position,
        total_positions=20,
    )

    assert positions.tolist() == expected

def test_smearing_factor_uses_exponentiated_residual_mean() -> None:
    residuals = pd.Series([0.0, np.log(2.0), np.log(4.0)])

    factor = estimate_oos_smearing_factor(
        residuals,
        minimum_residuals=3,
    )

    assert factor == pytest.approx((1.0 + 2.0 + 4.0) / 3.0)


def test_smearing_factor_ignores_missing_and_nonfinite_residuals() -> None:
    residuals = pd.Series([0.0, np.log(2.0), np.nan, np.inf])

    factor = estimate_oos_smearing_factor(
        residuals,
        minimum_residuals=2,
    )

    assert factor == pytest.approx(1.5)


def test_smearing_factor_requires_minimum_valid_residuals() -> None:
    residuals = pd.Series([0.0, np.nan, np.inf])

    with pytest.raises(ValueError, match="Insufficient eligible"):
        estimate_oos_smearing_factor(
            residuals,
            minimum_residuals=2,
        )


@pytest.mark.parametrize("minimum", [0, -1, True, 1.5])
def test_smearing_factor_rejects_invalid_minimum(minimum: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        estimate_oos_smearing_factor(
            pd.Series([0.0]),
            minimum_residuals=minimum,
        )


def test_smearing_factor_rejects_overflowing_residuals() -> None:
    residuals = pd.Series([1000.0])

    with pytest.raises(ValueError, match="non-finite"):
        estimate_oos_smearing_factor(
            residuals,
            minimum_residuals=1,
        )

from src.experiment_har_rv_five_day_oos_smearing import (
    build_oos_smearing_forecasts,
)
from tests.test_har_rv import add_five_day_target, make_features


def test_oos_experiment_preserves_forecast_index_and_maturity_boundary() -> None:
    features = add_five_day_target(make_features(600))
    results = build_oos_smearing_forecasts(
        features,
        development_start="2020-01-01",
        minimum_training_observations=60,
        reestimate_every=10,
        minimum_residuals=20,
    )

    # The experiment preserves the requested date range and original index.
    expected_index = features.loc["2020-01-01":].index.rename("Date")
    pd.testing.assert_index_equal(results.index, expected_index)

    # The final five origins cannot have fully observed five-day targets.
    assert results["target"].iloc[-5:].isna().all()
    assert results["baseline_forecast"].iloc[-5:].isna().all()

    # A forecast must never be recorded where predictors are unavailable.
    assert results["baseline_forecast"].notna().equals(
        results["baseline_forecast"].notna()
        & results["baseline_forecast"].gt(0)
        & np.isfinite(results["baseline_forecast"])
    )



def test_oos_correction_requires_prior_matured_residuals() -> None:
    features = add_five_day_target(make_features(600))
    results = build_oos_smearing_forecasts(
        features,
        development_start="2020-01-01",
        minimum_training_observations=60,
        reestimate_every=10,
        minimum_residuals=20,
    )

    corrected = results["oos_smearing_forecast"]
    factors = results["oos_smearing_factor"]

    assert corrected.notna().equals(factors.notna())
    assert corrected.notna().any()
    assert (corrected.dropna() > 0).all()
    assert np.isfinite(factors.dropna()).all()
    assert (factors.dropna() > 0).all()


def test_oos_corrected_forecast_is_baseline_times_factor() -> None:
    features = add_five_day_target(make_features(600))
    results = build_oos_smearing_forecasts(
        features,
        development_start="2020-01-01",
        minimum_training_observations=60,
        reestimate_every=10,
        minimum_residuals=20,
    )

    valid = results["oos_smearing_forecast"].notna()
    np.testing.assert_allclose(
        results.loc[valid, "oos_smearing_forecast"],
        results.loc[valid, "baseline_forecast"]
        * results.loc[valid, "oos_smearing_factor"],
        rtol=1e-12,
        atol=0,
    )