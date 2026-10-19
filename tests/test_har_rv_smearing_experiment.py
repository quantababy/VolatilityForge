
import numpy as np
import pandas as pd
import pytest

from src.experiment_har_rv_five_day_smearing import (
    build_experiment_forecasts,
    summarize,
)
from tests.test_har_rv import add_five_day_target, make_features


@pytest.fixture
def experiment_results() -> pd.DataFrame:
    features = add_five_day_target(make_features(500))
    return build_experiment_forecasts(
        features,
        development_start="2021-01-01",
        minimum_training_observations=60,
        reestimate_every=10,
    )


def test_experiment_forecasts_are_positive_and_share_dates(
    experiment_results: pd.DataFrame,
) -> None:
    baseline = experiment_results["baseline_forecast"]
    corrected = experiment_results["smearing_forecast"]
    factor = experiment_results["smearing_factor"]

    assert baseline.notna().any()
    assert baseline.notna().equals(corrected.notna())
    assert baseline.notna().equals(factor.notna())
    assert (baseline.dropna() > 0).all()
    assert (corrected.dropna() > 0).all()
    assert np.isfinite(factor.dropna()).all()
    assert (factor.dropna() > 0).all()


def test_smearing_forecast_equals_baseline_times_factor(
    experiment_results: pd.DataFrame,
) -> None:
    valid = experiment_results["baseline_forecast"].notna()
    expected = (
        experiment_results.loc[valid, "baseline_forecast"]
        * experiment_results.loc[valid, "smearing_factor"]
    )

    np.testing.assert_allclose(
        experiment_results.loc[valid, "smearing_forecast"],
        expected,
        rtol=1e-12,
        atol=0,
    )


def test_summary_uses_identical_positive_target_sample(
    experiment_results: pd.DataFrame,
) -> None:
    summary = summarize(experiment_results)

    assert len(summary) == 2
    assert summary["observations"].nunique() == 1
    assert summary["observations"].iloc[0] > 0
    assert summary["first_date"].nunique() == 1
    assert summary["last_date"].nunique() == 1
    assert np.isfinite(summary["mean_qlike"]).all()
    assert np.isfinite(summary["mse"]).all()


def test_summary_rejects_empty_common_sample() -> None:
    empty = pd.DataFrame(
        {
            "target": [np.nan],
            "baseline_forecast": [np.nan],
            "smearing_forecast": [np.nan],
            "smearing_factor": [np.nan],
        }
    )

    with pytest.raises(ValueError, match="No common valid observations"):
        summarize(empty)
