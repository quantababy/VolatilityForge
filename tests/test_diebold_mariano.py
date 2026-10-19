
import numpy as np
import pandas as pd
import pytest

from src.diebold_mariano import diebold_mariano_test


def test_equal_losses_have_zero_mean_differential() -> None:
    index = pd.RangeIndex(5)
    losses = pd.Series([1.0, 2.0, 1.5, 2.5, 3.0], index=index)

    with pytest.raises(ValueError, match="non-positive HAC variance"):
        diebold_mariano_test(losses, losses)



def test_lower_loss_for_model_a_gives_negative_statistic() -> None:
    index = pd.RangeIndex(10)
    loss_a = pd.Series(
        [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0],
        index=index,
    )
    loss_b = loss_a + pd.Series(
        [0.5, 1.0, 0.8, 1.2, 0.7, 1.1, 0.9, 1.3, 0.6, 1.0],
        index=index,
    )

    result = diebold_mariano_test(loss_a, loss_b)

    assert result["n_observations"] == 10
    assert result["mean_loss_differential"] < 0
    assert result["dm_statistic"] < 0
    assert result["p_value"] < 0.05

def test_rejects_internal_missing_pairs_with_hac_lags() -> None:
    index = pd.RangeIndex(5)
    loss_a = pd.Series([1.0, np.nan, 3.0, 4.0, 2.0], index=index)
    loss_b = pd.Series([2.0, 2.0, 3.5, 5.0, 4.0], index=index)

    with pytest.raises(ValueError, match="missing pairs"):
        diebold_mariano_test(loss_a, loss_b, hac_lags=1)


def test_rejects_mismatched_indexes() -> None:
    with pytest.raises(ValueError, match="identical indexes"):
        diebold_mariano_test(
            pd.Series([1.0, 2.0], index=[0, 1]),
            pd.Series([1.0, 2.0], index=[1, 2]),
        )


def test_rejects_too_few_pairs() -> None:
    with pytest.raises(ValueError, match="At least two"):
        diebold_mariano_test(
            pd.Series([1.0]),
            pd.Series([2.0]),
        )


@pytest.mark.parametrize("hac_lags", [-1, 1.5, True])
def test_rejects_invalid_hac_lags(hac_lags: int) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        diebold_mariano_test(
            pd.Series([1.0, 2.0, 3.0]),
            pd.Series([2.0, 3.0, 4.0]),
            hac_lags=hac_lags,
        )


def test_rejects_hac_lags_at_least_sample_size() -> None:
    with pytest.raises(ValueError, match="smaller than"):
        diebold_mariano_test(
            pd.Series([1.0, 2.0, 3.0]),
            pd.Series([2.0, 3.0, 4.0]),
            hac_lags=3,
        )


def test_rejects_nonfinite_losses() -> None:
    with pytest.raises(ValueError, match="finite"):
        diebold_mariano_test(
            pd.Series([1.0, np.inf, 3.0]),
            pd.Series([2.0, 3.0, 4.0]),
        )


def test_rejects_constant_nonzero_differential() -> None:
    with pytest.raises(ValueError, match="non-positive HAC variance"):
        diebold_mariano_test(
            pd.Series([1.0, 2.0, 3.0]),
            pd.Series([2.0, 3.0, 4.0]),
        )


def test_hln_correction_uses_forecast_horizon() -> None:
    index = pd.RangeIndex(20)
    loss_a = pd.Series(
        [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0,
         5.5, 7.0, 6.5, 8.0, 7.5, 9.0, 8.5, 10.0, 9.5, 11.0],
        index=index,
    )
    loss_b = loss_a + pd.Series(
        [0.5, 1.0, 0.8, 1.2, 0.7, 1.1, 0.9, 1.3, 0.6, 1.0,
         0.8, 1.2, 0.7, 1.1, 0.9, 1.3, 0.6, 1.0, 0.8, 1.2],
        index=index,
    )

    one_step = diebold_mariano_test(
        loss_a, loss_b, forecast_horizon=1
    )
    five_step = diebold_mariano_test(
        loss_a, loss_b, forecast_horizon=5, hac_lags=4
    )

    assert one_step["hln_correction_factor"] == pytest.approx(
    np.sqrt(19 / 20)
)
    assert five_step["hln_correction_factor"] < 1.0
    assert five_step["forecast_horizon"] == 5


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_rejects_invalid_forecast_horizon(horizon: int) -> None:
    with pytest.raises(ValueError, match="forecast_horizon"):
        diebold_mariano_test(
            pd.Series([1.0, 2.0, 3.0]),
            pd.Series([2.0, 3.0, 4.5]),
            forecast_horizon=horizon,
        )


def test_rejects_horizon_at_least_sample_size() -> None:
    with pytest.raises(ValueError, match="smaller than"):
        diebold_mariano_test(
            pd.Series([1.0, 2.0, 3.0]),
            pd.Series([2.0, 3.0, 4.5]),
            forecast_horizon=3,
        )

def test_raw_dm_statistic_matches_statsmodels_hac() -> None:
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hac

    differential = np.array([
        0.4, -0.2, 0.7, -0.1, 0.3, -0.5,
        0.8, -0.2, 0.1, 0.6, -0.4, 0.2,
    ])
    n = len(differential)

    result = diebold_mariano_test(
        pd.Series(differential),
        pd.Series(np.zeros(n)),
        hac_lags=3,
        forecast_horizon=1,
    )

    fitted = sm.OLS(
        differential,
        np.ones((n, 1)),
    ).fit()

    variance = cov_hac(
        fitted,
        nlags=3,
        use_correction=False,
    )[0, 0]

    expected_statistic = np.mean(differential) / np.sqrt(variance)

    assert result["raw_dm_statistic"] == pytest.approx(
        expected_statistic,
        rel=1e-12,
        abs=1e-12,
    )

def test_rejects_unsorted_loss_indexes() -> None:
    loss_a = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.DatetimeIndex(
            ["2024-01-03", "2024-01-01", "2024-01-02"]
        ),
    )
    loss_b = loss_a + 0.5

    with pytest.raises(ValueError, match="sorted chronologically"):
        diebold_mariano_test(loss_a, loss_b, hac_lags=1)


def test_rejects_duplicate_loss_indexes() -> None:
    index = pd.DatetimeIndex(
        ["2024-01-01", "2024-01-01", "2024-01-02"]
    )
    loss_a = pd.Series([1.0, 2.0, 3.0], index=index)
    loss_b = pd.Series([1.5, 2.5, 3.5], index=index)

    with pytest.raises(ValueError, match="duplicates"):
        diebold_mariano_test(loss_a, loss_b, hac_lags=1)