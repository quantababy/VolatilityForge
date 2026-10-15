import numpy as np
import pandas as pd
import pytest

from src.evaluation import evaluate_variance_forecasts


def test_calculates_normalized_qlike_and_mse() -> None:
    index = pd.RangeIndex(2)
    forecasts = pd.Series([1.0, 2.0], index=index)
    targets = pd.Series([1.0, 1.0], index=index)

    result = evaluate_variance_forecasts(forecasts, targets)

    expected_qlike = (0.0 + 0.5 - np.log(0.5) - 1.0) / 2
    assert result["n_observations"] == 2
    assert result["qlike"] == pytest.approx(expected_qlike)
    assert result["mse"] == pytest.approx(0.5)


def test_drops_missing_target_pairs() -> None:
    index = pd.RangeIndex(3)
    forecasts = pd.Series([1.0, 2.0, 3.0], index=index)
    targets = pd.Series([1.0, np.nan, 2.0], index=index)

    result = evaluate_variance_forecasts(forecasts, targets)

    assert result["n_observations"] == 2


def test_rejects_mismatched_indexes() -> None:
    with pytest.raises(ValueError, match="identical indexes"):
        evaluate_variance_forecasts(
            pd.Series([1.0], index=[0]),
            pd.Series([1.0], index=[1]),
        )


def test_rejects_nonpositive_forecasts() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        evaluate_variance_forecasts(
            pd.Series([0.0]),
            pd.Series([1.0]),
        )


@pytest.mark.parametrize("target", [0.0, -1.0])
def test_rejects_nonpositive_targets(target: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        evaluate_variance_forecasts(
            pd.Series([1.0]),
            pd.Series([target]),
        )


def test_rejects_no_valid_pairs() -> None:
    with pytest.raises(ValueError, match="No valid"):
        evaluate_variance_forecasts(
            pd.Series([np.nan]),
            pd.Series([np.nan]),
        )
