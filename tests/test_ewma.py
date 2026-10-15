import numpy as np
import pandas as pd
import pytest

from src.models.ewma import ewma_variance


def test_ewma_uses_previous_return_not_current_return():
    returns = pd.Series([0.01, 0.02, -0.03])
    result = ewma_variance(
        returns,
        decay=0.5,
        initial_variance=0.0001,
    )

    assert result.iloc[0] == pytest.approx(0.0001)
    assert result.iloc[1] == pytest.approx(
        0.5 * 0.0001 + 0.5 * 0.01**2
    )
    assert result.iloc[2] == pytest.approx(
        0.5 * result.iloc[1] + 0.5 * 0.02**2
    )


def test_ewma_preserves_index_and_output_name():
    returns = pd.Series(
        [0.01, -0.02, 0.005],
        index=pd.date_range("2024-01-01", periods=3),
    )

    result = ewma_variance(returns)

    assert result.index.equals(returns.index)
    assert result.name == "ewma_variance"
    assert (result > 0).all()


@pytest.mark.parametrize("decay", [0, 1, -0.1, 1.1])
def test_ewma_rejects_invalid_decay(decay):
    with pytest.raises(ValueError, match="decay"):
        ewma_variance(pd.Series([0.01, 0.02]), decay=decay)


def test_ewma_rejects_missing_returns():
    with pytest.raises(ValueError, match="missing"):
        ewma_variance(pd.Series([0.01, np.nan]))


def test_ewma_rejects_nonfinite_returns():
    with pytest.raises(ValueError, match="finite"):
        ewma_variance(pd.Series([0.01, np.inf]))


def test_ewma_rejects_invalid_initial_variance():
    with pytest.raises(ValueError, match="initial_variance"):
        ewma_variance(
            pd.Series([0.01, 0.02]),
            initial_variance=0,
        )


def test_ewma_rejects_empty_returns():
    with pytest.raises(ValueError, match="empty"):
        ewma_variance(pd.Series(dtype=float))
