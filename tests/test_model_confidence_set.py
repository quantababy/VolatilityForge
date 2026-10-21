import numpy as np
import pandas as pd
import pytest

from src.model_confidence_set import model_confidence_set


def test_retains_similar_models_and_returns_expected_columns():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=250, name="Date")
    common = rng.uniform(0.01, 1.0, size=250)
    losses = pd.DataFrame(
        {
            "Model A": common,
            "Model B": common + rng.normal(0, 0.002, size=250),
            "Poor model": common + 0.5,
        },
        index=dates,
    )

    result = model_confidence_set(
        losses, bootstrap_samples=300, block_length=5, random_seed=42
    )

    assert list(result.columns) == [
        "model", "mean_loss", "elimination_p_value", "in_confidence_set"
    ]
    assert "Poor model" not in result.loc[
        result["in_confidence_set"], "model"
    ].tolist()
    assert result["in_confidence_set"].any()


def test_rejects_nonfinite_losses():
    losses = pd.DataFrame({"A": [0.1, np.nan], "B": [0.2, 0.3]})
    with pytest.raises(ValueError, match="finite"):
        model_confidence_set(losses)


def test_rejects_duplicate_dates():
    dates = pd.to_datetime(["2024-01-01", "2024-01-01"])
    losses = pd.DataFrame({"A": [0.1, 0.2], "B": [0.2, 0.3]}, index=dates)
    with pytest.raises(ValueError, match="unique"):
        model_confidence_set(losses)


def test_rejects_invalid_block_length():
    losses = pd.DataFrame({"A": [0.1, 0.2], "B": [0.2, 0.3]})
    with pytest.raises(ValueError, match="block_length"):
        model_confidence_set(losses, block_length=3)


def test_rejects_negative_losses():
    losses = pd.DataFrame({"A": [-0.1, 0.2], "B": [0.2, 0.3]})
    with pytest.raises(ValueError, match="non-negative"):
        model_confidence_set(losses)
