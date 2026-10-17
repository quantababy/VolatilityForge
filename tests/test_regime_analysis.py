
import numpy as np
import pandas as pd
import pytest

from src.regime_analysis import (
    assign_volatility_regimes,
    evaluate_forecasts_by_regime,
)


def make_features(values: list[float]) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.DataFrame({"rv_22": values}, index=index)


def test_assigns_regimes_using_only_prior_observations() -> None:
    features = make_features([1.0] * 126 + [2.0, 0.5])

    regimes = assign_volatility_regimes(
        features,
        lookback=126,
        minimum_history=126,
    )

    assert regimes.iloc[:126].isna().all()
    assert regimes.iloc[126] == "high"
    assert regimes.iloc[127] == "low"


def test_new_observation_does_not_change_its_own_threshold() -> None:
    features = make_features([1.0] * 126 + [100.0])

    first = assign_volatility_regimes(
        features,
        lookback=126,
        minimum_history=126,
    )

    changed = features.copy()
    changed.iloc[-1, 0] = 0.1
    second = assign_volatility_regimes(
        changed,
        lookback=126,
        minimum_history=126,
    )

    assert first.iloc[-1] == "high"
    assert second.iloc[-1] == "low"


def test_rejects_missing_volatility_column() -> None:
    with pytest.raises(ValueError, match="Missing volatility column"):
        assign_volatility_regimes(pd.DataFrame({"other": [1.0]}))


def test_rejects_invalid_lookback() -> None:
    features = make_features([1.0, 2.0])

    with pytest.raises(ValueError, match="lookback must be positive"):
        assign_volatility_regimes(features, lookback=0)


def test_rejects_invalid_minimum_history() -> None:
    features = make_features([1.0, 2.0])

    with pytest.raises(ValueError, match="minimum_history"):
        assign_volatility_regimes(
            features,
            lookback=2,
            minimum_history=3,
        )


def test_evaluates_forecasts_by_regime() -> None:
    index = pd.bdate_range("2024-01-01", periods=4)
    forecasts = pd.Series([1.0, 2.0, 1.0, 4.0], index=index)
    targets = pd.Series([1.0, 1.0, 2.0, 2.0], index=index)
    regimes = pd.Series(["low", "low", "high", "high"], index=index)

    result = evaluate_forecasts_by_regime(forecasts, targets, regimes)

    assert result["regime"].tolist() == ["low", "high"]
    assert result["n_observations"].tolist() == [2, 2]
    assert result.loc[0, "mse"] == pytest.approx(0.5)
    assert result.loc[1, "mse"] == pytest.approx(2.5)


def test_skips_regimes_without_valid_forecasts() -> None:
    index = pd.bdate_range("2024-01-01", periods=2)
    forecasts = pd.Series([1.0, np.nan], index=index)
    targets = pd.Series([1.0, 2.0], index=index)
    regimes = pd.Series(["low", "high"], index=index)

    result = evaluate_forecasts_by_regime(forecasts, targets, regimes)

    assert result["regime"].tolist() == ["low"]


def test_rejects_mismatched_indexes() -> None:
    index = pd.bdate_range("2024-01-01", periods=2)

    with pytest.raises(ValueError, match="identical indexes"):
        evaluate_forecasts_by_regime(
            pd.Series([1.0, 2.0], index=index),
            pd.Series([1.0, 2.0], index=index[::-1]),
            pd.Series(["low", "high"], index=index),
        )


def test_mse_includes_zero_targets_but_qlike_excludes_them() -> None:
    index = pd.bdate_range("2024-01-01", periods=3)
    forecasts = pd.Series([1.0, 2.0, 3.0], index=index)
    targets = pd.Series([0.0, 1.0, 2.0], index=index)
    regimes = pd.Series(["low", "low", "low"], index=index)

    result = evaluate_forecasts_by_regime(
        forecasts,
        targets,
        regimes,
    )

    low = result.loc[result["regime"] == "low"].iloc[0]

    assert low["n_mse_observations"] == 3
    assert low["n_qlike_observations"] == 2
    assert low["n_observations"] == 3
    assert low["mse"] == pytest.approx(
        ((1.0 - 0.0) ** 2 + (2.0 - 1.0) ** 2 + (3.0 - 2.0) ** 2) / 3
    )
    assert np.isfinite(low["qlike"])
