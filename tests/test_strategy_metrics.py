import numpy as np
import pandas as pd
import pytest

from src.strategy_metrics import calculate_strategy_metrics


def test_calculates_compounded_and_annualized_returns() -> None:
    returns = pd.Series([0.10, -0.05])

    result = calculate_strategy_metrics(
        returns,
        trading_days_per_year=2,
    )

    assert result["n_observations"] == 2
    assert result["total_return"] == pytest.approx(1.10 * 0.95 - 1.0)
    assert result["annualized_return"] == pytest.approx(
        (1.10 * 0.95) ** (2 / 2) - 1.0
    )


def test_calculates_annualized_volatility_and_zero_rf_sharpe() -> None:
    returns = pd.Series([0.01, 0.03, 0.02])

    result = calculate_strategy_metrics(
        returns,
        trading_days_per_year=252,
    )

    expected_std = np.std(returns.to_numpy(), ddof=1)

    assert result["annualized_volatility"] == pytest.approx(
        expected_std * np.sqrt(252)
    )
    assert result["sharpe_ratio_zero_rf"] == pytest.approx(
        returns.mean() / expected_std * np.sqrt(252)
    )


def test_calculates_maximum_drawdown() -> None:
    returns = pd.Series([0.10, -0.20, 0.05])

    result = calculate_strategy_metrics(returns)

    expected_wealth = np.cumprod(1.0 + returns.to_numpy())
    running_peak = np.maximum.accumulate(
        np.concatenate(([1.0], expected_wealth))
    )[1:]
    expected_drawdown = np.min(expected_wealth / running_peak - 1.0)

    assert result["maximum_drawdown"] == pytest.approx(expected_drawdown)


def test_constant_returns_have_zero_volatility_and_nan_sharpe() -> None:
    result = calculate_strategy_metrics(pd.Series([0.01, 0.01, 0.01]))

    assert result["annualized_volatility"] == pytest.approx(0.0)
    assert np.isnan(result["sharpe_ratio_zero_rf"])


def test_rejects_empty_returns() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calculate_strategy_metrics(pd.Series(dtype=float))


@pytest.mark.parametrize(
    "values",
    [
        [0.01, np.nan],
        [0.01, np.inf],
        [0.01, -np.inf],
    ],
)
def test_rejects_missing_or_nonfinite_returns(values) -> None:
    with pytest.raises(ValueError, match="missing values|finite"):
        calculate_strategy_metrics(pd.Series(values))


def test_rejects_returns_below_minus_one() -> None:
    with pytest.raises(ValueError, match="less than -100%"):
        calculate_strategy_metrics(pd.Series([0.01, -1.01]))


@pytest.mark.parametrize("days", [0, -1, 2.5, True])
def test_rejects_invalid_trading_days(days) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        calculate_strategy_metrics(
            pd.Series([0.01, 0.02]),
            trading_days_per_year=days,
        )