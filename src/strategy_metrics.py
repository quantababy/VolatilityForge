"""Performance metrics for volatility-targeting strategy backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_strategy_metrics(
    returns: pd.Series,
    trading_days_per_year: int = 252,
) -> dict[str, float | int]:
    """Calculate annualized performance and risk metrics from periodic returns.

    Sharpe ratio assumes a zero risk-free rate. Returns are simple periodic
    returns, and annualized return is based on compounded wealth.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")
    if returns.empty:
        raise ValueError("returns must not be empty.")
    if returns.isna().any():
        raise ValueError("returns must not contain missing values.")

    values = returns.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("returns must be finite.")
    if (values < -1.0).any():
        raise ValueError("returns cannot be less than -100%.")

    if (
        not isinstance(trading_days_per_year, (int, np.integer))
        or isinstance(trading_days_per_year, (bool, np.bool_))
        or trading_days_per_year < 1
    ):
        raise ValueError("trading_days_per_year must be a positive integer.")

    wealth = np.cumprod(1.0 + values)
    total_return = float(wealth[-1] - 1.0)
    annualized_return = float(
        wealth[-1] ** (trading_days_per_year / len(values)) - 1.0
    ) if wealth[-1] > 0 else -1.0

    daily_volatility = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    annualized_volatility = daily_volatility * np.sqrt(trading_days_per_year)

    if daily_volatility > 0:
        sharpe_ratio = float(
            np.mean(values) / daily_volatility
            * np.sqrt(trading_days_per_year)
        )
    else:
        sharpe_ratio = float("nan")

    running_peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    drawdowns = wealth / running_peak - 1.0
    maximum_drawdown = float(min(0.0, np.min(drawdowns)))

    return {
        "n_observations": len(values),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio_zero_rf": sharpe_ratio,
        "maximum_drawdown": maximum_drawdown,
    }
