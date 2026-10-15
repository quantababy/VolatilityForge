"""Leakage-aware volatility feature and target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_volatility_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build historical features and forward variance targets.

    Features at date t use data available through date t.
    Targets measure variance over the subsequent trading sessions.
    """
    required_columns = {"Close", "VIX_Close"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Input market data must not be empty.")

    result = data.copy()
    result.index = pd.DatetimeIndex(result.index).normalize()
    result.index.name = "Date"

    if result.index.has_duplicates:
        raise ValueError("Input market data contains duplicate dates.")

    if not result.index.is_monotonic_increasing:
        raise ValueError("Input market data must be sorted by date.")

    if result[["Close", "VIX_Close"]].isna().any().any():
        raise ValueError("Close and VIX_Close must not contain missing values.")

    if (result["Close"] <= 0).any():
        raise ValueError("SPY closing prices must be positive.")

    if (result["VIX_Close"] <= 0).any():
        raise ValueError("VIX closing values must be positive.")

    log_return = np.log(result["Close"] / result["Close"].shift(1))
    squared_return = log_return.pow(2)

    result["log_return"] = log_return
    result["squared_return"] = squared_return

    # Historical variance proxies, including the return on date t.
    result["rv_5"] = squared_return.rolling(
        window=5, min_periods=5
    ).sum()
    result["rv_22"] = squared_return.rolling(
        window=22, min_periods=22
    ).sum()

    # Forward targets: future returns are labels, never predictors.
    result["target_rv_1"] = squared_return.shift(-1)
    result["target_rv_5"] = squared_return.rolling(
        window=5, min_periods=5
    ).sum().shift(-5)

    return result
