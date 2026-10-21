
"""Leakage-aware volatility feature and target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_volatility_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build historical features and forward variance targets.

    Features at date t use market data available through the close of date t.
    Targets measure variance over the subsequent trading sessions.
    """
    required_columns = {"Open", "High", "Low", "Close", "VIX_Close"}
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

    price_columns = ["Open", "High", "Low", "Close", "VIX_Close"]

    if result[price_columns].isna().any().any():
        raise ValueError(
            "Open, High, Low, Close, and VIX_Close must not contain "
            "missing values."
        )

    if (result[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("SPY OHLC prices must be positive.")

    if (result["VIX_Close"] <= 0).any():
        raise ValueError("VIX closing values must be positive.")

    if (
        (result["High"] < result[["Open", "Low", "Close"]].max(axis=1))
        | (result["Low"] > result[["Open", "High", "Close"]].min(axis=1))
    ).any():
        raise ValueError("OHLC prices must satisfy High >= Open, Low, Close "
                         "and Low <= Open, High, Close.")

    # Close-to-close log returns and squared returns.
    log_return = np.log(result["Close"] / result["Close"].shift(1))
    squared_return = log_return.pow(2)

    result["log_return"] = log_return
    result["squared_return"] = squared_return

    # OHLC-derived features available at the close of date t.
    log_high_low = np.log(result["High"] / result["Low"])
    log_close_open = np.log(result["Close"] / result["Open"])

    # Overnight gap: today's open relative to the previous close.
    result["overnight_gap"] = np.log(
        result["Open"] / result["Close"].shift(1)
    )

    # Parkinson range-based variance estimator.
    result["parkinson_variance"] = (
        log_high_low.pow(2) / (4.0 * np.log(2.0))
    )

    # Garman-Klass OHLC variance estimator.
    result["garman_klass_variance"] = (
        0.5 * log_high_low.pow(2)
        - (2.0 * np.log(2.0) - 1.0) * log_close_open.pow(2)
    )

    # Rogers-Satchell OHLC variance estimator.
    result["rogers_satchell_variance"] = (
        np.log(result["High"] / result["Open"])
        * np.log(result["High"] / result["Close"])
        + np.log(result["Low"] / result["Open"])
        * np.log(result["Low"] / result["Close"])
    )

    # Historical variance proxies, including returns through date t.
    result["rv_5"] = squared_return.rolling(
        window=5, min_periods=5
    ).sum()

    result["rv_22"] = squared_return.rolling(
        window=22, min_periods=22
    ).sum()

    # Forward targets: future returns are labels, never predictors.
    result["target_rv_1"] = squared_return.shift(-1)

    # Cumulative variance over the next five trading sessions.
    result["target_rv_5"] = squared_return.rolling(
        window=5, min_periods=5
    ).sum().shift(-5)

    return result
