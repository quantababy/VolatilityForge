
"""Leakage-aware volatility-targeting strategy backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_volatility_target_backtest(
    features: pd.DataFrame,
    forecasts: pd.Series,
    target_annual_volatility: float = 0.10,
    max_leverage: float = 1.0,
    transaction_cost_bps: float = 5.0,
    trading_days_per_year: int = 252,
    initial_exposure: float = 0.0,
) -> pd.DataFrame:
    """Backtest a long-only volatility-targeting strategy.

    A forecast at date t determines exposure for the return from
    close t to close t+1. Transaction costs are charged on changes
    in exposure. Forecast dates must be consecutive trading sessions.
    The first turnover is measured relative to ``initial_exposure``.

    Remove unavailable warm-up forecasts before calling this function.
    Missing, infinite, or nonpositive forecast values are rejected.
    """

    if features.empty:
        raise ValueError("features must not be empty.")
    if forecasts.empty:
        raise ValueError("forecasts must not be empty.")
    if "Close" not in features.columns:
        raise ValueError("features must contain a Close column.")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features must have a DatetimeIndex.")
    if not isinstance(forecasts.index, pd.DatetimeIndex):
        raise TypeError("forecasts must have a DatetimeIndex.")
    if features.index.has_duplicates or forecasts.index.has_duplicates:
        raise ValueError("Date indexes must not contain duplicates.")
    if not features.index.is_monotonic_increasing:
        raise ValueError("features must be sorted by date.")
    if not forecasts.index.is_monotonic_increasing:
        raise ValueError("forecasts must be sorted by date.")

    if (
        not np.isfinite(target_annual_volatility)
        or target_annual_volatility <= 0
    ):
        raise ValueError(
            "target_annual_volatility must be finite and positive."
        )
    if not np.isfinite(max_leverage) or max_leverage <= 0:
        raise ValueError("max_leverage must be finite and positive.")
    if (
        not np.isfinite(initial_exposure)
        or initial_exposure < 0
        or initial_exposure > max_leverage
    ):
        raise ValueError(
            "initial_exposure must be finite and between 0 and max_leverage."
        )

    if (
        not np.isfinite(transaction_cost_bps)
        or transaction_cost_bps < 0
    ):
        raise ValueError(
            "transaction_cost_bps must be finite and nonnegative."
        )
    if (
        not isinstance(trading_days_per_year, (int, np.integer))
        or isinstance(trading_days_per_year, (bool, np.bool_))
        or trading_days_per_year < 1
    ):
        raise ValueError("trading_days_per_year must be a positive integer.")

    close = features["Close"].astype(float)

    if close.isna().any() or not np.isfinite(close.to_numpy()).all():
        raise ValueError("Close prices must be finite and nonmissing.")
    if (close <= 0).any():
        raise ValueError("Close prices must be positive.")

    variance = forecasts.astype(float)

    if variance.isna().any() or not np.isfinite(variance.to_numpy()).all():
        raise ValueError("Forecast variances must be finite and nonmissing.")
    if (variance <= 0).any():
        raise ValueError("Forecast variances must be strictly positive.")

    forecast_positions = features.index.get_indexer(variance.index)

    if (forecast_positions < 0).any():
        raise ValueError("Every forecast date must exist in features.")

    if len(forecast_positions) > 1 and not np.all(
        np.diff(forecast_positions) == 1
    ):
        raise ValueError(
            "Forecast dates must be consecutive trading sessions."
        )

    # A forecast made after close t is applied to the next session's return.
    aligned = pd.DataFrame(index=variance.index)
    aligned["forecast_variance"] = variance
    aligned["close"] = close.reindex(variance.index)
    aligned["next_close"] = close.shift(-1).reindex(variance.index)
    aligned["next_return"] = (
        aligned["next_close"] / aligned["close"] - 1.0
    )
    aligned["next_date"] = pd.Series(
        features.index.to_series().shift(-1).reindex(variance.index),
        index=variance.index,
    )

    # The final feature date has no observed next-session return yet.
    aligned = aligned.dropna(subset=["next_return", "next_date"]).copy()

    if aligned.empty:
        raise ValueError(
            "No forecast dates have an available next-session return."
        )

    daily_target_volatility = target_annual_volatility / np.sqrt(
        trading_days_per_year
    )
    forecast_daily_volatility = np.sqrt(aligned["forecast_variance"])

    aligned["position"] = (
        daily_target_volatility / forecast_daily_volatility
    ).clip(lower=0.0, upper=max_leverage)

    # Charge entry turnover relative to the configured starting exposure.
    previous_position = aligned["position"].shift(1).fillna(
        initial_exposure
    )

    aligned["turnover"] = (
        aligned["position"] - previous_position
    ).abs()
    aligned["transaction_cost"] = (
        aligned["turnover"] * transaction_cost_bps / 10_000.0
    )

    aligned["gross_return"] = (
        aligned["position"] * aligned["next_return"]
    )
    aligned["net_return"] = (
        aligned["gross_return"] - aligned["transaction_cost"]
    )

    aligned.index.name = "Date"

    return aligned[
        [
            "next_date",
            "forecast_variance",
            "position",
            "next_return",
            "turnover",
            "transaction_cost",
            "gross_return",
            "net_return",
        ]
    ]
