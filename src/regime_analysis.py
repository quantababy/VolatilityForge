
"""Leakage-aware volatility regime assignment and forecast evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import evaluate_variance_forecasts

REGIME_COLUMN = "volatility_regime"


def assign_volatility_regimes(
    features: pd.DataFrame,
    volatility_column: str = "rv_22",
    lookback: int = 252,
    minimum_history: int = 126,
) -> pd.Series:
    """Classify dates using trailing volatility versus a past-only median.

    The threshold at date t is calculated from the preceding observations,
    excluding date t itself. No future observations are used.
    """
    if volatility_column not in features.columns:
        raise ValueError(
            f"Missing volatility column: {volatility_column}"
        )

    if lookback < 1:
        raise ValueError("lookback must be positive.")

    if minimum_history < 1 or minimum_history > lookback:
        raise ValueError(
            "minimum_history must be between 1 and lookback."
        )

    volatility = features[volatility_column]

    if not isinstance(volatility.index, pd.DatetimeIndex):
        raise TypeError("Features must use a DatetimeIndex.")

    if volatility.index.has_duplicates:
        raise ValueError("Features contain duplicate dates.")

    if not volatility.index.is_monotonic_increasing:
        raise ValueError("Features must be sorted by date.")

    if (volatility.dropna() < 0).any():
        raise ValueError("Historical variance cannot be negative.")

    historical_median = volatility.shift(1).rolling(
        window=lookback,
        min_periods=minimum_history,
    ).median()

    regimes = pd.Series(
        index=features.index,
        dtype="object",
        name=REGIME_COLUMN,
    )

    valid = volatility.notna() & historical_median.notna()
    regimes.loc[valid & volatility.le(historical_median)] = "low"
    regimes.loc[valid & volatility.gt(historical_median)] = "high"

    return regimes



def evaluate_forecasts_by_regime(
    forecasts: pd.Series,
    targets: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Evaluate QLIKE and MSE separately within each volatility regime."""
    if not forecasts.index.equals(targets.index):
        raise ValueError("Forecasts and targets must have identical indexes.")

    if not forecasts.index.equals(regimes.index):
        raise ValueError("Forecasts and regimes must have identical indexes.")

    rows: list[dict[str, str | float | int]] = []

    for regime in ("low", "high"):
        mask = regimes.eq(regime)
        regime_forecasts = forecasts.loc[mask]
        regime_targets = targets.loc[mask]

        finite_pairs = (
            regime_forecasts.notna()
            & regime_targets.notna()
            & np.isfinite(regime_forecasts)
            & np.isfinite(regime_targets)
            & regime_forecasts.gt(0)
            & regime_targets.ge(0)
        )

        valid_forecasts = regime_forecasts.loc[finite_pairs]
        valid_targets = regime_targets.loc[finite_pairs]

        if valid_forecasts.empty:
            continue

        # MSE includes zero targets; forecasts must remain positive.
        mse = float(((valid_forecasts - valid_targets) ** 2).mean())
        mse_observations = len(valid_targets)

        # QLIKE requires strictly positive targets.
        positive_targets = valid_targets.gt(0)
        qlike_forecasts = valid_forecasts.loc[positive_targets]
        qlike_targets = valid_targets.loc[positive_targets]

        if qlike_targets.empty:
            qlike = float("nan")
            qlike_observations = 0
        else:
            metrics = evaluate_variance_forecasts(
                qlike_forecasts,
                qlike_targets,
            )
            qlike = float(metrics["qlike"])
            qlike_observations = len(qlike_targets)

        rows.append({
            "regime": regime,
            "n_observations": mse_observations,
            "n_qlike_observations": qlike_observations,
            "n_mse_observations": mse_observations,
            "qlike": qlike,
            "mse": mse,
        })

    if not rows:
        raise ValueError("No valid forecast-target pairs have assigned regimes.")

    return pd.DataFrame(rows)
