"""Generate Model Confidence Set reports for both forecast horizons."""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch.bootstrap import MCS

from src.compare_five_day_forecasts import (
    build_comparison_frame as build_five_day,
)
from src.compare_forecasts import build_comparison_frame as build_one_day


def build_mcs_report(
    losses: pd.DataFrame,
    *,
    horizon: int,
    alpha: float = 0.10,
    bootstrap_reps: int = 1000,
    block_size: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute Hansen-Lunde-Nason MCS p-values from aligned QLIKE losses."""
    if not isinstance(losses, pd.DataFrame) or losses.empty:
        raise ValueError("Losses must be a non-empty DataFrame.")
    if losses.shape[1] < 2:
        raise ValueError("At least two models are required.")
    if losses.columns.has_duplicates:
        raise ValueError("Model names must be unique.")
    if not losses.index.is_unique or not losses.index.is_monotonic_increasing:
        raise ValueError("Loss dates must be unique and sorted.")
    if losses.isna().any().any():
        raise ValueError("Losses must not contain missing values.")
    if not np.isfinite(losses.to_numpy(dtype=float)).all():
        raise ValueError("Losses must be finite.")
    if (losses < 0).any().any():
        raise ValueError("QLIKE losses must be non-negative.")
    if horizon not in (1, 5):
        raise ValueError("horizon must be 1 or 5.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if bootstrap_reps < 100:
        raise ValueError("bootstrap_reps must be at least 100.")
    if not 1 <= block_size <= len(losses):
        raise ValueError("block_size must be between 1 and the sample size.")

    mcs = MCS(
        losses,
        size=alpha,
        reps=bootstrap_reps,
        block_size=block_size,
        method="R",
        bootstrap="circular",
        seed=seed,
    )
    mcs.compute()

    pvalues = mcs.pvalues.iloc[:, 0].rename("mcs_p_value")
    included = set(mcs.included)
    report = pd.DataFrame({
        "model": losses.columns,
        "mean_qlike": losses.mean().reindex(losses.columns).to_numpy(),
    }).set_index("model")
    report = report.join(pvalues, how="left")
    report["in_confidence_set"] = report.index.isin(included)
    report["forecast_horizon"] = horizon
    report["alpha"] = alpha
    report["bootstrap_reps"] = bootstrap_reps
    report["block_size"] = block_size
    report["bootstrap_method"] = "circular block bootstrap"
    report["mcs_test_statistic"] = "R"
    report["n_observations"] = len(losses)
    report["start_date"] = str(losses.index.min().date())
    report["end_date"] = str(losses.index.max().date())

    return report.reset_index().sort_values(
        ["in_confidence_set", "mean_qlike"],
        ascending=[False, True],
        ignore_index=True,
    )


def main() -> None:
    """Write separate one-day and five-day MCS reports."""
    _, one_day_losses = build_one_day()
    _, five_day_losses = build_five_day()

    reports = [
        (
            one_day_losses,
            1,
            "reports/mcs_one_day.csv",
        ),
        (
            five_day_losses,
            5,
            "reports/mcs_five_day.csv",
        ),
    ]

    for losses, horizon, path in reports:
        report = build_mcs_report(losses, horizon=horizon)
        report.to_csv(path, index=False, float_format="%.12g")
        print(f"\nHorizon: {horizon} day(s)")
        print(
            f"QLIKE sample: {report['start_date'].iloc[0]} to "
            f"{report['end_date'].iloc[0]}"
        )
        print(f"Observations: {len(losses)}")
        print(f"Saved: {path}")
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()

