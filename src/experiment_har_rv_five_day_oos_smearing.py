
"""Evaluate five-day HAR-RV with out-of-sample smearing correction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.evaluation_config import load_evaluation_config
from src.models.har_rv import _prepare_log_predictors


def estimate_oos_smearing_factor(
    residuals: pd.Series,
    minimum_residuals: int = 60,
) -> float:
    """Estimate Duan's smearing factor from past out-of-sample residuals."""
    if (
        not isinstance(minimum_residuals, int)
        or isinstance(minimum_residuals, bool)
        or minimum_residuals < 1
    ):
        raise ValueError("minimum_residuals must be a positive integer.")

    valid = residuals.dropna().to_numpy(dtype=float)
    valid = valid[np.isfinite(valid)]

    if len(valid) < minimum_residuals:
        raise ValueError("Insufficient eligible out-of-sample residuals.")

    with np.errstate(over="ignore", invalid="ignore"):
        exponentiated = np.exp(valid)

    if not np.isfinite(exponentiated).all():
        raise ValueError("Residuals produce non-finite smearing values.")

    factor = float(np.mean(exponentiated))

    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Invalid out-of-sample smearing factor.")

    return factor


def eligible_oos_residual_positions(
    forecast_position: int,
    total_positions: int,
) -> np.ndarray:
    """Return origins whose five-day targets mature before the forecast origin."""
    if (
        not isinstance(forecast_position, int)
        or isinstance(forecast_position, bool)
        or not isinstance(total_positions, int)
        or isinstance(total_positions, bool)
        or forecast_position < 0
        or total_positions < 0
    ):
        raise ValueError("Positions must be non-negative integers.")

    stop = min(max(0, forecast_position - 5), total_positions)
    return np.arange(stop, dtype=int)


def build_oos_smearing_forecasts(
    features: pd.DataFrame,
    development_start: str,
    minimum_training_observations: int = 252,
    reestimate_every: int = 21,
    minimum_residuals: int = 60,
) -> pd.DataFrame:
    """Compare baseline HAR-RV with a past-OOS-residual correction."""
    if (
        not isinstance(minimum_training_observations, int)
        or isinstance(minimum_training_observations, bool)
        or minimum_training_observations < 30
    ):
        raise ValueError(
            "minimum_training_observations must be an integer of at least 30."
        )

    if (
        not isinstance(reestimate_every, int)
        or isinstance(reestimate_every, bool)
        or reestimate_every < 1
    ):
        raise ValueError("reestimate_every must be a positive integer.")

    if (
        not isinstance(minimum_residuals, int)
        or isinstance(minimum_residuals, bool)
        or minimum_residuals < 1
    ):
        raise ValueError("minimum_residuals must be a positive integer.")

    data = features.copy()

    if data.empty:
        raise ValueError("features must not be empty.")

    data.index = pd.DatetimeIndex(data.index).normalize()
    data.index.name = "Date"

    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Features must have unique, sorted dates.")

    if "target_rv_5" not in data.columns:
        raise ValueError("Features must contain the 'target_rv_5' column.")

    predictors = _prepare_log_predictors(data)
    valid_predictors = (
        predictors.notna().all(axis=1)
        & np.isfinite(predictors).all(axis=1)
    )

    target = data["target_rv_5"]
    log_target = np.log(target.where(target > 0))
    valid_training = (
        valid_predictors
        & log_target.notna()
        & np.isfinite(log_target)
    )
    valid_positions = np.flatnonzero(valid_training.to_numpy())

    baseline = pd.Series(np.nan, index=data.index, dtype=float)
    corrected = pd.Series(np.nan, index=data.index, dtype=float)
    factors = pd.Series(np.nan, index=data.index, dtype=float)
    residuals = pd.Series(np.nan, index=data.index, dtype=float)

    model: LinearRegression | None = None
    last_fit_observations = 0

    # The last five rows do not have fully observed five-day targets.
    for position in range(max(0, len(data) - 5)):
        if not valid_predictors.iloc[position]:
            continue

        # Training label origin s must satisfy s + 5 < forecast origin t.
        eligible = valid_positions[valid_positions <= position - 6]
        observations_available = len(eligible)

        if observations_available < minimum_training_observations:
            continue

        if (
            model is None
            or observations_available - last_fit_observations
            >= reestimate_every
        ):
            model = LinearRegression()
            model.fit(
                predictors.iloc[eligible].to_numpy(),
                log_target.iloc[eligible].to_numpy(),
            )
            last_fit_observations = observations_available

        predicted_log = float(
            model.predict(predictors.iloc[[position]].to_numpy())[0]
        )
        baseline.iloc[position] = float(
            np.exp(np.clip(predicted_log, -700, 700))
        )

        # Only use historical forecast errors whose five-day outcomes have
        # matured strictly before this forecast origin.
        eligible_residuals = eligible_oos_residual_positions(
            position,
            len(residuals),
        )
        past_residuals = residuals.iloc[eligible_residuals].dropna()

        if len(past_residuals) >= minimum_residuals:
            factor = estimate_oos_smearing_factor(
                past_residuals,
                minimum_residuals=minimum_residuals,
            )
            factors.iloc[position] = factor
            corrected.iloc[position] = float(
                np.exp(
                    np.clip(
                        predicted_log + np.log(factor),
                        -700,
                        700,
                    )
                )
            )

        # Record the forecast error only after making the forecast.
        # This residual can only be used by a later eligible origin.
        if pd.notna(target.iloc[position]) and target.iloc[position] > 0:
            residuals.iloc[position] = (
                np.log(target.iloc[position]) - predicted_log
            )

    results = pd.DataFrame(
        {
            "target": target,
            "baseline_forecast": baseline,
            "oos_smearing_forecast": corrected,
            "oos_smearing_factor": factors,
            "oos_log_residual": residuals,
        },
        index=data.index,
    )

    return results.loc[pd.Timestamp(development_start):]


def summarize_oos_smearing(results: pd.DataFrame) -> pd.DataFrame:
    """Compare forecasts on identical valid observations."""
    valid = (
        results["target"].notna()
        & np.isfinite(results["target"])
        & results["target"].gt(0)
        & results["baseline_forecast"].notna()
        & np.isfinite(results["baseline_forecast"])
        & results["baseline_forecast"].gt(0)
        & results["oos_smearing_forecast"].notna()
        & np.isfinite(results["oos_smearing_forecast"])
        & results["oos_smearing_forecast"].gt(0)
    )
    sample = results.loc[valid]

    if sample.empty:
        raise ValueError("No common valid observations for comparison.")

    target = sample["target"].to_numpy()
    rows = []

    for label, column in (
        ("Baseline HAR-RV", "baseline_forecast"),
        ("OOS-smearing HAR-RV", "oos_smearing_forecast"),
    ):
        forecast = sample[column].to_numpy()
        ratio = target / forecast
        qlike = ratio - np.log(ratio) - 1.0

        rows.append(
            {
                "model": label,
                "observations": len(sample),
                "first_date": sample.index.min().date().isoformat(),
                "last_date": sample.index.max().date().isoformat(),
                "mean_qlike": float(np.mean(qlike)),
                "mse": float(np.mean((target - forecast) ** 2)),
                "mean_forecast": float(np.mean(forecast)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Run the experiment and save forecast-level and summary reports."""
    evaluation = load_evaluation_config()
    features = pd.read_parquet(
        "data/processed/volatility_features.parquet"
    )

    results = build_oos_smearing_forecasts(
        features,
        development_start=evaluation["development_start"],
        minimum_training_observations=evaluation[
            "minimum_training_observations"
        ],
        reestimate_every=evaluation["reestimate_every"],
    )

    Path("reports").mkdir(parents=True, exist_ok=True)

    forecast_path = Path(
        "reports/har_rv_five_day_oos_smearing_experiment.csv"
    )
    summary_path = Path(
        "reports/har_rv_five_day_oos_smearing_summary.csv"
    )

    summary = summarize_oos_smearing(results)

    results.to_csv(forecast_path, index_label="Date")
    summary.to_csv(summary_path, index=False)

    print("Saved forecast comparison:", forecast_path)
    print("Saved summary:", summary_path)
    print("\nForecast comparison:")
    print(summary.to_string(index=False))

    available_factors = results["oos_smearing_factor"].dropna()
    if available_factors.empty:
        print("\nNo OOS smearing factors were available.")
    else:
        print(
            "\nMean OOS smearing factor:",
            f"{available_factors.mean():.6f}",
        )


if __name__ == "__main__":
    main()
