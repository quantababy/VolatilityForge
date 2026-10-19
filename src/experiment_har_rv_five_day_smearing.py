
"""Compare five-day HAR-RV forecasts with and without Duan smearing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.evaluation_config import load_evaluation_config
from src.models.har_rv import _prepare_log_predictors


def build_experiment_forecasts(
    features: pd.DataFrame,
    development_start: str,
    minimum_training_observations: int,
    reestimate_every: int,
) -> pd.DataFrame:
    """Reproduce five-day HAR-RV and add training-residual smearing."""
    data = features.copy()
    data.index = pd.DatetimeIndex(data.index).normalize()
    data.index.name = "Date"

    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Features must have unique, sorted dates.")

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
    smearing_factors = pd.Series(np.nan, index=data.index, dtype=float)

    fitted_model: LinearRegression | None = None
    last_fit_observations = 0
    current_smearing = np.nan

    for position in range(max(0, len(data) - 5)):
        if not valid_predictors.iloc[position]:
            continue

        # Match the production model's conservative eligibility rule:
        # training label origin s must satisfy s + 5 < forecast origin t.
        eligible = valid_positions[valid_positions <= position - 6]
        observations_available = len(eligible)

        if observations_available < minimum_training_observations:
            continue

        if (
            fitted_model is None
            or observations_available - last_fit_observations
            >= reestimate_every
        ):
            fitted_model = LinearRegression()
            x_train = predictors.iloc[eligible].to_numpy()
            y_train = log_target.iloc[eligible].to_numpy()
            fitted_model.fit(x_train, y_train)

            residuals = y_train - fitted_model.predict(x_train)
            with np.errstate(over="ignore", invalid="ignore"):
                residual_smearing = np.exp(residuals)

            current_smearing = float(np.mean(residual_smearing))
            if (
                not np.isfinite(current_smearing)
                or current_smearing <= 0
            ):
                raise RuntimeError(
                    "Invalid smearing factor at "
                    f"{data.index[position].date()}."
                )

            last_fit_observations = observations_available

        predicted_log = float(
            fitted_model.predict(predictors.iloc[[position]].to_numpy())[0]
        )
        baseline.iloc[position] = float(np.exp(np.clip(predicted_log, -700, 700)))
        corrected.iloc[position] = float(
            np.exp(np.clip(predicted_log + np.log(current_smearing), -700, 700))
        )
        smearing_factors.iloc[position] = current_smearing

    result = pd.DataFrame(
        {
            "target": target,
            "baseline_forecast": baseline,
            "smearing_forecast": corrected,
            "smearing_factor": smearing_factors,
        },
        index=data.index,
    )

    # Keep forecasts only for the configured development period onward.
    result = result.loc[pd.Timestamp(development_start):]
    return result


def normalized_qlike(target: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    ratio = target / forecast
    return ratio - np.log(ratio) - 1.0


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Evaluate both variants on exactly the same valid observations."""
    valid = (
        results["target"].notna()
        & np.isfinite(results["target"])
        & results["target"].gt(0)
        & results["baseline_forecast"].notna()
        & np.isfinite(results["baseline_forecast"])
        & results["baseline_forecast"].gt(0)
        & results["smearing_forecast"].notna()
        & np.isfinite(results["smearing_forecast"])
        & results["smearing_forecast"].gt(0)
    )
    sample = results.loc[valid].copy()
    if sample.empty:
        raise ValueError("No common valid observations for comparison.")

    y = sample["target"].to_numpy()
    summary_rows = []

    for label, column in (
        ("Baseline HAR-RV", "baseline_forecast"),
        ("Smearing-corrected HAR-RV", "smearing_forecast"),
    ):
        forecast = sample[column].to_numpy()
        qlike = normalized_qlike(y, forecast)
        mse = np.mean((y - forecast) ** 2)
        summary_rows.append(
            {
                "model": label,
                "observations": len(sample),
                "first_date": sample.index.min().date().isoformat(),
                "last_date": sample.index.max().date().isoformat(),
                "mean_qlike": float(np.mean(qlike)),
                "mse": float(mse),
                "mean_forecast": float(np.mean(forecast)),
            }
        )

    return pd.DataFrame(summary_rows)


def main() -> None:
    evaluation = load_evaluation_config()
    features = pd.read_parquet(
        "data/processed/volatility_features.parquet"
    )

    results = build_experiment_forecasts(
        features,
        development_start=evaluation["development_start"],
        minimum_training_observations=evaluation[
            "minimum_training_observations"
        ],
        reestimate_every=evaluation["reestimate_every"],
    )

    output_path = (
        "reports/har_rv_five_day_smearing_experiment.csv"
    )
    summary_path = (
        "reports/har_rv_five_day_smearing_summary.csv"
    )

    from pathlib import Path

    Path("reports").mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index_label="Date")
    summary = summarize(results)
    summary.to_csv(summary_path, index=False)

    print("Saved forecast comparison:", output_path)
    print("Saved summary:", summary_path)
    print("\nForecast comparison:")
    print(summary.to_string(index=False))
    print(
        "\nMean smearing factor:",
        f"{results['smearing_factor'].mean():.6f}",
    )


if __name__ == "__main__":
    main()
