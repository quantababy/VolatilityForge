
"""Analyze five-day HAR-RV OOS-smearing forecast performance."""

from pathlib import Path

import numpy as np
import pandas as pd

REPORT = Path("reports/har_rv_five_day_oos_smearing_experiment.csv")
FEATURES = Path("data/processed/volatility_features.parquet")
OUTPUT = Path("reports/har_rv_five_day_oos_smearing_regime_analysis.csv")


def add_metrics(frame: pd.DataFrame, group_name: str, group_value: object) -> dict:
    target = frame["target"].to_numpy(dtype=float)
    rows = {"group_type": group_name, "group": str(group_value), "n": len(frame)}

    for label, column in (
        ("baseline", "baseline_forecast"),
        ("corrected", "oos_smearing_forecast"),
    ):
        forecast = frame[column].to_numpy(dtype=float)
        ratio = target / forecast
        rows[f"{label}_qlike"] = float(
            np.mean(ratio - np.log(ratio) - 1.0)
        )
        rows[f"{label}_mse"] = float(np.mean((target - forecast) ** 2))
        rows[f"{label}_mean_forecast"] = float(np.mean(forecast))

    rows["qlike_improvement_pct"] = (
        100
        * (rows["baseline_qlike"] - rows["corrected_qlike"])
        / rows["baseline_qlike"]
    )
    rows["mse_improvement_pct"] = (
        100 * (rows["baseline_mse"] - rows["corrected_mse"])
        / rows["baseline_mse"]
    )
    return rows


def main() -> None:
    forecasts = pd.read_csv(REPORT, parse_dates=["Date"]).set_index("Date")
    features = pd.read_parquet(FEATURES)
    features.index = pd.DatetimeIndex(features.index).normalize()

    common = forecasts.join(
        features[["rv_22", "VIX_Close"]],
        how="left",
    )

    valid = (
        common["target"].gt(0)
        & common["baseline_forecast"].gt(0)
        & common["oos_smearing_forecast"].gt(0)
        & np.isfinite(common["target"])
        & np.isfinite(common["baseline_forecast"])
        & np.isfinite(common["oos_smearing_forecast"])
    )
    sample = common.loc[valid].copy()

    if sample.empty:
        raise ValueError("No common valid forecast observations.")

    results = []

    for year, group in sample.groupby(sample.index.year):
        results.append(add_metrics(group, "year", year))

    # Define volatility regimes using tertiles of the evaluation targets.
    # This is descriptive and uses outcomes, so it is not a deployable
    # real-time regime classifier.
    low_cut, high_cut = sample["target"].quantile([1 / 3, 2 / 3])
    sample["volatility_regime"] = pd.cut(
        sample["target"],
        bins=[-np.inf, low_cut, high_cut, np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )

    for regime, group in sample.groupby("volatility_regime", observed=True):
        results.append(add_metrics(group, "realized_variance_regime", regime))

    output = pd.DataFrame(results)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)

    print("Evaluation observations:", len(sample))
    print("Evaluation dates:", sample.index.min().date(), "to", sample.index.max().date())
    print("Realized-variance tertile cutoffs:", low_cut, high_cut)
    print("\nRegime/year analysis:")
    print(output.to_string(index=False))
    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
