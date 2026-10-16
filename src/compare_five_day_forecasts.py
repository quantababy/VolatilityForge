
"""Compare five-day variance forecasts and test QLIKE differences."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from src.diebold_mariano import diebold_mariano_test

TEST_START = "2022-01-01"
HAC_LAGS = (4, 5, 10, 22)

FORECAST_FILES = {
    "EWMA": (
        "data/processed/ewma_forecasts.parquet",
        "ewma_variance_5d",
    ),
    "GARCH(1,1)": (
        "data/processed/garch_five_day_forecasts.parquet",
        "garch_variance_5d",
    ),
    "GJR-GARCH(1,1)": (
        "data/processed/gjr_garch_five_day_forecasts.parquet",
        "gjr_garch_variance_5d",
    ),
}


def normalized_qlike(
    target: pd.Series,
    forecast: pd.Series,
) -> pd.Series:
    """Calculate normalized QLIKE for positive variance values."""
    ratio = target / forecast
    return ratio - np.log(ratio) - 1.0


def build_comparison_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align five-day forecasts and targets on common valid dates."""
    features = pd.read_parquet(
        "data/processed/volatility_features.parquet"
    )
    target = features.loc[
        pd.Timestamp(TEST_START):, "target_rv_5"
    ]

    aligned = pd.DataFrame({"target": target})

    for model, (path, column) in FORECAST_FILES.items():
        forecasts = pd.read_parquet(path)
        aligned[model] = forecasts[column].reindex(target.index)

    model_columns = list(FORECAST_FILES)
    values = aligned[["target", *model_columns]]

    valid = (
        values.notna().all(axis=1)
        & np.isfinite(values).all(axis=1)
        & values.gt(0).all(axis=1)
    )
    aligned = aligned.loc[valid]

    if aligned.empty:
        raise ValueError("No common positive five-day observations.")

    losses = pd.DataFrame(index=aligned.index)
    for model in model_columns:
        losses[model] = normalized_qlike(
            aligned["target"], aligned[model]
        )

    return aligned, losses


def build_dm_report(losses: pd.DataFrame) -> pd.DataFrame:
    """Compare EWMA with alternatives across HAC lag choices."""
    rows: list[dict[str, str | int | float]] = []
    competitors = ("GARCH(1,1)", "GJR-GARCH(1,1)")

    for lag in HAC_LAGS:
        results = []

        for competitor in competitors:
            result = diebold_mariano_test(
                losses["EWMA"],
                losses[competitor],
                hac_lags=lag,
                forecast_horizon=5,
            )
            results.append((competitor, result))

        adjusted = multipletests(
            [result["p_value"] for _, result in results],
            method="holm",
        )[1]

        for (competitor, result), adjusted_p in zip(
            results, adjusted
        ):
            rows.append({
                "comparison": f"EWMA vs {competitor}",
                "n_observations": result["n_observations"],
                "start_date": str(losses.index.min().date()),
                "end_date": str(losses.index.max().date()),
                "forecast_horizon": 5,
                "hac_lags": lag,
                "mean_loss_differential": (
                    result["mean_loss_differential"]
                ),
                "dm_statistic": result["dm_statistic"],
                "p_value": result["p_value"],
                "holm_p_value": float(adjusted_p),
                "inference_method": (
                    "Newey-West HAC + HLN factor; t(n-1) reference"
                ),
            })

    return pd.DataFrame(rows)


def main() -> None:
    """Write five-day metrics and DM sensitivity reports."""
    aligned, losses = build_comparison_frame()

    summary_rows = []
    for model in FORECAST_FILES:
        errors = aligned[model] - aligned["target"]
        summary_rows.append({
            "model": model,
            "n_observations": len(aligned),
            "start_date": str(aligned.index.min().date()),
            "end_date": str(aligned.index.max().date()),
            "qlike": float(losses[model].mean()),
            "mse": float(np.mean(errors**2)),
        })

    summary = pd.DataFrame(summary_rows).sort_values("qlike")
    dm_report = build_dm_report(losses)

    summary_path = "reports/five_day_model_comparison.csv"
    dm_path = "reports/five_day_dm_sensitivity.csv"

    summary.to_csv(summary_path, index=False, float_format="%.12g")
    dm_report.to_csv(dm_path, index=False, float_format="%.12g")

    print("Evaluation: development-period evidence, not an untouched holdout")
    print(f"Observations: {len(aligned)}")
    print(
        f"Dates: {aligned.index.min().date()} to "
        f"{aligned.index.max().date()}"
    )
    print("\nFive-day model comparison:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.10g}"))
    print(f"\nSaved: {summary_path}")
    print(f"\nDM sensitivity report: {dm_path}")
    print(dm_report.to_string(
        index=False,
        float_format=lambda x: f"{x:.8g}",
    ))


if __name__ == "__main__":
    main()
