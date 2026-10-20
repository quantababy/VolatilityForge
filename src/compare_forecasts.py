
"""Reproducible QLIKE and DM-test sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from src.calibration import mincer_zarnowitz
from src.diebold_mariano import diebold_mariano_test
from src.evaluation_config import load_evaluation_config

HAC_LAGS = (5, 10, 22)
CALIBRATION_HAC_LAG = 5
FORECAST_FILES = {
    "EWMA": ("data/processed/ewma_forecasts.parquet", "ewma_variance_1d"),
    "GARCH(1,1)": ("data/processed/garch_forecasts.parquet", "garch_variance_1d"),
    "GJR-GARCH(1,1)": (
        "data/processed/gjr_garch_forecasts.parquet",
        "gjr_garch_variance_1d",
    ),
    "HAR-RV": ("data/processed/har_rv_forecasts.parquet", "har_rv_variance_1d"),
    "LightGBM": (
        "data/processed/lightgbm_forecasts.parquet",
        "lightgbm_variance_1d",
    ),
}


def _validate_datetime_index(index: pd.Index, label: str) -> None:
    """Reject indexes that cannot be safely aligned by timestamp."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{label} index must be a DatetimeIndex.")

    if not index.is_unique:
        raise ValueError(f"{label} index must contain unique timestamps.")

    if not index.is_monotonic_increasing:
        raise ValueError(
            f"{label} index must be sorted in ascending order."
        )


def normalized_qlike(target: pd.Series, forecast: pd.Series) -> pd.Series:
    """Return normalized QLIKE losses for strictly positive variances."""
    ratio = target / forecast
    return ratio - np.log(ratio) - 1.0


def build_comparison_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return common MSE observations and positive-target QLIKE losses."""
    test_start = load_evaluation_config()["development_start"]

    features = pd.read_parquet("data/processed/volatility_features.parquet")
    _validate_datetime_index(features.index, "Feature")

    if "target_rv_1" not in features.columns:
        raise ValueError("Feature data must contain target_rv_1.")

    target = features.loc[pd.Timestamp(test_start):, "target_rv_1"]
    aligned = pd.DataFrame({"target": target})

    for model, (path, column) in FORECAST_FILES.items():
        frame = pd.read_parquet(path)
        _validate_datetime_index(frame.index, f"{model} forecast")

        if column not in frame.columns:
            raise ValueError(
                f"{model} forecast data must contain {column}."
            )

        aligned[model] = frame[column].reindex(target.index)

    forecast_columns = list(FORECAST_FILES)
    valid_forecasts = (
        aligned[forecast_columns].notna()
        & np.isfinite(aligned[forecast_columns])
        & aligned[forecast_columns].gt(0)
    ).all(axis=1)

    valid_targets = aligned["target"].notna() & np.isfinite(aligned["target"])

    mse_mask = valid_targets & aligned["target"].ge(0) & valid_forecasts
    mse_aligned = aligned.loc[mse_mask].copy()

    if mse_aligned.empty:
        raise ValueError("No common valid observations are available for MSE.")

    qlike_aligned = mse_aligned.loc[mse_aligned["target"].gt(0)].copy()

    if qlike_aligned.empty:
        raise ValueError(
            "No common positive-target observations are available for QLIKE."
        )

    losses = pd.DataFrame(index=qlike_aligned.index)
    for model in FORECAST_FILES:
        losses[model] = normalized_qlike(
            qlike_aligned["target"], qlike_aligned[model]
        )

    return mse_aligned, losses


def run_dm_sensitivity(losses: pd.DataFrame) -> pd.DataFrame:
    """Run pairwise EWMA comparisons and Holm-adjust p-values by HAC lag."""
    rows: list[dict[str, str | int | float]] = []

    competitors = (
        "GARCH(1,1)",
        "GJR-GARCH(1,1)",
        "HAR-RV",
        "LightGBM",
    )

    for lag in HAC_LAGS:
        comparisons = []

        for competitor in competitors:
            result = diebold_mariano_test(
                losses["EWMA"],
                losses[competitor],
                hac_lags=lag,
                forecast_horizon=1,
            )
            comparisons.append((competitor, result))

        adjusted_p_values = multipletests(
            [result["p_value"] for _, result in comparisons],
            method="holm",
        )[1]

        for (competitor, result), adjusted_p in zip(
            comparisons, adjusted_p_values
        ):
            rows.append({
                "comparison": f"EWMA vs {competitor}",
                "n_observations": result["n_observations"],
                "start_date": str(losses.index.min().date()),
                "end_date": str(losses.index.max().date()),
                "forecast_horizon": result["forecast_horizon"],
                "hac_lags": lag,
                "mean_loss_differential": result["mean_loss_differential"],
                "dm_statistic": result["dm_statistic"],
                "p_value": result["p_value"],
                "holm_p_value": float(adjusted_p),
                "inference_method": (
                    "Newey-West HAC + HLN factor; t(n-1) reference"
                ),
            })

    return pd.DataFrame(rows)


def build_calibration_report(
    mse_aligned: pd.DataFrame,
    *,
    hac_lags: int = CALIBRATION_HAC_LAG,
) -> pd.DataFrame:
    """Build Mincer-Zarnowitz calibration diagnostics for one-day forecasts."""
    rows = []

    for model in FORECAST_FILES:
        result = mincer_zarnowitz(
            target=mse_aligned["target"],
            forecast=mse_aligned[model],
            hac_lags=hac_lags,
        )
        rows.append({
            "model": model,
            "forecast_horizon": 1,
            "start_date": str(mse_aligned.index.min().date()),
            "end_date": str(mse_aligned.index.max().date()),
            **result,
        })

    return pd.DataFrame(rows)



def main() -> None:
    """Generate the model-loss summary and DM sensitivity report."""
    mse_aligned, losses = build_comparison_frame()

    calibration_report = build_calibration_report(mse_aligned)
    calibration_path = "reports/one_day_calibration.csv"
    calibration_report.to_csv(
        calibration_path,
        index=False,
        float_format="%.12g",
    )

    summary_rows = []

    for model in FORECAST_FILES:
        mse_errors = mse_aligned[model] - mse_aligned["target"]

        summary_rows.append({
            "model": model,
            "forecast_horizon": 1,
            "n_mse_observations": len(mse_aligned),
            "n_qlike_observations": len(losses),
            "mse_start_date": str(mse_aligned.index.min().date()),
            "mse_end_date": str(mse_aligned.index.max().date()),
            "qlike_start_date": str(losses.index.min().date()),
            "qlike_end_date": str(losses.index.max().date()),
            "qlike": float(losses[model].mean()),
            "mse": float(np.mean(mse_errors**2)),
        })

    summary = pd.DataFrame(summary_rows).sort_values("qlike")
    summary_path = "reports/volatility_model_comparison.csv"
    summary.to_csv(summary_path, index=False, float_format="%.12g")

    report = run_dm_sensitivity(losses)
    report_path = "reports/dm_test_sensitivity.csv"
    report.to_csv(report_path, index=False)

    print(f"MSE observations: {len(mse_aligned)}")
    print(f"QLIKE/DM observations: {len(losses)}")
    print(
        "MSE evaluation period: "
        f"{mse_aligned.index.min().date()} to {mse_aligned.index.max().date()}"
    )
    print(
        "QLIKE/DM evaluation period: "
        f"{losses.index.min().date()} to {losses.index.max().date()}"
    )
    print("\nModel comparison:")
    print(summary.to_string(index=False))
    print(f"\nDM sensitivity report saved to {report_path}")
    print(report.to_string(index=False))
    print(f"\nCalibration report saved to {calibration_path}")
    print(calibration_report.to_string(index=False))
    print(f"\nModel comparison report saved to {summary_path}")

if __name__ == "__main__":
    main()
