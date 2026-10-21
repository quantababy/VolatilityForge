"""Compare five-day variance forecasts and test QLIKE differences."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from src.calibration import mincer_zarnowitz
from src.diebold_mariano import diebold_mariano_test
from src.evaluation_config import load_evaluation_config

PRIMARY_HAC_LAG = 4
HAC_SENSITIVITY_LAGS = (5, 10, 22)
HAC_LAGS = (PRIMARY_HAC_LAG, *HAC_SENSITIVITY_LAGS)
CALIBRATION_HAC_LAG = 4
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
    "HAR-RV": (
        "data/processed/har_rv_five_day_forecasts.parquet",
        "har_rv_variance_5d",
    ),
    "LightGBM": (
        "data/processed/lightgbm_five_day_forecasts.parquet",
        "lightgbm_variance_5d",
    ),
    "VIX-implied": (
        "data/processed/vix_implied_five_day_forecasts.parquet",
        "vix_implied_variance_5d",
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
    """Return common MSE observations and positive-target QLIKE losses."""
    test_start = load_evaluation_config()["development_start"]
    features = pd.read_parquet(
        "data/processed/volatility_features.parquet"
    )
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("Feature index must be a DatetimeIndex.")
    if features.index.has_duplicates:
        raise ValueError("Feature dates must be unique.")
    if not features.index.is_monotonic_increasing:
        raise ValueError("Feature dates must be sorted.")
    if "target_rv_5" not in features.columns:
        raise ValueError("'target_rv_5' column is required.")

    target = features.loc[
        pd.Timestamp(test_start):, "target_rv_5"
    ]

    aligned = pd.DataFrame({"target": target})

    for model, (path, column) in FORECAST_FILES.items():
        forecasts = pd.read_parquet(path)

        if not isinstance(forecasts.index, pd.DatetimeIndex):
            raise TypeError(
                f"{model} forecast index must be a DatetimeIndex."
            )
        if forecasts.index.has_duplicates:
            raise ValueError(
                f"{model} forecast dates must be unique."
            )
        if not forecasts.index.is_monotonic_increasing:
            raise ValueError(
                f"{model} forecast dates must be sorted."
            )
        if column not in forecasts.columns:
            raise ValueError(
                f"{column!r} not found in {path}."
            )

        aligned[model] = forecasts[column].reindex(target.index)

    model_columns = list(FORECAST_FILES)
    forecast_values = aligned[model_columns]
    valid_forecasts = (
        forecast_values.notna()
        & np.isfinite(forecast_values)
        & forecast_values.gt(0)
    ).all(axis=1)
    valid_targets = (
        aligned["target"].notna()
        & np.isfinite(aligned["target"])
    )

    mse_mask = (
        valid_targets
        & aligned["target"].ge(0)
        & valid_forecasts
    )
    mse_aligned = aligned.loc[mse_mask].copy()

    if mse_aligned.empty:
        raise ValueError("No common valid five-day observations for MSE.")

    qlike_aligned = mse_aligned.loc[
        mse_aligned["target"].gt(0)
    ].copy()

    if qlike_aligned.empty:
        raise ValueError(
            "No common positive-target five-day observations for QLIKE."
        )

    losses = pd.DataFrame(index=qlike_aligned.index)
    for model in model_columns:
        losses[model] = normalized_qlike(
            qlike_aligned["target"], qlike_aligned[model]
        )

    return mse_aligned, losses


def build_dm_report(losses: pd.DataFrame) -> pd.DataFrame:
    """Compare EWMA with alternatives using primary and sensitivity HAC lags."""
    rows: list[dict[str, str | int | float]] = []
    competitors = tuple(
        model for model in losses.columns if model != "EWMA"
    )

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

        for (competitor, result), adjusted_p in zip(results, adjusted):
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


def build_calibration_report(
    mse_aligned: pd.DataFrame,
    *,
    hac_lags: int = CALIBRATION_HAC_LAG,
) -> pd.DataFrame:
    """Build Mincer-Zarnowitz calibration diagnostics for five-day forecasts."""
    rows = []

    for model in FORECAST_FILES:
        result = mincer_zarnowitz(
            target=mse_aligned["target"],
            forecast=mse_aligned[model],
            hac_lags=hac_lags,
        )
        rows.append({
            "model": model,
            "forecast_horizon": 5,
            "start_date": str(mse_aligned.index.min().date()),
            "end_date": str(mse_aligned.index.max().date()),
            **result,
        })

    return pd.DataFrame(rows)


def main() -> None:
    """Write five-day metrics and DM sensitivity reports."""
    mse_aligned, losses = build_comparison_frame()

    calibration_report = build_calibration_report(mse_aligned)
    calibration_path = "reports/five_day_calibration.csv"
    calibration_report.to_csv(
        calibration_path,
        index=False,
        float_format="%.12g",
    )

    summary_rows = []
    for model in FORECAST_FILES:
        errors = mse_aligned[model] - mse_aligned["target"]
        summary_rows.append({
            "model": model,
            "n_mse_observations": len(mse_aligned),
            "n_qlike_observations": len(losses),
            "mse_start_date": str(mse_aligned.index.min().date()),
            "mse_end_date": str(mse_aligned.index.max().date()),
            "qlike_start_date": str(losses.index.min().date()),
            "qlike_end_date": str(losses.index.max().date()),
            "qlike": float(losses[model].mean()),
            "mse": float(np.mean(errors**2)),
            "forecast_horizon": 5,
        })

    summary = pd.DataFrame(summary_rows).sort_values("qlike")
    dm_report = build_dm_report(losses)

    summary_path = "reports/five_day_model_comparison.csv"
    dm_path = "reports/five_day_dm_sensitivity.csv"

    summary.to_csv(summary_path, index=False, float_format="%.12g")
    dm_report.to_csv(dm_path, index=False, float_format="%.12g")

    print("Evaluation: development-period evidence, not an untouched holdout")
    print(f"MSE observations: {len(mse_aligned)}")
    print(f"QLIKE/DM observations: {len(losses)}")
    print(
        f"MSE dates: {mse_aligned.index.min().date()} to "
        f"{mse_aligned.index.max().date()}"
    )
    print(
        f"QLIKE/DM dates: {losses.index.min().date()} to "
        f"{losses.index.max().date()}"
    )
    print("\nFive-day model comparison:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.10g}"))
    print(f"\nSaved: {summary_path}")
    print(f"\nDM sensitivity report: {dm_path}")
    print(dm_report.to_string(
        index=False,
        float_format=lambda x: f"{x:.8g}",
    ))
    print(f"\nCalibration report: {calibration_path}")
    print(calibration_report.to_string(
    index=False,
    float_format=lambda x: f"{x:.8g}",
    ))


if __name__ == "__main__":
    main()
