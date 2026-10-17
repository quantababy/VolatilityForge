
"""Generate reproducible volatility-regime forecast evaluation reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation_config import load_evaluation_config
from src.regime_analysis import (
    assign_volatility_regimes,
    evaluate_forecasts_by_regime,
)

FEATURES_PATH = Path("data/processed/volatility_features.parquet")
REPORTS_DIR = Path("reports")

FORECAST_FILES = {
    "EWMA": (
        "data/processed/ewma_forecasts.parquet",
        "ewma_variance_1d",
        "ewma_variance_5d",
    ),
    "GARCH(1,1)": (
        "data/processed/garch_forecasts.parquet",
        "garch_variance_1d",
        "data/processed/garch_five_day_forecasts.parquet",
    ),
    "GJR-GARCH(1,1)": (
        "data/processed/gjr_garch_forecasts.parquet",
        "gjr_garch_variance_1d",
        "data/processed/gjr_garch_five_day_forecasts.parquet",
    ),
    "HAR-RV": (
        "data/processed/har_rv_forecasts.parquet",
        "har_rv_variance_1d",
        "data/processed/har_rv_five_day_forecasts.parquet",
    ),
    "LightGBM": (
        "data/processed/lightgbm_forecasts.parquet",
        "lightgbm_variance_1d",
        None,
    ),
}


def build_regime_report(
    features: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Evaluate available forecasts separately in low/high regimes."""
    if horizon not in (1, 5):
        raise ValueError("horizon must be 1 or 5.")

    target_column = f"target_rv_{horizon}"
    if target_column not in features.columns:
        raise ValueError(f"Missing target column: {target_column}")

    regimes = assign_volatility_regimes(features)
    target = features[target_column]
    development_start = pd.Timestamp(
        load_evaluation_config()["development_start"]
    )

    rows: list[dict[str, str | int | float]] = []

    for model, (one_day_path, one_day_column, five_day_path) in (
        FORECAST_FILES.items()
    ):
        if horizon == 1:
            path = one_day_path
            forecast_column = one_day_column
        else:
            forecast_column = {
                "EWMA": "ewma_variance_5d",
                "GARCH(1,1)": "garch_variance_5d",
                "GJR-GARCH(1,1)": "gjr_garch_variance_5d",
                "HAR-RV": "har_rv_variance_5d",
            }.get(model)

            path = (
                one_day_path
                if model == "EWMA"
                else five_day_path
            )

        if forecast_column is None:
            continue

        forecast_frame = pd.read_parquet(path)
        if forecast_column not in forecast_frame.columns:
            raise ValueError(
                f"{model} forecast file lacks column {forecast_column}."
            )

        forecasts = forecast_frame[forecast_column].reindex(features.index)

        common_index = features.index[features.index >= development_start]
        forecasts = forecasts.reindex(common_index)
        model_target = target.reindex(common_index)
        model_regimes = regimes.reindex(common_index)

        valid = (
            forecasts.notna()
            & model_target.notna()
            & np.isfinite(forecasts)
            & np.isfinite(model_target)
            & forecasts.gt(0)
            & model_target.ge(0)
            & model_regimes.notna()
        )
        
        forecasts = forecasts.loc[valid]
        model_target = model_target.loc[valid]
        model_regimes = model_regimes.loc[valid]

        if forecasts.empty:
            continue

        metrics = evaluate_forecasts_by_regime(
            forecasts,
            model_target,
            model_regimes,
        )

        for record in metrics.to_dict(orient="records"):
            rows.append({
                "model": model,
                "horizon_days": horizon,
                "start_date": str(forecasts.index.min().date()),
                "end_date": str(forecasts.index.max().date()),
                **record,
            })

    return pd.DataFrame(rows)


def main() -> None:
    """Write one-day and five-day regime comparison reports."""
    features = pd.read_parquet(FEATURES_PATH)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for horizon in (1, 5):
        report = build_regime_report(features, horizon)
        output_path = REPORTS_DIR / f"regime_comparison_{horizon}d.csv"
        report.to_csv(output_path, index=False)

        print(f"\n{horizon}-day regime comparison")
        if report.empty:
            print("No valid forecast-regime observations were available.")
        else:
            print(report.to_string(index=False))
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
