
"""Generate latest-date prospective volatility forecasts."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation_config import load_evaluation_config
from src.forecasting import ewma_prospective_variance_forecast
from src.models.garch import (
    garch_five_day_prospective_variance_forecast,
    garch_prospective_variance_forecast,
)
from src.models.gjr_garch import (
    gjr_garch_five_day_prospective_variance_forecast,
    gjr_garch_prospective_variance_forecast,
)
from src.models.har_rv import (
    har_rv_five_day_prospective_variance_forecast,
    har_rv_prospective_variance_forecast,
)
from src.models.lightgbm_model import (
    lightgbm_prospective_variance_forecast,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "volatility_features.parquet"
ONE_DAY_OUTPUT = (
    ROOT / "data" / "processed" / "prospective_forecasts_1d.parquet"
)
FIVE_DAY_OUTPUT = (
    ROOT / "data" / "processed" / "prospective_forecasts_5d.parquet"
)



def _forecast_record(
    model: str,
    horizon_days: int,
    variance: float,
    forecast_date: pd.Timestamp,
    generated_at_utc: str,
) -> dict:
    """Validate and standardize one prospective forecast."""
    if not isinstance(variance, (int, float)) or not math.isfinite(variance):
        raise ValueError(f"{model} returned an invalid forecast.")

    if variance <= 0:
        raise ValueError(f"{model} returned a non-positive variance.")

    return {
        "forecast_date": forecast_date,
        "model": model,
        "horizon_days": horizon_days,
        "variance_forecast": float(variance),
        "generated_at_utc": generated_at_utc,
    }



def main() -> None:
    """Generate and save one-day and five-day forecasts."""
    with (ROOT / "configs" / "config.yaml").open(
        encoding="utf-8"
    ) as file:
        project_config = yaml.safe_load(file)

    evaluation = load_evaluation_config()
    ewma_config = project_config["models"]["ewma"]

    features = pd.read_parquet(FEATURES_PATH)

    if features.empty:
        raise ValueError(f"No feature data found in {FEATURES_PATH}.")

    features.index = pd.DatetimeIndex(features.index).normalize()
    features.index.name = "Date"

    if features.index.has_duplicates:
        raise ValueError("Feature data contains duplicate dates.")
    if not features.index.is_monotonic_increasing:
        raise ValueError("Feature data must be sorted by date.")
    if "log_return" not in features.columns:
        raise ValueError("Feature data must contain 'log_return'.")

    returns = features["log_return"].dropna()
    if returns.empty:
        raise ValueError("No observed returns are available.")

    latest_date = features.index[-1]
    generated_at_utc = datetime.now(UTC).isoformat()
    minimum_training = evaluation["minimum_training_observations"]
    reestimate_every = evaluation["reestimate_every"]
    random_seed = project_config.get("random_seed", 42)

    # Each function uses observations available through the latest close.
    one_day_forecasts = {
        "EWMA": ewma_prospective_variance_forecast(
            features,
            decay=ewma_config["lambda"],
            minimum_training_observations=minimum_training,
        ),
        "GARCH": garch_prospective_variance_forecast(
            returns,
            minimum_training_observations=minimum_training,
        ),
        "GJR-GARCH": gjr_garch_prospective_variance_forecast(
            returns,
            minimum_training_observations=minimum_training,
        ),
        "HAR-RV": har_rv_prospective_variance_forecast(
            features,
            minimum_training_observations=minimum_training,
            reestimate_every=reestimate_every,
        ),
        "LightGBM": lightgbm_prospective_variance_forecast(
            features,
            minimum_training_observations=minimum_training,
            reestimate_every=reestimate_every,
            random_state=random_seed,
        ),
    }

    five_day_forecasts = {
        "EWMA": 5.0 * one_day_forecasts["EWMA"],
        "GARCH": garch_five_day_prospective_variance_forecast(
            returns,
            minimum_training_observations=minimum_training,
        ),
        "GJR-GARCH": gjr_garch_five_day_prospective_variance_forecast(
            returns,
            minimum_training_observations=minimum_training,
        ),
        "HAR-RV": har_rv_five_day_prospective_variance_forecast(
            features,
            minimum_training_observations=minimum_training,
        ),
    }

    one_day_records = [
        _forecast_record(
            model=model,
            horizon_days=1,
            variance=variance,
            forecast_date=latest_date,
            generated_at_utc=generated_at_utc,
        )
        for model, variance in one_day_forecasts.items()
    ]

    five_day_records = [
        _forecast_record(
            model=model,
            horizon_days=5,
            variance=variance,
            forecast_date=latest_date,
            generated_at_utc=generated_at_utc,
        )
        for model, variance in five_day_forecasts.items()
    ]

    one_day_frame = pd.DataFrame(one_day_records)
    five_day_frame = pd.DataFrame(five_day_records)

    ONE_DAY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    one_day_frame.to_parquet(ONE_DAY_OUTPUT, index=False)
    five_day_frame.to_parquet(FIVE_DAY_OUTPUT, index=False)

    print(f"Latest available close: {latest_date.date()}")
    print(f"Generated at UTC: {generated_at_utc}")
    print("\nNext-day variance forecasts:")
    print(one_day_frame[["model", "variance_forecast"]].to_string(index=False))
    print("\nFive-session cumulative variance forecasts:")
    print(five_day_frame[["model", "variance_forecast"]].to_string(index=False))
    print(f"\nSaved: {ONE_DAY_OUTPUT}")
    print(f"Saved: {FIVE_DAY_OUTPUT}")


if __name__ == "__main__":
    main()
