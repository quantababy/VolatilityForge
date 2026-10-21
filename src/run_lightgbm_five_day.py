"""Generate and save LightGBM five-day variance forecasts."""

from __future__ import annotations

import pandas as pd
import yaml

from src.evaluation_config import load_evaluation_config
from src.models.lightgbm_model import lightgbm_five_day_variance_forecasts


def main() -> None:
    with open("configs/config.yaml", encoding="utf-8") as file:
        project_config = yaml.safe_load(file)

    evaluation = load_evaluation_config()
    features = pd.read_parquet("data/processed/volatility_features.parquet")

    forecasts = lightgbm_five_day_variance_forecasts(
        features,
        minimum_training_observations=evaluation[
            "minimum_training_observations"
        ],
        reestimate_every=evaluation["reestimate_every"],
        random_state=project_config["project"]["random_seed"],
    )

    output_path = "data/processed/lightgbm_five_day_forecasts.parquet"
    forecasts.to_frame().to_parquet(output_path)

    available = forecasts.dropna()
    if available.empty:
        raise ValueError("LightGBM produced no five-day forecasts.")

    print(f"Saved forecasts: {output_path}")
    print(f"Forecast count: {len(available)}")
    print(f"Forecast period: {available.index.min()} to {available.index.max()}")
    print(f"Minimum forecast: {available.min():.8f}")
    print(f"Maximum forecast: {available.max():.8f}")


if __name__ == "__main__":
    main()
