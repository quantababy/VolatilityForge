
"""Generate and save LightGBM next-day variance forecasts."""

from __future__ import annotations

import pandas as pd
import yaml

from src.evaluation_config import load_evaluation_config
from src.models.lightgbm_model import lightgbm_variance_forecasts


def main() -> None:
    """Load engineered features, generate forecasts, and save results."""
    with open("configs/config.yaml", encoding="utf-8") as file:
        project_config = yaml.safe_load(file)

    evaluation = load_evaluation_config()
    features = pd.read_parquet(
        "data/processed/volatility_features.parquet"
    )

    forecasts = lightgbm_variance_forecasts(
        features,
        minimum_training_observations=evaluation[
            "minimum_training_observations"
        ],
        reestimate_every=evaluation["reestimate_every"],
        random_state=project_config["project"]["random_seed"],
    )

    output_path = "data/processed/lightgbm_forecasts.parquet"
    forecasts.to_frame().to_parquet(output_path)

    available = forecasts.dropna()

    print(f"Forecasts saved to {output_path}")
    print(f"Available forecasts: {len(available)}")
    if not available.empty:
        print(
            "Forecast period: "
            f"{available.index.min().date()} to "
            f"{available.index.max().date()}"
        )
        print(f"Minimum forecast variance: {available.min():.10f}")
        print(f"Maximum forecast variance: {available.max():.10f}")


if __name__ == "__main__":
    main()
