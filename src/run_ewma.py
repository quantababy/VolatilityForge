"""Generate historical EWMA variance forecasts."""

from __future__ import annotations

import pandas as pd
import yaml

from src.evaluation_config import load_evaluation_config
from src.forecasting import generate_ewma_forecasts


def main() -> None:
    """Generate EWMA forecasts across the available historical period."""
    with open("configs/config.yaml", encoding="utf-8") as file:
        project_config = yaml.safe_load(file)

    evaluation = load_evaluation_config()
    ewma_config = project_config["models"]["ewma"]

    features = pd.read_parquet(
        "data/processed/volatility_features.parquet"
    )

    forecasts = generate_ewma_forecasts(
        features,
        test_start=evaluation["historical_forecast_start"],
        decay=ewma_config["lambda"],
        minimum_training_observations=evaluation[
            "minimum_training_observations"
        ],
    )

    output_path = "data/processed/ewma_forecasts.parquet"
    forecasts.to_parquet(output_path)

    for column in ("ewma_variance_1d", "ewma_variance_5d"):
        valid = forecasts[column].dropna()
        if valid.empty:
            print(f"{column}: no valid forecasts")
            continue

        print(
            f"{column}: {len(valid)} forecasts, "
            f"{valid.index.min().date()} to {valid.index.max().date()}"
        )

    print(f"Saved forecasts to {output_path}")


if __name__ == "__main__":
    main()
