"""Generate VIX-implied variance forecasts for both horizons."""

from __future__ import annotations

import pandas as pd
import yaml

from src.models.vix_implied import vix_implied_variance_forecasts


def main() -> None:
    with open("configs/config.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    trading_days = config["forecast"]["trading_days_per_year"]
    features = pd.read_parquet("data/processed/volatility_features.parquet")

    forecasts = vix_implied_variance_forecasts(
        features,
        trading_days_per_year=trading_days,
    )

    outputs = {
        "vix_implied_forecasts.parquet": "vix_implied_variance_1d",
        "vix_implied_five_day_forecasts.parquet": "vix_implied_variance_5d",
    }

    for filename, column in outputs.items():
        path = f"data/processed/{filename}"
        forecasts[[column]].to_parquet(path)
        print(f"Saved {path}: {forecasts[column].notna().sum()} forecasts")


if __name__ == "__main__":
    main()
