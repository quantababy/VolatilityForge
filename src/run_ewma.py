"""Generate historical EWMA variance forecasts."""

from __future__ import annotations

import pandas as pd

from src.forecasting import generate_ewma_forecasts


def main() -> None:
    """Generate EWMA forecasts across the available historical period."""
    features = pd.read_parquet(
        "data/processed/volatility_features.parquet"
    )

    forecasts = generate_ewma_forecasts(
        features,
        test_start="2001-01-03",
        decay=0.94,
        minimum_training_observations=252,
    )

    output_path = "data/processed/ewma_forecasts.parquet"
    forecasts.to_parquet(output_path)

    for column in ("ewma_variance_1d", "ewma_variance_5d"):
        valid = forecasts[column].dropna()
        print(
            f"{column}: {len(valid)} forecasts, "
            f"{valid.index.min().date()} to {valid.index.max().date()}"
        )

    print(f"Saved forecasts to {output_path}")


if __name__ == "__main__":
    main()

