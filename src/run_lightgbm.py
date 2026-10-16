
"""Generate and save LightGBM next-day variance forecasts."""

from __future__ import annotations

import pandas as pd

from src.models.lightgbm_model import lightgbm_variance_forecasts


def main() -> None:
    """Load engineered features, generate forecasts, and save results."""
    features = pd.read_parquet(
        "data/processed/volatility_features.parquet"
    )

    forecasts = lightgbm_variance_forecasts(
        features,
        minimum_training_observations=252,
        reestimate_every=21,
        random_state=42,
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
