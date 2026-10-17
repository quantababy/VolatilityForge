"""Generate and save historical one-day GJR-GARCH variance forecasts."""

from __future__ import annotations

import pandas as pd

from src.evaluation_config import load_evaluation_config
from src.models.gjr_garch import gjr_garch_variance_forecasts


def main() -> None:
    """Load returns, generate forecasts, and save them."""
    evaluation = load_evaluation_config()
    features = pd.read_parquet(
        "data/processed/volatility_features.parquet"
    )

    forecasts = gjr_garch_variance_forecasts(
        features["log_return"].dropna(),
        minimum_training_observations=evaluation[
            "minimum_training_observations"
        ],
        reestimate_every=evaluation["reestimate_every"],
    )

    output_path = "data/processed/gjr_garch_forecasts.parquet"
    forecasts.to_frame().to_parquet(output_path)

    available = forecasts.dropna()
    print(f"Forecasts saved to {output_path}")
    print(f"Available forecasts: {len(available)}")
    if not available.empty:
        print(
            f"Forecast period: {available.index.min().date()} "
            f"to {available.index.max().date()}"
        )
        print(f"Minimum variance: {available.min():.10f}")
        print(f"Maximum variance: {available.max():.10f}")


if __name__ == "__main__":
    main()
