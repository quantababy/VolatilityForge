
"""Generate and save five-day GJR-GARCH variance forecasts."""

from __future__ import annotations

import pandas as pd

from src.evaluation_config import load_evaluation_config
from src.models.gjr_garch import gjr_garch_five_day_variance_forecasts


def main() -> None:
    """Load returns, generate five-day forecasts, and save results."""
    evaluation = load_evaluation_config()
    features = pd.read_parquet(
        "data/processed/volatility_features.parquet"
    )

    if "log_return" not in features.columns:
        raise ValueError(
            "volatility_features.parquet must contain 'log_return'."
        )

    returns = features["log_return"].dropna()

    forecasts = gjr_garch_five_day_variance_forecasts(
        returns,
        minimum_training_observations=evaluation["minimum_training_observations"],
        reestimate_every=evaluation["reestimate_every"],
    )

    output_path = (
        "data/processed/gjr_garch_five_day_forecasts.parquet"
    )
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
