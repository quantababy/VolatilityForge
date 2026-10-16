
"""Generate and save five-day GARCH cumulative variance forecasts."""

from __future__ import annotations

import pandas as pd

from src.models.garch import garch_five_day_variance_forecasts


def main() -> None:
    """Load returns, generate five-day forecasts, and save results."""
    features = pd.read_parquet(
        "data/processed/volatility_features.parquet"
    )

    if "log_return" not in features.columns:
        raise ValueError(
            "volatility_features.parquet must contain 'log_return'."
        )

    returns = features["log_return"].dropna()

    forecasts = garch_five_day_variance_forecasts(
        returns,
        minimum_training_observations=252,
        reestimate_every=21,
    )

    output_path = "data/processed/garch_five_day_forecasts.parquet"
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
