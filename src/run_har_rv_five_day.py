
"""Generate five-day HAR-RV variance forecasts."""

from __future__ import annotations

import pandas as pd

from src.models.har_rv import har_rv_five_day_variance_forecasts


def main() -> None:
    """Generate and save five-day HAR-RV forecasts."""
    features = pd.read_parquet(
        "data/processed/volatility_features.parquet"
    )

    forecasts = har_rv_five_day_variance_forecasts(features)

    output_path = "data/processed/har_rv_five_day_forecasts.parquet"
    forecasts.to_frame().to_parquet(output_path)

    valid = forecasts.dropna()
    if valid.empty:
        raise ValueError("No valid five-day HAR-RV forecasts were generated.")

    print(f"Saved: {output_path}")
    print(f"Available forecasts: {len(valid)}")
    print(f"Date range: {valid.index.min().date()} to {valid.index.max().date()}")
    print(f"Minimum forecast variance: {valid.min():.10g}")
    print(f"Maximum forecast variance: {valid.max():.10g}")


if __name__ == "__main__":
    main()
