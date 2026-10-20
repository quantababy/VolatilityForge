
"""Build raw, aligned, and engineered VolatilityForge datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.data_alignment import align_spy_vix
from src.data_loader import download_market_data, save_market_data
from src.feature_engineering import build_volatility_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _report_dataset(name: str, data: pd.DataFrame, path: Path) -> None:
    """Print basic dataset metadata after saving."""
    print(f"\n{name}")
    print(f"  Path: {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")
    print(f"  Rows: {len(data):,}")
    print(f"  Columns: {len(data.columns)}")
    print(f"  Date range: {data.index.min().date()} to {data.index.max().date()}")


def run_pipeline(project_root: Path | None = None) -> dict[str, Path]:
    """Download market data and rebuild all pipeline datasets."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    config_path = root / "configs" / "config.yaml"

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_config = config["data"]
    if data_config.get("cache_format") != "parquet":
        raise ValueError("This pipeline currently supports only Parquet output.")

    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    start_date = data_config["start_date"]
    end_date = data_config.get("end_date")

    spy = download_market_data(
        data_config["primary_asset"],
        start_date,
        end_date,
    )
    vix = download_market_data(
        data_config["implied_volatility_asset"],
        start_date,
        end_date,
    )

    paths = {
        "spy": raw_dir / "spy.parquet",
        "vix": raw_dir / "vix.parquet",
        "aligned": processed_dir / "spy_vix.parquet",
        "features": processed_dir / "volatility_features.parquet",
    }

    # Save validated source datasets before alignment.
    save_market_data(spy, str(paths["spy"]))
    save_market_data(vix, str(paths["vix"]))

    aligned = align_spy_vix(spy, vix)
    aligned.to_parquet(paths["aligned"], index=True)

    features = build_volatility_features(aligned)
    features.to_parquet(paths["features"], index=True)

    for name, data, path in (
        ("SPY raw data", spy, paths["spy"]),
        ("VIX raw data", vix, paths["vix"]),
        ("Aligned SPY/VIX data", aligned, paths["aligned"]),
        ("Engineered volatility features", features, paths["features"]),
    ):
        _report_dataset(name, data, path)

    print("\nData pipeline completed successfully.")
    return paths


def main() -> None:
    """Run the data pipeline from the repository root."""
    run_pipeline()


if __name__ == "__main__":
    main()
