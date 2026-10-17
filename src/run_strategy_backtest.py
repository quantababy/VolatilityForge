
"""Run volatility-targeting backtests for one-day variance forecasts."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from src.strategy_backtest import run_volatility_target_backtest

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "volatility_features.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

FORECAST_FILES = {
    "ewma": ("ewma_forecasts.parquet", "ewma_variance_1d"),
    "garch": ("garch_forecasts.parquet", "garch_variance_1d"),
    "gjr_garch": ("gjr_garch_forecasts.parquet", "gjr_garch_variance_1d"),
    "har_rv": ("har_rv_forecasts.parquet", "har_rv_variance_1d"),
    "lightgbm": ("lightgbm_forecasts.parquet", "lightgbm_variance_1d"),
}


def load_strategy_config() -> tuple[dict, str, int]:
    """Load strategy settings and the configured development start date."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("Configuration must be a YAML mapping.")

    strategy = config.get("strategy")
    forecast = config.get("forecast")
    evaluation = config.get("evaluation")

    if not isinstance(strategy, dict):
        raise TypeError("Configuration must contain a strategy mapping.")
    if not isinstance(forecast, dict):
        raise TypeError("Configuration must contain a forecast mapping.")
    if not isinstance(evaluation, dict):
        raise TypeError("Configuration must contain an evaluation mapping.")

    required = {
        "target_annual_volatility",
        "max_leverage",
        "transaction_cost_bps",
    }
    missing = required.difference(strategy)
    if missing:
        raise ValueError(
            f"Missing strategy configuration keys: {sorted(missing)}"
        )

    trading_days = forecast.get("trading_days_per_year")
    if (
        not isinstance(trading_days, int)
        or isinstance(trading_days, bool)
        or trading_days < 1
    ):
        raise ValueError(
            "forecast.trading_days_per_year must be a positive integer."
        )

    development_start = evaluation.get("development_start")
    if not isinstance(development_start, str) or not development_start:
        raise ValueError(
            "evaluation.development_start must be a nonempty date string."
        )

    return strategy, development_start, trading_days


def run_all_strategy_backtests() -> dict[str, pd.DataFrame]:
    """Run each strategy and save its daily returns and exposures."""
    strategy, development_start, trading_days = load_strategy_config()

    features = pd.read_parquet(FEATURES_PATH)
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("Feature data must have a DatetimeIndex.")

    if "Close" not in features.columns:
        raise ValueError("Feature data must contain a Close column.")

    development_features = features.loc[
        features.index >= pd.Timestamp(development_start)
    ].copy()

    if development_features.empty:
        raise ValueError(
            f"No feature observations found from {development_start} onward."
        )

    results: dict[str, pd.DataFrame] = {}

    for model_name, (filename, forecast_column) in FORECAST_FILES.items():
        forecast_path = OUTPUT_DIR / filename
        forecasts_frame = pd.read_parquet(forecast_path)

        if forecast_column not in forecasts_frame.columns:
            raise ValueError(
                f"{forecast_column!r} not found in {forecast_path}."
            )
        if not isinstance(forecasts_frame.index, pd.DatetimeIndex):
            raise TypeError(
                f"{model_name} forecasts must have a DatetimeIndex."
            )

        forecasts = forecasts_frame[forecast_column]
        forecasts = forecasts.loc[
            forecasts.index >= pd.Timestamp(development_start)
        ]

        # Remove unavailable warm-up forecasts. Any remaining gaps are
        # rejected by the backtest instead of being silently skipped.
        forecasts = forecasts.dropna()

        if forecasts.empty:
            raise ValueError(
                f"No valid {model_name} forecasts from {development_start}."
            )

        model_results = run_volatility_target_backtest(
            features=development_features,
            forecasts=forecasts,
            target_annual_volatility=float(
                strategy["target_annual_volatility"]
            ),
            max_leverage=float(strategy["max_leverage"]),
            transaction_cost_bps=float(strategy["transaction_cost_bps"]),
            trading_days_per_year=trading_days,
        )

        output_path = OUTPUT_DIR / f"strategy_{model_name}.parquet"
        model_results.to_parquet(output_path)
        results[model_name] = model_results

        LOGGER.info(
            "%s: saved %d daily observations (%s to %s) to %s",
            model_name,
            len(model_results),
            model_results.index.min().date(),
            model_results.index.max().date(),
            output_path,
        )

    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    results = run_all_strategy_backtests()

    print("\nStrategy backtests completed:")
    for model_name, frame in results.items():
        print(
            f"{model_name:10s} "
            f"observations={len(frame):4d} "
            f"from={frame.index.min().date()} "
            f"to={frame.index.max().date()}"
        )


if __name__ == "__main__":
    main()
