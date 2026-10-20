"""Generate strategy sensitivity results across volatility targets and costs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.strategy_backtest import run_volatility_target_backtest
from src.strategy_metrics import calculate_strategy_metrics

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "volatility_features.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "strategy_sensitivity.csv"

FORECAST_FILES = {
    "ewma": ("ewma_forecasts.parquet", "ewma_variance_1d"),
    "garch": ("garch_forecasts.parquet", "garch_variance_1d"),
    "gjr_garch": ("gjr_garch_forecasts.parquet", "gjr_garch_variance_1d"),
    "har_rv": ("har_rv_forecasts.parquet", "har_rv_variance_1d"),
    "lightgbm": ("lightgbm_forecasts.parquet", "lightgbm_variance_1d"),
}

TARGET_VOLATILITIES = (0.08, 0.10, 0.12)
TRANSACTION_COSTS_BPS = (0.0, 5.0, 10.0, 20.0)


def generate_strategy_sensitivity(
    config_path: Path = CONFIG_PATH,
    features_path: Path = FEATURES_PATH,
    processed_dir: Path = PROCESSED_DIR,
    report_path: Path = REPORT_PATH,
) -> pd.DataFrame:
    """Run all model/target/cost combinations and write a CSV report."""
    with config_path.open("r", encoding="utf-8") as file:
        config: Any = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("Configuration must be a YAML mapping.")

    strategy = config.get("strategy")
    forecast_config = config.get("forecast")
    evaluation = config.get("evaluation")

    if not isinstance(strategy, dict):
        raise TypeError("Configuration must contain a strategy mapping.")
    if not isinstance(forecast_config, dict):
        raise TypeError("Configuration must contain a forecast mapping.")
    if not isinstance(evaluation, dict):
        raise TypeError("Configuration must contain an evaluation mapping.")

    max_leverage = float(strategy["max_leverage"])
    development_start = evaluation.get("development_start")
    trading_days = forecast_config.get("trading_days_per_year")

    if not isinstance(development_start, str) or not development_start:
        raise ValueError("evaluation.development_start must be a date string.")
    if (
        not isinstance(trading_days, int)
        or isinstance(trading_days, bool)
        or trading_days < 1
    ):
        raise ValueError(
            "forecast.trading_days_per_year must be a positive integer."
        )
    if not np.isfinite(max_leverage) or max_leverage <= 0:
        raise ValueError("strategy.max_leverage must be finite and positive.")

    features = pd.read_parquet(features_path)
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("Feature data must have a DatetimeIndex.")
    if "Close" not in features.columns:
        raise ValueError("Feature data must contain a Close column.")
    if features.index.has_duplicates or not features.index.is_monotonic_increasing:
        raise ValueError("Feature dates must be unique and sorted.")

    features = features.loc[
        features.index >= pd.Timestamp(development_start)
    ].copy()
    if features.empty:
        raise ValueError("No feature observations in the development period.")

    rows: list[dict[str, Any]] = []

    for model_name, (filename, forecast_column) in FORECAST_FILES.items():
        forecast_path = processed_dir / filename
        if not forecast_path.is_file():
            raise FileNotFoundError(f"Missing forecast file: {forecast_path}")

        forecast_frame = pd.read_parquet(forecast_path)
        if not isinstance(forecast_frame.index, pd.DatetimeIndex):
            raise TypeError(f"{model_name} forecasts must have a DatetimeIndex.")
        if forecast_column not in forecast_frame.columns:
            raise ValueError(
                f"{forecast_column!r} missing from {forecast_path}."
            )

        forecasts = forecast_frame[forecast_column]
        forecasts = forecasts.loc[
            forecasts.index >= pd.Timestamp(development_start)
        ].dropna()

        if forecasts.empty:
            raise ValueError(
                f"No valid {model_name} forecasts from {development_start}."
            )

        for target_volatility in TARGET_VOLATILITIES:
            for cost_bps in TRANSACTION_COSTS_BPS:
                result = run_volatility_target_backtest(
                    features=features,
                    forecasts=forecasts,
                    target_annual_volatility=target_volatility,
                    max_leverage=max_leverage,
                    transaction_cost_bps=cost_bps,
                    trading_days_per_year=trading_days,
                )

                metrics = calculate_strategy_metrics(
                    result["net_return"],
                    trading_days_per_year=trading_days,
                )

                rows.append(
                    {
                        "model": model_name,
                        "target_annual_volatility": target_volatility,
                        "max_leverage": max_leverage,
                        "transaction_cost_bps": cost_bps,
                        "evaluation_start": str(
                            pd.Timestamp(result["next_date"].min()).date()
                        ),
                        "evaluation_end": str(
                            pd.Timestamp(result["next_date"].max()).date()
                        ),
                        "n_observations": len(result),
                        "total_return_net": metrics["total_return"],
                        "annualized_return_net": metrics["annualized_return"],
                        "annualized_volatility_net": (
                            metrics["annualized_volatility"]
                        ),
                        "sharpe_ratio_zero_rf_net": (
                            metrics["sharpe_ratio_zero_rf"]
                        ),
                        "maximum_drawdown_net": metrics["maximum_drawdown"],
                        "average_exposure": float(result["position"].mean()),
                        "average_turnover": float(result["turnover"].mean()),
                        "cumulative_cost_return_units": float(
                            result["transaction_cost"].sum()
                        ),
                    }
                )

    report = pd.DataFrame(rows)
    expected_rows = (
        len(FORECAST_FILES)
        * len(TARGET_VOLATILITIES)
        * len(TRANSACTION_COSTS_BPS)
    )
    if len(report) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} sensitivity rows; got {len(report)}."
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    LOGGER.info("Saved %d sensitivity rows to %s", len(report), report_path)
    print(f"Saved sensitivity report: {report_path}")
    print(f"Rows: {len(report)}")
    print(f"Models: {report['model'].nunique()}")
    print(f"Observations per combination: {report['n_observations'].min()}–"
          f"{report['n_observations'].max()}")

    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_strategy_sensitivity()


if __name__ == "__main__":
    main()

