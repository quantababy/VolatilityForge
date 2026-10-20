
"""Generate comparable performance reports for saved strategy backtests."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.strategy_metrics import calculate_strategy_metrics

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

STRATEGY_FILES = {
    "EWMA": "strategy_ewma.parquet",
    "GARCH": "strategy_garch.parquet",
    "GJR-GARCH": "strategy_gjr_garch.parquet",
    "HAR-RV": "strategy_har_rv.parquet",
    "LightGBM": "strategy_lightgbm.parquet",
}

REQUIRED_COLUMNS = {
    "next_date",
    "position",
    "next_return",
    "turnover",
    "transaction_cost",
    "gross_return",
    "net_return",
}

REQUIRED_STRATEGY_CONFIG_KEYS = {
    "target_annual_volatility",
    "max_leverage",
    "transaction_cost_bps",
}

DEFAULT_STRATEGY_CONFIG = {
    "target_annual_volatility": 0.10,
    "max_leverage": 1.0,
    "transaction_cost_bps": 5.0,
}


def load_project_config(config_path: Path = CONFIG_PATH) -> tuple[
    dict[str, Any], int
]:
    """Load and validate strategy settings and annual trading-day count."""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("Configuration must be a YAML mapping.")

    strategy_config = config.get("strategy")
    forecast_config = config.get("forecast")

    if not isinstance(strategy_config, dict):
        raise TypeError("Configuration must contain a strategy mapping.")

    if not isinstance(forecast_config, dict):
        raise TypeError("Configuration must contain a forecast mapping.")

    trading_days = forecast_config.get("trading_days_per_year")

    if (
        not isinstance(trading_days, int)
        or isinstance(trading_days, bool)
        or trading_days < 1
    ):
        raise ValueError(
            "forecast.trading_days_per_year must be a positive integer."
        )

    _validate_strategy_config(strategy_config)
    return strategy_config, trading_days


def _validate_strategy_config(
    strategy_config: dict[str, Any],
) -> dict[str, float]:
    """Validate and normalize the assumptions used in the strategy report."""
    missing = REQUIRED_STRATEGY_CONFIG_KEYS.difference(strategy_config)

    if missing:
        raise ValueError(
            f"Missing strategy configuration keys: {sorted(missing)}"
        )

    try:
        target_volatility = float(
            strategy_config["target_annual_volatility"]
        )
        leverage_cap = float(strategy_config["max_leverage"])
        transaction_cost_bps = float(
            strategy_config["transaction_cost_bps"]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Strategy configuration values must be numeric."
        ) from exc

    if (
        not np.isfinite(target_volatility)
        or target_volatility <= 0
    ):
        raise ValueError(
            "target_annual_volatility must be finite and positive."
        )

    if not np.isfinite(leverage_cap) or leverage_cap <= 0:
        raise ValueError("max_leverage must be finite and positive.")

    if (
        not np.isfinite(transaction_cost_bps)
        or transaction_cost_bps < 0
    ):
        raise ValueError(
            "transaction_cost_bps must be finite and nonnegative."
        )

    return {
        "target_annual_volatility": target_volatility,
        "max_leverage": leverage_cap,
        "transaction_cost_bps": transaction_cost_bps,
    }


def _validate_strategy_frame(
    name: str,
    frame: pd.DataFrame,
) -> None:
    """Validate the saved daily strategy output before reporting."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing)}"
        )

    if frame.empty:
        raise ValueError(f"{name} has no backtest observations.")

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex.")

    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} dates must be unique and sorted.")

    next_dates = pd.DatetimeIndex(frame["next_date"])

    if next_dates.isna().any():
        raise ValueError(f"{name} contains missing next-session dates.")

    if not (next_dates > frame.index).all():
        raise ValueError(
            f"{name} has next-session dates that are not after "
            "forecast dates."
        )

    numeric_columns = [
        "position",
        "next_return",
        "turnover",
        "transaction_cost",
        "gross_return",
        "net_return",
    ]
    values = frame[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite strategy values.")

    if (frame["position"] < 0).any():
        raise ValueError(f"{name} contains negative positions.")

    if (frame["turnover"] < 0).any():
        raise ValueError(f"{name} contains negative turnover.")

    if (frame["transaction_cost"] < 0).any():
        raise ValueError(f"{name} contains negative transaction costs.")

    expected_net = frame["gross_return"] - frame["transaction_cost"]

    if not np.allclose(
        frame["net_return"].to_numpy(dtype=float),
        expected_net.to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            f"{name} net returns do not equal gross returns minus costs."
        )


def _summarize_strategy(
    name: str,
    frame: pd.DataFrame,
    returns_column: str,
    trading_days_per_year: int,
    strategy_config: dict[str, float],
    benchmark: bool = False,
) -> dict[str, str | float | int | bool]:
    """Combine return metrics, period metadata, and strategy assumptions."""
    returns = frame[returns_column]

    metrics = calculate_strategy_metrics(
        returns,
        trading_days_per_year=trading_days_per_year,
    )

    metrics.update(
        {
            "model": name,
            "return_basis": returns_column,
            "is_buy_and_hold": benchmark,
            "n_observations": len(frame),
            "forecast_origin_start_date": str(frame.index.min().date()),
            "forecast_origin_end_date": str(frame.index.max().date()),
            "realized_return_start_date": str(
                pd.Timestamp(frame["next_date"].min()).date()
            ),
            "realized_return_end_date": str(
                pd.Timestamp(frame["next_date"].max()).date()
            ),
            "average_exposure": float(frame["position"].mean()),
            "average_turnover": float(frame["turnover"].mean()),
            "cumulative_transaction_cost_return_units": float(
                frame["transaction_cost"].sum()
            ),
            "cost_convention": (
                "excluded_gross_benchmark"
                if benchmark
                else "turnover_based_cost_deducted_from_net_returns"
            ),
            "target_annual_volatility": (
                strategy_config["target_annual_volatility"]
            ),
            "max_leverage": strategy_config["max_leverage"],
            "transaction_cost_bps": (
                strategy_config["transaction_cost_bps"]
            ),
        }
    )

    return metrics


def generate_strategy_report(
    processed_dir: Path = PROCESSED_DIR,
    report_dir: Path = REPORT_DIR,
    trading_days_per_year: int = 252,
    strategy_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Create comparable strategy summaries and a same-period benchmark.

    The buy-and-hold benchmark uses constant 1x exposure to the same
    realized returns. It reports gross performance and excludes transaction
    costs. Strategy assumptions are included as explicit report metadata.
    """
    if (
        not isinstance(trading_days_per_year, int)
        or isinstance(trading_days_per_year, bool)
        or trading_days_per_year < 1
    ):
        raise ValueError("trading_days_per_year must be a positive integer.")

    if strategy_config is None:
        strategy_config = DEFAULT_STRATEGY_CONFIG.copy()

    validated_strategy_config = _validate_strategy_config(strategy_config)

    frames: dict[str, pd.DataFrame] = {}

    for name, filename in STRATEGY_FILES.items():
        path = processed_dir / filename

        if not path.is_file():
            raise FileNotFoundError(f"Missing strategy output: {path}")

        frame = pd.read_parquet(path)
        _validate_strategy_frame(name, frame)
        frames[name] = frame

    reference_name = next(iter(frames))
    reference = frames[reference_name]

    for name, frame in frames.items():
        if not frame.index.equals(reference.index):
            raise ValueError(
                f"{name} forecast dates differ from {reference_name}."
            )

        if not frame["next_date"].equals(reference["next_date"]):
            raise ValueError(
                f"{name} next-session dates differ from {reference_name}."
            )

        if not np.allclose(
            frame["next_return"].to_numpy(dtype=float),
            reference["next_return"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{name} realized returns differ from {reference_name}."
            )

    rows: list[dict[str, Any]] = []

    for name, frame in frames.items():
        for basis in ("gross_return", "net_return"):
            rows.append(
                _summarize_strategy(
                    name=f"{name} ({basis})",
                    frame=frame,
                    returns_column=basis,
                    trading_days_per_year=trading_days_per_year,
                    strategy_config=validated_strategy_config,
                )
            )

    benchmark_frame = reference.copy()
    benchmark_frame["position"] = 1.0
    benchmark_frame["turnover"] = 0.0
    benchmark_frame["transaction_cost"] = 0.0
    benchmark_frame["gross_return"] = benchmark_frame["next_return"]
    benchmark_frame["net_return"] = benchmark_frame["next_return"]

    rows.append(
        _summarize_strategy(
            name="Buy-and-hold (1x, gross)",
            frame=benchmark_frame,
            returns_column="gross_return",
            trading_days_per_year=trading_days_per_year,
            strategy_config=validated_strategy_config,
            benchmark=True,
        )
    )

    report = pd.DataFrame(rows)

    report_columns = [
        "model",
        "return_basis",
        "is_buy_and_hold",
        "n_observations",
        "forecast_origin_start_date",
        "forecast_origin_end_date",
        "realized_return_start_date",
        "realized_return_end_date",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio_zero_rf",
        "maximum_drawdown",
        "average_exposure",
        "average_turnover",
        "cumulative_transaction_cost_return_units",
        "cost_convention",
        "target_annual_volatility",
        "max_leverage",
        "transaction_cost_bps",
    ]

    report = report[report_columns]

    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "strategy_performance.csv"
    report.to_csv(output_path, index=False)

    LOGGER.info("Saved strategy report to %s", output_path)
    print(f"Saved report: {output_path}")
    print(f"Evaluation observations per model: {len(reference)}")
    print(
        report.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    return report


def main() -> None:
    """Generate the report using the project's actual YAML configuration."""
    logging.basicConfig(level=logging.INFO)

    strategy_config, trading_days = load_project_config()

    generate_strategy_report(
        trading_days_per_year=trading_days,
        strategy_config=strategy_config,
    )


if __name__ == "__main__":
    main()
