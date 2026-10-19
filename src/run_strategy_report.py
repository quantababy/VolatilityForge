"""Generate comparable performance reports for saved strategy backtests."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.strategy_metrics import calculate_strategy_metrics

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

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

    if frame["next_date"].isna().any():
        raise ValueError(f"{name} contains missing next-session dates.")

    if not (frame["next_date"].to_numpy() > frame.index.to_numpy()).all():
        raise ValueError(
            f"{name} has next-session dates that are not after forecast dates."
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
    benchmark: bool = False,
) -> dict[str, str | float | int]:
    """Combine return-series metrics with strategy-specific diagnostics."""
    returns = frame[returns_column]
    metrics = calculate_strategy_metrics(
        returns,
        trading_days_per_year=trading_days_per_year,
    )

    metrics.update(
        {
            "model": name,
            "return_basis": returns_column,
            "average_exposure": float(frame["position"].mean()),
            "average_turnover": float(frame["turnover"].mean()),
            "total_transaction_cost": float(
                frame["transaction_cost"].sum()
            ),
            "is_buy_and_hold": benchmark,
        }
    )
    return metrics


def generate_strategy_report(
    processed_dir: Path = PROCESSED_DIR,
    report_dir: Path = REPORT_DIR,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """Create net/gross strategy summaries and a same-period buy-and-hold row.

    The buy-and-hold row uses constant 1x exposure to the same realized
    returns. It reports gross performance and excludes transaction costs,
    so its cost convention is explicit rather than silently assumed equal.
    """
    if (
        not isinstance(trading_days_per_year, int)
        or isinstance(trading_days_per_year, bool)
        or trading_days_per_year < 1
    ):
        raise ValueError("trading_days_per_year must be a positive integer.")

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

    rows: list[dict[str, str | float | int]] = []

    for name, frame in frames.items():
        for basis in ("gross_return", "net_return"):
            rows.append(
                _summarize_strategy(
                    name=f"{name} ({basis})",
                    frame=frame,
                    returns_column=basis,
                    trading_days_per_year=trading_days_per_year,
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
            benchmark=True,
        )
    )

    report = pd.DataFrame(rows)
    report = report[
        [
            "model",
            "return_basis",
            "is_buy_and_hold",
            "n_observations",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio_zero_rf",
            "maximum_drawdown",
            "average_exposure",
            "average_turnover",
            "total_transaction_cost",
        ]
    ]

    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "strategy_performance.csv"
    report.to_csv(output_path, index=False)

    LOGGER.info("Saved strategy report to %s", output_path)
    print(f"Saved report: {output_path}")
    print(f"Evaluation observations per model: {len(reference)}")
    print(report.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_strategy_report()