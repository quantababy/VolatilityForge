from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.run_strategy_report import (
    STRATEGY_FILES,
    generate_strategy_report,
)


def make_strategy_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=5)
    next_dates = dates[1:].append(
        pd.DatetimeIndex([dates[-1] + pd.offsets.BDay(1)])
    )

    returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01])
    positions = np.array([0.5, 0.6, 0.4, 0.7, 0.5])
    turnover = np.abs(
        positions - np.concatenate(([0.0], positions[:-1]))
    )
    costs = turnover * 5.0 / 10_000.0
    gross = positions * returns
    net = gross - costs

    return pd.DataFrame(
        {
            "next_date": next_dates,
            "forecast_variance": np.full(5, 0.0004),
            "position": positions,
            "next_return": returns,
            "turnover": turnover,
            "transaction_cost": costs,
            "gross_return": gross,
            "net_return": net,
        },
        index=dates,
    ).rename_axis("Date")


def write_all_strategy_files(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame = make_strategy_frame()

    for filename in STRATEGY_FILES.values():
        frame.to_parquet(processed_dir / filename)


def test_generates_report_for_all_strategies_and_benchmark(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    report_dir = tmp_path / "reports"
    write_all_strategy_files(processed_dir)

    report = generate_strategy_report(
        processed_dir=processed_dir,
        report_dir=report_dir,
    )

    assert len(report) == len(STRATEGY_FILES) * 2 + 1
    assert report["n_observations"].eq(5).all()
    assert report["is_buy_and_hold"].sum() == 1
    assert (report_dir / "strategy_performance.csv").is_file()
    assert "cost_convention" in report.columns
    assert "cumulative_transaction_cost_return_units" in report.columns
    assert report.loc[
        ~report["is_buy_and_hold"], "cost_convention"
    ].eq("turnover_based_cost_deducted_from_net_returns").all()


def test_buy_and_hold_uses_same_period_realized_returns(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    write_all_strategy_files(processed_dir)

    report = generate_strategy_report(
        processed_dir=processed_dir,
        report_dir=tmp_path / "reports",
    )
    benchmark = report.loc[report["is_buy_and_hold"]].iloc[0]

    expected_return = (1.01 * 0.98 * 1.015 * 1.005 * 0.99) - 1.0

    assert benchmark["model"] == "Buy-and-hold (1x, gross)"
    assert benchmark["total_return"] == pytest.approx(expected_return)
    assert benchmark["average_exposure"] == pytest.approx(1.0)
    assert (
        benchmark["cumulative_transaction_cost_return_units"]
        == pytest.approx(0.0)
    )
    assert benchmark["cost_convention"] == "excluded_gross_benchmark"


def test_rejects_missing_strategy_file(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    report_dir = tmp_path / "reports"

    with pytest.raises(FileNotFoundError, match="Missing strategy output"):
        generate_strategy_report(
            processed_dir=processed_dir,
            report_dir=report_dir,
        )

def test_rejects_misaligned_model_dates(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    write_all_strategy_files(processed_dir)

    path = processed_dir / STRATEGY_FILES["GARCH"]
    frame = pd.read_parquet(path)

    shifted_index = frame.index + pd.Timedelta(days=1)
    frame.index = shifted_index
    frame["next_date"] = (
        shifted_index.to_series().shift(-1).to_numpy()
    )
    frame.iloc[-1, frame.columns.get_loc("next_date")] = (
        shifted_index[-1] + pd.offsets.BDay(1)
    )
    frame.to_parquet(path)

    with pytest.raises(ValueError, match="forecast dates differ"):
        generate_strategy_report(
            processed_dir=processed_dir,
            report_dir=tmp_path / "reports",
        )


def test_rejects_inconsistent_realized_returns(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    write_all_strategy_files(processed_dir)

    path = processed_dir / STRATEGY_FILES["GARCH"]
    frame = pd.read_parquet(path)
    frame.iloc[0, frame.columns.get_loc("next_return")] += 0.01
    frame.to_parquet(path)

    with pytest.raises(ValueError, match="realized returns differ"):
        generate_strategy_report(
            processed_dir=processed_dir,
            report_dir=tmp_path / "reports",
        )