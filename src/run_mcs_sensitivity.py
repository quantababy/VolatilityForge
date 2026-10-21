"""Check MCS membership sensitivity to bootstrap block size."""
from __future__ import annotations

import pandas as pd

from src.compare_five_day_forecasts import (
    build_comparison_frame as build_five_day,
)
from src.compare_forecasts import build_comparison_frame as build_one_day
from src.run_mcs_reports import build_mcs_report


def main() -> None:
    """Generate MCS sensitivity results for both forecast horizons."""
    _, one_day_losses = build_one_day()
    _, five_day_losses = build_five_day()

    rows = []
    for horizon, losses in (
        (1, one_day_losses),
        (5, five_day_losses),
    ):
        for block_size in (5, 10, 22):
            report = build_mcs_report(
                losses,
                horizon=horizon,
                bootstrap_reps=500,
                block_size=block_size,
                seed=42,
            )
            rows.append(report)
            included = report.loc[
                report["in_confidence_set"], "model"
            ].tolist()
            print(
                f"Horizon={horizon}d | block={block_size} | "
                f"observations={len(losses)} | MCS={included}"
            )

    result = pd.concat(rows, ignore_index=True)
    output_path = "reports/mcs_sensitivity.csv"
    result.to_csv(output_path, index=False, float_format="%.12g")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
