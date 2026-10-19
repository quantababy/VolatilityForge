
import math

import pandas as pd
import pytest

from src.run_prospective_forecasts import _forecast_record


@pytest.mark.parametrize(
    "variance",
    [float("nan"), float("inf"), float("-inf")],
)
def test_forecast_record_rejects_nonfinite_variance(variance):
    with pytest.raises(ValueError, match="invalid forecast"):
        _forecast_record(
            model="EWMA",
            horizon_days=1,
            variance=variance,
            forecast_date=pd.Timestamp("2026-10-19"),
            generated_at_utc="2026-10-20T00:00:00+00:00",
        )


def test_forecast_record_accepts_finite_positive_variance():
    record = _forecast_record(
        model="EWMA",
        horizon_days=1,
        variance=0.0004,
        forecast_date=pd.Timestamp("2026-10-19"),
        generated_at_utc="2026-10-20T00:00:00+00:00",
    )

    assert math.isfinite(record["variance_forecast"])
    assert record["variance_forecast"] == pytest.approx(0.0004)
