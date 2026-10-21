
import numpy as np
import pandas as pd
import pytest

from src import compare_five_day_forecasts, compare_forecasts


@pytest.mark.parametrize(
    ("module", "expected_models", "expected_lag"),
    [
        (compare_forecasts, 6, 5),
        (compare_five_day_forecasts, 6, 4),
    ],
)
def test_calibration_report_contains_each_model(
    module,
    expected_models,
    expected_lag,
):
    dates = pd.bdate_range("2022-01-03", periods=80)
    forecast = pd.Series(
        np.linspace(0.01, 0.08, len(dates)),
        index=dates,
    )
    frame = pd.DataFrame(
        {"target": 0.002 + 1.2 * forecast},
        index=dates,
    )

    for i, model in enumerate(module.FORECAST_FILES):
        frame[model] = forecast + i * 0.001

    report = module.build_calibration_report(frame)

    assert len(report) == expected_models
    assert report["model"].tolist() == list(module.FORECAST_FILES)
    assert report["n_observations"].eq(len(dates)).all()
    assert report["hac_lags"].eq(expected_lag).all()
    assert report["start_date"].eq(str(dates.min().date())).all()
    assert report["end_date"].eq(str(dates.max().date())).all()
    assert np.isfinite(report["intercept"]).all()
    assert np.isfinite(report["slope"]).all()
    assert np.isfinite(report["joint_p_value"]).all()


@pytest.mark.parametrize(
    "module",
    [compare_forecasts, compare_five_day_forecasts],
)
def test_calibration_report_rejects_insufficient_sample(module):
    dates = pd.bdate_range("2024-01-01", periods=3)
    frame = pd.DataFrame(
        {"target": [0.01, 0.02, 0.03]},
        index=dates,
    )

    for model in module.FORECAST_FILES:
        frame[model] = [0.02, 0.03, 0.04]

    with pytest.raises(ValueError, match="Insufficient observations"):
        module.build_calibration_report(frame)


@pytest.mark.parametrize(
    ("module", "target_column"),
    [
        (compare_forecasts, "target_rv_1"),
        (compare_five_day_forecasts, "target_rv_5"),
    ],
)
def test_calibration_report_uses_the_comparison_sample(
    monkeypatch,
    module,
    target_column,
):
    dates = pd.bdate_range("2022-01-03", periods=17)
    target = pd.Series(
        [np.nan, 0.0, *[i / 1000 for i in range(1, 15)], np.nan],
        index=dates,
        name=target_column,
    )
    features = pd.DataFrame({target_column: target})

    monkeypatch.setattr(
        module,
        "load_evaluation_config",
        lambda: {"development_start": "2022-01-01"},
    )

    def fake_read_parquet(path):
        if path == "data/processed/volatility_features.parquet":
            return features

        column = next(
            col
            for _, (file_path, col) in module.FORECAST_FILES.items()
            if file_path == path
        )
        forecasts = np.linspace(0.01, 0.03, len(dates))
        return pd.DataFrame({column: forecasts}, index=dates)

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    mse_aligned, _ = module.build_comparison_frame()
    report = module.build_calibration_report(mse_aligned)

    assert report["n_observations"].eq(len(mse_aligned)).all()
    assert report["start_date"].eq(
        str(mse_aligned.index.min().date())
    ).all()
    assert report["end_date"].eq(
        str(mse_aligned.index.max().date())
    ).all()
    assert dates[1] in mse_aligned.index  # Zero target remains eligible.
    assert dates[0] not in mse_aligned.index
    assert dates[-1] not in mse_aligned.index
