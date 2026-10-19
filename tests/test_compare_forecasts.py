
import numpy as np
import pandas as pd
import pytest

from src import compare_five_day_forecasts, compare_forecasts


@pytest.mark.parametrize(
    ("module", "target_column"),
    [
        (compare_forecasts, "target_rv_1"),
        (compare_five_day_forecasts, "target_rv_5"),
    ],
)
def test_comparison_aligns_dates_and_filters_invalid_values(
    monkeypatch, module, target_column
):
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    target = pd.Series(
        [1.0, 0.0, 2.0, 3.0, np.nan, 4.0],
        index=dates,
        name=target_column,
    )
    features = pd.DataFrame({target_column: target})

    monkeypatch.setattr(
        module,
        "load_evaluation_config",
        lambda: {"development_start": "2024-01-01"},
    )

    def fake_read_parquet(path):
        if path == "data/processed/volatility_features.parquet":
            return features

        # Deliberately omit one date to test reindexing.
        forecast_dates = dates.delete(2)
        column = next(
            column
            for _, (file_path, column) in module.FORECAST_FILES.items()
            if file_path == path
        )
        values = pd.Series(1.0, index=forecast_dates, name=column)

        # Include invalid values on dates that should be excluded.
        values.loc[dates[3]] = np.inf
        return values.to_frame()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    mse_frame, losses = module.build_comparison_frame()

    # Date 2 is absent from every forecast file; date 3 has an
    # infinite forecast; date 4 has a missing target.
    assert list(mse_frame.index) == [dates[0], dates[1], dates[5]]
    assert list(losses.index) == [dates[0], dates[5]]
    assert (mse_frame[list(module.FORECAST_FILES)] > 0).all().all()
    assert np.isfinite(losses.to_numpy()).all()


@pytest.mark.parametrize(
    ("module", "target_column", "expected_error"),
    [
        (
            compare_forecasts,
            "target_rv_1",
            "No common valid observations are available for MSE",
        ),
        (
            compare_five_day_forecasts,
            "target_rv_5",
            "No common valid five-day observations for MSE",
        ),
    ],
)
def test_comparison_rejects_empty_common_sample(
    monkeypatch, module, target_column, expected_error
):
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    features = pd.DataFrame(
        {target_column: [1.0, 2.0, 3.0]},
        index=dates,
    )

    monkeypatch.setattr(
        module,
        "load_evaluation_config",
        lambda: {"development_start": "2024-01-01"},
    )

    def fake_read_parquet(path):
        if path == "data/processed/volatility_features.parquet":
            return features

        column = next(
            column
            for _, (file_path, column) in module.FORECAST_FILES.items()
            if file_path == path
        )
        return pd.DataFrame(
            {column: [np.nan, np.nan, np.nan]},
            index=dates,
        )

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    with pytest.raises(ValueError, match=expected_error):
        module.build_comparison_frame()


@pytest.mark.parametrize(
    ("module", "target_column"),
    [
        (compare_forecasts, "target_rv_1"),
        (compare_five_day_forecasts, "target_rv_5"),
    ],
)
@pytest.mark.parametrize("invalid_forecast", [0.0, -1.0])
def test_comparison_excludes_nonpositive_forecasts(
    monkeypatch, module, target_column, invalid_forecast
):
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    features = pd.DataFrame(
        {target_column: [1.0, 2.0, 3.0, 4.0]},
        index=dates,
    )

    monkeypatch.setattr(
        module,
        "load_evaluation_config",
        lambda: {"development_start": "2024-01-01"},
    )

    def fake_read_parquet(path):
        if path == "data/processed/volatility_features.parquet":
            return features

        column = next(
            column
            for _, (file_path, column) in module.FORECAST_FILES.items()
            if file_path == path
        )
        values = pd.Series(1.0, index=dates, name=column)
        values.loc[dates[1]] = invalid_forecast
        return values.to_frame()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    mse_frame, losses = module.build_comparison_frame()

    # Every model has an invalid forecast on dates[1], so that date
    # must be removed from both evaluation samples.
    assert dates[1] not in mse_frame.index
    assert dates[1] not in losses.index
    assert len(mse_frame) == 3
    assert len(losses) == 3

def test_five_day_hac_lags_have_pre_specified_primary_and_sensitivities():
    assert compare_five_day_forecasts.PRIMARY_HAC_LAG == 4
    assert compare_five_day_forecasts.HAC_SENSITIVITY_LAGS == (5, 10, 22)
    assert compare_five_day_forecasts.HAC_LAGS == (4, 5, 10, 22)

def test_five_day_comparison_excludes_immature_targets(
    monkeypatch,
) -> None:
    module = compare_five_day_forecasts
    dates = pd.date_range("2024-01-01", periods=6, freq="D")

    # The last five-day target is immature at the final forecast origin.
    features = pd.DataFrame(
        {"target_rv_5": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]},
        index=dates,
    )

    monkeypatch.setattr(
        module,
        "load_evaluation_config",
        lambda: {"development_start": "2024-01-01"},
    )

    def fake_read_parquet(path):
        if path == "data/processed/volatility_features.parquet":
            return features

        column = next(
            column
            for _, (file_path, column) in module.FORECAST_FILES.items()
            if file_path == path
        )
        return pd.DataFrame({column: 1.0}, index=dates)

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    mse_frame, losses = module.build_comparison_frame()

    assert dates[-1] not in mse_frame.index
    assert dates[-1] not in losses.index
    assert len(mse_frame) == 5
    assert len(losses) == 5