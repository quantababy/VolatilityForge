import numpy as np
import pandas as pd
import pytest

from src import compare_five_day_forecasts as comparison


def setup_comparison_inputs(monkeypatch, forecast_index):
    dates = pd.bdate_range("2022-01-03", periods=8)

    features = pd.DataFrame(
        {"target_rv_5": [np.nan] * 5 + [0.01, 0.02, 0.03]},
        index=dates,
    )

    forecast_frames = {}
    for model, (_, column) in comparison.FORECAST_FILES.items():
        index = forecast_index if model == "EWMA" else dates
        forecast_frames[comparison.FORECAST_FILES[model][0]] = pd.DataFrame(
            {column: np.full(len(index), 0.015)},
            index=index,
        )

    original_read_parquet = pd.read_parquet

    def fake_read_parquet(path, *args, **kwargs):
        path_string = str(path)
        if path_string == "data/processed/volatility_features.parquet":
            return features
        if path_string in forecast_frames:
            return forecast_frames[path_string]
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(comparison, "load_evaluation_config", lambda: {
        "development_start": "2022-01-01"
    })
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)


def test_rejects_duplicate_forecast_dates(monkeypatch) -> None:
    dates = pd.bdate_range("2022-01-03", periods=8)
    duplicate_dates = dates.insert(3, dates[3])
    setup_comparison_inputs(monkeypatch, duplicate_dates)

    with pytest.raises(ValueError, match="EWMA forecast dates must be unique"):
        comparison.build_comparison_frame()


def test_rejects_unsorted_forecast_dates(monkeypatch) -> None:
    dates = pd.bdate_range("2022-01-03", periods=8)
    unsorted_dates = dates[[0, 2, 1, 3, 4, 5, 6, 7]]
    setup_comparison_inputs(monkeypatch, unsorted_dates)

    with pytest.raises(ValueError, match="EWMA forecast dates must be sorted"):
        comparison.build_comparison_frame()

def test_rejects_non_datetime_feature_index(monkeypatch) -> None:
    dates = pd.bdate_range("2022-01-03", periods=8)
    setup_comparison_inputs(monkeypatch, dates)

    features = pd.DataFrame(
        {"target_rv_5": [np.nan] * 5 + [0.01, 0.02, 0.03]},
        index=range(8),
    )

    original_read_parquet = pd.read_parquet

    def fake_read_parquet(path, *args, **kwargs):
        if str(path) == "data/processed/volatility_features.parquet":
            return features
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    with pytest.raises(TypeError, match="Feature index must be a DatetimeIndex"):
        comparison.build_comparison_frame()


def test_rejects_missing_five_day_target_column(monkeypatch) -> None:
    dates = pd.bdate_range("2022-01-03", periods=8)
    setup_comparison_inputs(monkeypatch, dates)

    features = pd.DataFrame(
        {"wrong_target": [np.nan] * 5 + [0.01, 0.02, 0.03]},
        index=dates,
    )

    original_read_parquet = pd.read_parquet

    def fake_read_parquet(path, *args, **kwargs):
        if str(path) == "data/processed/volatility_features.parquet":
            return features
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    with pytest.raises(ValueError, match="target_rv_5"):
        comparison.build_comparison_frame()
