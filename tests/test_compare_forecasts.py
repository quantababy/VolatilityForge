
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

        forecast_dates = dates.delete(2)
        column = next(
            column
            for _, (file_path, column) in module.FORECAST_FILES.items()
            if file_path == path
        )
        values = pd.Series(1.0, index=forecast_dates, name=column)
        values.loc[dates[3]] = np.inf
        return values.to_frame()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    mse_frame, losses = module.build_comparison_frame()

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

    assert dates[1] not in mse_frame.index
    assert dates[1] not in losses.index
    assert len(mse_frame) == 3
    assert len(losses) == 3


def test_five_day_hac_lags_have_pre_specified_primary_and_sensitivities():
    assert compare_five_day_forecasts.PRIMARY_HAC_LAG == 4
    assert compare_five_day_forecasts.HAC_SENSITIVITY_LAGS == (5, 10, 22)
    assert compare_five_day_forecasts.HAC_LAGS == (4, 5, 10, 22)


def test_five_day_comparison_excludes_immature_targets(monkeypatch) -> None:
    module = compare_five_day_forecasts
    dates = pd.date_range("2024-01-01", periods=6, freq="D")

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


@pytest.mark.parametrize(
    ("index", "expected_error"),
    [
        (
            pd.Index(["2024-01-01", "2024-01-02"]),
            "Feature index must be a DatetimeIndex",
        ),
        (
            pd.DatetimeIndex(["2024-01-01", "2024-01-01"]),
            "Feature index must contain unique timestamps",
        ),
        (
            pd.DatetimeIndex(["2024-01-02", "2024-01-01"]),
            "Feature index must be sorted in ascending order",
        ),
    ],
)
def test_one_day_comparison_rejects_invalid_feature_indexes(
    monkeypatch, index, expected_error
):
    module = compare_forecasts
    features = pd.DataFrame(
        {"target_rv_1": [1.0, 2.0]},
        index=index,
    )

    monkeypatch.setattr(
        module,
        "load_evaluation_config",
        lambda: {"development_start": "2024-01-01"},
    )
    monkeypatch.setattr(pd, "read_parquet", lambda path: features)

    expected_exception = (
        TypeError
        if not isinstance(index, pd.DatetimeIndex)
        else ValueError
    )

    with pytest.raises(expected_exception, match=expected_error):
        module.build_comparison_frame()


@pytest.mark.parametrize(
    ("index", "expected_error"),
    [
        (
            pd.Index(["2024-01-01", "2024-01-02"]),
            "forecast index must be a DatetimeIndex",
        ),
        (
            pd.DatetimeIndex(["2024-01-01", "2024-01-01"]),
            "forecast index must contain unique timestamps",
        ),
        (
            pd.DatetimeIndex(["2024-01-02", "2024-01-01"]),
            "forecast index must be sorted in ascending order",
        ),
    ],
)
def test_one_day_comparison_rejects_invalid_forecast_indexes(
    monkeypatch, index, expected_error
):
    module = compare_forecasts
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    features = pd.DataFrame(
        {"target_rv_1": [1.0, 2.0]},
        index=dates,
    )
    first_path, first_column = next(iter(module.FORECAST_FILES.values()))

    def fake_read_parquet(path):
        if path == "data/processed/volatility_features.parquet":
            return features
        if path == first_path:
            return pd.DataFrame({first_column: [1.0, 2.0]}, index=index)

        _, column = next(
            item
            for item in module.FORECAST_FILES.values()
            if item[0] == path
        )
        return pd.DataFrame({column: [1.0, 2.0]}, index=dates)

    monkeypatch.setattr(
        module,
        "load_evaluation_config",
        lambda: {"development_start": "2024-01-01"},
    )
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    expected_exception = (
        TypeError
        if not isinstance(index, pd.DatetimeIndex)
        else ValueError
    )

    with pytest.raises(expected_exception, match=expected_error):
        module.build_comparison_frame()


def test_one_day_comparison_rejects_missing_target_column(monkeypatch):
    module = compare_forecasts
    features = pd.DataFrame(
        {"other_column": [1.0]},
        index=pd.date_range("2024-01-01", periods=1),
    )

    monkeypatch.setattr(
        module,
        "load_evaluation_config",
        lambda: {"development_start": "2024-01-01"},
    )
    monkeypatch.setattr(pd, "read_parquet", lambda path: features)

    with pytest.raises(
        ValueError,
        match="Feature data must contain target_rv_1",
    ):
        module.build_comparison_frame()


def test_one_day_comparison_rejects_missing_forecast_column(monkeypatch):
    module = compare_forecasts
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    features = pd.DataFrame(
        {"target_rv_1": [1.0, 2.0]},
        index=dates,
    )
    first_path, _ = next(iter(module.FORECAST_FILES.values()))

    def fake_read_parquet(path):
        if path == "data/processed/volatility_features.parquet":
            return features
        if path == first_path:
            return pd.DataFrame({"wrong_column": [1.0, 2.0]}, index=dates)

        _, column = next(
            item
            for item in module.FORECAST_FILES.values()
            if item[0] == path
        )
        return pd.DataFrame({column: [1.0, 2.0]}, index=dates)

    monkeypatch.setattr(
        module,
        "load_evaluation_config",
        lambda: {"development_start": "2024-01-01"},
    )
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    with pytest.raises(
        ValueError,
        match="EWMA forecast data must contain ewma_variance_1d",
    ):
        module.build_comparison_frame()


def test_one_day_main_writes_comparison_report_with_metadata(
    monkeypatch, tmp_path
):
    module = compare_forecasts
    dates = pd.date_range("2024-01-02", periods=3, freq="D")


    models = list(module.FORECAST_FILES)

    mse_aligned = pd.DataFrame(
        {"target": [1.0, 2.0, 3.0]},
        index=dates,
    )
    losses = pd.DataFrame(index=dates)

    for i, model in enumerate(models, start=1):
        mse_aligned[model] = [
            1.0 + i * 0.1,
            2.0,
            3.0 - i * 0.1,
        ]
        losses[model] = [
            0.1 * i,
            0.2 * i,
            0.3 * i,
        ]


    monkeypatch.setattr(
        module,
        "build_comparison_frame",
        lambda: (mse_aligned, losses),
    )
    monkeypatch.setattr(
        module,
        "build_calibration_report",
        lambda frame: pd.DataFrame({"model": ["EWMA"]}),
    )
    monkeypatch.setattr(
        module,
        "run_dm_sensitivity",
        lambda frame: pd.DataFrame({"model": ["EWMA"]}),
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()

    module.main()

    report = pd.read_csv(
        tmp_path / "reports" / "volatility_model_comparison.csv"
    )

    assert set(report["model"]) == set(module.FORECAST_FILES)
    assert set(report["forecast_horizon"]) == {1}
    assert set(report["n_mse_observations"]) == {3}
    assert set(report["n_qlike_observations"]) == {3}
    assert set(report["mse_start_date"]) == {"2024-01-02"}
    assert set(report["mse_end_date"]) == {"2024-01-04"}
    assert set(report["qlike_start_date"]) == {"2024-01-02"}
    assert set(report["qlike_end_date"]) == {"2024-01-04"}
    assert report["qlike"].notna().all()
    assert report["mse"].notna().all()
