import pandas as pd
import pytest
import yaml

from src.run_strategy_sensitivity import (
    FORECAST_FILES,
    TARGET_VOLATILITIES,
    TRANSACTION_COSTS_BPS,
    generate_strategy_sensitivity,
)


def create_inputs(tmp_path):
    config_path = tmp_path / "config.yaml"
    features_path = tmp_path / "features.parquet"
    processed_dir = tmp_path / "processed"
    report_path = tmp_path / "reports" / "sensitivity.csv"
    processed_dir.mkdir()

    config = {
        "strategy": {"max_leverage": 1.0},
        "forecast": {"trading_days_per_year": 252},
        "evaluation": {"development_start": "2024-01-01"},
    }
    config_path.write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )

    dates = pd.bdate_range("2024-01-01", periods=6)
    features = pd.DataFrame(
        {"Close": [100.0, 101.0, 99.0, 102.0, 103.0, 101.0]},
        index=dates,
    )
    features.index.name = "Date"
    features.to_parquet(features_path)

    for filename, column in FORECAST_FILES.values():
        forecasts = pd.DataFrame(
            {column: [0.0004, 0.0005, 0.0003, 0.0004, 0.0005]},
            index=dates[:5],
        )
        forecasts.index.name = "Date"
        forecasts.to_parquet(processed_dir / filename)

    return config_path, features_path, processed_dir, report_path


def test_generates_complete_sensitivity_grid(tmp_path):
    config, features, processed, report = create_inputs(tmp_path)

    result = generate_strategy_sensitivity(
        config_path=config,
        features_path=features,
        processed_dir=processed,
        report_path=report,
    )

    expected_rows = (
        len(FORECAST_FILES)
        * len(TARGET_VOLATILITIES)
        * len(TRANSACTION_COSTS_BPS)
    )
    assert len(result) == expected_rows == 60
    assert result["model"].nunique() == 5
    assert set(result["target_annual_volatility"]) == set(
        TARGET_VOLATILITIES
    )
    assert set(result["transaction_cost_bps"]) == set(
        TRANSACTION_COSTS_BPS
    )
    assert (result["n_observations"] == 5).all()
    assert report.is_file()

    saved = pd.read_csv(report)
    pd.testing.assert_frame_equal(saved, result, check_dtype=False)


def test_rejects_invalid_configuration_mapping(tmp_path):
    config, features, processed, report = create_inputs(tmp_path)
    config.write_text("- invalid\\n- configuration\\n", encoding="utf-8")

    with pytest.raises(TypeError, match="YAML mapping"):
        generate_strategy_sensitivity(
            config_path=config,
            features_path=features,
            processed_dir=processed,
            report_path=report,
        )


def test_rejects_missing_forecast_file(tmp_path):
    config, features, processed, report = create_inputs(tmp_path)
    (processed / FORECAST_FILES["ewma"][0]).unlink()

    with pytest.raises(FileNotFoundError, match="Missing forecast file"):
        generate_strategy_sensitivity(
            config_path=config,
            features_path=features,
            processed_dir=processed,
            report_path=report,
        )
