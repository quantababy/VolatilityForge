from pathlib import Path

import yaml


def test_config_file_exists():
    assert Path("configs/config.yaml").is_file()


def test_project_configuration():
    with Path("configs/config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["project"]["name"] == "VolatilityForge"
    assert config["project"]["random_seed"] == 42


def test_forecast_horizons():
    with Path("configs/config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["forecast"]["horizons"] == [1, 5]


def test_development_period():
    with Path("configs/config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["evaluation"]["development_start"] == "2022-01-01"
    assert config["evaluation"]["method"] == "expanding_window"
