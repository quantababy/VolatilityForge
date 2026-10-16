import pytest

from src.evaluation_config import load_evaluation_config


def test_loads_development_start_from_config():
    config = load_evaluation_config()

    assert config["development_start"] == "2022-01-01"


def test_rejects_missing_development_start(monkeypatch):
    from src import evaluation_config

    monkeypatch.setattr(
        evaluation_config.yaml,
        "safe_load",
        lambda file: {"evaluation": {"method": "expanding_window"}},
    )

    with pytest.raises(ValueError, match="development_start"):
        load_evaluation_config()
