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
        lambda file: {"evaluation": {"method": "expanding_window", "historical_forecast_start": "2001-01-03"}},
    )

    with pytest.raises(ValueError, match="development_start"):
        load_evaluation_config()

def test_loads_training_settings():
    config = load_evaluation_config()

    assert config["minimum_training_observations"] == 252
    assert config["reestimate_every"] == 21


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("minimum_training_observations", 0),
        ("minimum_training_observations", -1),
        ("minimum_training_observations", True),
        ("minimum_training_observations", "252"),
        ("reestimate_every", 0),
        ("reestimate_every", -1),
        ("reestimate_every", False),
        ("reestimate_every", "21"),
    ],
)
def test_rejects_invalid_training_settings(monkeypatch, key, value):
    from src import evaluation_config

    monkeypatch.setattr(
        evaluation_config.yaml,
        "safe_load",
        lambda file: {
            "evaluation": {
                "historical_forecast_start": "2001-01-03",
                "development_start": "2022-01-01",
                "minimum_training_observations": 252,
                "reestimate_every": 21,
                key: value,
            }
        },
    )

    with pytest.raises(ValueError, match=key):
        load_evaluation_config()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("historical_forecast_start", None),
        ("historical_forecast_start", ""),
        ("historical_forecast_start", "not-a-date"),
        ("development_start", None),
        ("development_start", ""),
        ("development_start", "not-a-date"),
    ],
)
def test_rejects_invalid_evaluation_dates(monkeypatch, key, value):
    from src import evaluation_config

    monkeypatch.setattr(
        evaluation_config.yaml,
        "safe_load",
        lambda file: {
            "evaluation": {
                "historical_forecast_start": "2001-01-03",
                "development_start": "2022-01-01",
                "minimum_training_observations": 252,
                "reestimate_every": 21,
                key: value,
            }
        },
    )

    with pytest.raises(ValueError, match=key):
        load_evaluation_config()
