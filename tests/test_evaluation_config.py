
import pytest

from src.evaluation_config import load_evaluation_config

VALID_EVALUATION = {
    "method": "expanding_window",
    "historical_forecast_start": "2001-01-03",
    "development_start": "2022-01-01",
    "primary_metric": "qlike",
    "secondary_metrics": ["mse"],
    "minimum_training_observations": 252,
    "reestimate_every": 21,
}


def mock_evaluation_config(monkeypatch, evaluation):
    from src import evaluation_config

    monkeypatch.setattr(
        evaluation_config.yaml,
        "safe_load",
        lambda file: {"evaluation": evaluation},
    )


def test_loads_development_start_from_config():
    config = load_evaluation_config()

    assert config["development_start"] == "2022-01-01"


def test_loads_training_settings():
    config = load_evaluation_config()

    assert config["minimum_training_observations"] == 252
    assert config["reestimate_every"] == 21


def test_rejects_missing_development_start(monkeypatch):
    evaluation = VALID_EVALUATION.copy()
    evaluation.pop("development_start")
    mock_evaluation_config(monkeypatch, evaluation)

    with pytest.raises(ValueError, match="development_start"):
        load_evaluation_config()


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
    evaluation = VALID_EVALUATION.copy()
    evaluation[key] = value
    mock_evaluation_config(monkeypatch, evaluation)

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
    evaluation = VALID_EVALUATION.copy()
    evaluation[key] = value
    mock_evaluation_config(monkeypatch, evaluation)

    with pytest.raises(ValueError, match=key):
        load_evaluation_config()


@pytest.mark.parametrize(
    "method",
    ["rolling_window", "fixed_window", "", None],
)
def test_rejects_unsupported_evaluation_method(monkeypatch, method):
    evaluation = VALID_EVALUATION.copy()
    evaluation["method"] = method
    mock_evaluation_config(monkeypatch, evaluation)

    with pytest.raises(ValueError, match="evaluation.method"):
        load_evaluation_config()


@pytest.mark.parametrize("metric", ["mse", "rmse", "", None])
def test_rejects_unsupported_primary_metric(monkeypatch, metric):
    evaluation = VALID_EVALUATION.copy()
    evaluation["primary_metric"] = metric
    mock_evaluation_config(monkeypatch, evaluation)

    with pytest.raises(ValueError, match="evaluation.primary_metric"):
        load_evaluation_config()


@pytest.mark.parametrize(
    "secondary_metrics",
    [
        None,
        "mse",
        [],
        ["rmse"],
        ["mse", ""],
        ["mse", None],
    ],
)
def test_rejects_invalid_secondary_metrics(
    monkeypatch, secondary_metrics
):
    evaluation = VALID_EVALUATION.copy()
    evaluation["secondary_metrics"] = secondary_metrics
    mock_evaluation_config(monkeypatch, evaluation)

    with pytest.raises(ValueError, match="evaluation.secondary_metrics"):
        load_evaluation_config()


@pytest.mark.parametrize(
    ("historical_start", "development_start"),
    [
        ("2022-01-01", "2022-01-01"),
        ("2023-01-01", "2022-01-01"),
    ],
)
def test_rejects_invalid_evaluation_date_order(
    monkeypatch, historical_start, development_start
):
    evaluation = VALID_EVALUATION.copy()
    evaluation["historical_forecast_start"] = historical_start
    evaluation["development_start"] = development_start
    mock_evaluation_config(monkeypatch, evaluation)

    with pytest.raises(
        ValueError,
        match="historical_forecast_start must precede development_start",
    ):
        load_evaluation_config()
