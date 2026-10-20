
"""Shared evaluation-period configuration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path("configs/config.yaml")


def load_evaluation_config() -> dict:
    """Load and validate the evaluation settings from project config."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("Project configuration must be a YAML mapping.")

    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        raise TypeError("Evaluation configuration must be a mapping.")

    if evaluation.get("method") != "expanding_window":
        raise ValueError(
            "evaluation.method must be 'expanding_window'."
        )

    if evaluation.get("primary_metric") != "qlike":
        raise ValueError(
            "evaluation.primary_metric must be 'qlike'."
        )

    secondary_metrics = evaluation.get("secondary_metrics")
    if (
        not isinstance(secondary_metrics, list)
        or not all(
            isinstance(metric, str) and metric.strip()
            for metric in secondary_metrics
        )
        or "mse" not in secondary_metrics
    ):
        raise ValueError(
            "evaluation.secondary_metrics must be a list of non-empty "
            "strings containing 'mse'."
        )

    parsed_dates = {}

    for key in ("historical_forecast_start", "development_start"):
        value = evaluation.get(key)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"evaluation.{key} must be a non-empty date string."
            )

        try:
            parsed_date = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"evaluation.{key} must be a valid date."
            ) from exc

        if pd.isna(parsed_date):
            raise ValueError(
                f"evaluation.{key} must be a valid date."
            )

        parsed_dates[key] = parsed_date

    if (
        parsed_dates["historical_forecast_start"]
        >= parsed_dates["development_start"]
    ):
        raise ValueError(
            "historical_forecast_start must precede development_start."
        )

    for key in ("minimum_training_observations", "reestimate_every"):
        value = evaluation.get(key)

        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"evaluation.{key} must be a positive integer."
            )

    return evaluation
