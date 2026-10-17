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

    for key in ("minimum_training_observations", "reestimate_every"):
        value = evaluation.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"evaluation.{key} must be a positive integer."
            )

    return evaluation
