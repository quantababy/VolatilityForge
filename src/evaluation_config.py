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

    start = evaluation.get("development_start")
    if not isinstance(start, str) or not start.strip():
        raise ValueError(
            "evaluation.development_start must be a non-empty date string."
        )

    try:
        parsed_start = pd.Timestamp(start)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "evaluation.development_start must be a valid date."
        ) from exc

    if pd.isna(parsed_start):
        raise ValueError("evaluation.development_start must be a valid date.")

    return evaluation
