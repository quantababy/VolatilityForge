"""Moving-block-bootstrap Model Confidence Set procedure."""
from __future__ import annotations

import numpy as np
import pandas as pd


def model_confidence_set(
    losses: pd.DataFrame,
    *,
    alpha: float = 0.10,
    bootstrap_samples: int = 1000,
    block_length: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Sequentially eliminate inferior models using a paired block bootstrap.

    Rows are common forecast dates; columns are models; lower loss is better.
    The range statistic tests whether relative model losses are distinguishable.
    """
    if not isinstance(losses, pd.DataFrame):
        raise TypeError("losses must be a pandas DataFrame.")
    if losses.empty or losses.shape[1] < 2:
        raise ValueError("At least two models and one observation are required.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 100
    ):
        raise ValueError("bootstrap_samples must be an integer of at least 100.")
    if losses.columns.has_duplicates:
        raise ValueError("Model names must be unique.")
    if not losses.index.is_unique:
        raise ValueError("Loss dates must be unique.")
    if not losses.index.is_monotonic_increasing:
        raise ValueError("Loss dates must be sorted in ascending order.")

    values = losses.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Losses must be finite and aligned on a common sample.")
    if (values < 0).any():
        raise ValueError("Losses must be non-negative.")

    n_obs, n_models = values.shape
    if block_length is None:
        block_length = min(5, n_obs)
    elif (
        isinstance(block_length, bool)
        or not isinstance(block_length, int)
        or block_length < 1
        or block_length > n_obs
    ):
        raise ValueError("block_length must be between 1 and the number of rows.")

    rng = np.random.default_rng(random_seed)
    n_blocks = int(np.ceil(n_obs / block_length))
    offsets = np.arange(block_length)
    bootstrap_indices = np.empty((bootstrap_samples, n_obs), dtype=int)

    for b in range(bootstrap_samples):
        starts = rng.integers(0, n_obs, size=n_blocks)
        indices = ((starts[:, None] + offsets) % n_obs).ravel()
        bootstrap_indices[b] = indices[:n_obs]

    active = list(range(n_models))
    records: list[dict[str, object]] = []

    while len(active) > 1:
        current = values[:, active]
        means = current.mean(axis=0)

        # Pairwise loss differentials preserve the dependence between models.
        differences = current[:, :, None] - current[:, None, :]
        mean_differences = differences.mean(axis=0)
        centered = differences - mean_differences
        boot_differences = centered[bootstrap_indices].mean(axis=1)
        pair_se = boot_differences.std(axis=0, ddof=1)

        observed_t = np.zeros_like(mean_differences)
        np.divide(
            np.abs(mean_differences),
            pair_se,
            out=observed_t,
            where=pair_se > 0,
        )
        observed_stat = float(observed_t.max())

        bootstrap_t = np.zeros_like(boot_differences)
        np.divide(
            np.abs(boot_differences),
            pair_se[None, :, :],
            out=bootstrap_t,
            where=pair_se[None, :, :] > 0,
        )
        bootstrap_stats = bootstrap_t.max(axis=(1, 2))
        p_value = float(
            (1 + np.count_nonzero(bootstrap_stats >= observed_stat))
            / (bootstrap_samples + 1)
        )

        if p_value >= alpha:
            for position, model_index in enumerate(active):
                records.append({
                    "model": losses.columns[model_index],
                    "mean_loss": float(means[position]),
                    "elimination_p_value": np.nan,
                    "in_confidence_set": True,
                })
            break

        worst_position = int(np.argmax(means))
        worst_index = active.pop(worst_position)
        records.append({
            "model": losses.columns[worst_index],
            "mean_loss": float(values[:, worst_index].mean()),
            "elimination_p_value": p_value,
            "in_confidence_set": False,
        })

    if len(active) == 1:
        model_index = active[0]
        records.append({
            "model": losses.columns[model_index],
            "mean_loss": float(values[:, model_index].mean()),
            "elimination_p_value": np.nan,
            "in_confidence_set": True,
        })

    result = pd.DataFrame(records)
    return result.sort_values(
        ["in_confidence_set", "mean_loss"],
        ascending=[False, True],
        ignore_index=True,
    )
