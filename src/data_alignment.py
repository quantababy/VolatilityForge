"""Align market datasets on shared trading dates."""

from __future__ import annotations

import pandas as pd


def align_spy_vix(spy: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Inner-join SPY and VIX observations by normalized date."""
    if spy.empty or vix.empty:
        raise ValueError("SPY and VIX datasets must not be empty.")

    spy = spy.copy()
    vix = vix.copy()

    spy.index = pd.DatetimeIndex(spy.index).normalize()
    vix.index = pd.DatetimeIndex(vix.index).normalize()

    if spy.index.has_duplicates:
        raise ValueError("SPY contains duplicate dates.")

    if vix.index.has_duplicates:
        raise ValueError("VIX contains duplicate dates.")

    vix = vix.rename(columns={column: f"VIX_{column}" for column in vix.columns})

    aligned = spy.join(vix, how="inner", validate="one_to_one")
    aligned.index.name = "Date"

    if aligned.empty:
        raise ValueError("SPY and VIX have no dates in common.")

    return aligned.sort_index()
