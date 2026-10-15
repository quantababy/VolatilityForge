import pandas as pd
import pytest

from src.data_alignment import align_spy_vix


def make_data(dates, prefix=""):
    return pd.DataFrame(
        {
            f"{prefix}Close": [100.0 + i for i in range(len(dates))],
        },
        index=pd.to_datetime(dates),
    )


def test_alignment_keeps_only_common_dates():
    spy = make_data(["2024-01-02", "2024-01-03", "2024-01-04"])
    vix = make_data(
        ["2024-01-03", "2024-01-04", "2024-01-05"],
        prefix="",
    )

    aligned = align_spy_vix(spy, vix)

    assert len(aligned) == 2
    assert aligned.index.tolist() == list(
        pd.to_datetime(["2024-01-03", "2024-01-04"])
    )
    assert "Close" in aligned.columns
    assert "VIX_Close" in aligned.columns


def test_alignment_rejects_empty_dataset():
    empty = pd.DataFrame(columns=["Close"], index=pd.DatetimeIndex([]))
    nonempty = make_data(["2024-01-02"])

    with pytest.raises(ValueError, match="must not be empty"):
        align_spy_vix(empty, nonempty)


def test_alignment_rejects_duplicate_dates():
    dates = pd.to_datetime(["2024-01-02", "2024-01-02"])
    spy = pd.DataFrame({"Close": [100.0, 101.0]}, index=dates)
    vix = make_data(["2024-01-02"], prefix="")

    with pytest.raises(ValueError, match="SPY contains duplicate dates"):
        align_spy_vix(spy, vix)

