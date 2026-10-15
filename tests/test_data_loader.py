import pandas as pd
import pytest

from src.data_loader import download_market_data, save_market_data


def sample_market_data():
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        {
            "Open": [100.0, 102.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 101.0],
            "Close": [104.0, 103.0],
            "Volume": [1000, 1200],
        },
        index=index,
    ).rename_axis("Date")


def test_download_rejects_empty_ticker():
    with pytest.raises(ValueError, match="Ticker must not be empty"):
        download_market_data(" ", "2024-01-01")


def test_download_rejects_empty_response(monkeypatch):
    monkeypatch.setattr(
        "src.data_loader.yf.download",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="No market data returned"):
        download_market_data("SPY", "2024-01-01")


def test_save_and_reload_parquet(tmp_path):
    data = sample_market_data()
    path = tmp_path / "market_data.parquet"

    save_market_data(data, str(path))
    loaded = pd.read_parquet(path)

    pd.testing.assert_frame_equal(loaded, data)


def test_save_rejects_empty_data(tmp_path):
    with pytest.raises(ValueError, match="Cannot save an empty"):
        save_market_data(pd.DataFrame(), str(tmp_path / "empty.parquet"))
