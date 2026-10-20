
import numpy as np
import pandas as pd
import yaml

from src import run_data_pipeline


def make_market_data(
    dates: pd.DatetimeIndex,
    base_price: float,
) -> pd.DataFrame:
    close = base_price * np.exp(np.arange(len(dates)) * 0.001)
    open_price = close * 0.999

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": np.maximum(open_price, close) * 1.001,
            "Low": np.minimum(open_price, close) * 0.999,
            "Close": close,
            "Volume": np.full(len(dates), 100_000, dtype=int),
        },
        index=dates,
    ).rename_axis("Date")


def test_pipeline_builds_all_datasets_without_network(
    tmp_path,
    monkeypatch,
):
    dates = pd.bdate_range("2024-01-01", periods=40)
    spy = make_market_data(dates, 450.0)
    vix = make_market_data(dates, 20.0)

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "data": {
                    "primary_asset": "SPY",
                    "implied_volatility_asset": "^VIX",
                    "start_date": "2024-01-01",
                    "cache_format": "parquet",
                }
            }
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_download(ticker, start_date, end_date=None):
        calls.append((ticker, start_date, end_date))
        return spy.copy() if ticker == "SPY" else vix.copy()

    monkeypatch.setattr(
        run_data_pipeline,
        "download_market_data",
        fake_download,
    )

    paths = run_data_pipeline.run_pipeline(tmp_path)

    assert calls == [
        ("SPY", "2024-01-01", None),
        ("^VIX", "2024-01-01", None),
    ]

    expected_paths = {
        "spy": tmp_path / "data/raw/spy.parquet",
        "vix": tmp_path / "data/raw/vix.parquet",
        "aligned": tmp_path / "data/processed/spy_vix.parquet",
        "features": (
            tmp_path / "data/processed/volatility_features.parquet"
        ),
    }

    assert paths == expected_paths

    for path in expected_paths.values():
        assert path.is_file()

    aligned = pd.read_parquet(paths["aligned"])
    features = pd.read_parquet(paths["features"])

    assert len(aligned) == len(dates)
    assert "VIX_Close" in aligned.columns
    assert len(features) == len(dates)
    assert {"log_return", "rv_5", "rv_22", "target_rv_1", "target_rv_5"} <= (
        set(features.columns)
    )
    assert features["target_rv_1"].iloc[-1:].isna().all()
    assert features["target_rv_5"].iloc[-5:].isna().all()
