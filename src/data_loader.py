from __future__ import annotations

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_market_data(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV data for a ticker."""
    if not ticker.strip():
        raise ValueError("Ticker must not be empty.")

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    if data.empty:
        raise ValueError(f"No market data returned for ticker {ticker!r}.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data.loc[:, REQUIRED_COLUMNS].copy()
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"
    data = data.sort_index()

    if data.index.has_duplicates:
        raise ValueError("Market data contains duplicate dates.")

    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Market data contains missing OHLCV values.")

    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Market prices must be positive.")

    if (data["Volume"] < 0).any():
        raise ValueError("Trading volume cannot be negative.")

    price_tolerance = 1e-10

    highest_other_price = data[["Open", "Low", "Close"]].max(axis=1)
    lowest_other_price = data[["Open", "High", "Close"]].min(axis=1)

    if (highest_other_price - data["High"] > price_tolerance).any():
        raise ValueError("High price is inconsistent with OHLC prices.")

    if (data["Low"] - lowest_other_price > price_tolerance).any():
        raise ValueError("Low price is inconsistent with OHLC prices.")

    return data


def save_market_data(data: pd.DataFrame, path: str) -> None:
    """Save validated market data to a Parquet file."""
    if data.empty:
        raise ValueError("Cannot save an empty market-data DataFrame.")

    data.to_parquet(path, index=True)
