"""Data loading and public-market download helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


ReturnMethod = Literal["log", "simple"]


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV of asset prices without forward-filling missing observations.

    The first column is treated as a date/index column when it is clearly
    date-like or named as an index column. Asset columns are converted to numeric
    values, infinities are replaced with missing values, and all-empty assets are
    removed.
    """

    return _load_market_csv(path)


def load_returns_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV of asset returns with the same parsing rules as prices.

    Missing values are preserved for the graph engine's cleaning layer rather
    than imputed here.
    """

    return _load_market_csv(path)


def prices_to_returns(prices: pd.DataFrame, method: ReturnMethod = "log") -> pd.DataFrame:
    """Convert price levels into simple or log returns.

    Log returns require strictly positive observed prices. Rows where every
    asset return is missing are dropped, but partial missingness is retained for
    later cleaning by the graph-regime engine.
    """

    if method not in {"log", "simple"}:
        raise ValueError("method must be either 'log' or 'simple'.")
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if prices.empty:
        raise ValueError("prices must contain at least one row and one column.")

    numeric_prices = prices.apply(pd.to_numeric, errors="coerce")
    numeric_prices = numeric_prices.replace([np.inf, -np.inf], np.nan)
    numeric_prices = numeric_prices.dropna(axis=1, how="all")
    if numeric_prices.empty:
        raise ValueError("prices must contain at least one numeric asset column.")

    if method == "log":
        if (numeric_prices <= 0).to_numpy().any():
            raise ValueError("log returns require all observed prices to be positive.")
        returns = np.log(numeric_prices / numeric_prices.shift(1))
    else:
        returns = numeric_prices / numeric_prices.shift(1) - 1.0

    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(axis=0, how="all")
    return returns.astype(float)


def download_yfinance_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance via yfinance.

    This helper uses a free public data source and raises clear errors when the
    optional dependency, network access, or returned data are unavailable.
    """

    if not tickers:
        raise ValueError("tickers must contain at least one symbol.")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for online price downloads. Install the "
            "'data' optional dependencies or load local CSV files instead."
        ) from exc

    try:
        downloaded = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "yfinance price download failed. Check internet access, ticker "
            "symbols, or use load_prices_csv with a local CSV file."
        ) from exc

    prices = _extract_yfinance_price_frame(downloaded, tickers)
    if prices.empty:
        raise RuntimeError(
            "yfinance returned no adjusted close prices. Use a different date "
            "range, check ticker availability, or load local CSV files."
        )

    prices = prices.dropna(axis=1, how="all")
    prices = prices.replace([np.inf, -np.inf], np.nan)
    if isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.sort_index()
    return prices.astype(float)


def download_vix(start: str, end: str | None = None) -> pd.Series:
    """Download the CBOE VIX index from Yahoo Finance as a series named ``vix``."""

    try:
        prices = download_yfinance_prices(["^VIX"], start=start, end=end)
    except ImportError:
        raise
    except RuntimeError as exc:
        raise RuntimeError(
            "VIX download failed. Check internet access or load a local VIX CSV "
            "and pass it into the benchmark layer."
        ) from exc

    if prices.empty:
        raise RuntimeError("VIX download returned no data.")

    series = prices.iloc[:, 0].rename("vix")
    return series.dropna()


def _load_market_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    raw = pd.read_csv(csv_path)
    if raw.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    parsed = _maybe_use_first_column_as_index(raw)
    cleaned = parsed.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(axis=1, how="all")

    if isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned = cleaned.sort_index()

    return cleaned.astype(float)


def _maybe_use_first_column_as_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[1] < 2:
        return frame.copy()

    first_column = frame.columns[0]
    first_name = str(first_column).strip().lower()
    first_values = frame[first_column]
    index_like_names = {"", "date", "datetime", "timestamp", "time", "index", "unnamed: 0"}
    date_like_names = {"date", "datetime", "timestamp", "time"}

    if first_name in index_like_names or _looks_like_date_column(first_values):
        parsed_dates = pd.to_datetime(first_values, errors="coerce")
        if parsed_dates.notna().all() and (
            first_name in date_like_names
            or first_name in {"", "index", "unnamed: 0"}
            or _looks_like_date_column(first_values)
        ):
            output = frame.drop(columns=first_column).copy()
            output.index = pd.DatetimeIndex(parsed_dates, name=str(first_column))
            return output

        if first_name in {"index", "unnamed: 0"}:
            output = frame.drop(columns=first_column).copy()
            output.index = first_values
            output.index.name = str(first_column)
            return output

    return frame.copy()


def _looks_like_date_column(values: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(values):
        return False

    parsed = pd.to_datetime(values, errors="coerce")
    return bool(parsed.notna().mean() >= 0.9)


def _extract_yfinance_price_frame(
    downloaded: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()

    if isinstance(downloaded.columns, pd.MultiIndex):
        for field in ("Adj Close", "Close"):
            if field in downloaded.columns.get_level_values(0):
                prices = downloaded[field]
                break
        else:
            return pd.DataFrame()
    else:
        if "Adj Close" in downloaded.columns:
            prices = downloaded[["Adj Close"]].copy()
        elif "Close" in downloaded.columns:
            prices = downloaded[["Close"]].copy()
        else:
            return pd.DataFrame()

        prices.columns = tickers[:1]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.copy()
    prices.columns = [str(column) for column in prices.columns]
    return prices
