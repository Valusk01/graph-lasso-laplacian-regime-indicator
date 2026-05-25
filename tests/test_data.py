from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graph_regime.data import load_prices_csv, load_returns_csv, prices_to_returns


def test_load_prices_csv_parses_dates_and_numeric_values(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-01"],
            "SPY": ["101.0", "100.0"],
            "TLT": ["99.0", "100.0"],
            "empty": [np.nan, np.nan],
        }
    ).to_csv(csv_path, index=False)

    prices = load_prices_csv(csv_path)

    assert isinstance(prices, pd.DataFrame)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert prices.index.is_monotonic_increasing
    assert list(prices.columns) == ["SPY", "TLT"]
    assert prices.dtypes.tolist() == [float, float]


def test_prices_to_returns_supports_log_and_simple_returns() -> None:
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 105.0, 110.0],
            "TLT": [50.0, 49.0, 51.0],
        },
        index=pd.date_range("2020-01-01", periods=3),
    )

    log_returns = prices_to_returns(prices, method="log")
    simple_returns = prices_to_returns(prices, method="simple")

    assert np.allclose(log_returns.iloc[0]["SPY"], np.log(105.0 / 100.0))
    assert np.allclose(simple_returns.iloc[0]["TLT"], 49.0 / 50.0 - 1.0)
    assert len(log_returns) == 2
    assert len(simple_returns) == 2


def test_prices_to_returns_rejects_invalid_method() -> None:
    prices = pd.DataFrame({"SPY": [100.0, 101.0]})

    with pytest.raises(ValueError, match="method"):
        prices_to_returns(prices, method="arithmetic")  # type: ignore[arg-type]


def test_prices_to_returns_rejects_non_positive_log_prices() -> None:
    prices = pd.DataFrame({"SPY": [100.0, 0.0, 101.0]})

    with pytest.raises(ValueError, match="positive"):
        prices_to_returns(prices, method="log")


def test_load_returns_csv_parses_dates_and_numeric_values(tmp_path) -> None:
    csv_path = tmp_path / "returns.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "SPY": ["0.01", "0.02"],
            "TLT": ["-0.01", "0.00"],
        }
    ).to_csv(csv_path, index=False)

    returns = load_returns_csv(csv_path)

    assert isinstance(returns.index, pd.DatetimeIndex)
    assert list(returns.columns) == ["SPY", "TLT"]
    assert np.isfinite(returns.to_numpy()).all()
    assert returns.loc[pd.Timestamp("2020-01-02"), "SPY"] == 0.02
