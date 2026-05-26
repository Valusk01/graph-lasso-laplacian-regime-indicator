from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graph_regime.benchmarks import (
    compute_correlation_regime_score,
    compute_market_drawdown,
    compute_realized_volatility,
    create_stress_labels,
)


def test_compute_realized_volatility_series_input() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, 0.005], name="SPY")

    realized_vol = compute_realized_volatility(returns, window=2, annualize=False)

    assert realized_vol.name == "realized_volatility"
    assert (realized_vol.dropna() >= 0.0).all()


def test_compute_realized_volatility_dataframe_input() -> None:
    returns = pd.DataFrame(
        {
            "SPY": [0.01, -0.02, 0.015, 0.005],
            "TLT": [0.00, 0.01, -0.005, 0.002],
        }
    )

    realized_vol = compute_realized_volatility(returns, window=2, annualize=True)

    assert realized_vol.name == "realized_volatility"
    assert (realized_vol.dropna() >= 0.0).all()


def test_compute_market_drawdown_marks_new_highs_and_declines() -> None:
    prices = pd.Series([100.0, 110.0, 99.0, 120.0])

    drawdown = compute_market_drawdown(prices)

    assert drawdown.name == "drawdown"
    assert drawdown.iloc[0] == 0.0
    assert drawdown.iloc[1] == 0.0
    assert drawdown.iloc[2] < 0.0
    assert drawdown.iloc[3] == 0.0


def test_compute_correlation_regime_score_shape_and_finite_values() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    returns = pd.DataFrame(
        {
            "asset_a": [0.01, 0.02, -0.01, 0.03, 0.02, -0.02],
            "asset_b": [0.02, 0.01, -0.02, 0.02, 0.01, -0.01],
            "asset_c": [-0.01, 0.00, 0.01, -0.02, 0.01, 0.02],
        },
        index=index,
    )

    score = compute_correlation_regime_score(returns, window=3)

    assert score.name == "average_correlation"
    assert len(score) == 4
    assert score.index.equals(index[2:])
    assert np.isfinite(score.dropna().to_numpy()).all()


def test_create_stress_labels_from_all_benchmarks() -> None:
    index = pd.date_range("2020-01-01", periods=4)
    vix = pd.Series([20.0, 35.0, 25.0, 40.0], index=index)
    drawdown = pd.Series([0.0, -0.05, -0.12, -0.02], index=index)
    realized_vol = pd.Series([0.10, 0.20, 0.30, 0.40], index=index)
    correlation_score = pd.Series([0.20, 0.30, 0.85, 0.90], index=index)

    labels = create_stress_labels(
        vix=vix,
        drawdown=drawdown,
        realized_vol=realized_vol,
        correlation_score=correlation_score,
        vix_threshold=30.0,
        drawdown_threshold=-0.10,
        vol_quantile=0.75,
        correlation_quantile=0.75,
    )

    expected_columns = {
        "vix",
        "drawdown",
        "realized_volatility",
        "average_correlation",
        "vix_stress_label",
        "drawdown_stress_label",
        "realized_volatility_stress_label",
        "correlation_stress_label",
        "systemic_stress_label",
    }
    assert expected_columns.issubset(labels.columns)
    assert labels.loc[index[1], "vix_stress_label"] == 1
    assert labels.loc[index[2], "drawdown_stress_label"] == 1
    assert labels["systemic_stress_label"].max() == 1

def test_create_stress_labels_preserves_missing_benchmark_labels() -> None:
    index = pd.date_range("2020-01-01", periods=3)
    vix = pd.Series([20.0, np.nan, 40.0], index=index)

    labels = create_stress_labels(vix=vix, vix_threshold=30.0)

    assert labels.loc[index[0], "vix_stress_label"] == 0
    assert pd.isna(labels.loc[index[1], "vix_stress_label"])
    assert labels.loc[index[2], "vix_stress_label"] == 1
    assert labels.loc[index[2], "systemic_stress_label"] == 1

def test_create_stress_labels_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="At least one benchmark"):
        create_stress_labels()

def test_compute_correlation_regime_score_supports_absolute_correlations() -> None:
    index = pd.date_range("2020-01-01", periods=4)
    returns = pd.DataFrame(
        {
            "asset_a": [1.0, 2.0, 3.0, 4.0],
            "asset_b": [-1.0, -2.0, -3.0, -4.0],
            "asset_c": [1.0, 2.0, 3.0, 4.0],
        },
        index=index,
    )

    raw_score = compute_correlation_regime_score(returns, window=4, use_absolute=False)
    absolute_score = compute_correlation_regime_score(returns, window=4, use_absolute=True)

    assert absolute_score.iloc[0] >= raw_score.iloc[0]
    assert np.isclose(absolute_score.iloc[0], 1.0)