from __future__ import annotations

import os
import tempfile
from pathlib import Path

_MPL_CACHE = Path(tempfile.gettempdir()) / "graph_regime_matplotlib_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from graph_regime.plotting import (
    plot_feature_panel,
    plot_indicator_vs_benchmark,
    plot_regime_class_returns,
    plot_regime_indicator,
    plot_scatter_indicator_target,
    plot_stress_boxplot,
)


def test_plot_regime_indicator_returns_figure_and_saves(tmp_path) -> None:
    index = pd.date_range("2020-01-01", periods=5)
    indicator = pd.DataFrame({"regime_indicator": [0.1, 0.2, 1.0, 0.3, 1.2]}, index=index)
    labels = pd.DataFrame({"systemic_stress_label": [0, 0, 1, pd.NA, 1]}, index=index)
    output_path = tmp_path / "regime_indicator.png"

    fig = plot_regime_indicator(indicator, stress_labels=labels, output_path=output_path)

    assert isinstance(fig, Figure)
    assert output_path.exists()
    plt.close(fig)


def test_plot_indicator_vs_benchmark_returns_figure_and_saves(tmp_path) -> None:
    index = pd.date_range("2020-01-01", periods=5)
    indicator = pd.Series([0.1, 0.2, 0.4, 0.8, 1.0], index=index)
    benchmark = pd.Series([15.0, 18.0, 25.0, 30.0, 35.0], index=index, name="vix")
    output_path = tmp_path / "indicator_vs_vix.png"

    fig = plot_indicator_vs_benchmark(
        indicator,
        benchmark,
        benchmark_name="vix",
        output_path=output_path,
    )

    assert isinstance(fig, Figure)
    assert output_path.exists()
    plt.close(fig)


def test_plot_feature_panel_returns_figure() -> None:
    index = pd.date_range("2020-01-01", periods=4)
    features = pd.DataFrame(
        {
            "average_graph_strength": [0.1, 0.2, 0.3, 0.4],
            "algebraic_connectivity": [0.0, 0.1, 0.2, 0.3],
            "largest_laplacian_eigenvalue_share": [0.5, 0.4, 0.45, 0.5],
            "laplacian_frobenius_change": [np.nan, 0.1, 0.2, 0.15],
            "modularity": [np.nan, np.nan, np.nan, np.nan],
        },
        index=index,
    )

    fig = plot_feature_panel(features)

    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_stress_boxplot_returns_figure() -> None:
    aligned = pd.DataFrame(
        {
            "regime_indicator": [0.1, 0.2, 1.0, 1.2],
            "systemic_stress_label": [0, 0, 1, 1],
        }
    )

    fig = plot_stress_boxplot(aligned)

    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_stress_boxplot_raises_when_only_one_class_exists() -> None:
    aligned = pd.DataFrame(
        {
            "regime_indicator": [0.1, 0.2, 0.3],
            "systemic_stress_label": [0, 0, 0],
        }
    )

    with pytest.raises(ValueError, match="Both stress and non-stress"):
        plot_stress_boxplot(aligned)


def test_plot_scatter_indicator_target_returns_figure() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    indicator = pd.Series([0.0, 0.5, 1.0, 1.5, 2.0], index=index)
    target = pd.Series([0.1, 0.2, 0.4, 0.6, 0.9], index=index, name="future_vol")

    fig = plot_scatter_indicator_target(indicator, target)

    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_regime_class_returns_returns_figure() -> None:
    summary = pd.DataFrame(
        {
            "n_obs": [10, 20, 10],
            "mean_return": [0.001, 0.0, -0.001],
            "volatility": [0.01, 0.015, 0.03],
            "mean_absolute_return": [0.008, 0.01, 0.02],
        },
        index=[
            "low_systemic_connectedness",
            "normal",
            "high_systemic_connectedness",
        ],
    )

    fig = plot_regime_class_returns(summary, metric="volatility")

    assert isinstance(fig, Figure)
    plt.close(fig)
