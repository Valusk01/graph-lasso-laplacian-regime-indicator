"""Matplotlib visual diagnostics for graph-regime research workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


DEFAULT_FEATURE_COLUMNS = [
    "average_graph_strength",
    "algebraic_connectivity",
    "largest_laplacian_eigenvalue_share",
    "laplacian_frobenius_change",
    "modularity",
]


def plot_regime_indicator(
    indicator: pd.DataFrame | pd.Series,
    indicator_column: str = "regime_indicator",
    stress_labels: pd.DataFrame | None = None,
    stress_label_column: str = "systemic_stress_label",
    title: str = "Graph-Lasso Laplacian Regime Indicator",
    output_path: str | Path | None = None,
) -> Figure:
    """Plot the regime indicator through time, optionally marking stress periods."""

    indicator_series = _extract_indicator_series(indicator, indicator_column)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(indicator_series.index, indicator_series, label="Regime indicator", linewidth=1.5)

    if stress_labels is not None and stress_label_column in stress_labels.columns:
        stress = pd.to_numeric(
            stress_labels[stress_label_column].reindex(indicator_series.index),
            errors="coerce",
        )
        stress_points = indicator_series.loc[stress == 1]
        if not stress_points.empty:
            ax.scatter(
                stress_points.index,
                stress_points,
                color="tab:red",
                s=18,
                alpha=0.75,
                label="Benchmark stress",
                zorder=3,
            )

    ax.set_title(title)
    ax.set_ylabel("Regime indicator")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    _save_figure(fig, output_path)
    return fig


def plot_indicator_vs_benchmark(
    indicator: pd.DataFrame | pd.Series,
    benchmark: pd.Series,
    indicator_column: str = "regime_indicator",
    benchmark_name: str | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """Plot the regime indicator and one continuous benchmark on twin y-axes."""

    if not isinstance(benchmark, pd.Series):
        raise TypeError("benchmark must be a pandas Series.")

    indicator_series = _extract_indicator_series(indicator, indicator_column)
    benchmark_label = benchmark_name or benchmark.name or "benchmark"
    benchmark_series = pd.to_numeric(benchmark, errors="coerce").rename(benchmark_label)
    aligned = pd.concat([indicator_series, benchmark_series], axis=1, join="inner").sort_index()
    if aligned.empty:
        raise ValueError("indicator and benchmark have no overlapping observations.")

    fig, ax_indicator = plt.subplots(figsize=(11, 5))
    ax_benchmark = ax_indicator.twinx()

    indicator_line = ax_indicator.plot(
        aligned.index,
        aligned["regime_indicator"],
        label="Regime indicator",
        linewidth=1.5,
    )
    benchmark_line = ax_benchmark.plot(
        aligned.index,
        aligned[benchmark_label],
        color="tab:orange",
        label=benchmark_label,
        linewidth=1.3,
        alpha=0.85,
    )

    ax_indicator.set_title(title or f"Regime Indicator vs {benchmark_label}")
    ax_indicator.set_ylabel("Regime indicator")
    ax_benchmark.set_ylabel(benchmark_label)
    ax_indicator.grid(True, alpha=0.25)

    lines = indicator_line + benchmark_line
    ax_indicator.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.autofmt_xdate()
    _save_figure(fig, output_path)
    return fig


def plot_feature_panel(
    features: pd.DataFrame,
    feature_columns: list[str] | None = None,
    title: str = "Rolling Graph-Laplacian Features",
    output_path: str | Path | None = None,
) -> Figure:
    """Plot selected rolling graph-Laplacian features in stacked panels."""

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")

    if feature_columns is None:
        candidate_columns = [column for column in DEFAULT_FEATURE_COLUMNS if column in features.columns]
    else:
        missing = [column for column in feature_columns if column not in features.columns]
        if missing:
            raise KeyError(f"features is missing required columns: {missing}")
        candidate_columns = feature_columns

    valid_columns = [
        column
        for column in candidate_columns
        if pd.to_numeric(features[column], errors="coerce").notna().any()
    ]
    if not valid_columns:
        raise ValueError("No valid feature columns are available to plot.")

    fig, axes = plt.subplots(
        nrows=len(valid_columns),
        ncols=1,
        figsize=(11, 2.6 * len(valid_columns)),
        sharex=True,
    )
    axes_array = np.atleast_1d(axes)

    for ax, column in zip(axes_array, valid_columns, strict=True):
        series = pd.to_numeric(features[column], errors="coerce")
        ax.plot(series.index, series, linewidth=1.3)
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.25)

    axes_array[0].set_title(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_stress_boxplot(
    aligned: pd.DataFrame,
    indicator_column: str = "regime_indicator",
    stress_label_column: str = "systemic_stress_label",
    title: str = "Regime Indicator During Stress vs Non-Stress Periods",
    output_path: str | Path | None = None,
) -> Figure:
    """Compare regime indicator distributions during stress and non-stress periods."""

    _require_columns(aligned, [indicator_column, stress_label_column])
    data = aligned[[indicator_column, stress_label_column]].copy()
    data[indicator_column] = pd.to_numeric(data[indicator_column], errors="coerce")
    data[stress_label_column] = pd.to_numeric(data[stress_label_column], errors="coerce")
    data = data.dropna()

    non_stress = data.loc[data[stress_label_column] == 0, indicator_column]
    stress = data.loc[data[stress_label_column] == 1, indicator_column]
    if non_stress.empty or stress.empty:
        raise ValueError("Both stress and non-stress groups are required for a boxplot.")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([non_stress, stress], patch_artist=True)
    ax.set_xticklabels(["Non-stress", "Stress"])
    ax.set_title(title)
    ax.set_ylabel("Regime indicator")
    ax.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, output_path)
    return fig


def plot_scatter_indicator_target(
    indicator: pd.Series,
    target: pd.Series,
    target_name: str | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """Scatter plot the regime indicator against a forward target with an OLS line."""

    if not isinstance(indicator, pd.Series):
        raise TypeError("indicator must be a pandas Series.")
    if not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series.")

    indicator_series = pd.to_numeric(indicator, errors="coerce").rename("regime_indicator")
    target_label = target_name or target.name or "target"
    target_series = pd.to_numeric(target, errors="coerce").rename(target_label)
    aligned = pd.concat([indicator_series, target_series], axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("indicator and target have no overlapping finite observations.")

    fig, ax = plt.subplots(figsize=(7, 5))
    x = aligned["regime_indicator"].to_numpy(dtype=float)
    y = aligned[target_label].to_numpy(dtype=float)
    ax.scatter(x, y, alpha=0.75)

    if len(aligned) >= 3 and not np.isclose(np.std(x), 0.0):
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        y_line = intercept + slope * x_line
        ax.plot(x_line, y_line, color="tab:red", linewidth=1.3, label="OLS fit")
        ax.legend(loc="best")

    ax.set_title(title or f"Regime Indicator vs {target_label}")
    ax.set_xlabel("Regime indicator")
    ax.set_ylabel(target_label)
    ax.grid(True, alpha=0.25)
    _save_figure(fig, output_path)
    return fig


def plot_regime_class_returns(
    regime_class_summary: pd.DataFrame,
    metric: str = "volatility",
    title: str | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """Plot a selected return or risk metric by regime class."""

    if not isinstance(regime_class_summary, pd.DataFrame):
        raise TypeError("regime_class_summary must be a pandas DataFrame.")
    if metric not in regime_class_summary.columns:
        raise KeyError(f"regime_class_summary is missing metric column {metric!r}.")

    series = pd.to_numeric(regime_class_summary[metric], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"metric {metric!r} contains no finite values to plot.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(series.index.astype(str), series.to_numpy(dtype=float))
    ax.set_title(title or f"{metric} by Regime Class")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def _extract_indicator_series(
    indicator: pd.DataFrame | pd.Series,
    indicator_column: str,
) -> pd.Series:
    if isinstance(indicator, pd.DataFrame):
        if indicator_column not in indicator.columns:
            raise KeyError(f"indicator is missing column {indicator_column!r}.")
        series = indicator[indicator_column]
    elif isinstance(indicator, pd.Series):
        series = indicator
    else:
        raise TypeError("indicator must be a pandas DataFrame or Series.")

    numeric = pd.to_numeric(series, errors="coerce").rename("regime_indicator")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).sort_index()
    if numeric.dropna().empty:
        raise ValueError("indicator contains no finite values to plot.")
    return numeric


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("input must be a pandas DataFrame.")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"input is missing required columns: {missing}")


def _save_figure(fig: Figure, output_path: str | Path | None) -> None:
    if output_path is None:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
