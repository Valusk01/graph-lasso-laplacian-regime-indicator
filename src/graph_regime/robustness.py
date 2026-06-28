"""Robustness and transition diagnostics for graph-regime research."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

from graph_regime.evaluation import (
    align_indicator_and_benchmarks,
    compute_forward_targets,
    summarize_regime_classes,
)
from graph_regime.indicator import compute_regime_indicator, compute_rolling_graph_features


def run_robustness_grid(
    returns: pd.DataFrame,
    benchmarks: pd.DataFrame,
    alpha_values: list[float] | tuple[float, ...] = (0.10,),
    windows: list[int] | tuple[int, ...] = (126,),
    partial_corr_thresholds: list[float] | tuple[float, ...] = (1e-6,),
    compute_modularity_values: list[bool] | tuple[bool, ...] = (False,),
    min_non_missing: float = 0.95,
    max_iter: int = 1000,
    tol: float = 1e-4,
    enet_tol: float = 1e-4,
    mode: str = "cd",
    forward_horizon: int = 21,
) -> pd.DataFrame:
    """Run a lightweight rolling-graph parameter grid and summarize diagnostics.

    The default grid is intentionally tiny. This helper is meant for research
    robustness checks, not exhaustive hyperparameter search.
    """

    _validate_grid(alpha_values, windows, partial_corr_thresholds, compute_modularity_values)
    rows: list[dict[str, float | int | bool]] = []

    for alpha, window, threshold, compute_modularity in product(
        alpha_values,
        windows,
        partial_corr_thresholds,
        compute_modularity_values,
    ):
        features = compute_rolling_graph_features(
            returns,
            window=window,
            alpha=alpha,
            min_non_missing=min_non_missing,
            partial_corr_threshold=threshold,
            max_iter=max_iter,
            compute_modularity=compute_modularity,
            tol=tol,
            enet_tol=enet_tol,
            mode=mode,
            on_non_convergence="record",
        )
        indicator = compute_regime_indicator(features)
        summary = summarize_robustness_diagnostics(
            indicator["regime_indicator"],
            benchmarks=benchmarks,
            returns=returns,
            convergence=features.get("graph_lasso_converged"),
            forward_horizon=forward_horizon,
        )
        summary.update(
            {
                "alpha": float(alpha),
                "window": int(window),
                "partial_corr_threshold": float(threshold),
                "compute_modularity": bool(compute_modularity),
            },
        )
        rows.append(summary)

    columns = [
        "alpha",
        "window",
        "partial_corr_threshold",
        "compute_modularity",
        "n_ri_observations",
        "convergence_rate",
        "ri_standard_deviation",
        "correlation_with_vix",
        "correlation_with_realized_volatility",
        "correlation_with_average_correlation",
        "stress_non_stress_ri_difference",
        "correlation_with_future_realized_volatility",
        "correlation_with_future_max_drawdown",
        "high_regime_volatility",
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_robustness_diagnostics(
    indicator: pd.Series,
    benchmarks: pd.DataFrame,
    returns: pd.Series | pd.DataFrame | None = None,
    convergence: pd.Series | None = None,
    forward_horizon: int = 21,
    stress_label_column: str = "systemic_stress_label",
) -> dict[str, float | int]:
    """Summarize robustness diagnostics for one RI series."""

    if not isinstance(indicator, pd.Series):
        raise TypeError("indicator must be a pandas Series.")
    if not isinstance(benchmarks, pd.DataFrame):
        raise TypeError("benchmarks must be a pandas DataFrame.")

    ri = pd.to_numeric(indicator, errors="coerce").rename("regime_indicator")
    aligned = align_indicator_and_benchmarks(ri, benchmarks)

    convergence_rate = np.nan
    if convergence is not None:
        aligned_convergence = _as_bool_numeric(convergence).reindex(ri.index)
        convergence_rate = float(aligned_convergence.dropna().mean())

    summary: dict[str, float | int] = {
        "n_ri_observations": int(ri.dropna().shape[0]),
        "convergence_rate": convergence_rate,
        "ri_standard_deviation": float(ri.std(ddof=0)),
        "correlation_with_vix": _pearson_from_aligned(aligned, "regime_indicator", "vix"),
        "correlation_with_realized_volatility": _pearson_from_aligned(
            aligned,
            "regime_indicator",
            "realized_volatility",
        ),
        "correlation_with_average_correlation": _pearson_from_aligned(
            aligned,
            "regime_indicator",
            "average_correlation",
        ),
        "stress_non_stress_ri_difference": _stress_difference(
            aligned,
            stress_label_column=stress_label_column,
        ),
        "correlation_with_future_realized_volatility": np.nan,
        "correlation_with_future_max_drawdown": np.nan,
        "high_regime_volatility": np.nan,
    }

    if returns is not None:
        targets = compute_forward_targets(returns, horizons=[forward_horizon])
        summary["correlation_with_future_realized_volatility"] = _pearson_pair(
            ri,
            targets[f"forward_realized_volatility_{forward_horizon}d"],
        )
        summary["correlation_with_future_max_drawdown"] = _pearson_pair(
            ri,
            targets[f"forward_max_drawdown_{forward_horizon}d"],
        )
        regime_summary = summarize_regime_classes(ri, returns=returns)
        if "high_systemic_connectedness" in regime_summary.index:
            summary["high_regime_volatility"] = float(
                regime_summary.loc["high_systemic_connectedness", "volatility"],
            )

    return summary


def compute_benchmark_changes(benchmarks: pd.DataFrame) -> pd.DataFrame:
    """Compute one-period changes in benchmark stress variables."""

    if not isinstance(benchmarks, pd.DataFrame):
        raise TypeError("benchmarks must be a pandas DataFrame.")

    columns = [
        "vix",
        "realized_volatility",
        "average_correlation",
        "average_absolute_correlation",
        "drawdown",
    ]
    changes: dict[str, pd.Series] = {}
    for column in columns:
        if column in benchmarks.columns:
            values = pd.to_numeric(benchmarks[column], errors="coerce").replace(
                [np.inf, -np.inf],
                np.nan,
            )
            changes[f"{column}_change"] = values.diff()

    if not changes:
        raise ValueError("benchmarks contains none of the supported columns.")

    return pd.DataFrame(changes).sort_index()


def compute_transition_correlations(
    indicator: pd.Series,
    benchmark_changes: pd.DataFrame,
) -> pd.DataFrame:
    """Correlate RI with changes in benchmark variables."""

    if not isinstance(indicator, pd.Series):
        raise TypeError("indicator must be a pandas Series.")
    if not isinstance(benchmark_changes, pd.DataFrame):
        raise TypeError("benchmark_changes must be a pandas DataFrame.")

    ri = pd.to_numeric(indicator, errors="coerce").rename("regime_indicator")
    rows: list[dict[str, float | str]] = []

    for column in benchmark_changes.columns:
        change = pd.to_numeric(benchmark_changes[column], errors="coerce")
        pair = pd.concat([ri, change.rename("benchmark_change")], axis=1).dropna()
        pearson, spearman = _correlations(pair["regime_indicator"], pair["benchmark_change"])
        rows.append(
            {
                "diagnostic_type": "benchmark_change_correlation",
                "benchmark": column,
                "n_obs": float(len(pair)),
                "pearson_correlation": pearson,
                "spearman_correlation": spearman,
            },
        )

    return pd.DataFrame(rows)


def create_stress_onset_labels(
    stress_labels: pd.Series | pd.DataFrame,
    min_gap: int = 20,
    stress_label_column: str = "systemic_stress_label",
) -> pd.Series:
    """Label 0-to-1 stress onsets after at least ``min_gap`` calm observations."""

    if min_gap < 0:
        raise ValueError("min_gap must be non-negative.")

    if isinstance(stress_labels, pd.DataFrame):
        if stress_label_column in stress_labels.columns:
            labels = stress_labels[stress_label_column]
        elif stress_labels.shape[1] == 1:
            labels = stress_labels.iloc[:, 0]
        else:
            raise KeyError(f"stress_labels is missing {stress_label_column!r}.")
    elif isinstance(stress_labels, pd.Series):
        labels = stress_labels
    else:
        raise TypeError("stress_labels must be a pandas Series or DataFrame.")

    numeric_labels = pd.to_numeric(labels, errors="coerce").fillna(0).astype(int)
    onset = pd.Series(0, index=numeric_labels.index, dtype=int, name="stress_onset_label")

    calm_run = 0
    previous = 0
    for index_value, current in numeric_labels.items():
        if current == 1 and previous == 0 and calm_run >= min_gap:
            onset.loc[index_value] = 1

        if current == 0:
            calm_run += 1
        else:
            calm_run = 0
        previous = current

    return onset


def compare_converged_vs_all_diagnostics(
    indicator: pd.Series,
    benchmarks: pd.DataFrame,
    convergence: pd.Series,
    returns: pd.Series | pd.DataFrame | None = None,
    forward_horizon: int = 21,
) -> pd.DataFrame:
    """Compare diagnostics using all windows versus converged windows only."""

    if not isinstance(convergence, pd.Series):
        raise TypeError("convergence must be a pandas Series.")

    all_summary = summarize_robustness_diagnostics(
        indicator,
        benchmarks=benchmarks,
        returns=returns,
        convergence=convergence,
        forward_horizon=forward_horizon,
    )
    converged_mask = _as_bool_numeric(convergence).reindex(indicator.index).fillna(0).astype(bool)
    converged_indicator = indicator.loc[converged_mask]
    converged_summary = summarize_robustness_diagnostics(
        converged_indicator,
        benchmarks=benchmarks,
        returns=returns,
        convergence=convergence.reindex(converged_indicator.index),
        forward_horizon=forward_horizon,
    )

    output = pd.DataFrame(
        [
            {"sample": "all_windows", **all_summary},
            {"sample": "converged_only", **converged_summary},
        ],
    )
    return output


def _validate_grid(
    alpha_values: list[float] | tuple[float, ...],
    windows: list[int] | tuple[int, ...],
    partial_corr_thresholds: list[float] | tuple[float, ...],
    compute_modularity_values: list[bool] | tuple[bool, ...],
) -> None:
    if not alpha_values or any(alpha <= 0 for alpha in alpha_values):
        raise ValueError("alpha_values must contain positive values.")
    if not windows or any(window <= 1 for window in windows):
        raise ValueError("windows must contain values greater than one.")
    if not partial_corr_thresholds or any(
        threshold < 0 for threshold in partial_corr_thresholds
    ):
        raise ValueError("partial_corr_thresholds must contain non-negative values.")
    if not compute_modularity_values:
        raise ValueError("compute_modularity_values must not be empty.")


def _stress_difference(
    aligned: pd.DataFrame,
    stress_label_column: str,
) -> float:
    if stress_label_column not in aligned.columns:
        return np.nan

    values = aligned[["regime_indicator", stress_label_column]].copy()
    values["regime_indicator"] = pd.to_numeric(values["regime_indicator"], errors="coerce")
    values[stress_label_column] = pd.to_numeric(values[stress_label_column], errors="coerce")
    values = values.dropna()

    stress = values.loc[values[stress_label_column] == 1, "regime_indicator"]
    non_stress = values.loc[values[stress_label_column] == 0, "regime_indicator"]
    if stress.empty or non_stress.empty:
        return np.nan
    return float(stress.mean() - non_stress.mean())


def _pearson_from_aligned(frame: pd.DataFrame, left: str, right: str) -> float:
    if right not in frame.columns or left not in frame.columns:
        return np.nan
    return _pearson_pair(frame[left], frame[right])


def _pearson_pair(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat(
        [
            pd.to_numeric(left, errors="coerce").rename("left"),
            pd.to_numeric(right, errors="coerce").rename("right"),
        ],
        axis=1,
        sort=False,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    pearson, _ = _correlations(pair["left"], pair["right"])
    return pearson


def _correlations(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    if len(left) < 3 or _is_constant(left) or _is_constant(right):
        return np.nan, np.nan
    pearson = stats.pearsonr(left.to_numpy(dtype=float), right.to_numpy(dtype=float)).statistic
    spearman = stats.spearmanr(
        left.to_numpy(dtype=float),
        right.to_numpy(dtype=float),
        nan_policy="omit",
    ).statistic
    return float(pearson), float(spearman)


def _is_constant(values: pd.Series) -> bool:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return finite.size < 2 or bool(np.isclose(finite.std(ddof=0), 0.0))


def _as_bool_numeric(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.astype(float)

    mapped = values.map(
        {
            True: 1.0,
            False: 0.0,
            "True": 1.0,
            "False": 0.0,
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
            1: 1.0,
            0: 0.0,
        },
    )
    numeric = pd.to_numeric(values, errors="coerce")
    return mapped.where(mapped.notna(), numeric)
