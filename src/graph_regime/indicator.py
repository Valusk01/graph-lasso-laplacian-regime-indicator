"""Rolling graph-feature computation and regime-indicator construction."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from graph_regime.features import compute_graph_features
from graph_regime.graph_lasso import (
    fit_graphical_lasso_with_diagnostics,
    precision_to_partial_correlation,
)
from graph_regime.laplacian import (
    adjacency_to_laplacian,
    partial_correlation_to_adjacency,
)
from graph_regime.preprocessing import clean_returns, standardize_window


REGIME_FEATURES = {
    "average_graph_strength": 1.0,
    "algebraic_connectivity": 1.0,
    "largest_laplacian_eigenvalue_share": 1.0,
    "modularity": -1.0,
    "laplacian_frobenius_change": 1.0,
}

GRAPH_LASSO_DIAGNOSTIC_COLUMNS = [
    "graph_lasso_converged",
    "graph_lasso_n_iter",
    "graph_lasso_alpha",
    "graph_lasso_max_iter",
    "graph_lasso_tol",
    "graph_lasso_enet_tol",
    "graph_lasso_mode",
    "graph_lasso_warning_count",
    "graph_lasso_warning_message",
]


def compute_rolling_graph_features(
    returns: pd.DataFrame,
    window: int,
    alpha: float,
    min_periods: int | None = None,
    min_non_missing: float = 0.95,
    partial_corr_threshold: float = 1e-8,
    max_iter: int = 200,
    compute_modularity: bool = False,
    tol: float = 1e-4,
    enet_tol: float = 1e-4,
    mode: str = "cd",
    on_non_convergence: str = "record",
) -> pd.DataFrame:
    """Estimate rolling graphical-lasso graphs and Laplacian features.

    Each row corresponds to one rolling window end-date. The sparse precision
    matrix is converted to partial correlations, then to a weighted adjacency
    matrix, then to a graph Laplacian before feature extraction.

    The same fixed alpha is used in every window to avoid injecting graph
    instability from per-window hyperparameter selection. The partial-correlation
    threshold is also fixed across windows so that changes in edge structure are
    not driven by changing filtering rules.

    Modularity is disabled by default because NetworkX community detection can
    be computationally expensive, unavailable, or unstable in repeated rolling
    windows. When disabled, modularity is reported as NaN and its regime
    indicator z-score is neutralized to zero.

    Graphical-lasso convergence diagnostics are recorded for every rolling
    window. ConvergenceWarning messages are captured inside the estimator so
    workflows do not spam the terminal. on_non_convergence controls whether
    non-converged windows are only recorded, summarized with one warning, or
    treated as an error.
    """

    if window <= 1:
        raise ValueError("window must be greater than one.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if min_periods is None:
        min_periods = window
    if min_periods <= 1:
        raise ValueError("min_periods must be greater than one.")
    if min_periods > window:
        raise ValueError("min_periods cannot exceed window.")
    if not 0 < min_non_missing <= 1:
        raise ValueError("min_non_missing must be in the interval (0, 1].")
    if partial_corr_threshold < 0:
        raise ValueError("partial_corr_threshold must be non-negative.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if enet_tol <= 0:
        raise ValueError("enet_tol must be positive.")
    if mode not in {"cd", "lars"}:
        raise ValueError("mode must be either 'cd' or 'lars'.")
    if on_non_convergence not in {"record", "warn", "raise"}:
        raise ValueError("on_non_convergence must be 'record', 'warn', or 'raise'.")

    cleaned_returns = clean_returns(returns, min_non_missing=min_non_missing)

    if cleaned_returns.shape[1] < 2:
        raise ValueError("returns must contain at least two usable assets.")

    if cleaned_returns.shape[0] < min_periods:
        raise ValueError("returns must contain enough complete observations.")

    rows: list[dict[str, object]] = []
    index: list[object] = []
    previous_laplacian: np.ndarray | None = None
    non_converged_dates: list[object] = []

    for end_position in range(min_periods, len(cleaned_returns) + 1):
        start_position = max(0, end_position - window)
        raw_window = cleaned_returns.iloc[start_position:end_position]
        if len(raw_window) < min_periods:
            continue

        standardized_window = standardize_window(raw_window)
        if standardized_window.shape[0] < min_periods:
            continue

        fit = fit_graphical_lasso_with_diagnostics(
            standardized_window,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            enet_tol=enet_tol,
            mode=mode,
        )
        precision = fit.precision
        partial_corr = precision_to_partial_correlation(precision)
        adjacency = partial_correlation_to_adjacency(
            partial_corr,
            threshold=partial_corr_threshold,
        )
        laplacian = adjacency_to_laplacian(adjacency)
        feature_row = compute_graph_features(
            adjacency=adjacency,
            laplacian=laplacian,
            previous_laplacian=previous_laplacian,
            compute_modularity=compute_modularity,
        )
        feature_row.update(
            {
                "graph_lasso_converged": fit.converged,
                "graph_lasso_n_iter": fit.n_iter,
                "graph_lasso_alpha": fit.alpha,
                "graph_lasso_max_iter": fit.max_iter,
                "graph_lasso_tol": fit.tol,
                "graph_lasso_enet_tol": fit.enet_tol,
                "graph_lasso_mode": fit.mode,
                "graph_lasso_warning_count": len(fit.warning_messages),
                "graph_lasso_warning_message": (
                    fit.warning_messages[0] if fit.warning_messages else ""
                ),
            }
        )

        rows.append(feature_row)
        window_end_date = cleaned_returns.index[end_position - 1]
        index.append(window_end_date)
        if not fit.converged:
            non_converged_dates.append(window_end_date)
        previous_laplacian = laplacian

    if non_converged_dates:
        non_convergence_count = len(non_converged_dates)
        total_windows = len(rows)
        if on_non_convergence == "raise":
            raise RuntimeError(
                "GraphicalLasso did not converge in "
                f"{non_convergence_count}/{total_windows} rolling windows. "
                f"First failed window end-date: {non_converged_dates[0]!r}."
            )
        if on_non_convergence == "warn":
            warnings.warn(
                "GraphicalLasso did not converge in "
                f"{non_convergence_count}/{total_windows} rolling windows. "
                "See graph_lasso_converged and graph_lasso_warning_count.",
                RuntimeWarning,
                stacklevel=2,
            )

    return pd.DataFrame(rows, index=pd.Index(index, name=returns.index.name))


def compute_regime_indicator(features: pd.DataFrame) -> pd.DataFrame:
    """Add standardized feature components and the composite regime indicator.

    High values indicate a more connected, less modular, more systemic market
    state. Z-scores are computed across the available feature time series, not
    within each rolling window. Constant or unavailable components receive a
    neutral zero z-score to avoid division-by-zero explosions.
    """

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")

    missing_columns = [column for column in REGIME_FEATURES if column not in features.columns]
    if missing_columns:
        raise KeyError(f"features is missing required columns: {missing_columns}")

    output = features.copy()
    regime_indicator = pd.Series(0.0, index=output.index, dtype=float)

    for feature_name, sign in REGIME_FEATURES.items():
        z_column = f"z_{feature_name}"
        output[z_column] = _safe_z_score(output[feature_name])
        regime_indicator = regime_indicator + sign * output[z_column]

    output["regime_indicator"] = regime_indicator.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return output


def summarize_graph_lasso_convergence(features: pd.DataFrame) -> dict[str, float | int]:
    """Summarize rolling graphical-lasso convergence diagnostics."""

    if "graph_lasso_converged" not in features.columns:
        raise KeyError("features is missing graph_lasso_converged.")

    converged = features["graph_lasso_converged"].astype(bool)
    n_windows = int(len(converged))
    n_converged = int(converged.sum())
    n_non_converged = int(n_windows - n_converged)
    convergence_rate = float(n_converged / n_windows) if n_windows else np.nan

    return {
        "n_windows": n_windows,
        "n_converged": n_converged,
        "n_non_converged": n_non_converged,
        "convergence_rate": convergence_rate,
    }


def _safe_z_score(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = numeric.mean(skipna=True)
    std = numeric.std(skipna=True, ddof=0)

    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        return pd.Series(0.0, index=series.index, dtype=float)

    z_score = (numeric - mean) / std
    return z_score.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
