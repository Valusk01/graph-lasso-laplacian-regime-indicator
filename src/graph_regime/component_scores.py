"""Graph-Laplacian component scores and benchmark residualization."""

from __future__ import annotations

import numpy as np
import pandas as pd

GRAPH_COMPONENT_COLUMNS = [
    "average_graph_strength",
    "algebraic_connectivity",
    "laplacian_frobenius_change",
    "largest_laplacian_eigenvalue_share",
    "weighted_edge_density",
    "average_node_strength",
]

BENCHMARK_CONTROL_COLUMNS = [
    "vix",
    "realized_volatility",
    "drawdown",
    "average_correlation",
    "average_absolute_correlation",
]

ORTHOGONAL_COMPONENT_MAP = {
    "average_graph_strength": "orthogonal_average_graph_strength",
    "algebraic_connectivity": "orthogonal_algebraic_connectivity",
    "laplacian_frobenius_change": "orthogonal_laplacian_frobenius_change",
    "largest_laplacian_eigenvalue_share": "orthogonal_largest_eigenvalue_share",
}


def add_component_scores(
    graph_features: pd.DataFrame,
    benchmarks: pd.DataFrame | None = None,
    residualization_method: str = "expanding",
    min_history: int = 60,
) -> pd.DataFrame:
    """Add graph component block scores and optional benchmark residuals.

    The block scores test whether individual graph-Laplacian features are more
    informative than the composite RI. Their z-scores are full-sample research
    diagnostics; use expanding or training-sample scaling for a strict
    real-time deployment study. Residualized columns remove linear benchmark
    exposure using expanding regressions by default, so each residual uses only
    prior benchmark/component history.
    """

    if not isinstance(graph_features, pd.DataFrame):
        raise TypeError("graph_features must be a pandas DataFrame.")
    if residualization_method not in {"expanding", "full_sample"}:
        raise ValueError("residualization_method must be 'expanding' or 'full_sample'.")
    if min_history <= 1:
        raise ValueError("min_history must be greater than one.")

    output = graph_features.copy()
    available_components = [
        column for column in GRAPH_COMPONENT_COLUMNS if column in output
    ]
    for column in available_components:
        output[column] = pd.to_numeric(output[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        output[f"z_{column}"] = _safe_z(output[column])

    output["connectivity_score"] = _sum_available_z(
        output,
        ["average_graph_strength", "algebraic_connectivity", "weighted_edge_density"],
    )
    output["transition_score"] = output.get(
        "z_laplacian_frobenius_change",
        pd.Series(0.0, index=output.index),
    )
    output["spectral_score"] = output.get(
        "z_largest_laplacian_eigenvalue_share",
        pd.Series(0.0, index=output.index),
    )
    output["graph_components_equal_weight_score"] = _mean_available_z(
        output,
        available_components,
    )

    if benchmarks is not None:
        orthogonal = residualize_graph_components(
            output,
            benchmarks=benchmarks,
            method=residualization_method,
            min_history=min_history,
        )
        output = output.join(orthogonal, how="left")
        orthogonal_columns = list(orthogonal.columns)
        component_columns = [
            column
            for column in orthogonal_columns
            if column != "orthogonal_graph_score"
        ]
        if component_columns:
            output["orthogonal_graph_score"] = _mean_safe_z(output[component_columns])

    return output


def residualize_graph_components(
    graph_features: pd.DataFrame,
    benchmarks: pd.DataFrame,
    method: str = "expanding",
    min_history: int = 60,
) -> pd.DataFrame:
    """Residualize selected graph components against benchmark risk variables.

    ``method="expanding"`` fits each residual using only prior observations.
    ``method="full_sample"`` is diagnostics-only because it uses the full
    sample to estimate residualization coefficients.
    """

    if not isinstance(graph_features, pd.DataFrame):
        raise TypeError("graph_features must be a pandas DataFrame.")
    if not isinstance(benchmarks, pd.DataFrame):
        raise TypeError("benchmarks must be a pandas DataFrame.")
    if method not in {"expanding", "full_sample"}:
        raise ValueError("method must be 'expanding' or 'full_sample'.")
    if min_history <= 1:
        raise ValueError("min_history must be greater than one.")

    controls = [
        column for column in BENCHMARK_CONTROL_COLUMNS if column in benchmarks.columns
    ]
    if not controls:
        return pd.DataFrame(index=graph_features.index)

    aligned_controls = benchmarks[controls].apply(pd.to_numeric, errors="coerce")
    rows: dict[str, pd.Series] = {}
    for source_column, output_column in ORTHOGONAL_COMPONENT_MAP.items():
        if source_column not in graph_features.columns:
            continue
        component = pd.to_numeric(graph_features[source_column], errors="coerce")
        frame = pd.concat(
            [component.rename("component"), aligned_controls],
            axis=1,
            join="inner",
        ).replace([np.inf, -np.inf], np.nan)
        if method == "full_sample":
            rows[output_column] = _full_sample_residuals(frame).reindex(
                graph_features.index
            )
        else:
            rows[output_column] = _expanding_residuals(
                frame,
                min_history=min_history,
            ).reindex(graph_features.index)

    output = pd.DataFrame(rows, index=graph_features.index)
    if not output.empty:
        output["orthogonal_graph_score"] = _mean_safe_z(output)
    return output


def _full_sample_residuals(frame: pd.DataFrame) -> pd.Series:
    valid = frame.dropna()
    residuals = pd.Series(np.nan, index=frame.index, dtype=float)
    if valid.shape[0] <= valid.shape[1]:
        return residuals
    y = valid["component"].to_numpy(dtype=float)
    x = valid.drop(columns=["component"]).to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.pinv(x_design) @ y
    residuals.loc[valid.index] = y - x_design @ beta
    return residuals


def _expanding_residuals(frame: pd.DataFrame, min_history: int) -> pd.Series:
    residuals = pd.Series(np.nan, index=frame.index, dtype=float)
    for position, index_value in enumerate(frame.index):
        current = frame.iloc[position]
        if current.isna().any():
            continue
        history = frame.iloc[:position].dropna()
        if history.shape[0] < min_history:
            continue
        x_history = history.drop(columns=["component"]).to_numpy(dtype=float)
        y_history = history["component"].to_numpy(dtype=float)
        if history.shape[0] <= x_history.shape[1] + 1:
            continue
        x_design = np.column_stack([np.ones(len(x_history)), x_history])
        beta = np.linalg.pinv(x_design) @ y_history
        x_current = np.r_[1.0, current.drop(labels=["component"]).to_numpy(dtype=float)]
        residuals.loc[index_value] = float(current["component"] - x_current @ beta)
    return residuals


def _sum_available_z(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    z_columns = [f"z_{column}" for column in columns if f"z_{column}" in frame.columns]
    if not z_columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return frame[z_columns].sum(axis=1, skipna=True).astype(float)


def _mean_available_z(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    z_columns = [f"z_{column}" for column in columns if f"z_{column}" in frame.columns]
    if not z_columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return frame[z_columns].mean(axis=1, skipna=True).fillna(0.0).astype(float)


def _mean_safe_z(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    z_frame = pd.DataFrame(
        {column: _safe_z(frame[column]) for column in frame.columns},
        index=frame.index,
    )
    return z_frame.mean(axis=1, skipna=True).fillna(0.0).astype(float)


def _safe_z(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = numeric.mean(skipna=True)
    std = numeric.std(skipna=True, ddof=0)
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
   