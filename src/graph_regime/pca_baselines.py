"""Rolling PCA/correlation-spectrum baselines for regime research."""

from __future__ import annotations

import numpy as np
import pandas as pd

PCA_FEATURE_COLUMNS = [
    "pca_first_eigenvalue",
    "pca_first_eigenvalue_share",
    "pca_effective_rank",
    "pca_first_eigenvalue_change",
    "pca_effective_rank_change",
]


def compute_rolling_pca_features(
    returns: pd.DataFrame,
    window: int = 126,
    min_periods: int | None = None,
    min_non_missing: float = 0.8,
) -> pd.DataFrame:
    """Compute rolling correlation-spectrum features without look-ahead.

    Each row uses only returns in the trailing window ending at that date. The
    eigenvalues are estimated from rolling correlation matrices, not covariance
    matrices, so the baseline is scale-normalized and comparable with graph
    features that are computed after window-level standardization.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns must not be empty.")
    if window <= 1:
        raise ValueError("window must be greater than one.")
    if min_periods is None:
        min_periods = window
    if min_periods <= 1:
        raise ValueError("min_periods must be greater than one.")
    if min_periods > window:
        raise ValueError("min_periods cannot exceed window.")
    if not 0 < min_non_missing <= 1:
        raise ValueError("min_non_missing must be in the interval (0, 1].")

    numeric_returns = returns.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    rows: list[dict[str, float]] = []
    index: list[object] = []
    previous_first: float | None = None
    previous_rank: float | None = None

    for end_position in range(min_periods, len(numeric_returns) + 1):
        start_position = max(0, end_position - window)
        window_frame = numeric_returns.iloc[start_position:end_position]
        feature_row = _window_pca_features(
            window_frame,
            min_non_missing=min_non_missing,
        )

        first = feature_row["pca_first_eigenvalue"]
        rank = feature_row["pca_effective_rank"]
        feature_row["pca_first_eigenvalue_change"] = (
            first - previous_first
            if previous_first is not None and np.isfinite(first)
            else np.nan
        )
        feature_row["pca_effective_rank_change"] = (
            rank - previous_rank
            if previous_rank is not None and np.isfinite(rank)
            else np.nan
        )

        if np.isfinite(first):
            previous_first = first
        if np.isfinite(rank):
            previous_rank = rank

        rows.append(feature_row)
        index.append(numeric_returns.index[end_position - 1])

    return pd.DataFrame(rows, index=pd.Index(index, name=returns.index.name))


def _window_pca_features(
    window_frame: pd.DataFrame,
    min_non_missing: float,
) -> dict[str, float]:
    min_count = max(2, int(np.ceil(min_non_missing * len(window_frame))))
    usable_columns = [
        column
        for column in window_frame.columns
        if window_frame[column].count() >= min_count
    ]
    if len(usable_columns) < 2:
        return _nan_features()

    corr = window_frame[usable_columns].corr(min_periods=2)
    corr = corr.reindex(index=usable_columns, columns=usable_columns)
    matrix = corr.to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)

    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return _nan_features()

    eigenvalues = np.sort(np.clip(eigenvalues, 0.0, None))[::-1]
    total = float(eigenvalues.sum())
    if total <= 0 or not np.isfinite(total):
        return _nan_features()

    probabilities = eigenvalues / total
    positive_probabilities = probabilities[probabilities > 0]
    entropy = -float(np.sum(positive_probabilities * np.log(positive_probabilities)))
    effective_rank = float(np.exp(entropy))
    first_eigenvalue = float(eigenvalues[0])

    return {
        "pca_first_eigenvalue": first_eigenvalue,
        "pca_first_eigenvalue_share": float(first_eigenvalue / total),
        "pca_effective_rank": effective_rank,
        "pca_first_eigenvalue_change": np.nan,
        "pca_effective_rank_change": np.nan,
    }


def _nan_features() -> dict[str, float]:
    return {column: np.nan for column in PCA_FEATURE_COLUMNS}
