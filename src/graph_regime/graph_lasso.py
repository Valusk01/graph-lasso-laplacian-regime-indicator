"""Sparse precision-matrix estimation and partial-correlation conversion."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_graphical_lasso(
    window: pd.DataFrame,
    alpha: float,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a sparse precision matrix from a standardized return window.

    Graphical lasso estimates a sparse inverse covariance matrix. Under Gaussian
    assumptions, zero precision entries correspond to conditional independence.
    Outside the Gaussian case, the safer interpretation is sparse linear
    conditional-dependence / partial-correlation topology.

    Graphical lasso does not produce a graph Laplacian directly; the precision
    matrix must first be converted into partial correlations before graph
    construction.
    """

    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if not isinstance(window, pd.DataFrame):
        raise TypeError("window must be a pandas DataFrame.")
    if window.shape[0] < 2:
        raise ValueError("window must contain at least two observations.")
    if window.shape[1] < 2:
        raise ValueError("window must contain at least two assets.")

    try:
        from sklearn.covariance import GraphicalLasso
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for fit_graphical_lasso. "
            "Install the project dependencies before running the rolling engine."
        ) from exc

    values = window.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("window must contain only finite values.")

    model = GraphicalLasso(alpha=alpha, max_iter=max_iter, assume_centered=True)
    model.fit(values)

    return np.asarray(model.covariance_, dtype=float), np.asarray(
        model.precision_,
        dtype=float,
    )


def precision_to_partial_correlation(precision: np.ndarray) -> np.ndarray:
    """Convert a sparse precision matrix into partial correlations.

    Partial correlations normalize each off-diagonal precision entry by the
    conditional variances, yielding signed conditional-dependence strengths
    between asset pairs after controlling for the rest of the universe.
    """

    precision_array = np.asarray(precision, dtype=float)
    if precision_array.ndim != 2 or precision_array.shape[0] != precision_array.shape[1]:
        raise ValueError("precision must be a square matrix.")
    if not np.isfinite(precision_array).all():
        raise ValueError("precision must contain only finite values.")

    diagonal = np.diag(precision_array)
    if np.any(diagonal <= 0):
        raise ValueError("precision diagonal entries must be positive.")

    denominator = np.sqrt(np.outer(diagonal, diagonal))
    with np.errstate(divide="raise", invalid="raise"):
        partial_corr = -precision_array / denominator

    partial_corr = np.clip(partial_corr, -1.0, 1.0)
    np.fill_diagonal(partial_corr, 1.0)
    partial_corr = 0.5 * (partial_corr + partial_corr.T)

    return partial_corr
