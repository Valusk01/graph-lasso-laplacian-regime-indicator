"""Conversion from partial correlations to weighted graph Laplacians."""

from __future__ import annotations

import numpy as np


def partial_correlation_to_adjacency(
    partial_corr: np.ndarray,
    threshold: float = 1e-8,
) -> np.ndarray:
    """Build a weighted adjacency matrix from absolute partial correlations.

    Edge weights represent the magnitude of conditional dependence between
    assets. The sign is discarded because this Phase 1 regime signal focuses on
    systemic connectedness rather than hedge direction.
    """

    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    partial_corr_array = np.asarray(partial_corr, dtype=float)
    if (
        partial_corr_array.ndim != 2
        or partial_corr_array.shape[0] != partial_corr_array.shape[1]
    ):
        raise ValueError("partial_corr must be a square matrix.")
    if not np.isfinite(partial_corr_array).all():
        raise ValueError("partial_corr must contain only finite values.")

    adjacency = np.abs(partial_corr_array)
    np.fill_diagonal(adjacency, 0.0)
    adjacency[adjacency <= threshold] = 0.0
    adjacency = 0.5 * (adjacency + adjacency.T)

    return adjacency


def adjacency_to_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Convert a weighted asset-dependence graph into its combinatorial Laplacian.

    The Laplacian summarizes how strongly each asset is tied to the rest of the
    conditional-dependence network and is the basis for the spectral features
    used as systemic connectedness indicators.
    """

    adjacency_array = np.asarray(adjacency, dtype=float)
    if adjacency_array.ndim != 2 or adjacency_array.shape[0] != adjacency_array.shape[1]:
        raise ValueError("adjacency must be a square matrix.")
    if not np.isfinite(adjacency_array).all():
        raise ValueError("adjacency must contain only finite values.")
    if np.any(adjacency_array < 0):
        raise ValueError("adjacency weights must be non-negative.")

    adjacency_array = adjacency_array.copy()
    np.fill_diagonal(adjacency_array, 0.0)
    adjacency_array = 0.5 * (adjacency_array + adjacency_array.T)

    degrees = adjacency_array.sum(axis=1)
    laplacian = np.diag(degrees) - adjacency_array
    laplacian = 0.5 * (laplacian + laplacian.T)

    return laplacian
