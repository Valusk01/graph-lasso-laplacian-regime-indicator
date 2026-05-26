"""Graph and Laplacian spectral features for regime research."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigvalsh


def compute_laplacian_spectrum(laplacian: np.ndarray) -> np.ndarray:
    """Return sorted eigenvalues of a weighted graph Laplacian.

    The spectrum describes global network structure: near-zero low eigenvalues
    indicate disconnected or weakly connected components, while larger dominant
    eigenvalues can indicate concentration of dependence in systemic modes.
    """

    laplacian_array = np.asarray(laplacian, dtype=float)
    if laplacian_array.ndim != 2 or laplacian_array.shape[0] != laplacian_array.shape[1]:
        raise ValueError("laplacian must be a square matrix.")
    if not np.isfinite(laplacian_array).all():
        raise ValueError("laplacian must contain only finite values.")

    symmetric_laplacian = 0.5 * (laplacian_array + laplacian_array.T)
    eigenvalues = np.sort(eigvalsh(symmetric_laplacian))
    eigenvalues[np.isclose(eigenvalues, 0.0, atol=1e-12)] = 0.0

    return eigenvalues


def compute_graph_features(
    adjacency: np.ndarray,
    laplacian: np.ndarray,
    previous_laplacian: np.ndarray | None = None,
    compute_modularity: bool = False,
) -> dict[str, float]:
    """Compute connectedness features for one rolling asset-return graph.

    Feature meanings:
    average_graph_strength / average_node_strength measure the typical total
    conditional-dependence weight attached to an asset. weighted_edge_density
    measures how much partial-correlation weight exists relative to a fully
    connected graph. algebraic_connectivity is the second-smallest Laplacian
    eigenvalue and rises when the network is harder to split into isolated
    blocks. largest_laplacian_eigenvalue and its share summarize dominance of a
    global spectral mode. modularity is high when the graph separates into
    communities, so lower modularity is consistent with less diversified,
    system-wide stress. Modularity is optional because community detection can
    be computationally expensive, unavailable, or unstable; when it is not
    computed it is reported as NaN and receives a neutral z-score in the regime
    indicator. laplacian_frobenius_change is NaN for the first window and then
    measures topology turnover relative to the previous window.
    """

    adjacency_array = np.asarray(adjacency, dtype=float)
    laplacian_array = np.asarray(laplacian, dtype=float)

    if adjacency_array.ndim != 2 or adjacency_array.shape[0] != adjacency_array.shape[1]:
        raise ValueError("adjacency must be a square matrix.")
    if laplacian_array.shape != adjacency_array.shape:
        raise ValueError("laplacian and adjacency must have the same shape.")
    if not np.isfinite(adjacency_array).all() or not np.isfinite(laplacian_array).all():
        raise ValueError("adjacency and laplacian must contain only finite values.")

    n_assets = adjacency_array.shape[0]
    upper_triangle = np.triu_indices(n_assets, k=1)
    edge_weights = adjacency_array[upper_triangle]
    total_edge_weight = float(edge_weights.sum())
    possible_edges = n_assets * (n_assets - 1) / 2
    number_of_edges = float(np.count_nonzero(edge_weights > 0.0))
    node_strengths = adjacency_array.sum(axis=1)
    average_node_strength = float(node_strengths.mean()) if n_assets else 0.0
    weighted_edge_density = (
        float(total_edge_weight / possible_edges) if possible_edges > 0 else 0.0
    )

    spectrum = compute_laplacian_spectrum(laplacian_array)
    algebraic_connectivity = float(max(spectrum[1], 0.0)) if n_assets >= 2 else 0.0
    largest_laplacian_eigenvalue = float(max(spectrum[-1], 0.0)) if n_assets else 0.0
    spectral_mass = float(np.clip(spectrum, 0.0, None).sum())
    largest_laplacian_eigenvalue_share = (
        float(largest_laplacian_eigenvalue / spectral_mass)
        if spectral_mass > 0
        else 0.0
    )

    if previous_laplacian is None:
        laplacian_frobenius_change = np.nan
    else:
        previous = np.asarray(previous_laplacian, dtype=float)
        if previous.shape != laplacian_array.shape:
            raise ValueError("previous_laplacian must match laplacian shape.")
        laplacian_frobenius_change = float(np.linalg.norm(laplacian_array - previous, ord="fro"))

    modularity = _compute_modularity(adjacency_array) if compute_modularity else np.nan

    return {
        "average_graph_strength": average_node_strength,
        "weighted_edge_density": weighted_edge_density,
        "algebraic_connectivity": algebraic_connectivity,
        "largest_laplacian_eigenvalue": largest_laplacian_eigenvalue,
        "largest_laplacian_eigenvalue_share": largest_laplacian_eigenvalue_share,
        "modularity": modularity,
        "laplacian_frobenius_change": laplacian_frobenius_change,
        "number_of_edges": number_of_edges,
        "average_node_strength": average_node_strength,
    }


def _compute_modularity(adjacency: np.ndarray) -> float:
    """Compute weighted modularity when NetworkX community tools are available."""

    if adjacency.shape[0] < 2 or np.count_nonzero(np.triu(adjacency, k=1)) == 0:
        return np.nan

    try:
        import networkx as nx
    except ImportError:
        return np.nan

    try:
        graph = nx.from_numpy_array(adjacency)
        communities = nx.community.greedy_modularity_communities(graph, weight="weight")
        if not communities:
            return np.nan
        return float(nx.community.modularity(graph, communities, weight="weight"))
    except Exception:
        return np.nan
