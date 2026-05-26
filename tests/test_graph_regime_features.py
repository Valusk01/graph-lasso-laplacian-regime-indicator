import numpy as np

from graph_regime.features import compute_graph_features
from graph_regime.laplacian import adjacency_to_laplacian


REQUIRED_FEATURE_KEYS = {
    "average_graph_strength",
    "weighted_edge_density",
    "algebraic_connectivity",
    "largest_laplacian_eigenvalue",
    "largest_laplacian_eigenvalue_share",
    "modularity",
    "laplacian_frobenius_change",
    "number_of_edges",
    "average_node_strength",
}


def test_compute_graph_features_contains_required_keys_and_valid_connectivity() -> None:
    adjacency = np.array(
        [
            [0.0, 0.3, 0.1],
            [0.3, 0.0, 0.2],
            [0.1, 0.2, 0.0],
        ]
    )
    laplacian = adjacency_to_laplacian(adjacency)

    features = compute_graph_features(adjacency, laplacian)

    assert REQUIRED_FEATURE_KEYS.issubset(features)
    assert features["algebraic_connectivity"] >= -1e-10
    assert np.isnan(features["laplacian_frobenius_change"])
    assert np.isnan(features["modularity"])


def test_compute_graph_features_reports_frobenius_change_with_previous_laplacian() -> None:
    previous_adjacency = np.array(
        [
            [0.0, 0.1, 0.0],
            [0.1, 0.0, 0.2],
            [0.0, 0.2, 0.0],
        ]
    )
    current_adjacency = np.array(
        [
            [0.0, 0.2, 0.1],
            [0.2, 0.0, 0.2],
            [0.1, 0.2, 0.0],
        ]
    )
    previous_laplacian = adjacency_to_laplacian(previous_adjacency)
    current_laplacian = adjacency_to_laplacian(current_adjacency)

    features = compute_graph_features(
        current_adjacency,
        current_laplacian,
        previous_laplacian=previous_laplacian,
    )

    assert features["laplacian_frobenius_change"] > 0.0


def test_compute_graph_features_skips_modularity_by_default() -> None:
    adjacency = np.array(
        [
            [0.0, 0.4, 0.1],
            [0.4, 0.0, 0.3],
            [0.1, 0.3, 0.0],
        ]
    )
    laplacian = adjacency_to_laplacian(adjacency)

    features = compute_graph_features(
        adjacency,
        laplacian,
        compute_modularity=False,
    )

    assert "modularity" in features
    assert np.isnan(features["modularity"])


def test_disconnected_graph_has_zero_algebraic_connectivity() -> None:
    adjacency = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    laplacian = adjacency_to_laplacian(adjacency)

    features = compute_graph_features(adjacency, laplacian)

    assert np.isclose(features["algebraic_connectivity"], 0.0)
