import numpy as np

from graph_regime.laplacian import (
    adjacency_to_laplacian,
    partial_correlation_to_adjacency,
)


def test_partial_correlation_to_adjacency_has_valid_weights() -> None:
    partial_corr = np.array(
        [
            [1.0, -0.25, 0.0],
            [-0.25, 1.0, 0.5],
            [0.0, 0.5, 1.0],
        ]
    )

    adjacency = partial_correlation_to_adjacency(partial_corr)

    assert np.allclose(np.diag(adjacency), 0.0)
    assert np.allclose(adjacency, adjacency.T)
    assert np.all(adjacency >= 0.0)


def test_adjacency_to_laplacian_has_laplacian_structure() -> None:
    adjacency = np.array(
        [
            [0.0, 0.3, 0.2],
            [0.3, 0.0, 0.4],
            [0.2, 0.4, 0.0],
        ]
    )

    laplacian = adjacency_to_laplacian(adjacency)
    off_diagonal = laplacian[~np.eye(laplacian.shape[0], dtype=bool)]

    assert np.allclose(laplacian.sum(axis=1), 0.0)
    assert np.all(np.diag(laplacian) >= 0.0)
    assert np.all(off_diagonal <= 0.0)
