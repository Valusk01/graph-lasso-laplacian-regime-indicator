import numpy as np
import pandas as pd

import graph_regime.indicator as indicator_module
from graph_regime.indicator import (
    compute_regime_indicator,
    compute_rolling_graph_features,
)


def _synthetic_returns(n_observations: int = 10, n_assets: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    market = rng.normal(scale=0.8, size=(n_observations, 1))
    idiosyncratic = rng.normal(scale=0.5, size=(n_observations, n_assets))
    loadings = np.linspace(0.4, 0.9, n_assets)
    values = market @ loadings.reshape(1, -1) + idiosyncratic
    index = pd.date_range("2020-01-01", periods=n_observations, freq="D")
    columns = [f"asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(values, index=index, columns=columns)


def test_compute_rolling_graph_features_shape_columns_and_no_infinite_values(monkeypatch) -> None:
    returns = _synthetic_returns()
    window = 5

    def fake_fit_graphical_lasso(window_frame, alpha, max_iter=200):
        n_assets = window_frame.shape[1]
        covariance = np.eye(n_assets)
        precision = np.eye(n_assets)
        precision[precision == 0.0] = -0.2
        return covariance, precision

    monkeypatch.setattr(
        indicator_module,
        "fit_graphical_lasso",
        fake_fit_graphical_lasso,
    )

    features = compute_rolling_graph_features(
        returns,
        window=window,
        alpha=0.8,
        min_non_missing=1.0,
        partial_corr_threshold=1e-6,
        max_iter=50,
        compute_modularity=False,
    )

    expected_columns = {
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
    assert features.shape[0] == len(returns) - window + 1
    assert expected_columns.issubset(features.columns)
    assert not np.isinf(features.to_numpy(dtype=float)).any()
    assert features["modularity"].isna().all()

    indicator = compute_regime_indicator(features)
    assert "regime_indicator" in indicator.columns
    assert np.isfinite(indicator["regime_indicator"].to_numpy()).all()


def test_compute_regime_indicator_adds_z_scores_and_handles_constant_columns() -> None:
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    features = pd.DataFrame(
        {
            "average_graph_strength": [1.0, 1.0, 1.0, 1.0],
            "weighted_edge_density": [0.2, 0.2, 0.2, 0.2],
            "algebraic_connectivity": [0.1, 0.1, 0.1, 0.1],
            "largest_laplacian_eigenvalue": [1.5, 1.5, 1.5, 1.5],
            "largest_laplacian_eigenvalue_share": [0.4, 0.4, 0.4, 0.4],
            "modularity": [0.3, 0.3, 0.3, 0.3],
            "laplacian_frobenius_change": [np.nan, 0.0, 0.0, 0.0],
            "number_of_edges": [3.0, 3.0, 3.0, 3.0],
            "average_node_strength": [1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )

    indicator = compute_regime_indicator(features)

    expected_z_columns = {
        "z_average_graph_strength",
        "z_algebraic_connectivity",
        "z_largest_laplacian_eigenvalue_share",
        "z_modularity",
        "z_laplacian_frobenius_change",
    }
    assert "regime_indicator" in indicator.columns
    assert expected_z_columns.issubset(indicator.columns)
    finite_columns = list(expected_z_columns | {"regime_indicator"})
    assert np.isfinite(indicator[finite_columns].to_numpy()).all()
    assert np.allclose(indicator["regime_indicator"], 0.0)

def test_compute_regime_indicator_handles_nan_modularity() -> None:
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    features = pd.DataFrame(
        {
            "average_graph_strength": [1.0, 1.2, 1.4, 1.3, 1.5],
            "weighted_edge_density": [0.2, 0.25, 0.3, 0.28, 0.35],
            "algebraic_connectivity": [0.1, 0.15, 0.2, 0.18, 0.25],
            "largest_laplacian_eigenvalue": [1.5, 1.6, 1.8, 1.7, 2.0],
            "largest_laplacian_eigenvalue_share": [0.4, 0.42, 0.45, 0.43, 0.47],
            "modularity": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "laplacian_frobenius_change": [np.nan, 0.1, 0.2, 0.15, 0.25],
            "number_of_edges": [3.0, 4.0, 5.0, 5.0, 6.0],
            "average_node_strength": [1.0, 1.2, 1.4, 1.3, 1.5],
        },
        index=index,
    )

    indicator = compute_regime_indicator(features)

    assert "regime_indicator" in indicator.columns
    assert "z_modularity" in indicator.columns
    assert np.isfinite(indicator["regime_indicator"].to_numpy()).all()
    assert np.allclose(indicator["z_modularity"], 0.0)
