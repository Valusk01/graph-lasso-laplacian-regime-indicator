import numpy as np
import pandas as pd

from graph_regime.component_scores import (
    add_component_scores,
    residualize_graph_components,
)


def _graph_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    trend = np.linspace(0.0, 1.0, len(index))
    return pd.DataFrame(
        {
            "average_graph_strength": trend,
            "algebraic_connectivity": trend * 0.5,
            "laplacian_frobenius_change": np.abs(np.gradient(trend)),
            "largest_laplacian_eigenvalue_share": trend * 0.2,
            "weighted_edge_density": trend * 0.3,
            "average_node_strength": trend * 0.7,
        },
        index=index,
    )


def test_add_component_scores_creates_blocks() -> None:
    index = pd.date_range("2020-01-01", periods=20)
    scored = add_component_scores(_graph_features(index))

    assert "connectivity_score" in scored.columns
    assert "transition_score" in scored.columns
    assert "spectral_score" in scored.columns
    assert "graph_components_equal_weight_score" in scored.columns


def test_full_sample_orthogonalization_reduces_benchmark_correlation() -> None:
    index = pd.date_range("2020-01-01", periods=80)
    vix = pd.Series(np.linspace(10.0, 30.0, len(index)), index=index)
    graph_features = _graph_features(index)
    graph_features["average_graph_strength"] = 2.0 * vix.to_numpy() + np.sin(
        np.arange(len(index)),
    )
    benchmarks = pd.DataFrame(
        {
            "vix": vix,
            "realized_volatility": np.linspace(0.1, 0.2, len(index)),
        },
        index=index,
    )

    residuals = residualize_graph_components(
        graph_features,
        benchmarks=benchmarks,
        method="full_sample",
    )

    raw_corr = graph_features["average_graph_strength"].corr(benchmarks["vix"])
    residual_corr = residuals["orthogonal_average_graph_strength"].corr(
        benchmarks["vix"]
    )
    assert abs(residual_corr) < abs(raw_corr)
