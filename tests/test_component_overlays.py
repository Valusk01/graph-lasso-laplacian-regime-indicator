import numpy as np
import pandas as pd

from graph_regime.component_overlays import (
    build_component_overlay_signals,
    compute_component_overlay_exposures,
    evaluate_component_overlays,
)


def test_component_overlay_exposure_is_shifted() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    signals = pd.DataFrame(
        {"regime_indicator": [-3.0, -2.0, -1.0, 10.0, 0.0]}, index=index
    )

    exposures = compute_component_overlay_exposures(
        signals,
        method="full_sample",
        high_quantile=0.6,
        extreme_quantile=0.8,
    )

    exposure = exposures["regime_indicator_exposure"]
    assert exposure.iloc[0] == 1.0
    assert exposure.iloc[3] == 1.0
    assert exposure.iloc[4] == 0.5


def test_component_overlay_evaluation_outputs_metrics() -> None:
    index = pd.date_range("2020-01-01", periods=35)
    signal_index = index[5:]
    returns = pd.DataFrame(
        {
            "a": np.sin(np.arange(35)) * 0.01,
            "b": np.cos(np.arange(35)) * 0.01,
        },
        index=index,
    )
    graph_features = pd.DataFrame(
        {
            "regime_indicator": np.linspace(-1.0, 1.0, 30),
            "average_graph_strength": np.linspace(0.1, 0.5, 30),
            "algebraic_connectivity": np.linspace(0.2, 0.6, 30),
            "laplacian_frobenius_change": np.linspace(0.0, 0.2, 30),
            "largest_laplacian_eigenvalue_share": np.linspace(0.2, 0.3, 30),
            "weighted_edge_density": np.linspace(0.1, 0.2, 30),
            "average_node_strength": np.linspace(0.1, 0.4, 30),
        },
        index=signal_index,
    )

    signals = build_component_overlay_signals(graph_features)
    comparison, costs = evaluate_component_overlays(
        returns,
        signals[["regime_indicator", "connectivity_score"]],
        method="expanding",
        min_history=5,
        cost_bps_values=(0.0, 5.0),
    )

    assert {"baseline", "regime_indicator"} <= set(comparison["signal"])
    assert set(costs["cost_bps"]) == {0.0, 5.0}
    as