import numpy as np
import pandas as pd

import graph_regime.robustness as robustness_module
from graph_regime.robustness import (
    compute_benchmark_changes,
    compute_transition_correlations,
    create_stress_onset_labels,
    run_robustness_grid,
    summarize_robustness_diagnostics,
)


def test_create_stress_onset_labels_identifies_zero_to_one_after_gap() -> None:
    labels = pd.Series(
        [0, 0, 1, 1, 0, 1, 0, 0, 1],
        index=pd.date_range("2020-01-01", periods=9),
    )

    onset = create_stress_onset_labels(labels, min_gap=2)

    assert onset.iloc[2] == 1
    assert onset.iloc[5] == 0
    assert onset.iloc[8] == 1
    assert onset.sum() == 2


def test_benchmark_changes_and_transition_correlations() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    benchmarks = pd.DataFrame(
        {
            "vix": [10.0, 12.0, 15.0, 14.0, 20.0],
            "realized_volatility": [0.1, 0.2, 0.3, 0.2, 0.4],
        },
        index=index,
    )
    indicator = pd.Series([0.0, 1.0, 2.0, 1.0, 3.0], index=index)

    changes = compute_benchmark_changes(benchmarks)
    correlations = compute_transition_correlations(indicator, changes)

    assert "vix_change" in changes.columns
    assert "realized_volatility_change" in changes.columns
    assert set(correlations["benchmark"]) == {"vix_change", "realized_volatility_change"}
    assert "pearson_correlation" in correlations.columns


def test_summarize_robustness_diagnostics_small_synthetic_data() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    indicator = pd.Series(np.linspace(-1.0, 1.0, 8), index=index)
    benchmarks = pd.DataFrame(
        {
            "vix": np.linspace(10.0, 20.0, 8),
            "realized_volatility": np.linspace(0.1, 0.2, 8),
            "average_correlation": np.linspace(0.2, 0.4, 8),
            "systemic_stress_label": [0, 0, 0, 0, 1, 1, 1, 1],
        },
        index=index,
    )
    returns = pd.DataFrame(
        {
            "a": [0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02],
            "b": [0.02, -0.02, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01],
        },
        index=index,
    )
    convergence = pd.Series([True, True, False, True, True, True, True, True], index=index)

    summary = summarize_robustness_diagnostics(
        indicator,
        benchmarks=benchmarks,
        returns=returns,
        convergence=convergence,
        forward_horizon=2,
    )

    assert summary["n_ri_observations"] == 8
    assert np.isclose(summary["convergence_rate"], 7 / 8)
    assert "stress_non_stress_ri_difference" in summary
    assert "high_regime_volatility" in summary


def test_run_robustness_grid_with_monkeypatched_rolling_engine(monkeypatch) -> None:
    index = pd.date_range("2020-01-01", periods=8)
    returns = pd.DataFrame(
        {
            "a": np.linspace(-0.02, 0.03, 8),
            "b": np.linspace(0.01, -0.01, 8),
        },
        index=index,
    )
    benchmarks = pd.DataFrame(
        {
            "vix": np.linspace(10.0, 20.0, 8),
            "realized_volatility": np.linspace(0.1, 0.3, 8),
            "average_correlation": np.linspace(0.2, 0.5, 8),
            "systemic_stress_label": [0, 0, 0, 1, 1, 0, 1, 1],
        },
        index=index,
    )

    def fake_compute_rolling_graph_features(
        returns,
        window,
        alpha,
        min_non_missing=0.95,
        partial_corr_threshold=1e-6,
        max_iter=1000,
        compute_modularity=False,
        tol=1e-4,
        enet_tol=1e-4,
        mode="cd",
        on_non_convergence="record",
    ):
        feature_index = returns.index[window - 1 :]
        n = len(feature_index)
        return pd.DataFrame(
            {
                "average_graph_strength": np.linspace(1.0, 1.3, n),
                "weighted_edge_density": np.linspace(0.1, 0.2, n),
                "algebraic_connectivity": np.linspace(0.2, 0.4, n),
                "largest_laplacian_eigenvalue": np.linspace(1.0, 1.5, n),
                "largest_laplacian_eigenvalue_share": np.linspace(0.2, 0.3, n),
                "modularity": np.nan,
                "laplacian_frobenius_change": [np.nan] + list(np.linspace(0.1, 0.2, n - 1)),
                "number_of_edges": np.arange(n) + 3,
                "average_node_strength": np.linspace(1.0, 1.3, n),
                "graph_lasso_converged": [True] * n,
            },
            index=feature_index,
        )

    monkeypatch.setattr(
        robustness_module,
        "compute_rolling_graph_features",
        fake_compute_rolling_graph_features,
    )

    grid = run_robustness_grid(
        returns,
        benchmarks=benchmarks,
        alpha_values=(0.1,),
        windows=(3,),
        partial_corr_thresholds=(1e-6,),
        compute_modularity_values=(False,),
        forward_horizon=2,
    )

    assert grid.shape[0] == 1
    assert grid.loc[0, "alpha"] == 0.1
    assert grid.loc[0, "window"] == 3
    assert "correlation_with_future_realized_volatility" in grid.columns
