import numpy as np
import pandas as pd

from graph_regime.component_scores import add_component_scores
from graph_regime.pca_baselines import compute_rolling_pca_features
from graph_regime.phase6 import SampleSplits
from graph_regime.phase7 import (
    run_graph_component_ablation,
    run_model_ladder_incremental_tests,
)


def _synthetic_inputs() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    rng = np.random.default_rng(11)
    index = pd.date_range("2020-01-01", periods=280, freq="B")
    signal = np.linspace(-1.0, 1.0, len(index))
    returns = pd.DataFrame(
        {
            "a": 0.001 * signal + rng.normal(0.0, 0.01, len(index)),
            "b": -0.001 * signal + rng.normal(0.0, 0.012, len(index)),
            "c": rng.normal(0.0, 0.009, len(index)),
        },
        index=index,
    )
    graph_features = pd.DataFrame(
        {
            "regime_indicator": signal + rng.normal(0.0, 0.1, len(index)),
            "average_graph_strength": signal,
            "algebraic_connectivity": signal * 0.5,
            "laplacian_frobenius_change": np.abs(np.gradient(signal)),
            "largest_laplacian_eigenvalue_share": signal * 0.2,
            "weighted_edge_density": signal * 0.3,
            "average_node_strength": signal * 0.4,
        },
        index=index,
    )
    benchmarks = pd.DataFrame(
        {
            "vix": 20.0 + signal,
            "realized_volatility": 0.1 + 0.01 * signal,
            "drawdown": -0.05 + 0.01 * signal,
            "average_correlation": 0.2 + 0.01 * signal,
            "average_absolute_correlation": 0.25 + 0.01 * signal,
            "systemic_stress_label": (signal > 0.7).astype(int),
        },
        index=index,
    )
    pca = compute_rolling_pca_features(returns, window=20)
    components = add_component_scores(
        graph_features,
        benchmarks=benchmarks,
        residualization_method="full_sample",
    )
    return returns, graph_features, benchmarks, pca.join(components, how="right")


def test_model_ladder_returns_expected_columns() -> None:
    returns, graph_features, benchmarks, combined = _synthetic_inputs()
    pca = combined[[column for column in combined.columns if column.startswith("pca_")]]
    components = combined.drop(columns=pca.columns)
    splits = SampleSplits(
        development_start="2020-01-01",
        development_end="2020-05-31",
        validation_start="2020-06-01",
        validation_end="2020-09-30",
        test_start="2020-10-01",
    )

    ladder = run_model_ladder_incremental_tests(
        graph_features,
        benchmarks=benchmarks,
        returns=returns,
        pca_features=pca,
        component_scores=components,
        splits=splits,
        horizons=(5,),
    )

    assert not ladder.empty
    assert {"target", "model", "regressor", "adjusted_r2", "oos_auc", "f1"} <= set(
        ladder.columns,
    )
    assert "M4_benchmarks_plus_pca_plus_graph_components" in set(ladder["model"])


def test_graph_component_ablation_outputs_expected_configurations() -> None:
    returns, graph_features, benchmarks, combined = _synthetic_inputs()
    components = add_component_scores(graph_features)
    splits = SampleSplits(
        development_start="2020-01-01",
        development_end="2020-05-31",
        validation_start="2020-06-01",
        validation_end="2020-09-30",
        test_start="2020-10-01",
    )

    ablation = run_graph_component_ablation(
        components,
        benchmarks=benchmarks,
        returns=returns,
        splits=splits,
        horizon=5,
        min_history=5,
    )

    assert {
        "all_graph_components",
        "excluding_laplacian_frobenius_change",
        "connectivity_only",
        "transition_only",
        "spectral_only",
    } <= set(ablation["ablation"])
    assert {"overlay_sharpe", "overlay_calmar", "annualized_turnover_proxy"} <= set(
        ablation.columns,
    )
