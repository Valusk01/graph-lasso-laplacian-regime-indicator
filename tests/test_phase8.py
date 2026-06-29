import numpy as np
import pandas as pd

from graph_regime.component_scores import add_component_scores
from graph_regime.phase6 import SampleSplits, compute_exposure_turnover
from graph_regime.phase8 import (
    build_final_candidate_selection_matrix,
    build_phase8_candidate_scores,
    compare_information_sets,
    compute_algebraic_connectivity_diagnostics,
    compute_low_turnover_exposure,
    evaluate_score_overlay_set,
    evaluate_turnover_reduction_overlays,
    make_rolling_origin_splits,
    rank_information_sets,
    run_rolling_origin_oos,
)


def _synthetic_inputs() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    rng = np.random.default_rng(17)
    index = pd.date_range("2019-01-01", periods=520, freq="B")
    trend = np.linspace(-1.0, 1.0, len(index))
    returns = pd.DataFrame(
        {
            "a": 0.0005 * trend + rng.normal(0.0, 0.01, len(index)),
            "b": -0.0003 * trend + rng.normal(0.0, 0.012, len(index)),
            "c": rng.normal(0.0, 0.009, len(index)),
        },
        index=index,
    )
    graph = pd.DataFrame(
        {
            "regime_indicator": trend + rng.normal(0.0, 0.1, len(index)),
            "average_graph_strength": trend,
            "algebraic_connectivity": 0.4 * trend + 0.1 * np.sin(np.arange(len(index))),
            "laplacian_frobenius_change": np.abs(np.gradient(trend)),
            "largest_laplacian_eigenvalue_share": 0.2 + 0.1 * trend,
            "weighted_edge_density": 0.2 + 0.2 * trend,
            "average_node_strength": 0.3 + 0.3 * trend,
        },
        index=index,
    )
    benchmarks = pd.DataFrame(
        {
            "vix": 20.0 + 2.0 * trend,
            "realized_volatility": 0.12 + 0.02 * trend,
            "drawdown": -0.05 - 0.02 * trend,
            "average_correlation": 0.25 + 0.03 * trend,
            "average_absolute_correlation": 0.3 + 0.03 * trend,
            "systemic_stress_label": 0,
        },
        index=index,
    )
    benchmarks.loc[index[120:150], "systemic_stress_label"] = 1
    benchmarks.loc[index[320:350], "systemic_stress_label"] = 1
    pca = pd.DataFrame(
        {
            "pca_first_eigenvalue": 1.5 + trend,
            "pca_first_eigenvalue_share": 0.4 + 0.1 * trend,
            "pca_effective_rank": 2.5 - 0.2 * trend,
            "pca_first_eigenvalue_change": np.gradient(1.5 + trend),
            "pca_effective_rank_change": np.gradient(2.5 - 0.2 * trend),
        },
        index=index,
    )
    return returns, graph, benchmarks, pca


def test_information_set_comparison_returns_expected_model_labels() -> None:
    returns, graph, benchmarks, pca = _synthetic_inputs()
    components = add_component_scores(graph)
    splits = SampleSplits(
        development_start="2019-01-01",
        development_end="2019-12-31",
        validation_start="2020-01-01",
        validation_end="2020-06-30",
        test_start="2020-07-01",
    )

    comparison = compare_information_sets(
        graph,
        benchmarks=benchmarks,
        returns=returns,
        pca_features=pca,
        component_scores=components,
        splits=splits,
        horizons=(5,),
    )
    ranking = rank_information_sets(comparison)

    assert {
        "benchmarks_only",
        "ri_only",
        "pca_only",
        "graph_components_only",
        "pca_plus_graph_components",
        "benchmarks_plus_pca_plus_graph_components",
    } <= set(comparison["information_set"])
    assert "future_realized_volatility_5d" in set(comparison["target"])
    assert "best_information_set" in ranking.columns


def test_algebraic_connectivity_diagnostics_handle_missing_component() -> None:
    returns, graph, benchmarks, pca = _synthetic_inputs()
    graph = graph.drop(columns=["algebraic_connectivity"])

    diagnostics = compute_algebraic_connectivity_diagnostics(
        graph,
        benchmarks=benchmarks,
        returns=returns,
        pca_features=pca,
    )

    row = diagnostics.iloc[0]
    assert row["diagnostic_type"] == "availability"
    assert row["value"] == 0.0


def test_turnover_reduction_variant_does_not_increase_changes() -> None:
    index = pd.date_range("2020-01-01", periods=40, freq="B")
    score = pd.Series(np.tile([0.0, 10.0], 20), index=index)

    daily = compute_low_turnover_exposure(
        score,
        variant="daily",
        method="full_sample",
        high_quantile=0.5,
        extreme_quantile=0.9,
    )
    weekly = compute_low_turnover_exposure(
        score,
        variant="weekly",
        method="full_sample",
        high_quantile=0.5,
        extreme_quantile=0.9,
    )

    daily_changes = compute_exposure_turnover(daily)["number_of_exposure_changes"]
    weekly_changes = compute_exposure_turnover(weekly)["number_of_exposure_changes"]
    assert weekly_changes <= daily_changes


def test_hysteresis_uses_separate_enter_exit_thresholds() -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    score = pd.Series([-1.0, 0.0, 5.0, 4.0, 3.0, -2.0], index=index)

    exposure = compute_low_turnover_exposure(
        score,
        variant="hysteresis",
        method="full_sample",
        high_quantile=0.5,
        extreme_quantile=0.95,
        hysteresis_exit_quantile=0.25,
    )

    assert exposure.iloc[3] < 1.0
    assert exposure.iloc[4] < 1.0


def test_cooldown_prevents_immediate_repeated_changes() -> None:
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    score = pd.Series(np.tile([0.0, 10.0], 6), index=index)

    daily = compute_low_turnover_exposure(
        score,
        variant="daily",
        method="full_sample",
        high_quantile=0.5,
        extreme_quantile=0.9,
    )
    cooldown = compute_low_turnover_exposure(
        score,
        variant="cooldown_5d",
        method="full_sample",
        high_quantile=0.5,
        extreme_quantile=0.9,
    )

    assert (
        compute_exposure_turnover(cooldown)["number_of_exposure_changes"]
        < compute_exposure_turnover(daily)["number_of_exposure_changes"]
    )


def test_rolling_origin_split_never_uses_future_test_data_for_selection() -> None:
    splits = make_rolling_origin_splits(test_years=(2020, 2021))

    for split in splits:
        assert split.selection_end < split.test_start
        assert split.test_start.year == split.test_year


def test_final_candidate_selection_matrix_contains_required_candidates() -> None:
    comparison = pd.DataFrame(
        {
            "target": ["stress_onset_label", "future_realized_volatility_21d"],
            "information_set": [
                "benchmarks_plus_graph_components",
                "benchmarks_plus_graph_components",
            ],
            "oos_auc": [0.7, np.nan],
            "oos_r2": [np.nan, 0.1],
        },
    )
    rolling = pd.DataFrame(
        {
            "candidate": ["ri", "graph_components_equal_weight_score"] * 2,
            "score": ["ri", "graph_components_equal_weight_score"] * 2,
            "variant": ["daily", "daily"] * 2,
            "cost_bps": [0.0, 0.0, 10.0, 10.0],
            "mean_sharpe": [0.4, 0.6, 0.3, 0.5],
            "mean_calmar": [0.2, 0.4, 0.15, 0.3],
            "mean_max_drawdown": [-0.2, -0.15, -0.22, -0.18],
            "mean_sortino": [0.5, 0.7, 0.4, 0.6],
            "mean_worst_5pct_return": [-0.02, -0.015, -0.025, -0.02],
            "mean_annualized_turnover_proxy": [5.0, 4.0, 5.0, 4.0],
            "share_years_positive_sharpe": [1.0, 1.0, 1.0, 1.0],
        },
    )

    matrix = build_final_candidate_selection_matrix(
        comparison,
        rolling,
        candidates=["ri", "graph_components_equal_weight_score"],
    )

    assert {"ri", "graph_components_equal_weight_score"} <= set(matrix["candidate"])
    assert {"oos_sharpe", "cost_10bps_sharpe", "selection_score"} <= set(
        matrix.columns,
    )


def test_date_alignment_is_consistent_across_overlay_scores() -> None:
    index = pd.date_range("2020-01-01", periods=60, freq="B")
    returns = pd.DataFrame(
        {
            "a": np.sin(np.arange(60)) * 0.01,
            "b": np.cos(np.arange(60)) * 0.01,
        },
        index=index,
    )
    scores = pd.DataFrame(
        {
            "score_a": np.linspace(-1.0, 1.0, 50),
            "score_b": np.linspace(1.0, -1.0, 50),
        },
        index=index[10:],
    )

    comparison, _ = evaluate_score_overlay_set(
        returns,
        scores,
        method="expanding",
        min_history=5,
    )

    assert comparison["n_obs"].nunique() == 1


def test_rolling_origin_oos_runs_on_synthetic_scores() -> None:
    returns, graph, _, pca = _synthetic_inputs()
    scores = build_phase8_candidate_scores(graph, pca_features=pca)
    splits = make_rolling_origin_splits(
        test_years=(2020,),
        selection_start="2019-01-01",
    )

    performance, summary = run_rolling_origin_oos(
        returns,
        scores,
        candidate_specs=[
            {"candidate": "ri", "score": "ri", "variant": "daily"},
            {
                "candidate": "best_low_turnover_score",
                "score": "ri",
                "variant": "daily",
                "select": "low_turnover",
            },
        ],
        splits=splits,
        cost_bps_values=(0.0,),
    )

    assert not performance.empty
    assert not summary.empty
    assert (performance["selection_end"] < performance["test_start"]).all()
    dynamic = performance.loc[performance["candidate"] == "best_low_turnover_score"]
    assert set(dynamic["score"]) == {"selection_period_best_low_turnover"}
    assert dynamic["selected_score"].ne("").all()
