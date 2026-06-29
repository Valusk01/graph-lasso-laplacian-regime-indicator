import numpy as np
import pandas as pd

import graph_regime.phase6 as phase6_module
from graph_regime.phase6 import (
    SampleSplits,
    apply_transaction_costs,
    assign_sample_period,
    build_phase6_tables_from_indicator,
    build_oos_outputs,
    evaluate_transaction_cost_sensitivity,
    make_phase6_parameter_grid,
    run_incremental_information_tests,
    run_phase6_robustness_grid,
    select_oos_configuration,
)


def test_phase6_parameter_grid_has_expected_full_size() -> None:
    grid = make_phase6_parameter_grid()

    assert len(grid) == 24
    assert {config.alpha for config in grid} == {0.05, 0.10, 0.15, 0.20}
    assert {config.window for config in grid} == {63, 126, 252}
    assert {config.partial_corr_threshold for config in grid} == {0.0, 0.03}


def test_assign_sample_period_uses_split_dates() -> None:
    index = pd.to_datetime(["2019-12-31", "2020-01-02", "2023-01-03"])
    labels = assign_sample_period(index)

    assert labels.iloc[0] == "development"
    assert labels.iloc[1] == "validation"
    assert labels.iloc[2] == "test"


def test_transaction_cost_adjustment_charges_exposure_changes() -> None:
    index = pd.date_range("2020-01-01", periods=4)
    returns = pd.Series([0.01, 0.01, 0.01, 0.01], index=index)
    exposure = pd.Series([1.0, 0.5, 0.5, 1.0], index=index)

    adjusted = apply_transaction_costs(returns, exposure, cost_bps=10.0)

    assert np.isclose(adjusted.iloc[0], 0.01)
    assert np.isclose(adjusted.iloc[1], 0.01 - 0.5 * 0.001)
    assert np.isclose(adjusted.iloc[2], 0.01)
    assert np.isclose(adjusted.iloc[3], 0.01 - 0.5 * 0.001)


def test_transaction_cost_sensitivity_outputs_metrics() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    portfolio_returns = pd.Series([0.01, -0.01, 0.02, -0.01] * 2, index=index)
    exposure = pd.Series([1.0, 1.0, 0.7, 0.7, 0.5, 0.5, 1.0, 1.0], index=index)

    sensitivity = evaluate_transaction_cost_sensitivity(
        portfolio_returns,
        exposure,
        cost_bps_values=(0.0, 5.0),
        trading_days=4,
    )

    assert list(sensitivity["cost_bps"]) == [0.0, 5.0]
    assert "annualized_turnover_proxy" in sensitivity.columns
    assert np.isfinite(sensitivity.loc[0, "sharpe"])


def test_oos_selection_uses_validation_not_test() -> None:
    rows = []
    for config_id, validation_sharpe, test_sharpe in [
        ("config_a", 0.1, 2.0),
        ("config_b", 1.0, 0.1),
    ]:
        for sample, overlay_sharpe in [
            ("validation", validation_sharpe),
            ("test", test_sharpe),
        ]:
            rows.append(_metric_row(config_id, sample, "baseline", 0.0))
            rows.append(_metric_row(config_id, sample, "ri_overlay", overlay_sharpe))
    metrics = pd.DataFrame(rows)

    selection = select_oos_configuration(metrics)
    selected = selection.loc[selection["selected"], "config_id"].iloc[0]
    outputs = build_oos_outputs(metrics, metrics, selection)

    assert selected == "config_b"
    assert set(outputs) == {"oos_test_performance", "oos_overlay_comparison"}


def test_incremental_information_tests_return_finite_outputs() -> None:
    rng = np.random.default_rng(7)
    index = pd.date_range("2020-01-01", periods=260, freq="B")
    signal = np.linspace(-1.0, 1.0, len(index))
    returns = pd.DataFrame(
        {
            "a": 0.001 * signal + rng.normal(0.0, 0.01, len(index)),
            "b": 0.001 * signal + rng.normal(0.0, 0.012, len(index)),
        },
        index=index,
    )
    indicator = pd.DataFrame(
        {
            "regime_indicator": signal + rng.normal(0.0, 0.1, len(index)),
            "average_graph_strength": signal,
            "algebraic_connectivity": signal * 0.5,
            "laplacian_frobenius_change": np.abs(np.gradient(signal)),
            "largest_laplacian_eigenvalue_share": signal * 0.2,
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
    splits = SampleSplits(
        development_start="2020-01-01",
        development_end="2020-05-31",
        validation_start="2020-06-01",
        validation_end="2020-09-30",
        test_start="2020-10-01",
    )

    tests = run_incremental_information_tests(
        indicator,
        benchmarks=benchmarks,
        returns=returns,
        splits=splits,
        horizons=(5,),
    )

    assert not tests.empty
    assert {"target", "model", "regressor", "adjusted_r2", "oos_r2"} <= set(
        tests.columns
    )
    assert np.isfinite(tests["adjusted_r2"].dropna()).any()


def test_robustness_grid_summary_creation_with_monkeypatch(monkeypatch) -> None:
    index = pd.date_range("2020-01-01", periods=12)
    returns = pd.DataFrame(
        {
            "a": np.linspace(-0.02, 0.03, 12),
            "b": np.linspace(0.01, -0.01, 12),
        },
        index=index,
    )
    benchmarks = pd.DataFrame(
        {
            "vix": np.linspace(10.0, 20.0, 12),
            "realized_volatility": np.linspace(0.1, 0.3, 12),
            "drawdown": np.linspace(0.0, -0.1, 12),
            "average_correlation": np.linspace(0.2, 0.5, 12),
            "average_absolute_correlation": np.linspace(0.25, 0.55, 12),
            "systemic_stress_label": [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
        },
        index=index,
    )

    def fake_compute_rolling_graph_features(
        returns,
        window,
        alpha,
        min_non_missing=0.95,
        partial_corr_threshold=1e-8,
        max_iter=200,
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
                "laplacian_frobenius_change": [np.nan]
                + list(np.linspace(0.1, 0.2, n - 1)),
                "number_of_edges": np.arange(n) + 3,
                "average_node_strength": np.linspace(1.0, 1.3, n),
                "graph_lasso_converged": [True] * n,
                "graph_lasso_n_iter": [5] * n,
                "graph_lasso_warning_count": [0] * n,
            },
            index=feature_index,
        )

    monkeypatch.setattr(
        phase6_module,
        "compute_rolling_graph_features",
        fake_compute_rolling_graph_features,
    )

    tables = run_phase6_robustness_grid(
        returns,
        benchmarks,
        configs=make_phase6_parameter_grid(
            alpha_values=(0.1,),
            windows=(3,),
            partial_corr_thresholds=(0.03,),
        ),
        splits=SampleSplits(
            development_start="2020-01-01",
            development_end="2020-01-04",
            validation_start="2020-01-05",
            validation_end="2020-01-08",
            test_start="2020-01-09",
        ),
        forward_horizon=2,
    )

    assert set(tables) == {
        "robustness_grid_summary",
        "robustness_overlay_metrics",
        "robustness_convergence_summary",
        "robustness_benchmark_comparison",
    }
    assert tables["robustness_grid_summary"].shape[0] == 1
    assert "convergence_rate" in tables["robustness_convergence_summary"].columns


def test_cached_phase6_tables_from_indicator() -> None:
    index = pd.date_range("2020-01-01", periods=20)
    returns = pd.DataFrame(
        {
            "a": np.sin(np.arange(20)) * 0.01,
            "b": np.cos(np.arange(20)) * 0.01,
        },
        index=index,
    )
    indicator = pd.DataFrame(
        {
            "regime_indicator": np.linspace(-1.0, 1.0, 20),
            "graph_lasso_converged": [True] * 19 + [False],
            "graph_lasso_n_iter": [10] * 20,
            "graph_lasso_warning_count": [0] * 19 + [1],
        },
        index=index,
    )
    benchmarks = pd.DataFrame(
        {
            "vix": np.linspace(10.0, 25.0, 20),
            "realized_volatility": np.linspace(0.1, 0.2, 20),
            "drawdown": np.linspace(0.0, -0.1, 20),
            "average_correlation": np.linspace(0.2, 0.5, 20),
            "average_absolute_correlation": np.linspace(0.25, 0.55, 20),
            "systemic_stress_label": [0] * 10 + [1] * 10,
        },
        index=index,
    )

    tables = build_phase6_tables_from_indicator(
        indicator,
        returns=returns,
        benchmarks=benchmarks,
        splits=SampleSplits(
            development_start="2020-01-01",
            development_end="2020-01-07",
            validation_start="2020-01-08",
            validation_end="2020-01-14",
            test_start="2020-01-15",
        ),
        forward_horizon=2,
    )

    assert tables["robustness_grid_summary"].shape[0] == 1
    assert tables["robustness_convergence_summary"].loc[0, "n_non_converged"] == 1
    assert {"baseline", "ri_overlay"} <= set(
        tables["robustness_overlay_metrics"]["strategy"]
    )


def _metric_row(
    config_id: str,
    sample: str,
    strategy: str,
    sharpe: float,
) -> dict[str, float | str]:
    return {
        "config_id": config_id,
        "alpha": 0.1,
        "window": 126,
        "partial_corr_threshold": 0.03,
        "sample": sample,
        "strategy": strategy,
        "annualized_return": sharpe * 0.1,
        "annualized_volatility": 0.1 if strategy == "ri_overlay" else 0.2,
        "sharpe": sharpe,
        "sortino": sharpe,
        "max_drawdown": -0.1 if strategy == "ri_overlay" else -0.2,
        "calmar": sharpe,
        "mean_return": 0.001,
        "mean_absolute_return": 0.01,
        "worst_1pct_return": -0.03,
        "worst_5pct_return": -0.02 if strategy == "ri_overlay" else -0.04,
        "hit_rate": 0.5,
        "n_obs": 50.0,
        "average_exposure": 0.9,
        "time_in_reduced_exposure": 0.2,
        "number_of_exposure_changes": 3.0,
        "annualized_turnover_proxy": 2.0,
        "mean_absolute_exposure_change": 0.01,
    }
