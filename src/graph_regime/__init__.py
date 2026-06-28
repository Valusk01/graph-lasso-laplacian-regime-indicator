"""Graph-lasso Laplacian regime indicator research engine."""

from graph_regime.benchmarks import (
    compute_correlation_regime_score,
    compute_market_drawdown,
    compute_realized_volatility,
    create_stress_labels,
    load_fred_recession_indicator,
)
from graph_regime.data import (
    download_vix,
    download_yfinance_prices,
    load_prices_csv,
    load_returns_csv,
    prices_to_returns,
)
from graph_regime.evaluation import (
    align_indicator_and_benchmarks,
    classify_regimes_by_quantile,
    compute_contemporaneous_diagnostics,
    compute_event_study,
    compute_forward_targets,
    evaluate_predictive_power,
    summarize_regime_classes,
)
from graph_regime.features import compute_graph_features, compute_laplacian_spectrum
from graph_regime.graph_lasso import (
    GraphicalLassoFit,
    fit_graphical_lasso,
    fit_graphical_lasso_with_diagnostics,
    precision_to_partial_correlation,
)
from graph_regime.indicator import (
    GRAPH_LASSO_DIAGNOSTIC_COLUMNS,
    compute_regime_indicator,
    compute_rolling_graph_features,
    summarize_graph_lasso_convergence,
)
from graph_regime.laplacian import (
    adjacency_to_laplacian,
    partial_correlation_to_adjacency,
)
from graph_regime.preprocessing import clean_returns, standardize_window
from graph_regime.risk_overlay import (
    apply_risk_overlay,
    compare_overlay_to_baseline,
    compute_benchmark_exposures,
    compute_exposure_from_indicator,
    compute_performance_metrics,
    compute_portfolio_returns,
    summarize_exposures,
)
from graph_regime.robustness import (
    compare_converged_vs_all_diagnostics,
    compute_benchmark_changes,
    compute_transition_correlations,
    create_stress_onset_labels,
    run_robustness_grid,
    summarize_robustness_diagnostics,
)

__all__ = [
    "adjacency_to_laplacian",
    "align_indicator_and_benchmarks",
    "apply_risk_overlay",
    "classify_regimes_by_quantile",
    "clean_returns",
    "compare_converged_vs_all_diagnostics",
    "compare_overlay_to_baseline",
    "compute_benchmark_changes",
    "compute_benchmark_exposures",
    "compute_contemporaneous_diagnostics",
    "compute_correlation_regime_score",
    "compute_event_study",
    "compute_exposure_from_indicator",
    "compute_forward_targets",
    "compute_graph_features",
    "compute_laplacian_spectrum",
    "compute_market_drawdown",
    "compute_performance_metrics",
    "compute_portfolio_returns",
    "compute_regime_indicator",
    "compute_realized_volatility",
    "compute_rolling_graph_features",
    "compute_transition_correlations",
    "create_stress_labels",
    "create_stress_onset_labels",
    "download_vix",
    "download_yfinance_prices",
    "evaluate_predictive_power",
    "fit_graphical_lasso",
    "fit_graphical_lasso_with_diagnostics",
    "GRAPH_LASSO_DIAGNOSTIC_COLUMNS",
    "GraphicalLassoFit",
    "load_fred_recession_indicator",
    "load_prices_csv",
    "load_returns_csv",
    "partial_correlation_to_adjacency",
    "precision_to_partial_correlation",
    "prices_to_returns",
    "run_robustness_grid",
    "standardize_window",
    "summarize_exposures",
    "summarize_regime_classes",
    "summarize_graph_lasso_convergence",
    "summarize_robustness_diagnostics",
]
