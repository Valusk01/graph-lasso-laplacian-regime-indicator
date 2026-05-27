"""Run the Phase 3 empirical evaluation example with public market data."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from graph_regime.benchmarks import (  # noqa: E402
    compute_correlation_regime_score,
    compute_market_drawdown,
    compute_realized_volatility,
    create_stress_labels,
)
from graph_regime.data import download_vix, download_yfinance_prices, prices_to_returns  # noqa: E402
from graph_regime.evaluation import (  # noqa: E402
    align_indicator_and_benchmarks,
    classify_regimes_by_quantile,
    compute_contemporaneous_diagnostics,
    compute_forward_targets,
    evaluate_predictive_power,
    summarize_regime_classes,
)
from graph_regime.indicator import (  # noqa: E402
    GRAPH_LASSO_DIAGNOSTIC_COLUMNS,
    compute_regime_indicator,
    compute_rolling_graph_features,
    summarize_graph_lasso_convergence,
)


ETF_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "HYG",
    "LQD",
    "GLD",
    "USO",
    "XLK",
    "XLF",
    "XLE",
    "XLU",
]


def main() -> int:
    output_dir = PROJECT_ROOT / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        prices = download_yfinance_prices(ETF_UNIVERSE, start="2015-01-01")
        returns = prices_to_returns(prices, method="log")

        graph_features = compute_rolling_graph_features(
            returns,
            window=126,
            alpha=0.10,
            min_non_missing=0.95,
            partial_corr_threshold=1e-6,
            max_iter=1000,
            tol=1e-4,
            enet_tol=1e-4,
            mode="cd",
            compute_modularity=False,
            on_non_convergence="warn",
        )
        convergence_summary = summarize_graph_lasso_convergence(graph_features)
        print(
            "GraphicalLasso convergence: "
            f"{convergence_summary['n_converged']}/{convergence_summary['n_windows']} "
            f"windows converged "
            f"({convergence_summary['convergence_rate']:.1%}); "
            f"{convergence_summary['n_non_converged']} non-converged."
        )
        graph_indicator = compute_regime_indicator(graph_features)

        vix = download_vix(start=str(returns.index.min().date()))
        spy_drawdown = compute_market_drawdown(prices["SPY"])
        realized_volatility = compute_realized_volatility(returns, window=21)
        average_correlation = compute_correlation_regime_score(returns, window=126)
        average_absolute_correlation = compute_correlation_regime_score(
            returns,
            window=126,
            use_absolute=True,
        )
        benchmark_labels = create_stress_labels(
            vix=vix,
            drawdown=spy_drawdown,
            realized_vol=realized_volatility,
            correlation_score=average_correlation,
        ).join(average_absolute_correlation, how="outer").sort_index()

        aligned = align_indicator_and_benchmarks(graph_indicator, benchmark_labels)
        contemporaneous_diagnostics = compute_contemporaneous_diagnostics(aligned)
        forward_targets = compute_forward_targets(returns, horizons=[5, 21, 63])
        predictive_diagnostics = evaluate_predictive_power(
            graph_indicator["regime_indicator"],
            forward_targets,
        )
        _regime_classes = classify_regimes_by_quantile(graph_indicator["regime_indicator"])
        regime_class_summary = summarize_regime_classes(
            graph_indicator["regime_indicator"],
            returns=returns,
        )

    except (ImportError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Empirical evaluation example could not be completed: {exc}")
        print(
            "Use local CSV files instead, for example:\n"
            "  prices = load_prices_csv('data/raw/prices.csv')\n"
            "  returns = prices_to_returns(prices, method='log')\n"
            "Then compute graph features, load local benchmark series, and pass "
            "them into the evaluation functions."
        )
        return 0

    graph_features.to_csv(output_dir / "graph_regime_features.csv")
    graph_indicator.to_csv(output_dir / "graph_regime_indicator.csv")
    benchmark_labels.to_csv(output_dir / "benchmark_stress_labels.csv")
    graph_features[GRAPH_LASSO_DIAGNOSTIC_COLUMNS].to_csv(
        output_dir / "graph_lasso_convergence_diagnostics.csv",
    )
    contemporaneous_diagnostics.to_csv(
        output_dir / "contemporaneous_diagnostics.csv",
        index=False,
    )
    predictive_diagnostics.to_csv(output_dir / "predictive_diagnostics.csv", index=False)
    regime_class_summary.to_csv(output_dir / "regime_class_summary.csv")

    print(f"Saved graph features to {output_dir / 'graph_regime_features.csv'}")
    print(f"Saved graph indicator to {output_dir / 'graph_regime_indicator.csv'}")
    print(f"Saved benchmark stress labels to {output_dir / 'benchmark_stress_labels.csv'}")
    print(
        "Saved convergence diagnostics to "
        f"{output_dir / 'graph_lasso_convergence_diagnostics.csv'}"
    )
    print(
        "Saved contemporaneous diagnostics to "
        f"{output_dir / 'contemporaneous_diagnostics.csv'}",
    )
    print(f"Saved predictive diagnostics to {output_dir / 'predictive_diagnostics.csv'}")
    print(f"Saved regime class summary to {output_dir / 'regime_class_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
