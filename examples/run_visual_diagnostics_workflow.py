"""Run the Phase 4 visual diagnostics workflow with public market data."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


_MPL_CACHE = Path(tempfile.gettempdir()) / "graph_regime_matplotlib_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

from matplotlib import pyplot as plt


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
from graph_regime.data import (  # noqa: E402
    download_vix,
    download_yfinance_prices,
    load_prices_csv,
    load_returns_csv,
    prices_to_returns,
)
from graph_regime.evaluation import (  # noqa: E402
    align_indicator_and_benchmarks,
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
from graph_regime.plotting import (  # noqa: E402
    plot_feature_panel,
    plot_indicator_vs_benchmark,
    plot_regime_class_returns,
    plot_regime_indicator,
    plot_scatter_indicator_target,
    plot_stress_boxplot,
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
    table_dir = PROJECT_ROOT / "outputs" / "tables"
    figure_dir = PROJECT_ROOT / "outputs" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # To use local data instead of online downloads, replace the download block
    # below with:
    #   prices = load_prices_csv("data/raw/prices.csv")
    #   returns = load_returns_csv("data/raw/returns.csv")
    # or:
    #   prices = load_prices_csv("data/raw/prices.csv")
    #   returns = prices_to_returns(prices, method="log")
    # Local VIX or benchmark CSV files can be loaded with pandas/read helpers
    # and then passed into create_stress_labels.
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
        regime_class_summary = summarize_regime_classes(
            graph_indicator["regime_indicator"],
            returns=returns,
        )

    except (ImportError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Visual diagnostics workflow could not be completed: {exc}")
        print(
            "Use local CSV files instead, for example:\n"
            "  prices = load_prices_csv('data/raw/prices.csv')\n"
            "  returns = load_returns_csv('data/raw/returns.csv')\n"
            "or:\n"
            "  prices = load_prices_csv('data/raw/prices.csv')\n"
            "  returns = prices_to_returns(prices, method='log')\n"
            "Then load local benchmark series and pass them into the workflow."
        )
        return 0

    graph_features.to_csv(table_dir / "graph_regime_features.csv")
    graph_indicator.to_csv(table_dir / "graph_regime_indicator.csv")
    benchmark_labels.to_csv(table_dir / "benchmark_stress_labels.csv")
    graph_features[GRAPH_LASSO_DIAGNOSTIC_COLUMNS].to_csv(
        table_dir / "graph_lasso_convergence_diagnostics.csv",
    )
    contemporaneous_diagnostics.to_csv(
        table_dir / "contemporaneous_diagnostics.csv",
        index=False,
    )
    predictive_diagnostics.to_csv(table_dir / "predictive_diagnostics.csv", index=False)
    regime_class_summary.to_csv(table_dir / "regime_class_summary.csv")

    _try_save_and_close(
        "regime_indicator",
        lambda: plot_regime_indicator(
            graph_indicator,
            stress_labels=benchmark_labels,
            output_path=figure_dir / "regime_indicator.png",
        ),
    )
    _try_save_and_close(
        "regime_indicator_vs_vix",
        lambda: plot_indicator_vs_benchmark(
            graph_indicator,
            benchmark_labels["vix"],
            benchmark_name="vix",
            output_path=figure_dir / "regime_indicator_vs_vix.png",
        ),
    )
    _try_save_and_close(
        "regime_indicator_vs_drawdown",
        lambda: plot_indicator_vs_benchmark(
            graph_indicator,
            benchmark_labels["drawdown"],
            benchmark_name="drawdown",
            output_path=figure_dir / "regime_indicator_vs_drawdown.png",
        ),
    )
    _try_save_and_close(
        "regime_indicator_vs_realized_volatility",
        lambda: plot_indicator_vs_benchmark(
            graph_indicator,
            benchmark_labels["realized_volatility"],
            benchmark_name="realized_volatility",
            output_path=figure_dir / "regime_indicator_vs_realized_volatility.png",
        ),
    )
    _try_save_and_close(
        "regime_indicator_vs_average_correlation",
        lambda: plot_indicator_vs_benchmark(
            graph_indicator,
            benchmark_labels["average_correlation"],
            benchmark_name="average_correlation",
            output_path=figure_dir / "regime_indicator_vs_average_correlation.png",
        ),
    )
    _try_save_and_close(
        "regime_indicator_vs_average_absolute_correlation",
        lambda: plot_indicator_vs_benchmark(
            graph_indicator,
            benchmark_labels["average_absolute_correlation"],
            benchmark_name="average_absolute_correlation",
            output_path=figure_dir / "regime_indicator_vs_average_absolute_correlation.png",
        ),
    )
    _try_save_and_close(
        "graph_feature_panel",
        lambda: plot_feature_panel(
            graph_features,
            output_path=figure_dir / "graph_feature_panel.png",
        ),
    )
    _try_save_and_close(
        "stress_boxplot",
        lambda: plot_stress_boxplot(
            aligned,
            output_path=figure_dir / "stress_boxplot.png",
        ),
    )
    _try_save_and_close(
        "ri_vs_forward_realized_volatility_21d",
        lambda: plot_scatter_indicator_target(
            graph_indicator["regime_indicator"],
            forward_targets["forward_realized_volatility_21d"],
            target_name="forward_realized_volatility_21d",
            output_path=figure_dir / "ri_vs_forward_realized_volatility_21d.png",
        ),
    )
    _try_save_and_close(
        "regime_class_volatility",
        lambda: plot_regime_class_returns(
            regime_class_summary,
            metric="volatility",
            output_path=figure_dir / "regime_class_volatility.png",
        ),
    )

    print(f"Saved tables to {table_dir}")
    print(f"Saved figures to {figure_dir}")
    print(
        "Saved convergence diagnostics to "
        f"{table_dir / 'graph_lasso_convergence_diagnostics.csv'}"
    )
    return 0


def _try_save_and_close(plot_name: str, figure_factory) -> None:
    try:
        fig = figure_factory()
    except (ValueError, KeyError) as exc:
        print(f"Skipped {plot_name}: {exc}")
        return

    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
