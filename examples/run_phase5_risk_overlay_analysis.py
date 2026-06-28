"""Run Phase 5 robustness and risk-overlay diagnostics from saved outputs."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


_MPL_CACHE = Path(tempfile.gettempdir()) / "graph_regime_matplotlib_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from graph_regime.evaluation import compute_contemporaneous_diagnostics  # noqa: E402
from graph_regime.risk_overlay import (  # noqa: E402
    apply_risk_overlay,
    compute_benchmark_exposures,
    compute_exposure_from_indicator,
    compute_performance_metrics,
    compute_portfolio_returns,
    summarize_exposures,
)
from graph_regime.robustness import (  # noqa: E402
    compare_converged_vs_all_diagnostics,
    compute_benchmark_changes,
    compute_transition_correlations,
    create_stress_onset_labels,
)


REQUIRED_TABLES = [
    "graph_regime_indicator.csv",
    "returns.csv",
    "benchmark_stress_labels.csv",
    "graph_lasso_convergence_diagnostics.csv",
]


def main() -> int:
    input_dir = PROJECT_ROOT / "outputs" / "tables"
    table_dir = PROJECT_ROOT / "outputs" / "phase5_tables"
    figure_dir = PROJECT_ROOT / "outputs" / "phase5_figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    missing = [filename for filename in REQUIRED_TABLES if not (input_dir / filename).exists()]
    if missing:
        print("Phase 5 analysis could not be completed because required tables are missing:")
        for filename in missing:
            print(f"  - {input_dir / filename}")
        print("Run .venv/bin/python examples/run_visual_diagnostics_workflow.py first.")
        return 0

    graph_indicator = _read_table(input_dir / "graph_regime_indicator.csv")
    returns = _read_table(input_dir / "returns.csv")
    benchmarks = _read_table(input_dir / "benchmark_stress_labels.csv")
    convergence = _read_table(input_dir / "graph_lasso_convergence_diagnostics.csv")

    if "regime_indicator" not in graph_indicator.columns:
        print("graph_regime_indicator.csv is missing regime_indicator.")
        return 0
    if "graph_lasso_converged" not in convergence.columns:
        print("graph_lasso_convergence_diagnostics.csv is missing graph_lasso_converged.")
        return 0

    regime_indicator = graph_indicator["regime_indicator"]
    base_returns = compute_portfolio_returns(returns)
    ri_exposure = compute_exposure_from_indicator(regime_indicator, method="expanding").rename(
        "ri_exposure",
    )
    ri_overlay_returns = apply_risk_overlay(base_returns, ri_exposure).rename("ri_overlay")

    benchmark_exposures = compute_benchmark_exposures(benchmarks, method="expanding")
    exposure_frame = pd.concat(
        [ri_exposure, benchmark_exposures],
        axis=1,
        sort=False,
    ).sort_index()

    strategy_returns: dict[str, pd.Series] = {
        "baseline": base_returns,
        "ri_overlay": ri_overlay_returns,
    }
    for column in benchmark_exposures.columns:
        strategy_name = column.replace("_exposure", "_overlay")
        strategy_returns[strategy_name] = apply_risk_overlay(
            base_returns,
            benchmark_exposures[column],
        ).rename(strategy_name)

    metrics = _strategy_metrics(strategy_returns)
    overlay_comparison = metrics.loc[["baseline", "ri_overlay"]].copy()
    benchmark_overlay_comparison = metrics.copy()
    benchmark_overlay_comparison = _add_metric_differences(
        benchmark_overlay_comparison,
        baseline_name="baseline",
    )
    exposure_summary = summarize_exposures(exposure_frame)

    converged_diagnostics = compare_converged_vs_all_diagnostics(
        regime_indicator,
        benchmarks=benchmarks,
        convergence=convergence["graph_lasso_converged"],
        returns=returns,
        forward_horizon=21,
    )

    transition_diagnostics = _transition_diagnostics(
        regime_indicator=regime_indicator,
        benchmarks=benchmarks,
    )

    overlay_comparison.to_csv(table_dir / "overlay_performance_comparison.csv")
    exposure_summary.to_csv(table_dir / "overlay_exposure_summary.csv", index=False)
    benchmark_overlay_comparison.to_csv(table_dir / "benchmark_overlay_comparison.csv")
    transition_diagnostics.to_csv(table_dir / "transition_diagnostics.csv", index=False)
    converged_diagnostics.to_csv(
        table_dir / "non_converged_exclusion_diagnostics.csv",
        index=False,
    )

    _plot_cumulative_returns(
        base_returns,
        ri_overlay_returns,
        output_path=figure_dir / "cumulative_returns_baseline_vs_ri_overlay.png",
    )
    _plot_exposure(
        ri_exposure,
        output_path=figure_dir / "exposure_over_time.png",
    )
    _plot_drawdowns(
        base_returns,
        ri_overlay_returns,
        output_path=figure_dir / "drawdown_baseline_vs_ri_overlay.png",
    )

    _print_key_overlay_summary(metrics)
    print(f"Saved Phase 5 tables to {table_dir}")
    print(f"Saved Phase 5 figures to {figure_dir}")
    return 0


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _strategy_metrics(strategy_returns: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for strategy, returns in strategy_returns.items():
        metrics = compute_performance_metrics(returns)
        metrics.name = strategy
        rows.append(metrics)
    output = pd.DataFrame(rows)
    output.index.name = "strategy"
    return output


def _add_metric_differences(
    metrics: pd.DataFrame,
    baseline_name: str,
) -> pd.DataFrame:
    output = metrics.copy()
    if baseline_name not in output.index:
        return output

    baseline = output.loc[baseline_name]
    for column in ["sharpe", "max_drawdown", "annualized_volatility", "calmar"]:
        if column in output.columns:
            output[f"{column}_minus_baseline"] = output[column] - baseline[column]
    return output


def _transition_diagnostics(
    regime_indicator: pd.Series,
    benchmarks: pd.DataFrame,
) -> pd.DataFrame:
    changes = compute_benchmark_changes(benchmarks)
    change_correlations = compute_transition_correlations(regime_indicator, changes)

    onset = create_stress_onset_labels(benchmarks, min_gap=20)
    aligned = pd.concat(
        [
            regime_indicator.rename("regime_indicator"),
            benchmarks[["systemic_stress_label"]]
            if "systemic_stress_label" in benchmarks.columns
            else pd.DataFrame(index=benchmarks.index),
            onset,
        ],
        axis=1,
        join="inner",
    )
    onset_diagnostics = compute_contemporaneous_diagnostics(
        aligned,
        stress_label_columns=["stress_onset_label", "systemic_stress_label"],
        continuous_benchmark_columns=[],
    )
    onset_diagnostics = onset_diagnostics.rename(columns={"metric": "statistic"})
    onset_diagnostics["diagnostic_type"] = "stress_label_comparison"

    change_correlations = change_correlations.rename(
        columns={
            "n_obs": "value_n_obs",
            "pearson_correlation": "value_pearson_correlation",
            "spearman_correlation": "value_spearman_correlation",
        },
    )
    change_rows = change_correlations.melt(
        id_vars=["diagnostic_type", "benchmark"],
        var_name="statistic",
        value_name="value",
    )
    onset_rows = onset_diagnostics[
        ["diagnostic_type", "benchmark", "statistic", "value"]
    ]
    return pd.concat([change_rows, onset_rows], axis=0, ignore_index=True)


def _wealth(returns: pd.Series) -> pd.Series:
    numeric_returns = pd.to_numeric(returns, errors="coerce").dropna().clip(lower=-0.999999)
    return (1.0 + numeric_returns).cumprod()


def _drawdown(returns: pd.Series) -> pd.Series:
    wealth = _wealth(returns)
    return wealth / wealth.cummax() - 1.0


def _plot_cumulative_returns(
    base_returns: pd.Series,
    ri_overlay_returns: pd.Series,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    _wealth(base_returns).plot(ax=ax, label="Baseline")
    _wealth(ri_overlay_returns).plot(ax=ax, label="RI overlay")
    ax.set_title("Baseline vs RI Overlay Cumulative Returns")
    ax.set_ylabel("Wealth index")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_exposure(exposure: pd.Series, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    exposure.plot(ax=ax, linewidth=1.2)
    ax.set_title("RI Overlay Exposure")
    ax.set_ylabel("Exposure")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_drawdowns(
    base_returns: pd.Series,
    ri_overlay_returns: pd.Series,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    _drawdown(base_returns).plot(ax=ax, label="Baseline")
    _drawdown(ri_overlay_returns).plot(ax=ax, label="RI overlay")
    ax.set_title("Baseline vs RI Overlay Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _print_key_overlay_summary(metrics: pd.DataFrame) -> None:
    if "baseline" not in metrics.index or "ri_overlay" not in metrics.index:
        return

    baseline = metrics.loc["baseline"]
    overlay = metrics.loc["ri_overlay"]
    print("Phase 5 RI overlay summary versus baseline:")
    for metric in ["sharpe", "max_drawdown", "annualized_volatility", "calmar"]:
        print(
            f"  {metric}: baseline={baseline[metric]:.4f}, "
            f"ri_overlay={overlay[metric]:.4f}, "
            f"difference={overlay[metric] - baseline[metric]:.4f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
