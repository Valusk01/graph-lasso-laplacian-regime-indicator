"""Run Phase 6 robustness and out-of-sample research evaluation."""

from __future__ import annotations

import argparse
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

from graph_regime.phase6 import (  # noqa: E402
    Phase6Config,
    build_phase6_tables_from_indicator,
    build_oos_outputs,
    evaluate_transaction_cost_sensitivity,
    make_phase6_parameter_grid,
    make_phase6_quick_grid,
    run_incremental_information_tests,
    run_phase6_robustness_grid,
    select_oos_configuration,
)
from graph_regime.risk_overlay import (  # noqa: E402
    compute_exposure_from_indicator,
    compute_portfolio_returns,
)

REQUIRED_TABLES = [
    "returns.csv",
    "benchmark_stress_labels.csv",
]

CACHED_SMOKE_TABLES = [
    "graph_regime_indicator.csv",
    "graph_lasso_convergence_diagnostics.csv",
]


def main() -> int:
    args = _parse_args()
    input_dir = PROJECT_ROOT / "outputs" / "tables"
    table_dir = PROJECT_ROOT / "outputs" / "phase6_tables"
    figure_dir = PROJECT_ROOT / "outputs" / "phase6_figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    required_tables = list(REQUIRED_TABLES)
    if not args.full_grid and not args.recompute_quick:
        required_tables.extend(CACHED_SMOKE_TABLES)
    missing = [
        filename for filename in required_tables if not (input_dir / filename).exists()
    ]
    if missing:
        print(
            "Phase 6 analysis could not be completed because required tables are missing:"
        )
        for filename in missing:
            print(f"  - {input_dir / filename}")
        print("Run .venv/bin/python examples/run_visual_diagnostics_workflow.py first.")
        return 0

    returns = _read_table(input_dir / "returns.csv")
    benchmarks = _read_table(input_dir / "benchmark_stress_labels.csv")

    if args.full_grid or args.recompute_quick:
        configs = (
            make_phase6_parameter_grid() if args.full_grid else make_phase6_quick_grid()
        )
        if args.full_grid:
            print(
                f"Running full Phase 6 robustness grid with {len(configs)} configurations."
            )
        else:
            print(
                f"Running recomputed quick Phase 6 grid with {len(configs)} configuration."
            )

        tables, indicator_frames = run_phase6_robustness_grid(
            returns=returns,
            benchmarks=benchmarks,
            configs=configs,
            max_iter=args.max_iter,
            min_history=args.min_history,
            return_indicator_frames=True,
        )
    else:
        print(
            "Running cached Phase 6 smoke evaluation from saved Phase 4 tables. "
            "Use --recompute-quick for one rolling recomputation or --full-grid "
            "for the controlled 24-configuration run."
        )
        selected_config = Phase6Config(
            config_id="cached_phase4_alpha0.10_window126_threshold0.000001",
            alpha=0.10,
            window=126,
            partial_corr_threshold=1e-6,
        )
        selected_indicator = _read_table(input_dir / "graph_regime_indicator.csv")
        convergence = _read_table(input_dir / "graph_lasso_convergence_diagnostics.csv")
        tables = build_phase6_tables_from_indicator(
            selected_indicator,
            returns=returns,
            benchmarks=benchmarks,
            convergence=convergence,
            config=selected_config,
            min_history=args.min_history,
        )
        indicator_frames = {selected_config.config_id: selected_indicator}
        configs = [selected_config]

    for name, frame in tables.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)

    oos_selection = select_oos_configuration(tables["robustness_overlay_metrics"])
    oos_outputs = build_oos_outputs(
        tables["robustness_overlay_metrics"],
        tables["robustness_benchmark_comparison"],
        oos_selection,
    )
    oos_selection.to_csv(table_dir / "oos_selection_summary.csv", index=False)
    for name, frame in oos_outputs.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)

    selected_config_id = str(
        oos_selection.loc[oos_selection["selected"], "config_id"].iloc[0]
    )
    selected_config = next(
        config for config in configs if config.config_id == selected_config_id
    )
    selected_indicator = indicator_frames[selected_config.config_id]

    incremental_tests = run_incremental_information_tests(
        selected_indicator,
        benchmarks=benchmarks,
        returns=returns,
    )
    incremental_tests.to_csv(
        table_dir / "incremental_information_tests.csv", index=False
    )

    base_returns = compute_portfolio_returns(returns)
    exposure = compute_exposure_from_indicator(
        selected_indicator["regime_indicator"],
        method="expanding",
        min_history=args.min_history,
    )
    transaction_costs = evaluate_transaction_cost_sensitivity(
        base_returns,
        exposure=exposure,
        cost_bps_values=(0.0, 5.0, 10.0),
    )
    transaction_costs.to_csv(
        table_dir / "transaction_cost_sensitivity.csv", index=False
    )

    _plot_metric_heatmap(
        tables["robustness_overlay_metrics"],
        metric="sharpe",
        output_path=figure_dir / "robustness_sharpe_heatmap.png",
    )
    _plot_metric_heatmap(
        tables["robustness_overlay_metrics"],
        metric="max_drawdown",
        output_path=figure_dir / "robustness_max_drawdown_heatmap.png",
    )
    _plot_metric_heatmap(
        tables["robustness_overlay_metrics"],
        metric="calmar",
        output_path=figure_dir / "robustness_calmar_heatmap.png",
    )

    _print_summary(tables["robustness_overlay_metrics"], oos_selection, oos_outputs)
    print(f"Saved Phase 6 tables to {table_dir}")
    print(f"Saved Phase 6 figures to {figure_dir}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Run the full 24-configuration Phase 6 grid. Default is a quick smoke grid.",
    )
    parser.add_argument(
        "--recompute-quick",
        action="store_true",
        help="Recompute the one-configuration quick grid instead of using saved Phase 4 RI.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum GraphicalLasso iterations per rolling window.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=20,
        help="Minimum expanding-history length before overlay exposure reductions.",
    )
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _plot_metric_heatmap(
    metrics: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    rows = metrics.loc[
        (metrics["sample"] == "all") & (metrics["strategy"] == "ri_overlay")
    ].copy()
    if rows.empty or metric not in rows.columns:
        return

    pivot = rows.pivot_table(
        index="window",
        columns="alpha",
        values=metric,
        aggfunc="mean",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_title(f"RI Overlay {metric.replace('_', ' ').title()} Robustness")
    ax.set_xlabel("Graphical-lasso alpha")
    ax.set_ylabel("Rolling window")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{value:.2f}" for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index])
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _print_summary(
    overlay_metrics: pd.DataFrame,
    selection: pd.DataFrame,
    oos_outputs: dict[str, pd.DataFrame],
) -> None:
    selected = selection.loc[selection["selected"]].iloc[0]
    print(
        "Selected validation configuration: "
        f"{selected['config_id']} (score={selected['selection_score']:.3f})."
    )

    test = oos_outputs["oos_test_performance"]
    baseline = test.loc[test["strategy"] == "baseline"]
    overlay = test.loc[test["strategy"] == "ri_overlay"]
    if not baseline.empty and not overlay.empty:
        baseline_row = baseline.iloc[0]
        overlay_row = overlay.iloc[0]
        print("OOS test RI overlay summary versus baseline:")
        for metric in ["sharpe", "max_drawdown", "annualized_volatility", "calmar"]:
            print(
                f"  {metric}: baseline={baseline_row[metric]:.4f}, "
                f"ri_overlay={overlay_row[metric]:.4f}, "
                f"difference={overlay_row[metric] - baseline_row[metric]:.4f}"
            )

    all_rows = overlay_metrics.loc[
        (overlay_metrics["sample"] == "all")
        & (overlay_metrics["strategy"] == "ri_overlay")
    ]
    if not all_rows.empty:
        best_sharpe = all_rows.sort_values("sharpe", ascending=False).iloc[0]
        print(
            "Best full-sample RI-overlay Sharpe in this grid: "
            f"{best_sharpe['sharpe']:.4f} ({best_sharpe['config_id']})."
        )


if __name__ == "__main__":
    raise SystemExit(main())
