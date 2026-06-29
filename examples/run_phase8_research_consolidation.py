"""Run Phase 8 research consolidation diagnostics.

This script reuses saved Phase 4/7 outputs where possible. It does not download
data and does not modify earlier generated outputs.
"""

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

from graph_regime.component_scores import add_component_scores  # noqa: E402
from graph_regime.pca_baselines import compute_rolling_pca_features  # noqa: E402
from graph_regime.phase8 import (  # noqa: E402
    build_final_candidate_selection_matrix,
    build_phase8_candidate_scores,
    compare_information_sets,
    compute_algebraic_connectivity_diagnostics,
    evaluate_algebraic_connectivity_overlays,
    evaluate_turnover_reduction_overlays,
    make_rolling_origin_splits,
    rank_information_sets,
    run_rolling_origin_oos,
    select_low_turnover_candidate,
)

REQUIRED_TABLES = [
    "returns.csv",
    "benchmark_stress_labels.csv",
    "graph_regime_indicator.csv",
]


def main() -> int:
    input_dir = PROJECT_ROOT / "outputs" / "tables"
    phase7_dir = PROJECT_ROOT / "outputs" / "phase7_tables"
    table_dir = PROJECT_ROOT / "outputs" / "phase8_tables"
    figure_dir = PROJECT_ROOT / "outputs" / "phase8_figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        filename for filename in REQUIRED_TABLES if not (input_dir / filename).exists()
    ]
    if missing:
        print("Phase 8 analysis could not be completed; required tables are missing:")
        for filename in missing:
            print(f"  - {input_dir / filename}")
        print("Run .venv/bin/python examples/run_visual_diagnostics_workflow.py first.")
        return 0

    returns = _read_table(input_dir / "returns.csv")
    benchmarks = _read_table(input_dir / "benchmark_stress_labels.csv")
    graph_features = _read_table(input_dir / "graph_regime_indicator.csv")

    pca_path = phase7_dir / "pca_baseline_features.csv"
    component_path = phase7_dir / "graph_component_scores.csv"
    pca_features = (
        _read_table(pca_path)
        if pca_path.exists()
        else compute_rolling_pca_features(returns, window=126)
    )
    component_scores = (
        _read_table(component_path)
        if component_path.exists()
        else add_component_scores(
            graph_features,
            benchmarks=benchmarks,
            residualization_method="expanding",
            min_history=126,
        )
    )

    information_set_comparison = compare_information_sets(
        graph_features=graph_features,
        benchmarks=benchmarks,
        returns=returns,
        pca_features=pca_features,
        component_scores=component_scores,
    )
    information_set_ranking = rank_information_sets(information_set_comparison)
    candidate_scores = build_phase8_candidate_scores(
        graph_features,
        pca_features=pca_features,
        component_scores=component_scores,
    )
    algebraic_diagnostics = compute_algebraic_connectivity_diagnostics(
        graph_features,
        benchmarks=benchmarks,
        returns=returns,
        pca_features=pca_features,
    )
    algebraic_overlay, algebraic_costs = evaluate_algebraic_connectivity_overlays(
        graph_features,
        returns=returns,
    )
    turnover_comparison, turnover_costs = evaluate_turnover_reduction_overlays(
        returns=returns,
        scores=candidate_scores,
    )
    best_low_turnover = select_low_turnover_candidate(turnover_comparison)
    rolling_specs = _rolling_candidate_specs(best_low_turnover)
    rolling_performance, rolling_summary = run_rolling_origin_oos(
        returns=returns,
        candidate_scores=candidate_scores,
        candidate_specs=rolling_specs,
        splits=make_rolling_origin_splits(),
    )
    selection_matrix = build_final_candidate_selection_matrix(
        information_set_comparison=information_set_comparison,
        rolling_origin_summary=rolling_summary,
        turnover_comparison=turnover_comparison,
    )

    information_set_comparison.to_csv(
        table_dir / "information_set_model_comparison.csv",
        index=False,
    )
    information_set_ranking.to_csv(
        table_dir / "information_set_ranking.csv",
        index=False,
    )
    algebraic_diagnostics.to_csv(
        table_dir / "algebraic_connectivity_diagnostics.csv",
        index=False,
    )
    algebraic_overlay.to_csv(
        table_dir / "algebraic_connectivity_overlay_comparison.csv",
        index=False,
    )
    algebraic_costs.to_csv(
        table_dir / "algebraic_connectivity_cost_sensitivity.csv",
        index=False,
    )
    turnover_comparison.to_csv(
        table_dir / "turnover_reduction_overlay_comparison.csv",
        index=False,
    )
    turnover_costs.to_csv(
        table_dir / "turnover_reduction_cost_sensitivity.csv",
        index=False,
    )
    rolling_performance.to_csv(
        table_dir / "rolling_origin_oos_performance.csv",
        index=False,
    )
    rolling_summary.to_csv(
        table_dir / "rolling_origin_oos_summary.csv",
        index=False,
    )
    selection_matrix.to_csv(
        table_dir / "final_candidate_selection_matrix.csv",
        index=False,
    )

    _plot_algebraic_correlations(
        algebraic_diagnostics,
        figure_dir / "algebraic_connectivity_vs_other_components.png",
    )
    _plot_scatter(
        turnover_comparison,
        x="annualized_turnover_proxy",
        y="sharpe",
        output_path=figure_dir / "turnover_vs_sharpe.png",
    )
    _plot_scatter(
        turnover_comparison,
        x="annualized_turnover_proxy",
        y="calmar",
        output_path=figure_dir / "turnover_vs_calmar.png",
    )
    _plot_cost_adjusted(
        turnover_costs, figure_dir / "cost_adjusted_overlay_comparison.png"
    )
    _plot_rolling_metric(
        rolling_performance,
        metric="sharpe",
        output_path=figure_dir / "rolling_origin_sharpe_by_year.png",
    )
    _plot_rolling_metric(
        rolling_performance,
        metric="calmar",
        output_path=figure_dir / "rolling_origin_calmar_by_year.png",
    )
    _plot_rolling_metric(
        rolling_performance,
        metric="max_drawdown",
        output_path=figure_dir / "rolling_origin_max_drawdown_by_year.png",
    )

    _print_summary(
        information_set_ranking,
        algebraic_overlay,
        turnover_comparison,
        rolling_summary,
        selection_matrix,
    )
    print(f"Saved Phase 8 tables to {table_dir}")
    print(f"Saved Phase 8 figures to {figure_dir}")
    return 0


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _rolling_candidate_specs(best_low_turnover: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"candidate": "ri", "score": "ri", "variant": "daily"},
        {
            "candidate": "graph_components_equal_weight_score",
            "score": "graph_components_equal_weight_score",
            "variant": "daily",
        },
        {
            "candidate": "transition_score",
            "score": "transition_score",
            "variant": "daily",
        },
        {"candidate": "pca_score", "score": "pca_score", "variant": "daily"},
        {
            "candidate": "pca_plus_graph_score",
            "score": "pca_plus_graph_score",
            "variant": "daily",
        },
        {
            "candidate": "excluding_algebraic_connectivity_score",
            "score": "excluding_algebraic_connectivity_score",
            "variant": "daily",
        },
        {
            "candidate": "best_low_turnover_score",
            "score": best_low_turnover.get(
                "score", "graph_components_equal_weight_score"
            ),
            "variant": best_low_turnover.get("variant", "hysteresis"),
            "select": "low_turnover",
        },
    ]


def _plot_algebraic_correlations(diagnostics: pd.DataFrame, output_path: Path) -> None:
    rows = diagnostics.loc[
        (diagnostics["diagnostic_type"] == "correlation")
        & (diagnostics["metric"] == "pearson")
    ].copy()
    rows = rows.dropna(subset=["value"]).sort_values("value")
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(rows["variable"], rows["value"])
    ax.set_title("Algebraic Connectivity Correlations")
    ax.set_xlabel("Pearson correlation")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    output_path: Path,
) -> None:
    if frame.empty or x not in frame or y not in frame:
        return
    rows = frame.dropna(subset=[x, y])
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rows[x], rows[y], alpha=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y.replace('_', ' ').title()} vs {x.replace('_', ' ').title()}")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_cost_adjusted(costs: pd.DataFrame, output_path: Path) -> None:
    if costs.empty or "sharpe" not in costs:
        return
    rows = costs.dropna(subset=["sharpe"]).copy()
    if rows.empty:
        return
    rows["name"] = rows["score"] + " / " + rows["variant"]
    best = (
        rows.groupby("name")["sharpe"].max().sort_values(ascending=False).head(8).index
    )
    rows = rows.loc[rows["name"].isin(best)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, group in rows.groupby("name"):
        group = group.sort_values("cost_bps")
        ax.plot(group["cost_bps"], group["sharpe"], marker="o", label=name)
    ax.set_xlabel("Cost bps")
    ax.set_ylabel("Sharpe")
    ax.set_title("Cost-Adjusted Overlay Sharpe")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_rolling_metric(
    performance: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    if performance.empty or metric not in performance:
        return
    rows = performance.loc[performance["cost_bps"] == 0.0].dropna(subset=[metric])
    if rows.empty:
        return
    pivot = rows.pivot_table(index="test_year", columns="candidate", values=metric)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(ax=ax)
    ax.set_title(f"Rolling-Origin {metric.replace('_', ' ').title()} by Year")
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _print_summary(
    ranking: pd.DataFrame,
    algebraic_overlay: pd.DataFrame,
    turnover: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    selection_matrix: pd.DataFrame,
) -> None:
    stress = ranking.loc[
        (ranking["target"] == "stress_onset_label") & (ranking["metric"] == "oos_auc")
    ]
    if not stress.empty:
        row = stress.iloc[0]
        print(
            "Best Phase 8 stress-onset information set by OOS AUC: "
            f"{row['best_information_set']} ({row['best_value']:.4f})."
        )
    if not algebraic_overlay.empty:
        best = algebraic_overlay.sort_values("sharpe", ascending=False).iloc[0]
        print(
            "Best algebraic-connectivity diagnostic overlay by Sharpe: "
            f"{best['score']} ({best['sharpe']:.4f})."
        )
    if not turnover.empty:
        best_turnover = turnover.sort_values("calmar", ascending=False).iloc[0]
        print(
            "Best low-turnover candidate by Calmar screen: "
            f"{best_turnover['score']} / {best_turnover['variant']} "
            f"({best_turnover['calmar']:.4f})."
        )
    if not rolling_summary.empty:
        best_oos = rolling_summary.loc[rolling_summary["cost_bps"] == 0.0]
        if not best_oos.empty:
            best_oos = best_oos.sort_values("mean_sharpe", ascending=False).iloc[0]
            print(
                "Best rolling-origin candidate by mean Sharpe: "
                f"{best_oos['candidate']} / {best_oos['variant']} "
                f"({best_oos['mean_sharpe']:.4f})."
            )
    if not selection_matrix.empty:
        best = selection_matrix.iloc[0]
        print(
            "Phase 8 selection-matrix leader: "
            f"{best['candidate']} (score {best['selection_score']:.4f})."
        )


if __name__ == "__main__":
    raise SystemExit(main())
