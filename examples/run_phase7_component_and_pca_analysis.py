"""Run Phase 7 PCA baseline and graph-component research analysis."""

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

from graph_regime.component_overlays import (  # noqa: E402
    build_component_overlay_signals,
    evaluate_component_overlays,
)
from graph_regime.component_scores import add_component_scores  # noqa: E402
from graph_regime.pca_baselines import compute_rolling_pca_features  # noqa: E402
from graph_regime.phase7 import (  # noqa: E402
    run_graph_component_ablation,
    run_model_ladder_incremental_tests,
)

REQUIRED_TABLES = [
    "returns.csv",
    "benchmark_stress_labels.csv",
    "graph_regime_indicator.csv",
]


def main() -> int:
    input_dir = PROJECT_ROOT / "outputs" / "tables"
    table_dir = PROJECT_ROOT / "outputs" / "phase7_tables"
    figure_dir = PROJECT_ROOT / "outputs" / "phase7_figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        filename for filename in REQUIRED_TABLES if not (input_dir / filename).exists()
    ]
    if missing:
        print(
            "Phase 7 analysis could not be completed because required tables are missing:"
        )
        for filename in missing:
            print(f"  - {input_dir / filename}")
        print("Run .venv/bin/python examples/run_visual_diagnostics_workflow.py first.")
        return 0

    returns = _read_table(input_dir / "returns.csv")
    benchmarks = _read_table(input_dir / "benchmark_stress_labels.csv")
    graph_indicator = _read_table(input_dir / "graph_regime_indicator.csv")

    pca_features = compute_rolling_pca_features(returns, window=126)
    component_scores = add_component_scores(
        graph_indicator,
        benchmarks=benchmarks,
        residualization_method="expanding",
        min_history=126,
    )
    model_ladder = run_model_ladder_incremental_tests(
        graph_indicator,
        benchmarks=benchmarks,
        returns=returns,
        pca_features=pca_features,
        component_scores=component_scores,
    )
    overlay_signals = build_component_overlay_signals(component_scores, pca_features)
    overlay_comparison, overlay_costs = evaluate_component_overlays(
        returns,
        overlay_signals,
        method="expanding",
        min_history=20,
    )
    ablation = run_graph_component_ablation(
        component_scores,
        benchmarks=benchmarks,
        returns=returns,
        min_history=20,
    )

    pca_features.to_csv(table_dir / "pca_baseline_features.csv")
    component_scores.to_csv(table_dir / "graph_component_scores.csv")
    model_ladder.to_csv(table_dir / "model_ladder_incremental_tests.csv", index=False)
    overlay_comparison.to_csv(
        table_dir / "component_overlay_comparison.csv", index=False
    )
    overlay_costs.to_csv(
        table_dir / "component_overlay_transaction_costs.csv",
        index=False,
    )
    ablation.to_csv(table_dir / "graph_component_ablation.csv", index=False)

    _plot_overlay_bar(
        overlay_comparison,
        metric="sharpe",
        output_path=figure_dir / "component_overlay_sharpe_bar.png",
    )
    _plot_overlay_bar(
        overlay_comparison,
        metric="calmar",
        output_path=figure_dir / "component_overlay_calmar_bar.png",
    )
    _plot_model_metric_bar(
        model_ladder,
        target="stress_onset_label",
        metric="oos_auc",
        output_path=figure_dir / "model_ladder_oos_auc_bar.png",
    )
    _plot_model_metric_bar(
        model_ladder,
        target="forward_realized_volatility_21d",
        metric="oos_r2",
        output_path=figure_dir / "model_ladder_oos_r2_bar.png",
    )

    _print_summary(model_ladder, overlay_comparison, ablation)
    print(f"Saved Phase 7 tables to {table_dir}")
    print(f"Saved Phase 7 figures to {figure_dir}")
    return 0


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _plot_overlay_bar(
    overlay_comparison: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    if metric not in overlay_comparison.columns:
        return
    rows = overlay_comparison.loc[overlay_comparison["signal"] != "baseline"].copy()
    rows = rows.dropna(subset=[metric]).sort_values(metric, ascending=False).head(12)
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(rows["signal"], rows[metric])
    ax.set_title(f"Component Overlay {metric.replace('_', ' ').title()}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_model_metric_bar(
    model_ladder: pd.DataFrame,
    target: str,
    metric: str,
    output_path: Path,
) -> None:
    if metric not in model_ladder.columns:
        return
    rows = model_ladder.loc[model_ladder["target"] == target]
    rows = rows.groupby("model", as_index=False)[metric].first().dropna()
    if rows.empty:
        return
    rows = rows.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(rows["model"], rows[metric])
    ax.set_title(f"{target} {metric.replace('_', ' ').upper()} by Model")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _print_summary(
    model_ladder: pd.DataFrame,
    overlay_comparison: pd.DataFrame,
    ablation: pd.DataFrame,
) -> None:
    overlay_rows = overlay_comparison.loc[overlay_comparison["signal"] != "baseline"]
    if not overlay_rows.empty:
        best_overlay = overlay_rows.sort_values("sharpe", ascending=False).iloc[0]
        print(
            "Best Phase 7 component/PCA overlay by Sharpe: "
            f"{best_overlay['signal']} ({best_overlay['sharpe']:.4f})."
        )

    stress_rows = model_ladder.loc[model_ladder["target"] == "stress_onset_label"]
    if not stress_rows.empty and "oos_auc" in stress_rows:
        best_model = stress_rows.groupby("model")["oos_auc"].first().dropna()
        if not best_model.empty:
            print(
                "Best stress-onset OOS AUC model: "
                f"{best_model.idxmax()} ({best_model.max():.4f})."
            )

    if not ablation.empty:
        best_ablation = ablation.sort_values("overlay_sharpe", ascending=False).iloc[0]
        print(
            "Best graph-component ablation by overlay Sharpe: "
            f"{best_ablation['ablation']} ({best_ablation['overlay_sharpe']:.4f})."
        )


if __name__ == "__main__":
    raise SystemExit(main())
