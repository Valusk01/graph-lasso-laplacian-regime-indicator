# Graph-Lasso Laplacian Regime Indicator

Research-grade Python tools for testing whether rolling graphical-lasso
Laplacian features can behave like a systemic stress or recession regime
indicator. The project currently includes the Phase 1 through Phase 6 research
workflow: core graph features, benchmark data, empirical diagnostics, visual
diagnostics, research-only risk-overlay analysis, and robustness/out-of-sample
evaluation.

## Research Hypothesis

During crisis, recessionary, or systemic stress regimes, the conditional
dependence topology of asset returns changes. A sparse inverse-covariance
network estimated through graphical lasso may become more globally connected,
less modular, and more dominated by systemic market structure. Rolling graph
Laplacian features can summarize those changes into a systemic connectedness
regime indicator.

## Implemented Phases

- Phase 1: core graph-regime engine for rolling graphical-lasso precision
  estimation, partial-correlation adjacency construction, Laplacian spectral
  features, convergence diagnostics, and a composite regime indicator.
- Phase 2: data loading helpers, optional public-data download helpers, and
  benchmark stress variables for later empirical comparison.
- Phase 3: empirical evaluation helpers for contemporaneous benchmark
  diagnostics, forward-target predictive diagnostics, event-study extraction,
  and regime-class summaries.
- Phase 4: reusable matplotlib visual diagnostics and a reproducible workflow
  that writes diagnostic tables and figures.
- Phase 5: robustness, topology-transition diagnostics, and research-only
  risk-overlay evaluation against simple benchmark overlays.
- Phase 6: controlled robustness grid, walk-forward-style out-of-sample
  selection, incremental-information tests, and transaction-cost/turnover
  sensitivity for the risk-overlay hypothesis.

Not implemented yet: dashboards, notebooks, live trading, broker integration,
or final empirical conclusions.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
```

The base package includes `matplotlib` for visual diagnostics.

For optional online data examples:

```bash
.venv/bin/python -m pip install -e ".[test,data]"
```

## Run Tests

```bash
.venv/bin/python -m pytest -q
```

Tests use synthetic data and temporary CSV files only. They do not require
internet access.

## Run The Data Layer Example

```bash
.venv/bin/python examples/run_data_layer_example.py
```

The example tries to download public Yahoo Finance data with `yfinance`, creates
returns and benchmark stress labels, and writes:

- `outputs/tables/returns.csv`
- `outputs/tables/benchmark_stress_labels.csv`

Downloaded market data and generated outputs are intentionally ignored by git
and should not be committed.

## Run The Empirical Evaluation Example

```bash
.venv/bin/python examples/run_empirical_evaluation_example.py
```

The Phase 3 example tries to download public Yahoo Finance data, compute the
graph regime indicator, build benchmark stress labels, run contemporaneous and
predictive diagnostics, and write:

- `outputs/tables/graph_regime_features.csv`
- `outputs/tables/graph_regime_indicator.csv`
- `outputs/tables/benchmark_stress_labels.csv`
- `outputs/tables/graph_lasso_convergence_diagnostics.csv`
- `outputs/tables/contemporaneous_diagnostics.csv`
- `outputs/tables/predictive_diagnostics.csv`
- `outputs/tables/regime_class_summary.csv`

Online market data and generated output tables are ignored by git and should not
be committed.

## Run The Visual Diagnostics Workflow

```bash
.venv/bin/python examples/run_visual_diagnostics_workflow.py
```

The Phase 4 workflow tries to download public Yahoo Finance data, compute the
graph regime indicator, run empirical diagnostics, and write the same Phase 3
tables plus visual diagnostics.

Output tables:

- `outputs/tables/graph_regime_features.csv`
- `outputs/tables/graph_regime_indicator.csv`
- `outputs/tables/benchmark_stress_labels.csv`
- `outputs/tables/graph_lasso_convergence_diagnostics.csv`
- `outputs/tables/contemporaneous_diagnostics.csv`
- `outputs/tables/predictive_diagnostics.csv`
- `outputs/tables/regime_class_summary.csv`

Output figures:

- `outputs/figures/regime_indicator.png`
- `outputs/figures/regime_indicator_vs_vix.png`
- `outputs/figures/regime_indicator_vs_drawdown.png`
- `outputs/figures/regime_indicator_vs_realized_volatility.png`
- `outputs/figures/regime_indicator_vs_average_correlation.png`
- `outputs/figures/regime_indicator_vs_average_absolute_correlation.png`
- `outputs/figures/graph_feature_panel.png`
- `outputs/figures/stress_boxplot.png`
- `outputs/figures/ri_vs_forward_realized_volatility_21d.png`
- `outputs/figures/regime_class_volatility.png`

Generated figures and tables under `outputs/` are ignored by git and should not
be committed.

## Run The Phase 5 Risk Overlay Analysis

```bash
.venv/bin/python examples/run_phase5_risk_overlay_analysis.py
```

This workflow reads the saved Phase 4 tables from `outputs/tables/`. If they
are missing, run the visual diagnostics workflow first. Phase 5 is research
only: it tests whether the regime indicator can act as a risk-overlay feature,
not whether it is a live trading system.

Output tables:

- `outputs/phase5_tables/overlay_performance_comparison.csv`
- `outputs/phase5_tables/overlay_exposure_summary.csv`
- `outputs/phase5_tables/benchmark_overlay_comparison.csv`
- `outputs/phase5_tables/transition_diagnostics.csv`
- `outputs/phase5_tables/non_converged_exclusion_diagnostics.csv`

Output figures:

- `outputs/phase5_figures/cumulative_returns_baseline_vs_ri_overlay.png`
- `outputs/phase5_figures/exposure_over_time.png`
- `outputs/phase5_figures/drawdown_baseline_vs_ri_overlay.png`

Phase 5 writes to `outputs/phase5_tables/` and `outputs/phase5_figures/`.
Generated Phase 5 outputs are also ignored by git. Interpret them as
exploratory diagnostics until robustness checks, transaction-cost assumptions,
turnover analysis, and out-of-sample tests are complete.

## Run The Phase 6 Research Evaluation

```bash
.venv/bin/python examples/run_phase6_phd_research_evaluation.py
```

The default run is a fast smoke workflow that reuses the saved Phase 4
`graph_regime_indicator.csv` and convergence diagnostics. To recompute the
one-configuration quick grid:

```bash
.venv/bin/python examples/run_phase6_phd_research_evaluation.py --recompute-quick
```

To run the controlled 24-configuration grid:

```bash
.venv/bin/python examples/run_phase6_phd_research_evaluation.py --full-grid
```

This workflow reads saved Phase 4 tables from `outputs/tables/`, runs
robustness and out-of-sample diagnostics, evaluates incremental information
beyond simple benchmarks, and adds transaction-cost sensitivity for the RI
overlay. With `--recompute-quick` or `--full-grid`, it recomputes RI for the
selected Phase 6 configurations.

Output tables:

- `outputs/phase6_tables/robustness_grid_summary.csv`
- `outputs/phase6_tables/robustness_overlay_metrics.csv`
- `outputs/phase6_tables/robustness_convergence_summary.csv`
- `outputs/phase6_tables/robustness_benchmark_comparison.csv`
- `outputs/phase6_tables/oos_selection_summary.csv`
- `outputs/phase6_tables/oos_test_performance.csv`
- `outputs/phase6_tables/oos_overlay_comparison.csv`
- `outputs/phase6_tables/incremental_information_tests.csv`
- `outputs/phase6_tables/transaction_cost_sensitivity.csv`

Optional output figures:

- `outputs/phase6_figures/robustness_sharpe_heatmap.png`
- `outputs/phase6_figures/robustness_max_drawdown_heatmap.png`
- `outputs/phase6_figures/robustness_calmar_heatmap.png`

Phase 6 is research-only. It is intended to test whether the RI overlay has
robust incremental risk-management value beyond simple VIX, volatility,
drawdown, and correlation overlays. It is not a trading system, not a broker
integration, and not validation of a profitable strategy. Generated Phase 6
tables and figures are ignored by git.
