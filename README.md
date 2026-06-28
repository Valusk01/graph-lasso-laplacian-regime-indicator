# Graph-Lasso Laplacian Regime Indicator

Research-grade Python tools for testing whether rolling graphical-lasso
Laplacian features can behave like a systemic stress or recession regime
indicator.

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

Generated Phase 5 outputs are also ignored by git. Interpret them as
exploratory diagnostics until robustness checks and out-of-sample tests are
complete.
