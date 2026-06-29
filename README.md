# Graph-Lasso Laplacian Regime Indicator

Research-grade Python tools for testing whether rolling graphical-lasso
Laplacian features can behave like a systemic stress or recession regime
indicator. The project currently includes the Phase 1 through Phase 7 research
workflow: core graph features, benchmark data, empirical diagnostics, visual
diagnostics, research-only risk-overlay analysis, and robustness/out-of-sample
evaluation, plus PCA baselines and graph-component studies.

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
- Phase 7: PCA/correlation-spectrum baselines, graph-component model ladders,
  component overlays, and graph-component ablation diagnostics.

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
- `outputs/tables/con