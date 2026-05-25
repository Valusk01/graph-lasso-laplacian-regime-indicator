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
  features, and a composite regime indicator.
- Phase 2: data loading helpers, optional public-data download helpers, and
  benchmark stress variables for later empirical comparison.

Not implemented yet: dashboards, notebooks, external benchmark evaluation,
predictive testing, robustness checks, or final empirical conclusions.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
```

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
