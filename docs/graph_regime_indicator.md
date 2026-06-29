# Graph-Lasso Laplacian Regime Indicator

This project tests whether rolling conditional-dependence topology in asset
returns can act as a financial regime indicator. It is a quantitative research
project, not a production trading system.

The project now implements:

- Phase 1: graph-lasso/Laplacian feature engine.
- Phase 2: benchmark and data layer.
- Phase 3: empirical evaluation diagnostics.
- Phase 4: visual diagnostics workflow.
- Phase 5: robustness and research-only risk-overlay diagnostics.
- Phase 6: controlled robustness, out-of-sample, incremental-information,
  transaction-cost, and turnover diagnostics.

## Research Hypothesis

During crisis, recessionary, or systemic stress regimes, the conditional
dependence topology of asset returns changes. A sparse inverse-covariance
network estimated through graphical lasso may become more globally connected,
less modular, and more dominated by systemic market structure. Rolling graph
Laplacian features can summarize those changes into a systemic connectedness
regime indicator.

The indicator should not be interpreted as proving that crises are caused by
network topology changes. The objective is more modest: to test whether changes
in sparse partial-correlation network structure contain useful contemporaneous
or predictive information about financial stress regimes.

## Why Graphical Lasso

Simple return correlations measure pairwise co-movement without controlling for
the rest of the asset universe. Graphical lasso instead estimates a sparse
precision matrix, also called the inverse covariance matrix.

Under Gaussian assumptions, zero off-diagonal entries in the precision matrix
correspond to conditional independence relationships between variables. In
financial return applications, this interpretation should be used carefully
because returns are heavy-tailed, nonlinear, heteroskedastic, and affected by
volatility clustering. Outside the Gaussian case, the safer interpretation is
that graphical lasso provides a sparse estimate of linear conditional-dependence
structure through partial correlations.

Graphical lasso does not directly generate a graph Laplacian. The pipeline is:

```text
returns -> standardized rolling window -> sparse precision matrix
        -> partial correlation matrix -> weighted adjacency matrix
        -> graph Laplacian -> spectral and topological features
```

## Why Convert Precision to Partial Correlations

Graphical lasso outputs a sparse precision matrix. To build a weighted graph
with interpretable edge weights, the precision matrix is converted into partial
correlations:

```text
partial_corr_ij = -Theta_ij / sqrt(Theta_ii * Theta_jj)
```

The module then uses the absolute value of each off-diagonal partial
correlation as the weighted adjacency matrix. This Phase 1 signal focuses on
dependence magnitude and systemic connectedness rather than the sign of the
relationship.

The resulting graph should be interpreted as a sparse partial-correlation
network. It is not a distribution-free causal graph and should not be described
as a direct estimator of causal relationships.

## Why Build a Laplacian

The weighted graph Laplacian is defined as:

```text
D_ii = sum_j A_ij
L = D - A
```

The Laplacian links local edge weights to global graph structure. Its spectrum
summarizes connectivity, fragmentation, and dominance of large-scale network
modes. In this project, the Laplacian transforms a rolling partial-correlation
network into numerical features that can be tracked through time and compared
with known financial stress indicators.

## Why These Features

The Phase 1 engine extracts graph and spectral features designed to summarize
systemic connectedness:

- `average_graph_strength`: typical total conditional-dependence weight
  attached to each asset.
- `weighted_edge_density`: total partial-correlation weight relative to the
  number of possible edges.
- `algebraic_connectivity`: the second-smallest Laplacian eigenvalue; higher
  values imply the graph is harder to split into weakly connected blocks.
- `largest_laplacian_eigenvalue`: scale measure for the strongest Laplacian
  spectral mode.
- `largest_laplacian_eigenvalue_share`: largest eigenvalue divided by total
  Laplacian spectral mass.
- `modularity`: optional estimated community separation. Lower modularity is
  consistent with less segmented market structure, but it is disabled by
  default for speed and stability.
- `laplacian_frobenius_change`: window-to-window topology turnover.

The composite regime indicator is:

```text
RI_t =
  z(average_graph_strength_t)
  + z(algebraic_connectivity_t)
  + z(largest_laplacian_eigenvalue_share_t)
  - z(modularity_t)
  + z(laplacian_frobenius_change_t)
```

Z-scores are computed across the available feature time series, not inside each
rolling window. Constant or unavailable components receive a neutral zero
z-score. When modularity is disabled or unavailable, it is reported as `NaN`
and `z_modularity` is neutralized to zero.

High values of the regime indicator are intended to represent a more systemic,
more connected, less diversified market state. The empirical outputs show that
this interpretation remains a hypothesis, not a validated fact.

## Why Fixed Alpha Matters

The graphical-lasso penalty alpha is fixed across windows by default. Re-tuning
the penalty inside every rolling window could create artificial topology
changes that reflect the model-selection procedure rather than genuine market
structure.

Changes in the estimated graph should come primarily from changes in the return
data, not from changes in the regularization rule. Robustness checks should
still evaluate whether conclusions are sensitive to alternative fixed values of
alpha.

## Volatility Contamination Risk

Returns are standardized within each window to reduce direct volatility
contamination. This does not fully eliminate the risk that volatility regimes,
heavy tails, or outliers influence the estimated dependence graph.

Crises often involve both higher volatility and stronger co-movement. A
graph-based indicator may rise because conditional-dependence topology is
genuinely changing, but it may also partly reflect volatility clustering or
extreme observations. Later interpretation should therefore compare the graph
indicator against simpler volatility and correlation benchmarks.

## Phase 2 Benchmark Layer

Phase 2 adds data-loading utilities and benchmark stress variables so the graph
regime indicator can be compared against familiar market stress measures.

The benchmark layer supports:

- VIX, an option-implied volatility index for the S&P 500.
- Market drawdown, defined as loss from the running high-water mark.
- Realized volatility, computed from rolling returns.
- Average rolling correlation.
- Average absolute rolling correlation.
- Optional recession indicators from FRED.

These benchmarks are imperfect but useful. VIX is equity-option-specific,
drawdown is path-dependent, realized volatility is backward-looking, average
correlation is unconditional, and recession indicators are low-frequency. They
are still useful anchors because each captures a different observable aspect of
stress.

Benchmark agreement is not causal proof. A graph regime indicator can co-move
with VIX, drawdowns, volatility, or correlation without proving that network
topology causes stress or predicts future crises.

## Phase 3 Empirical Evaluation Layer

Phase 3 adds table-based diagnostics for comparing the graph regime indicator
with benchmark stress variables and future market outcomes.

Contemporaneous diagnostics compare RI in stress versus non-stress periods and
compute correlations with continuous benchmarks such as VIX, realized
volatility, drawdown, and rolling correlation.

Predictive diagnostics construct forward targets using returns from `t+1`
through `t+h` only. Targets include forward realized volatility, forward return
sums, forward absolute return sums, and forward maximum drawdown. These are
screening statistics, not causal or trading evidence.

The evaluation layer also includes event-study extraction and quantile-based
regime class summaries.

## Phase 4 Visual Diagnostics Workflow

Phase 4 adds reusable matplotlib plotting functions and an example workflow
that writes diagnostic figures and tables under `outputs/`.

The visual diagnostics include:

- RI time-series plot with stress markers.
- RI overlays against VIX, drawdown, realized volatility, average correlation,
  and average absolute correlation.
- Graph feature panel.
- Stress versus non-stress boxplot.
- RI versus forward-risk scatter plot.
- Regime-class volatility bar chart.

These figures are exploratory visual evidence. They are not robustness checks
and do not validate the indicator.

## Graphical-Lasso Convergence Diagnostics

Rolling graphical-lasso estimation can fail to converge in some windows. A
non-converged window does not necessarily crash the workflow, but it weakens
confidence in that window's precision matrix and therefore in the partial
correlation graph and Laplacian features derived from it.

The rolling feature table includes:

- `graph_lasso_converged`
- `graph_lasso_n_iter`
- `graph_lasso_alpha`
- `graph_lasso_max_iter`
- `graph_lasso_tol`
- `graph_lasso_enet_tol`
- `graph_lasso_mode`
- `graph_lasso_warning_count`
- `graph_lasso_warning_message`

Workflows capture `ConvergenceWarning` messages so repeated rolling fits do not
spam the terminal. Users can choose whether non-convergence should be recorded,
warned once, or treated as an error.

Convergence diagnostics are part of model-risk control. They should be reported
alongside empirical results because unstable precision-matrix estimates can
change the inferred network topology.

## Phase 5 Robustness and Risk-Overlay Layer

Phase 5 adds robustness and research-only risk-overlay diagnostics. It is not a
trading system, does not connect to a broker, and does not validate the
indicator.

The motivation for a risk overlay is modest. The current empirical evidence is
more consistent with RI as a topology-transition or instability feature than as
a direct persistent stress-level signal. A risk overlay asks whether exposure
scaling during high-RI regimes changes realized portfolio risk, drawdowns, and
risk-adjusted performance. It does not attempt to forecast return direction.

The first Phase 5 RI overlay run improved several metrics versus the baseline
equal-weight portfolio:

- Sharpe improved from `0.5720` to `0.6976`.
- Max drawdown improved from `-0.3308` to `-0.2034`.
- Annualized volatility improved from `0.1354` to `0.1154`.
- Calmar improved from `0.2340` to `0.3958`.

Compared with simple benchmark overlays, the RI overlay had the best Sharpe and
Calmar in this run. Some benchmark overlays, especially VIX, drawdown, and
realized-volatility overlays, had slightly lower volatility or slightly better
max drawdown. This is encouraging for the refined risk-overlay hypothesis, but
it is preliminary and requires robustness and out-of-sample testing.

All overlay tests should be compared against simple baselines. Relevant
benchmarks include VIX, realized volatility, drawdown, average correlation,
average absolute correlation, and simple correlation-spectrum measures such as
the first PCA eigenvalue of the correlation matrix.

Exposure thresholds can be computed in full-sample mode for diagnostics or in
expanding mode for more realistic evaluation. Expanding thresholds are
preferred because each date uses only prior information. Phase 5 defaults to
expanding thresholds and shifts exposure by one observation before applying it
to returns to reduce look-ahead bias.

Phase 5 also introduces transition diagnostics. These compare RI against
changes in benchmark variables and against stress-onset labels, rather than
only persistent stress labels.

Generated Phase 5 results remain exploratory until out-of-sample tests are
completed. Robustness checks should include alpha grids, rolling-window grids,
partial-correlation thresholds, non-converged-window exclusions, alternative
asset universes, combinatorial versus normalized Laplacian features, signed
versus absolute partial-correlation networks, transaction costs, and turnover
analysis.

## Phase 6 Robustness and Out-of-Sample Evaluation

Phase 6 implements a research-grade evaluation framework for the refined
risk-overlay hypothesis: RI may be useful as a topology-transition feature that
complements conventional risk indicators, even if it is not validated as a
persistent stress-level indicator.

The controlled robustness grid varies:

- graphical-lasso alpha: `0.05`, `0.10`, `0.15`, `0.20`;
- rolling window length: `63`, `126`, `252` trading days;
- partial-correlation threshold: `0.00`, `0.03`.

The Laplacian type remains combinatorial and the network remains based on
absolute partial correlations. Holding those dimensions fixed keeps Phase 6
focused on the most important current degrees of freedom without turning the
project into an unconstrained in-sample optimizer.

For each configuration, the framework can recompute graph features and RI,
record graphical-lasso convergence, compute stress and transition diagnostics,
evaluate forward-risk relationships, and compare RI risk overlays against
simple benchmark overlays. Results are written to `outputs/phase6_tables/`;
optional robustness heatmaps are written to `outputs/phase6_figures/`.

Phase 6 also adds a walk-forward-style protocol:

- development sample: 2015-07-06 to 2019-12-31;
- validation sample: 2020-01-01 to 2022-12-31;
- test sample: 2023-01-01 onward.

Configuration selection is based on validation-period risk-overlay diagnostics,
not test-period performance. The selection score combines Sharpe, Calmar,
drawdown reduction, volatility reduction, downside-tail improvement, and a
turnover penalty. This is still a simple research protocol, but it prevents
choosing configurations purely because they maximize in-sample Sharpe.

Incremental-information tests evaluate whether RI adds explanatory information
beyond VIX, realized volatility, drawdown, average correlation, and average
absolute correlation. The tests cover forward realized volatility, forward max
drawdown, and stress-onset labels. They report adjusted R-squared, out-of-sample
R-squared where feasible, coefficient signs, robust/Newey-West-style
t-statistics, and classification diagnostics for stress-onset labels.

Transaction-cost sensitivity reports average exposure, time in reduced
exposure, number of exposure changes, annualized turnover proxy, and performance
after simple cost assumptions of 0, 5, and 10 basis points per unit exposure
change. These are not full execution-cost models, but they are necessary before
interpreting any overlay result as economically meaningful.

Phase 6 is still research-only. It does not prove profitability, validate RI as
a standalone trading signal, or establish that graph topology causes risk
reduction. Its purpose is to test whether the Phase 5 result survives
reasonable parameter changes, no-lookahead sample discipline, simple benchmark
comparisons, and cost/turnover assumptions.

## Current Validation Status

The project currently provides the machinery needed to estimate the graph-based
regime indicator, construct benchmark stress variables, produce empirical
diagnostic tables, generate visual diagnostic figures, and run Phase 5 and
Phase 6 research-only risk-overlay diagnostics.

It is not yet a validated recession, crisis, drawdown, VIX, or trading signal.
Dashboards, notebooks, live trading, broker integration, and final empirical
conclusions are intentionally out of scope.

The updated empirical outputs show mixed stress-level evidence, suggestive
topology-transition behavior, weak predictive diagnostics, and encouraging but
preliminary risk-overlay results. The next research step is to interpret Phase
6 robustness, out-of-sample, incremental-information, and transaction-cost
outputs, not deployment.
