# Graph-Lasso Laplacian Regime Indicator

This module tests whether rolling conditional-dependence topology in asset
returns can act as a financial regime indicator. It implements only the Phase 1
core engine: rolling graph estimation, Laplacian feature extraction, and a
standardized composite indicator.

## Why graphical lasso

Simple return correlations measure pairwise co-movement without controlling for
the rest of the asset universe. Graphical lasso instead estimates a sparse
precision matrix, the inverse covariance matrix. Nonzero off-diagonal precision
entries indicate conditional dependence between two assets after accounting for
all other assets in the window.

This distinction matters for regime research because a crisis may change the
conditional dependence network, not only the unconditional correlation matrix.

## Why convert precision to partial correlations

Graphical lasso does not directly generate a graph Laplacian. Its output is a
sparse precision matrix. To build a weighted graph with interpretable edge
weights, the precision matrix is converted into partial correlations:

```text
partial_corr_ij = -Theta_ij / sqrt(Theta_ii * Theta_jj)
```

The module then uses the absolute value of each off-diagonal partial correlation
as the weighted adjacency matrix. This Phase 1 signal focuses on dependence
magnitude and systemic connectedness rather than the sign of the relationship.

## Why build a Laplacian

The weighted graph Laplacian is defined as:

```text
D_ii = sum_j A_ij
L = D - A
```

The Laplacian links local edge weights to global graph structure. Its spectrum
summarizes connectivity, fragmentation, and dominance of large-scale network
modes.

## Why these features

- `average_graph_strength`: typical total conditional-dependence weight attached
  to each asset. Higher values suggest stronger market-wide connectedness.
- `weighted_edge_density`: total partial-correlation weight relative to the
  number of possible edges. Higher values suggest a denser dependence network.
- `algebraic_connectivity`: the second-smallest Laplacian eigenvalue. Higher
  values imply the graph is harder to split into weakly connected blocks.
- `largest_laplacian_eigenvalue`: a scale measure for the strongest Laplacian
  spectral mode.
- `largest_laplacian_eigenvalue_share`: the largest eigenvalue divided by total
  Laplacian spectral mass. Higher values suggest more concentrated systemic
  structure.
- `modularity`: estimated community separation when NetworkX community tools are
  available. Lower modularity is consistent with less segmented, less
  diversified market structure.
- `laplacian_frobenius_change`: window-to-window topology turnover. The first
  window is reported as `NaN` because no previous Laplacian exists.

The composite regime indicator is:

```text
RI_t =
  z(average_graph_strength_t)
  + z(algebraic_connectivity_t)
  + z(largest_laplacian_eigenvalue_share_t)
  - z(modularity_t)
  + z(laplacian_frobenius_change_t)
```

Z-scores are computed across the available time series of each feature, not
inside each rolling window. Constant or unavailable components receive a neutral
zero z-score.

## Why fixed alpha matters

The graphical-lasso penalty `alpha` is fixed across windows by default. Re-tuning
the penalty inside every rolling window could create artificial topology changes
that reflect the model-selection procedure rather than genuine market structure.

## Volatility contamination risk

Returns are standardized within each window to reduce direct volatility
contamination. This does not fully eliminate the risk that volatility regimes,
heavy tails, or outliers influence the estimated dependence graph. Later phases
should test whether the indicator adds information beyond simpler volatility and
correlation benchmarks.

## Validation status

This engine is not yet a validated recession, crisis, drawdown, or VIX signal.
Benchmark comparisons, external data downloads, dashboards, notebooks, reports,
and predictive evaluation are intentionally left for later phases.

## Phase 2 Benchmark Layer

Phase 2 adds data-loading utilities and benchmark stress variables so the graph
regime indicator can later be compared against familiar market stress measures.
The purpose is not to prove the indicator is correct, but to create a clean
comparison layer for future validation.

Benchmark comparison matters because the graph-lasso Laplacian indicator is a
new derived topology signal. If it rises during known stress episodes, broad
market drawdowns, volatility shocks, or high-correlation regimes, that behavior
would be consistent with the systemic connectedness hypothesis. If it does not,
the feature design, estimation choices, or hypothesis may need revision.

The Phase 2 benchmarks are:

- VIX: an option-implied volatility index for the S&P 500. It captures expected
  near-term equity volatility and is often elevated during market stress.
- Market drawdown: loss from a running high-water mark. It captures realized
  market damage and crisis depth rather than expected volatility.
- Realized volatility: rolling standard deviation of returns, optionally
  annualized. It captures recent turbulence in observed returns.
- Average rolling correlation: the average off-diagonal correlation across
  assets. It captures the familiar tendency for diversification to weaken when
  many assets move together.
- Recession indicators: optional FRED series such as `USREC` mark official or
  model-based macro recession periods, depending on the selected series.

These benchmarks are imperfect but useful. VIX is equity-option-specific,
drawdown is path-dependent, realized volatility can miss slow-moving stress,
average correlation is unconditional rather than conditional, and recession
indicators are low-frequency and often backward-looking. They are still helpful
anchors because each captures a different observable aspect of stress.

Benchmark agreement is not causal proof. A graph regime indicator can co-move
with VIX, drawdowns, volatility, or recession labels without proving that network
topology causes stress or predicts future crises. Causal interpretation,
out-of-sample evaluation, robustness checks, and predictive testing remain later
phase work.

## Interpretation of Graphical Lasso

Graphical lasso estimates a sparse inverse covariance matrix, also called a
precision matrix. Under Gaussian assumptions, zero off-diagonal entries in the
precision matrix correspond to conditional independence relationships between
variables.

In financial return applications, this interpretation should be used carefully.
Asset returns are not generally Gaussian, and their dependence structure may be
time-varying, heavy-tailed, and affected by volatility clustering. Therefore,
outside the Gaussian case, the safer interpretation is that graphical lasso
provides a sparse estimate of linear conditional-dependence structure through
partial correlations.

For this reason, the regime indicator should not be described as a
distribution-free causal or conditional-independence estimator. It should be
described as a systemic connectedness indicator based on the rolling topology
of sparse partial-correlation networks.

The graphical lasso does not directly generate a graph Laplacian. The pipeline
is:

returns → standardized rolling window → sparse precision matrix → partial
correlation matrix → weighted adjacency matrix → graph Laplacian → spectral and
topological features.
