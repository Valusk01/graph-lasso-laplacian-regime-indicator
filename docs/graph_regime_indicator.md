# Graph-Lasso Laplacian Regime Indicator

This module tests whether rolling conditional-dependence topology in asset
returns can act as a financial regime indicator. It currently implements the
Phase 1 core engine and the Phase 2 benchmark data layer.

Phase 1 covers rolling graph estimation, Laplacian feature extraction, and a
standardized composite regime indicator. Phase 2 adds data-loading utilities and
benchmark stress variables for later empirical validation.

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
correspond to conditional independence relationships between variables. In that
setting, nonzero off-diagonal precision entries indicate conditional dependence
between two assets after accounting for all other assets in the window.

In financial return applications, this interpretation should be used carefully.
Asset returns are not generally Gaussian, and their dependence structure may be
time-varying, heavy-tailed, nonlinear, and affected by volatility clustering.
Therefore, outside the Gaussian case, the safer interpretation is that
graphical lasso provides a sparse estimate of linear conditional-dependence
structure through partial correlations.

This distinction matters for regime research because a crisis may change the
conditional dependence network, not only the unconditional correlation matrix.

## Why Convert Precision to Partial Correlations

Graphical lasso does not directly generate a graph Laplacian. Its output is a
sparse precision matrix. To build a weighted graph with interpretable edge
weights, the precision matrix is converted into partial correlations:

```text
partial_corr_ij = -Theta_ij / sqrt(Theta_ii * Theta_jj)
The module then uses the absolute value of each off-diagonal partial correlation
as the weighted adjacency matrix. This Phase 1 signal focuses on dependence
magnitude and systemic connectedness rather than the sign of the relationship.

The resulting graph should therefore be interpreted as a sparse
partial-correlation network. It is not a distribution-free causal graph and
should not be described as a direct estimator of causal relationships.

Why Build a Laplacian

The weighted graph Laplacian is defined as:
D_ii = sum_j A_ij
L = D - A

The Laplacian links local edge weights to global graph structure. Its spectrum
summarizes connectivity, fragmentation, and dominance of large-scale network
modes.

In this project, the Laplacian is used to transform a rolling
partial-correlation network into numerical features that can be tracked through
time and compared with known financial stress indicators.

Why These Features
average_graph_strength: typical total conditional-dependence weight attached
to each asset. Higher values suggest stronger market-wide connectedness.
weighted_edge_density: total partial-correlation weight relative to the
number of possible edges. Higher values suggest a denser dependence network.
algebraic_connectivity: the second-smallest Laplacian eigenvalue. Higher
values imply the graph is harder to split into weakly connected blocks.
largest_laplacian_eigenvalue: a scale measure for the strongest Laplacian
spectral mode.
largest_laplacian_eigenvalue_share: the largest eigenvalue divided by total
Laplacian spectral mass. Higher values suggest more concentrated systemic
structure.
modularity: estimated community separation when NetworkX community tools are
available. Lower modularity is consistent with less segmented, less
diversified market structure.
laplacian_frobenius_change: window-to-window topology turnover. The first
window is reported as NaN because no previous Laplacian exists.

The composite regime indicator is:
RI_t =
  z(average_graph_strength_t)
  + z(algebraic_connectivity_t)
  + z(largest_laplacian_eigenvalue_share_t)
  - z(modularity_t)
  + z(laplacian_frobenius_change_t)

Z-scores are computed across the available time series of each feature, not
inside each rolling window. Constant or unavailable components receive a neutral
zero z-score.

High values of the regime indicator are intended to represent a more systemic,
more connected, less diversified market state.

Why Fixed Alpha Matters

The graphical-lasso penalty alpha is fixed across windows by default. Re-tuning
the penalty inside every rolling window could create artificial topology changes
that reflect the model-selection procedure rather than genuine market structure.

For this reason, changes in the estimated graph should come primarily from
changes in the return data, not from changes in the regularization rule.

Later robustness checks should still evaluate whether the main conclusions are
sensitive to alternative fixed values of alpha.

Volatility Contamination Risk

Returns are standardized within each window to reduce direct volatility
contamination. This does not fully eliminate the risk that volatility regimes,
heavy tails, or outliers influence the estimated dependence graph.

This issue is important in financial applications because crises often involve
both higher volatility and stronger co-movement. A graph-based indicator may rise
because the conditional-dependence topology is genuinely changing, but it may
also partly reflect volatility clustering or extreme observations.

Later phases should test whether the indicator adds information beyond simpler
volatility and correlation benchmarks.

Phase 2 Benchmark Layer

Phase 2 adds data-loading utilities and benchmark stress variables so the graph
regime indicator can later be compared against familiar market stress measures.
The purpose is not to prove the indicator is correct, but to create a clean
comparison layer for future validation.

Benchmark comparison matters because the graph-lasso Laplacian indicator is a
new derived topology signal. If it rises during known stress episodes, broad
market drawdowns, volatility shocks, high-correlation regimes, or recessionary
periods, that behavior would be consistent with the systemic connectedness
hypothesis. If it does not, the feature design, estimation choices, or
hypothesis may need revision.

The Phase 2 benchmarks are:

VIX: an option-implied volatility index for the S&P 500. It captures expected
near-term equity volatility and is often elevated during market stress.
Market drawdown: loss from a running high-water mark. It captures realized
market damage and crisis depth rather than expected volatility.
Realized volatility: rolling standard deviation of returns, optionally
annualized. It captures recent turbulence in observed returns.
Average rolling correlation: the average off-diagonal correlation across
assets. It captures the familiar tendency for diversification to weaken when
many assets move together.
Average absolute rolling correlation: the average absolute off-diagonal
correlation across assets. This can be useful when the universe contains
assets such as equities, bonds, commodities, and defensive assets, because
stress may strengthen dependence even when some relationships are negative.
Recession indicators: optional FRED series such as USREC mark official or
model-based macro recession periods, depending on the selected series.

These benchmarks are imperfect but useful. VIX is equity-option-specific,
drawdown is path-dependent, realized volatility can miss slow-moving stress,
average raw correlation is unconditional rather than conditional, average
absolute correlation ignores sign, and recession indicators are low-frequency
and often backward-looking. They are still helpful anchors because each captures
a different observable aspect of stress.

Benchmark agreement is not causal proof. A graph regime indicator can co-move
with VIX, drawdowns, volatility, correlation, or recession labels without
proving that network topology causes stress or predicts future crises. Causal
interpretation, out-of-sample evaluation, robustness checks, and predictive
testing remain later phase work.

Phase 3 Empirical Evaluation Layer

Phase 3 adds table-based diagnostics for comparing the graph regime indicator
with benchmark stress variables and future market outcomes. These diagnostics
support empirical inspection, but they do not claim final validation.

Contemporaneous diagnostics align the regime indicator with benchmark variables
on common dates. For binary stress labels, the evaluation layer compares stress
and non-stress periods by reporting observation counts, stress and non-stress
means and medians, differences in means, stress-to-non-stress ratios, and Welch
t-statistics when both groups have enough observations. This tests whether the
indicator tends to be higher during benchmark stress states.

For continuous benchmark variables, the diagnostics report Pearson and Spearman
correlations. Pearson correlation measures linear co-movement with variables
such as VIX, realized volatility, drawdown, or average correlation. Spearman
rank correlation checks whether the indicator is monotonic with those benchmarks
even when the relationship is not linear.

Predictive diagnostics are harder than contemporaneous validation because they
must avoid look-ahead. Phase 3 constructs forward targets using returns from
t+1 through t+h only. The targets include forward realized volatility, forward
return sums, forward absolute return sums, and forward maximum drawdown over
several horizons. The evaluation then reports correlations and simple
univariate OLS diagnostics of each forward target on the current regime
indicator.

The event-study helper extracts the indicator path around supplied event dates.
Each event is matched to the closest available indicator date less than or equal
to the event date, then an observation-window path is returned using relative
day offsets. This supports later crisis-window inspection without requiring a
plotting layer.

The quantile-classification helper labels indicator values as
low_systemic_connectedness, normal, or high_systemic_connectedness. Summary
tables can then compare observation counts, average returns, volatility, and
average absolute returns across indicator regimes.

Positive alignment with stress benchmarks or future volatility targets does not
prove causality. It may reflect shared exposure to volatility, market beta,
sample-specific crisis periods, or other omitted variables. Phase 3 produces
diagnostic tables for later research; robustness checks, out-of-sample
validation, and causal interpretation remain future work.

Interpretation of Graphical Lasso

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
described as a systemic connectedness indicator based on the rolling topology of
sparse partial-correlation networks.

The graphical lasso does not directly generate a graph Laplacian. The pipeline
is:
returns → standardized rolling window → sparse precision matrix → partial
correlation matrix → weighted adjacency matrix → graph Laplacian → spectral and
topological features

Current Validation Status

The project currently provides the machinery needed to estimate the graph-based
regime indicator, construct benchmark stress variables, and produce empirical
diagnostic tables.

It is not yet a validated recession, crisis, drawdown, or VIX signal. Visual
diagnostics, robustness checks, dashboards, notebooks, and final empirical
conclusions are intentionally left for later phases.

The Phase 3 diagnostics can be used to evaluate whether the graph-lasso
Laplacian regime indicator:

rises during known stress regimes;
co-moves with VIX, drawdowns, realized volatility, and correlation benchmarks;
distinguishes stress from non-stress periods;
contains predictive information for future volatility, drawdowns, or
correlation spikes.

Later phases should focus on robustness, visual diagnostics, and careful
interpretation of these empirical results.


---
