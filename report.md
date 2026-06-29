# Graph-Lasso Laplacian Regime Indicator — Updated Output Interpretation

## 1. Executive Summary

The updated workflow outputs are now consistent with the current convergence-diagnostics code. The regenerated feature table contains graphical-lasso solver diagnostics, and `outputs/tables/graph_lasso_convergence_diagnostics.csv` is present.

The implementation remains methodologically coherent: graphical lasso is used to estimate sparse precision matrices; precision matrices are converted into partial correlations; absolute partial correlations form weighted adjacency matrices; graph Laplacians are computed as `L = D - A`; and Laplacian graph features are combined into a time-series z-scored regime indicator.

The convergence result is reassuring but not perfect. Out of `2,761` rolling windows, `2,733` converged and `28` did not, for a convergence rate of `98.99%`. This is acceptable for exploratory interpretation, provided the failed windows are reported and not ignored. The failed windows are not random-looking: `20` of the `28` failed windows occur during the combined systemic stress label, and `8` occur in 2020. That concentration means crisis-period interpretations should remain cautious.

The empirical evidence is mixed. The regime indicator clearly spikes around some stress onsets, especially February-March 2020, and high-RI regimes are associated with higher volatility and larger absolute returns. But the indicator is not higher on average during the combined systemic stress label. In the contemporaneous diagnostics, systemic-stress periods have mean RI of `-0.478` versus `0.255` in non-stress periods, a difference of `-0.733` with Welch t-statistic `-8.057`.

The strongest honest interpretation is that the current RI is not yet a validated persistent stress-level indicator. It looks more like a noisy topology-transition and risk-overlay feature: it can spike near abrupt market-structure changes, and high-RI regimes are associated with higher subsequent risk, but it does not consistently track sustained VIX, drawdown, realized-volatility, or rolling-correlation stress.

The predictive diagnostics are weak. RI has small positive Pearson correlations with future realized volatility over 5, 21, and 63 days, but R-squared values are tiny, ranging from about `0.0040` to `0.0114`. Higher RI is also associated with lower forward return sums and somewhat worse short-horizon forward drawdowns, but the explanatory power is too small for a standalone trading claim.

Bottom line: the codebase is ready for deeper research, and the updated outputs provide useful preliminary evidence. They do not validate the original hypothesis yet, and they are not sufficient for a standalone trading signal.

## 2. Current Output State

Verified output tables:

- `outputs/tables/graph_regime_features.csv`
- `outputs/tables/graph_regime_indicator.csv`
- `outputs/tables/benchmark_stress_labels.csv`
- `outputs/tables/contemporaneous_diagnostics.csv`
- `outputs/tables/predictive_diagnostics.csv`
- `outputs/tables/regime_class_summary.csv`
- `outputs/tables/graph_lasso_convergence_diagnostics.csv`
- `outputs/tables/returns.csv`

Verified output figures:

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

Main table shapes:

| Table | Shape | Date range |
| --- | ---: | --- |
| `graph_regime_features.csv` | `2761 x 18` | 2015-07-06 to 2026-06-26 |
| `graph_regime_indicator.csv` | `2761 x 24` | 2015-07-06 to 2026-06-26 |
| `benchmark_stress_labels.csv` | `2888 x 10` | 2015-01-02 to 2026-06-26 |
| `graph_lasso_convergence_diagnostics.csv` | `2761 x 9` | 2015-07-06 to 2026-06-26 |
| `contemporaneous_diagnostics.csv` | `65 x 4` | tidy diagnostics |
| `predictive_diagnostics.csv` | `84 x 3` | tidy diagnostics |
| `regime_class_summary.csv` | `3 x 5` | class summary |

`git status --short` showed `?? report.md` before this update. Generated outputs are ignored by git and were not modified by this review.

## 3. Methodological Recap

The pipeline is implemented in the correct order:

1. Start with an ETF return panel.
2. Clean returns and standardize each rolling window.
3. Estimate a sparse precision matrix with fixed-alpha graphical lasso.
4. Convert precision to partial correlations.
5. Build a weighted adjacency matrix from absolute partial correlations.
6. Compute the combinatorial graph Laplacian as `L = D - A`.
7. Extract graph and Laplacian spectral features.
8. Z-score selected features through time.
9. Combine them into a regime indicator.
10. Compare the indicator with benchmark stress labels, continuous benchmarks, forward targets, and visual diagnostics.

This avoids the central conceptual mistake of treating graphical lasso as if it directly produced a Laplacian. It does not: graphical lasso estimates a precision matrix, and the graph/Laplacian objects are derived later.

The interpretation should remain financial-statistical rather than causal. Under Gaussian assumptions, sparse precision entries have a conditional-independence interpretation. Financial ETF returns are heavy-tailed, heteroskedastic, and time-varying, so the safer interpretation is sparse linear conditional-dependence topology.

## 4. Convergence Diagnostics

The convergence diagnostics table exists and contains the expected columns:

- `graph_lasso_converged`
- `graph_lasso_n_iter`
- `graph_lasso_alpha`
- `graph_lasso_max_iter`
- `graph_lasso_tol`
- `graph_lasso_enet_tol`
- `graph_lasso_mode`
- `graph_lasso_warning_count`
- `graph_lasso_warning_message`

Summary:

| Metric | Value |
| --- | ---: |
| Rolling windows | 2,761 |
| Converged windows | 2,733 |
| Non-converged windows | 28 |
| Convergence rate | 98.99% |
| Total captured warnings | 28 |
| Alpha | 0.10 |
| Max iterations | 1,000 |
| Tolerance | 0.0001 |
| Enet tolerance | 0.0001 |
| Solver mode | `cd` |

Iteration counts:

| Statistic | Iterations |
| --- | ---: |
| Mean | 170.3 |
| Median | 105 |
| 25th percentile | 44 |
| 75th percentile | 229 |
| Maximum | 1,000 |

All non-converged windows reached `1,000` iterations and emitted one captured convergence warning.

Non-converged dates:

- 2015-08-26
- 2015-08-28
- 2015-11-25
- 2016-02-18
- 2016-03-14
- 2016-05-24
- 2016-06-23
- 2016-08-19
- 2018-03-07
- 2019-03-13
- 2019-09-16
- 2019-09-25
- 2020-04-24
- 2020-04-29
- 2020-05-15
- 2020-05-20
- 2020-07-06
- 2020-07-13
- 2020-07-16
- 2020-11-09
- 2023-01-09
- 2023-01-17
- 2023-01-31
- 2023-03-31
- 2025-07-16
- 2025-07-17
- 2025-09-04
- 2025-09-09

Failed windows by year:

| Year | Failed windows |
| --- | ---: |
| 2015 | 3 |
| 2016 | 5 |
| 2018 | 1 |
| 2019 | 3 |
| 2020 | 8 |
| 2023 | 4 |
| 2025 | 4 |

Non-convergence is somewhat concentrated around stress-like periods. Among the 28 failed windows:

| Stress label | Failed-window count |
| --- | ---: |
| `systemic_stress_label = 1` | 20 |
| `vix_stress_label = 1` | 5 |
| `drawdown_stress_label = 1` | 8 |
| `realized_volatility_stress_label = 1` | 14 |
| `correlation_stress_label = 1` | 15 |

This matters because stress periods are exactly where precision-matrix reliability is most important. The 99.0% convergence rate is reassuring for exploratory analysis, but the failed windows should be flagged in any empirical table or chart. A future robustness pass should rerun key diagnostics after excluding non-converged windows.

Average RI by convergence status:

| Status | Count | Mean RI | Median RI | Min RI | Max RI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Converged | 2,733 | 0.014 | 0.079 | -5.584 | 12.796 |
| Non-converged | 28 | -1.339 | -1.044 | -5.634 | 6.213 |

The correlation between a binary convergence indicator and RI is only `0.064`, so non-convergence does not mechanically explain the RI series. However, the failed windows have a lower average RI and include several deeply negative COVID-aftershock windows. They do not coincide with the top RI dates, but they do overlap with benchmark stress periods.

Recommendation: do not discard the entire empirical run. Report the 28 failed windows, flag them in downstream analysis, and add an exclusion sensitivity check in the next phase.

## 5. Regime Indicator Behavior

The updated regime indicator has 2,761 observations from 2015-07-06 to 2026-06-26.

RI summary statistics:

| Statistic | Value |
| --- | ---: |
| Mean | approximately 0.000 |
| Standard deviation | 2.131 |
| Minimum | -5.634 |
| 25th percentile | -1.183 |
| Median | 0.069 |
| 75th percentile | 1.303 |
| Maximum | 12.796 |

Top 10 RI dates:

| Date | RI | Systemic stress label |
| --- | ---: | ---: |
| 2020-03-09 | 12.796 | 1 |
| 2020-02-28 | 11.415 | 1 |
| 2020-10-08 | 11.139 | 1 |
| 2016-11-09 | 10.664 | 0 |
| 2025-04-04 | 10.049 | 1 |
| 2020-03-12 | 9.936 | 1 |
| 2020-03-13 | 9.270 | 1 |
| 2017-05-12 | 8.745 | 0 |
| 2017-05-15 | 8.586 | 0 |
| 2020-03-19 | 8.497 | 1 |

Bottom 10 RI dates:

| Date | RI |
| --- | ---: |
| 2020-05-20 | -5.634 |
| 2020-05-19 | -5.584 |
| 2020-05-08 | -5.577 |
| 2020-07-07 | -5.559 |
| 2020-05-22 | -5.547 |
| 2020-06-24 | -5.547 |
| 2020-05-21 | -5.544 |
| 2020-06-23 | -5.537 |
| 2020-06-18 | -5.535 |
| 2020-05-07 | -5.519 |

The top dates strongly include the initial COVID shock, which is consistent with the hypothesis that network topology changes around crisis onsets. But the bottom dates are also during the COVID stress aftermath, when benchmark volatility and correlation were still elevated. This is the key empirical tension: the indicator reacts strongly to some transition moments, then becomes low during parts of sustained stress.

Feature summary:

| Feature | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| `average_graph_strength` | 0.907 | 0.821 | 1.005 |
| `weighted_edge_density` | 0.0698 | 0.0631 | 0.0773 |
| `algebraic_connectivity` | 0.203 | 0.109 | 0.334 |
| `largest_laplacian_eigenvalue_share` | 0.145 | 0.117 | 0.160 |
| `laplacian_frobenius_change` | 0.0758 | 0.0080 | 0.691 |
| `number_of_edges` | 44.97 | 36 | 59 |

Component correlations with RI:

| Component | Correlation with RI |
| --- | ---: |
| `z_algebraic_connectivity` | 0.711 |
| `z_average_graph_strength` | 0.629 |
| `z_laplacian_frobenius_change` | 0.549 |
| `z_largest_laplacian_eigenvalue_share` | 0.241 |
| `z_modularity` | not active |

The indicator is mainly driven by algebraic connectivity, average graph strength, and Frobenius topology change. Largest eigenvalue share contributes less. Modularity is inactive: all modularity values are `NaN`, and `z_modularity` is neutralized to zero.

Interpretation: the current RI is not dominated by one feature, but its main contributors are connectivity and topology-turnover measures. Its behavior looks closer to a topology-transition signal than a simple persistent stress-level signal.

## 6. Benchmark Comparison

The benchmark table contains 2,888 dates. The systemic stress label is active on 961 dates and inactive on 1,927 dates. The contemporaneous diagnostics use 2,761 aligned RI observations.

### 6.1 Stress vs Non-Stress Diagnostics

The stress-vs-non-stress results are mixed and mostly not supportive of the original stress-level hypothesis.

| Label | Stress n | Non-stress n | Stress mean RI | Non-stress mean RI | Difference | Welch t-stat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `systemic_stress_label` | 961 | 1800 | -0.478 | 0.255 | -0.733 | -8.057 |
| `correlation_stress_label` | 553 | 2208 | -0.847 | 0.212 | -1.059 | -9.068 |
| `realized_volatility_stress_label` | 574 | 2187 | -0.478 | 0.125 | -0.603 | -4.834 |
| `drawdown_stress_label` | 407 | 2354 | 0.022 | -0.004 | 0.025 | 0.192 |
| `vix_stress_label` | 158 | 2603 | 0.209 | -0.013 | 0.222 | 0.686 |

The VIX and drawdown labels have slightly higher RI during stress, but the differences are small and not statistically meaningful in this diagnostic. The systemic, realized-volatility, and correlation stress labels show lower RI during stress. The correlation stress label is especially contrary to the original expectation.

This should be reported plainly: the updated outputs do not validate the claim that RI is generally higher during benchmark stress labels.

### 6.2 Continuous Benchmark Correlations

Continuous benchmark correlations:

| Benchmark | Pearson correlation | Spearman correlation |
| --- | ---: | ---: |
| `vix` | 0.028 | -0.000 |
| `drawdown` | 0.048 | 0.140 |
| `realized_volatility` | -0.085 | -0.094 |
| `average_correlation` | -0.163 | -0.152 |
| `average_absolute_correlation` | -0.315 | -0.280 |

The VIX relationship is essentially zero. The drawdown correlation is positive, but drawdown is negative by construction, so this means RI tends to be slightly higher when drawdowns are less severe, not more severe. The realized-volatility and correlation relationships are negative.

The negative relationship with average absolute correlation is important because the original hypothesis expected systemic connectedness to rise during stress. In the current specification, the graph-lasso Laplacian RI is not behaving like a raw-correlation stress proxy.

### 6.3 Stress-Level vs Topology-Transition Interpretation

The top RI dates suggest a more nuanced interpretation.

Top RI values include:

- 2020-03-09: VIX `54.46`, drawdown `-18.9%`, systemic stress active.
- 2020-02-28: VIX `40.11`, drawdown `-12.4%`, systemic stress active.
- 2020-03-12: VIX `75.47`, drawdown `-26.7%`, systemic stress active.
- 2020-03-13: VIX `57.83`, drawdown `-20.4%`, systemic stress active.
- 2020-03-19: VIX `72.00`, drawdown `-28.9%`, systemic stress active.

Those dates are consistent with abrupt stress onset and market-structure transition. However, the RI becomes sharply negative in May-July 2020, while stress labels remain active and rolling correlations remain high.

This suggests the current indicator may be sensitive to abrupt topology transitions, edge reconfiguration, and onset dynamics rather than persistent stress levels. That is a plausible and potentially useful signal, but it is different from the original broad claim that high RI should generally indicate systemic stress.

## 7. Predictive Diagnostics

Predictive diagnostics use forward targets and should be interpreted as simple screening statistics, not causal or trading evidence.

Forward realized volatility:

| Target | Pearson | Spearman | Beta t-stat | R-squared |
| --- | ---: | ---: | ---: | ---: |
| `forward_realized_volatility_5d` | 0.107 | -0.048 | 5.639 | 0.0114 |
| `forward_realized_volatility_21d` | 0.079 | -0.064 | 4.149 | 0.0062 |
| `forward_realized_volatility_63d` | 0.063 | -0.037 | 3.288 | 0.0040 |

RI has positive Pearson correlations with future realized volatility. This is directionally useful for risk management, but the effect is weak. Spearman correlations are slightly negative, implying the relationship is not robustly monotonic.

Forward return sums:

| Target | Pearson | Spearman | Beta t-stat | R-squared |
| --- | ---: | ---: | ---: | ---: |
| `forward_return_sum_5d` | -0.084 | -0.027 | -4.433 | 0.0071 |
| `forward_return_sum_21d` | -0.055 | -0.028 | -2.862 | 0.0030 |
| `forward_return_sum_63d` | -0.073 | -0.075 | -3.814 | 0.0054 |

Higher RI is associated with lower future return sums. This is consistent with a risk-off interpretation, but the explanatory power is very small.

Forward max drawdown:

| Target | Pearson | Spearman | Beta t-stat | R-squared |
| --- | ---: | ---: | ---: | ---: |
| `forward_max_drawdown_5d` | -0.101 | 0.016 | -5.304 | 0.0101 |
| `forward_max_drawdown_21d` | -0.064 | 0.028 | -3.338 | 0.0041 |
| `forward_max_drawdown_63d` | -0.031 | 0.075 | -1.614 | 0.0010 |

Because max drawdowns are negative, negative Pearson correlations mean higher RI is associated with worse future drawdowns. The 5-day result is the strongest of these, but still has only about `1.0%` R-squared.

Forward absolute return sums:

| Target | Pearson | Spearman | R-squared |
| --- | ---: | ---: | ---: |
| `forward_absolute_return_sum_5d` | 0.066 | -0.048 | 0.0044 |
| `forward_absolute_return_sum_21d` | -0.022 | -0.096 | 0.0005 |
| `forward_absolute_return_sum_63d` | -0.049 | -0.088 | 0.0024 |

The absolute-return evidence is mixed and weak.

Overall, predictive evidence is weak to modest. It is suggestive of some risk-overlay value, especially for short-horizon realized volatility and drawdown, but it is not strong enough for a standalone trading signal.

## 8. Regime Class Summary

Quantile-based regime classes:

| Regime class | Observations | Mean return | Volatility | Mean absolute return |
| --- | ---: | ---: | ---: | ---: |
| `high_systemic_connectedness` | 553 | -0.000863 | 0.012482 | 0.007326 |
| `low_systemic_connectedness` | 553 | 0.000931 | 0.009090 | 0.006263 |
| `normal` | 1655 | 0.000541 | 0.006627 | 0.004947 |

This is one of the more supportive diagnostics. High-RI regimes have:

- the lowest average return;
- the highest volatility;
- the highest mean absolute return.

That supports a risk-overlay interpretation. If RI is high, the market environment is more turbulent on average. However, low-RI regimes also have higher volatility and larger absolute returns than normal regimes, so the relationship is not a clean monotonic stress ladder. The result is useful but still exploratory.

## 9. Visual Diagnostics

`regime_indicator.png` shows visible RI spikes around several benchmark-stress points, especially the COVID onset. It also shows many stress markers during low or normal RI periods. This supports a transition interpretation more than a persistent stress-level interpretation.

`regime_indicator_vs_vix.png` shows a sharp RI spike around the initial COVID VIX surge. After that, RI drops while VIX remains high, so the two series do not move together consistently.

`regime_indicator_vs_drawdown.png` shows that RI spikes during some drawdown accelerations but does not remain elevated throughout deep drawdown periods.

`regime_indicator_vs_realized_volatility.png` shows similar behavior: RI reacts to some volatility shocks but does not consistently track realized volatility levels.

`regime_indicator_vs_average_correlation.png` and `regime_indicator_vs_average_absolute_correlation.png` visually support the negative correlation diagnostics. The RI is often low when rolling correlation measures are high.

`graph_feature_panel.png` shows that average graph strength and algebraic connectivity vary smoothly, while Frobenius change is spiky and captures abrupt topology shifts. This is consistent with the composite RI partly reflecting network-transition events.

`stress_boxplot.png` contradicts the simple stress-level hypothesis. The stress distribution has a lower median RI than the non-stress distribution, though stress periods include several high positive outliers.

`ri_vs_forward_realized_volatility_21d.png` shows a shallow positive fitted line with wide dispersion. The relationship is visually weak.

`regime_class_volatility.png` supports the risk-overlay interpretation: high-RI regimes have the highest volatility, followed by low-RI regimes, then normal regimes.

## 10. Trading and Risk-Management Interpretation

The RI should not be used as a standalone trading signal today.

What the current evidence supports:

- RI captures meaningful variation in rolling graph topology.
- RI spikes around some abrupt stress onsets.
- High-RI regimes are associated with higher realized volatility and larger absolute returns.
- Higher RI has weak positive association with future realized volatility.
- Higher RI has weak association with worse short-horizon future drawdowns.

What the current evidence does not support:

- RI is not validated as a persistent systemic stress-level indicator.
- RI does not consistently rise during benchmark stress labels.
- RI does not co-move positively with rolling average correlation or average absolute correlation.
- Predictive R-squared values are too small for a standalone forecasting claim.
- There is no transaction-cost-aware trading rule, out-of-sample test, or robustness grid.

The most appropriate current use case is exploratory risk overlay research. RI may be useful as one input in a broader monitoring system, especially if future work confirms that it detects topology transitions before or during stress events.

## 11. Main Limitations

The Gaussian graphical model assumption is strong. ETF returns are heavy-tailed, heteroskedastic, and affected by volatility clustering. Precision-matrix sparsity should not be overinterpreted as true conditional independence.

Volatility contamination remains possible. Window standardization helps, but crisis windows mix volatility shocks, tail events, changing correlations, liquidity stress, and sector rotations.

The fixed graphical-lasso alpha is methodologically clean for rolling comparability, but results may be sensitive to alpha. The current run uses `alpha = 0.10`; future phases should test multiple fixed alpha values.

There are 28 non-converged windows. The rate is low, but the failures are concentrated more in systemic stress periods, so they must be flagged and included in model-risk reporting.

The ETF universe mixes equities, sectors, credit, bonds, commodities, and defensive assets. That breadth is useful, but it can make raw and conditional dependence behave differently from an equity-only stress indicator.

Benchmark labels are imperfect. VIX captures option-implied equity volatility, drawdown captures path-dependent realized loss, realized volatility is backward-looking, and average correlation is unconditional rather than conditional. The combined stress label can remain active after the topology transition has already occurred.

The current Phase 5 results are sample-dependent. Phase 6 now adds the
machinery for robustness grids, validation/test discipline, incremental
information tests, and transaction-cost sensitivity, but those outputs still
need to be interpreted as research diagnostics rather than validation. There is
still no alternative-universe analysis, no full walk-forward production study,
and no transaction-cost-aware trading conclusion.

The current indicator uses absolute partial correlations and a combinatorial Laplacian. Signed relationships, normalized Laplacians, alternative spectral features, and feature weighting choices remain open research questions.

## 12. Recommended Next Quantitative Investigations

1. Run and interpret the full Phase 6 24-configuration robustness grid across
   alpha, window, and partial-correlation threshold settings.
2. Compare RI overlays against VIX, realized-volatility, drawdown,
   average-correlation, average-absolute-correlation, and PCA first-eigenvalue
   baselines across Sharpe, Calmar, volatility, drawdown, turnover, and
   downside-tail metrics.
3. Inspect turnover, average exposure, time in reduced-exposure states, and
   transaction-cost sensitivity at 0, 5, and 10 basis points.
4. Use the Phase 6 validation/test split to check whether any selected
   configuration survives out of sample.
5. Evaluate exclusion of the 28 non-converged graphical-lasso windows across
   all overlay and incremental-information diagnostics.
6. Test whether stress-onset labels are more relevant than persistent stress
   labels.
7. Test whether RI adds incremental information after controlling for VIX,
   realized volatility, drawdown, average correlation, and average absolute
   correlation.
8. Test alternative ETF universes, including equity-only, sector-only,
   cross-asset, and reduced-collinearity universes.
9. Add richer cost and slippage assumptions before drawing any economic
   conclusion from overlay results.
10. Only after robustness and OOS evidence is stable, consider whether RI is
    useful as one component in a broader risk-monitoring system.

## 13. Research Interpretation and Next Steps

The current empirical run shows that the project has reached a useful research checkpoint. The implementation is methodologically coherent, and the updated workflow now records convergence diagnostics. Graphical lasso converged in `2,733` of `2,761` rolling windows, for a convergence rate of `98.99%`. That rate is high enough to support exploratory interpretation, but the remaining `28` non-converged windows should be flagged in all empirical work, especially because many overlap with benchmark stress periods.

The original persistent stress-level hypothesis is not validated by the current outputs. RI spikes around some stress onsets, especially early COVID, but it is not higher on average during the combined systemic stress label. It is also negatively related to average correlation and average absolute correlation, which contradicts a simple interpretation that the graph-based indicator is just a higher-is-more-stress systemic connectedness proxy.

The evidence is more consistent with a refined hypothesis: RI may capture topology transitions or instability in the conditional-dependence network. The large spikes around abrupt stress onsets suggest that the indicator may respond to network reconfiguration rather than the sustained level of conventional stress variables. This would explain why RI can be high at the beginning of a shock but low during later periods when VIX, realized volatility, or rolling correlations remain elevated.

Predictive evidence is weak. The indicator has small positive relationships with future realized volatility and some association with worse short-horizon drawdowns, but the R-squared values are too small to support a standalone forecasting or trading claim. High-RI regimes are associated with higher volatility and larger absolute returns, making RI more plausible as a research feature for risk overlays than as a direct directional signal.

The current indicator should not be interpreted as a validated crisis predictor or standalone trading signal. Its most promising role is as a topology-transition feature that may complement conventional risk indicators. The next research decision is whether the Phase 5 risk-overlay improvement survives robustness checks, baseline comparisons, non-converged-window exclusions, stress-onset event studies, transaction-cost assumptions, and out-of-sample tests.

## Phase 5 Risk-Overlay Results

Phase 5 tested a simple research-only exposure-scaling overlay. The overlay
reduces equal-weight ETF portfolio exposure when RI is high, using expanding
quantile thresholds and a one-period exposure shift to reduce look-ahead bias.
This is not a live trading system and should not be interpreted as validation
or proof of profitability.

Baseline versus RI overlay metrics:

| Metric | Baseline | RI overlay | Change |
| --- | ---: | ---: | ---: |
| Annualized return | 0.0774 | 0.0805 | 0.0031 |
| Annualized volatility | 0.1354 | 0.1154 | -0.0200 |
| Sharpe | 0.5720 | 0.6976 | 0.1257 |
| Sortino | 0.6674 | 0.8785 | 0.2111 |
| Max drawdown | -0.3308 | -0.2034 | 0.1275 |
| Calmar | 0.2340 | 0.3958 | 0.1617 |
| Mean absolute return | 0.0056 | 0.0051 | -0.0006 |
| Worst 5% return | -0.0123 | -0.0114 | 0.0009 |

In this first run, the RI overlay improved Sharpe, max drawdown, volatility,
and Calmar versus the baseline equal-weight portfolio. The cumulative-return
figure shows a similar long-run wealth path with materially smoother behavior
during the COVID drawdown. The drawdown figure shows the most visible benefit:
the RI overlay reduced the depth of the largest drawdown.

Comparison with simple benchmark overlays:

| Strategy | Volatility | Sharpe | Max drawdown | Calmar |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.1354 | 0.5720 | -0.3308 | 0.2340 |
| RI overlay | 0.1154 | 0.6976 | -0.2034 | 0.3958 |
| VIX overlay | 0.1000 | 0.6448 | -0.1905 | 0.3383 |
| Realized-volatility overlay | 0.1040 | 0.6794 | -0.2000 | 0.3534 |
| Average-correlation overlay | 0.1086 | 0.5471 | -0.2527 | 0.2350 |
| Average-absolute-correlation overlay | 0.1104 | 0.6266 | -0.2006 | 0.3449 |
| Drawdown overlay | 0.0997 | 0.6836 | -0.1941 | 0.3513 |

The RI overlay had the best Sharpe and Calmar among the tested overlays in
this run. However, VIX, drawdown, and realized-volatility overlays produced
lower volatility and slightly better max drawdown. This is encouraging for the
refined RI risk-overlay hypothesis, but it also shows that simpler benchmark
overlays remain highly competitive.

The RI exposure summary shows that exposure was reduced on about `21.8%` of RI
observations and cut to `0.5` or lower on about `12.2%` of observations. This
means the overlay is not always defensive; it is active in selected high-RI
states. Turnover and transaction costs have not yet been evaluated.

The non-converged-window exclusion diagnostic is reassuring but still
preliminary. Using all windows, the convergence rate is `98.99%`; using only
converged windows leaves the main contemporaneous relationships qualitatively
similar. For example, the stress/non-stress RI difference is `-0.7330` for all
windows and `-0.7176` for converged windows only. Future robustness checks
should still exclude non-converged windows in all overlay tests.

### Research Interpretation of Phase 5

The first Phase 5 result is encouraging for the refined risk-overlay
hypothesis. The RI overlay improved several risk-adjusted metrics relative to
the baseline equal-weight portfolio. This is more consistent with RI being
useful for dynamic exposure management than for direct directional forecasting.

However, the result remains exploratory because it is one sample, one universe,
one parameterization, and one overlay rule. It still requires out-of-sample
validation, transaction-cost assumptions, turnover analysis, and robustness
across parameter grids. The result should be treated as suggestive evidence
that RI may complement conventional risk indicators, not as proof that RI is a
tradable signal.

### What Phase 5 Does Not Prove

Phase 5 does not prove profitability. It does not validate RI as a standalone
trading signal. It does not prove that graph topology causes risk reduction. It
does not show robustness across all reasonable graph-lasso, Laplacian,
threshold, universe, or overlay parameter choices. It also does not replace
simpler overlays unless robustness tests confirm incremental value beyond VIX,
realized volatility, drawdown, average correlation, average absolute
correlation, and PCA-style correlation-spectrum baselines.

## Phase 6 Robustness and Out-of-Sample Evaluation Framework

Phase 6 implements the next research layer needed to test the refined
risk-overlay hypothesis. The goal is no longer to show that RI is simply high
during persistent stress labels. The sharper question is whether RI provides
robust incremental risk-management information beyond simple VIX, volatility,
drawdown, and correlation overlays.

The Phase 6 framework adds a controlled robustness grid over:

- graphical-lasso alpha values: `0.05`, `0.10`, `0.15`, `0.20`;
- rolling windows: `63`, `126`, `252` trading days;
- partial-correlation thresholds: `0.00`, `0.03`.

The grid keeps the Laplacian definition fixed as combinatorial and the network
definition fixed as absolute partial correlations. That restraint is
important: Phase 6 is designed to test stability across reasonable parameter
choices, not to optimize every modeling decision until the in-sample result
looks good.

For each configuration, the framework can recompute graph features and RI,
record graphical-lasso convergence rates, compute contemporaneous and
transition diagnostics, evaluate forward-risk relationships, compare RI
overlay performance with simple benchmark overlays, and summarize the result
in dedicated Phase 6 output tables.

The out-of-sample protocol separates the sample into:

- development: 2015-07-06 to 2019-12-31;
- validation: 2020-01-01 to 2022-12-31;
- test: 2023-01-01 onward.

Configuration selection is based on validation-period diagnostics only. The
selection score combines Sharpe, Calmar, drawdown reduction, volatility
reduction, downside-tail improvement, and a turnover penalty. Test-period
results are then reported only for the frozen selected configuration. This
does not eliminate all research degrees of freedom, but it is a substantial
improvement over judging the overlay only on full-sample performance.

Phase 6 also adds incremental-information tests. These ask whether RI and its
graph components add explanatory information beyond VIX, realized volatility,
drawdown, average correlation, and average absolute correlation for future
realized volatility, future max drawdown, and stress-onset labels. The tests
report adjusted R-squared, out-of-sample R-squared where feasible, coefficient
signs, Newey-West-style t-statistics, and classification diagnostics for
stress-onset labels.

Finally, Phase 6 adds turnover and transaction-cost sensitivity. The overlay
is evaluated at 0, 5, and 10 basis points per unit exposure change, alongside
average exposure, time in reduced exposure, number of exposure changes, and an
annualized turnover proxy. This is still a simplified cost model, but it
prevents the research from ignoring one of the most obvious ways an overlay
can look attractive on paper and fail economically.

Phase 6 should be interpreted as a robustness and falsification framework. If
the RI overlay only works for one alpha, one window, one threshold, or one
sample split, the refined risk-overlay hypothesis is weak. If the result is
stable across the grid, survives validation/test separation, adds information
beyond simple benchmarks, and remains plausible after costs and turnover, then
the project would have stronger evidence that RI is useful as a research
feature for risk management. Even then, it would still not prove causality or
justify a standalone trading strategy.

### Initial Phase 6 Smoke Output

The default Phase 6 example uses saved Phase 4 outputs as a fast smoke test.
It is not the full 24-configuration robustness grid. The cached configuration
is `alpha = 0.10`, `window = 126`, and partial-correlation threshold
`0.000001`.

In the test sample beginning in 2023, the cached RI overlay improved the
baseline equal-weight portfolio on the main risk-adjusted metrics:

| Metric | Baseline test | RI overlay test | Change |
| --- | ---: | ---: | ---: |
| Sharpe | 1.5784 | 1.8374 | 0.2589 |
| Max drawdown | -0.1322 | -0.0969 | 0.0353 |
| Annualized volatility | 0.1089 | 0.0938 | -0.0151 |
| Calmar | 1.3003 | 1.7788 | 0.4784 |

This is encouraging, but it should be treated as a smoke result rather than a
robust finding. Only one cached configuration is present in the default run, so
there is no genuine parameter-grid selection in that output.

Against simple benchmark overlays in the same test sample, the RI overlay had
the highest Sharpe and Calmar in the cached smoke run. VIX and RI overlays had
almost identical max drawdowns, while several simple overlays remained
competitive. This means the RI result is promising but must still be compared
systematically against simpler risk variables across the full grid.

Transaction-cost sensitivity materially weakens the full-sample RI overlay:

| Cost assumption | Sharpe | Calmar |
| --- | ---: | ---: |
| 0 bps | 0.6976 | 0.3958 |
| 5 bps | 0.6335 | 0.3584 |
| 10 bps | 0.5698 | 0.3216 |

At 10 bps, the full-sample Sharpe is roughly back to the baseline level. This
does not invalidate the overlay, but it shows that turnover and implementation
costs are central to the research question.

The incremental-information smoke tests are mixed. Adding RI modestly improves
in-sample adjusted R-squared for forward realized volatility and forward max
drawdown, and graph components improve short-horizon realized-volatility OOS
R-squared in this run. However, some longer-horizon OOS R-squared values
deteriorate when graph components are added. Stress-onset AUC improves from
about `0.792` for the baseline model to about `0.835` with graph components
in-sample, but OOS AUC remains close to `0.798`. The correct interpretation is
that RI and graph features may contain incremental information, but the current
evidence is not uniformly strong and requires full-grid and out-of-sample
confirmation.

## Phase 7 PCA Baselines and Component-Level Results

Phase 7 extends the research question from "does the composite RI work?" to
"which graph-Laplacian components, if any, contain useful incremental
information?" This is an important shift. A composite indicator can be useful
for monitoring, but it can also blend together features with different
economic meanings and different predictive value.

The Phase 7 workflow adds rolling PCA/correlation-spectrum baselines, graph
component block scores, benchmark-orthogonalized graph components, a model
ladder, component overlays, and graph-component ablations. These outputs are
saved under `outputs/phase7_tables/` and `outputs/phase7_figures/`.

### Model Ladder Findings

The model ladder compares:

- `M0`: benchmarks only;
- `M1`: benchmarks plus composite RI;
- `M2`: benchmarks plus PCA/correlation-spectrum features;
- `M3`: benchmarks plus graph components;
- `M4`: benchmarks plus PCA features and graph components;
- `M5`: benchmarks plus orthogonalized graph components.

The strongest Phase 7 evidence is that graph components and PCA features are
more informative than the composite RI alone in several diagnostics.

For forward realized volatility:

| Target | M0 OOS R2 | M1 RI OOS R2 | M2 PCA OOS R2 | M3 graph OOS R2 | M4 PCA+graph OOS R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5d realized volatility | 0.1759 | 0.1810 | 0.1745 | 0.2140 | 0.2483 |
| 21d realized volatility | 0.0384 | 0.0303 | 0.0617 | -0.0089 | 0.1310 |

For realized-volatility targets, the combined PCA-plus-graph model performs
best in this run. The composite RI alone adds only modest incremental
information, while graph components and PCA features together appear more
powerful.

For forward max drawdown:

| Target | M0 OOS R2 | M1 RI OOS R2 | M2 PCA OOS R2 | M3 graph OOS R2 | M4 PCA+graph OOS R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5d max drawdown | 0.0205 | 0.0248 | -0.0273 | -0.0106 | -0.0355 |
| 21d max drawdown | -0.0294 | -0.0133 | -0.0783 | -0.1414 | 0.0038 |

The drawdown results are weaker and less stable. RI helps slightly relative to
the benchmark-only model for short-horizon drawdown, but PCA and graph
components do not produce uniformly better OOS drawdown prediction.

For stress-onset classification:

| Model | In-sample AUC | OOS AUC | Brier score | OOS Brier |
| --- | ---: | ---: | ---: | ---: |
| M0 benchmarks | 0.7922 | 0.7962 | 0.00564 | 0.00670 |
| M1 benchmarks + RI | 0.7918 | 0.7982 | 0.00564 | 0.00670 |
| M2 benchmarks + PCA | 0.8814 | 0.8712 | 0.00526 | 0.00571 |
| M3 benchmarks + graph components | 0.8354 | 0.7983 | 0.00562 | 0.00669 |
| M4 benchmarks + PCA + graph components | 0.9034 | 0.8890 | 0.00524 | 0.00573 |
| M5 benchmarks + orthogonal graph | 0.8566 | 0.8347 | 0.00517 | 0.00670 |

This is the clearest Phase 7 result: PCA features materially improve
stress-onset classification, and PCA plus graph components gives the best OOS
AUC in this run. Composite RI alone does not materially improve stress-onset
AUC beyond benchmarks.

### Component Overlay Findings

Component-based overlays also improve on the original RI overlay in this run.
The baseline row below is evaluated on the same dates as the component overlay
signals, so it is directly comparable with the overlay rows.

| Overlay signal | Sharpe | Calmar | Max drawdown | Turnover proxy |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.5970 | 0.2471 | -0.3308 | n/a |
| Composite RI | 0.6976 | 0.3958 | -0.2034 | 13.70 |
| Graph components equal weight | 0.7375 | 0.4030 | -0.2121 | 9.22 |
| Transition score | 0.7223 | 0.4160 | -0.1999 | 25.38 |
| Laplacian Frobenius change | 0.7202 | 0.4148 | -0.1999 | 25.33 |
| Connectivity score | 0.7122 | 0.3862 | -0.2205 | 4.47 |
| PCA first eigenvalue change | 0.7177 | 0.3344 | -0.2590 | 29.18 |
| PCA first eigenvalue share | 0.6693 | 0.3580 | -0.2085 | 1.52 |

The best overlay by Sharpe is the equal-weight graph-component score. The best
Calmar among these signals is the transition score, which is essentially
driven by Laplacian Frobenius topology change. The PCA first-eigenvalue change
is competitive on Sharpe but has high turnover and weaker drawdown/Calmar.

Transaction costs change the interpretation. For the equal-weight graph
component score, Sharpe declines from `0.7375` at 0 bps to `0.6944` at 5 bps
and `0.6514` at 10 bps. For the transition score, Sharpe declines from
`0.7223` to `0.6038` and then `0.4866`. The transition signal is useful but
turnover-intensive, so its economic interpretation is much more cost-sensitive.

### Ablation Findings

The graph-component ablation table suggests that no single interpretation is
fully dominant.

| Ablation | OOS R2 | OOS AUC | Overlay Sharpe | Overlay Calmar |
| --- | ---: | ---: | ---: | ---: |
| All graph components | -0.0089 | 0.7983 | 0.7375 | 0.4030 |
| Excluding Frobenius change | -0.0182 | 0.8053 | 0.7063 | 0.3371 |
| Excluding algebraic connectivity | 0.0135 | 0.8080 | 0.7919 | 0.4358 |
| Excluding average graph strength | -0.0089 | 0.7983 | 0.7454 | 0.4165 |
| Excluding largest eigenvalue share | -0.0156 | 0.8072 | 0.7363 | 0.4027 |
| Connectivity only | 0.0398 | 0.8116 | 0.7122 | 0.3862 |
| Transition only | 0.0481 | 0.8003 | 0.7223 | 0.4160 |
| Spectral only | -0.0144 | 0.8133 | 0.5483 | 0.2147 |

The best ablation by overlay Sharpe is "excluding algebraic connectivity,"
which is a useful warning against assuming every component in the composite RI
is helpful for overlay design. Connectivity-only and transition-only have the
best OOS R2 among the ablation rows, while spectral-only has the best OOS AUC
but poor overlay performance. This supports a component-specific research
interpretation rather than a single universal graph score.

### Phase 7 Research Interpretation

Phase 7 strengthens the topology-transition/risk-overlay hypothesis, but in a
more nuanced way than the original RI hypothesis. The composite RI is not the
end of the research object. Individual graph components, graph-component
blocks, and PCA spectral baselines contain information that RI alone can
dilute.

The most encouraging result is that PCA plus graph components improves
stress-onset OOS AUC and forward-realized-volatility OOS R2. The most useful
overlay result is that graph-component scores can outperform the composite RI
on Sharpe and Calmar, although the best signals differ depending on whether
turnover and transaction costs are emphasized.

The most important caution is that PCA baselines are strong. If PCA features
continue to explain stress onsets better than graph components in broader
tests, then the project must show that graph-lasso Laplacian features add
incremental value beyond simpler correlation-spectrum information. The Phase 7
result is therefore encouraging for component research, but it also raises the
standard of evidence for the graph-lasso layer.

## Phase 8 Research Consolidation

Phase 8 consolidates the post-Phase-7 research direction. The question is no
longer whether the original composite RI should be accepted as-is. The more
precise question is which transparent information set or overlay object is the
best candidate for continued research: RI, graph components, transition
features, PCA features, PCA plus graph features, or a lower-turnover variant.

### PCA-Only vs Graph-Only vs PCA-Plus-Graph

The corrected Phase 8 information-set comparison supports the Phase 7 finding
that PCA and graph components are more useful together than the composite RI
alone for several targets.

| Target | Best OOS information set | OOS value |
| --- | --- | ---: |
| 5d future realized volatility R2 | benchmarks + PCA + graph components | 0.2483 |
| 21d future realized volatility R2 | benchmarks + PCA + graph components | 0.1310 |
| 5d future max drawdown R2 | graph components only | 0.0349 |
| 21d future max drawdown R2 | RI only | 0.0084 |
| Stress-onset OOS AUC | benchmarks + PCA + graph components | 0.8890 |

The strongest evidence is for stress-onset classification and future realized
volatility. Drawdown prediction remains weak and unstable. The fact that
PCA-plus-graph is best for stress-onset OOS AUC and realized-volatility OOS R2
is suggestive, but it also reinforces that PCA is a strong benchmark. Graph
features need to justify themselves as incremental information beyond the
correlation spectrum, not merely as a more complex version of it.

### Algebraic Connectivity Diagnostic

Phase 8 investigated why excluding algebraic connectivity improved overlay
Sharpe in Phase 7. The evidence suggests algebraic connectivity is not useless,
but it is not a strong standalone overlay feature in this configuration.

Algebraic connectivity is strongly correlated with RI (`0.711`) and moderately
correlated with average graph strength and weighted edge density (`0.594` each).
It is only weakly correlated with VIX (`0.096`), drawdown (`0.076`), average
absolute correlation (`0.042`), realized volatility (`0.006`), and average
correlation (`-0.008`). Its stress-onset AUC is only `0.549`, close to weak
classification value.

Overlay diagnostics are consistent with the ablation result:

| Overlay | Sharpe | Calmar | Max drawdown | Turnover |
| --- | ---: | ---: | ---: | ---: |
| All graph components | 0.7375 | 0.4030 | -0.2121 | 9.15 |
| Excluding algebraic connectivity | 0.7919 | 0.4358 | -0.2121 | 10.19 |
| Algebraic connectivity only | 0.6573 | 0.3443 | -0.2257 | 3.92 |
| Connectivity block with algebraic connectivity | 0.7122 | 0.3862 | -0.2205 | 4.44 |
| Connectivity block without algebraic connectivity | 0.6960 | 0.3382 | -0.2509 | 3.05 |

The most plausible interpretation is that algebraic connectivity is partly
redundant with broader graph-strength information and may dilute overlay timing
when included in an equal-weight component score. It may still contain useful
topological information, but Phase 8 does not support treating it as a
standalone overlay driver.

### Turnover-Reduction Results

Turnover remains a central practical constraint. Phase 8 tested weekly
rebalancing, smoothing, hysteresis, cooldown, and combined smoothing plus
hysteresis variants.

The best low-turnover screen by Calmar was `pca_first_eigenvalue_change` with
the `smoothed_5d_hysteresis` rule:

| Score / variant | Sharpe | Calmar | Max drawdown | Turnover |
| --- | ---: | ---: | ---: | ---: |
| PCA first eigenvalue change / smoothed 5d hysteresis | 0.8090 | 0.4610 | -0.1978 | 8.04 |
| PCA first eigenvalue change / smoothed 5d | 0.8506 | 0.4530 | -0.2172 | 10.12 |
| RI / cooldown 5d | 0.7126 | 0.4406 | -0.1827 | 5.35 |
| Graph components equal weight / hysteresis | 0.7604 | 0.4227 | -0.1994 | 4.82 |
| PCA plus graph / smoothed 5d hysteresis | 0.7565 | 0.4162 | -0.1958 | 4.11 |

This is important because the high-turnover transition score was attractive in
earlier phases but costly. Phase 8 shows that smoother or hysteresis-based
rules can preserve much of the risk-overlay benefit with materially lower
turnover. The result is encouraging for a risk-overlay research direction, not
for a live trading claim.

### Rolling-Origin OOS Results

The yearly rolling-origin OOS evaluation is stricter than the single-sample
overlay diagnostics. Each test year uses only the preceding selection period
to set thresholds.

The leading rolling-origin candidates at 0 bps were:

| Candidate | Mean Sharpe | Mean Calmar | Mean max drawdown | Turnover | Positive-Sharpe years |
| --- | ---: | ---: | ---: | ---: | ---: |
| Excluding algebraic connectivity | 1.0083 | 1.6798 | -0.1088 | 12.74 | 0.83 |
| Graph components equal weight | 0.9439 | 1.4832 | -0.1082 | 9.70 | 0.83 |
| Transition score | 0.9058 | 1.5613 | -0.1096 | 24.56 | 0.83 |
| RI | 0.8965 | 1.6179 | -0.1043 | 14.22 | 0.83 |
| Best low-turnover score | 0.8786 | 1.3211 | -0.1137 | 8.37 | 0.83 |
| PCA plus graph score | 0.8530 | 1.4518 | -0.1034 | 24.70 | 0.83 |
| PCA score | 0.8047 | 1.2661 | -0.1166 | 13.19 | 0.67 |

The year-by-year table shows a shared weakness: 2022 is negative for all
candidates. This means the overlay evidence is not uniformly strong across
every regime. The results are not simply a COVID-only artifact, because several
candidates perform well in 2021, 2023, 2024, and 2025, but 2022 is a clear
stress test that prevents overclaiming.

At 10 bps, the excluding-algebraic-connectivity score still has the higher
average Sharpe (`0.8921`) than the dynamically selected low-turnover score
(`0.8114`). The low-turnover candidate does reduce turnover, but the corrected
rolling-origin result no longer supports treating it as the leading
cost-adjusted object.

### Final Object Selection

The Phase 8 selection matrix ranks `excluding_algebraic_connectivity_score`
first, followed by the equal-weight graph component score and RI. The dynamic
low-turnover candidate is useful as a turnover-control reference but is not the
top-ranked candidate after the rolling-origin correction:

| Candidate | OOS Sharpe | OOS Calmar | OOS max drawdown | Turnover | 10 bps Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| Excluding algebraic connectivity | 1.0083 | 1.6798 | -0.1088 | 12.74 | 0.8921 |
| Graph components equal weight | 0.9439 | 1.4832 | -0.1082 | 9.70 | 0.8548 |
| RI | 0.8965 | 1.6179 | -0.1043 | 14.22 | 0.7564 |
| Best low-turnover score | 0.8786 | 1.3211 | -0.1137 | 8.37 | 0.8114 |
| Transition score | 0.9058 | 1.5613 | -0.1096 | 24.56 | 0.6719 |
| PCA plus graph score | 0.8530 | 1.4518 | -0.1034 | 24.70 | 0.6258 |
| PCA score | 0.8047 | 1.2661 | -0.1166 | 13.19 | 0.6951 |

The recommended research interpretation is:

- keep the original RI as an interpretable baseline, not the final object;
- use PCA-plus-graph models for stress-onset classification and volatility
  prediction research;
- use the excluding-algebraic-connectivity graph score as the leading
  no-cost and cost-adjusted overlay candidate in this run;
- keep the best low-turnover variant as a secondary candidate because it
  reduces turnover, but do not treat it as the leading cost-adjusted object;
- continue treating PCA as a strong benchmark that graph features must beat or
  complement.

### What Phase 8 Can And Cannot Claim

Phase 8 strengthens the topology-transition/risk-overlay hypothesis. The
evidence is now more consistent with component-level graph and PCA information
being useful for risk management than with the original composite RI being a
validated stress-level indicator.

It still does not validate a trading strategy. It does not prove profitability,
causality, or universal robustness. The results remain conditional on this ETF
universe, this sample, the saved graph-lasso configuration, the current
transaction-cost proxy, and the current candidate-score definitions. The next
research step should test alternative universes, stricter walk-forward
normalization of component scores, more realistic transaction-cost and turnover
assumptions, and whether the Phase 8 leaders retain value after controlling
directly for PCA features and conventional risk benchmarks.
