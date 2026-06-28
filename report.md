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

The current results are sample-dependent. There is no robustness grid, no alternative universe analysis, no out-of-sample validation, no walk-forward parameter discipline, and no economic trading evaluation.

The current indicator uses absolute partial correlations and a combinatorial Laplacian. Signed relationships, normalized Laplacians, alternative spectral features, and feature weighting choices remain open research questions.

## 12. Recommended Next Quantitative Investigations

1. Rerun all diagnostics excluding the 28 non-converged windows.
2. Compare results across fixed alpha values such as `0.05`, `0.10`, `0.15`, and `0.20`.
3. Compare rolling windows such as 63, 126, and 252 trading days.
4. Test whether RI is better at detecting stress onsets than persistent stress levels.
5. Create event windows around known market shocks and inspect RI before, during, and after the event.
6. Compare against simple baselines: VIX, realized volatility, average correlation, average absolute correlation, and rolling drawdown.
7. Split the sample into development and holdout periods.
8. Test equity-only, cross-asset, sector-only, and reduced-collinearity universes.
9. Evaluate whether excluding or down-weighting Frobenius change changes the stress-label results.
10. Investigate whether normalized Laplacian features behave differently from combinatorial Laplacian features.
11. Add out-of-sample risk-overlay tests before considering any trading interpretation.

## 13. Research Interpretation and Next Steps

The current empirical run shows that the project has reached a useful research checkpoint. The implementation is methodologically coherent, and the updated workflow now records convergence diagnostics. Graphical lasso converged in `2,733` of `2,761` rolling windows, for a convergence rate of `98.99%`. That rate is high enough to support exploratory interpretation, but the remaining `28` non-converged windows should be flagged in all empirical work, especially because many overlap with benchmark stress periods.

The original persistent stress-level hypothesis is not validated by the current outputs. RI spikes around some stress onsets, especially early COVID, but it is not higher on average during the combined systemic stress label. It is also negatively related to average correlation and average absolute correlation, which contradicts a simple interpretation that the graph-based indicator is just a higher-is-more-stress systemic connectedness proxy.

The evidence is more consistent with a refined hypothesis: RI may capture topology transitions or instability in the conditional-dependence network. The large spikes around abrupt stress onsets suggest that the indicator may respond to network reconfiguration rather than the sustained level of conventional stress variables. This would explain why RI can be high at the beginning of a shock but low during later periods when VIX, realized volatility, or rolling correlations remain elevated.

Predictive evidence is weak. The indicator has small positive relationships with future realized volatility and some association with worse short-horizon drawdowns, but the R-squared values are too small to support a standalone forecasting or trading claim. The strongest practical interpretation is that high-RI regimes are associated with higher volatility and larger absolute returns, making RI a candidate research feature for risk overlays rather than a direct trading signal.

The current indicator should not be interpreted as a validated crisis predictor or standalone trading signal. Its most promising role is as a topology-transition feature that may complement conventional risk indicators. The next phase should test this refined hypothesis through robustness checks, baseline comparisons, exclusion of non-converged windows, event studies around stress onsets, and out-of-sample risk-overlay experiments.

Phase 5 should focus on quantitative robustness and incremental-information tests:

1. Exclude the 28 non-converged windows and rerun all key diagnostics.
2. Test a fixed-alpha grid: `0.05`, `0.10`, `0.15`, and `0.20`.
3. Test a rolling-window grid: 63, 126, and 252 trading days.
4. Compare combinatorial Laplacian features with normalized Laplacian features.
5. Compare absolute partial-correlation networks with signed partial-correlation network variants.
6. Compare RI against simple baselines: VIX, realized volatility, drawdown, average correlation, average absolute correlation, and the first PCA eigenvalue of the correlation matrix.
7. Test stress-onset labels separately from persistent stress labels.
8. Run event studies around known market shocks.
9. Test whether RI adds incremental information in regressions controlling for VIX, realized volatility, drawdown, and average correlation.
10. Test risk-overlay rules out of sample before considering any trading application.

The next research decision is therefore not whether the current RI is "validated"; it is not. The next decision is whether the topology-transition interpretation survives robustness checks and adds information beyond simpler stress proxies.

## Phase 5 Research Plan

Phase 5 implements the next research step implied by the updated empirical
interpretation: test whether RI is robust across reasonable graph settings,
whether it behaves more like a topology-transition feature than a persistent
stress-level indicator, and whether it has any incremental risk-overlay value
relative to simple benchmarks.

The planned diagnostics are intentionally research-only. They should not be
read as a trading system or validation claim. The Phase 5 workflow should
compare an RI-based exposure overlay against a baseline equal-weight portfolio
and against simple benchmark overlays based on VIX, realized volatility,
drawdown, average correlation, and average absolute correlation. Any apparent
improvement must then be checked out of sample before it can be interpreted as
economically meaningful.

The most important Phase 5 tests are:

1. Exclude the 28 non-converged graphical-lasso windows and rerun diagnostics.
2. Test fixed alpha values of `0.05`, `0.10`, `0.15`, and `0.20`.
3. Test rolling windows of 63, 126, and 252 trading days.
4. Compare combinatorial and normalized Laplacian variants.
5. Compare absolute and signed partial-correlation network variants.
6. Compare RI against VIX, realized volatility, drawdown, average correlation,
   average absolute correlation, and PCA first-eigenvalue baselines.
7. Test stress-onset labels separately from persistent stress labels.
8. Run event studies around known market shocks.
9. Test whether RI adds information after controlling for conventional stress
   variables.
10. Run out-of-sample risk-overlay experiments before considering any trading
    application.
