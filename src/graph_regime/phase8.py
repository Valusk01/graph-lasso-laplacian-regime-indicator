"""Phase 8 research consolidation utilities.

This module compares PCA, graph, and combined information sets; diagnoses the
algebraic-connectivity ablation; evaluates lower-turnover overlay variants; and
builds rolling-origin OOS and final candidate-selection tables. The utilities
are research-only and intentionally use transparent linear models and quantile
overlay rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from graph_regime.component_scores import GRAPH_COMPONENT_COLUMNS, add_component_scores
from graph_regime.evaluation import compute_forward_targets
from graph_regime.phase6 import (
    DEFAULT_SAMPLE_SPLITS,
    SampleSplits,
    apply_transaction_costs,
    assign_sample_period,
    compute_exposure_turnover,
)
from graph_regime.phase7 import (
    BASELINE_COLUMNS,
    _auc,
    _classification_metrics,
    _coefficient_sign,
    _fit_ols,
    _oos_metrics,
)
from graph_regime.risk_overlay import (
    apply_risk_overlay,
    compute_performance_metrics,
    compute_portfolio_returns,
)
from graph_regime.robustness import create_stress_onset_labels

ExposureVariant = Literal[
    "daily",
    "weekly",
    "smoothed_5d",
    "smoothed_10d",
    "hysteresis",
    "cooldown_5d",
    "smoothed_5d_hysteresis",
]

REGRESSION_TARGETS = [
    "future_realized_volatility_5d",
    "future_realized_volatility_21d",
    "future_max_drawdown_5d",
    "future_max_drawdown_21d",
]
CLASSIFICATION_TARGET = "stress_onset_label"
INFORMATION_SETS = [
    "benchmarks_only",
    "ri_only",
    "pca_only",
    "graph_components_only",
    "pca_plus_graph_components",
    "benchmarks_plus_ri",
    "benchmarks_plus_pca",
    "benchmarks_plus_graph_components",
    "benchmarks_plus_pca_plus_graph_components",
]
DEFAULT_OVERLAY_VARIANTS: tuple[ExposureVariant, ...] = (
    "daily",
    "weekly",
    "smoothed_5d",
    "smoothed_10d",
    "hysteresis",
    "cooldown_5d",
    "smoothed_5d_hysteresis",
)
DEFAULT_CANDIDATES = [
    "ri",
    "graph_components_equal_weight_score",
    "transition_score",
    "pca_score",
    "pca_plus_graph_score",
    "excluding_algebraic_connectivity_score",
    "best_low_turnover_score",
]


@dataclass(frozen=True)
class RollingOriginSplit:
    """Selection and test dates for one rolling-origin OOS year."""

    test_year: int
    selection_start: pd.Timestamp
    selection_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def compare_information_sets(
    graph_features: pd.DataFrame,
    benchmarks: pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    pca_features: pd.DataFrame | None = None,
    component_scores: pd.DataFrame | None = None,
    splits: SampleSplits = DEFAULT_SAMPLE_SPLITS,
    horizons: tuple[int, ...] = (5, 21),
    newey_west_lags: int = 5,
) -> pd.DataFrame:
    """Compare PCA-only, graph-only, RI, benchmark, and combined models.

    In-sample metrics are diagnostics-only. OOS metrics are fit on the
    development plus validation samples and evaluated on the held-out test
    sample defined by ``splits``.
    """

    regressors = _build_phase8_regressors(
        graph_features=graph_features,
        benchmarks=benchmarks,
        pca_features=pca_features,
        component_scores=component_scores,
    )
    specs = _information_set_specs(regressors)
    targets = _build_phase8_targets(returns, benchmarks, horizons=horizons)
    rows: list[dict[str, object]] = []

    for target in _phase8_target_columns(horizons):
        if target not in targets.columns:
            continue
        y = pd.to_numeric(targets[target], errors="coerce").rename(target)
        classification = target == CLASSIFICATION_TARGET
        for information_set, columns in specs.items():
            if not columns:
                continue
            frame = (
                pd.concat([y, regressors[columns]], axis=1, join="inner")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if frame.empty:
                continue

            fit = _fit_ols(
                frame[target],
                frame[columns],
                newey_west_lags=newey_west_lags,
            )
            sample_labels = assign_sample_period(frame.index, splits)
            oos = _oos_metrics(
                frame[target],
                frame[columns],
                sample_labels=sample_labels,
                classification=classification,
                newey_west_lags=newey_west_lags,
            )
            cls = (
                _classification_metrics(frame[target], fit["fitted"])
                if classification
                else {}
            )
            coefficients = fit["coefficients"].drop(
                labels=["intercept"], errors="ignore"
            )
            signs = {
                name: _coefficient_sign(float(value))
                for name, value in coefficients.items()
            }
            rows.append(
                {
                    "target": target,
                    "target_type": "classification" if classification else "regression",
                    "information_set": information_set,
                    "n_obs": float(len(frame)),
                    "n_positive_events": (
                        float((frame[target] == 1).sum()) if classification else np.nan
                    ),
                    "n_regressors": float(len(columns)),
                    "regressors": ",".join(columns),
                    "adjusted_r2": fit["adjusted_r2"],
                    "oos_r2": oos["oos_r2"],
                    "rmse": fit["rmse"],
                    "mae": fit["mae"],
                    "oos_rmse": oos["oos_rmse"],
                    "oos_mae": oos["oos_mae"],
                    "auc": cls.get("auc", np.nan),
                    "oos_auc": oos["oos_auc"],
                    "brier_score": cls.get("brier_score", np.nan),
                    "oos_brier_score": oos["oos_brier_score"],
                    "precision": cls.get("precision", np.nan),
                    "recall": cls.get("recall", np.nan),
                    "f1": cls.get("f1", np.nan),
                    "oos_precision": oos["oos_precision"],
                    "oos_recall": oos["oos_recall"],
                    "oos_f1": oos["oos_f1"],
                    "coefficient_signs": ";".join(
                        f"{name}:{sign}" for name, sign in signs.items()
                    ),
                    "diagnostic_note": (
                        "in_sample_metrics_are_diagnostics_only"
                        if not classification
                        else "classification_scores_are_linear_probability_diagnostics"
                    ),
                },
            )

    return pd.DataFrame(rows)


def rank_information_sets(comparison: pd.DataFrame) -> pd.DataFrame:
    """Rank information sets by target and metric."""

    if not isinstance(comparison, pd.DataFrame):
        raise TypeError("comparison must be a pandas DataFrame.")
    required = {"target", "information_set", "target_type"}
    missing = required - set(comparison.columns)
    if missing:
        raise KeyError(f"comparison is missing columns: {sorted(missing)}")

    metric_specs = {
        "regression": [
            ("oos_r2", "max", False),
            ("adjusted_r2", "max", True),
            ("rmse", "min", True),
            ("mae", "min", True),
            ("oos_rmse", "min", False),
            ("oos_mae", "min", False),
        ],
        "classification": [
            ("oos_auc", "max", False),
            ("auc", "max", True),
            ("oos_brier_score", "min", False),
            ("brier_score", "min", True),
        ],
    }
    rows: list[dict[str, object]] = []
    for (target, target_type), group in comparison.groupby(
        ["target", "target_type"],
        sort=False,
    ):
        for metric, direction, diagnostics_only in metric_specs.get(target_type, []):
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            valid = group.loc[values.notna()].copy()
            if valid.empty:
                continue
            best_index = (
                pd.to_numeric(valid[metric], errors="coerce").idxmax()
                if direction == "max"
                else pd.to_numeric(valid[metric], errors="coerce").idxmin()
            )
            best = valid.loc[best_index]
            rows.append(
                {
                    "target": target,
                    "target_type": target_type,
                    "metric": metric,
                    "rank_direction": direction,
                    "best_information_set": best["information_set"],
                    "best_value": float(best[metric]),
                    "n_candidates": float(len(valid)),
                    "diagnostics_only": bool(diagnostics_only),
                },
            )
    return pd.DataFrame(rows)


def build_phase8_candidate_scores(
    graph_features: pd.DataFrame,
    pca_features: pd.DataFrame | None = None,
    component_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build transparent candidate final-object scores for Phase 8."""

    if component_scores is None:
        component_scores = add_component_scores(graph_features)
    scored = pd.concat([graph_features, component_scores], axis=1)
    scored = scored.loc[:, ~scored.columns.duplicated()]
    if pca_features is not None:
        scored = scored.join(pca_features, how="outer", rsuffix="_pca")

    output = pd.DataFrame(index=scored.index)
    for column in GRAPH_COMPONENT_COLUMNS:
        if column in scored:
            output[column] = _numeric_series(scored[column])
    if "regime_indicator" in scored:
        output["ri"] = _numeric_series(scored["regime_indicator"])

    graph_columns = [column for column in GRAPH_COMPONENT_COLUMNS if column in scored]
    if graph_columns:
        output["graph_components_equal_weight_score"] = _mean_z(scored[graph_columns])
        excluding_algebraic = [
            column for column in graph_columns if column != "algebraic_connectivity"
        ]
        if excluding_algebraic:
            output["excluding_algebraic_connectivity_score"] = _mean_z(
                scored[excluding_algebraic],
            )
    if "laplacian_frobenius_change" in scored:
        output["transition_score"] = _safe_z(scored["laplacian_frobenius_change"])
    if "algebraic_connectivity" in scored:
        output["algebraic_connectivity_only"] = _safe_z(
            scored["algebraic_connectivity"],
        )
    connectivity_with = [
        column
        for column in [
            "average_graph_strength",
            "algebraic_connectivity",
            "weighted_edge_density",
        ]
        if column in scored
    ]
    if connectivity_with:
        output["connectivity_block_with_algebraic_connectivity"] = _mean_z(
            scored[connectivity_with],
        )
    connectivity_without = [
        column
        for column in ["average_graph_strength", "weighted_edge_density"]
        if column in scored
    ]
    if connectivity_without:
        output["connectivity_block_without_algebraic_connectivity"] = _mean_z(
            scored[connectivity_without],
        )

    pca_columns = [
        column
        for column in ["pca_first_eigenvalue_share", "pca_first_eigenvalue_change"]
        if column in scored
    ]
    for column in pca_columns:
        output[column] = _numeric_series(scored[column])
    if pca_columns:
        output["pca_score"] = _mean_z(scored[pca_columns])
    combined_columns = [
        column
        for column in [
            "graph_components_equal_weight_score",
            "transition_score",
            "pca_score",
        ]
        if column in output
    ]
    if combined_columns:
        output["pca_plus_graph_score"] = _mean_z(output[combined_columns])

    return output.replace([np.inf, -np.inf], np.nan).sort_index()


def compute_algebraic_connectivity_diagnostics(
    graph_features: pd.DataFrame,
    benchmarks: pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    pca_features: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = (5, 21),
) -> pd.DataFrame:
    """Diagnose algebraic connectivity against components, benchmarks, targets."""

    if "algebraic_connectivity" not in graph_features.columns:
        return pd.DataFrame(
            [
                {
                    "diagnostic_type": "availability",
                    "variable": "algebraic_connectivity",
                    "metric": "available",
                    "value": 0.0,
                    "n_obs": 0.0,
                },
            ],
        )

    algebraic = _numeric_series(graph_features["algebraic_connectivity"]).rename(
        "algebraic_connectivity",
    )
    rows: list[dict[str, object]] = []
    valid = algebraic.dropna()
    rows.append(
        {
            "diagnostic_type": "availability",
            "variable": "algebraic_connectivity",
            "metric": "available",
            "value": 1.0,
            "n_obs": float(len(valid)),
        },
    )
    for metric, value in {
        "mean": valid.mean(),
        "std": valid.std(ddof=0),
        "min": valid.min(),
        "p25": valid.quantile(0.25),
        "median": valid.median(),
        "p75": valid.quantile(0.75),
        "max": valid.max(),
    }.items():
        rows.append(
            {
                "diagnostic_type": "summary",
                "variable": "algebraic_connectivity",
                "metric": metric,
                "value": float(value) if np.isfinite(value) else np.nan,
                "n_obs": float(len(valid)),
            },
        )

    comparison = _algebraic_comparison_frame(graph_features, benchmarks, pca_features)
    for column in comparison.columns:
        if column == "algebraic_connectivity":
            continue
        pair = pd.concat([algebraic, comparison[column]], axis=1).dropna()
        if pair.empty:
            continue
        rows.extend(
            [
                {
                    "diagnostic_type": "correlation",
                    "variable": column,
                    "metric": "pearson",
                    "value": float(pair.iloc[:, 0].corr(pair.iloc[:, 1])),
                    "n_obs": float(len(pair)),
                },
                {
                    "diagnostic_type": "correlation",
                    "variable": column,
                    "metric": "spearman",
                    "value": float(
                        pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
                    ),
                    "n_obs": float(len(pair)),
                },
            ],
        )

    targets = _build_phase8_targets(returns, benchmarks, horizons=horizons)
    for target in _phase8_target_columns(horizons):
        if target not in targets:
            continue
        pair = pd.concat([algebraic, targets[target]], axis=1).dropna()
        if pair.empty:
            continue
        if target == CLASSIFICATION_TARGET:
            rows.append(
                {
                    "diagnostic_type": "predictive",
                    "variable": target,
                    "metric": "auc",
                    "value": _auc(
                        pair[target].astype(int), pair["algebraic_connectivity"]
                    ),
                    "n_obs": float(len(pair)),
                },
            )
        else:
            rows.append(
                {
                    "diagnostic_type": "predictive",
                    "variable": target,
                    "metric": "pearson",
                    "value": float(pair.iloc[:, 0].corr(pair.iloc[:, 1])),
                    "n_obs": float(len(pair)),
                },
            )
    return pd.DataFrame(rows)


def evaluate_algebraic_connectivity_overlays(
    graph_features: pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    cost_bps_values: tuple[float, ...] = (0.0, 5.0, 10.0),
    method: str = "expanding",
    min_history: int = 20,
    trading_days: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare algebraic-connectivity overlay variants and costs."""

    scores = build_phase8_candidate_scores(graph_features)
    requested = [
        "graph_components_equal_weight_score",
        "excluding_algebraic_connectivity_score",
        "algebraic_connectivity_only",
        "connectivity_block_with_algebraic_connectivity",
        "connectivity_block_without_algebraic_connectivity",
    ]
    available = [column for column in requested if column in scores]
    if not available:
        raise ValueError("No algebraic-connectivity overlay scores are available.")
    return evaluate_score_overlay_set(
        returns=returns,
        scores=scores[available],
        variant="daily",
        method=method,
        min_history=min_history,
        cost_bps_values=cost_bps_values,
        trading_days=trading_days,
    )


def compute_low_turnover_exposure(
    score: pd.Series,
    variant: ExposureVariant = "daily",
    method: str = "expanding",
    high_quantile: float = 0.8,
    extreme_quantile: float = 0.9,
    normal_exposure: float = 1.0,
    high_exposure: float = 0.7,
    extreme_exposure: float = 0.5,
    min_history: int = 20,
    hysteresis_exit_quantile: float = 0.6,
    cooldown_days: int = 5,
) -> pd.Series:
    """Compute shifted exposure using a lower-turnover overlay variant."""

    _validate_overlay_variant_inputs(
        variant,
        high_quantile,
        extreme_quantile,
        hysteresis_exit_quantile,
        method,
        min_history,
    )
    signal = _numeric_series(score).sort_index()
    if variant == "smoothed_5d":
        signal = signal.rolling(5, min_periods=1).mean()
        raw = _quantile_raw_exposure(
            signal,
            method,
            high_quantile,
            extreme_quantile,
            normal_exposure,
            high_exposure,
            extreme_exposure,
            min_history,
        )
    elif variant == "smoothed_10d":
        signal = signal.rolling(10, min_periods=1).mean()
        raw = _quantile_raw_exposure(
            signal,
            method,
            high_quantile,
            extreme_quantile,
            normal_exposure,
            high_exposure,
            extreme_exposure,
            min_history,
        )
    elif variant == "hysteresis":
        raw = _hysteresis_raw_exposure(
            signal,
            method,
            high_quantile,
            extreme_quantile,
            hysteresis_exit_quantile,
            normal_exposure,
            high_exposure,
            extreme_exposure,
            min_history,
        )
    elif variant == "cooldown_5d":
        raw = _quantile_raw_exposure(
            signal,
            method,
            high_quantile,
            extreme_quantile,
            normal_exposure,
            high_exposure,
            extreme_exposure,
            min_history,
        )
        raw = _apply_cooldown(raw, cooldown_days=cooldown_days)
    elif variant == "weekly":
        raw = _quantile_raw_exposure(
            signal,
            method,
            high_quantile,
            extreme_quantile,
            normal_exposure,
            high_exposure,
            extreme_exposure,
            min_history,
        )
        raw = _weekly_hold(raw, normal_exposure=normal_exposure)
    elif variant == "smoothed_5d_hysteresis":
        signal = signal.rolling(5, min_periods=1).mean()
        raw = _hysteresis_raw_exposure(
            signal,
            method,
            high_quantile,
            extreme_quantile,
            hysteresis_exit_quantile,
            normal_exposure,
            high_exposure,
            extreme_exposure,
            min_history,
        )
    elif variant == "daily":
        raw = _quantile_raw_exposure(
            signal,
            method,
            high_quantile,
            extreme_quantile,
            normal_exposure,
            high_exposure,
            extreme_exposure,
            min_history,
        )
    else:
        raise ValueError(f"Unsupported overlay variant: {variant}")

    return raw.shift(1).fillna(normal_exposure).rename("exposure")


def evaluate_turnover_reduction_overlays(
    returns: pd.Series | pd.DataFrame,
    scores: pd.DataFrame,
    score_columns: list[str] | None = None,
    variants: tuple[ExposureVariant, ...] = DEFAULT_OVERLAY_VARIANTS,
    cost_bps_values: tuple[float, ...] = (0.0, 5.0, 10.0),
    method: str = "expanding",
    min_history: int = 20,
    trading_days: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate low-turnover variants for selected candidate scores."""

    if score_columns is None:
        score_columns = [
            column
            for column in [
                "ri",
                "graph_components_equal_weight_score",
                "transition_score",
                "pca_first_eigenvalue_share",
                "pca_first_eigenvalue_change",
                "pca_plus_graph_score",
            ]
            if column in scores
        ]
    rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    base_returns = _portfolio_series(returns)
    score_frame = scores[score_columns].apply(pd.to_numeric, errors="coerce")
    common_index = base_returns.index.intersection(score_frame.dropna(how="any").index)
    score_frame = score_frame.reindex(common_index)
    base_returns = base_returns.reindex(common_index).dropna()
    if base_returns.empty:
        raise ValueError("returns and scores do not share any valid dates.")

    for score_name in score_columns:
        if score_name not in score_frame:
            continue
        score = score_frame[score_name].dropna()
        for variant in variants:
            exposure = compute_low_turnover_exposure(
                score,
                variant=variant,
                method=method,
                min_history=min_history,
            )
            overlay_returns = apply_risk_overlay(base_returns, exposure)
            metrics = compute_performance_metrics(
                overlay_returns,
                trading_days=trading_days,
            )
            turnover = compute_exposure_turnover(exposure, trading_days=trading_days)
            row = {
                "score": score_name,
                "variant": variant,
                **metrics.to_dict(),
                **turnover.to_dict(),
                "time_in_extreme_reduction": _time_in_extreme_reduction(exposure),
            }
            rows.append(row)
            for cost_bps in cost_bps_values:
                adjusted = apply_transaction_costs(
                    overlay_returns,
                    exposure=exposure,
                    cost_bps=float(cost_bps),
                )
                adjusted_metrics = compute_performance_metrics(
                    adjusted,
                    trading_days=trading_days,
                )
                cost_rows.append(
                    {
                        "score": score_name,
                        "variant": variant,
                        "cost_bps": float(cost_bps),
                        **adjusted_metrics.to_dict(),
                        **turnover.to_dict(),
                    },
                )

    return pd.DataFrame(rows), pd.DataFrame(cost_rows)


def evaluate_score_overlay_set(
    returns: pd.Series | pd.DataFrame,
    scores: pd.DataFrame,
    variant: ExposureVariant = "daily",
    method: str = "expanding",
    min_history: int = 20,
    cost_bps_values: tuple[float, ...] = (0.0, 5.0, 10.0),
    trading_days: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a score set using one exposure variant."""

    comparison, costs = evaluate_turnover_reduction_overlays(
        returns=returns,
        scores=scores,
        score_columns=list(scores.columns),
        variants=(variant,),
        cost_bps_values=cost_bps_values,
        method=method,
        min_history=min_history,
        trading_days=trading_days,
    )
    return comparison, costs


def make_rolling_origin_splits(
    test_years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025),
    selection_start: str = "2015-07-06",
) -> list[RollingOriginSplit]:
    """Create expanding-selection, one-year-test rolling-origin splits."""

    start = pd.Timestamp(selection_start)
    splits: list[RollingOriginSplit] = []
    for year in test_years:
        splits.append(
            RollingOriginSplit(
                test_year=int(year),
                selection_start=start,
                selection_end=pd.Timestamp(f"{year - 1}-12-31"),
                test_start=pd.Timestamp(f"{year}-01-01"),
                test_end=pd.Timestamp(f"{year}-12-31"),
            ),
        )
    return splits


def run_rolling_origin_oos(
    returns: pd.Series | pd.DataFrame,
    candidate_scores: pd.DataFrame,
    candidate_specs: list[dict[str, str]] | None = None,
    splits: list[RollingOriginSplit] | None = None,
    cost_bps_values: tuple[float, ...] = (0.0, 5.0, 10.0),
    trading_days: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-selection, yearly OOS overlay evaluation.

    Thresholds are estimated only on each split's selection period. The test
    year is never used to estimate thresholds or select candidates.
    """

    if splits is None:
        splits = make_rolling_origin_splits()
    if candidate_specs is None:
        candidate_specs = _default_rolling_candidate_specs(candidate_scores)

    base_returns = _portfolio_series(returns)
    performance_rows: list[dict[str, object]] = []
    for split in splits:
        for spec in candidate_specs:
            candidate = spec["candidate"]
            score_column = spec["score"]
            variant = spec.get("variant", "daily")
            selection_mask = (candidate_scores.index >= split.selection_start) & (
                candidate_scores.index <= split.selection_end
            )
            test_mask = (candidate_scores.index >= split.test_start) & (
                candidate_scores.index <= split.test_end
            )
            if selection_mask.sum() < 20 or test_mask.sum() < 5:
                continue
            split_scores = _candidate_scores_for_split(candidate_scores, selection_mask)
            reported_score = score_column
            reported_variant = variant
            selected_score = ""
            selected_variant = ""
            if spec.get("select") == "low_turnover":
                selected = _select_low_turnover_for_split(
                    returns=base_returns,
                    split_scores=split_scores,
                    selection_start=split.selection_start,
                    selection_end=split.selection_end,
                )
                score_column = selected["score"]
                variant = selected["variant"]
                selected_score = score_column
                selected_variant = variant
                reported_score = "selection_period_best_low_turnover"
                reported_variant = "dynamic"
            if score_column not in split_scores:
                continue
            score = _numeric_series(split_scores[score_column]).sort_index()
            selection_mask = (score.index >= split.selection_start) & (
                score.index <= split.selection_end
            )
            test_mask = (score.index >= split.test_start) & (
                score.index <= split.test_end
            )
            exposure = _fixed_threshold_exposure_for_test(
                score=score,
                selection_mask=selection_mask,
                test_mask=test_mask,
                variant=variant,  # type: ignore[arg-type]
            )
            test_returns = base_returns.loc[
                (base_returns.index >= split.test_start)
                & (base_returns.index <= split.test_end)
            ]
            overlay_returns = apply_risk_overlay(test_returns, exposure)
            for cost_bps in cost_bps_values:
                adjusted = apply_transaction_costs(
                    overlay_returns,
                    exposure=exposure,
                    cost_bps=float(cost_bps),
                )
                metrics = compute_performance_metrics(
                    adjusted,
                    trading_days=trading_days,
                )
                turnover = compute_exposure_turnover(
                    exposure,
                    trading_days=trading_days,
                )
                performance_rows.append(
                    {
                        "candidate": candidate,
                        "score": reported_score,
                        "variant": reported_variant,
                        "selected_score": selected_score,
                        "selected_variant": selected_variant,
                        "test_year": float(split.test_year),
                        "cost_bps": float(cost_bps),
                        "selection_start": split.selection_start,
                        "selection_end": split.selection_end,
                        "test_start": split.test_start,
                        "test_end": split.test_end,
                        **metrics.to_dict(),
                        **turnover.to_dict(),
                    },
                )

    performance = pd.DataFrame(performance_rows)
    summary = summarize_rolling_origin_oos(performance)
    return performance, summary


def summarize_rolling_origin_oos(performance: pd.DataFrame) -> pd.DataFrame:
    """Summarize rolling-origin OOS performance across test years."""

    if performance.empty:
        return pd.DataFrame()
    metrics = [
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "worst_5pct_return",
        "annualized_turnover_proxy",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in performance.groupby(
        ["candidate", "score", "variant", "cost_bps"]
    ):
        candidate, score, variant, cost_bps = keys
        row: dict[str, object] = {
            "candidate": candidate,
            "score": score,
            "variant": variant,
            "cost_bps": float(cost_bps),
            "n_test_years": float(group["test_year"].nunique()),
        }
        for metric in metrics:
            if metric in group:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[f"mean_{metric}"] = float(values.mean())
                row[f"median_{metric}"] = float(values.median())
        row["share_years_positive_sharpe"] = float(
            (pd.to_numeric(group["sharpe"], errors="coerce") > 0).mean(),
        )
        row["share_years_drawdown_better_than_20pct"] = float(
            (pd.to_numeric(group["max_drawdown"], errors="coerce") > -0.20).mean(),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_final_candidate_selection_matrix(
    information_set_comparison: pd.DataFrame,
    rolling_origin_summary: pd.DataFrame,
    turnover_comparison: pd.DataFrame | None = None,
    candidates: list[str] | None = None,
) -> pd.DataFrame:
    """Create a final research decision matrix across candidate objects."""

    if candidates is None:
        candidates = DEFAULT_CANDIDATES
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        rolling = _candidate_rolling_row(
            rolling_origin_summary, candidate, cost_bps=0.0
        )
        cost_10 = _candidate_rolling_row(
            rolling_origin_summary,
            candidate,
            cost_bps=10.0,
        )
        info_set = _candidate_information_set(candidate)
        stress_auc = _metric_for_information_set(
            information_set_comparison,
            info_set,
            CLASSIFICATION_TARGET,
            "oos_auc",
        )
        vol_r2 = _metric_for_information_set(
            information_set_comparison,
            info_set,
            "future_realized_volatility_21d",
            "oos_r2",
        )
        row = {
            "candidate": candidate,
            "oos_sharpe": rolling.get("mean_sharpe", np.nan),
            "oos_calmar": rolling.get("mean_calmar", np.nan),
            "oos_max_drawdown": rolling.get("mean_max_drawdown", np.nan),
            "oos_sortino": rolling.get("mean_sortino", np.nan),
            "oos_worst_5pct": rolling.get("mean_worst_5pct_return", np.nan),
            "oos_stress_onset_auc": stress_auc,
            "oos_volatility_prediction_r2": vol_r2,
            "turnover": rolling.get("mean_annualized_turnover_proxy", np.nan),
            "cost_10bps_sharpe": cost_10.get("mean_sharpe", np.nan),
            "simplicity_interpretability_score": _simplicity_score(candidate),
            "depends_on_pca": candidate in {"pca_score", "pca_plus_graph_score"},
            "depends_on_graph_components": candidate
            not in {"pca_score", "best_low_turnover_score"},
            "robustness_across_years": rolling.get(
                "share_years_positive_sharpe", np.nan
            ),
        }
        rows.append(row)
    matrix = pd.DataFrame(rows)
    if matrix.empty:
        return matrix
    score = pd.Series(0.0, index=matrix.index, dtype=float)
    for column in [
        "oos_sharpe",
        "oos_calmar",
        "cost_10bps_sharpe",
        "robustness_across_years",
    ]:
        score += _safe_z(matrix[column])
    score -= 0.5 * _safe_z(matrix["turnover"])
    matrix["selection_score"] = score
    return matrix.sort_values("selection_score", ascending=False).reset_index(drop=True)


def select_low_turnover_candidate(turnover_comparison: pd.DataFrame) -> dict[str, str]:
    """Select a low-turnover candidate using Calmar, drawdown, Sharpe, turnover."""

    if turnover_comparison.empty:
        return {
            "candidate": "best_low_turnover_score",
            "score": "graph_components_equal_weight_score",
            "variant": "smoothed_5d_hysteresis",
        }
    frame = turnover_comparison.copy()
    score = (
        _safe_z(frame["calmar"])
        + _safe_z(frame["sharpe"])
        + _safe_z(frame["max_drawdown"])
        - 0.5 * _safe_z(frame["annualized_turnover_proxy"])
    )
    best = frame.loc[score.idxmax()]
    return {
        "candidate": "best_low_turnover_score",
        "score": str(best["score"]),
        "variant": str(best["variant"]),
    }


def _build_phase8_regressors(
    graph_features: pd.DataFrame,
    benchmarks: pd.DataFrame,
    pca_features: pd.DataFrame | None,
    component_scores: pd.DataFrame | None,
) -> pd.DataFrame:
    if component_scores is None:
        component_scores = graph_features
    frames = [
        benchmarks[[column for column in BASELINE_COLUMNS if column in benchmarks]]
    ]
    if "regime_indicator" in graph_features:
        frames.append(graph_features[["regime_indicator"]])
    graph_columns = [
        column for column in GRAPH_COMPONENT_COLUMNS if column in component_scores
    ]
    if graph_columns:
        frames.append(component_scores[graph_columns])
    if pca_features is not None:
        frames.append(pca_features)
    return (
        pd.concat(frames, axis=1, join="outer")
        .loc[
            :,
            lambda frame: ~frame.columns.duplicated(),
        ]
        .replace([np.inf, -np.inf], np.nan)
    )


def _information_set_specs(regressors: pd.DataFrame) -> dict[str, list[str]]:
    benchmarks = [column for column in BASELINE_COLUMNS if column in regressors]
    ri = ["regime_indicator"] if "regime_indicator" in regressors else []
    pca = [column for column in regressors.columns if column.startswith("pca_")]
    graph = [column for column in GRAPH_COMPONENT_COLUMNS if column in regressors]
    return {
        "benchmarks_only": benchmarks,
        "ri_only": ri,
        "pca_only": pca,
        "graph_components_only": graph,
        "pca_plus_graph_components": pca + graph,
        "benchmarks_plus_ri": benchmarks + ri,
        "benchmarks_plus_pca": benchmarks + pca,
        "benchmarks_plus_graph_components": benchmarks + graph,
        "benchmarks_plus_pca_plus_graph_components": benchmarks + pca + graph,
    }


def _build_phase8_targets(
    returns: pd.Series | pd.DataFrame,
    benchmarks: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    forward = compute_forward_targets(returns, horizons=list(horizons)).rename(
        columns=lambda column: (
            column.replace("forward_", "future_", 1)
            if isinstance(column, str)
            else column
        ),
    )
    onset = create_stress_onset_labels(benchmarks, min_gap=20).rename(
        CLASSIFICATION_TARGET,
    )
    return pd.concat([forward, onset], axis=1, join="outer")


def _phase8_target_columns(horizons: tuple[int, ...]) -> list[str]:
    columns: list[str] = []
    for horizon in horizons:
        columns.extend(
            [
                f"future_realized_volatility_{horizon}d",
                f"future_max_drawdown_{horizon}d",
            ],
        )
    columns.append(CLASSIFICATION_TARGET)
    return columns


def _algebraic_comparison_frame(
    graph_features: pd.DataFrame,
    benchmarks: pd.DataFrame,
    pca_features: pd.DataFrame | None,
) -> pd.DataFrame:
    columns = [
        "regime_indicator",
        "average_graph_strength",
        "weighted_edge_density",
        "laplacian_frobenius_change",
        "largest_laplacian_eigenvalue_share",
        "algebraic_connectivity",
    ]
    frames = [
        graph_features[[column for column in columns if column in graph_features]]
    ]
    if pca_features is not None and "pca_first_eigenvalue_share" in pca_features:
        frames.append(pca_features[["pca_first_eigenvalue_share"]])
    benchmark_columns = [
        column for column in BASELINE_COLUMNS if column in benchmarks.columns
    ]
    if benchmark_columns:
        frames.append(benchmarks[benchmark_columns])
    return pd.concat(frames, axis=1, join="outer").apply(
        pd.to_numeric,
        errors="coerce",
    )


def _quantile_raw_exposure(
    signal: pd.Series,
    method: str,
    high_quantile: float,
    extreme_quantile: float,
    normal_exposure: float,
    high_exposure: float,
    extreme_exposure: float,
    min_history: int,
) -> pd.Series:
    raw = pd.Series(normal_exposure, index=signal.index, dtype=float)
    if signal.dropna().empty:
        return raw
    if method == "full_sample":
        high_threshold = signal.quantile(high_quantile)
        extreme_threshold = signal.quantile(extreme_quantile)
        enough = pd.Series(True, index=signal.index)
    else:
        expanding = signal.expanding(min_periods=1)
        high_threshold = expanding.quantile(high_quantile).shift(1)
        extreme_threshold = expanding.quantile(extreme_quantile).shift(1)
        enough = expanding.count().shift(1) >= min_history

    raw.loc[((signal >= high_threshold) & enough).fillna(False)] = high_exposure
    raw.loc[((signal >= extreme_threshold) & enough).fillna(False)] = extreme_exposure
    return raw


def _hysteresis_raw_exposure(
    signal: pd.Series,
    method: str,
    high_quantile: float,
    extreme_quantile: float,
    exit_quantile: float,
    normal_exposure: float,
    high_exposure: float,
    extreme_exposure: float,
    min_history: int,
) -> pd.Series:
    if method == "full_sample":
        high_threshold = pd.Series(signal.quantile(high_quantile), index=signal.index)
        extreme_threshold = pd.Series(
            signal.quantile(extreme_quantile),
            index=signal.index,
        )
        exit_threshold = pd.Series(signal.quantile(exit_quantile), index=signal.index)
        enough = pd.Series(True, index=signal.index)
    else:
        expanding = signal.expanding(min_periods=1)
        high_threshold = expanding.quantile(high_quantile).shift(1)
        extreme_threshold = expanding.quantile(extreme_quantile).shift(1)
        exit_threshold = expanding.quantile(exit_quantile).shift(1)
        enough = expanding.count().shift(1) >= min_history

    raw = pd.Series(normal_exposure, index=signal.index, dtype=float)
    state = normal_exposure
    for index_value, value in signal.items():
        if pd.isna(value) or not bool(enough.loc[index_value]):
            raw.loc[index_value] = state
            continue
        high = high_threshold.loc[index_value]
        extreme = extreme_threshold.loc[index_value]
        exit_level = exit_threshold.loc[index_value]
        if pd.isna(high) or pd.isna(extreme) or pd.isna(exit_level):
            raw.loc[index_value] = state
            continue
        if value >= extreme:
            state = extreme_exposure
        elif value >= high:
            state = high_exposure
        elif value < exit_level:
            state = normal_exposure
        raw.loc[index_value] = state
    return raw


def _apply_cooldown(raw: pd.Series, cooldown_days: int) -> pd.Series:
    output = raw.copy()
    last_change_position = -(10**9)
    previous = output.iloc[0] if len(output) else np.nan
    for position in range(1, len(output)):
        current = output.iloc[position]
        if (
            not np.isclose(current, previous)
            and position - last_change_position <= cooldown_days
        ):
            output.iloc[position] = previous
        elif not np.isclose(current, previous):
            last_change_position = position
            previous = current
        else:
            previous = current
    return output


def _weekly_hold(raw: pd.Series, normal_exposure: float) -> pd.Series:
    if raw.empty:
        return raw
    if isinstance(raw.index, pd.DatetimeIndex):
        update_mask = pd.Series(raw.index.dayofweek == 0, index=raw.index)
    else:
        update_mask = pd.Series(np.arange(len(raw)) % 5 == 0, index=raw.index)
    update_mask.iloc[0] = True
    held = raw.where(update_mask).ffill().fillna(normal_exposure)
    return held.astype(float)


def _fixed_threshold_exposure_for_test(
    score: pd.Series,
    selection_mask: np.ndarray,
    test_mask: np.ndarray,
    variant: ExposureVariant,
    high_quantile: float = 0.8,
    extreme_quantile: float = 0.9,
    normal_exposure: float = 1.0,
    high_exposure: float = 0.7,
    extreme_exposure: float = 0.5,
) -> pd.Series:
    test_index = score.index[test_mask]
    selection = score.loc[selection_mask].dropna()
    high_threshold = selection.quantile(high_quantile)
    extreme_threshold = selection.quantile(extreme_quantile)
    exit_threshold = selection.quantile(0.6)
    window = score.loc[selection_mask | test_mask]
    if variant in {"smoothed_5d", "smoothed_5d_hysteresis"}:
        window = window.rolling(5, min_periods=1).mean()
    elif variant == "smoothed_10d":
        window = window.rolling(10, min_periods=1).mean()

    if variant in {"hysteresis", "smoothed_5d_hysteresis"}:
        raw = _fixed_hysteresis_raw(
            window,
            high_threshold,
            extreme_threshold,
            exit_threshold,
            normal_exposure,
            high_exposure,
            extreme_exposure,
        )
    else:
        raw = pd.Series(normal_exposure, index=window.index, dtype=float)
        raw.loc[window >= high_threshold] = high_exposure
        raw.loc[window >= extreme_threshold] = extreme_exposure
        if variant == "weekly":
            raw = _weekly_hold(raw, normal_exposure)
        elif variant == "cooldown_5d":
            raw = _apply_cooldown(raw, cooldown_days=5)
    exposure = raw.shift(1).fillna(normal_exposure)
    return exposure.reindex(test_index).rename("exposure")


def _fixed_hysteresis_raw(
    signal: pd.Series,
    high_threshold: float,
    extreme_threshold: float,
    exit_threshold: float,
    normal_exposure: float,
    high_exposure: float,
    extreme_exposure: float,
) -> pd.Series:
    raw = pd.Series(normal_exposure, index=signal.index, dtype=float)
    state = normal_exposure
    for index_value, value in signal.items():
        if pd.isna(value):
            raw.loc[index_value] = state
            continue
        if value >= extreme_threshold:
            state = extreme_exposure
        elif value >= high_threshold:
            state = high_exposure
        elif value < exit_threshold:
            state = normal_exposure
        raw.loc[index_value] = state
    return raw


def _default_rolling_candidate_specs(
    candidate_scores: pd.DataFrame,
) -> list[dict[str, str]]:
    specs = [
        {"candidate": "ri", "score": "ri", "variant": "daily"},
        {
            "candidate": "graph_components_equal_weight_score",
            "score": "graph_components_equal_weight_score",
            "variant": "daily",
        },
        {
            "candidate": "transition_score",
            "score": "transition_score",
            "variant": "daily",
        },
        {"candidate": "pca_score", "score": "pca_score", "variant": "daily"},
        {
            "candidate": "pca_plus_graph_score",
            "score": "pca_plus_graph_score",
            "variant": "daily",
        },
        {
            "candidate": "excluding_algebraic_connectivity_score",
            "score": "excluding_algebraic_connectivity_score",
            "variant": "daily",
        },
    ]
    return [spec for spec in specs if spec["score"] in candidate_scores]


def _candidate_scores_for_split(
    candidate_scores: pd.DataFrame,
    selection_mask: np.ndarray,
) -> pd.DataFrame:
    output = pd.DataFrame(index=candidate_scores.index)
    raw_graph = [
        column for column in GRAPH_COMPONENT_COLUMNS if column in candidate_scores
    ]
    if "ri" in candidate_scores:
        output["ri"] = _selection_scaled_ri(candidate_scores, selection_mask)
    if raw_graph:
        output["graph_components_equal_weight_score"] = _mean_selection_z(
            candidate_scores[raw_graph],
            selection_mask,
        )
        excluding = [
            column for column in raw_graph if column != "algebraic_connectivity"
        ]
        if excluding:
            output["excluding_algebraic_connectivity_score"] = _mean_selection_z(
                candidate_scores[excluding],
                selection_mask,
            )
    if "laplacian_frobenius_change" in candidate_scores:
        output["transition_score"] = _selection_z(
            candidate_scores["laplacian_frobenius_change"],
            selection_mask,
        )
    pca_columns = [
        column
        for column in ["pca_first_eigenvalue_share", "pca_first_eigenvalue_change"]
        if column in candidate_scores
    ]
    for column in pca_columns:
        output[column] = _numeric_series(candidate_scores[column])
    if pca_columns:
        output["pca_score"] = _mean_selection_z(
            candidate_scores[pca_columns],
            selection_mask,
        )
    combined = [
        column
        for column in [
            "graph_components_equal_weight_score",
            "transition_score",
            "pca_score",
        ]
        if column in output
    ]
    if combined:
        output["pca_plus_graph_score"] = _mean_selection_z(
            output[combined],
            selection_mask,
        )
    return output.replace([np.inf, -np.inf], np.nan)


def _selection_scaled_ri(
    candidate_scores: pd.DataFrame,
    selection_mask: np.ndarray,
) -> pd.Series:
    terms = []
    for column, sign in [
        ("average_graph_strength", 1.0),
        ("algebraic_connectivity", 1.0),
        ("largest_laplacian_eigenvalue_share", 1.0),
        ("modularity", -1.0),
        ("laplacian_frobenius_change", 1.0),
    ]:
        if column in candidate_scores:
            terms.append(sign * _selection_z(candidate_scores[column], selection_mask))
    if terms:
        return pd.concat(terms, axis=1).sum(axis=1, skipna=True).fillna(0.0)
    return _numeric_series(candidate_scores["ri"])


def _select_low_turnover_for_split(
    returns: pd.Series,
    split_scores: pd.DataFrame,
    selection_start: pd.Timestamp,
    selection_end: pd.Timestamp,
) -> dict[str, str]:
    selection_scores = split_scores.loc[
        (split_scores.index >= selection_start) & (split_scores.index <= selection_end)
    ]
    selection_returns = returns.loc[
        (returns.index >= selection_start) & (returns.index <= selection_end)
    ]
    comparison, _ = evaluate_turnover_reduction_overlays(
        returns=selection_returns,
        scores=selection_scores,
    )
    selected = select_low_turnover_candidate(comparison)
    return {"score": selected["score"], "variant": selected["variant"]}


def _candidate_rolling_row(
    summary: pd.DataFrame,
    candidate: str,
    cost_bps: float | None = None,
) -> pd.Series:
    if summary.empty or "candidate" not in summary:
        return pd.Series(dtype=float)
    frame = summary.loc[summary["candidate"] == candidate]
    if cost_bps is not None and "cost_bps" in frame:
        frame = frame.loc[
            pd.to_numeric(frame["cost_bps"], errors="coerce") == float(cost_bps)
        ]
    if frame.empty:
        return pd.Series(dtype=float)
    if "mean_sharpe" in frame:
        return frame.sort_values("mean_sharpe", ascending=False).iloc[0]
    return frame.iloc[0]


def _candidate_information_set(candidate: str) -> str:
    return {
        "ri": "benchmarks_plus_ri",
        "graph_components_equal_weight_score": "benchmarks_plus_graph_components",
        "transition_score": "benchmarks_plus_graph_components",
        "pca_score": "benchmarks_plus_pca",
        "pca_plus_graph_score": "benchmarks_plus_pca_plus_graph_components",
        "excluding_algebraic_connectivity_score": "benchmarks_plus_graph_components",
        "best_low_turnover_score": "benchmarks_plus_graph_components",
    }.get(candidate, "benchmarks_plus_graph_components")


def _metric_for_information_set(
    comparison: pd.DataFrame,
    information_set: str,
    target: str,
    metric: str,
) -> float:
    if comparison.empty or metric not in comparison:
        return np.nan
    rows = comparison.loc[
        (comparison["information_set"] == information_set)
        & (comparison["target"] == target)
    ]
    if rows.empty:
        return np.nan
    value = pd.to_numeric(rows.iloc[0][metric], errors="coerce")
    return float(value) if np.isfinite(value) else np.nan


def _simplicity_score(candidate: str) -> float:
    return {
        "ri": 4.0,
        "transition_score": 5.0,
        "pca_score": 5.0,
        "graph_components_equal_weight_score": 3.5,
        "excluding_algebraic_connectivity_score": 3.0,
        "pca_plus_graph_score": 2.5,
        "best_low_turnover_score": 2.5,
    }.get(candidate, 3.0)


def _portfolio_series(returns: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(returns, pd.DataFrame):
        return compute_portfolio_returns(returns)
    if isinstance(returns, pd.Series):
        return _numeric_series(returns).rename("portfolio_returns")
    raise TypeError("returns must be a pandas Series or DataFrame.")


def _validate_overlay_variant_inputs(
    variant: str,
    high_quantile: float,
    extreme_quantile: float,
    hysteresis_exit_quantile: float,
    method: str,
    min_history: int,
) -> None:
    if not 0 <= high_quantile < extreme_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= high < extreme <= 1.")
    if (
        variant in {"hysteresis", "smoothed_5d_hysteresis"}
        and not 0 <= hysteresis_exit_quantile < high_quantile
    ):
        raise ValueError(
            "hysteresis_exit_quantile must be lower than high_quantile.",
        )
    if method not in {"expanding", "full_sample"}:
        raise ValueError("method must be 'expanding' or 'full_sample'.")
    if min_history <= 0:
        raise ValueError("min_history must be positive.")


def _time_in_extreme_reduction(exposure: pd.Series) -> float:
    valid = pd.to_numeric(exposure, errors="coerce").dropna()
    return float((valid <= 0.5).mean()) if not valid.empty else np.nan


def _mean_z(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    z_frame = pd.DataFrame(
        {column: _safe_z(frame[column]) for column in frame.columns},
        index=frame.index,
    )
    return z_frame.mean(axis=1, skipna=True).fillna(0.0)


def _mean_selection_z(frame: pd.DataFrame, selection_mask: np.ndarray) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    z_frame = pd.DataFrame(
        {
            column: _selection_z(frame[column], selection_mask)
            for column in frame.columns
        },
        index=frame.index,
    )
    return z_frame.mean(axis=1, skipna=True).fillna(0.0)


def _selection_z(series: pd.Series, selection_mask: np.ndarray) -> pd.Series:
    numeric = _numeric_series(series)
    selection = numeric.loc[selection_mask].dropna()
    mean = selection.mean()
    std = selection.std(ddof=0)
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        return pd.Series(0.0, index=series.index, dtype=float)
    return ((numeric - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_z(series: pd.Series) -> pd.Series:
    numeric = _numeric_series(series)
    mean = numeric.mean()
    std = numeric.std(ddof=0)
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        return pd.Series(0.0, index=series.index, dtype=float)
    return ((numeric - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
