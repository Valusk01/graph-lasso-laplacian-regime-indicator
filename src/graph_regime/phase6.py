"""Phase 6 robustness, OOS, and incremental-information research utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd

from graph_regime.evaluation import (
    compute_contemporaneous_diagnostics,
    compute_forward_targets,
)
from graph_regime.indicator import (
    compute_regime_indicator,
    compute_rolling_graph_features,
    summarize_graph_lasso_convergence,
)
from graph_regime.risk_overlay import (
    apply_risk_overlay,
    compute_benchmark_exposures,
    compute_exposure_from_indicator,
    compute_performance_metrics,
    compute_portfolio_returns,
)
from graph_regime.robustness import (
    compute_benchmark_changes,
    compute_transition_correlations,
    create_stress_onset_labels,
    summarize_robustness_diagnostics,
)

SampleName = Literal["development", "validation", "test"]


@dataclass(frozen=True)
class Phase6Config:
    """Controlled graph-regime configuration for robustness research."""

    config_id: str
    alpha: float
    window: int
    partial_corr_threshold: float
    laplacian_type: str = "combinatorial"
    network_type: str = "absolute_partial_correlation"
    compute_modularity: bool = False


@dataclass(frozen=True)
class SampleSplits:
    """Date boundaries for development, validation, and test samples."""

    development_start: str = "2015-07-06"
    development_end: str = "2019-12-31"
    validation_start: str = "2020-01-01"
    validation_end: str = "2022-12-31"
    test_start: str = "2023-01-01"


DEFAULT_SAMPLE_SPLITS = SampleSplits()


def make_phase6_parameter_grid(
    alpha_values: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
    windows: tuple[int, ...] = (63, 126, 252),
    partial_corr_thresholds: tuple[float, ...] = (0.00, 0.03),
) -> list[Phase6Config]:
    """Create the controlled Phase 6 robustness grid.

    The default grid has 24 configurations: four graphical-lasso penalties,
    three rolling windows, and two partial-correlation thresholds. Laplacian
    type and network construction are intentionally held fixed in this phase.
    """

    _validate_grid_inputs(alpha_values, windows, partial_corr_thresholds)
    configs: list[Phase6Config] = []
    for alpha, window, threshold in product(
        alpha_values, windows, partial_corr_thresholds
    ):
        configs.append(
            Phase6Config(
                config_id=f"alpha{alpha:.2f}_window{window}_threshold{threshold:.2f}",
                alpha=float(alpha),
                window=int(window),
                partial_corr_threshold=float(threshold),
            ),
        )
    return configs


def make_phase6_quick_grid() -> list[Phase6Config]:
    """Create a small Phase 6 grid for smoke tests and local iteration."""

    return make_phase6_parameter_grid(
        alpha_values=(0.20,),
        windows=(63,),
        partial_corr_thresholds=(0.03,),
    )


def assign_sample_period(
    index: pd.Index,
    splits: SampleSplits = DEFAULT_SAMPLE_SPLITS,
) -> pd.Series:
    """Assign each date to development, validation, test, or missing period."""

    dates = pd.to_datetime(index)
    labels = pd.Series(pd.NA, index=index, dtype="object", name="sample")

    development = (dates >= pd.Timestamp(splits.development_start)) & (
        dates <= pd.Timestamp(splits.development_end)
    )
    validation = (dates >= pd.Timestamp(splits.validation_start)) & (
        dates <= pd.Timestamp(splits.validation_end)
    )
    test = dates >= pd.Timestamp(splits.test_start)

    labels.loc[development] = "development"
    labels.loc[validation] = "validation"
    labels.loc[test] = "test"
    return labels


def compute_exposure_turnover(
    exposure: pd.Series,
    trading_days: int = 252,
) -> pd.Series:
    """Summarize exposure level, reduced-exposure time, and turnover."""

    if not isinstance(exposure, pd.Series):
        raise TypeError("exposure must be a pandas Series.")
    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")

    numeric = pd.to_numeric(exposure, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(
            {
                "average_exposure": np.nan,
                "time_in_reduced_exposure": np.nan,
                "number_of_exposure_changes": 0.0,
                "annualized_turnover_proxy": np.nan,
                "mean_absolute_exposure_change": np.nan,
            },
        )

    changes = valid.diff().abs().fillna(0.0)
    n_changes = float((changes > 1e-12).sum())
    annualized_turnover = float(changes.sum() * trading_days / len(valid))
    return pd.Series(
        {
            "average_exposure": float(valid.mean()),
            "time_in_reduced_exposure": float((valid < 1.0).mean()),
            "number_of_exposure_changes": n_changes,
            "annualized_turnover_proxy": annualized_turnover,
            "mean_absolute_exposure_change": float(changes.mean()),
        },
    )


def apply_transaction_costs(
    overlay_returns: pd.Series,
    exposure: pd.Series,
    cost_bps: float,
) -> pd.Series:
    """Subtract a simple exposure-turnover transaction-cost proxy.

    Costs are charged on absolute changes in exposure. A value of 5 bps means
    0.0005 return cost for a full one-unit exposure change.
    """

    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative.")
    if not isinstance(overlay_returns, pd.Series):
        raise TypeError("overlay_returns must be a pandas Series.")
    if not isinstance(exposure, pd.Series):
        raise TypeError("exposure must be a pandas Series.")

    aligned = pd.concat(
        [
            pd.to_numeric(overlay_returns, errors="coerce").rename("returns"),
            pd.to_numeric(exposure, errors="coerce").rename("exposure"),
        ],
        axis=1,
        join="inner",
    ).replace([np.inf, -np.inf], np.nan)
    aligned = aligned.dropna(subset=["returns"])
    aligned["exposure"] = aligned["exposure"].fillna(1.0)

    turnover = aligned["exposure"].diff().abs().fillna(0.0)
    costs = turnover * (cost_bps / 10_000.0)
    return (aligned["returns"] - costs).rename(f"overlay_returns_after_{cost_bps:g}bps")


def evaluate_transaction_cost_sensitivity(
    portfolio_returns: pd.Series,
    exposure: pd.Series,
    cost_bps_values: tuple[float, ...] = (0.0, 5.0, 10.0),
    trading_days: int = 252,
) -> pd.DataFrame:
    """Evaluate an overlay under simple transaction-cost assumptions."""

    overlay_returns = apply_risk_overlay(portfolio_returns, exposure)
    turnover = compute_exposure_turnover(exposure, trading_days=trading_days)
    rows: list[dict[str, float]] = []

    for cost_bps in cost_bps_values:
        cost_adjusted = apply_transaction_costs(
            overlay_returns,
            exposure=exposure,
            cost_bps=float(cost_bps),
        )
        metrics = compute_performance_metrics(cost_adjusted, trading_days=trading_days)
        row = {"cost_bps": float(cost_bps), **metrics.to_dict(), **turnover.to_dict()}
        rows.append(row)

    return pd.DataFrame(rows)


def run_phase6_robustness_grid(
    returns: pd.DataFrame,
    benchmarks: pd.DataFrame,
    configs: list[Phase6Config],
    splits: SampleSplits = DEFAULT_SAMPLE_SPLITS,
    min_non_missing: float = 0.95,
    max_iter: int = 1000,
    tol: float = 1e-4,
    enet_tol: float = 1e-4,
    mode: str = "cd",
    exposure_method: str = "expanding",
    min_history: int = 20,
    trading_days: int = 252,
    forward_horizon: int = 21,
    return_indicator_frames: bool = False,
) -> dict[str, pd.DataFrame] | tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Run Phase 6 robustness configurations and collect summary tables."""

    if not configs:
        raise ValueError("configs must contain at least one Phase6Config.")
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if not isinstance(benchmarks, pd.DataFrame):
        raise TypeError("benchmarks must be a pandas DataFrame.")

    base_returns = compute_portfolio_returns(returns)
    benchmark_exposures = _safe_benchmark_exposures(
        benchmarks,
        method=exposure_method,
        min_history=min_history,
    )

    summary_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    overlay_metric_frames: list[pd.DataFrame] = []
    benchmark_metric_frames: list[pd.DataFrame] = []
    indicator_frames: dict[str, pd.DataFrame] = {}

    for config in configs:
        features = compute_rolling_graph_features(
            returns,
            window=config.window,
            alpha=config.alpha,
            min_non_missing=min_non_missing,
            partial_corr_threshold=config.partial_corr_threshold,
            max_iter=max_iter,
            compute_modularity=config.compute_modularity,
            tol=tol,
            enet_tol=enet_tol,
            mode=mode,
            on_non_convergence="record",
        )
        indicator_frame = compute_regime_indicator(features)
        regime_indicator = indicator_frame["regime_indicator"]
        config_fields = _config_fields(config)
        if return_indicator_frames:
            indicator_frames[config.config_id] = indicator_frame

        summary = summarize_robustness_diagnostics(
            regime_indicator,
            benchmarks=benchmarks,
            returns=returns,
            convergence=features.get("graph_lasso_converged"),
            forward_horizon=forward_horizon,
        )
        summary.update(_transition_summary(regime_indicator, benchmarks))
        summary_rows.append({**config_fields, **summary})

        convergence_rows.append(
            {
                **config_fields,
                **_convergence_summary(features),
            },
        )

        ri_exposure = compute_exposure_from_indicator(
            regime_indicator,
            method=exposure_method,
            min_history=min_history,
        ).rename("ri_exposure")
        ri_overlay_returns = apply_risk_overlay(base_returns, ri_exposure).rename(
            "ri_overlay"
        )

        overlay_metric_frames.append(
            _strategy_metric_table(
                {"baseline": base_returns, "ri_overlay": ri_overlay_returns},
                {"ri_overlay": ri_exposure},
                config_fields=config_fields,
                splits=splits,
                trading_days=trading_days,
            ),
        )

        strategies = {"baseline": base_returns, "ri_overlay": ri_overlay_returns}
        exposures = {"ri_overlay": ri_exposure}
        for exposure_column in benchmark_exposures.columns:
            strategy_name = exposure_column.replace("_exposure", "_overlay")
            exposure = benchmark_exposures[exposure_column]
            strategies[strategy_name] = apply_risk_overlay(
                base_returns, exposure
            ).rename(
                strategy_name,
            )
            exposures[strategy_name] = exposure

        benchmark_metric_frames.append(
            _strategy_metric_table(
                strategies,
                exposures,
                config_fields=config_fields,
                splits=splits,
                trading_days=trading_days,
            ),
        )

    robustness_overlay_metrics = _add_baseline_differences(
        pd.concat(overlay_metric_frames, ignore_index=True),
    )
    robustness_benchmark_comparison = _add_baseline_differences(
        pd.concat(benchmark_metric_frames, ignore_index=True),
    )

    tables = {
        "robustness_grid_summary": pd.DataFrame(summary_rows),
        "robustness_overlay_metrics": robustness_overlay_metrics,
        "robustness_convergence_summary": pd.DataFrame(convergence_rows),
        "robustness_benchmark_comparison": robustness_benchmark_comparison,
    }
    if return_indicator_frames:
        return tables, indicator_frames
    return tables


def build_phase6_tables_from_indicator(
    indicator: pd.DataFrame | pd.Series,
    returns: pd.DataFrame,
    benchmarks: pd.DataFrame,
    convergence: pd.Series | pd.DataFrame | None = None,
    config: Phase6Config | None = None,
    splits: SampleSplits = DEFAULT_SAMPLE_SPLITS,
    exposure_method: str = "expanding",
    min_history: int = 20,
    trading_days: int = 252,
    forward_horizon: int = 21,
) -> dict[str, pd.DataFrame]:
    """Build Phase 6 diagnostics from an already-computed RI table.

    This is useful for fast smoke workflows that reuse saved Phase 4 outputs.
    Full robustness analysis should still recompute RI through
    ``run_phase6_robustness_grid``.
    """

    indicator_frame = _indicator_frame(indicator)
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if not isinstance(benchmarks, pd.DataFrame):
        raise TypeError("benchmarks must be a pandas DataFrame.")

    if config is None:
        config = Phase6Config(
            config_id="cached_indicator",
            alpha=np.nan,
            window=0,
            partial_corr_threshold=np.nan,
        )

    config_fields = _config_fields(config)
    regime_indicator = indicator_frame["regime_indicator"]
    convergence_series = _convergence_series(indicator_frame, convergence)
    base_returns = compute_portfolio_returns(returns)
    benchmark_exposures = _safe_benchmark_exposures(
        benchmarks,
        method=exposure_method,
        min_history=min_history,
    )

    summary = summarize_robustness_diagnostics(
        regime_indicator,
        benchmarks=benchmarks,
        returns=returns,
        convergence=convergence_series,
        forward_horizon=forward_horizon,
    )
    summary.update(_transition_summary(regime_indicator, benchmarks))

    convergence_frame = _convergence_frame(indicator_frame, convergence)
    convergence_summary = (
        _convergence_summary(convergence_frame)
        if "graph_lasso_converged" in convergence_frame
        else {
            "n_windows": int(len(regime_indicator.dropna())),
            "n_converged": np.nan,
            "n_non_converged": np.nan,
            "convergence_rate": np.nan,
            "graph_lasso_warning_count_total": np.nan,
            "graph_lasso_n_iter_mean": np.nan,
            "graph_lasso_n_iter_max": np.nan,
        }
    )

    ri_exposure = compute_exposure_from_indicator(
        regime_indicator,
        method=exposure_method,
        min_history=min_history,
    ).rename("ri_exposure")
    ri_overlay_returns = apply_risk_overlay(base_returns, ri_exposure).rename(
        "ri_overlay"
    )

    overlay_metrics = _strategy_metric_table(
        {"baseline": base_returns, "ri_overlay": ri_overlay_returns},
        {"ri_overlay": ri_exposure},
        config_fields=config_fields,
        splits=splits,
        trading_days=trading_days,
    )

    strategies = {"baseline": base_returns, "ri_overlay": ri_overlay_returns}
    exposures = {"ri_overlay": ri_exposure}
    for exposure_column in benchmark_exposures.columns:
        strategy_name = exposure_column.replace("_exposure", "_overlay")
        exposure = benchmark_exposures[exposure_column]
        strategies[strategy_name] = apply_risk_overlay(base_returns, exposure).rename(
            strategy_name,
        )
        exposures[strategy_name] = exposure

    benchmark_metrics = _strategy_metric_table(
        strategies,
        exposures,
        config_fields=config_fields,
        splits=splits,
        trading_days=trading_days,
    )

    return {
        "robustness_grid_summary": pd.DataFrame([{**config_fields, **summary}]),
        "robustness_overlay_metrics": _add_baseline_differences(overlay_metrics),
        "robustness_convergence_summary": pd.DataFrame(
            [{**config_fields, **convergence_summary}],
        ),
        "robustness_benchmark_comparison": _add_baseline_differences(benchmark_metrics),
    }


def select_oos_configuration(
    robustness_overlay_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Select a configuration using validation-only risk-overlay diagnostics."""

    required = {
        "config_id",
        "sample",
        "strategy",
        "sharpe",
        "calmar",
        "max_drawdown",
        "annualized_volatility",
        "worst_5pct_return",
        "annualized_turnover_proxy",
    }
    missing = sorted(required - set(robustness_overlay_metrics.columns))
    if missing:
        raise KeyError(f"robustness_overlay_metrics is missing columns: {missing}")

    rows: list[dict[str, object]] = []
    validation = robustness_overlay_metrics.loc[
        robustness_overlay_metrics["sample"] == "validation"
    ]
    if validation.empty:
        raise ValueError("No validation sample rows are available for selection.")

    for config_id, group in validation.groupby("config_id", sort=False):
        baseline = _single_strategy_row(group, "baseline")
        overlay = _single_strategy_row(group, "ri_overlay")
        if baseline is None or overlay is None:
            continue
        row = {
            "config_id": config_id,
            "alpha": overlay.get("alpha", np.nan),
            "window": overlay.get("window", np.nan),
            "partial_corr_threshold": overlay.get("partial_corr_threshold", np.nan),
            "validation_sharpe": overlay["sharpe"],
            "validation_calmar": overlay["calmar"],
            "validation_max_drawdown_reduction": overlay["max_drawdown"]
            - baseline["max_drawdown"],
            "validation_volatility_reduction": baseline["annualized_volatility"]
            - overlay["annualized_volatility"],
            "validation_downside_improvement": overlay["worst_5pct_return"]
            - baseline["worst_5pct_return"],
            "validation_annualized_turnover_proxy": overlay[
                "annualized_turnover_proxy"
            ],
        }
        rows.append(row)

    selection = pd.DataFrame(rows)
    if selection.empty:
        raise ValueError(
            "No candidate configuration has validation baseline and RI rows."
        )

    positive_columns = [
        "validation_sharpe",
        "validation_calmar",
        "validation_max_drawdown_reduction",
        "validation_volatility_reduction",
        "validation_downside_improvement",
    ]
    score = pd.Series(0.0, index=selection.index, dtype=float)
    for column in positive_columns:
        score += _safe_z(selection[column])
    score -= 0.25 * _safe_z(selection["validation_annualized_turnover_proxy"])
    selection["selection_score"] = score
    best_index = selection["selection_score"].idxmax()
    selection["selected"] = False
    selection.loc[best_index, "selected"] = True
    return selection.sort_values("selection_score", ascending=False).reset_index(
        drop=True
    )


def build_oos_outputs(
    robustness_overlay_metrics: pd.DataFrame,
    robustness_benchmark_comparison: pd.DataFrame,
    oos_selection_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build final test-only OOS performance tables for the selected config."""

    if "selected" not in oos_selection_summary.columns:
        raise KeyError("oos_selection_summary is missing selected.")
    selected = oos_selection_summary.loc[oos_selection_summary["selected"]]
    if selected.empty:
        raise ValueError(
            "No selected configuration is marked in oos_selection_summary."
        )

    selected_config = str(selected.iloc[0]["config_id"])
    test_overlay = robustness_overlay_metrics.loc[
        (robustness_overlay_metrics["config_id"] == selected_config)
        & (robustness_overlay_metrics["sample"] == "test")
    ].copy()
    test_comparison = robustness_benchmark_comparison.loc[
        (robustness_benchmark_comparison["config_id"] == selected_config)
        & (robustness_benchmark_comparison["sample"] == "test")
    ].copy()

    return {
        "oos_test_performance": test_overlay.reset_index(drop=True),
        "oos_overlay_comparison": test_comparison.reset_index(drop=True),
    }


def run_incremental_information_tests(
    indicator: pd.DataFrame | pd.Series,
    benchmarks: pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    splits: SampleSplits = DEFAULT_SAMPLE_SPLITS,
    horizons: tuple[int, ...] = (5, 21),
    baseline_columns: tuple[str, ...] = (
        "vix",
        "realized_volatility",
        "drawdown",
        "average_correlation",
        "average_absolute_correlation",
    ),
    graph_component_columns: tuple[str, ...] = (
        "average_graph_strength",
        "algebraic_connectivity",
        "laplacian_frobenius_change",
        "largest_laplacian_eigenvalue_share",
    ),
    newey_west_lags: int = 5,
) -> pd.DataFrame:
    """Test whether RI adds information beyond simple benchmark regressors."""

    indicator_frame = _indicator_frame(indicator)
    if not isinstance(benchmarks, pd.DataFrame):
        raise TypeError("benchmarks must be a pandas DataFrame.")

    available_baselines = [
        column for column in baseline_columns if column in benchmarks.columns
    ]
    available_graph = [
        column for column in graph_component_columns if column in indicator_frame
    ]
    regressors = pd.concat(
        [
            benchmarks[available_baselines],
            indicator_frame[["regime_indicator"] + available_graph],
        ],
        axis=1,
        join="outer",
    )

    forward_targets = compute_forward_targets(returns, horizons=list(horizons))
    onset = create_stress_onset_labels(benchmarks, min_gap=20).rename(
        "stress_onset_label"
    )
    targets = pd.concat([forward_targets, onset], axis=1, join="outer")

    rows: list[dict[str, object]] = []
    model_specs = {
        "baseline": available_baselines,
        "baseline_plus_ri": available_baselines + ["regime_indicator"],
        "baseline_plus_graph_components": available_baselines
        + ["regime_indicator"]
        + available_graph,
    }

    for target_column in _phase6_target_columns(targets, horizons):
        y = pd.to_numeric(targets[target_column], errors="coerce").rename(target_column)
        for model_name, columns in model_specs.items():
            if not columns:
                continue
            model_frame = (
                pd.concat([y, regressors[columns]], axis=1, join="inner")
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
            )
            if model_frame.empty:
                rows.append(_empty_incremental_row(target_column, model_name, "model"))
                continue

            y_values = model_frame[target_column]
            x_values = model_frame[columns]
            fit = _fit_ols(y_values, x_values, newey_west_lags=newey_west_lags)
            sample_labels = assign_sample_period(model_frame.index, splits)
            oos_r2 = _oos_r2(y_values, x_values, columns, sample_labels)

            classification = target_column == "stress_onset_label"
            fitted = fit["fitted"]
            auc = _auc_score(y_values, fitted) if classification else np.nan
            brier = _brier_score(y_values, fitted) if classification else np.nan
            oos_auc, oos_brier = (
                _oos_classification_metrics(
                    y_values,
                    x_values,
                    columns,
                    sample_labels,
                )
                if classification
                else (np.nan, np.nan)
            )

            for regressor in columns:
                coefficient = fit["coefficients"].get(regressor, np.nan)
                t_stat = fit["t_stats"].get(regressor, np.nan)
                rows.append(
                    {
                        "target": target_column,
                        "model": model_name,
                        "regressor": regressor,
                        "n_obs": float(len(model_frame)),
                        "adjusted_r2": fit["adjusted_r2"],
                        "oos_r2": oos_r2,
                        "coefficient": coefficient,
                        "coefficient_sign": _coefficient_sign(coefficient),
                        "t_stat": t_stat,
                        "auc": auc,
                        "brier_score": brier,
                        "oos_auc": oos_auc,
                        "oos_brier_score": oos_brier,
                    },
                )

    return pd.DataFrame(rows)


def _strategy_metric_table(
    strategy_returns: dict[str, pd.Series],
    exposures: dict[str, pd.Series],
    config_fields: dict[str, object],
    splits: SampleSplits,
    trading_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample_labels = assign_sample_period(
        pd.Index(
            sorted(
                set().union(*(returns.index for returns in strategy_returns.values()))
            )
        ),
        splits,
    )
    sample_names = ["all", "development", "validation", "test"]

    for sample_name in sample_names:
        for strategy_name, returns in strategy_returns.items():
            if sample_name == "all":
                sample_returns = returns
                sample_exposure = exposures.get(strategy_name)
            else:
                labels = sample_labels.reindex(returns.index)
                sample_returns = returns.loc[labels == sample_name]
                exposure = exposures.get(strategy_name)
                sample_exposure = (
                    exposure.loc[labels.reindex(exposure.index) == sample_name]
                    if exposure is not None
                    else None
                )

            metrics = compute_performance_metrics(
                sample_returns, trading_days=trading_days
            )
            turnover = (
                compute_exposure_turnover(sample_exposure, trading_days=trading_days)
                if sample_exposure is not None
                else _empty_turnover()
            )
            rows.append(
                {
                    **config_fields,
                    "sample": sample_name,
                    "strategy": strategy_name,
                    **metrics.to_dict(),
                    **turnover.to_dict(),
                },
            )

    return pd.DataFrame(rows)


def _safe_benchmark_exposures(
    benchmarks: pd.DataFrame,
    method: str,
    min_history: int,
) -> pd.DataFrame:
    try:
        return compute_benchmark_exposures(
            benchmarks,
            method=method,
            min_history=min_history,
        )
    except ValueError:
        return pd.DataFrame(index=benchmarks.index)


def _add_baseline_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    difference_columns = [
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "worst_5pct_return",
    ]
    for column in difference_columns:
        output[f"{column}_minus_baseline"] = np.nan

    for (_, sample), group in output.groupby(["config_id", "sample"], sort=False):
        baseline = group.loc[group["strategy"] == "baseline"]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        for idx in group.index:
            for column in difference_columns:
                output.loc[idx, f"{column}_minus_baseline"] = (
                    output.loc[idx, column] - baseline_row[column]
                )
    return output


def _transition_summary(
    indicator: pd.Series,
    benchmarks: pd.DataFrame,
) -> dict[str, float]:
    summary: dict[str, float] = {
        "stress_onset_ri_difference": np.nan,
        "correlation_with_vix_change": np.nan,
        "correlation_with_realized_volatility_change": np.nan,
        "correlation_with_average_correlation_change": np.nan,
        "correlation_with_average_absolute_correlation_change": np.nan,
        "correlation_with_drawdown_change": np.nan,
    }

    try:
        changes = compute_benchmark_changes(benchmarks)
        correlations = compute_transition_correlations(indicator, changes)
        for _, row in correlations.iterrows():
            benchmark = str(row["benchmark"])
            key = f"correlation_with_{benchmark}"
            summary[key] = float(row["pearson_correlation"])
    except ValueError:
        pass

    onset = create_stress_onset_labels(benchmarks, min_gap=20)
    aligned = pd.concat(
        [indicator.rename("regime_indicator"), onset],
        axis=1,
        join="inner",
    )
    diagnostics = compute_contemporaneous_diagnostics(
        aligned,
        stress_label_columns=["stress_onset_label"],
        continuous_benchmark_columns=[],
    )
    difference = diagnostics.loc[
        (diagnostics["benchmark"] == "stress_onset_label")
        & (diagnostics["metric"] == "difference_in_means"),
        "value",
    ]
    if not difference.empty:
        summary["stress_onset_ri_difference"] = float(difference.iloc[0])

    return summary


def _convergence_summary(features: pd.DataFrame) -> dict[str, float | int]:
    summary = summarize_graph_lasso_convergence(features)
    output: dict[str, float | int] = {
        "n_windows": summary["n_windows"],
        "n_converged": summary["n_converged"],
        "n_non_converged": summary["n_non_converged"],
        "convergence_rate": summary["convergence_rate"],
        "graph_lasso_warning_count_total": np.nan,
        "graph_lasso_n_iter_mean": np.nan,
        "graph_lasso_n_iter_max": np.nan,
    }
    if "graph_lasso_warning_count" in features:
        output["graph_lasso_warning_count_total"] = float(
            pd.to_numeric(features["graph_lasso_warning_count"], errors="coerce").sum(),
        )
    if "graph_lasso_n_iter" in features:
        n_iter = pd.to_numeric(features["graph_lasso_n_iter"], errors="coerce")
        output["graph_lasso_n_iter_mean"] = float(n_iter.mean())
        output["graph_lasso_n_iter_max"] = float(n_iter.max())
    return output


def _indicator_frame(indicator: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(indicator, pd.Series):
        return indicator.rename("regime_indicator").to_frame()
    if not isinstance(indicator, pd.DataFrame):
        raise TypeError("indicator must be a pandas DataFrame or Series.")
    if "regime_indicator" not in indicator.columns:
        raise KeyError("indicator is missing regime_indicator.")
    return indicator.copy()


def _convergence_series(
    indicator_frame: pd.DataFrame,
    convergence: pd.Series | pd.DataFrame | None,
) -> pd.Series | None:
    if isinstance(convergence, pd.Series):
        return convergence
    if isinstance(convergence, pd.DataFrame) and "graph_lasso_converged" in convergence:
        return convergence["graph_lasso_converged"]
    if "graph_lasso_converged" in indicator_frame:
        return indicator_frame["graph_lasso_converged"]
    return None


def _convergence_frame(
    indicator_frame: pd.DataFrame,
    convergence: pd.Series | pd.DataFrame | None,
) -> pd.DataFrame:
    if isinstance(convergence, pd.DataFrame):
        return convergence
    if isinstance(convergence, pd.Series):
        return convergence.rename("graph_lasso_converged").to_frame()
    columns = [
        column
        for column in indicator_frame.columns
        if column.startswith("graph_lasso_")
    ]
    return (
        indicator_frame[columns].copy()
        if columns
        else pd.DataFrame(index=indicator_frame.index)
    )


def _phase6_target_columns(
    targets: pd.DataFrame, horizons: tuple[int, ...]
) -> list[str]:
    columns: list[str] = []
    for horizon in horizons:
        columns.extend(
            [
                f"forward_realized_volatility_{horizon}d",
                f"forward_max_drawdown_{horizon}d",
            ],
        )
    columns.append("stress_onset_label")
    return [column for column in columns if column in targets.columns]


def _fit_ols(
    y: pd.Series,
    x: pd.DataFrame,
    newey_west_lags: int,
) -> dict[str, object]:
    y_arr = y.to_numpy(dtype=float)
    x_arr = x.to_numpy(dtype=float)
    n_obs = len(y_arr)
    x_design = np.column_stack([np.ones(n_obs), x_arr])
    names = ["intercept"] + list(x.columns)

    if n_obs <= x_design.shape[1] or _is_constant(y):
        return {
            "coefficients": pd.Series(np.nan, index=names),
            "t_stats": pd.Series(np.nan, index=names),
            "adjusted_r2": np.nan,
            "fitted": pd.Series(np.nan, index=y.index),
        }

    beta = np.linalg.pinv(x_design) @ y_arr
    fitted = x_design @ beta
    residuals = y_arr - fitted
    r2 = _r_squared(y_arr, fitted)
    k = x_design.shape[1] - 1
    adjusted_r2 = (
        1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - k - 1)
        if np.isfinite(r2) and n_obs > k + 1
        else np.nan
    )
    t_stats = _newey_west_t_stats(x_design, residuals, beta, lags=newey_west_lags)

    return {
        "coefficients": pd.Series(beta, index=names),
        "t_stats": pd.Series(t_stats, index=names),
        "adjusted_r2": float(adjusted_r2),
        "fitted": pd.Series(fitted, index=y.index),
    }


def _oos_r2(
    y: pd.Series,
    x: pd.DataFrame,
    columns: list[str],
    sample_labels: pd.Series,
) -> float:
    train_mask = sample_labels.isin(["development", "validation"]).to_numpy()
    test_mask = (sample_labels == "test").to_numpy()
    if train_mask.sum() <= len(columns) + 1 or test_mask.sum() < 3:
        return np.nan

    train_y = y.iloc[train_mask]
    train_x = x.iloc[train_mask][columns]
    test_y = y.iloc[test_mask]
    test_x = x.iloc[test_mask][columns]
    fit = _fit_ols(train_y, train_x, newey_west_lags=5)
    coefficients = fit["coefficients"]
    if coefficients.isna().all():
        return np.nan
    x_design = np.column_stack([np.ones(len(test_x)), test_x.to_numpy(dtype=float)])
    beta = coefficients.reindex(["intercept"] + columns).to_numpy(dtype=float)
    predictions = x_design @ beta
    sse = float(np.square(test_y.to_numpy(dtype=float) - predictions).sum())
    baseline_sse = float(np.square(test_y.to_numpy(dtype=float) - train_y.mean()).sum())
    return float(1.0 - sse / baseline_sse) if baseline_sse > 0 else np.nan


def _oos_classification_metrics(
    y: pd.Series,
    x: pd.DataFrame,
    columns: list[str],
    sample_labels: pd.Series,
) -> tuple[float, float]:
    train_mask = sample_labels.isin(["development", "validation"]).to_numpy()
    test_mask = (sample_labels == "test").to_numpy()
    if train_mask.sum() <= len(columns) + 1 or test_mask.sum() < 3:
        return np.nan, np.nan

    fit = _fit_ols(y.iloc[train_mask], x.iloc[train_mask][columns], newey_west_lags=5)
    coefficients = fit["coefficients"]
    if coefficients.isna().all():
        return np.nan, np.nan
    test_x = x.iloc[test_mask][columns]
    x_design = np.column_stack([np.ones(len(test_x)), test_x.to_numpy(dtype=float)])
    beta = coefficients.reindex(["intercept"] + columns).to_numpy(dtype=float)
    predictions = pd.Series(x_design @ beta, index=test_x.index)
    test_y = y.iloc[test_mask]
    return _auc_score(test_y, predictions), _brier_score(test_y, predictions)


def _newey_west_t_stats(
    x_design: np.ndarray,
    residuals: np.ndarray,
    beta: np.ndarray,
    lags: int,
) -> np.ndarray:
    n_obs, n_params = x_design.shape
    if n_obs <= n_params:
        return np.full(n_params, np.nan)

    max_lag = max(0, min(int(lags), n_obs - 1))
    xtx_inv = np.linalg.pinv(x_design.T @ x_design)
    weighted = x_design * residuals[:, None]
    s_matrix = weighted.T @ weighted
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = weighted[lag:].T @ weighted[:-lag]
        s_matrix += weight * (gamma + gamma.T)

    covariance = xtx_inv @ s_matrix @ xtx_inv
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(standard_errors > 0, beta / standard_errors, np.nan)


def _r_squared(y: np.ndarray, fitted: np.ndarray) -> float:
    sst = float(np.square(y - y.mean()).sum())
    if sst <= 0:
        return np.nan
    sse = float(np.square(y - fitted).sum())
    return float(1.0 - sse / sst)


def _auc_score(y: pd.Series, scores: pd.Series) -> float:
    y_binary = pd.to_numeric(y, errors="coerce").astype(float)
    score_values = pd.to_numeric(scores, errors="coerce").astype(float)
    pair = pd.concat(
        [y_binary.rename("y"), score_values.rename("score")], axis=1
    ).dropna()
    positives = pair["y"] == 1
    negatives = pair["y"] == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pair["score"].rank(method="average")
    rank_sum_pos = float(ranks.loc[positives].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _brier_score(y: pd.Series, scores: pd.Series) -> float:
    pair = pd.concat(
        [
            pd.to_numeric(y, errors="coerce").rename("y"),
            pd.to_numeric(scores, errors="coerce").clip(0.0, 1.0).rename("score"),
        ],
        axis=1,
    ).dropna()
    if pair.empty:
        return np.nan
    return float(np.square(pair["score"] - pair["y"]).mean())


def _empty_incremental_row(
    target: str, model: str, regressor: str
) -> dict[str, object]:
    return {
        "target": target,
        "model": model,
        "regressor": regressor,
        "n_obs": 0.0,
        "adjusted_r2": np.nan,
        "oos_r2": np.nan,
        "coefficient": np.nan,
        "coefficient_sign": "missing",
        "t_stat": np.nan,
        "auc": np.nan,
        "brier_score": np.nan,
        "oos_auc": np.nan,
        "oos_brier_score": np.nan,
    }


def _coefficient_sign(coefficient: float) -> str:
    if not np.isfinite(coefficient) or np.isclose(coefficient, 0.0):
        return "zero_or_unstable"
    return "positive" if coefficient > 0 else "negative"


def _safe_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    mean = numeric.mean()
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std <= 0:
        return pd.Series(0.0, index=values.index)
    return (numeric - mean) / std


def _single_strategy_row(group: pd.DataFrame, strategy: str) -> pd.Series | None:
    rows = group.loc[group["strategy"] == strategy]
    if rows.empty:
        return None
    return rows.iloc[0]


def _config_fields(config: Phase6Config) -> dict[str, object]:
    return asdict(config)


def _empty_turnover() -> pd.Series:
    return pd.Series(
        {
            "average_exposure": np.nan,
            "time_in_reduced_exposure": np.nan,
            "number_of_exposure_changes": np.nan,
            "annualized_turnover_proxy": np.nan,
            "mean_absolute_exposure_change": np.nan,
        },
    )


def _is_constant(values: pd.Series) -> bool:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return len(numeric) < 2 or bool(np.isclose(numeric.std(ddof=0), 0.0))


def _validate_grid_inputs(
    alpha_values: tuple[float, ...],
    windows: tuple[int, ...],
    partial_corr_thresholds: tuple[float, ...],
) -> None:
    if not alpha_values or any(alpha <= 0 for alpha in alpha_values):
        raise ValueError("alpha_values must contain positive values.")
    if not windows or any(window <= 1 for window in windows):
        raise ValueError("windows must contain values greater than one.")
    if not partial_corr_thresholds or any(
        threshold < 0 for threshold in partial_corr_thresholds
    ):
        raise ValueError("partial_corr_thresholds must contain non-negative values.")
