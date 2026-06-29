"""Phase 7 PCA baselines, component model ladders, and ablations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from graph_regime.component_scores import GRAPH_COMPONENT_COLUMNS
from graph_regime.evaluation import compute_forward_targets
from graph_regime.phase6 import (
    DEFAULT_SAMPLE_SPLITS,
    SampleSplits,
    assign_sample_period,
    compute_exposure_turnover,
)
from graph_regime.risk_overlay import (
    apply_risk_overlay,
    compute_exposure_from_indicator,
    compute_performance_metrics,
    compute_portfolio_returns,
)
from graph_regime.robustness import create_stress_onset_labels

BASELINE_COLUMNS = [
    "vix",
    "realized_volatility",
    "drawdown",
    "average_correlation",
    "average_absolute_correlation",
]

ORTHOGONAL_COLUMNS = [
    "orthogonal_average_graph_strength",
    "orthogonal_algebraic_connectivity",
    "orthogonal_laplacian_frobenius_change",
    "orthogonal_largest_eigenvalue_share",
    "orthogonal_graph_score",
]


def run_model_ladder_incremental_tests(
    graph_features: pd.DataFrame,
    benchmarks: pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    pca_features: pd.DataFrame | None = None,
    component_scores: pd.DataFrame | None = None,
    splits: SampleSplits = DEFAULT_SAMPLE_SPLITS,
    horizons: tuple[int, ...] = (5, 21),
    newey_west_lags: int = 5,
) -> pd.DataFrame:
    """Run Phase 7 model ladder tests for benchmarks, PCA, RI, and components."""

    if not isinstance(graph_features, pd.DataFrame):
        raise TypeError("graph_features must be a pandas DataFrame.")
    if not isinstance(benchmarks, pd.DataFrame):
        raise TypeError("benchmarks must be a pandas DataFrame.")
    if component_scores is None:
        component_scores = graph_features
    if not isinstance(component_scores, pd.DataFrame):
        raise TypeError("component_scores must be a pandas DataFrame.")

    regressors = _build_regressor_frame(
        graph_features=graph_features,
        benchmarks=benchmarks,
        pca_features=pca_features,
        component_scores=component_scores,
    )
    model_specs = _model_specs(regressors)
    targets = _build_targets(returns, benchmarks, horizons=horizons)
    rows: list[dict[str, object]] = []

    for target_column in _target_columns(horizons):
        if target_column not in targets:
            continue
        y = pd.to_numeric(targets[target_column], errors="coerce").rename(target_column)
        for model_name, columns in model_specs.items():
            if not columns:
                continue
            model_frame = (
                pd.concat([y, regressors[columns]], axis=1, join="inner")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if model_frame.empty:
                continue
            fit = _fit_ols(
                model_frame[target_column],
                model_frame[columns],
                newey_west_lags=newey_west_lags,
            )
            sample_labels = assign_sample_period(model_frame.index, splits)
            oos = _oos_metrics(
                model_frame[target_column],
                model_frame[columns],
                sample_labels=sample_labels,
                classification=target_column == "stress_onset_label",
                newey_west_lags=newey_west_lags,
            )
            classification = (
                _classification_metrics(model_frame[target_column], fit["fitted"])
                if target_column == "stress_onset_label"
                else {}
            )

            for regressor in columns:
                coefficient = fit["coefficients"].get(regressor, np.nan)
                rows.append(
                    {
                        "target": target_column,
                        "model": model_name,
                        "regressor": regressor,
                        "n_obs": float(len(model_frame)),
                        "adjusted_r2": fit["adjusted_r2"],
                        "oos_r2": oos["oos_r2"],
                        "rmse": fit["rmse"],
                        "mae": fit["mae"],
                        "oos_rmse": oos["oos_rmse"],
                        "oos_mae": oos["oos_mae"],
                        "coefficient": coefficient,
                        "coefficient_sign": _coefficient_sign(coefficient),
                        "t_stat": fit["t_stats"].get(regressor, np.nan),
                        "auc": classification.get("auc", np.nan),
                        "oos_auc": oos["oos_auc"],
                        "brier_score": classification.get("brier_score", np.nan),
                        "oos_brier_score": oos["oos_brier_score"],
                        "precision": classification.get("precision", np.nan),
                        "recall": classification.get("recall", np.nan),
                        "f1": classification.get("f1", np.nan),
                        "oos_precision": oos["oos_precision"],
                        "oos_recall": oos["oos_recall"],
                        "oos_f1": oos["oos_f1"],
                    },
                )

    return pd.DataFrame(rows)


def run_graph_component_ablation(
    component_scores: pd.DataFrame,
    benchmarks: pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    splits: SampleSplits = DEFAULT_SAMPLE_SPLITS,
    horizon: int = 21,
    min_history: int = 20,
) -> pd.DataFrame:
    """Run graph-component ablations for prediction and overlay diagnostics."""

    if not isinstance(component_scores, pd.DataFrame):
        raise TypeError("component_scores must be a pandas DataFrame.")
    if not isinstance(benchmarks, pd.DataFrame):
        raise TypeError("benchmarks must be a pandas DataFrame.")

    targets = _build_targets(returns, benchmarks, horizons=(horizon,))
    benchmark_columns = [column for column in BASELINE_COLUMNS if column in benchmarks]
    base_returns = (
        compute_portfolio_returns(returns)
        if isinstance(returns, pd.DataFrame)
        else pd.to_numeric(returns, errors="coerce").rename("portfolio_returns")
    )
    rows: list[dict[str, object]] = []

    for ablation_name, requested_columns in _ablation_specs().items():
        columns = [column for column in requested_columns if column in component_scores]
        if not columns:
            continue
        score = _ablation_score(component_scores, columns).rename(ablation_name)
        regressors = pd.concat(
            [benchmarks[benchmark_columns], component_scores[columns]],
            axis=1,
            join="outer",
        )

        regression_metrics = _single_model_summary(
            targets[f"forward_realized_volatility_{horizon}d"],
            regressors,
            splits=splits,
            classification=False,
        )
        classification_metrics = _single_model_summary(
            targets["stress_onset_label"],
            regressors,
            splits=splits,
            classification=True,
        )
        exposure = compute_exposure_from_indicator(
            score,
            method="expanding",
            min_history=min_history,
        )
        overlay_returns = apply_risk_overlay(base_returns, exposure)
        overlay_metrics = compute_performance_metrics(overlay_returns)
        turnover = compute_exposure_turnover(exposure)

        rows.append(
            {
                "ablation": ablation_name,
                "component_count": len(columns),
                "components": ",".join(columns),
                "adjusted_r2": regression_metrics["adjusted_r2"],
                "oos_r2": regression_metrics["oos_r2"],
                "auc": classification_metrics["auc"],
                "oos_auc": classification_metrics["oos_auc"],
                "overlay_sharpe": overlay_metrics["sharpe"],
                "overlay_calmar": overlay_metrics["calmar"],
                "overlay_max_drawdown": overlay_metrics["max_drawdown"],
                "annualized_turnover_proxy": turnover["annualized_turnover_proxy"],
                "time_in_reduced_exposure": turnover["time_in_reduced_exposure"],
            },
        )

    return pd.DataFrame(rows)


def _build_regressor_frame(
    graph_features: pd.DataFrame,
    benchmarks: pd.DataFrame,
    pca_features: pd.DataFrame | None,
    component_scores: pd.DataFrame,
) -> pd.DataFrame:
    frames = [
        benchmarks[[column for column in BASELINE_COLUMNS if column in benchmarks]]
    ]
    graph_columns = [
        column
        for column in ["regime_indicator"]
        + GRAPH_COMPONENT_COLUMNS
        + ORTHOGONAL_COLUMNS
        if column in graph_features.columns or column in component_scores.columns
    ]
    graph_frame = pd.concat([graph_features, component_scores], axis=1)
    frames.append(graph_frame.loc[:, ~graph_frame.columns.duplicated()][graph_columns])
    if pca_features is not None:
        frames.append(pca_features)
    return pd.concat(frames, axis=1, join="outer").replace([np.inf, -np.inf], np.nan)


def _model_specs(regressors: pd.DataFrame) -> dict[str, list[str]]:
    benchmark_columns = [column for column in BASELINE_COLUMNS if column in regressors]
    pca_columns = [column for column in regressors.columns if column.startswith("pca_")]
    graph_columns = [
        column for column in GRAPH_COMPONENT_COLUMNS if column in regressors
    ]
    orthogonal_columns = [
        column for column in ORTHOGONAL_COLUMNS if column in regressors
    ]

    specs = {
        "M0_benchmarks": benchmark_columns,
        "M1_benchmarks_plus_ri": benchmark_columns
        + (["regime_indicator"] if "regime_indicator" in regressors else []),
        "M2_benchmarks_plus_pca": benchmark_columns + pca_columns,
        "M3_benchmarks_plus_graph_components": benchmark_columns + graph_columns,
        "M4_benchmarks_plus_pca_plus_graph_components": benchmark_columns
        + pca_columns
        + graph_columns,
    }
    if orthogonal_columns:
        specs["M5_benchmarks_plus_orthogonal_graph"] = (
            benchmark_columns + orthogonal_columns
        )
    return {name: _dedupe(columns) for name, columns in specs.items()}


def _build_targets(
    returns: pd.Series | pd.DataFrame,
    benchmarks: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    forward = compute_forward_targets(returns, horizons=list(horizons))
    onset = create_stress_onset_labels(benchmarks, min_gap=20).rename(
        "stress_onset_label"
    )
    return pd.concat([forward, onset], axis=1, join="outer")


def _target_columns(horizons: tuple[int, ...]) -> list[str]:
    columns: list[str] = []
    for horizon in horizons:
        columns.extend(
            [
                f"forward_realized_volatility_{horizon}d",
                f"forward_max_drawdown_{horizon}d",
            ],
        )
    columns.append("stress_onset_label")
    return columns


def _single_model_summary(
    y: pd.Series,
    x: pd.DataFrame,
    splits: SampleSplits,
    classification: bool,
) -> dict[str, float]:
    frame = (
        pd.concat([pd.to_numeric(y, errors="coerce").rename("target"), x], axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if frame.empty:
        return _empty_summary(classification=classification)
    columns = list(frame.columns.drop("target"))
    fit = _fit_ols(frame["target"], frame[columns], newey_west_lags=5)
    sample_labels = assign_sample_period(frame.index, splits)
    oos = _oos_metrics(
        frame["target"],
        frame[columns],
        sample_labels=sample_labels,
        classification=classification,
        newey_west_lags=5,
    )
    if classification:
        cls = _classification_metrics(frame["target"], fit["fitted"])
        return {"auc": cls["auc"], "oos_auc": oos["oos_auc"]}
    return {"adjusted_r2": fit["adjusted_r2"], "oos_r2": oos["oos_r2"]}


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
        fitted = pd.Series(np.nan, index=y.index)
        return {
            "coefficients": pd.Series(np.nan, index=names),
            "t_stats": pd.Series(np.nan, index=names),
            "adjusted_r2": np.nan,
            "fitted": fitted,
            "rmse": np.nan,
            "mae": np.nan,
        }

    beta = np.linalg.pinv(x_design) @ y_arr
    fitted_values = x_design @ beta
    residuals = y_arr - fitted_values
    r2 = _r_squared(y_arr, fitted_values)
    k = x_design.shape[1] - 1
    adjusted_r2 = (
        1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - k - 1)
        if np.isfinite(r2) and n_obs > k + 1
        else np.nan
    )
    fitted = pd.Series(fitted_values, index=y.index)
    return {
        "coefficients": pd.Series(beta, index=names),
        "t_stats": pd.Series(
            _newey_west_t_stats(x_design, residuals, beta, newey_west_lags),
            index=names,
        ),
        "adjusted_r2": float(adjusted_r2),
        "fitted": fitted,
        "rmse": _rmse(y_arr, fitted_values),
        "mae": _mae(y_arr, fitted_values),
    }


def _oos_metrics(
    y: pd.Series,
    x: pd.DataFrame,
    sample_labels: pd.Series,
    classification: bool,
    newey_west_lags: int,
) -> dict[str, float]:
    train_mask = sample_labels.isin(["development", "validation"]).to_numpy()
    test_mask = (sample_labels == "test").to_numpy()
    output = {
        "oos_r2": np.nan,
        "oos_rmse": np.nan,
        "oos_mae": np.nan,
        "oos_auc": np.nan,
        "oos_brier_score": np.nan,
        "oos_precision": np.nan,
        "oos_recall": np.nan,
        "oos_f1": np.nan,
    }
    if train_mask.sum() <= x.shape[1] + 1 or test_mask.sum() < 3:
        return output

    columns = list(x.columns)
    train_y = y.iloc[train_mask]
    train_x = x.iloc[train_mask][columns]
    test_y = y.iloc[test_mask]
    test_x = x.iloc[test_mask][columns]
    fit = _fit_ols(train_y, train_x, newey_west_lags=newey_west_lags)
    coefficients = fit["coefficients"].reindex(["intercept"] + columns)
    if coefficients.isna().all():
        return output
    x_design = np.column_stack([np.ones(len(test_x)), test_x.to_numpy(dtype=float)])
    predictions = pd.Series(
        x_design @ coefficients.to_numpy(dtype=float),
        index=test_x.index,
    )
    output["oos_rmse"] = _rmse(
        test_y.to_numpy(dtype=float), predictions.to_numpy(dtype=float)
    )
    output["oos_mae"] = _mae(
        test_y.to_numpy(dtype=float), predictions.to_numpy(dtype=float)
    )
    if classification:
        output.update(
            {
                f"oos_{k}": v
                for k, v in _classification_metrics(test_y, predictions).items()
            }
        )
    else:
        baseline_sse = float(
            np.square(test_y.to_numpy(dtype=float) - train_y.mean()).sum(),
        )
        sse = float(np.square(test_y.to_numpy(dtype=float) - predictions).sum())
        output["oos_r2"] = 1.0 - sse / baseline_sse if baseline_sse > 0 else np.nan
    return output


def _classification_metrics(y: pd.Series, scores: pd.Series) -> dict[str, float]:
    pair = pd.concat(
        [
            pd.to_numeric(y, errors="coerce").rename("y"),
            pd.to_numeric(scores, errors="coerce").rename("score"),
        ],
        axis=1,
    ).dropna()
    if pair.empty:
        return {
            "auc": np.nan,
            "brier_score": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }
    clipped = pair["score"].clip(0.0, 1.0)
    prediction = (clipped >= 0.5).astype(int)
    y_binary = pair["y"].astype(int)
    tp = float(((prediction == 1) & (y_binary == 1)).sum())
    fp = float(((prediction == 1) & (y_binary == 0)).sum())
    fn = float(((prediction == 0) & (y_binary == 1)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else np.nan
    )
    return {
        "auc": _auc(y_binary, pair["score"]),
        "brier_score": float(np.square(clipped - y_binary).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _auc(y: pd.Series, scores: pd.Series) -> float:
    pair = pd.concat([y.rename("y"), scores.rename("score")], axis=1).dropna()
    positives = pair["y"] == 1
    negatives = pair["y"] == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pair["score"].rank(method="average")
    rank_sum = float(ranks.loc[positives].sum())
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


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


def _ablation_specs() -> dict[str, list[str]]:
    return {
        "all_graph_components": GRAPH_COMPONENT_COLUMNS,
        "excluding_laplacian_frobenius_change": [
            column
            for column in GRAPH_COMPONENT_COLUMNS
            if column != "laplacian_frobenius_change"
        ],
        "excluding_algebraic_connectivity": [
            column
            for column in GRAPH_COMPONENT_COLUMNS
            if column != "algebraic_connectivity"
        ],
        "excluding_average_graph_strength": [
            column
            for column in GRAPH_COMPONENT_COLUMNS
            if column != "average_graph_strength"
        ],
        "excluding_largest_eigenvalue_share": [
            column
            for column in GRAPH_COMPONENT_COLUMNS
            if column != "largest_laplacian_eigenvalue_share"
        ],
        "connectivity_only": ["connectivity_score"],
        "transition_only": ["transition_score"],
        "spectral_only": ["spectral_score"],
    }


def _ablation_score(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    z_frame = pd.DataFrame({column: _safe_z(frame[column]) for column in columns})
    return z_frame.mean(axis=1, skipna=True).fillna(0.0)


def _safe_z(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = numeric.mean()
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std <= 0:
        return pd.Series(0.0, index=series.index)
    return ((numeric - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _r_squared(y: np.ndarray, fitted: np.ndarray) -> float:
    sst = float(np.square(y - y.mean()).sum())
    return float(1.0 - np.square(y - fitted).sum() / sst) if sst > 0 else np.nan


def _rmse(y: np.ndarray, fitted: np.ndarray) -> float:
    return float(np.sqrt(np.square(y - fitted).mean())) if len(y) else np.nan


def _mae(y: np.ndarray, fitted: np.ndarray) -> float:
    return float(np.abs(y - fitted).mean()) if len(y) else np.nan


def _is_constant(values: pd.Series) -> bool:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return len(numeric) < 2 or bool(np.isclose(numeric.std(ddof=0), 0.0))


def _coefficient_sign(coefficient: float) -> str:
    if not np.isfinite(coefficient) or np.isclose(coefficient, 0.0):
        return "zero_or_unstable"
    return "positive" if coefficient > 0 else "negative"


def _dedupe(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _empty_summary(classification: bool) -> dict[str, float]:
    if classification:
        return {"auc": np.nan, "oos_auc": np.nan}
    return {"adjusted_r2": np.nan, "oos_r2": np.nan}
