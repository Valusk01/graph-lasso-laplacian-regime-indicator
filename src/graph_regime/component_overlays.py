"""Research-only component-based risk-overlay comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from graph_regime.component_scores import add_component_scores
from graph_regime.phase6 import (
    apply_transaction_costs,
    compute_exposure_turnover,
)
from graph_regime.risk_overlay import (
    apply_risk_overlay,
    compute_exposure_from_indicator,
    compute_performance_metrics,
    compute_portfolio_returns,
)

DEFAULT_OVERLAY_SIGNALS = [
    "regime_indicator",
    "average_graph_strength",
    "algebraic_connectivity",
    "laplacian_frobenius_change",
    "largest_laplacian_eigenvalue_share",
    "connectivity_score",
    "transition_score",
    "spectral_score",
    "graph_components_equal_weight_score",
    "pca_first_eigenvalue_share",
    "pca_first_eigenvalue_change",
]


def build_component_overlay_signals(
    graph_features: pd.DataFrame,
    pca_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble graph-component and PCA signals for overlay evaluation."""

    if not isinstance(graph_features, pd.DataFrame):
        raise TypeError("graph_features must be a pandas DataFrame.")

    scored = add_component_scores(graph_features)
    if pca_features is not None:
        if not isinstance(pca_features, pd.DataFrame):
            raise TypeError("pca_features must be a pandas DataFrame.")
        scored = scored.join(pca_features, how="left", rsuffix="_pca")

    available = [
        column for column in DEFAULT_OVERLAY_SIGNALS if column in scored.columns
    ]
    if not available:
        raise ValueError("No supported component overlay signals are available.")

    signals = scored[available].apply(pd.to_numeric, errors="coerce")
    return signals.replace([np.inf, -np.inf], np.nan)


def compute_component_overlay_exposures(
    signals: pd.DataFrame,
    high_quantile: float = 0.8,
    extreme_quantile: float = 0.9,
    normal_exposure: float = 1.0,
    high_exposure: float = 0.7,
    extreme_exposure: float = 0.5,
    method: str = "expanding",
    min_history: int = 20,
) -> pd.DataFrame:
    """Convert component/PCA signals into shifted exposure series."""

    if not isinstance(signals, pd.DataFrame):
        raise TypeError("signals must be a pandas DataFrame.")
    if signals.empty:
        raise ValueError("signals must not be empty.")

    exposures = {}
    for column in signals.columns:
        signal = pd.to_numeric(signals[column], errors="coerce")
        if signal.dropna().empty:
            continue
        exposures[f"{column}_exposure"] = compute_exposure_from_indicator(
            signal.rename(column),
            high_quantile=high_quantile,
            extreme_quantile=extreme_quantile,
            normal_exposure=normal_exposure,
            high_exposure=high_exposure,
            extreme_exposure=extreme_exposure,
            method=method,
            min_history=min_history,
        )

    if not exposures:
        raise ValueError("No finite overlay signals are available.")
    return pd.DataFrame(exposures).sort_index()


def evaluate_component_overlays(
    returns: pd.DataFrame | pd.Series,
    signals: pd.DataFrame,
    cost_bps_values: tuple[float, ...] = (0.0, 5.0, 10.0),
    method: str = "expanding",
    min_history: int = 20,
    trading_days: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate component overlays and transaction-cost sensitivity."""

    if isinstance(returns, pd.DataFrame):
        base_returns = compute_portfolio_returns(returns)
    elif isinstance(returns, pd.Series):
        base_returns = pd.to_numeric(returns, errors="coerce").rename(
            "portfolio_returns"
        )
    else:
        raise TypeError("returns must be a pandas Series or DataFrame.")

    exposures = compute_component_overlay_exposures(
        signals,
        method=method,
        min_history=min_history,
    )
    comparison_index = base_returns.index.intersection(exposures.index)
    comparison_base_returns = base_returns.reindex(comparison_index).dropna()
    if comparison_base_returns.empty:
        raise ValueError("returns and overlay signals do not share any valid dates.")

    baseline = compute_performance_metrics(
        comparison_base_returns,
        trading_days=trading_days,
    )
    comparison_rows: list[dict[str, object]] = [
        {"signal": "baseline", "strategy": "baseline", **baseline.to_dict()}
    ]
    cost_rows: list[dict[str, object]] = []

    for exposure_column in exposures.columns:
        signal = exposure_column.removesuffix("_exposure")
        exposure = exposures[exposure_column].reindex(comparison_index)
        overlay_returns = apply_risk_overlay(
            comparison_base_returns,
            exposure,
        ).rename(signal)
        metrics = compute_performance_metrics(
            overlay_returns, trading_days=trading_days
        )
        turnover = compute_exposure_turnover(exposure, trading_days=trading_days)
        row = {
            "signal": signal,
            "strategy": f"{signal}_overlay",
            **metrics.to_dict(),
            **turnover.to_dict(),
        }
        comparison_rows.append(row)

        for cost_bps in cost_bps_values:
            cost_adjusted = apply_transaction_costs(
                overlay_returns,
                exposure=exposure,
                cost_bps=float(cost_bps),
            )
            cost_metrics = compute_performance_metrics(
                cost_adjusted,
                trading_days=trading_days,
            )
            cost_rows.append(
                {
                    "signal": signal,
                    "cost_bps": float(cost_bps),
                    **cost_metrics.to_dict(),
                    **turnover.to_dict(),
                },
            )

    comparison = pd.DataFrame(comparison_rows)
    comparison = _add_baseline_differences(comparison)
    transaction_costs = pd.DataFrame(cost_rows)
    return comparison, transaction_costs


def _add_baseline_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    baseline = output.loc[output["strategy"] == "baseline"]
    if baseline.empty:
        return output
    baseline_row = baseline.iloc[0]
    for column in [
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "worst_5pct_return",
    ]:
        if column in output:
            output[f"{column}_minus_baseline"] = output[column] - baseline_row[column]
    return output
