"""Research-only risk-overlay utilities for regime-indicator evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd


ExposureMethod = Literal["expanding", "full_sample"]


def compute_exposure_from_indicator(
    indicator: pd.Series,
    high_quantile: float = 0.8,
    extreme_quantile: float = 0.9,
    normal_exposure: float = 1.0,
    high_exposure: float = 0.7,
    extreme_exposure: float = 0.5,
    method: ExposureMethod = "expanding",
    min_history: int = 20,
) -> pd.Series:
    """Convert a risk indicator into a one-period-lagged exposure series.

    This is a research-only risk overlay. Exposure reductions are triggered by
    high indicator quantiles. The returned exposure is shifted by one
    observation so return at ``t`` uses only information available through
    ``t-1``. ``method="expanding"`` avoids look-ahead by estimating quantiles
    from prior history; ``method="full_sample"`` is for diagnostics only.
    """

    _validate_exposure_inputs(
        high_quantile=high_quantile,
        extreme_quantile=extreme_quantile,
        normal_exposure=normal_exposure,
        high_exposure=high_exposure,
        extreme_exposure=extreme_exposure,
        method=method,
        min_history=min_history,
    )
    if not isinstance(indicator, pd.Series):
        raise TypeError("indicator must be a pandas Series.")

    signal = pd.to_numeric(indicator, errors="coerce").replace([np.inf, -np.inf], np.nan)
    raw_exposure = pd.Series(normal_exposure, index=signal.index, dtype=float)

    if signal.dropna().empty:
        return raw_exposure.rename("exposure")

    if method == "full_sample":
        high_threshold = signal.quantile(high_quantile)
        extreme_threshold = signal.quantile(extreme_quantile)
        high_mask = signal >= high_threshold
        extreme_mask = signal >= extreme_threshold
    else:
        expanding = signal.expanding(min_periods=1)
        history_count = expanding.count().shift(1)
        high_threshold = expanding.quantile(high_quantile).shift(1)
        extreme_threshold = expanding.quantile(extreme_quantile).shift(1)
        enough_history = history_count >= min_history
        high_mask = (signal >= high_threshold) & enough_history
        extreme_mask = (signal >= extreme_threshold) & enough_history

    raw_exposure.loc[high_mask.fillna(False)] = high_exposure
    raw_exposure.loc[extreme_mask.fillna(False)] = extreme_exposure

    shifted = raw_exposure.shift(1).fillna(normal_exposure)
    return shifted.rename("exposure")


def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: str | pd.Series = "equal_weight",
) -> pd.Series:
    """Compute portfolio returns from an asset-return panel.

    ``weights="equal_weight"`` computes a simple equal-weight portfolio across
    available assets. A weight Series is aligned to return columns and
    normalized to sum to one.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns must not be empty.")

    numeric_returns = returns.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if isinstance(weights, str) and weights == "equal_weight":
        portfolio = numeric_returns.mean(axis=1, skipna=True)
    elif isinstance(weights, pd.Series):
        numeric_weights = pd.to_numeric(weights, errors="coerce").reindex(
            numeric_returns.columns,
        ).fillna(0.0)
        weight_sum = numeric_weights.sum()
        if not np.isfinite(weight_sum) or np.isclose(weight_sum, 0.0):
            raise ValueError("weights must contain at least one non-zero finite value.")
        normalized_weights = numeric_weights / weight_sum
        portfolio = numeric_returns.mul(normalized_weights, axis=1).sum(
            axis=1,
            min_count=1,
        )
    elif isinstance(weights, str):
        raise ValueError("weights must be 'equal_weight' or a pandas Series.")
    else:
        raise ValueError("weights must be 'equal_weight' or a pandas Series.")

    return portfolio.replace([np.inf, -np.inf], np.nan).rename("portfolio_returns")


def apply_risk_overlay(
    portfolio_returns: pd.Series,
    exposure: pd.Series,
) -> pd.Series:
    """Apply an exposure series to portfolio returns."""

    if not isinstance(portfolio_returns, pd.Series):
        raise TypeError("portfolio_returns must be a pandas Series.")
    if not isinstance(exposure, pd.Series):
        raise TypeError("exposure must be a pandas Series.")

    aligned = pd.concat(
        [
            pd.to_numeric(portfolio_returns, errors="coerce").rename("returns"),
            pd.to_numeric(exposure, errors="coerce").rename("exposure"),
        ],
        axis=1,
        join="inner",
    ).replace([np.inf, -np.inf], np.nan)
    aligned = aligned.dropna(subset=["returns"])
    aligned["exposure"] = aligned["exposure"].fillna(1.0)

    return (aligned["returns"] * aligned["exposure"]).rename("overlay_returns")


def compute_performance_metrics(
    returns: pd.Series,
    trading_days: int = 252,
) -> pd.Series:
    """Compute simple performance and downside metrics for return series."""

    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")
    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")

    numeric_returns = pd.to_numeric(returns, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    if numeric_returns.empty:
        return pd.Series(_empty_metrics(), name="performance")

    wealth = _wealth_index(numeric_returns)
    annualized_return = _annualized_return(wealth, len(numeric_returns), trading_days)
    annualized_volatility = float(numeric_returns.std(ddof=1) * np.sqrt(trading_days))
    sharpe = (
        annualized_return / annualized_volatility
        if annualized_volatility > 0 and np.isfinite(annualized_return)
        else np.nan
    )

    downside = numeric_returns.loc[numeric_returns < 0.0]
    downside_volatility = float(downside.std(ddof=1) * np.sqrt(trading_days))
    sortino = (
        annualized_return / downside_volatility
        if downside_volatility > 0 and np.isfinite(annualized_return)
        else np.nan
    )

    max_drawdown = _max_drawdown_from_wealth(wealth)
    calmar = (
        annualized_return / abs(max_drawdown)
        if max_drawdown < 0 and np.isfinite(annualized_return)
        else np.nan
    )

    metrics = {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "mean_return": float(numeric_returns.mean()),
        "mean_absolute_return": float(numeric_returns.abs().mean()),
        "worst_1pct_return": float(numeric_returns.quantile(0.01)),
        "worst_5pct_return": float(numeric_returns.quantile(0.05)),
        "hit_rate": float((numeric_returns > 0.0).mean()),
        "n_obs": float(len(numeric_returns)),
    }
    return pd.Series(metrics, name="performance")


def compare_overlay_to_baseline(
    base_returns: pd.Series,
    overlay_returns: pd.Series,
) -> pd.DataFrame:
    """Compare baseline and overlay performance metrics."""

    baseline = compute_performance_metrics(base_returns).rename("baseline")
    overlay = compute_performance_metrics(overlay_returns).rename("overlay")
    comparison = pd.concat([baseline, overlay], axis=1)
    comparison["overlay_minus_baseline"] = comparison["overlay"] - comparison["baseline"]
    comparison.index.name = "metric"
    return comparison


def compute_benchmark_exposures(
    benchmarks: pd.DataFrame,
    benchmark_columns: Iterable[str] | None = None,
    high_quantile: float = 0.8,
    extreme_quantile: float = 0.9,
    normal_exposure: float = 1.0,
    high_exposure: float = 0.7,
    extreme_exposure: float = 0.5,
    method: ExposureMethod = "expanding",
    min_history: int = 20,
) -> pd.DataFrame:
    """Build benchmark risk-overlay exposures using the same quantile rules.

    Higher values are treated as higher risk for most benchmark variables. For
    drawdown, the risk score is ``-drawdown`` because deeper drawdowns are more
    negative.
    """

    if not isinstance(benchmarks, pd.DataFrame):
        raise TypeError("benchmarks must be a pandas DataFrame.")

    if benchmark_columns is None:
        benchmark_columns = [
            "vix",
            "realized_volatility",
            "average_correlation",
            "average_absolute_correlation",
            "drawdown",
        ]

    exposures: dict[str, pd.Series] = {}
    for column in benchmark_columns:
        if column not in benchmarks.columns:
            continue
        risk_score = pd.to_numeric(benchmarks[column], errors="coerce")
        if column == "drawdown":
            risk_score = -risk_score
        exposures[f"{column}_exposure"] = compute_exposure_from_indicator(
            risk_score.rename(column),
            high_quantile=high_quantile,
            extreme_quantile=extreme_quantile,
            normal_exposure=normal_exposure,
            high_exposure=high_exposure,
            extreme_exposure=extreme_exposure,
            method=method,
            min_history=min_history,
        )

    if not exposures:
        raise ValueError("No requested benchmark columns are available.")

    return pd.DataFrame(exposures).sort_index()


def summarize_exposures(exposures: pd.DataFrame) -> pd.DataFrame:
    """Summarize exposure series for research diagnostics."""

    if not isinstance(exposures, pd.DataFrame):
        raise TypeError("exposures must be a pandas DataFrame.")

    rows: list[dict[str, float | str]] = []
    for column in exposures.columns:
        exposure = pd.to_numeric(exposures[column], errors="coerce").dropna()
        if exposure.empty:
            rows.append(
                {
                    "exposure": column,
                    "n_obs": 0.0,
                    "mean": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "share_reduced": np.nan,
                    "share_half_or_less": np.nan,
                },
            )
            continue

        rows.append(
            {
                "exposure": column,
                "n_obs": float(len(exposure)),
                "mean": float(exposure.mean()),
                "min": float(exposure.min()),
                "max": float(exposure.max()),
                "share_reduced": float((exposure < 1.0).mean()),
                "share_half_or_less": float((exposure <= 0.5).mean()),
            },
        )

    return pd.DataFrame(rows)


def _validate_exposure_inputs(
    high_quantile: float,
    extreme_quantile: float,
    normal_exposure: float,
    high_exposure: float,
    extreme_exposure: float,
    method: str,
    min_history: int,
) -> None:
    if not 0 <= high_quantile < extreme_quantile <= 1:
        raise ValueError(
            "quantiles must satisfy 0 <= high_quantile < extreme_quantile <= 1.",
        )
    if method not in {"expanding", "full_sample"}:
        raise ValueError("method must be either 'expanding' or 'full_sample'.")
    if min_history <= 0:
        raise ValueError("min_history must be positive.")
    for value, name in [
        (normal_exposure, "normal_exposure"),
        (high_exposure, "high_exposure"),
        (extreme_exposure, "extreme_exposure"),
    ]:
        if value < 0 or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite non-negative value.")
    if not normal_exposure >= high_exposure >= extreme_exposure:
        raise ValueError(
            "exposures must satisfy normal_exposure >= high_exposure >= "
            "extreme_exposure.",
        )


def _empty_metrics() -> dict[str, float]:
    return {
        "annualized_return": np.nan,
        "annualized_volatility": np.nan,
        "sharpe": np.nan,
        "sortino": np.nan,
        "max_drawdown": np.nan,
        "calmar": np.nan,
        "mean_return": np.nan,
        "mean_absolute_return": np.nan,
        "worst_1pct_return": np.nan,
        "worst_5pct_return": np.nan,
        "hit_rate": np.nan,
        "n_obs": 0.0,
    }


def _wealth_index(returns: pd.Series) -> pd.Series:
    clipped_returns = returns.clip(lower=-0.999999)
    return (1.0 + clipped_returns).cumprod()


def _annualized_return(
    wealth: pd.Series,
    n_obs: int,
    trading_days: int,
) -> float:
    if n_obs <= 0 or wealth.empty or wealth.iloc[-1] <= 0:
        return np.nan
    return float(wealth.iloc[-1] ** (trading_days / n_obs) - 1.0)


def _max_drawdown_from_wealth(wealth: pd.Series) -> float:
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())
