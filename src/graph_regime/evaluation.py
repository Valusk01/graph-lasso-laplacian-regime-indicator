"""Empirical diagnostics for graph-regime indicator evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def align_indicator_and_benchmarks(
    indicator: pd.DataFrame | pd.Series,
    benchmarks: pd.DataFrame,
    indicator_column: str = "regime_indicator",
) -> pd.DataFrame:
    """Align the graph regime indicator with benchmark stress variables.

    An inner join gives clean contemporaneous diagnostics over dates where both
    the indicator index and benchmark index are present. Missing benchmark
    values are preserved so nullable stress labels remain informative.
    """

    if isinstance(indicator, pd.DataFrame):
        if indicator_column not in indicator.columns:
            raise KeyError(f"indicator is missing column {indicator_column!r}.")
        indicator_series = indicator[indicator_column]
    elif isinstance(indicator, pd.Series):
        indicator_series = indicator
    else:
        raise TypeError("indicator must be a pandas DataFrame or Series.")

    if not isinstance(benchmarks, pd.DataFrame):
        raise TypeError("benchmarks must be a pandas DataFrame.")

    indicator_frame = pd.to_numeric(indicator_series, errors="coerce").rename(
        "regime_indicator",
    ).to_frame()
    aligned = indicator_frame.join(benchmarks, how="inner").sort_index()
    return aligned.dropna(subset=["regime_indicator"])


def compute_contemporaneous_diagnostics(
    aligned: pd.DataFrame,
    indicator_column: str = "regime_indicator",
    stress_label_columns: list[str] | None = None,
    continuous_benchmark_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compare the regime indicator with stress labels and continuous benchmarks.

    The output is a tidy diagnostics table. Stress-label rows measure whether
    the indicator is higher during benchmark stress periods. Continuous rows
    measure Pearson and Spearman co-movement with variables such as VIX,
    drawdown, realized volatility, or average correlation.
    """

    if indicator_column not in aligned.columns:
        raise KeyError(f"aligned is missing column {indicator_column!r}.")

    if stress_label_columns is None:
        stress_label_columns = _detect_stress_label_columns(aligned)

    if continuous_benchmark_columns is None:
        excluded = set(stress_label_columns) | {indicator_column}
        continuous_benchmark_columns = [
            column
            for column in aligned.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(aligned[column])
        ]

    rows: list[dict[str, object]] = []
    indicator_values = pd.to_numeric(aligned[indicator_column], errors="coerce")

    for label_column in stress_label_columns:
        if label_column not in aligned.columns:
            continue
        label_values = pd.to_numeric(aligned[label_column], errors="coerce")
        valid = pd.concat([indicator_values, label_values], axis=1).dropna()
        valid.columns = ["indicator", "label"]

        stress_group = valid.loc[valid["label"] == 1, "indicator"]
        non_stress_group = valid.loc[valid["label"] == 0, "indicator"]

        stress_mean = _mean_or_nan(stress_group)
        non_stress_mean = _mean_or_nan(non_stress_group)
        ratio = (
            stress_mean / non_stress_mean
            if np.isfinite(non_stress_mean) and non_stress_mean != 0
            else np.nan
        )

        metrics = {
            "n_total": float(len(valid)),
            "n_stress": float(len(stress_group)),
            "n_non_stress": float(len(non_stress_group)),
            "mean_indicator_stress": stress_mean,
            "mean_indicator_non_stress": non_stress_mean,
            "median_indicator_stress": _median_or_nan(stress_group),
            "median_indicator_non_stress": _median_or_nan(non_stress_group),
            "difference_in_means": stress_mean - non_stress_mean,
            "ratio_stress_to_non_stress": ratio,
            "welch_t_statistic": _welch_t_statistic(stress_group, non_stress_group),
        }
        rows.extend(
            _tidy_rows(
                diagnostic_type="stress_label",
                benchmark=label_column,
                metrics=metrics,
            ),
        )

    for benchmark_column in continuous_benchmark_columns:
        if benchmark_column not in aligned.columns:
            continue
        benchmark_values = pd.to_numeric(aligned[benchmark_column], errors="coerce")
        pair = pd.concat([indicator_values, benchmark_values], axis=1).dropna()
        pair.columns = ["indicator", "benchmark"]
        pearson, spearman = _correlations(pair["indicator"], pair["benchmark"])

        rows.extend(
            _tidy_rows(
                diagnostic_type="continuous_benchmark",
                benchmark=benchmark_column,
                metrics={
                    "n_obs": float(len(pair)),
                    "pearson_correlation": pearson,
                    "spearman_correlation": spearman,
                },
            ),
        )

    return pd.DataFrame(rows, columns=["diagnostic_type", "benchmark", "metric", "value"])


def compute_forward_targets(
    returns: pd.Series | pd.DataFrame,
    horizons: list[int] = [5, 21, 63],
    market_return_column: str | None = None,
    annualize_volatility: bool = False,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Compute forward-looking market targets without including return at ``t``.

    For each date ``t`` and horizon ``h``, targets use returns from ``t+1``
    through ``t+h`` only. The return target is named as a sum to avoid ambiguity
    between simple and log-return compounding conventions.
    """

    if not horizons:
        raise ValueError("horizons must contain at least one positive integer.")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("all horizons must be positive integers.")
    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")

    market_returns = _market_return_series(returns, market_return_column)
    targets = pd.DataFrame(index=market_returns.index)

    for horizon in horizons:
        vol_column = f"forward_realized_volatility_{horizon}d"
        sum_column = f"forward_return_sum_{horizon}d"
        abs_sum_column = f"forward_absolute_return_sum_{horizon}d"
        drawdown_column = f"forward_max_drawdown_{horizon}d"

        volatility_values: list[float] = []
        return_sum_values: list[float] = []
        absolute_sum_values: list[float] = []
        max_drawdown_values: list[float] = []

        for position in range(len(market_returns)):
            future = market_returns.iloc[position + 1 : position + horizon + 1]
            if len(future) < horizon:
                volatility_values.append(np.nan)
                return_sum_values.append(np.nan)
                absolute_sum_values.append(np.nan)
                max_drawdown_values.append(np.nan)
                continue

            observed_future = future.dropna()
            if observed_future.empty:
                volatility_values.append(np.nan)
                return_sum_values.append(np.nan)
                absolute_sum_values.append(np.nan)
                max_drawdown_values.append(np.nan)
                continue

            volatility = observed_future.std(ddof=1)
            if annualize_volatility and np.isfinite(volatility):
                volatility = volatility * np.sqrt(trading_days)

            return_sum = observed_future.sum()
            volatility_values.append(float(volatility))
            return_sum_values.append(float(return_sum))
            absolute_sum_values.append(float(abs(return_sum)))
            max_drawdown_values.append(_forward_max_drawdown(observed_future))

        targets[vol_column] = volatility_values
        targets[sum_column] = return_sum_values
        targets[abs_sum_column] = absolute_sum_values
        targets[drawdown_column] = max_drawdown_values

    return targets


def evaluate_predictive_power(
    indicator: pd.Series,
    forward_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate whether today's indicator co-moves with future target variables.

    Diagnostics are deliberately simple univariate statistics: correlations and
    ordinary least squares of each forward target on the current indicator.
    They are useful screening tools, not causal evidence.
    """

    if not isinstance(indicator, pd.Series):
        raise TypeError("indicator must be a pandas Series.")
    if not isinstance(forward_targets, pd.DataFrame):
        raise TypeError("forward_targets must be a pandas DataFrame.")

    indicator_values = pd.to_numeric(indicator, errors="coerce").rename("indicator")
    rows: list[dict[str, object]] = []

    for target_column in forward_targets.columns:
        target_values = pd.to_numeric(forward_targets[target_column], errors="coerce")
        pair = pd.concat([indicator_values, target_values.rename("target")], axis=1).dropna()
        metrics = _predictive_metrics(pair["indicator"], pair["target"])
        rows.extend(
            {"target": target_column, "metric": metric, "value": value}
            for metric, value in metrics.items()
        )

    return pd.DataFrame(rows, columns=["target", "metric", "value"])


def compute_event_study(
    indicator: pd.Series,
    event_dates: list[pd.Timestamp] | list[str],
    window_before: int = 60,
    window_after: int = 60,
) -> pd.DataFrame:
    """Extract indicator paths around supplied event dates.

    Event dates are matched to the closest available indicator date less than or
    equal to the event date. If an event predates the sample, the first available
    date is used so edge events do not fail the study.
    """

    if not isinstance(indicator, pd.Series):
        raise TypeError("indicator must be a pandas Series.")
    if indicator.empty:
        raise ValueError("indicator must not be empty.")
    if window_before < 0 or window_after < 0:
        raise ValueError("event windows must be non-negative.")

    indicator_series = pd.to_numeric(indicator, errors="coerce").sort_index()
    datetime_index = pd.to_datetime(indicator_series.index)
    if datetime_index.isna().any():
        raise ValueError("indicator index must be datetime-like for event studies.")

    indicator_series.index = pd.DatetimeIndex(datetime_index)
    rows: list[dict[str, object]] = []

    for raw_event_date in event_dates:
        event_date = pd.Timestamp(raw_event_date)
        matched_position = int(indicator_series.index.searchsorted(event_date, side="right") - 1)
        matched_position = max(0, min(matched_position, len(indicator_series) - 1))

        start_position = max(0, matched_position - window_before)
        end_position = min(len(indicator_series), matched_position + window_after + 1)
        matched_date = indicator_series.index[matched_position]

        for position in range(start_position, end_position):
            rows.append(
                {
                    "event_date": event_date,
                    "matched_date": matched_date,
                    "relative_day": position - matched_position,
                    "regime_indicator": float(indicator_series.iloc[position]),
                },
            )

    return pd.DataFrame(
        rows,
        columns=["event_date", "matched_date", "relative_day", "regime_indicator"],
    )


def classify_regimes_by_quantile(
    indicator: pd.Series,
    high_quantile: float = 0.80,
    low_quantile: float = 0.20,
) -> pd.Series:
    """Classify indicator values into low, normal, and high connectedness regimes."""

    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= low_quantile < high_quantile <= 1.")
    if not isinstance(indicator, pd.Series):
        raise TypeError("indicator must be a pandas Series.")

    numeric_indicator = pd.to_numeric(indicator, errors="coerce")
    high_threshold = numeric_indicator.quantile(high_quantile)
    low_threshold = numeric_indicator.quantile(low_quantile)

    classes = pd.Series(pd.NA, index=indicator.index, dtype="object", name="regime_class")
    valid = numeric_indicator.notna()
    classes.loc[valid] = "normal"
    classes.loc[valid & (numeric_indicator <= low_threshold)] = "low_systemic_connectedness"
    classes.loc[valid & (numeric_indicator >= high_threshold)] = "high_systemic_connectedness"

    return classes


def summarize_regime_classes(
    indicator: pd.Series,
    returns: pd.Series | pd.DataFrame | None = None,
    high_quantile: float = 0.80,
    low_quantile: float = 0.20,
) -> pd.DataFrame:
    """Summarize observations and return behavior by indicator quantile class."""

    regime_class = classify_regimes_by_quantile(
        indicator,
        high_quantile=high_quantile,
        low_quantile=low_quantile,
    )

    if returns is None:
        summary = regime_class.dropna().value_counts().rename("n_obs").to_frame()
        summary.index.name = "regime_class"
        return summary

    market_returns = _market_return_series(returns, market_return_column=None)
    aligned = pd.concat(
        [regime_class, market_returns.rename("return")],
        axis=1,
        join="inner",
    ).dropna(subset=["regime_class"])

    grouped = aligned.groupby("regime_class", sort=True)["return"]
    summary = pd.DataFrame(
        {
            "n_obs": grouped.count(),
            "mean_return": grouped.mean(),
            "volatility": grouped.std(),
            "mean_absolute_return": grouped.apply(lambda values: values.abs().mean()),
        },
    )
    summary.index.name = "regime_class"
    return summary


def _detect_stress_label_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if str(column).endswith("_stress_label")]
    if "systemic_stress_label" in frame.columns and "systemic_stress_label" not in columns:
        columns.append("systemic_stress_label")
    return columns


def _tidy_rows(
    diagnostic_type: str,
    benchmark: str,
    metrics: dict[str, float],
) -> list[dict[str, object]]:
    return [
        {
            "diagnostic_type": diagnostic_type,
            "benchmark": benchmark,
            "metric": metric,
            "value": value,
        }
        for metric, value in metrics.items()
    ]


def _mean_or_nan(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else np.nan


def _median_or_nan(values: pd.Series) -> float:
    return float(values.median()) if len(values) else np.nan


def _welch_t_statistic(stress_group: pd.Series, non_stress_group: pd.Series) -> float:
    if len(stress_group) < 2 or len(non_stress_group) < 2:
        return np.nan

    result = stats.ttest_ind(
        stress_group.to_numpy(dtype=float),
        non_stress_group.to_numpy(dtype=float),
        equal_var=False,
        nan_policy="omit",
    )
    return float(result.statistic)


def _correlations(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    if len(left) < 2 or _is_constant(left) or _is_constant(right):
        return np.nan, np.nan

    pearson = stats.pearsonr(left.to_numpy(dtype=float), right.to_numpy(dtype=float)).statistic
    spearman = stats.spearmanr(
        left.to_numpy(dtype=float),
        right.to_numpy(dtype=float),
        nan_policy="omit",
    ).statistic
    return float(pearson), float(spearman)


def _predictive_metrics(indicator: pd.Series, target: pd.Series) -> dict[str, float]:
    n_obs = float(len(indicator))
    unstable_metrics = {
        "n_obs": n_obs,
        "pearson_correlation": np.nan,
        "spearman_correlation": np.nan,
        "ols_beta": np.nan,
        "ols_intercept": np.nan,
        "ols_beta_t_stat": np.nan,
        "r_squared": np.nan,
    }
    if len(indicator) < 3 or _is_constant(indicator) or _is_constant(target):
        return unstable_metrics

    x = indicator.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean
    sxx = float(np.sum(x_centered**2))
    if sxx <= 0:
        return unstable_metrics

    beta = float(np.sum(x_centered * y_centered) / sxx)
    intercept = float(y_mean - beta * x_mean)
    fitted = intercept + beta * x
    residuals = y - fitted
    sse = float(np.sum(residuals**2))
    sst = float(np.sum(y_centered**2))
    r_squared = float(1.0 - sse / sst) if sst > 0 else np.nan

    degrees_of_freedom = len(x) - 2
    if degrees_of_freedom > 0:
        residual_variance = sse / degrees_of_freedom
        beta_standard_error = np.sqrt(residual_variance / sxx) if residual_variance >= 0 else np.nan
        beta_t_stat = beta / beta_standard_error if beta_standard_error > 0 else np.nan
    else:
        beta_t_stat = np.nan

    pearson, spearman = _correlations(indicator, target)
    return {
        "n_obs": n_obs,
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "ols_beta": beta,
        "ols_intercept": intercept,
        "ols_beta_t_stat": float(beta_t_stat),
        "r_squared": r_squared,
    }


def _market_return_series(
    returns: pd.Series | pd.DataFrame,
    market_return_column: str | None,
) -> pd.Series:
    if isinstance(returns, pd.DataFrame):
        if returns.empty:
            raise ValueError("returns must not be empty.")
        numeric_returns = returns.apply(pd.to_numeric, errors="coerce")
        if market_return_column is not None and market_return_column in numeric_returns.columns:
            market_returns = numeric_returns[market_return_column]
        else:
            market_returns = numeric_returns.mean(axis=1, skipna=True)
    elif isinstance(returns, pd.Series):
        if returns.empty:
            raise ValueError("returns must not be empty.")
        market_returns = pd.to_numeric(returns, errors="coerce")
    else:
        raise TypeError("returns must be a pandas Series or DataFrame.")

    return market_returns.replace([np.inf, -np.inf], np.nan).rename("market_return")


def _forward_max_drawdown(forward_returns: Sequence[float] | pd.Series) -> float:
    returns = pd.Series(forward_returns, dtype=float).dropna()
    if returns.empty:
        return np.nan

    cumulative_return_path = returns.cumsum()
    approximate_wealth = pd.concat(
        [pd.Series([1.0]), 1.0 + cumulative_return_path.reset_index(drop=True)],
        ignore_index=True,
    )
    running_max = approximate_wealth.cummax()
    drawdown = approximate_wealth / running_max - 1.0
    return float(drawdown.min())


def _is_constant(values: pd.Series) -> bool:
    finite_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return finite_values.size < 2 or bool(np.isclose(finite_values.std(ddof=0), 0.0))
