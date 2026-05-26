"""Benchmark stress variables for graph-regime indicator comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_realized_volatility(
    returns: pd.Series | pd.DataFrame,
    window: int = 21,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """Compute rolling realized volatility from returns.

    DataFrame input is converted to an equal-weight portfolio return first, so
    the benchmark captures broad realized market turbulence rather than one
    asset's idiosyncratic volatility.
    """

    if window <= 1:
        raise ValueError("window must be greater than one.")
    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")

    if isinstance(returns, pd.DataFrame):
        if returns.empty:
            raise ValueError("returns must not be empty.")
        numeric_returns = returns.apply(pd.to_numeric, errors="coerce")
        return_series = numeric_returns.mean(axis=1, skipna=True)
    elif isinstance(returns, pd.Series):
        if returns.empty:
            raise ValueError("returns must not be empty.")
        return_series = pd.to_numeric(returns, errors="coerce")
    else:
        raise TypeError("returns must be a pandas Series or DataFrame.")

    realized_volatility = return_series.replace([np.inf, -np.inf], np.nan).rolling(
        window=window,
        min_periods=window,
    ).std()
    if annualize:
        realized_volatility = realized_volatility * np.sqrt(trading_days)

    return realized_volatility.rename("realized_volatility")


def compute_market_drawdown(price_series: pd.Series) -> pd.Series:
    """Compute drawdown from a broad-market price series.

    Drawdown measures loss from the running high-water mark and is a simple
    benchmark for market stress and crisis depth.
    """

    if not isinstance(price_series, pd.Series):
        raise TypeError("price_series must be a pandas Series.")
    if price_series.empty:
        raise ValueError("price_series must not be empty.")

    prices = pd.to_numeric(price_series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if prices.dropna().empty:
        raise ValueError("price_series must contain at least one numeric value.")
    if (prices.dropna() <= 0).any():
        raise ValueError("price_series must contain positive observed prices.")

    running_max = prices.cummax()
    drawdown = prices / running_max - 1.0
    return drawdown.rename("drawdown")


def compute_correlation_regime_score(
    returns: pd.DataFrame,
    window: int = 126,
    use_absolute: bool = False,
) -> pd.Series:
    """Compute rolling average pairwise asset correlation.

    The default score uses raw off-diagonal correlations, which measure
    same-direction co-movement across assets. If use_absolute=True, the score
    uses absolute off-diagonal correlations, which measure dependence strength
    regardless of sign. The absolute version can be useful for mixed asset
    universes containing equities, bonds, commodities, credit, and defensive
    assets.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.shape[1] < 2:
        raise ValueError("returns must contain at least two assets.")
    if window <= 1:
        raise ValueError("window must be greater than one.")

    numeric_returns = returns.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )

    values: list[float] = []
    index: list[object] = []

    for end_position in range(window, len(numeric_returns) + 1):
        window_returns = numeric_returns.iloc[end_position - window : end_position]
        correlation = window_returns.corr()
        mask = np.triu(np.ones(correlation.shape, dtype=bool), k=1)
        pairwise_values = correlation.to_numpy(dtype=float)[mask]

        if use_absolute:
            pairwise_values = np.abs(pairwise_values)

        finite_values = pairwise_values[np.isfinite(pairwise_values)]
        values.append(float(finite_values.mean()) if finite_values.size else np.nan)
        index.append(numeric_returns.index[end_position - 1])

    score_name = "average_absolute_correlation" if use_absolute else "average_correlation"

    return pd.Series(
        values,
        index=pd.Index(index, name=returns.index.name),
        name=score_name,
    )


def create_stress_labels(
    vix: pd.Series | None = None,
    drawdown: pd.Series | None = None,
    realized_vol: pd.Series | None = None,
    correlation_score: pd.Series | None = None,
    vix_threshold: float = 30.0,
    drawdown_threshold: float = -0.10,
    vol_quantile: float = 0.80,
    correlation_quantile: float = 0.80,
) -> pd.DataFrame:
    """Create binary benchmark stress labels from available public variables.

    Inputs are outer-joined on index. The combined systemic label is one when at
    least one available benchmark flags stress on that date.
    """

    _validate_quantile(vol_quantile, "vol_quantile")
    _validate_quantile(correlation_quantile, "correlation_quantile")

    series_by_name: dict[str, pd.Series] = {}
    if vix is not None:
        series_by_name["vix"] = _as_numeric_series(vix, "vix")
    if drawdown is not None:
        series_by_name["drawdown"] = _as_numeric_series(drawdown, "drawdown")
    if realized_vol is not None:
        series_by_name["realized_volatility"] = _as_numeric_series(
            realized_vol,
            "realized_volatility",
        )
    if correlation_score is not None:
        series_by_name["average_correlation"] = _as_numeric_series(
            correlation_score,
            "average_correlation",
        )

    if not series_by_name:
        raise ValueError("At least one benchmark series must be provided.")

    output = pd.concat(series_by_name.values(), axis=1, join="outer").sort_index()
    label_columns: list[str] = []

    if "vix" in output:
        output["vix_stress_label"] = _threshold_label(
            output["vix"],
            output["vix"] >= vix_threshold,
        )
        label_columns.append("vix_stress_label")

    if "drawdown" in output:
        output["drawdown_stress_label"] = _threshold_label(
            output["drawdown"],
            output["drawdown"] <= drawdown_threshold,
        )
        label_columns.append("drawdown_stress_label")

    if "realized_volatility" in output:
        threshold = output["realized_volatility"].quantile(vol_quantile)
        output["realized_volatility_stress_label"] = _threshold_label(
            output["realized_volatility"],
            output["realized_volatility"] >= threshold,
        )
        label_columns.append("realized_volatility_stress_label")

    if "average_correlation" in output:
        threshold = output["average_correlation"].quantile(correlation_quantile)
        output["correlation_stress_label"] = _threshold_label(
            output["average_correlation"],
            output["average_correlation"] >= threshold,
        )
        label_columns.append("correlation_stress_label")

    output["systemic_stress_label"] = (
        output[label_columns].fillna(0).max(axis=1).astype(int)
    )
    return output


def load_fred_recession_indicator(
    series_id: str = "USREC",
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Load a FRED recession indicator using pandas-datareader when available."""

    try:
        from pandas_datareader import data as web
    except ImportError as exc:
        raise ImportError(
            "pandas-datareader is required for FRED recession indicators. "
            "Install the 'data' optional dependencies or load a local CSV file."
        ) from exc

    try:
        frame = web.DataReader(series_id, "fred", start=start, end=end)
    except Exception as exc:
        raise RuntimeError(
            "FRED recession indicator download failed. Check internet access, "
            "the series id, or use a local CSV file."
        ) from exc

    if frame.empty or series_id not in frame.columns:
        raise RuntimeError(f"FRED returned no data for series id {series_id!r}.")

    return frame[series_id].rename(series_id.lower())


def _validate_quantile(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in the interval [0, 1].")


def _as_numeric_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")

    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric.rename(name)

def _threshold_label(series: pd.Series, condition: pd.Series) -> pd.Series:
    label = condition.astype("Int64")
    label = label.mask(series.isna(), pd.NA)
    return label