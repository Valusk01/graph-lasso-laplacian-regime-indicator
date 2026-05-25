"""Preprocessing utilities for rolling return windows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_returns(
    returns: pd.DataFrame,
    min_non_missing: float = 0.95,
) -> pd.DataFrame:
    """Clean an asset-return panel before rolling graph estimation.

    Assets with too many missing observations are removed, and remaining rows
    with missing or infinite values are dropped. This keeps each graphical-lasso
    window on a stable asset universe while avoiding implicit return imputation.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if not 0 < min_non_missing <= 1:
        raise ValueError("min_non_missing must be in the interval (0, 1].")

    numeric_returns = returns.apply(pd.to_numeric, errors="coerce")
    numeric_returns = numeric_returns.replace([np.inf, -np.inf], np.nan)

    keep_columns = numeric_returns.notna().mean(axis=0) >= min_non_missing
    cleaned = numeric_returns.loc[:, keep_columns].dropna(axis=0, how="any")

    return cleaned.astype(float)


def standardize_window(window: pd.DataFrame) -> pd.DataFrame:
    """Standardize returns within one rolling window asset by asset.

    Centering and scaling inside the window makes the graph primarily reflect
    conditional dependence topology rather than raw volatility level. Constant
    columns are left as zeros after centering to avoid division by zero.
    """

    if not isinstance(window, pd.DataFrame):
        raise TypeError("window must be a pandas DataFrame.")
    if window.empty:
        raise ValueError("window must contain at least one row and one column.")

    numeric_window = window.apply(pd.to_numeric, errors="coerce")
    numeric_window = numeric_window.replace([np.inf, -np.inf], np.nan)
    numeric_window = numeric_window.dropna(axis=0, how="any")

    if numeric_window.empty:
        raise ValueError("window contains no complete observations.")

    means = numeric_window.mean(axis=0)
    stds = numeric_window.std(axis=0, ddof=0)
    safe_stds = stds.mask((stds <= 0) | ~np.isfinite(stds), 1.0)

    standardized = (numeric_window - means) / safe_stds
    return standardized.astype(float)
