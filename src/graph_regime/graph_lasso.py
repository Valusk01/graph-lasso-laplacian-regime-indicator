"""Sparse precision-matrix estimation and partial-correlation conversion."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GraphicalLassoFit:
    """Graphical-lasso fit result with convergence diagnostics."""

    covariance: np.ndarray
    precision: np.ndarray
    converged: bool
    n_iter: int | None
    warning_messages: tuple[str, ...]
    alpha: float
    max_iter: int
    tol: float
    enet_tol: float
    mode: str


def fit_graphical_lasso(
    window: pd.DataFrame,
    alpha: float,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a sparse precision matrix from a standardized return window.

    Graphical lasso estimates a sparse inverse covariance matrix. Under Gaussian
    assumptions, zero precision entries correspond to conditional independence.
    Outside the Gaussian case, the safer interpretation is sparse linear
    conditional-dependence / partial-correlation topology.

    Graphical lasso does not produce a graph Laplacian directly; the precision
    matrix must first be converted into partial correlations before graph
    construction.
    """

    fit = fit_graphical_lasso_with_diagnostics(
        window=window,
        alpha=alpha,
        max_iter=max_iter,
    )
    return fit.covariance, fit.precision


def fit_graphical_lasso_with_diagnostics(
    window: pd.DataFrame,
    alpha: float,
    max_iter: int = 1000,
    tol: float = 1e-4,
    enet_tol: float = 1e-4,
    mode: str = "cd",
    assume_centered: bool = True,
) -> GraphicalLassoFit:
    """Estimate graphical lasso and return convergence diagnostics.

    ConvergenceWarning messages from scikit-learn are captured instead of being
    printed repeatedly during rolling workflows. Non-convergence is recorded in
    the returned fit object so callers can decide whether to record, warn, or
    raise.
    """

    _validate_graphical_lasso_inputs(
        window=window,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        enet_tol=enet_tol,
        mode=mode,
    )

    try:
        from sklearn.covariance import GraphicalLasso
        from sklearn.exceptions import ConvergenceWarning
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for fit_graphical_lasso_with_diagnostics. "
            "Install the project dependencies before running the rolling engine."
        ) from exc

    values = window.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("window must contain only finite values.")

    model = GraphicalLasso(
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        enet_tol=enet_tol,
        mode=mode,
        assume_centered=assume_centered,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(values)

    convergence_warnings = tuple(
        str(warning.message)
        for warning in caught_warnings
        if issubclass(warning.category, ConvergenceWarning)
    )

    return GraphicalLassoFit(
        covariance=np.asarray(model.covariance_, dtype=float),
        precision=np.asarray(model.precision_, dtype=float),
        converged=len(convergence_warnings) == 0,
        n_iter=getattr(model, "n_iter_", None),
        warning_messages=convergence_warnings,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        enet_tol=enet_tol,
        mode=mode,
    )


def _validate_graphical_lasso_inputs(
    window: pd.DataFrame,
    alpha: float,
    max_iter: int,
    tol: float,
    enet_tol: float,
    mode: str,
) -> None:
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if enet_tol <= 0:
        raise ValueError("enet_tol must be positive.")
    if mode not in {"cd", "lars"}:
        raise ValueError("mode must be either 'cd' or 'lars'.")
    if not isinstance(window, pd.DataFrame):
        raise TypeError("window must be a pandas DataFrame.")
    if window.shape[0] < 2:
        raise ValueError("window must contain at least two observations.")
    if window.shape[1] < 2:
        raise ValueError("window must contain at least two assets.")


def precision_to_partial_correlation(precision: np.ndarray) -> np.ndarray:
    """Convert a sparse precision matrix into partial correlations.

    Partial correlations normalize each off-diagonal precision entry by the
    conditional variances, yielding signed conditional-dependence strengths
    between asset pairs after controlling for the rest of the universe.
    """

    precision_array = np.asarray(precision, dtype=float)
    if precision_array.ndim != 2 or precision_array.shape[0] != precision_array.shape[1]:
        raise ValueError("precision must be a square matrix.")
    if not np.isfinite(precision_array).all():
        raise ValueError("precision must contain only finite values.")

    diagonal = np.diag(precision_array)
    if np.any(diagonal <= 0):
        raise ValueError("precision diagonal entries must be positive.")

    denominator = np.sqrt(np.outer(diagonal, diagonal))
    with np.errstate(divide="raise", invalid="raise"):
        partial_corr = -precision_array / denominator

    partial_corr = np.clip(partial_corr, -1.0, 1.0)
    np.fill_diagonal(partial_corr, 1.0)
    partial_corr = 0.5 * (partial_corr + partial_corr.T)

    return partial_corr
