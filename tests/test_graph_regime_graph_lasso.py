import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from graph_regime.graph_lasso import (
    GraphicalLassoFit,
    fit_graphical_lasso,
    fit_graphical_lasso_with_diagnostics,
    precision_to_partial_correlation,
)


def test_precision_to_partial_correlation_is_symmetric_finite_with_unit_diagonal() -> None:
    precision = np.array(
        [
            [2.0, -0.4, 0.1],
            [-0.4, 1.5, -0.2],
            [0.1, -0.2, 1.2],
        ]
    )

    partial_corr = precision_to_partial_correlation(precision)

    assert np.allclose(np.diag(partial_corr), 1.0)
    assert np.allclose(partial_corr, partial_corr.T)
    assert np.isfinite(partial_corr).all()


def test_fit_graphical_lasso_rejects_zero_alpha() -> None:
    window = pd.DataFrame(
        {
            "asset_a": [0.1, -0.2, 0.3, -0.1],
            "asset_b": [0.2, -0.1, 0.1, -0.3],
            "asset_c": [-0.1, 0.1, -0.2, 0.2],
        }
    )

    with pytest.raises(ValueError, match="alpha must be positive"):
        fit_graphical_lasso(window, alpha=0.0)


def test_fit_graphical_lasso_with_diagnostics_returns_fit_metadata(monkeypatch) -> None:
    _install_fake_sklearn(monkeypatch)
    window = pd.DataFrame(
        {
            "asset_a": [0.1, -0.2, 0.3, -0.1, 0.2],
            "asset_b": [0.2, -0.1, 0.1, -0.3, 0.1],
            "asset_c": [-0.1, 0.1, -0.2, 0.2, -0.1],
        }
    )

    fit = fit_graphical_lasso_with_diagnostics(
        window,
        alpha=0.5,
        max_iter=50,
        tol=1e-4,
        enet_tol=1e-4,
        mode="cd",
    )

    assert isinstance(fit, GraphicalLassoFit)
    assert fit.covariance.shape == (3, 3)
    assert fit.precision.shape == (3, 3)
    assert isinstance(fit.converged, bool)
    assert fit.n_iter is None or fit.n_iter > 0
    assert fit.alpha == 0.5
    assert fit.max_iter == 50
    assert fit.tol == 1e-4
    assert fit.enet_tol == 1e-4
    assert fit.mode == "cd"
    assert isinstance(fit.warning_messages, tuple)
    assert fit.converged


def test_fit_graphical_lasso_with_diagnostics_captures_convergence_warning(monkeypatch) -> None:
    _install_fake_sklearn(monkeypatch, emit_warning=True)
    window = pd.DataFrame(
        {
            "asset_a": [0.1, -0.2, 0.3],
            "asset_b": [0.2, -0.1, 0.1],
        }
    )

    fit = fit_graphical_lasso_with_diagnostics(window, alpha=0.5)

    assert not fit.converged
    assert len(fit.warning_messages) == 1
    assert "did not converge" in fit.warning_messages[0]


def test_fit_graphical_lasso_with_diagnostics_rejects_invalid_tol() -> None:
    window = pd.DataFrame({"asset_a": [0.1, 0.2], "asset_b": [0.2, 0.1]})

    with pytest.raises(ValueError, match="tol must be positive"):
        fit_graphical_lasso_with_diagnostics(window, alpha=0.5, tol=0.0)


def test_fit_graphical_lasso_with_diagnostics_rejects_invalid_enet_tol() -> None:
    window = pd.DataFrame({"asset_a": [0.1, 0.2], "asset_b": [0.2, 0.1]})

    with pytest.raises(ValueError, match="enet_tol must be positive"):
        fit_graphical_lasso_with_diagnostics(window, alpha=0.5, enet_tol=0.0)


def test_fit_graphical_lasso_with_diagnostics_rejects_invalid_mode() -> None:
    window = pd.DataFrame({"asset_a": [0.1, 0.2], "asset_b": [0.2, 0.1]})

    with pytest.raises(ValueError, match="mode must be either"):
        fit_graphical_lasso_with_diagnostics(window, alpha=0.5, mode="bad")


def _install_fake_sklearn(monkeypatch, emit_warning: bool = False) -> None:
    class FakeConvergenceWarning(UserWarning):
        pass

    class FakeGraphicalLasso:
        def __init__(
            self,
            alpha,
            max_iter,
            tol,
            enet_tol,
            mode,
            assume_centered,
        ):
            self.alpha = alpha
            self.max_iter = max_iter
            self.tol = tol
            self.enet_tol = enet_tol
            self.mode = mode
            self.assume_centered = assume_centered

        def fit(self, values):
            if emit_warning:
                warnings.warn(
                    "graphical_lasso: did not converge",
                    FakeConvergenceWarning,
                    stacklevel=2,
                )
            n_assets = values.shape[1]
            self.covariance_ = np.eye(n_assets)
            self.precision_ = np.eye(n_assets)
            self.n_iter_ = min(self.max_iter, 4)
            return self

    sklearn_module = types.ModuleType("sklearn")
    sklearn_module.__path__ = []
    covariance_module = types.ModuleType("sklearn.covariance")
    covariance_module.GraphicalLasso = FakeGraphicalLasso
    exceptions_module = types.ModuleType("sklearn.exceptions")
    exceptions_module.ConvergenceWarning = FakeConvergenceWarning

    monkeypatch.setitem(sys.modules, "sklearn", sklearn_module)
    monkeypatch.setitem(sys.modules, "sklearn.covariance", covariance_module)
    monkeypatch.setitem(sys.modules, "sklearn.exceptions", exceptions_module)
