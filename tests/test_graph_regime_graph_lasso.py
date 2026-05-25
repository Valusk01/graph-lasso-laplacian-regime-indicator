import numpy as np
import pandas as pd
import pytest

from graph_regime.graph_lasso import (
    fit_graphical_lasso,
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
