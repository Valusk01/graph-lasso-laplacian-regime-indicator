import numpy as np
import pandas as pd

from graph_regime.preprocessing import clean_returns, standardize_window


def test_clean_returns_drops_sparse_assets_and_missing_rows() -> None:
    returns = pd.DataFrame(
        {
            "asset_a": [0.01, 0.02, 0.03, 0.04],
            "asset_b": [0.02, np.nan, 0.01, 0.03],
            "asset_c": [np.nan, np.nan, 0.01, np.nan],
            "asset_d": [0.03, 0.01, np.inf, 0.02],
        }
    )

    cleaned = clean_returns(returns, min_non_missing=0.75)

    assert list(cleaned.columns) == ["asset_a", "asset_b", "asset_d"]
    assert cleaned.index.tolist() == [0, 3]
    assert np.isfinite(cleaned.to_numpy()).all()


def test_standardize_window_centers_and_scales_assets() -> None:
    window = pd.DataFrame(
        {
            "asset_a": [1.0, 2.0, 3.0, 4.0],
            "asset_b": [2.0, 4.0, 6.0, 8.0],
            "constant_asset": [5.0, 5.0, 5.0, 5.0],
        }
    )

    standardized = standardize_window(window)

    assert np.allclose(standardized[["asset_a", "asset_b"]].mean(axis=0), 0.0)
    assert np.allclose(
        standardized[["asset_a", "asset_b"]].std(axis=0, ddof=0),
        1.0,
    )
    assert np.allclose(standardized["constant_asset"], 0.0)
