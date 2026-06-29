import numpy as np
import pandas as pd

from graph_regime.pca_baselines import compute_rolling_pca_features


def test_rolling_pca_features_are_finite_and_bounded() -> None:
    index = pd.date_range("2020-01-01", periods=12)
    returns = pd.DataFrame(
        {
            "a": np.linspace(-0.02, 0.03, 12),
            "b": np.linspace(0.03, -0.01, 12),
            "c": np.sin(np.arange(12)) * 0.01,
        },
        index=index,
    )

    features = compute_rolling_pca_features(returns, window=5)

    assert features.index[0] == index[4]
    assert "pca_first_eigenvalue_share" in features.columns
    assert features["pca_effective_rank"].dropna().gt(0).all()
    assert features["pca_first_eigenvalue_share"].dropna().between(0, 1).all()


def test_rolling_pca_features_do_not_use_future_data() -> None:
    index = pd.date_range("2020-01-01", periods=10)
    returns = pd.DataFrame(
        {
            "a": np.arange(10, dtype=float) / 100.0,
            "b": np.arange(10, 20, dtype=float) / 100.0,
            "c": np.cos(np.arange(10)) / 100.0,
        },
        index=index,
    )
    changed_future = returns.copy()
    changed_future.iloc[-1, :] = [10.0, -10.0, 5.0]

    original = compute_rolling_pca_features(returns, window=4)
    modified = compute_rolling_pca_features(changed_future, window=4)

    early_index = original.index[:-1]
    assert np.allclose(
        original.loc[early_index, "pca_first_eigenvalue_share"],
        modified.loc[early_index, "pca_first_eigenvalue_share"],
        equal_nan=True,
    )
