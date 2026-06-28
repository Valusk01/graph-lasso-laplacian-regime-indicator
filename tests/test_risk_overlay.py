import numpy as np
import pandas as pd

from graph_regime.risk_overlay import (
    apply_risk_overlay,
    compute_benchmark_exposures,
    compute_exposure_from_indicator,
    compute_performance_metrics,
    compute_portfolio_returns,
)


def test_exposure_is_shifted_to_avoid_lookahead() -> None:
    indicator = pd.Series(
        [-3.0, -2.0, -1.0, 10.0, 0.0],
        index=pd.date_range("2020-01-01", periods=5),
    )

    exposure = compute_exposure_from_indicator(
        indicator,
        high_quantile=0.6,
        extreme_quantile=0.8,
        method="full_sample",
    )

    assert exposure.iloc[0] == 1.0
    assert exposure.iloc[3] == 1.0
    assert exposure.iloc[4] == 0.5


def test_expanding_exposure_uses_prior_history_only() -> None:
    indicator = pd.Series(
        [0.0, 0.0, 0.0, 10.0, 0.0],
        index=pd.date_range("2020-01-01", periods=5),
    )

    exposure = compute_exposure_from_indicator(
        indicator,
        high_quantile=0.8,
        extreme_quantile=0.9,
        method="expanding",
        min_history=3,
    )

    assert exposure.iloc[:4].eq(1.0).all()
    assert exposure.iloc[4] == 0.5


def test_overlay_returns_align_with_exposure() -> None:
    returns = pd.Series(
        [0.01, 0.02, -0.03],
        index=pd.date_range("2020-01-01", periods=3),
    )
    exposure = pd.Series(
        [1.0, 0.5, 0.5],
        index=pd.date_range("2020-01-01", periods=3),
    )

    overlay = apply_risk_overlay(returns, exposure)

    assert np.allclose(overlay.to_numpy(), [0.01, 0.01, -0.015])


def test_compute_portfolio_returns_equal_weight_and_custom_weights() -> None:
    returns = pd.DataFrame(
        {
            "a": [0.01, 0.03],
            "b": [0.03, 0.01],
        },
        index=pd.date_range("2020-01-01", periods=2),
    )

    equal_weight = compute_portfolio_returns(returns)
    weighted = compute_portfolio_returns(returns, weights=pd.Series({"a": 0.75, "b": 0.25}))

    assert np.allclose(equal_weight.to_numpy(), [0.02, 0.02])
    assert np.allclose(weighted.to_numpy(), [0.015, 0.025])


def test_performance_metrics_are_finite_and_max_drawdown_is_correct() -> None:
    returns = pd.Series([0.10, -0.10, 0.05, -0.02])

    metrics = compute_performance_metrics(returns, trading_days=4)

    assert metrics["n_obs"] == 4
    assert np.isfinite(metrics["annualized_volatility"])
    assert np.isclose(metrics["max_drawdown"], -0.10)


def test_benchmark_overlays_reduce_exposure_for_high_risk_values() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    benchmarks = pd.DataFrame(
        {
            "vix": [10.0, 10.0, 10.0, 50.0, 10.0],
            "drawdown": [0.0, 0.0, 0.0, -0.3, 0.0],
        },
        index=index,
    )

    exposures = compute_benchmark_exposures(
        benchmarks,
        benchmark_columns=["vix", "drawdown"],
        method="expanding",
        min_history=3,
    )

    assert exposures.loc[index[-1], "vix_exposure"] == 0.5
    assert exposures.loc[index[-1], "drawdown_exposure"] == 0.5
