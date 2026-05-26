from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graph_regime.evaluation import (
    align_indicator_and_benchmarks,
    classify_regimes_by_quantile,
    compute_contemporaneous_diagnostics,
    compute_event_study,
    compute_forward_targets,
    evaluate_predictive_power,
    summarize_regime_classes,
)


def test_align_indicator_and_benchmarks_sorts_and_preserves_columns() -> None:
    index = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"])
    indicator = pd.DataFrame({"regime_indicator": [3.0, 1.0, 2.0]}, index=index)
    benchmarks = pd.DataFrame(
        {
            "systemic_stress_label": pd.Series([1, 0, 1], index=index, dtype="Int64"),
            "vix": [35.0, 20.0, 30.0],
        },
        index=index,
    )

    aligned = align_indicator_and_benchmarks(indicator, benchmarks)

    assert "regime_indicator" in aligned.columns
    assert "systemic_stress_label" in aligned.columns
    assert "vix" in aligned.columns
    assert aligned.index.is_monotonic_increasing


def test_compute_contemporaneous_diagnostics_detects_stress_difference_and_correlations() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    aligned = pd.DataFrame(
        {
            "regime_indicator": [0.1, 0.2, 0.3, 1.2, 1.4, 1.6],
            "systemic_stress_label": [0, 0, 0, 1, 1, 1],
            "vix": [15.0, 16.0, 18.0, 30.0, 35.0, 40.0],
        },
        index=index,
    )

    diagnostics = compute_contemporaneous_diagnostics(aligned)

    diff = diagnostics.loc[
        (diagnostics["benchmark"] == "systemic_stress_label")
        & (diagnostics["metric"] == "difference_in_means"),
        "value",
    ].iloc[0]
    continuous_metrics = diagnostics.loc[
        (diagnostics["benchmark"] == "vix")
        & (diagnostics["metric"].isin(["pearson_correlation", "spearman_correlation"]))
    ]

    assert diff > 0.0
    assert len(continuous_metrics) == 2


def test_compute_forward_targets_use_future_values_only() -> None:
    returns = pd.Series(
        [0.10, 0.01, 0.02, -0.03, 0.04],
        index=pd.date_range("2020-01-01", periods=5),
    )

    targets = compute_forward_targets(returns, horizons=[2, 3])

    expected_columns = {
        "forward_realized_volatility_2d",
        "forward_return_sum_2d",
        "forward_absolute_return_sum_2d",
        "forward_max_drawdown_2d",
        "forward_realized_volatility_3d",
        "forward_return_sum_3d",
        "forward_absolute_return_sum_3d",
        "forward_max_drawdown_3d",
    }
    assert expected_columns.issubset(targets.columns)
    assert np.isclose(targets.iloc[0]["forward_return_sum_2d"], 0.01 + 0.02)
    assert not np.isclose(targets.iloc[0]["forward_return_sum_2d"], 0.10 + 0.01)
    assert targets["forward_return_sum_3d"].tail(3).isna().all()


def test_evaluate_predictive_power_reports_positive_correlation_and_beta() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    indicator = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=index)
    forward_targets = pd.DataFrame(
        {"forward_realized_volatility_5d": [0.1, 0.2, 0.4, 0.7, 0.9, 1.2]},
        index=index,
    )

    diagnostics = evaluate_predictive_power(indicator, forward_targets)
    pearson = diagnostics.loc[
        diagnostics["metric"] == "pearson_correlation",
        "value",
    ].iloc[0]

    assert pearson > 0.0
    assert "ols_beta" in diagnostics["metric"].to_list()


def test_compute_event_study_returns_relative_path() -> None:
    indicator = pd.Series(
        np.arange(10, dtype=float),
        index=pd.date_range("2020-01-01", periods=10, freq="D"),
    )

    event_study = compute_event_study(
        indicator,
        event_dates=["2020-01-05", pd.Timestamp("2020-01-08")],
        window_before=2,
        window_after=1,
    )

    assert "relative_day" in event_study.columns
    assert "event_date" in event_study.columns
    assert "matched_date" in event_study.columns
    assert 0 in event_study["relative_day"].to_list()


def test_classify_regimes_by_quantile_labels_and_rejects_invalid_quantiles() -> None:
    indicator = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    classes = classify_regimes_by_quantile(
        indicator,
        low_quantile=0.20,
        high_quantile=0.80,
    )

    assert classes.iloc[0] == "low_systemic_connectedness"
    assert classes.iloc[2] == "normal"
    assert classes.iloc[-1] == "high_systemic_connectedness"

    with pytest.raises(ValueError):
        classify_regimes_by_quantile(indicator, low_quantile=0.90, high_quantile=0.80)


def test_summarize_regime_classes_with_returns() -> None:
    indicator = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        index=pd.date_range("2020-01-01", periods=5),
    )
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03], index=indicator.index)

    summary = summarize_regime_classes(
        indicator,
        returns=returns,
        low_quantile=0.20,
        high_quantile=0.80,
    )

    assert "n_obs" in summary.columns
    assert "mean_return" in summary.columns
    assert "volatility" in summary.columns
    assert summary["n_obs"].sum() == 5
