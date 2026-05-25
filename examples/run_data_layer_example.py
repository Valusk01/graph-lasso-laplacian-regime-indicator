"""Run the Phase 2 public-data and benchmark-label example."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from graph_regime.benchmarks import (  # noqa: E402
    compute_correlation_regime_score,
    compute_market_drawdown,
    compute_realized_volatility,
    create_stress_labels,
)
from graph_regime.data import (  # noqa: E402
    download_vix,
    download_yfinance_prices,
    load_prices_csv,
    prices_to_returns,
)


ETF_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "HYG",
    "LQD",
    "GLD",
    "USO",
    "XLK",
    "XLF",
    "XLE",
    "XLU",
]


def main() -> int:
    output_dir = PROJECT_ROOT / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        prices = download_yfinance_prices(ETF_UNIVERSE, start="2015-01-01")
        returns = prices_to_returns(prices, method="log")
        vix = download_vix(start=str(returns.index.min().date()))
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"Online example could not be completed: {exc}")
        print(
            "Use local CSV files instead, for example:\n"
            "  prices = load_prices_csv('data/raw/prices.csv')\n"
            "  returns = prices_to_returns(prices, method='log')\n"
            "Then pass local benchmark series into create_stress_labels."
        )
        return 0

    spy_drawdown = compute_market_drawdown(prices["SPY"])
    realized_volatility = compute_realized_volatility(returns, window=21)
    average_correlation = compute_correlation_regime_score(returns, window=126)
    stress_labels = create_stress_labels(
        vix=vix,
        drawdown=spy_drawdown,
        realized_vol=realized_volatility,
        correlation_score=average_correlation,
    )

    returns.to_csv(output_dir / "returns.csv")
    stress_labels.to_csv(output_dir / "benchmark_stress_labels.csv")

    print(f"Saved returns to {output_dir / 'returns.csv'}")
    print(f"Saved stress labels to {output_dir / 'benchmark_stress_labels.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
