"""How to put your own strategy through the validation gates.

This is the template to copy when you have an idea you want to test honestly.
It demonstrates the sequence that matters:

1. Run the idea on data containing no edge, and confirm it finds nothing.
2. Compare against both benchmarks, net of costs.
3. Sweep parameters while *keeping every result*, not just the winner.
4. Deflate the winner's Sharpe by the true number of trials.
5. Check the probability of backtest overfitting.
6. Re-select parameters walk-forward and see what actually survives.

Usage::

    python examples/02_validate_your_own_strategy.py
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module

import numpy as np
import pandas as pd

from cryptoquant.config import load_config
from cryptoquant.data.synthetic import make_feature_dataset
from cryptoquant.logging_utils import setup_logging
from cryptoquant.pipeline import buy_and_hold, evaluate, parameter_sweep
from cryptoquant.research.validation import (
    deflated_sharpe,
    probability_of_backtest_overfitting,
    walk_forward_splits,
)
from cryptoquant.trading.metrics import performance_stats
from cryptoquant.trading.strategies import Strategy

quickstart = import_module("examples.01_quickstart") if False else None


class MeanReversion(Strategy):
    """Fade short-horizon moves, scaled by their size in volatility units.

    A worked example of a strategy worth testing: it has an economic rationale
    (liquidity provision earns a premium when flow is one-sided) and exactly one
    parameter, so the multiple-testing burden stays small.

    Attributes:
        lookback: Horizon in bars over which the move is measured.
        name: Strategy identifier.
    """

    def __init__(self, lookback: int = 24) -> None:
        self.lookback = lookback
        self.name = f"mean_reversion_{lookback}"

    def signal(self, data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Return a signal that is short after rallies and long after selloffs.

        Args:
            data: Feature frames keyed by symbol.

        Returns:
            A signal frame in ``[-1, 1]``.
        """
        signals = {}
        for symbol, frame in data.items():
            close = frame["close"]
            move = np.log(close / close.shift(self.lookback))
            scale = move.rolling(720, min_periods=360).std()
            signals[symbol] = -np.tanh(move / scale.replace(0, np.nan))
        return pd.DataFrame(signals).fillna(0.0)


def main() -> None:
    """Run the full validation sequence on the example strategy."""
    setup_logging()
    config = load_config()

    def build(trend_strength: float):
        return make_feature_dataset(
            ("A", "B", "C"), n=12_000, seed=42, trend_strength=trend_strength
        )

    # ---- Gate 0: the null test ------------------------------------------
    print("=" * 60)
    print("GATE 0 - null test: this data contains no edge")
    print("=" * 60)
    features, prices, funding = build(trend_strength=0.0)
    null_result, _ = evaluate(MeanReversion(), features, prices, funding, config)
    print(f"  Sharpe on pure noise: {null_result.stats['sharpe']:+.2f}")
    print("  A large positive number here would mean the process is broken.\n")

    # ---- Gates 1-2: benchmarks and costs ---------------------------------
    features, prices, funding = build(trend_strength=0.03)
    benchmark = buy_and_hold(prices, config)
    result, _ = evaluate(MeanReversion(), features, prices, funding, config)

    print("=" * 60)
    print("GATES 1-2 - benchmarks and costs")
    print("=" * 60)
    print(f"  buy and hold      Sharpe {benchmark.stats['sharpe']:+.2f}")
    print(f"  strategy (net)    Sharpe {result.stats['sharpe']:+.2f}")
    print(f"  strategy (gross)  Sharpe {result.stats['gross_sharpe']:+.2f}")
    print(f"  Sharpe lost to costs     {result.stats['cost_drag_sharpe']:.2f}")
    print(f"  annual turnover          {result.stats['ann_turnover']:.0f}\n")

    # ---- Gates 3: sweep, deflate, and measure overfitting ----------------
    grid = [{"lookback": lb} for lb in (6, 12, 24, 48, 72, 120, 168, 336)]
    perf_matrix, table = parameter_sweep(MeanReversion, grid, features, prices, funding, config)
    n_trials = perf_matrix.shape[1]
    best = table["sharpe"].idxmax()

    per_bar_sharpes = (perf_matrix.mean() / perf_matrix.std(ddof=1)).to_numpy()
    deflated = deflated_sharpe(
        perf_matrix[best], n_trials=n_trials, all_trial_sharpes=per_bar_sharpes
    )
    pbo = probability_of_backtest_overfitting(perf_matrix, s=12)

    print("=" * 60)
    print(f"GATE 3 - {n_trials} variants tried; deflate the winner")
    print("=" * 60)
    print(f"  best variant             {best} (Sharpe {table.loc[best, 'sharpe']:+.2f})")
    print(f"  deflated Sharpe          {deflated['dsr']:.3f}   (needs > 0.90)")
    print(f"  PBO                      {pbo['pbo']:.2f}    (needs < 0.50)")
    print(f"  P(out-of-sample loss)    {pbo['prob_oos_loss']:.2f}")
    print(f"  degradation slope        {pbo['degradation_slope']:+.2f}\n")

    # ---- Gate 4: walk-forward selection ----------------------------------
    out_of_sample = []
    for train_idx, test_idx in walk_forward_splits(len(perf_matrix), n_splits=6, min_train=0.4):
        in_sample = perf_matrix.iloc[train_idx]
        chosen = (in_sample.mean() / in_sample.std(ddof=1)).idxmax()
        out_of_sample.append(perf_matrix[chosen].iloc[test_idx])
    stitched = pd.concat(out_of_sample)
    walk_forward = performance_stats(stitched, bars_per_year=config.get("backtest.bars_per_year"))

    print("=" * 60)
    print("GATE 4 - walk-forward: choose on the past, trade forward")
    print("=" * 60)
    print(f"  in-sample best Sharpe    {table.loc[best, 'sharpe']:+.2f}")
    print(f"  walk-forward Sharpe      {walk_forward['sharpe']:+.2f}")
    print(f"  max drawdown             {walk_forward['max_drawdown']:.1%}")
    print(
        "\n  The walk-forward number is the honest one. If it is much worse "
        "than the in-sample best, your parameter selection is not transferring."
    )


if __name__ == "__main__":
    main()
