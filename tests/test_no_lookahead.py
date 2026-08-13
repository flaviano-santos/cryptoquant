"""Tests that the backtester cannot see the future.

These are the most important tests in the suite. A backtester with lookahead
produces beautiful results that are entirely fictional, and the bug is invisible
in the output: the equity curve simply looks good.

The approach is adversarial. Hand the engine a signal built from information it
should not have, and assert that it refuses to profit from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoquant.trading.backtest import Costs, run_backtest

FREE = Costs(taker_fee_bps=0.0, maker_fee_bps=0.0, slippage_bps=0.0)


def _perfect_foresight(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a signal equal to the sign of the *next* bar's return."""
    forward = prices["X"].pct_change().shift(-1)
    return pd.DataFrame({"X": np.sign(forward).fillna(0.0)})


def test_zero_lag_exposes_future_information(prices):
    """With lag=0 a future-peeking signal must look absurdly profitable.

    This is a control. It proves the engine can register the edge at all, so
    that the lag=1 result below is meaningful rather than vacuous.
    """
    result = run_backtest(prices, _perfect_foresight(prices), costs=FREE, lag=0)
    assert result.stats["sharpe"] > 10


def test_execution_lag_destroys_future_information(prices):
    """With lag=1 the same signal must lose its edge entirely."""
    result = run_backtest(prices, _perfect_foresight(prices), costs=FREE, lag=1)
    assert abs(result.stats["sharpe"]) < 3


def test_constant_signal_reproduces_buy_and_hold(prices):
    """An always-long, cost-free strategy must equal buy-and-hold exactly.

    Adapted from the author's earlier ``bitcoin-quant-lab`` test suite. It is a
    strong invariant: one assertion catches sign errors, off-by-one shifts and
    incorrect compounding.
    """
    weights = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    result = run_backtest(prices, weights, costs=FREE, lag=1)

    expected = float(prices["X"].iloc[-1] / prices["X"].iloc[1] - 1)
    actual = float(result.equity.iloc[-1] / result.equity.iloc[0] - 1)
    assert actual == pytest.approx(expected, rel=1e-9)


def test_position_does_not_earn_its_own_bar(prices):
    """A position switched on at bar t must not earn the return of bar t."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    switch_on = 100
    weights.iloc[switch_on] = 1.0

    result = run_backtest(prices, weights, costs=FREE, lag=1)
    assert result.returns.iloc[switch_on] == 0.0
    assert result.returns.iloc[switch_on + 1] == 0.0
    assert result.returns.iloc[switch_on + 2] != 0.0


def test_features_do_not_correlate_with_future_returns(ohlcv):
    """No engineered feature may be strongly correlated with the next return.

    A high correlation here is the signature of an accidental negative shift.
    Real predictive features in liquid markets correlate at a few percent; a
    reading above 0.3 means information has leaked backwards.
    """
    from cryptoquant.research.features import build_features
    from cryptoquant.research.validation import leakage_check

    features = build_features(ohlcv)
    future_return = features["close"].pct_change().shift(-1)
    suspicious = leakage_check(features.drop(columns="close"), future_return, threshold=0.30)
    assert suspicious.empty, f"features appear to leak: {list(suspicious.index)}"
