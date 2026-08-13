"""Tests for the backtest engine and its performance statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoquant.trading.backtest import Costs, run_backtest
from cryptoquant.trading.metrics import drawdown_series, performance_stats


def test_costs_reduce_returns(prices, rng):
    """Charging fees must lower the Sharpe of a churning strategy."""
    noise = pd.DataFrame({"X": rng.choice([-1.0, 1.0], len(prices))}, index=prices.index)
    free = run_backtest(prices, noise, costs=Costs(0, 0, 0), lag=1)
    paid = run_backtest(prices, noise, costs=Costs(4.5, 2.0, 2.0), lag=1)
    assert paid.stats["sharpe"] < free.stats["sharpe"]
    assert paid.costs.sum() > 0


def test_zero_turnover_incurs_zero_cost(prices):
    """A strategy that never trades must pay nothing."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    result = run_backtest(prices, weights, costs=Costs(50, 50, 50), lag=1)
    assert result.costs.sum() == 0.0
    assert result.returns.abs().sum() == 0.0


def test_funding_is_charged_to_longs(prices):
    """A long position must lose money to positive funding, and a short gain."""
    long_book = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    short_book = -long_book
    funding = pd.DataFrame(0.0001, index=prices.index, columns=prices.columns)

    long_result = run_backtest(prices, long_book, costs=Costs(0, 0, 0), funding=funding, lag=1)
    short_result = run_backtest(prices, short_book, costs=Costs(0, 0, 0), funding=funding, lag=1)

    assert long_result.funding_cost.sum() > 0
    assert short_result.funding_cost.sum() < 0


def test_gross_leverage_cap_binds(prices):
    """Weights exceeding the gross cap must be scaled down proportionally."""
    wide = pd.concat([prices] * 3, axis=1)
    wide.columns = ["A", "B", "C"]
    weights = pd.DataFrame(1.0, index=wide.index, columns=wide.columns)
    result = run_backtest(wide, weights, costs=Costs(0, 0, 0), lag=1, max_gross=1.5)
    assert result.positions.abs().sum(axis=1).max() == pytest.approx(1.5, abs=1e-9)


def test_sharpe_uses_the_standard_formula():
    """Sharpe must be mean/std scaled by sqrt(T), not a compounded numerator.

    Compounding the mean return while scaling volatility by the square root of
    time inflates the ratio whenever returns are large. On daily crypto data the
    overstatement is roughly a factor of two, which is large enough to turn an
    unremarkable strategy into an apparently excellent one.
    """
    returns = pd.Series(np.full(365, 0.001))
    returns.iloc[::2] = -0.0005  # give it non-zero variance

    stats = performance_stats(returns, bars_per_year=365)
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(365)
    assert stats["sharpe"] == pytest.approx(expected, rel=1e-9)

    inflated = ((1 + returns.mean()) ** 365 - 1) / (returns.std(ddof=1) * np.sqrt(365))
    assert stats["sharpe"] < inflated


def test_drawdown_is_negative_and_bounded():
    """Drawdowns must be non-positive and never below -100%."""
    returns = pd.Series([0.1, -0.5, 0.2, -0.1, 0.05])
    drawdowns = drawdown_series(returns)
    assert (drawdowns <= 0).all()
    assert (drawdowns > -1).all()
    assert drawdowns.min() == pytest.approx(-0.5, abs=1e-9)


def test_stats_are_empty_for_degenerate_input():
    """Constant returns have no defined Sharpe; the stats must not raise."""
    assert performance_stats(pd.Series(np.zeros(100))).empty
