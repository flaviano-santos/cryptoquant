"""Tests for the strategy implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoquant.research.features import build_features
from cryptoquant.trading.strategies import (
    AdaptiveMultiFactor,
    Ensemble,
    MovingAverageCrossover,
    Strategy,
    TrendCarry,
)


@pytest.fixture(scope="module")
def features(universe) -> dict[str, pd.DataFrame]:
    """Feature frames for the shared synthetic universe."""
    return {symbol: build_features(frame) for symbol, frame in universe.items()}


def test_base_strategy_requires_an_implementation():
    with pytest.raises(NotImplementedError):
        Strategy().signal({})


class TestMovingAverageCrossover:
    """The long-only trend benchmark."""

    def test_rejects_inverted_windows(self):
        with pytest.raises(ValueError, match="must be <"):
            MovingAverageCrossover(fast_window=100, slow_window=20)

    def test_long_only_by_default(self, features):
        signal = MovingAverageCrossover().signal(features)
        assert signal.min().min() >= 0.0
        assert signal.max().max() <= 1.0

    def test_short_leg_can_be_enabled(self, features):
        signal = MovingAverageCrossover(allow_short=True).signal(features)
        assert signal.min().min() == -1.0

    def test_flat_before_the_slow_window_fills(self, features):
        signal = MovingAverageCrossover(fast_window=20, slow_window=100).signal(features)
        assert (signal.iloc[:99] == 0).all().all()

    def test_signal_is_actually_a_crossover(self, features):
        """Where the signal is long, the fast average must exceed the slow one."""
        symbol = next(iter(features))
        close = features[symbol]["close"]
        fast = close.rolling(20, min_periods=20).mean()
        slow = close.rolling(100, min_periods=100).mean()
        signal = MovingAverageCrossover().signal(features)[symbol]
        longs = signal == 1.0
        assert (fast[longs] > slow[longs]).all()


class TestTrendCarry:
    """The blended trend-and-carry baseline."""

    def test_signal_is_bounded(self, features):
        signal = TrendCarry().signal(features)
        assert signal.abs().max().max() <= 1.0

    def test_covers_every_symbol(self, features):
        assert set(TrendCarry().signal(features).columns) == set(features)

    def test_no_missing_values(self, features):
        assert not TrendCarry().signal(features).isna().any().any()

    def test_weights_change_the_signal(self, features):
        trend_only = TrendCarry(w_ts=1.0, w_cs=0.0, w_carry=0.0).signal(features)
        cross_only = TrendCarry(w_ts=0.0, w_cs=1.0, w_carry=0.0).signal(features)
        assert not np.allclose(trend_only.to_numpy(), cross_only.to_numpy())


class TestAdaptiveMultiFactor:
    """Regime-adaptive diversified signal."""

    def test_signal_is_bounded_complete_and_finite(self, features):
        signal = AdaptiveMultiFactor().signal(features)
        assert set(signal.columns) == set(features)
        assert signal.abs().max().max() <= 1.0
        assert np.isfinite(signal.to_numpy()).all()

    def test_rejects_invalid_parameters(self):
        with pytest.raises(ValueError, match="non-negative"):
            AdaptiveMultiFactor(w_trend=-0.1)
        with pytest.raises(ValueError, match="at least one"):
            AdaptiveMultiFactor(
                w_trend=0,
                w_breakout=0,
                w_mean_reversion=0,
                w_cross_sectional=0,
                w_carry=0,
            )
        with pytest.raises(ValueError, match="at least one bar"):
            AdaptiveMultiFactor(signal_halflife=0)

    def test_future_mutation_cannot_change_past_signals(self, features):
        strategy = AdaptiveMultiFactor()
        original = strategy.signal(features)
        changed = {symbol: frame.copy() for symbol, frame in features.items()}
        for frame in changed.values():
            feature_cols = [c for c in frame.columns if c != "close"]
            frame.loc[frame.index[-20:], feature_cols] = 999.0
        mutated = strategy.signal(changed)
        pd.testing.assert_frame_equal(original.iloc[:-20], mutated.iloc[:-20])

    def test_default_persistence_reduces_signal_turnover(self, features):
        raw = AdaptiveMultiFactor(signal_halflife=1).signal(features)
        smooth = AdaptiveMultiFactor().signal(features)
        assert smooth.diff().abs().sum().sum() < raw.diff().abs().sum().sum()


class TestEnsemble:
    """Averaging strategies together."""

    def test_averages_its_members(self, features):
        first, second = MovingAverageCrossover(), TrendCarry()
        combined = Ensemble(members=[first, second]).signal(features)
        expected = (first.signal(features) + second.signal(features)) / 2
        pd.testing.assert_frame_equal(combined, expected.clip(-1, 1), check_dtype=False)

    def test_respects_explicit_weights(self, features):
        member = MovingAverageCrossover()
        combined = Ensemble(members=[member], weights=[1.0]).signal(features)
        pd.testing.assert_frame_equal(combined, member.signal(features).clip(-1, 1))
