"""Tests for position sizing and risk controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoquant.trading.risk import (
    KillSwitch,
    RiskConfig,
    apply_turnover_buffer,
    drawdown_throttle,
    kelly_fraction_from_returns,
    vol_target_weights,
)


class TestVolatilityTargeting:
    """Converting a signal into risk-scaled position weights."""

    def test_realised_volatility_approaches_the_target(self, universe):
        """The whole point: realised book volatility should track the target."""
        prices = pd.DataFrame({s: d.set_index("ts")["close"] for s, d in universe.items()})
        signal = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
        config = RiskConfig(target_annual_vol=0.20, bars_per_year=8_760)

        weights = vol_target_weights(signal, prices, config)
        book = (weights.shift(1) * np.log(prices).diff()).sum(axis=1).dropna()
        realised = book.std() * np.sqrt(8_760)
        assert 0.10 < realised < 0.40

    def test_per_asset_cap_binds(self, universe):
        prices = pd.DataFrame({s: d.set_index("ts")["close"] for s, d in universe.items()})
        signal = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
        config = RiskConfig(max_position_per_asset=0.15, target_annual_vol=5.0)
        weights = vol_target_weights(signal, prices, config)
        assert weights.abs().max().max() <= 0.15 + 1e-9

    def test_gross_leverage_cap_binds(self, universe):
        prices = pd.DataFrame({s: d.set_index("ts")["close"] for s, d in universe.items()})
        signal = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
        config = RiskConfig(max_gross_leverage=1.0, target_annual_vol=5.0)
        weights = vol_target_weights(signal, prices, config)
        assert weights.abs().sum(axis=1).max() <= 1.0 + 1e-9

    def test_zero_signal_gives_zero_weight(self, universe):
        prices = pd.DataFrame({s: d.set_index("ts")["close"] for s, d in universe.items()})
        signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        weights = vol_target_weights(signal, prices, RiskConfig())
        assert (weights == 0).all().all()


class TestTurnoverBuffer:
    """The proportional no-trade band."""

    def test_buffer_reduces_turnover(self, rng):
        target = pd.DataFrame({"X": rng.standard_normal(2_000) * 0.2})
        raw = target.diff().abs().sum().sum()
        buffered = apply_turnover_buffer(target, buffer=0.3).diff().abs().sum().sum()
        assert buffered < raw

    def test_zero_buffer_is_a_passthrough(self, rng):
        target = pd.DataFrame({"X": rng.standard_normal(200) * 0.2})
        result = apply_turnover_buffer(target, buffer=0.0, min_band=0.0)
        pd.testing.assert_frame_equal(result, target, check_dtype=False)

    def test_output_shape_is_preserved(self, rng):
        target = pd.DataFrame(rng.standard_normal((100, 3)), columns=["A", "B", "C"])
        assert apply_turnover_buffer(target).shape == target.shape


class TestKillSwitch:
    """The runtime circuit breaker."""

    def test_daily_loss_limit_trips(self):
        switch = KillSwitch(RiskConfig(max_daily_loss=0.03))
        now = pd.Timestamp("2026-01-01 09:00", tz="UTC")
        assert switch.update(10_000, now) is None
        assert switch.update(9_600, now + pd.Timedelta(hours=1)) is not None
        assert not switch.ok

    def test_drawdown_limit_trips(self):
        switch = KillSwitch(RiskConfig(max_drawdown=0.20, max_daily_loss=0.99))
        base = pd.Timestamp("2026-01-01", tz="UTC")
        switch.update(10_000, base)
        switch.update(12_000, base + pd.Timedelta(days=1))
        assert switch.update(9_000, base + pd.Timedelta(days=2)) is not None

    def test_stays_tripped_after_recovery(self):
        """A breach must require a manual reset, not resolve itself."""
        switch = KillSwitch(RiskConfig(max_daily_loss=0.03))
        now = pd.Timestamp("2026-01-01 09:00", tz="UTC")
        switch.update(10_000, now)
        switch.update(9_000, now + pd.Timedelta(hours=1))
        switch.update(11_000, now + pd.Timedelta(hours=2))
        assert not switch.ok

    def test_new_day_resets_the_daily_baseline(self):
        switch = KillSwitch(RiskConfig(max_daily_loss=0.03))
        day_one = pd.Timestamp("2026-01-01 09:00", tz="UTC")
        switch.update(10_000, day_one)
        switch.update(9_800, day_one + pd.Timedelta(hours=2))
        assert switch.ok
        switch.update(9_800, day_one + pd.Timedelta(days=1))
        assert switch.ok


def test_drawdown_throttle_scales_between_zero_and_one():
    equity = pd.Series([100, 110, 105, 95, 88, 80])
    scale = drawdown_throttle(equity, start=0.05, stop=0.20)
    assert (scale >= 0).all() and (scale <= 1).all()
    assert scale.iloc[1] == pytest.approx(1.0)
    assert scale.iloc[-1] < scale.iloc[1]


def test_kelly_fraction_is_bounded(rng):
    returns = pd.Series(rng.standard_normal(2_000) * 0.01 + 0.0005)
    fraction = kelly_fraction_from_returns(returns, window=500, cap=1.0)
    assert fraction.abs().max() <= 1.0
