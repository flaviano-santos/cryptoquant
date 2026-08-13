"""Causal, regime-adaptive multi-factor strategy.

The strategy deliberately combines return drivers rather than indicator names:
trend, breakout, range mean reversion, cross-sectional momentum and perpetual
funding carry.  All inputs are backward-looking features and the portfolio is
still executed through :mod:`cryptoquant.pipeline`, which adds an execution lag,
costs, volatility targeting and hard risk limits.

This is a research candidate, not a claim of profitability.  Its purpose is to
offer a richer *pre-registered* alternative to ``TrendCarry`` that can be
accepted or rejected by the existing validation gates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import Strategy

__all__ = ["AdaptiveMultiFactor"]


@dataclass
class AdaptiveMultiFactor(Strategy):
    """Blend complementary signals and route them by market regime.

    ``trend`` and ``breakout`` receive more weight when momentum agrees across
    horizons.  ``mean_reversion`` is activated only when that agreement is
    weak and short-run return autocorrelation is non-positive.  This avoids the
    common error of averaging trend and mean-reversion signals at all times.

    The default weights are fixed economic priors.  They should not be tuned
    on the full history; compare the defaults out of sample first.
    """

    w_trend: float = 0.35
    w_breakout: float = 0.20
    w_mean_reversion: float = 0.15
    w_cross_sectional: float = 0.20
    w_carry: float = 0.10
    trend_temperature: float = 1.25
    signal_halflife: int = 12
    name: str = "adaptive_multifactor"

    def __post_init__(self) -> None:
        weights = (
            self.w_trend,
            self.w_breakout,
            self.w_mean_reversion,
            self.w_cross_sectional,
            self.w_carry,
        )
        if any(w < 0 for w in weights):
            raise ValueError("factor weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one factor weight must be positive")
        if self.trend_temperature <= 0:
            raise ValueError("trend_temperature must be positive")
        if self.signal_halflife < 1:
            raise ValueError("signal_halflife must be at least one bar")

    @staticmethod
    def _panel(data: Mapping[str, pd.DataFrame], column: str) -> pd.DataFrame:
        """Build a symbol-wide panel for a feature, omitting absent columns."""
        return pd.DataFrame({s: X[column] for s, X in data.items() if column in X})

    def signal(self, data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Return bounded target signals using information available at each bar."""
        if not data:
            return pd.DataFrame()

        # Multi-horizon volatility-scaled trend. Agreement is a robust regime
        # measure: one noisy horizon cannot switch the whole book.
        horizons = (24, 72, 168, 336, 720)
        trend_by_symbol: dict[str, pd.Series] = {}
        agreement_by_symbol: dict[str, pd.Series] = {}
        for symbol, X in data.items():
            cols = [f"tsmom_{h}" for h in horizons if f"tsmom_{h}" in X]
            if not cols:
                continue
            parts = X[cols]
            trend_by_symbol[symbol] = parts.mean(axis=1)
            agreement_by_symbol[symbol] = parts.apply(np.sign).mean(axis=1).abs()

        raw_trend = pd.DataFrame(trend_by_symbol)
        if raw_trend.empty:
            index = next(iter(data.values())).index
            return pd.DataFrame(0.0, index=index, columns=list(data))
        trend = np.tanh(raw_trend / self.trend_temperature)
        agreement = pd.DataFrame(agreement_by_symbol).reindex_like(trend).fillna(0.0)

        # Smooth routing prevents a hard threshold from causing excess churn.
        trend_regime = ((agreement - 0.25) / 0.55).clip(0.0, 1.0)

        breakout = self._panel(data, "pos_range_168").reindex_like(trend)
        breakout = np.tanh(breakout.fillna(0.0) / 0.25)

        rsi = self._panel(data, "rsi_14").reindex_like(trend).fillna(0.0)
        mean_reversion = -np.tanh(rsi / 0.20)
        ac1 = self._panel(data, "ac1_168").reindex_like(trend)
        if not ac1.empty:
            range_regime = (1.0 - trend_regime) * (1.0 - ac1.fillna(0.0).clip(0, 0.25) / 0.25)
        else:
            range_regime = 1.0 - trend_regime

        momentum = self._panel(data, "mom_168").reindex_like(trend)
        if momentum.shape[1] > 1:
            cross_sectional = (momentum.rank(axis=1, pct=True) - 0.5) * 2.0
            cross_sectional = cross_sectional.sub(cross_sectional.mean(axis=1), axis=0)
        else:
            cross_sectional = pd.DataFrame(0.0, index=trend.index, columns=trend.columns)

        funding = self._panel(data, "funding_z_168").reindex_like(trend)
        carry = -np.tanh(funding.fillna(0.0) / 2.0)

        total_weight = (
            self.w_trend
            + self.w_breakout
            + self.w_mean_reversion
            + self.w_cross_sectional
            + self.w_carry
        )
        out = (
            self.w_trend * trend * trend_regime
            + self.w_breakout * breakout * trend_regime
            + self.w_mean_reversion * mean_reversion * range_regime
            + self.w_cross_sectional * cross_sectional
            + self.w_carry * carry
        ) / total_weight

        # Volatility shocks and deteriorating liquidity deserve less exposure,
        # not a heroic prediction about direction.
        vol_ratio = self._panel(data, "vol_ratio").reindex_like(trend).fillna(1.0)
        vol_gate = (1.5 / vol_ratio.clip(lower=1.0)).clip(0.25, 1.0)
        amihud = self._panel(data, "amihud").reindex_like(trend).fillna(0.0)
        liquidity_gate = (1.0 - 0.20 * amihud.clip(lower=0.0, upper=4.0)).clip(0.25, 1.0)

        out = (out * vol_gate * liquidity_gate).clip(-1.0, 1.0).fillna(0.0)

        # A desired position should not change radically because one hourly bar
        # changed rank.  Causal exponential persistence lowers turnover and
        # makes the research signal closer to something an exchange can fill.
        return out.ewm(halflife=self.signal_halflife, adjust=False, min_periods=1).mean()
