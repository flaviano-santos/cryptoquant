"""Time-series momentum, cross-sectional momentum and funding carry.

This is the baseline every other strategy is measured against, and it is built
only from effects with decades of out-of-sample evidence across asset classes:

* **Time-series momentum** - assets that have risen tend to keep rising over
  horizons of weeks to months (Moskowitz, Ooi & Pedersen, 2012).
* **Cross-sectional momentum** - ranking within a universe hedges out the market
  factor, which in crypto dominates everything else.
* **Carry** - the perpetual funding rate is a directly observable price of
  leverage. Extreme positive funding means longs are crowded and paying to stay.

Every component is normalised to roughly unit scale before blending, so the
weights mean what they appear to mean. If a more elaborate model cannot beat
this net of costs, it has not earned its complexity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import Strategy

__all__ = ["TrendCarry"]


@dataclass
class TrendCarry(Strategy):
    """Signal = w_ts * time-series momentum + w_cs * cross-sectional momentum + w_carry * negative funding z-score gated by a trend-quality filter.

    Every component is normalised to roughly unit scale before blending, so the
    weights mean what they look like.
    """

    lookbacks: tuple[int, ...] = (72, 168, 336, 720)
    w_ts: float = 0.5
    w_cs: float = 0.3
    w_carry: float = 0.2
    vol_window: int = 168
    squash: float = 1.0  # tanh temperature; lower = more binary
    name: str = "trend_carry"

    def signal(self, data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Blend momentum and carry into a single signal per symbol.

        Args:
            data: Feature frames keyed by symbol, as produced by
                :func:`cryptoquant.research.features.build_features`.

        Returns:
            A frame indexed by timestamp with one column per symbol, clipped to
            ``[-1, 1]``.
        """
        # ---- time-series momentum, averaged over horizons ------------------
        ts_parts = {}
        for sym, X in data.items():
            cols = [f"tsmom_{k}" for k in self.lookbacks if f"tsmom_{k}" in X]
            if not cols:
                continue
            ts_parts[sym] = X[cols].mean(axis=1)
        ts = pd.DataFrame(ts_parts)
        ts = np.tanh(ts / self.squash)

        # ---- cross-sectional momentum: rank within the universe ------------
        raw_mom = pd.DataFrame({s: X["mom_168"] for s, X in data.items() if "mom_168" in X})
        if raw_mom.shape[1] > 1:
            cs = (raw_mom.rank(axis=1, pct=True) - 0.5) * 2.0
            cs = cs.sub(cs.mean(axis=1), axis=0)  # dollar-neutral
        else:
            cs = pd.DataFrame(0.0, index=ts.index, columns=ts.columns)

        # ---- carry: pay attention when funding is extreme ------------------
        if any("funding_z_168" in X for X in data.values()):
            fz = pd.DataFrame(
                {s: X.get("funding_z_168") for s, X in data.items() if "funding_z_168" in X}
            )
            carry = -np.tanh(fz / 2.0)  # crowded longs -> fade
        else:
            carry = pd.DataFrame(0.0, index=ts.index, columns=ts.columns)

        cs = cs.reindex_like(ts).fillna(0.0)
        carry = carry.reindex_like(ts).fillna(0.0)
        sig = self.w_ts * ts.fillna(0.0) + self.w_cs * cs + self.w_carry * carry

        # ---- regime gate: stand down when trend quality is poor ------------
        gate = self._trend_gate(data, sig.index, sig.columns)
        sig = sig * gate

        return sig.clip(-1, 1).fillna(0.0)

    @staticmethod
    def _trend_gate(data, index, columns) -> pd.DataFrame:
        """Scale down when short-horizon vol spikes relative to long-horizon vol.

        Vol explosions are where trend strategies give back a year of gains.
        """
        g = {}
        for sym in columns:
            X = data.get(sym)
            if X is None or "vol_ratio" not in X:
                g[sym] = pd.Series(1.0, index=index)
                continue
            vr = X["vol_ratio"].reindex(index)
            g[sym] = (1.0 / vr.clip(lower=0.5)).clip(0.25, 1.0).fillna(1.0)
        return pd.DataFrame(g)
