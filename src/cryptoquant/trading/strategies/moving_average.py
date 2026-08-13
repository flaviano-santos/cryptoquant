"""Moving-average crossover.

A long-only trend filter: hold the asset while a fast moving average sits above
a slow one, stand aside otherwise. It is the oldest systematic rule in the book
and it remains a fair benchmark, because most of what trend-following earns in
crypto comes from being flat during drawdowns rather than from timing entries
well.

This implementation is adapted from the author's earlier ``bitcoin-quant-lab``
project. It is retained deliberately: a simple, well-understood strategy that
has survived walk-forward validation is far more useful as a benchmark than an
elaborate one that has not.

A caution carried over from that project's review. On daily BTC from 2018,
the fixed 20/100 parameterisation produced a higher out-of-sample Sharpe than
the pair chosen by sweeping 24 combinations and taking the winner. Tuning these
windows against a backtest tends to *reduce* live performance. Treat the
defaults as a benchmark, not a starting point for optimisation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from .base import Strategy

__all__ = ["MovingAverageCrossover"]


@dataclass
class MovingAverageCrossover(Strategy):
    """Long when the fast moving average exceeds the slow one.

    Attributes:
        fast_window: Lookback of the fast moving average, in bars.
        slow_window: Lookback of the slow moving average, in bars.
        allow_short: If ``True``, go short instead of flat when the fast average
            is below the slow one. Shorting a positive-drift asset is a
            materially different bet, so this is off by default.
        name: Strategy identifier.
    """

    fast_window: int = 20
    slow_window: int = 100
    allow_short: bool = False
    name: str = "ma_crossover"

    def __post_init__(self) -> None:
        """Validate the window configuration.

        Raises:
            ValueError: If ``fast_window`` is not strictly less than
                ``slow_window``, which would make the signal meaningless.
        """
        if self.fast_window >= self.slow_window:
            raise ValueError(
                f"fast_window ({self.fast_window}) must be < slow_window ({self.slow_window})"
            )

    def signal(self, data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute the crossover signal for every symbol.

        Both averages are computed from closes up to and including the current
        bar. The backtester applies the execution lag, so no shift is applied
        here; doing it in both places would double-count the delay.

        Args:
            data: Feature frames keyed by symbol, each containing a ``close``
                column and indexed by UTC timestamp.

        Returns:
            A frame indexed by timestamp with one column per symbol, valued
            ``1.0`` (long), ``0.0`` (flat) or ``-1.0`` (short, when
            ``allow_short`` is set). Bars before the slow window has filled are
            ``0.0``.
        """
        signals: dict[str, pd.Series] = {}
        for symbol, frame in data.items():
            if "close" not in frame:
                continue
            close = frame["close"]
            fast = close.rolling(self.fast_window, min_periods=self.fast_window).mean()
            slow = close.rolling(self.slow_window, min_periods=self.slow_window).mean()

            long_leg = (fast > slow).astype(float)
            if self.allow_short:
                long_leg = long_leg - (fast < slow).astype(float)

            # Bars where either average is undefined must be flat, not long.
            signals[symbol] = long_leg.where(fast.notna() & slow.notna(), 0.0)

        return pd.DataFrame(signals).fillna(0.0)
