"""Averaging several strategies together.

Diversification across *models* is the cheapest genuine Sharpe improvement
available, because model error is far less correlated than market exposure is.
Two mediocre, weakly correlated signals routinely beat either one alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from .base import Strategy

__all__ = ["Ensemble"]


@dataclass
class Ensemble(Strategy):
    """Average the signals of several strategies.

    Attributes:
        members: Strategies to combine.
        weights: Blending weights. Defaults to equal weighting.
        name: Strategy identifier.
    """

    members: list[Strategy] = field(default_factory=list)
    weights: list[float] | None = None
    name: str = "ensemble"

    def fit(self, data: Mapping[str, pd.DataFrame]) -> Ensemble:
        """Fit every member strategy.

        Args:
            data: Feature frames keyed by symbol.

        Returns:
            ``self``.
        """
        for m in self.members:
            m.fit(data)
        return self

    def signal(self, data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine member signals into a weighted average.

        Args:
            data: Feature frames keyed by symbol.

        Returns:
            The blended signal, clipped to ``[-1, 1]``.
        """
        sigs = [m.signal(data) for m in self.members]
        w = self.weights or [1.0 / len(sigs)] * len(sigs)
        out = sum(s * wi for s, wi in zip(sigs, w, strict=True))
        return out.clip(-1, 1)
