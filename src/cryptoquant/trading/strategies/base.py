"""Strategy interface.

A strategy maps a dictionary of per-symbol feature frames onto a signal frame:
one column per symbol, one row per bar, values in roughly ``[-1, 1]`` where the
sign is the desired direction and the magnitude is conviction.

Signals are deliberately *not* position sizes. Converting conviction into
capital is the job of :mod:`cryptoquant.trading.risk`, which applies volatility
targeting and hard limits. Keeping the two separate means a signal can be
evaluated independently of how aggressively it happens to be sized.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

__all__ = ["Strategy"]


class Strategy:
    """Base class for all signal generators.

    Subclasses must implement :meth:`signal`. Strategies that learn from data
    should also override :meth:`fit`; stateless rule-based strategies can rely
    on the default no-op implementation.

    Attributes:
        name: Short identifier used in reports and plot titles.
    """

    name: str = "base"

    def fit(self, data: Mapping[str, pd.DataFrame]) -> Strategy:
        """Estimate any parameters the strategy needs from historical data.

        The default implementation does nothing, which is correct for
        rule-based strategies with no learned state.

        Args:
            data: Feature frames keyed by symbol, each indexed by UTC timestamp.

        Returns:
            ``self``, so that ``fit`` can be chained.
        """
        return self

    def signal(self, data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Produce the target signal for every symbol and bar.

        Implementations must only use information available at or before each
        bar's close. Introducing a forward-looking value here silently
        invalidates every downstream result.

        Args:
            data: Feature frames keyed by symbol, each indexed by UTC timestamp.

        Returns:
            A frame indexed by timestamp with one column per symbol and values
            in roughly ``[-1, 1]``.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError
