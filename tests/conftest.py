"""Shared pytest fixtures.

Fixtures are session-scoped where generation is expensive, because several test
modules need the same synthetic market and regenerating it per test would
dominate the suite's runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoquant.data.synthetic import make_ohlcv, make_universe
from cryptoquant.research.labeling import cusum_events, daily_vol, triple_barrier


@pytest.fixture(scope="session")
def ohlcv() -> pd.DataFrame:
    """A single synthetic OHLCV series with no predictable structure."""
    return make_ohlcv(n=4_000, seed=1)


@pytest.fixture(scope="session")
def close(ohlcv: pd.DataFrame) -> pd.Series:
    """Close prices from :func:`ohlcv`, indexed by timestamp."""
    return ohlcv.set_index("ts")["close"]


@pytest.fixture(scope="session")
def prices(close: pd.Series) -> pd.DataFrame:
    """A one-column wide price frame, the shape the backtester expects."""
    return close.to_frame("X")


@pytest.fixture(scope="session")
def universe() -> dict[str, pd.DataFrame]:
    """A small correlated multi-asset synthetic universe."""
    return make_universe(("A", "B", "C"), n=3_000, seed=3)


@pytest.fixture(scope="session")
def labels(close: pd.Series) -> pd.DataFrame:
    """Triple-barrier labels on CUSUM-sampled events from :func:`close`."""
    volatility = daily_vol(close)
    events = cusum_events(close, volatility.fillna(volatility.median()))
    return triple_barrier(close, events, volatility, num_bars=48)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded random generator, so failures are reproducible."""
    return np.random.default_rng(12345)
