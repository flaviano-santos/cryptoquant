"""Data acquisition and local storage.

Three sources, in descending order of research value:

1. :mod:`~cryptoquant.data.binance_vision` - free bulk historical archives.
2. :mod:`~cryptoquant.data.feeds` - CCXT access for recent and live data.
3. :mod:`~cryptoquant.data.alt_data` - free sentiment and on-chain series.

Everything lands in :class:`~cryptoquant.data.store.Store`, a Parquet lake
queried through DuckDB. :mod:`~cryptoquant.data.synthetic` generates artificial
markets for null testing.
"""

from .alt_data import defillama_tvl, fear_greed, stablecoin_supply
from .binance_vision import BinanceVision
from .feeds import ccxt_ohlcv, fetch_price, list_usdt_pairs
from .store import Store, ingest
from .synthetic import make_feature_dataset, make_ohlcv, make_universe

__all__ = [
    "BinanceVision",
    "Store",
    "ccxt_ohlcv",
    "defillama_tvl",
    "fear_greed",
    "fetch_price",
    "ingest",
    "list_usdt_pairs",
    "make_feature_dataset",
    "make_ohlcv",
    "make_universe",
    "stablecoin_supply",
]
