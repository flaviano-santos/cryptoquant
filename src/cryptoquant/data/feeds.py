"""Live and recent market data through CCXT.

`CCXT <https://github.com/ccxt/ccxt>`_ normalises the REST APIs of roughly a
hundred exchanges behind a single interface. Public market data needs no
credentials; only order placement does.

Use these helpers for the recent tail of data and for live trading. For anything
historical, :mod:`cryptoquant.data.binance_vision` is faster, unmetered and goes
back much further.
"""

from __future__ import annotations

import logging

import pandas as pd

from ._timeutils import to_datetime_utc

logger = logging.getLogger(__name__)

__all__ = ["ccxt_ohlcv", "fetch_price", "list_usdt_pairs"]

_OHLCV_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]
_MAX_BATCH = 1000


def ccxt_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str = "1h",
    since: str | None = None,
    limit_total: int = 5_000,
) -> pd.DataFrame:
    """Download recent OHLCV bars from any CCXT-supported exchange.

    Pages through the exchange's kline endpoint until ``limit_total`` bars have
    been collected or the exchange stops returning data. No API key is required.

    Args:
        exchange_id: A CCXT exchange identifier, e.g. ``"binanceusdm"``.
        symbol: Unified CCXT symbol, e.g. ``"BTC/USDT:USDT"``.
        timeframe: Bar size understood by the exchange, e.g. ``"1h"``.
        since: ISO date to start from. ``None`` starts at the exchange default.
        limit_total: Upper bound on the number of bars to return.

    Returns:
        A frame with a UTC ``ts`` column plus open/high/low/close/volume,
        de-duplicated and sorted ascending. Empty if the exchange returned
        nothing.
    """
    import ccxt

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    since_ms = int(pd.Timestamp(since).timestamp() * 1000) if since else None

    rows: list[list[float]] = []
    while len(rows) < limit_total:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=_MAX_BATCH)
        if not batch:
            break
        rows += batch
        since_ms = int(batch[-1][0]) + 1
        if len(batch) < _MAX_BATCH:
            break

    frame = pd.DataFrame(rows, columns=_OHLCV_COLUMNS)
    if frame.empty:
        return frame
    frame["ts"] = to_datetime_utc(frame["open_time"])
    return (
        frame.drop(columns="open_time")
        .drop_duplicates("ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )


def list_usdt_pairs(exchange_id: str = "binance", limit: int | None = None) -> list[str]:
    """List active USDT-quoted markets on an exchange.

    Useful for building a trading universe programmatically instead of
    hard-coding symbols. Note that the result reflects markets listed *today*,
    so backtesting over it introduces survivorship bias.

    Args:
        exchange_id: A CCXT exchange identifier.
        limit: Optional cap on the number of symbols returned.

    Returns:
        Sorted unified symbols such as ``"BTC/USDT"``. Empty on failure.
    """
    import ccxt

    try:
        exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        markets = exchange.load_markets()
    except Exception:
        logger.exception("could not load markets from %s", exchange_id)
        return []

    pairs = sorted(s for s, m in markets.items() if s.endswith("/USDT") and m.get("active"))
    return pairs[:limit] if limit else pairs


def fetch_price(symbol: str, exchange_id: str = "binance") -> float | None:
    """Fetch the last traded price for a symbol.

    Args:
        symbol: Unified CCXT symbol, e.g. ``"ETH/USDT"``.
        exchange_id: A CCXT exchange identifier.

    Returns:
        The last traded price, or ``None`` if the request failed.
    """
    import ccxt

    try:
        exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        return float(exchange.fetch_ticker(symbol)["last"])
    except Exception:
        logger.exception("could not fetch price for %s on %s", symbol, exchange_id)
        return None
