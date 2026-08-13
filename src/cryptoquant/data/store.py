"""Local columnar warehouse: Parquet files queried through DuckDB.

DuckDB reads Parquet directly, so a laptop can run SQL over tens of gigabytes of
bars with no server process and no separate ingest step. Datasets are partitioned
on disk by symbol and interval, which keeps partial re-downloads cheap.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import duckdb
import pandas as pd

from .binance_vision import BinanceVision

logger = logging.getLogger(__name__)

__all__ = ["Store", "ingest"]


class Store:
    """Parquet lake + DuckDB. DuckDB reads Parquet directly, so you get SQL over tens of gigabytes of bars on a laptop with no server and no ingest step."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "raw").mkdir(exist_ok=True)
        self.con = duckdb.connect(str(self.root / "cq.duckdb"))

    # -- paths --------------------------------------------------------------
    def _path(self, dataset: str, symbol: str, interval: str | None = None) -> Path:
        """Resolve the on-disk Parquet path for a dataset partition."""
        parts = [dataset, f"symbol={symbol}"]
        if interval:
            parts.append(f"interval={interval}")
        p = self.root / Path(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p / "data.parquet"

    # -- write / read -------------------------------------------------------
    def write(
        self,
        df: pd.DataFrame,
        dataset: str,
        symbol: str,
        interval: str | None = None,
        merge: bool = True,
    ) -> Path:
        """Write a frame to the lake, optionally merging with existing rows.

        Args:
            df: Data to store; must contain a ``ts`` column.
            dataset: Logical dataset name, e.g. ``"klines"``.
            symbol: Instrument the data belongs to.
            interval: Bar interval, for datasets that have one.
            merge: If ``True``, combine with any existing partition and drop
                duplicate timestamps, keeping the newly written rows. This makes
                incremental top-ups idempotent.

        Returns:
            The path written to.
        """
        path = self._path(dataset, symbol, interval)
        if merge and path.exists():
            old = pd.read_parquet(path)
            df = (
                pd.concat([old, df], ignore_index=True)
                .drop_duplicates(subset="ts", keep="last")
                .sort_values("ts")
                .reset_index(drop=True)
            )
        df.to_parquet(path, index=False, compression="zstd")
        return path

    def read(
        self,
        dataset: str,
        symbol: str,
        interval: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Read one dataset partition, optionally restricted to a date range.

        Args:
            dataset: Logical dataset name.
            symbol: Instrument to read.
            interval: Bar interval, for datasets that have one.
            start: Inclusive lower bound on ``ts``.
            end: Inclusive upper bound on ``ts``.

        Returns:
            The stored rows, or an empty frame if the partition does not exist.
        """
        path = self._path(dataset, symbol, interval)
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if start is not None:
            df = df[df["ts"] >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df["ts"] <= pd.Timestamp(end, tz="UTC")]
        return df.reset_index(drop=True)

    def read_many(
        self,
        dataset: str,
        symbols: Iterable[str],
        interval: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Read several symbols at once.

        Args:
            dataset: Logical dataset name.
            symbols: Instruments to read.
            interval: Bar interval, for datasets that have one.
            start: Inclusive lower bound on ``ts``.
            end: Inclusive upper bound on ``ts``.

        Returns:
            A mapping of symbol to frame.
        """
        return {s: self.read(dataset, s, interval, start, end) for s in symbols}

    def sql(self, query: str) -> pd.DataFrame:
        """Execute raw DuckDB SQL against the lake.

        Reference Parquet directly, e.g.
        ``read_parquet('<root>/klines/**/*.parquet')``.

        Args:
            query: A SQL statement.

        Returns:
            The result as a frame.
        """
        return self.con.execute(query).df()

    def panel(
        self,
        symbols: Sequence[str],
        interval: str,
        field: str = "close",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Assemble a wide panel of one field across several symbols.

        Args:
            symbols: Instruments to include as columns.
            interval: Bar interval.
            field: Column to extract, e.g. ``"close"``.
            start: Inclusive lower bound on ``ts``.
            end: Inclusive upper bound on ``ts``.

        Returns:
            A frame indexed by UTC timestamp with one column per symbol.
        """
        cols = {}
        for s in symbols:
            d = self.read("klines", s, interval, start, end)
            if not d.empty:
                cols[s] = d.set_index("ts")[field]
        if not cols:
            return pd.DataFrame()
        return pd.DataFrame(cols).sort_index()


# ---------------------------------------------------------------------------
# One-call ingestion
# ---------------------------------------------------------------------------
def ingest(
    cfg, symbols: Sequence[str] | None = None, with_funding: bool = True, with_metrics: bool = False
) -> Store:
    """Download everything the config asks for into the local store."""
    store = Store(cfg.data_root)
    bv = BinanceVision(cache_dir=store.root / "raw", market=cfg.get("data.market", "futures_um"))
    symbols = list(symbols or cfg.get("data.symbols"))
    interval = cfg.get("data.interval", "1h")
    start, end = cfg.get("data.start"), cfg.get("data.end")

    for sym in symbols:
        logger.info("klines %s", sym)
        k = bv.klines(sym, interval, start, end)
        if not k.empty:
            store.write(k, "klines", sym, interval)
            logger.info(
                "  %s bars %s -> %s", len(k), k["ts"].iloc[0].date(), k["ts"].iloc[-1].date()
            )
        if with_funding:
            f = bv.funding(sym, start, end)
            if not f.empty:
                store.write(f, "funding", sym)
        if with_metrics:
            m = bv.metrics(sym, max(str(start), "2023-01-01"), end)
            if not m.empty:
                store.write(m, "metrics", sym)
    return store
