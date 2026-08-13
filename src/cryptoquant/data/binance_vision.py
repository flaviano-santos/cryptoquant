"""Bulk historical market data from Binance Vision.

`data.binance.vision <https://data.binance.vision>`_ publishes free ZIP archives
of klines, aggregated trades, funding rates and open-interest metrics for every
symbol Binance lists, going back to 2017. There is no API key, no rate limit and
no cost, which makes it the natural foundation for any local research stack.

Downloads are cached on disk and 404s are remembered, so re-running an ingest
only fetches archives that are genuinely new.
"""

from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from ._timeutils import day_range, month_range, to_datetime_utc

logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data"
"""Root of the public archive."""

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]
"""Column order of the kline CSVs, which ship without a header before 2025."""

FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
"""Column order of the funding-rate CSVs."""

METRICS_COLUMNS = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]
"""Column order of the open-interest / positioning CSVs."""

MARKET_PATHS = {"spot": "spot", "futures_um": "futures/um", "futures_cm": "futures/cm"}
"""Maps a friendly market name onto its path segment in the archive."""

__all__ = [
    "BASE_URL",
    "FUNDING_COLUMNS",
    "KLINE_COLUMNS",
    "MARKET_PATHS",
    "METRICS_COLUMNS",
    "BinanceVision",
]


@dataclass
class BinanceVision:
    """Downloads and caches Binance's public historical dumps."""

    cache_dir: Path
    market: str = "futures_um"
    workers: int = 8
    timeout: int = 60

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.market not in MARKET_PATHS:
            raise ValueError(f"market must be one of {list(MARKET_PATHS)}")
        self._session = requests.Session()

    # -- url builders -------------------------------------------------------
    def _kline_url(self, symbol: str, interval: str, y: int, m: int, d: int | None) -> str:
        mp = MARKET_PATHS[self.market]
        if d is None:
            return f"{BASE_URL}/{mp}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{y:04d}-{m:02d}.zip"
        return f"{BASE_URL}/{mp}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{y:04d}-{m:02d}-{d:02d}.zip"

    def _funding_url(self, symbol: str, y: int, m: int) -> str:
        mp = MARKET_PATHS[self.market]
        return (
            f"{BASE_URL}/{mp}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{y:04d}-{m:02d}.zip"
        )

    def _metrics_url(self, symbol: str, d: date) -> str:
        mp = MARKET_PATHS[self.market]
        return f"{BASE_URL}/{mp}/daily/metrics/{symbol}/{symbol}-metrics-{d:%Y-%m-%d}.zip"

    # -- fetching -----------------------------------------------------------
    def _fetch_zip(self, url: str) -> bytes | None:
        """Returns raw zip bytes, cached on disk. None if the file does not exist."""
        name = url.split("/data/", 1)[-1].replace("/", "__")
        cached = self.cache_dir / name
        missing_marker = cached.with_suffix(".missing")
        if cached.exists():
            return cached.read_bytes()
        if missing_marker.exists():
            return None
        try:
            r = self._session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:  # transient network problem
            logger.warning("fetch failed %s: %s", url, exc)
            return None
        if r.status_code == 404:
            missing_marker.touch()  # remember: not published
            return None
        r.raise_for_status()
        cached.write_bytes(r.content)
        return r.content

    @staticmethod
    def _read_csv_from_zip(blob: bytes, cols: Sequence[str]) -> pd.DataFrame:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                head = fh.read(2048)
        first = head.split(b"\n", 1)[0].decode("utf8", "ignore").lower()
        # newer dumps ship a header row; older ones do not
        has_header = any(
            c.split("_")[0] in first for c in ("open_time", "calc_time", "create_time")
        )
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            name = zf.namelist()[0]
            df = pd.read_csv(
                zf.open(name),
                header=0 if has_header else None,
                names=None if has_header else list(cols),
                low_memory=False,
            )
        if has_header:
            df.columns = [str(c).strip() for c in df.columns]
            if len(df.columns) == len(cols):
                df.columns = list(cols)
        return df

    def _parallel(self, jobs: Sequence[tuple[str, Sequence[str]]]) -> Iterator[pd.DataFrame]:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futs = {pool.submit(self._fetch_zip, url): (url, cols) for url, cols in jobs}
            for fut in as_completed(futs):
                url, cols = futs[fut]
                blob = fut.result()
                if not blob:
                    continue
                try:
                    yield self._read_csv_from_zip(blob, cols)
                except Exception as exc:  # corrupt cache entry
                    logger.warning("parse failed %s: %s", url, exc)

    # -- public API ---------------------------------------------------------
    def klines(
        self,
        symbol: str,
        interval: str,
        start: str,
        end: str | None = None,
        fill_recent_days: bool = True,
    ) -> pd.DataFrame:
        """OHLCV bars. Monthly archives for the bulk, daily archives to fill the current partial month."""
        jobs = [
            (self._kline_url(symbol, interval, y, m, None), KLINE_COLUMNS)
            for y, m in month_range(start, end)
        ]
        if fill_recent_days:
            this_month_start = pd.Timestamp.utcnow().tz_localize(None).normalize().replace(day=1)
            for d in day_range(max(pd.Timestamp(start), this_month_start), end):
                jobs.append(
                    (self._kline_url(symbol, interval, d.year, d.month, d.day), KLINE_COLUMNS)
                )

        frames = list(self._parallel(jobs))
        if not frames:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        df = pd.concat(frames, ignore_index=True)
        df["ts"] = to_datetime_utc(df["open_time"])
        num = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
        ]
        for c in num:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = (
            df.drop(columns=[c for c in ("open_time", "close_time", "ignore") if c in df])
            .dropna(subset=["ts", "close"])
            .drop_duplicates(subset="ts")
            .sort_values("ts")
            .reset_index(drop=True)
        )
        df.insert(0, "symbol", symbol)
        return df

    def funding(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        """Realised 8-hourly perpetual funding rates. Only for futures markets."""
        if not self.market.startswith("futures"):
            return pd.DataFrame(columns=["ts", "funding_rate"])
        jobs = [
            (self._funding_url(symbol, y, m), FUNDING_COLUMNS) for y, m in month_range(start, end)
        ]
        frames = list(self._parallel(jobs))
        if not frames:
            return pd.DataFrame(columns=["ts", "funding_rate"])
        df = pd.concat(frames, ignore_index=True)
        df["ts"] = to_datetime_utc(df["calc_time"])
        df["funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")
        out = (
            df[["ts", "funding_rate"]]
            .dropna()
            .drop_duplicates("ts")
            .sort_values("ts")
            .reset_index(drop=True)
        )
        out.insert(0, "symbol", symbol)
        return out

    def metrics(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        """Open interest and positioning ratios at 5-minute resolution.

        Published daily, history begins around 2023 for most symbols.
        These are genuinely informative features - OI changes tell you whether
        a move is being driven by new leverage or by position unwinding.
        """
        if not self.market.startswith("futures"):
            return pd.DataFrame()
        jobs = [(self._metrics_url(symbol, d), METRICS_COLUMNS) for d in day_range(start, end)]
        frames = list(self._parallel(jobs))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["ts"] = to_datetime_utc(df["create_time"])
        for c in METRICS_COLUMNS[2:]:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return (
            df.drop(columns=["create_time"])
            .dropna(subset=["ts"])
            .drop_duplicates("ts")
            .sort_values("ts")
            .reset_index(drop=True)
        )
