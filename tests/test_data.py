"""Tests for the data layer.

Network access is not required: the Binance archive parser is exercised against
synthetic ZIP files that reproduce every format variation the real archive has
shipped, including the 2025 switch from millisecond to microsecond timestamps.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from cryptoquant.data._timeutils import day_range, month_range, to_datetime_utc
from cryptoquant.data.binance_vision import FUNDING_COLUMNS, KLINE_COLUMNS, BinanceVision
from cryptoquant.data.store import Store

MILLIS = 1_704_067_200_000  # 2024-01-01T00:00:00Z
MICROS = MILLIS * 1_000


def _make_zip(rows: list[list], header: list[str] | None = None) -> bytes:
    """Build an in-memory ZIP in the layout Binance publishes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        body = "" if header is None else ",".join(header) + "\n"
        body += "\n".join(",".join(map(str, row)) for row in rows)
        archive.writestr("data.csv", body)
    return buffer.getvalue()


def _kline_rows(epoch: int, step: int) -> list[list]:
    return [
        [
            epoch + i * step,
            42_000 + i,
            42_100 + i,
            41_900 + i,
            42_050 + i,
            10.5,
            epoch + i * step + step - 1,
            441_000,
            120,
            5.2,
            218_000,
            0,
        ]
        for i in range(5)
    ]


class TestTimestampInference:
    """Epoch units are inferred from magnitude, not assumed."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (MILLIS, "2024-01-01"),
            (MICROS, "2024-01-01"),
            (MILLIS // 1_000, "2024-01-01"),
        ],
    )
    def test_units_are_detected(self, value, expected):
        result = to_datetime_utc(pd.Series([value] * 5))
        assert str(result.iloc[0].date()) == expected

    def test_output_is_utc_aware(self):
        assert to_datetime_utc(pd.Series([MILLIS])).dt.tz is not None


class TestArchiveParsing:
    """The CSV reader handles every published archive format."""

    @pytest.mark.parametrize(
        ("header", "epoch", "step"),
        [
            (None, MILLIS, 3_600_000),  # pre-2025, no header
            (KLINE_COLUMNS, MILLIS, 3_600_000),  # header, milliseconds
            (KLINE_COLUMNS, MICROS, 3_600_000_000),  # 2025+, microseconds
        ],
    )
    def test_kline_variants_parse_identically(self, header, epoch, step):
        blob = _make_zip(_kline_rows(epoch, step), header)
        frame = BinanceVision._read_csv_from_zip(blob, KLINE_COLUMNS)
        assert len(frame) == 5
        assert float(frame["close"].iloc[0]) == 42_050
        assert str(to_datetime_utc(frame["open_time"]).iloc[0].date()) == "2024-01-01"

    def test_funding_rows_parse(self):
        rows = [[MILLIS + i * 8 * 3_600_000, 8, 0.0001 * (i + 1)] for i in range(3)]
        frame = BinanceVision._read_csv_from_zip(_make_zip(rows, FUNDING_COLUMNS), FUNDING_COLUMNS)
        assert frame["last_funding_rate"].tolist() == [0.0001, 0.0002, 0.0003]


class TestDateRanges:
    """Range helpers exclude periods the archive has not published yet."""

    def test_month_range_excludes_the_current_month(self):
        months = month_range("2024-01-01", None)
        now = pd.Timestamp.utcnow().tz_localize(None)
        assert (now.year, now.month) not in months

    def test_day_range_excludes_today(self):
        days = day_range("2024-01-01", None)
        assert pd.Timestamp.utcnow().date() not in days

    def test_inverted_range_is_empty(self):
        assert month_range("2030-01-01", "2020-01-01") == []


class TestStore:
    """The Parquet lake."""

    def test_roundtrip(self, tmp_path: Path, ohlcv):
        store = Store(tmp_path)
        store.write(ohlcv, "klines", "TESTUSDT", "1h")
        restored = store.read("klines", "TESTUSDT", "1h")
        assert len(restored) == len(ohlcv)
        assert restored["close"].iloc[0] == pytest.approx(ohlcv["close"].iloc[0])

    def test_write_merges_and_deduplicates(self, tmp_path: Path, ohlcv):
        store = Store(tmp_path)
        store.write(ohlcv.iloc[:100], "klines", "TESTUSDT", "1h")
        store.write(ohlcv.iloc[50:200], "klines", "TESTUSDT", "1h")
        assert len(store.read("klines", "TESTUSDT", "1h")) == 200

    def test_missing_dataset_returns_empty(self, tmp_path: Path):
        assert Store(tmp_path).read("klines", "NOPE", "1h").empty

    def test_date_filtering(self, tmp_path: Path, ohlcv):
        store = Store(tmp_path)
        store.write(ohlcv, "klines", "TESTUSDT", "1h")
        cutoff = ohlcv["ts"].iloc[500]
        filtered = store.read("klines", "TESTUSDT", "1h", start=str(cutoff))
        assert (filtered["ts"] >= cutoff).all()

    def test_panel_is_wide(self, tmp_path: Path, universe):
        store = Store(tmp_path)
        for symbol, frame in universe.items():
            store.write(frame, "klines", symbol, "1h")
        panel = store.panel(list(universe), "1h")
        assert list(panel.columns) == list(universe)
