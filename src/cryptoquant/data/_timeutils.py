"""Timestamp helpers shared by the data loaders.

Binance's public dumps are not internally consistent about time units: files
published before 2025 use milliseconds, later ones use microseconds, and a few
endpoints return seconds. Guessing from the column name is unreliable, so these
helpers infer the unit from the magnitude of the values themselves.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["day_range", "month_range", "to_datetime_utc"]


def to_datetime_utc(values: pd.Series) -> pd.Series:
    """Convert an integer epoch column to timezone-aware UTC timestamps.

    The epoch unit (seconds, milliseconds, microseconds or nanoseconds) is
    inferred from the median magnitude of ``values`` rather than assumed, which
    makes the loader robust to Binance changing units between archive vintages.

    Args:
        values: Integer or float epoch offsets.

    Returns:
        A UTC-localised ``datetime64[ns, UTC]`` series aligned to ``values``.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    median = float(np.nanmedian(numeric.to_numpy(dtype="float64")))
    unit: Literal["s", "ms", "us", "ns"]
    if median > 1e17:
        unit = "ns"
    elif median > 1e14:
        unit = "us"
    elif median > 1e11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(numeric, unit=unit, utc=True)


def month_range(start: str | date, end: str | date | None) -> list[tuple[int, int]]:
    """List the ``(year, month)`` pairs covered by a date range.

    The upper bound is clamped to the last *fully published* month, because
    Binance only uploads a monthly archive once the month has closed.

    Args:
        start: First date to include.
        end: Last date to include, or ``None`` for "as recent as possible".

    Returns:
        Chronologically ordered ``(year, month)`` tuples; empty if the range
        contains no complete month.
    """
    first = pd.Timestamp(start).to_period("M")
    requested = (pd.Timestamp(end) if end else pd.Timestamp.utcnow().tz_localize(None)).to_period(
        "M"
    )
    last_complete = pd.Timestamp.utcnow().tz_localize(None).to_period("M") - 1
    last = min(requested, last_complete)
    if last < first:
        return []
    return [(p.year, p.month) for p in pd.period_range(first, last, freq="M")]


def day_range(start: str | date, end: str | date | None) -> list[date]:
    """List the calendar days covered by a date range, excluding today.

    Today is excluded because the current day's archive is still being written.

    Args:
        start: First date to include.
        end: Last date to include, or ``None`` for "up to yesterday".

    Returns:
        Chronologically ordered ``datetime.date`` objects.
    """
    first = pd.Timestamp(start).normalize()
    requested = (
        pd.Timestamp(end).normalize()
        if end
        else pd.Timestamp.utcnow().tz_localize(None).normalize()
    )
    last = min(
        requested, pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)
    )
    if last < first:
        return []
    return [ts.date() for ts in pd.date_range(first, last, freq="D")]
