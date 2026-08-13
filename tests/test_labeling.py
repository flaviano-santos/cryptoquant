"""Tests for triple-barrier labeling and sample weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoquant.research.labeling import (
    count_concurrent,
    cusum_events,
    daily_vol,
    sample_uniqueness,
    time_decay,
    triple_barrier,
    vertical_barriers,
)


def test_cusum_samples_fewer_events_than_bars(close):
    """The filter must reduce redundancy, not label every bar."""
    volatility = daily_vol(close)
    events = cusum_events(close, volatility.fillna(volatility.median()))
    assert 0 < len(events) < len(close) / 2


def test_cusum_threshold_controls_event_count(close):
    """A wider threshold must produce strictly fewer events."""
    volatility = daily_vol(close).fillna(0.02)
    few = cusum_events(close, volatility * 3)
    many = cusum_events(close, volatility * 0.5)
    assert len(few) < len(many)


def test_labels_are_binary_or_signed(labels):
    assert set(labels["bin"].unique()) <= {-1, 0, 1}


def test_exit_never_precedes_entry(labels):
    assert (labels["t1"] >= labels.index).all()


def test_vertical_barrier_respects_the_horizon(close):
    events = close.index[100:200]
    barriers = vertical_barriers(close.index, events, num_bars=24)
    gaps = close.index.searchsorted(barriers.to_numpy()) - close.index.searchsorted(events)
    assert (gaps <= 24).all()


def test_meta_labels_are_zero_or_one(close):
    """With a side supplied, labels become 'take the bet' / 'skip it'."""
    volatility = daily_vol(close)
    events = cusum_events(close, volatility.fillna(volatility.median()))
    side = pd.Series(1.0, index=events)
    meta = triple_barrier(close, events, volatility, num_bars=24, side=side)
    assert set(meta["bin"].unique()) <= {0, 1}


def test_uniqueness_is_below_one_when_labels_overlap(close, labels):
    """Overlapping labels are not independent observations."""
    uniqueness = sample_uniqueness(close.index, labels)
    assert 0 < uniqueness.mean() < 1


def test_effective_sample_is_smaller_than_the_row_count(close, labels):
    """The number of rows overstates how much information you actually have."""
    uniqueness = sample_uniqueness(close.index, labels)
    assert uniqueness.sum() < 0.95 * len(labels)


def test_concurrency_is_non_negative(close, labels):
    counts = count_concurrent(close.index, labels)
    assert (counts >= 0).all()


def test_time_decay_favours_recent_observations(close, labels):
    """Older samples must receive no more weight than newer ones."""
    uniqueness = sample_uniqueness(close.index, labels)
    weights = time_decay(uniqueness, last_weight=0.5)
    assert weights.iloc[0] <= weights.iloc[-1]
    assert (weights >= 0).all()


def test_daily_vol_is_positive(close):
    volatility = daily_vol(close).dropna()
    assert (volatility > 0).all()
    assert np.isfinite(volatility).all()
