"""Tests for the meta-labeled machine-learning strategy.

These are marked slow because they fit gradient-boosted models under
cross-validation. Deselect them with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cryptoquant.data.synthetic import make_feature_dataset
from cryptoquant.trading.strategies import MetaLabelML, TrendCarry

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def fitted():
    """A model fitted on a small synthetic universe containing a real edge."""
    features, prices, funding = make_feature_dataset(
        ("A", "B"), n=6_000, seed=17, trend_strength=0.05
    )
    model = MetaLabelML(primary=TrendCarry(), n_splits=3)
    model.fit(features)
    return model, features, prices, funding


def test_fit_produces_a_model_per_symbol(fitted):
    model, features, _, _ = fitted
    assert set(model.models_) <= set(features)
    assert model.models_


def test_events_carry_side_exit_and_probability(fitted):
    model, _, _, _ = fitted
    for events in model.events_.values():
        assert {"side", "t1", "proba", "size"} <= set(events.columns)
        assert (events["t1"] >= events.index).all()


def test_out_of_sample_signal_is_bounded(fitted):
    model, features, prices, _ = fitted
    signal = model.signal(features, use_oos=True)
    assert signal.abs().max().max() <= 1.0
    assert not signal.isna().any().any()
    assert set(signal.columns) == set(prices.columns)


def test_positions_are_held_not_recomputed_each_bar(fitted):
    """A bet must persist to its barrier rather than flickering with the model.

    Recomputing the position every bar is the standard mistake: it produces
    turnover driven by model noise, which costs more than the signal is worth.
    """
    model, features, _, _ = fitted
    signal = model.signal(features, use_oos=True)
    column = signal.iloc[:, 0]
    active = column[column != 0]
    if active.empty:
        pytest.skip("no bets were taken on this sample")
    changes = (column.diff() != 0).mean()
    assert changes < 0.5


def test_live_signal_uses_the_full_sample_model(fitted):
    model, features, _, _ = fitted
    live = model.signal(features, use_oos=False)
    oos = model.signal(features, use_oos=True)
    assert live.shape == oos.shape
    assert np.isfinite(live.to_numpy()).all()


def test_signal_before_fit_raises():
    model = MetaLabelML(primary=TrendCarry())
    with pytest.raises(RuntimeError, match="fit"):
        model.signal({})


def test_feature_importance_is_recorded(fitted):
    model, _, _, _ = fitted
    assert model.feature_importance_ is not None
    assert "gain" in model.feature_importance_
