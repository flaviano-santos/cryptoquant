"""Tests for the anti-overfitting statistics.

These verify the properties that make the statistics useful: that purging
actually removes leaking samples, that the multiple-testing correction gets
harsher as trials accumulate, and that the overfitting detector fires on data
where every apparent edge is known to be noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoquant.research.validation import (
    PurgedKFold,
    bootstrap_sharpe_ci,
    deflated_sharpe,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe,
    probability_of_backtest_overfitting,
    walk_forward_splits,
)


class TestPurgedKFold:
    """Purged, embargoed cross-validation."""

    def test_no_training_label_overlaps_a_test_window(self, labels):
        """The defining property: zero overlap between train and test lifespans."""
        features = pd.DataFrame(index=labels.index).assign(dummy=1.0)
        splitter = PurgedKFold(n_splits=5, t1=labels["t1"], embargo_pct=0.02)

        for train_idx, test_idx in splitter.split(features):
            test_start = labels.index[test_idx[0]]
            test_end = labels["t1"].iloc[test_idx].max()
            train_ends = labels["t1"].iloc[train_idx]
            train_starts = labels.index[train_idx]
            overlap = ((train_ends > test_start) & (train_starts < test_end)).sum()
            assert overlap == 0

    def test_naive_kfold_would_have_leaked(self, labels):
        """Confirms the fix is load-bearing rather than decorative."""
        leaked = 0
        for fold in np.array_split(np.arange(len(labels)), 5):
            test_start = labels.index[fold[0]]
            test_end = labels["t1"].iloc[fold].max()
            mask = np.ones(len(labels), dtype=bool)
            mask[fold] = False
            leaked += int(
                ((labels["t1"][mask] > test_start) & (labels.index[mask] < test_end)).sum()
            )
        assert leaked > 0

    def test_train_and_test_never_intersect(self, labels):
        features = pd.DataFrame(index=labels.index).assign(dummy=1.0)
        splitter = PurgedKFold(n_splits=4, t1=labels["t1"], embargo_pct=0.01)
        for train_idx, test_idx in splitter.split(features):
            assert not set(train_idx) & set(test_idx)

    def test_requires_matching_index(self, labels):
        splitter = PurgedKFold(n_splits=3, t1=labels["t1"])
        mismatched = pd.DataFrame(index=labels.index[:-5]).assign(dummy=1.0)
        with pytest.raises(ValueError, match="same index"):
            list(splitter.split(mismatched))


class TestWalkForward:
    """Anchored and rolling walk-forward splits."""

    def test_test_always_follows_train(self):
        for train_idx, test_idx in walk_forward_splits(1_000, n_splits=5):
            assert train_idx.max() < test_idx.min()

    def test_expanding_window_grows(self):
        splits = walk_forward_splits(1_000, n_splits=5, expanding=True)
        sizes = [len(train) for train, _ in splits]
        assert sizes == sorted(sizes)

    def test_embargo_creates_a_gap(self):
        for train_idx, test_idx in walk_forward_splits(1_000, n_splits=4, embargo=20):
            assert test_idx.min() - train_idx.max() > 20


class TestMultipleTesting:
    """Deflated Sharpe and the expected maximum under the null."""

    def test_luck_threshold_rises_with_trial_count(self):
        assert expected_max_sharpe(500, 0.01) > expected_max_sharpe(10, 0.01) > 0

    def test_single_trial_needs_no_haircut(self):
        assert expected_max_sharpe(1, 0.01) == 0.0

    def test_same_returns_deflate_harder_after_more_trials(self, rng):
        returns = pd.Series(rng.standard_normal(8_760) * 0.01 + 0.0004)
        one = deflated_sharpe(returns, n_trials=1, sr_variance=0.0004)
        many = deflated_sharpe(returns, n_trials=500, sr_variance=0.0004)
        assert many["dsr"] < one["dsr"]

    def test_probabilistic_sharpe_is_a_probability(self, rng):
        returns = pd.Series(rng.standard_normal(2_000) * 0.01)
        assert 0.0 <= probabilistic_sharpe(returns) <= 1.0

    def test_track_record_length_is_infinite_below_target(self, rng):
        losing = pd.Series(rng.standard_normal(1_000) * 0.01 - 0.001)
        assert min_track_record_length(losing, target_sr=0.0) == np.inf


class TestOverfittingDetection:
    """Probability of backtest overfitting via combinatorially symmetric CV."""

    def test_pbo_is_high_on_pure_noise(self):
        """Selecting the best of many noise series must not generalise.

        The estimator is averaged over several draws because the combinatorial
        splits overlap heavily, so a single draw has a standard deviation of
        roughly 0.17 around the theoretical value of 0.5. Asserting on one seed
        would make this test flaky rather than informative.
        """
        estimates = []
        for seed in range(8):
            generator = np.random.default_rng(seed)
            noise = pd.DataFrame(generator.standard_normal((3_000, 20)) * 0.01)
            estimates.append(probability_of_backtest_overfitting(noise, s=10)["pbo"])
        assert np.mean(estimates) > 0.35

    def test_selection_on_noise_yields_no_out_of_sample_edge(self, rng):
        """The variant chosen in-sample must be worthless out-of-sample."""
        noise = pd.DataFrame(rng.standard_normal((3_000, 20)) * 0.01)
        result = probability_of_backtest_overfitting(noise, s=10)
        assert abs(result["oos_sharpe_mean"]) < 0.02

    def test_pbo_is_low_when_one_variant_genuinely_dominates(self, rng):
        """A real, persistent edge must survive out-of-sample selection."""
        noise = rng.standard_normal((4_000, 10)) * 0.01
        panel = pd.DataFrame(noise)
        panel[0] += 0.004  # one variant with a large, stable edge
        result = probability_of_backtest_overfitting(panel, s=10)
        assert result["pbo"] < 0.2

    def test_single_variant_is_undefined(self):
        result = probability_of_backtest_overfitting(pd.DataFrame({"only": [0.1, -0.1]}))
        assert np.isnan(result["pbo"])


class TestBootstrap:
    """Stationary block bootstrap confidence intervals."""

    def test_interval_brackets_the_point_estimate(self, rng):
        returns = pd.Series(rng.standard_normal(2_000) * 0.01 + 0.0003)
        result = bootstrap_sharpe_ci(returns, n_boot=200, bars_per_year=8_760)
        assert result["ci_low"] < result["sharpe"] < result["ci_high"]

    def test_degenerate_input_does_not_raise(self):
        result = bootstrap_sharpe_ci(pd.Series(np.zeros(500)), n_boot=50)
        assert np.isnan(result["ci_low"])
