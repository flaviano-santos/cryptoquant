"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoquant.research.features import (
    build_features,
    frac_diff,
    frac_diff_weights,
    garman_klass,
    hurst,
    rsi,
    yang_zhang,
    zscore,
)


class TestFractionalDifferentiation:
    """Fractional differencing: stationarity while retaining memory."""

    def test_d_zero_is_the_identity(self, close):
        result = frac_diff(close, d=0.0).dropna()
        pd.testing.assert_series_equal(
            result, close.loc[result.index], check_names=False, rtol=1e-9
        )

    def test_d_one_is_the_first_difference(self, close):
        result = frac_diff(close, d=1.0).dropna()
        expected = close.diff().loc[result.index]
        assert np.corrcoef(result, expected)[0, 1] > 0.999

    def test_weights_alternate_and_shrink(self):
        weights = frac_diff_weights(0.5)
        assert weights[-1] == 1.0
        assert abs(weights[0]) < abs(weights[-1])

    def test_partial_differencing_retains_correlation_with_the_level(self, close):
        """The point of fractional d: more memory than a first difference."""
        partial = frac_diff(np.log(close), 0.4).dropna()
        full = np.log(close).diff().dropna()
        common = partial.index.intersection(full.index)
        level = np.log(close).loc[common]
        assert abs(np.corrcoef(partial.loc[common], level)[0, 1]) > abs(
            np.corrcoef(full.loc[common], level)[0, 1]
        )

    def test_vectorised_result_matches_trailing_dot_product(self):
        values = pd.Series(np.linspace(1.0, 4.0, 200))
        values.iloc[100] = np.nan
        weights = frac_diff_weights(0.4, threshold=1e-3)
        result = frac_diff(values, 0.4, threshold=1e-3)
        expected = pd.Series(np.nan, index=values.index)
        for i in range(len(weights) - 1, len(values)):
            window = values.iloc[i - len(weights) + 1 : i + 1]
            if window.notna().all():
                expected.iloc[i] = float(np.dot(weights, window))
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestVolatilityEstimators:
    """Range-based volatility estimators."""

    def test_estimators_are_positive(self, ohlcv):
        frame = ohlcv.set_index("ts")
        for estimator in (garman_klass, yang_zhang):
            values = estimator(frame).dropna()
            assert (values >= 0).all()
            assert np.isfinite(values).all()

    def test_range_estimator_is_less_noisy_than_close_to_close(self, ohlcv):
        """Using the whole bar is more efficient than using the close alone."""
        frame = ohlcv.set_index("ts")
        close_to_close = np.log(frame["close"]).diff().rolling(24).std().dropna()
        range_based = garman_klass(frame, 24).dropna()
        common = close_to_close.index.intersection(range_based.index)
        assert range_based.loc[common].std() < close_to_close.loc[common].std()


def test_rsi_is_bounded(close):
    values = rsi(close).dropna()
    assert values.min() >= 0 and values.max() <= 100


def test_zscore_is_standardised(close):
    values = zscore(close, 200).dropna()
    assert abs(values.mean()) < 1.0
    assert 0.5 < values.std() < 2.0


def test_hurst_is_in_a_plausible_range(close):
    values = hurst(close, window=300).dropna()
    assert len(values) > 0
    assert values.between(-0.5, 1.5).mean() > 0.9


class TestFeatureMatrix:
    """The assembled feature matrix."""

    def test_index_is_preserved(self, ohlcv):
        features = build_features(ohlcv)
        assert features.index.equals(pd.DatetimeIndex(ohlcv["ts"]))

    def test_close_is_retained_for_downstream_use(self, ohlcv):
        assert "close" in build_features(ohlcv)

    def test_no_infinities_survive(self, ohlcv):
        features = build_features(ohlcv)
        assert not np.isinf(features.select_dtypes("number")).any().any()

    def test_funding_features_appear_when_funding_is_supplied(self, ohlcv):
        from cryptoquant.data.synthetic import make_funding

        funding = make_funding(pd.DatetimeIndex(ohlcv["ts"]), seed=1)
        features = build_features(ohlcv, funding)
        assert "funding" in features
        assert "funding_z_168" in features
