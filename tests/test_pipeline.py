"""Tests for the end-to-end orchestration layer."""

from __future__ import annotations

import numpy as np
import pytest

from cryptoquant.config import Config, load_config
from cryptoquant.data.synthetic import make_feature_dataset
from cryptoquant.pipeline import (
    buy_and_hold,
    costs_from_config,
    evaluate,
    full_report,
    make_live_signal_fn,
    parameter_sweep,
    plot_report,
    risk_from_config,
)
from cryptoquant.trading.strategies import MovingAverageCrossover, TrendCarry


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


@pytest.fixture(scope="module")
def dataset():
    """A small synthetic dataset with a genuine edge in it."""
    return make_feature_dataset(("A", "B"), n=3_000, seed=11, trend_strength=0.05)


class TestConfigMapping:
    def test_risk_config_reads_the_file(self, config):
        risk = risk_from_config(config)
        assert risk.target_annual_vol == config.get("risk.target_annual_vol")
        assert risk.max_gross_leverage == config.get("risk.max_gross_leverage")

    def test_costs_read_the_file(self, config):
        costs = costs_from_config(config)
        assert costs.taker_fee_bps == config.get("costs.taker_fee_bps")
        assert costs.per_turnover > 0


class TestEvaluate:
    def test_returns_a_result_and_weights(self, dataset, config):
        features, prices, funding = dataset
        result, weights = evaluate(MovingAverageCrossover(), features, prices, funding, config)
        assert len(result.returns) == len(prices)
        assert weights.shape == prices.shape

    def test_respects_the_gross_leverage_cap(self, dataset, config):
        features, prices, funding = dataset
        result, _ = evaluate(TrendCarry(), features, prices, funding, config)
        cap = config.get("risk.max_gross_leverage")
        assert result.positions.abs().sum(axis=1).max() <= cap + 1e-9

    def test_costs_are_charged(self, dataset, config):
        features, prices, funding = dataset
        result, _ = evaluate(TrendCarry(), features, prices, funding, config)
        assert result.costs.sum() > 0
        # Net must be worse than gross in aggregate. Bar by bar it need not be,
        # because a short position *receives* positive funding.
        assert result.returns.sum() < result.gross_returns.sum()

    def test_turnover_buffer_reduces_trading(self, dataset, config):
        features, prices, funding = dataset
        tight, _ = evaluate(TrendCarry(), features, prices, funding, config, turnover_buffer=0.0)
        loose, _ = evaluate(TrendCarry(), features, prices, funding, config, turnover_buffer=0.5)
        assert loose.turnover.sum() < tight.turnover.sum()


def test_buy_and_hold_is_fully_invested(dataset, config):
    _, prices, _ = dataset
    result = buy_and_hold(prices, config)
    assert result.positions.sum(axis=1).iloc[-1] == pytest.approx(1.0)


class TestParameterSweep:
    def test_keeps_every_variant(self, dataset, config):
        """The whole matrix is required to compute PBO and the deflated Sharpe."""
        features, prices, funding = dataset
        grid = [{"w_ts": w} for w in (0.3, 0.5, 0.7)]
        matrix, table = parameter_sweep(TrendCarry, grid, features, prices, funding, config)
        assert matrix.shape[1] == len(grid)
        assert len(table) == len(grid)
        assert "sharpe" in table

    def test_a_failing_variant_is_skipped_not_fatal(self, dataset, config):
        features, prices, funding = dataset
        grid = [{"w_ts": 0.5}, {"nonexistent_parameter": 1.0}]
        matrix, _ = parameter_sweep(TrendCarry, grid, features, prices, funding, config)
        assert matrix.shape[1] == 1


class TestReporting:
    def test_full_report_sections(self, dataset, config):
        features, prices, funding = dataset
        result, _ = evaluate(MovingAverageCrossover(), features, prices, funding, config)
        report = full_report(result, config, n_trials=1, benchmark=buy_and_hold(prices, config))
        assert {"performance", "validation", "monthly", "benchmark"} <= set(report)
        assert "information_ratio" in report

    def test_plot_writes_a_file(self, dataset, config, tmp_path):
        features, prices, funding = dataset
        result, _ = evaluate(MovingAverageCrossover(), features, prices, funding, config)
        path = plot_report(result, tmp_path / "out" / "equity.png", title="test")
        assert path.is_file() and path.stat().st_size > 0


def test_live_signal_function_matches_the_backtest_path(dataset, config):
    """The live path must reuse the backtest's feature and sizing code."""
    features, prices, _ = dataset
    bars = {
        symbol: frame.reset_index()
        .rename(columns={"index": "ts"})
        .assign(
            open=frame["close"], high=frame["close"] * 1.001, low=frame["close"] * 0.999, volume=1.0
        )
        for symbol, frame in features.items()
    }
    weights = make_live_signal_fn(config, MovingAverageCrossover())(bars)
    assert set(weights.columns) == set(prices.columns)
    assert np.isfinite(weights.to_numpy()).all()
