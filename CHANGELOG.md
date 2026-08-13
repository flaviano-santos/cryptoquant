# Changelog

## Unreleased

- Add a causal, regime-adaptive multi-factor research strategy combining
  trend, breakout, range mean reversion, cross-sectional momentum and funding
  carry, with volatility/liquidity gates and signal persistence.
- Include the fixed-default candidate in research reports while keeping it out
  of the live path pending forward paper validation.
- Add future-mutation, boundedness, configuration and turnover tests.
- Accelerate fractional differentiation with FFT convolution while preserving
  missing-window semantics.
- Add a frozen, hash-chained forward-evidence ledger and daily public-data
  update command; it is completely separate from order execution.

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `LiveRunner` now sets `reduceOnly` when closing a position to flat. The
  previous sign comparison treated a full close as an opening trade, because
  `np.sign(0)` is `0`.
- `parameter_sweep` no longer aborts the whole sweep when one grid entry fails
  to construct.

### Added

- A fake-exchange test double, so the live loop is tested without network
  access. Test coverage is now 67%.

## [0.1.0] - 2026-08-01

Initial release.

### Added

- **Data layer.** Binance Vision bulk historical loader (klines, funding rates,
  open-interest metrics) with on-disk caching and automatic epoch-unit
  detection; CCXT feeds for recent and live data; free alternative data
  (Fear & Greed, DefiLlama TVL, stablecoin supply); a DuckDB/Parquet store.
- **Research layer.** Feature engineering including fractional differentiation
  and range-based volatility estimators; triple-barrier labeling with CUSUM
  event sampling and sample-uniqueness weighting; a validation module providing
  purged k-fold with embargo, walk-forward splits, the deflated Sharpe ratio,
  probability of backtest overfitting via CSCV, and the stationary bootstrap.
- **Trading layer.** A vectorised backtester with enforced execution lag,
  transaction costs, slippage and perpetual funding; volatility targeting,
  fractional Kelly, proportional no-trade bands and a kill switch; strategies
  `MovingAverageCrossover`, `TrendCarry`, `MetaLabelML` and `Ensemble`; a live
  runner defaulting to testnet and dry-run.
- **Synthetic market generator** for null testing the research process.
- `cryptoquant` command-line interface with `download`, `research`, `selftest`
  and `paper` subcommands.
- Test suite covering lookahead, leakage, cost accounting and the
  multiple-testing statistics.
- CI running lint, type-check, a test matrix across three Python versions and
  two operating systems, plus the null test.

### Notes

- `MovingAverageCrossover` is adapted from the author's earlier
  `bitcoin-quant-lab` project. Its Sharpe calculation had a bug that inflated
  results by roughly 2x; the corrected formula lives in
  `cryptoquant.trading.metrics`.
