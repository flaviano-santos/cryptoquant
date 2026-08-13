# cryptoquant

[![CI](https://github.com/flavianosantos/cryptoquant/actions/workflows/ci.yml/badge.svg)](https://github.com/flavianosantos/cryptoquant/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A local, zero-cost research and trading stack for cryptocurrency markets: bulk
historical data, feature engineering, path-aware labeling, a cost-realistic
backtester, the statistics that decide whether a result is real, and a live
execution layer that defaults to doing nothing.

No paid APIs. No subscriptions. No cloud. It runs on a laptop.

> **This is research software, not financial advice.** No backtest result
> implies a live result. Cryptocurrency trading carries substantial risk of
> total loss, and leveraged perpetual futures can lose more than the initial
> margin.

---

## Why another backtester

Most retail trading code fails the same test: give it a signal derived from
information it should not have, and it happily reports a Sharpe of 100. This
project inverts the priority. The backtester is ordinary; the **validation
layer is the product**.

Run this before anything else:

```bash
cryptoquant selftest --trend 0.0 --sweep
```

That generates synthetic markets containing **no predictable structure
whatsoever**, then runs the full research process over them — features, labels,
a 36-variant parameter sweep, the works. A trustworthy process finds nothing.
Ours reports a deflated Sharpe of 0.03 and a 79% probability of out-of-sample
loss on the sweep winner, which is the correct answer. If your own tooling finds
an edge in that data, it will find edges everywhere, and you will fund one.

---

## Installation

```bash
git clone https://github.com/flavianosantos/cryptoquant
cd cryptoquant
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev,live]"
```

Requires Python 3.10 or newer. The `live` extra pulls in CCXT and is only
needed for trading; research works without it.

## Quick start

```bash
# 1. Prove the machinery is honest (no download required)
pytest
cryptoquant selftest --trend 0.0        # must find nothing
cryptoquant selftest --trend 0.03       # a real edge is injected; should find it

# 2. Download free history (Binance Vision; no API key)
cryptoquant download

# 3. Research
cryptoquant research --ml --sweep

# 4. Paper trade (testnet + dry-run by default: logs orders, sends nothing)
cryptoquant paper --once
```

---

## Free data sources

| Source | Content | Notes |
|---|---|---|
| **[data.binance.vision](https://data.binance.vision)** | Klines, agg trades, funding rates, open interest — all symbols, from 2017 | The most valuable free resource in crypto. Vendors resell this. No key, no rate limit. |
| **[CCXT](https://github.com/ccxt/ccxt)** | ~100 exchanges, unified API | Public market data needs no credentials. |
| **[DefiLlama](https://defillama.com/docs/api)** | TVL, DEX volume, stablecoin supply | Unauthenticated. Slow macro context. |
| **alternative.me** | Fear & Greed index, daily from 2018 | Weak alone; useful as a regime conditioner. |
| **Binance testnet** | Full API with fake money | Free paper trading against a real matching engine. |

Storage is a Parquet lake queried through **DuckDB** — SQL over tens of
gigabytes of bars with no server and no ingest step.

---

## Architecture

```
src/cryptoquant/
├── config.py             configuration loading (never secrets)
├── cli.py                the `cryptoquant` command
├── pipeline.py           end-to-end orchestration
├── data/
│   ├── binance_vision.py bulk historical archives, cached
│   ├── feeds.py          CCXT live and recent data
│   ├── alt_data.py       free sentiment and on-chain series
│   ├── store.py          DuckDB + Parquet warehouse
│   └── synthetic.py      null-model market generator
├── research/
│   ├── features.py       technical, microstructure, fractional differentiation
│   ├── labeling.py       triple-barrier, CUSUM, sample uniqueness
│   └── validation.py     purged CV, deflated Sharpe, PBO, block bootstrap
└── trading/
    ├── backtest.py       vectorised engine, enforced execution lag
    ├── metrics.py        performance statistics
    ├── risk.py           vol targeting, no-trade bands, kill switch
    ├── live.py           CCXT broker and runner
    └── strategies/       MA, trend + carry, adaptive multi-factor, meta-ML
```

## Adaptive multi-factor research candidate

`AdaptiveMultiFactor` is a fixed-default, causal candidate combining
multi-horizon trend, breakout, range-regime mean reversion, cross-sectional
relative strength and perpetual-funding carry. Volatility shocks and poor
liquidity reduce exposure; 12-bar signal persistence controls turnover. It is
included in `cryptoquant research`, but deliberately not enabled for paper or
live trading until it passes a longer forward test.

The first untouched evaluation used hourly Binance USD-M data for BTC, ETH,
SOL, BNB and XRP from 2020 through 2026-08-12, with default fees, slippage,
realized funding, execution lag and risk limits:

| Test | Net Sharpe | CAGR | Max drawdown |
|---|---:|---:|---:|
| Full sample | 1.61 | 36.8% | -19.0% |
| 2020-2021 | 2.49 | 65.3% | -14.5% |
| 2022-2023 | 1.78 | 42.4% | -11.6% |
| 2024-2025 | 1.07 | 21.8% | -16.5% |
| 2026 YTD | **-0.19** | **-5.6%** | **-19.0%** |
| Double costs + extra execution bar | 1.32 | 28.7% | -20.4% |

These are research results, not a forecast. The weakening by period and the
negative 2026 result are material. Defaults were frozen before the real-data
run; tuning them against this sample would destroy its out-of-sample value.
The next valid evidence is forward paper trading.

### Frozen forward evidence

The current forward test was frozen at `2026-08-12 23:00 UTC`. Its manifest
stores SHA-256 hashes of `AdaptiveMultiFactor` and `config.yaml`; subsequent
snapshots form an append-only hash chain. Verification fails if the frozen
strategy, configuration, manifest or any earlier ledger entry changes.

```bash
python -m cryptoquant.forward verify --config config.yaml --directory forward_evidence
python -m cryptoquant.forward update --config config.yaml --directory forward_evidence
```

`update` downloads newly completed public Binance archives and evaluates only
bars later than the frozen cutoff. It never places an order. Thirty calendar
days is the minimum useful checkpoint; 90 days spanning more than one regime is
materially better.

The repository includes `.github/workflows/forward-evidence.yml`, which runs
this collection daily on a GitHub-hosted runner and commits only
`forward_evidence/ledger.jsonl`. It requires no exchange credentials and cannot
place trades. Repository Actions must have read/write workflow permission; a
protected default branch may instead require a pull-request workflow.

### The five ideas that carry the weight

**1. Fractional differentiation.** Raw prices are non-stationary; a model will
memorise 2021 price levels. Returns are stationary but discard all memory.
Fractional differencing applies the *minimum* differencing that achieves
stationarity, keeping the rest.

**2. Triple-barrier labeling.** "Predict the next bar's sign" ignores volatility
and ignores path — a trade that hits your stop before your target is a loss
regardless of where price ends up. Triple-barrier labels ask which barrier came
first.

**3. Sample uniqueness.** Overlapping labels are near-duplicates. In our test
fixture, 294 labelled rows carry only **128 effective independent
observations**. Passing uniqueness as `sample_weight` removes a large amount of
illusory model skill, which is the point.

**4. Purged, embargoed cross-validation.** Standard k-fold leaks: a training
label spanning bars 100–124 shares nearly all its information with a test label
spanning 101–125. Purging drops the overlaps; the embargo drops what follows,
because serial correlation leaks backwards too.

**5. Deflated Sharpe and PBO.** Try 500 variants and keep the best, and the best
Sharpe is mostly luck. The deflated Sharpe computes how high a Sharpe pure luck
would produce given your trial count. PBO runs your *selection procedure* over
combinatorial splits and measures how often the in-sample winner lands in the
bottom half out-of-sample. On pure noise it correctly reports 0.58.

References: López de Prado, *Advances in Financial Machine Learning*, ch. 3–8;
Bailey & López de Prado (2014); Bailey et al. (2017).

---

## The validation gates

Run in order. Each kills ideas cheaply.

| Gate | Test | Pass condition |
|---|---|---|
| **0** | Null test on synthetic data | Process finds no edge |
| **1** | Beat both benchmarks | Better than buy-and-hold *and* MA crossover, net of costs |
| **2** | Survive costs | `cost_drag_sharpe` below half of gross Sharpe |
| **3** | Survive deflation | DSR > 0.90 **and** PBO < 0.5, with an honest trial count |
| **4** | Survive walk-forward | Anchored, always training on the past |
| **5** | Survive perturbation | Extra bar of lag, doubled fees, best month dropped, signal shuffled |
| **6** | Paper trade ≥ 1 month | On testnet, with the real runner |
| **7** | Live, small | Money you can lose entirely |

On gate 3, be honest about the trial count: it includes every parameter you
nudged by hand, not just the ones you wrote down.

### A worked example

Applying these gates to a plain moving-average crossover on daily BTC
(2018–2025, 2 bps slippage on top of 4 bps fees):

| | Sharpe | Max DD | CAGR |
|---|---|---|---|
| Buy and hold | 0.68 | −81.5% | 25.9% |
| MA(20,100), never tuned | **0.92** | −59.0% | 36.9% |
| Best of 24 swept variants | 1.20 | — | — |
| **Honest walk-forward selection** | **0.65** | −39.8% | 19.6% |

Read the last two rows together. **Optimising the parameters made the strategy
worse** — 0.92 untuned versus 0.65 after honest walk-forward selection. The 1.20
never existed; it is visible only with hindsight. This is why
`MovingAverageCrossover` ships with fixed defaults and a docstring warning
against tuning them.

---

## Realistic expectations

- **Where a careful individual can still compete:** medium-frequency (hours to
  days), cross-sectional signals across many altcoins, funding and basis carry,
  small-cap pairs beneath institutional size thresholds, regime-aware sizing.
  Not latency, not market making.
- **What a good retail result looks like:** net Sharpe 0.8–1.5, 20–40% max
  drawdown, on capital small enough that market impact is negligible. Anything
  above 2.5 in a backtest is usually a bug, a fee assumption, or overfitting —
  check in that order.
- **Costs dominate.** At ~370× annual turnover, fees and slippage cost roughly
  **1.0–1.3 of Sharpe**. Turnover control is frequently the whole difference
  between profitable and not.
- **Crypto is roughly a one-factor market.** Most coins are ~0.8 correlated to
  BTC. A "diversified" long book of ten alts is one bet.
- **The base rate is bad.** Most strategies that look good in backtest do not
  survive live. That is the reason this package exists.

---

## Live trading

Three design commitments in `trading/live.py`:

1. **State comes from the exchange, never from memory.** Every cycle re-reads
   actual positions. Where local and remote disagree, remote wins.
2. **Everything is off by default.** `testnet: true` and `dry_run: true`.
3. **The kill switch is checked before every order** — daily loss and max
   drawdown — and once tripped it flattens and halts until reset by hand.

Credentials come from the environment only:

```bash
export CQ_API_KEY=...
export CQ_API_SECRET=...
```

Create keys with **trading enabled, withdrawals disabled, and an IP allowlist**.
See [SECURITY.md](SECURITY.md).

---

## Development

```bash
pre-commit install              # ruff, mypy and a private-key detector on every commit
pytest                          # full suite (~40 s)
pytest -m "not slow"            # skip the model-fitting tests
pytest --cov                    # with coverage
ruff check . && ruff format --check . && mypy
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests adding strategies are
judged on evidence, not returns.

## Further reading

- López de Prado, *Advances in Financial Machine Learning* — the source of most
  of the validation machinery here. If you read one book, this is it.
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*.
- Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest
  Overfitting*.
- Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum* — why `TrendCarry`
  is built the way it is.
- Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns* — on
  why a t-statistic of 2 is not enough.

## Related projects

- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) — the
  natural next step for execution realism once a strategy clears gate 5.
- [skfolio](https://github.com/skfolio/skfolio) — scikit-learn-compatible
  portfolio optimisation with combinatorial purged CV.
- [Freqtrade](https://github.com/freqtrade/freqtrade) — stronger operational
  plumbing for deployment.

## License

MIT — see [LICENSE](LICENSE).
