"""Minimal end-to-end example: synthetic data, a strategy, a validated result.

Runs in about a minute and downloads nothing, so it is a safe first thing to
execute after installing. Swap :func:`~cryptoquant.data.synthetic.make_feature_dataset`
for :func:`cryptoquant.pipeline.load_feature_data` once you have run
``cryptoquant download``.

Usage::

    python examples/01_quickstart.py
"""

from __future__ import annotations

from cryptoquant.config import load_config
from cryptoquant.data.synthetic import make_feature_dataset
from cryptoquant.logging_utils import setup_logging
from cryptoquant.pipeline import buy_and_hold, evaluate, full_report, print_report
from cryptoquant.trading.strategies import MovingAverageCrossover, TrendCarry


def main() -> None:
    """Evaluate two strategies against a passive benchmark and print the report."""
    setup_logging()
    config = load_config()

    features, prices, funding = make_feature_dataset(
        ("A", "B", "C"), n=12_000, seed=42, trend_strength=0.03
    )
    benchmark = buy_and_hold(prices, config)

    for strategy in (MovingAverageCrossover(), TrendCarry()):
        result, weights = evaluate(strategy, features, prices, funding, config)
        print(f"\n{'=' * 60}\n{strategy.name}\n{'=' * 60}")
        print_report(full_report(result, config, n_trials=1, benchmark=benchmark))
        print(f"\n  mean gross exposure: {weights.abs().sum(axis=1).mean():.3f}")

    print(
        "\nNote: a single trial needs no multiple-testing haircut. The moment "
        "you try a second parameterisation, pass the true count as n_trials."
    )


if __name__ == "__main__":
    main()
