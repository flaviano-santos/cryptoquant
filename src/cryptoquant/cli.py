"""Command-line interface.

Installs as the ``cryptoquant`` command and exposes the three stages of the
workflow as subcommands::

    cryptoquant download                 # fetch free historical data
    cryptoquant research --sweep --ml    # backtest and validate
    cryptoquant selftest                 # null test: prove the process is honest
    cryptoquant paper --once             # dry-run the live loop

Argparse is used rather than a third-party CLI framework to keep the dependency
footprint small; this package is meant to be auditable end to end.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .config import load_config
from .data.synthetic import make_feature_dataset
from .logging_utils import setup_logging

logger = logging.getLogger(__name__)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser.

    Returns:
        A parser with one subparser per workflow stage.
    """
    parser = argparse.ArgumentParser(
        prog="cryptoquant",
        description="Local, zero-cost crypto trading research stack.",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="fetch free historical data")
    download.add_argument("--symbols", nargs="*", default=None)
    download.add_argument(
        "--metrics", action="store_true", help="also fetch open interest and positioning (large)"
    )
    download.add_argument("--no-funding", action="store_true")

    research = subparsers.add_parser("research", help="backtest and validate")
    research.add_argument("--ml", action="store_true", help="also run the meta-labeled model")
    research.add_argument("--sweep", action="store_true", help="parameter sweep with PBO")
    research.add_argument("--outdir", default="reports")

    selftest = subparsers.add_parser(
        "selftest", help="run the pipeline on synthetic data with no edge in it"
    )
    selftest.add_argument(
        "--trend",
        type=float,
        default=0.0,
        help="0.0 = pure null; ~0.03 injects a real, modest edge",
    )
    selftest.add_argument("--sweep", action="store_true")
    selftest.add_argument("--outdir", default="reports")

    paper = subparsers.add_parser("paper", help="run the live loop (testnet, dry-run)")
    paper.add_argument("--once", action="store_true", help="single cycle then exit")

    return parser


def _cmd_download(args: argparse.Namespace) -> int:
    """Fetch historical data into the local store.

    Args:
        args: Parsed arguments carrying ``config``, ``symbols`` and flags.

    Returns:
        A process exit code.
    """
    from .data.store import ingest

    config = load_config(args.config)
    store = ingest(
        config,
        symbols=args.symbols,
        with_funding=not args.no_funding,
        with_metrics=args.metrics,
    )
    interval = config.get("data.interval")
    print(f"\ndata written to {store.root}")
    for symbol in args.symbols or config.get("data.symbols"):
        frame = store.read("klines", symbol, interval)
        if frame.empty:
            print(f"  {symbol:<10} no data")
        else:
            print(
                f"  {symbol:<10} {len(frame):>8,} bars  "
                f"{frame['ts'].iloc[0]:%Y-%m-%d} -> {frame['ts'].iloc[-1]:%Y-%m-%d}"
            )
    return 0


def _run_research(args: argparse.Namespace, synthetic_trend: float | None) -> int:
    """Shared implementation of the ``research`` and ``selftest`` commands.

    Args:
        args: Parsed arguments.
        synthetic_trend: ``None`` to use real stored data, otherwise the trend
            strength of a synthetic universe.

    Returns:
        A process exit code.
    """
    from .pipeline import (
        buy_and_hold,
        evaluate,
        full_report,
        load_feature_data,
        parameter_sweep,
        plot_report,
        print_report,
    )
    from .trading.strategies import (
        AdaptiveMultiFactor,
        MetaLabelML,
        MovingAverageCrossover,
        TrendCarry,
    )

    config = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if synthetic_trend is None:
        features, prices, funding = load_feature_data(config)
    else:
        features, prices, funding = make_feature_dataset(trend_strength=synthetic_trend)
        print(
            f"\nSYNTHETIC DATA (trend_strength={synthetic_trend}). "
            f"{'There is no edge to find - any Sharpe below is your process fooling itself.' if synthetic_trend == 0 else 'A real edge is present; the pipeline should find it.'}\n"
        )

    benchmark = buy_and_hold(prices, config)

    print("\n" + "=" * 60)
    print("BENCHMARK 1: equal-weight buy and hold")
    print("=" * 60)
    print_report({"performance": benchmark.stats})

    print("\n" + "=" * 60)
    print("BENCHMARK 2: moving-average crossover (20/100)")
    print("=" * 60)
    ma_result, _ = evaluate(MovingAverageCrossover(), features, prices, funding, config)
    print_report(full_report(ma_result, config, n_trials=1, benchmark=benchmark))

    print("\n" + "=" * 60)
    print("STRATEGY: trend + carry")
    print("=" * 60)
    result, _ = evaluate(TrendCarry(), features, prices, funding, config)
    print_report(full_report(result, config, n_trials=1, benchmark=benchmark))
    plot_report(result, outdir / "trend_carry.png", benchmark, "trend + carry")

    print("\n" + "=" * 60)
    print("RESEARCH CANDIDATE: adaptive multi-factor (fixed defaults)")
    print("=" * 60)
    adaptive_result, _ = evaluate(AdaptiveMultiFactor(), features, prices, funding, config)
    print_report(full_report(adaptive_result, config, n_trials=1, benchmark=benchmark))
    plot_report(
        adaptive_result,
        outdir / "adaptive_multifactor.png",
        benchmark,
        "adaptive multi-factor",
    )

    n_trials = 1
    if args.sweep:
        grid = [
            {"w_ts": a, "w_cs": b, "w_carry": c, "squash": s}
            for a, b, c in itertools.product((0.3, 0.5, 0.7), (0.0, 0.3), (0.0, 0.2, 0.4))
            for s in (0.5, 1.0)
        ]
        perf_matrix, table = parameter_sweep(TrendCarry, grid, features, prices, funding, config)
        n_trials = perf_matrix.shape[1]
        table.to_csv(outdir / "sweep.csv")

        print("\n" + "=" * 60)
        print(f"SWEEP: {n_trials} variants")
        print("=" * 60)
        print(
            table.sort_values("sharpe", ascending=False)
            .loc[:, ["w_ts", "w_cs", "w_carry", "squash", "sharpe", "max_drawdown", "ann_turnover"]]
            .head(10)
            .round(3)
            .to_string()
        )

        best = table["sharpe"].idxmax()
        params = {k: table.loc[best, k] for k in ("w_ts", "w_cs", "w_carry", "squash")}
        best_result, _ = evaluate(TrendCarry(**params), features, prices, funding, config)
        print(f"\nBEST VARIANT {best}: {params}")
        print_report(
            full_report(
                best_result, config, n_trials=n_trials, perf_matrix=perf_matrix, benchmark=benchmark
            )
        )
        print("\n  Read the deflated Sharpe and PBO, not the raw Sharpe.")
        print("  DSR < 0.90 or PBO > 0.5 means you have found noise.")

    if args.ml:
        print("\n" + "=" * 60)
        print("STRATEGY: meta-labeled ML (purged out-of-sample)")
        print("=" * 60)
        ml_result, _ = evaluate(
            MetaLabelML(primary=TrendCarry()), features, prices, funding, config, use_oos=True
        )
        print_report(
            full_report(ml_result, config, n_trials=max(n_trials, 20), benchmark=benchmark)
        )
        plot_report(ml_result, outdir / "meta_ml.png", benchmark, "meta-labeled ML")

    print(f"\nreports written to {outdir.resolve()}")
    return 0


def _cmd_paper(args: argparse.Namespace) -> int:
    """Run the live trading loop in its safe default configuration.

    Args:
        args: Parsed arguments carrying ``config`` and ``once``.

    Returns:
        A process exit code.
    """
    from .pipeline import make_live_signal_fn, risk_from_config
    from .trading.live import Broker, LiveRunner
    from .trading.strategies import TrendCarry

    config = load_config(args.config)
    symbols = [s.replace("USDT", "/USDT:USDT") for s in config.get("data.symbols")]

    broker = Broker(
        exchange_id=config.get("live.exchange", "binanceusdm"),
        testnet=config.get("live.testnet", True),
        dry_run=config.get("live.dry_run", True),
    )
    runner = LiveRunner(
        broker=broker,
        symbols=symbols,
        signal_fn=make_live_signal_fn(config, TrendCarry()),
        risk=risk_from_config(config),
        timeframe=config.get("data.interval", "1h"),
        poll_seconds=config.get("live.poll_seconds", 60),
        state_path=Path(config.data_root) / "live_state.json",
    )

    if args.once:
        state = runner.step()
        print(pd.Series(state["weights"]).round(4).to_string())
    else:
        runner.run()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``cryptoquant`` console script.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code, ``0`` on success.
    """
    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    handlers = {
        "download": _cmd_download,
        "research": lambda a: _run_research(a, synthetic_trend=None),
        "selftest": lambda a: _run_research(a, synthetic_trend=a.trend),
        "paper": _cmd_paper,
    }
    if args.command == "selftest" and not hasattr(args, "ml"):
        args.ml = False
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
