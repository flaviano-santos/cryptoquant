"""End-to-end orchestration: data to features to signal to backtest to validation.

This module is the glue. It reads a :class:`~cryptoquant.config.Config`, loads
whatever the data layer has stored, builds features, applies a strategy, sizes
the resulting signal through the risk layer and evaluates the outcome.

The important convention: :func:`evaluate` always applies volatility targeting
and the execution lag, so two strategies compared through it are compared on
equal terms rather than on how aggressively each happens to be scaled.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data.store import Store
from .research.features import build_features
from .research.validation import (
    probability_of_backtest_overfitting,
    validation_report,
)
from .trading.backtest import BacktestResult, Costs, run_backtest
from .trading.metrics import drawdown_series, monthly_table, rolling_sharpe
from .trading.risk import RiskConfig, apply_turnover_buffer, vol_target_weights
from .trading.strategies import Strategy

log = logging.getLogger(__name__)


def risk_from_config(cfg: Config) -> RiskConfig:
    """Build a :class:`RiskConfig` from the ``risk`` section of a config.

    Args:
        cfg: Loaded configuration.

    Returns:
        A populated risk configuration, using library defaults for any key the
        config file omits.
    """
    r = cfg.get("risk", {}) or {}
    return RiskConfig(
        target_annual_vol=r.get("target_annual_vol", 0.20),
        max_gross_leverage=r.get("max_gross_leverage", 2.0),
        max_position_per_asset=r.get("max_position_per_asset", 0.5),
        vol_lookback=r.get("vol_lookback", 72),
        kelly_fraction=r.get("kelly_fraction", 0.25),
        max_daily_loss=r.get("max_daily_loss", 0.03),
        max_drawdown=r.get("max_drawdown", 0.20),
        bars_per_year=cfg.get("backtest.bars_per_year", 8760),
    )


def costs_from_config(cfg: Config) -> Costs:
    """Build a :class:`Costs` model from the ``costs`` section of a config.

    Args:
        cfg: Loaded configuration.

    Returns:
        The cost model applied by the backtester.
    """
    c = cfg.get("costs", {}) or {}
    return Costs(
        taker_fee_bps=c.get("taker_fee_bps", 4.5),
        maker_fee_bps=c.get("maker_fee_bps", 2.0),
        slippage_bps=c.get("slippage_bps", 2.0),
    )


# ---------------------------------------------------------------------------
def load_feature_data(
    cfg: Config, symbols: Sequence[str] | None = None
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Returns (features per symbol, wide close prices, wide per-bar funding)."""
    store = Store(cfg.data_root)
    symbols = list(symbols or cfg.get("data.symbols"))
    interval = cfg.get("data.interval", "1h")

    feats, closes, fundings = {}, {}, {}
    for sym in symbols:
        k = store.read("klines", sym, interval)
        if k.empty:
            log.warning("no klines for %s - run scripts/01_download_data.py", sym)
            continue
        f = store.read("funding", sym)
        m = store.read("metrics", sym)
        X = build_features(k, f if not f.empty else None, m if not m.empty else None)
        feats[sym] = X
        closes[sym] = X["close"]
        if "funding" in X:
            # funding is charged every 8h; spread it so per-bar accounting works
            bars_per_funding = int(pd.Timedelta("8h") / pd.Timedelta(interval))
            fundings[sym] = X["funding"].fillna(0.0) / max(bars_per_funding, 1)

    if not feats:
        raise RuntimeError("no data found - download first")

    prices = pd.DataFrame(closes).sort_index()
    funding = (
        pd.DataFrame(fundings).reindex(prices.index).fillna(0.0) if fundings else pd.DataFrame()
    )
    # align every symbol onto the common index
    for s in feats:
        feats[s] = feats[s].reindex(prices.index)
    return feats, prices, funding


# ---------------------------------------------------------------------------
def evaluate(
    strategy: Strategy,
    feats: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    funding: pd.DataFrame,
    cfg: Config,
    turnover_buffer: float = 0.1,
    fit: bool = True,
    **signal_kwargs,
) -> tuple[BacktestResult, pd.DataFrame]:
    """Fit (if needed), size, backtest. Returns the result and the final weights."""
    if fit:
        strategy.fit(feats)
    sig = strategy.signal(feats, **signal_kwargs)
    sig = sig.reindex(prices.index).reindex(columns=prices.columns).fillna(0.0)

    rk = risk_from_config(cfg)
    w = vol_target_weights(sig, prices, rk)
    if turnover_buffer:
        w = apply_turnover_buffer(w, turnover_buffer)

    res = run_backtest(
        prices=prices,
        weights=w,
        costs=costs_from_config(cfg),
        funding=funding if not funding.empty else None,
        lag=1,
        bars_per_year=rk.bars_per_year,
        initial_equity=cfg.get("backtest.initial_equity", 10_000),
        max_gross=rk.max_gross_leverage,
    )
    return res, w


def make_live_signal_fn(cfg: Config, strategy: Strategy):
    """Build the callable the live runner uses to turn fresh bars into weights.

    The live path must apply exactly the same feature construction and risk
    sizing as the backtest, otherwise the strategy you validated is not the
    strategy you are running. Sharing this function is what guarantees that.

    Args:
        cfg: Loaded configuration.
        strategy: A fitted (or stateless) strategy.

    Returns:
        A function mapping ``{symbol: ohlcv_frame}`` onto a weight frame.
    """
    risk = risk_from_config(cfg)

    def signal_fn(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        feats = {s: build_features(df) for s, df in bars.items()}
        index = pd.DatetimeIndex(sorted(set().union(*(f.index for f in feats.values()))))
        feats = {s: f.reindex(index) for s, f in feats.items()}
        prices = pd.DataFrame({s: f["close"] for s, f in feats.items()})
        signal = strategy.signal(feats).reindex(prices.index).fillna(0.0)
        return apply_turnover_buffer(vol_target_weights(signal, prices, risk), 0.1)

    return signal_fn


def buy_and_hold(prices: pd.DataFrame, cfg: Config) -> BacktestResult:
    """Equal-weight long-only benchmark. If you cannot beat this, stop."""
    w = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
    return run_backtest(
        prices,
        w,
        costs=costs_from_config(cfg),
        lag=1,
        bars_per_year=cfg.get("backtest.bars_per_year", 8760),
        initial_equity=cfg.get("backtest.initial_equity", 10_000),
    )


# ---------------------------------------------------------------------------
def parameter_sweep(
    strategy_factory, grid: list[dict], feats, prices, funding, cfg, **kw
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every parameter combination and keep *all* the return series.

    You need the full matrix, not just the winner, because probability of
    backtest overfitting and the deflated Sharpe both require knowing how many
    things you tried and how they were distributed. Keeping only the best
    result is how people fool themselves.
    """
    rets, rows = {}, []
    for i, params in enumerate(grid):
        try:
            strat = strategy_factory(**params)
            res, _ = evaluate(strat, feats, prices, funding, cfg, **kw)
        except Exception as exc:
            log.warning("variant %d failed: %s", i, exc)
            continue
        key = f"v{i:03d}"
        rets[key] = res.returns
        rows.append({"variant": key, **params, **res.stats.to_dict()})
    return pd.DataFrame(rets), pd.DataFrame(rows).set_index("variant")


def full_report(
    res: BacktestResult,
    cfg: Config,
    n_trials: int = 1,
    perf_matrix: pd.DataFrame | None = None,
    benchmark: BacktestResult | None = None,
) -> dict:
    """Performance + the validation statistics that decide go / no-go."""
    bpy = cfg.get("backtest.bars_per_year", 8760)
    out = {
        "performance": res.stats,
        "validation": validation_report(res.returns, n_trials=n_trials, bars_per_year=bpy),
        "monthly": monthly_table(res.returns),
    }
    if perf_matrix is not None and perf_matrix.shape[1] > 1:
        out["pbo"] = pd.Series(probability_of_backtest_overfitting(perf_matrix))
    if benchmark is not None:
        out["benchmark"] = benchmark.stats
        excess = res.returns - benchmark.returns.reindex(res.returns.index).fillna(0)
        if excess.std() > 0:
            out["information_ratio"] = float(excess.mean() / excess.std() * np.sqrt(bpy))
    return out


def print_report(report: dict) -> None:
    """Print a report produced by :func:`full_report` to stdout.

    Args:
        report: Mapping of section name to a stats ``Series`` or frame. Missing
            sections are skipped, so partial reports print cleanly.
    """
    for name in ("performance", "benchmark", "validation", "pbo"):
        if name not in report:
            continue
        print(f"\n=== {name.upper()} ===")
        s = report[name]
        for k, v in s.items():
            print(
                f"  {k:<24} {v: >12.4f}"
                if isinstance(v, (int, float, np.floating))
                else f"  {k:<24} {v}"
            )
    if "information_ratio" in report:
        print(f"\n  information_ratio vs benchmark: {report['information_ratio']:.3f}")
    if "monthly" in report and not report["monthly"].empty:
        print("\n=== MONTHLY RETURNS ===")
        print((report["monthly"] * 100).round(1).fillna("").to_string())


def plot_report(
    res: BacktestResult,
    path: str | Path,
    benchmark: BacktestResult | None = None,
    title: str = "strategy",
) -> Path:
    """Render an equity, drawdown and rolling-Sharpe chart to a PNG file.

    The rolling Sharpe panel is the informative one: a strategy whose edge is
    confined to a single stretch of the sample shows up immediately there, and
    nowhere in the headline statistics.

    Args:
        res: Backtest to plot.
        path: Destination file. Parent directories are created as needed.
        benchmark: Optional benchmark to overlay on the equity panel.
        title: Label used in the legend and title.

    Returns:
        The path written to.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        3, 1, figsize=(12, 11), sharex=True, gridspec_kw={"height_ratios": [3, 1.4, 1.4]}
    )
    (res.equity / res.equity.iloc[0]).plot(ax=ax[0], label=title, lw=1.3)
    if benchmark is not None:
        (benchmark.equity / benchmark.equity.iloc[0]).plot(
            ax=ax[0], label="buy & hold", lw=1.0, alpha=0.7
        )
    ax[0].set_yscale("log")
    ax[0].set_ylabel("equity (log)")
    ax[0].legend()
    ax[0].set_title(
        f"{title} | Sharpe {res.stats.get('sharpe', float('nan')):.2f} "
        f"| maxDD {res.stats.get('max_drawdown', float('nan')):.1%}"
    )

    drawdown_series(res.returns).plot(ax=ax[1], color="firebrick", lw=0.9)
    ax[1].set_ylabel("drawdown")

    rolling_sharpe(res.returns, 24 * 90, res.bars_per_year).plot(ax=ax[2], lw=0.9)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_ylabel("rolling 90d Sharpe")

    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
