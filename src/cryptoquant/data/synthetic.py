"""Synthetic market generator.

Two uses:
  1. Test the pipeline without downloading anything.
  2. Far more important: a **null-model check**. Run your whole research
     process on synthetic data that contains no predictable structure. If it
     still finds a Sharpe of 1.5, your process manufactures edge out of noise
     and every result you have is worthless. Do this before you trust anything.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def make_ohlcv(
    n: int = 20_000,
    seed: int = 0,
    start: str = "2021-01-01",
    freq: str = "1h",
    ann_vol: float = 0.8,
    drift: float = 0.3,
    trend_strength: float = 0.0,
    momentum_halflife: int = 336,
    vol_cluster: float = 0.9,
    s0: float = 30_000.0,
) -> pd.DataFrame:
    """GARCH-like volatility clustering plus an optional slow-moving latent drift.

    trend_strength > 0 injects a persistent expected-return process with the
    given half-life, i.e. a *real* trend-following edge that a correct
    implementation should find. trend_strength = 0 is a pure null: any Sharpe
    your research process reports on it is manufactured.
    """
    rng = np.random.default_rng(seed)
    bars_per_year = {"1h": 8760, "4h": 2190, "1d": 365, "15m": 35040}[freq]
    dt = 1 / bars_per_year
    sigma = ann_vol * np.sqrt(dt)

    eps = rng.standard_normal(n)
    vol = np.empty(n)
    vol[0] = sigma
    for t in range(1, n):
        vol[t] = np.sqrt(
            0.02 * sigma**2
            + vol_cluster * vol[t - 1] ** 2
            + (0.98 - vol_cluster) * (vol[t - 1] * eps[t - 1]) ** 2
        )

    ret = drift * dt + vol * eps
    if trend_strength:
        phi = float(np.exp(-np.log(2) / max(momentum_halflife, 1)))
        shock = rng.standard_normal(n) * sigma * trend_strength * np.sqrt(1 - phi**2)
        mu = np.empty(n)
        mu[0] = shock[0]
        for t in range(1, n):
            mu[t] = phi * mu[t - 1] + shock[t]
        ret = ret + mu

    close = s0 * np.exp(np.cumsum(ret))
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")

    noise = np.abs(rng.standard_normal(n)) * vol * close
    open_ = np.concatenate([[s0], close[:-1]])
    high = np.maximum(open_, close) + noise * 0.5
    low = np.minimum(open_, close) - noise * 0.5
    volume = np.exp(rng.standard_normal(n) * 0.5 + 6) * (1 + 10 * np.abs(ret))

    return pd.DataFrame(
        {
            "symbol": "SYNTH",
            "ts": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": (volume * 3).astype(int),
            "taker_buy_base": volume * (0.5 + 0.1 * rng.standard_normal(n)).clip(0.05, 0.95),
        }
    )


def make_funding(index: pd.DatetimeIndex, seed: int = 0, mean_bps_8h: float = 1.0) -> pd.DataFrame:
    """Generate a plausible perpetual funding-rate series.

    The rate follows a mean-reverting AR(1) process around a small positive
    average, which is what real perpetual funding looks like in practice: longs
    pay shorts most of the time, with occasional sharp excursions.

    Args:
        index: Bar timestamps the funding series should span.
        seed: Random seed.
        mean_bps_8h: Average funding rate per 8-hour period, in basis points.

    Returns:
        A frame with ``symbol``, ``ts`` and ``funding_rate`` columns, sampled
        every eighth bar.
    """
    rng = np.random.default_rng(seed)
    eight_h = index[::8] if len(index) > 8 else index
    x = np.zeros(len(eight_h))
    for t in range(1, len(x)):
        x[t] = 0.95 * x[t - 1] + rng.standard_normal() * 0.5
    rate = (mean_bps_8h + x * 2) / 1e4
    return pd.DataFrame({"symbol": "SYNTH", "ts": eight_h, "funding_rate": rate})


def make_universe(
    symbols=("A", "B", "C", "D"), n: int = 20_000, seed: int = 0, beta: float = 0.7, **kw
) -> dict[str, pd.DataFrame]:
    """Correlated universe with a shared market factor - crypto is roughly a one-factor market, and a backtest on independent assets will flatter any diversification you claim."""
    market = make_ohlcv(n=n, seed=seed, **kw)
    mret = np.log(market["close"]).diff().fillna(0).to_numpy()
    out = {}
    for i, s in enumerate(symbols):
        d = make_ohlcv(n=n, seed=seed + 100 + i, **kw)
        idio = np.log(d["close"]).diff().fillna(0).to_numpy()
        blended = beta * mret + np.sqrt(max(1 - beta**2, 0)) * idio
        px = float(d["close"].iloc[0]) * np.exp(np.cumsum(blended))
        scale = px / d["close"].to_numpy()
        for c in ("open", "high", "low", "close"):
            d[c] = d[c].to_numpy() * scale
        d["symbol"] = s
        out[s] = d
    return out


def make_feature_dataset(
    symbols: Sequence[str] = ("A", "B", "C", "D"),
    n: int = 17_000,
    seed: int = 7,
    trend_strength: float = 0.0,
    bars_per_funding: int = 8,
    **kwargs: float,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Build a ready-to-evaluate synthetic dataset.

    Produces exactly the triple that :func:`cryptoquant.pipeline.evaluate`
    expects, so a strategy can be tested against a known ground truth before it
    ever sees real data.

    Args:
        symbols: Names of the synthetic instruments.
        n: Number of hourly bars per instrument.
        seed: Random seed.
        trend_strength: Strength of the injected latent trend. ``0.0`` produces
            data with no predictable structure whatsoever, which is the setting
            to use when testing whether your research process manufactures edge.
        bars_per_funding: Bars per funding interval, used to spread the 8-hourly
            funding rate across bars for per-bar accounting.
        **kwargs: Forwarded to :func:`make_ohlcv`.

    Returns:
        A ``(features, prices, funding)`` triple.
    """
    from ..research.features import build_features

    universe = make_universe(
        tuple(symbols), n=n, seed=seed, trend_strength=trend_strength, **kwargs
    )

    features, closes, fundings = {}, {}, {}
    for symbol, klines in universe.items():
        funding = make_funding(pd.DatetimeIndex(klines["ts"]), seed=abs(hash(symbol)) % 1000)
        frame = build_features(klines, funding)
        features[symbol] = frame
        closes[symbol] = frame["close"]
        fundings[symbol] = frame["funding"].fillna(0.0) / max(bars_per_funding, 1)

    prices = pd.DataFrame(closes).sort_index()
    funding_panel = pd.DataFrame(fundings).reindex(prices.index).fillna(0.0)
    return features, prices, funding_panel
