"""Feature engineering.

Design rules that matter more than the specific features:

1. Every feature must be computable using only information available at the
   close of the bar it is stamped on. Any rolling window is backward-looking.
   One accidental `.shift(-1)` invalidates an entire research programme.

2. Prefer stationary transforms. Raw price is non-stationary and a model will
   happily memorise 2021 price levels. Returns are stationary but throw away
   all memory. Fractional differentiation sits between the two: the minimum
   differencing that passes a stationarity test while keeping as much memory
   as possible (Lopez de Prado, Advances in Financial Machine Learning, ch.5).

3. Scale features by volatility, not by absolute size. A 2% move means
   different things in different regimes.

4. Crypto-specific features carry real information that equity factors do not:
   perpetual funding rate (the cost of crowded leverage), open-interest
   changes, taker buy/sell imbalance, and the basis between perp and spot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# fractional differentiation
# ---------------------------------------------------------------------------
def frac_diff_weights(d: float, threshold: float = 1e-5, max_len: int = 10_000) -> np.ndarray:
    """Binomial weights for the fixed-width fractional differencing window."""
    w = [1.0]
    for k in range(1, max_len):
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < threshold:
            break
        w.append(w_)
    return np.array(w[::-1])


def frac_diff(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """Fixed-width-window fractional differentiation. Preserves memory."""
    w = frac_diff_weights(d, threshold)
    width = len(w)
    vals = series.to_numpy(dtype="float64")
    out = np.full(len(vals), np.nan)
    if len(vals) < width:
        return pd.Series(out, index=series.index, name=f"{series.name}_fd{d}")

    # FFT convolution performs the same trailing dot products in O(n log n).
    # The prior Python loop made a five-asset hourly research run spend several
    # minutes on this feature alone. A cumulative finite-value count preserves
    # the old rule that a window containing a missing observation is missing.
    clean = np.nan_to_num(vals, nan=0.0)
    computed = fftconvolve(clean, w[::-1], mode="valid")
    finite = np.isfinite(vals).astype(int)
    cumulative = np.concatenate(([0], np.cumsum(finite)))
    valid = cumulative[width:] - cumulative[:-width]
    out[width - 1 :] = np.where(valid == width, computed, np.nan)
    return pd.Series(out, index=series.index, name=f"{series.name}_fd{d}")


def min_ffd(series: pd.Series, ds: np.ndarray | None = None, pvalue: float = 0.05) -> float:
    """Smallest d for which the fractionally differenced series is stationary (ADF test). Requires statsmodels; falls back to 0.4, an empirically decent default for crypto log-prices."""
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return 0.4
    ds = np.linspace(0, 1, 11) if ds is None else ds
    for d in ds:
        x = frac_diff(series, float(d)).dropna()
        if len(x) < 100:
            continue
        if adfuller(x, maxlag=1, regression="c", autolag=None)[1] < pvalue:
            return float(d)
    return 1.0


# ---------------------------------------------------------------------------
# volatility estimators
# ---------------------------------------------------------------------------
def realised_vol(returns: pd.Series, window: int = 72) -> pd.Series:
    """Rolling standard deviation of returns.

    Args:
        returns: Per-bar returns.
        window: Window length in bars.

    Returns:
        Rolling volatility, ``NaN`` until the window is half full.
    """
    return returns.rolling(window, min_periods=window // 2).std()


def ewma_vol(returns: pd.Series, halflife: int = 36) -> pd.Series:
    """Exponentially weighted volatility.

    Reacts faster to regime changes than an equally weighted window, which
    matters because crypto volatility clusters strongly.

    Args:
        returns: Per-bar returns.
        halflife: Half-life of the weighting, in bars.

    Returns:
        EWMA volatility.
    """
    return returns.ewm(halflife=halflife, min_periods=halflife).std()


def garman_klass(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """Range-based volatility. Uses the whole bar (O,H,L,C) instead of just the close, so it is roughly 5-7x more efficient than close-to-close for the same sample size. Free accuracy."""
    hl = np.log(df["high"] / df["low"]) ** 2
    co = np.log(df["close"] / df["open"]) ** 2
    gk = 0.5 * hl - (2 * np.log(2) - 1) * co
    return np.sqrt(gk.rolling(window, min_periods=window // 2).mean().clip(lower=0))


def yang_zhang(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """Drift- and gap-robust volatility estimator."""
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    log_ho, log_lo = np.log(high / open_), np.log(low / open_)
    log_co = np.log(close / open_)
    log_oc = np.log(open_ / close.shift(1))
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    close_vol = np.log(close / close.shift(1)).rolling(window).var()
    open_vol = log_oc.rolling(window).var()
    rs_vol = rs.rolling(window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    return np.sqrt((open_vol + k * close_vol + (1 - k) * rs_vol).clip(lower=0))


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def zscore(s: pd.Series, window: int) -> pd.Series:
    """Standardise a series against its own rolling mean and standard deviation.

    Args:
        s: Series to standardise.
        window: Rolling window length in bars.

    Returns:
        The rolling z-score.
    """
    m = s.rolling(window, min_periods=window // 2).mean()
    v = s.rolling(window, min_periods=window // 2).std()
    return (s - m) / v.replace(0, np.nan)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative strength index.

    Args:
        close: Close prices.
        window: Smoothing period.

    Returns:
        RSI values in ``[0, 100]``.
    """
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average true range.

    Args:
        df: OHLC frame.
        window: Smoothing period.

    Returns:
        The smoothed true range, in price units.
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window).mean()


def hurst(
    series: pd.Series, window: int = 200, lags: tuple[int, ...] = (2, 4, 8, 16, 32)
) -> pd.Series:
    """Rolling Hurst exponent via the variance-of-lagged-differences estimator.

    H > 0.5 trending, H < 0.5 mean-reverting. Useful as a regime gate: run
    momentum when H is high, run mean reversion when it is low.
    """
    logp = np.log(series)

    def _h(x: np.ndarray) -> float:
        try:
            tau = [np.sqrt(np.std(x[lag:] - x[:-lag])) for lag in lags]
            if min(tau) <= 0:
                return np.nan
            return float(np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0)
        except Exception:
            return np.nan

    return logp.rolling(window).apply(_h, raw=True)


# ---------------------------------------------------------------------------
# feature blocks
# ---------------------------------------------------------------------------
def price_features(df: pd.DataFrame, d_frac: float = 0.4) -> pd.DataFrame:
    """Return / momentum / volatility / microstructure features from OHLCV."""
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    ret = np.log(c).diff()

    out["ret_1"] = ret
    for k in (4, 12, 24, 72, 168, 336, 720):
        out[f"mom_{k}"] = np.log(c / c.shift(k))
    for k in (24, 72, 168):
        out[f"vol_{k}"] = ret.rolling(k, min_periods=k // 2).std()

    # volatility-normalised momentum: the single most robust family of
    # signals in every liquid asset class studied
    for k in (24, 72, 168, 336, 720):
        out[f"tsmom_{k}"] = out[f"mom_{k}"] / (out["vol_168"] * np.sqrt(k))

    out["gk_vol"] = garman_klass(df)
    out["yz_vol"] = yang_zhang(df)
    out["vol_of_vol"] = out["vol_24"].rolling(168, min_periods=84).std()
    out["vol_ratio"] = out["vol_24"] / out["vol_168"].replace(0, np.nan)

    out["rsi_14"] = rsi(c, 14) / 100 - 0.5
    out["atr_pct"] = atr(df, 14) / c
    out["hl_range"] = (df["high"] - df["low"]) / c
    out["close_loc"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan) - 0.5

    # distance from rolling extremes: proxy for breakout / stop clusters
    for k in (24, 168, 720):
        hi = df["high"].rolling(k, min_periods=k // 2).max()
        lo = df["low"].rolling(k, min_periods=k // 2).min()
        out[f"pos_range_{k}"] = (c - lo) / (hi - lo).replace(0, np.nan) - 0.5

    # stationary price memory
    out["fd_logp"] = frac_diff(np.log(c), d_frac)
    out["fd_logp_z"] = zscore(out["fd_logp"], 336)

    # order-flow proxies available in free kline data
    if "taker_buy_base" in df and "volume" in df:
        imb = df["taker_buy_base"] / df["volume"].replace(0, np.nan)
        out["taker_imb"] = imb - 0.5
        out["taker_imb_z"] = zscore(out["taker_imb"], 168)
    if "trades" in df:
        out["trade_intensity"] = zscore(np.log1p(df["trades"]), 168)
    if "quote_volume" in df:
        out["dollar_vol_z"] = zscore(np.log1p(df["quote_volume"]), 168)
        # Amihud illiquidity: |return| per dollar traded. Rising illiquidity
        # precedes violent moves.
        out["amihud"] = zscore(
            (ret.abs() / df["quote_volume"].replace(0, np.nan)).rolling(24).mean(), 336
        )

    # signed volume / return autocorrelation
    out["ret_skew_168"] = ret.rolling(168, min_periods=84).skew()
    out["ret_kurt_168"] = ret.rolling(168, min_periods=84).kurt()
    out["ac1_168"] = ret.rolling(168, min_periods=84).apply(
        lambda x: pd.Series(x).autocorr(1), raw=False
    )
    return out


def funding_features(funding: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Perpetual funding is a direct, observable price of leverage. Extreme positive funding means longs are crowded and paying to stay - historically one of the better-documented contrarian signals in crypto."""
    out = pd.DataFrame(index=index)
    if funding is None or funding.empty:
        return out
    f = funding.set_index("ts")["funding_rate"].sort_index()
    f = f.reindex(index.union(f.index)).ffill().reindex(index)
    out["funding"] = f
    out["funding_z_168"] = zscore(f, 168)
    out["funding_cum_24"] = f.rolling(24).sum()
    out["funding_cum_168"] = f.rolling(168).sum()
    out["funding_sign_persist"] = np.sign(f).rolling(72).mean()
    return out


def oi_features(metrics: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Open interest and positioning. Distinguishes new leverage from unwinds."""
    out = pd.DataFrame(index=index)
    if metrics is None or metrics.empty:
        return out
    m = metrics.set_index("ts").sort_index()
    oi = pd.to_numeric(m.get("sum_open_interest"), errors="coerce")
    if oi is None or oi.dropna().empty:
        return out
    oi = oi.reindex(index.union(oi.index)).ffill().reindex(index)
    out["oi_chg_24"] = np.log(oi / oi.shift(24))
    out["oi_z_168"] = zscore(np.log(oi), 168)
    for col, name in (
        ("count_long_short_ratio", "ls_ratio"),
        ("sum_taker_long_short_vol_ratio", "taker_ls"),
    ):
        if col in m:
            s = pd.to_numeric(m[col], errors="coerce")
            s = s.reindex(index.union(s.index)).ffill().reindex(index)
            out[name] = np.log(s.replace(0, np.nan))
            out[f"{name}_z"] = zscore(out[name], 168)
    return out


def cross_sectional(features: dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    """Rank a feature across the universe at each timestamp. Cross-sectional signals hedge out the market beta that dominates crypto returns, which usually improves Sharpe far more than a better time-series signal does."""
    panel = pd.DataFrame({s: f[col] for s, f in features.items() if col in f})
    ranks = panel.rank(axis=1, pct=True) - 0.5
    return ranks


def build_features(
    klines: pd.DataFrame,
    funding: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
    extra: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Assemble the full feature matrix for one symbol, indexed by timestamp."""
    df = klines.set_index("ts").sort_index()
    blocks = [price_features(df)]
    if funding is not None and not funding.empty:
        blocks.append(funding_features(funding, df.index))
    if metrics is not None and not metrics.empty:
        blocks.append(oi_features(metrics, df.index))
    X = pd.concat(blocks, axis=1)
    if extra:
        for name, s in extra.items():
            s = s.reindex(X.index.union(s.index)).ffill().reindex(X.index)
            X[name] = s
    X = X.replace([np.inf, -np.inf], np.nan)
    X["close"] = df["close"]
    return X
