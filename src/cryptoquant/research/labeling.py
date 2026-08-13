"""Labeling.

The default in most retail ML-for-trading code is "predict the sign of the
next bar's return". It is close to worthless, for three reasons:

  * Fixed horizons ignore volatility. The same +0.5% is noise in one regime
    and a real move in another.
  * It ignores path. A trade that hits your stop before your target is a loss
    even if the end-of-horizon return is positive.
  * Overlapping labels are massively autocorrelated, which destroys the IID
    assumption behind every cross-validation score you compute.

The triple-barrier method fixes the first two. Sample-uniqueness weighting and
purged cross-validation fix the third.

Reference: Lopez de Prado, Advances in Financial Machine Learning, ch. 3-4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_vol(close: pd.Series, span: int = 100, bars_per_day: int = 24) -> pd.Series:
    """EWMA volatility of returns measured over one day, used to scale barriers."""
    idx = close.index
    ret = close / close.shift(bars_per_day) - 1
    return ret.ewm(span=span, min_periods=span // 2).std().reindex(idx)


def vertical_barriers(
    index: pd.DatetimeIndex, t_events: pd.DatetimeIndex, num_bars: int
) -> pd.Series:
    """Timeout barrier: the bar `num_bars` ahead of each event."""
    pos = index.searchsorted(t_events)
    end = np.minimum(pos + num_bars, len(index) - 1)
    return pd.Series(index[end], index=t_events, name="t1")


def triple_barrier(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    target: pd.Series,
    pt_sl: tuple[float, float] = (2.0, 1.0),
    min_target: float = 0.001,
    num_bars: int = 24,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """For each event, which barrier is touched first?

    close       : price series indexed by timestamp
    t_events    : timestamps at which a bet is considered
    target      : volatility estimate at each event (barrier width unit)
    pt_sl       : (profit-take, stop-loss) multiples of target
    num_bars    : vertical barrier (max holding period)
    side        : if given, barriers become asymmetric around the bet direction
                  and the returned label is meta (1 = take the bet, 0 = skip).

    Returns a frame with t1 (exit time), ret (realised return), bin (label),
    and trgt.
    """
    t_events = pd.DatetimeIndex(t_events)
    target = target.reindex(t_events).ffill()
    keep = target > min_target
    t_events, target = t_events[keep.values], target[keep.values]
    if len(t_events) == 0:
        return pd.DataFrame(columns=["t1", "ret", "bin", "trgt", "side"])

    t1 = vertical_barriers(close.index, t_events, num_bars)
    side_ = pd.Series(1.0, index=t_events) if side is None else side.reindex(t_events).fillna(0.0)

    pt = pt_sl[0] * target
    sl = -pt_sl[1] * target

    out = pd.DataFrame(index=t_events, columns=["t1", "ret", "bin", "trgt", "side"], dtype="object")
    out["trgt"] = target
    out["side"] = side_

    arr_idx = close.index
    prices = close.to_numpy(dtype="float64")
    pos_start = arr_idx.searchsorted(t_events)
    pos_end = arr_idx.searchsorted(t1.to_numpy())

    for i, (p0, p1) in enumerate(zip(pos_start, pos_end, strict=True)):
        if p1 <= p0:
            continue
        path = prices[p0 : p1 + 1] / prices[p0] - 1.0
        path = path * side_.iloc[i]
        hit_pt = np.argmax(path > pt.iloc[i]) if (path > pt.iloc[i]).any() else None
        hit_sl = np.argmax(path < sl.iloc[i]) if (path < sl.iloc[i]).any() else None
        candidates = [c for c in (hit_pt, hit_sl) if c is not None]
        touch = min(candidates) if candidates else len(path) - 1
        out.iat[i, out.columns.get_loc("t1")] = arr_idx[p0 + touch]
        out.iat[i, out.columns.get_loc("ret")] = float(path[touch])

    out = out.dropna(subset=["t1"])
    out["ret"] = out["ret"].astype("float64")
    out["t1"] = pd.to_datetime(out["t1"], utc=True)

    if side is None:
        out["bin"] = np.sign(out["ret"]).replace(0, 0).astype("int8")
    else:
        # meta-labeling: 1 if acting on the primary signal would have made money
        out["bin"] = (out["ret"] > 0).astype("int8")
    return out


def count_concurrent(bar_index: pd.DatetimeIndex, events: pd.DataFrame) -> pd.Series:
    """How many labels are live at each bar."""
    t1 = events["t1"]
    counts = pd.Series(0.0, index=bar_index)
    starts = bar_index.searchsorted(events.index)
    ends = bar_index.searchsorted(t1.to_numpy())
    delta = np.zeros(len(bar_index) + 1)
    for s, e in zip(starts, ends, strict=True):
        delta[s] += 1
        delta[min(e + 1, len(bar_index))] -= 1
    counts.iloc[:] = np.cumsum(delta[:-1])
    return counts


def sample_uniqueness(bar_index: pd.DatetimeIndex, events: pd.DataFrame) -> pd.Series:
    """Average uniqueness of each label: 1 / (number of labels overlapping it), averaged over its life. Feed this to the model as sample_weight so that a cluster of 24 near-identical overlapping labels does not count as 24 independent observations. This single line removes a lot of illusory skill."""
    conc = count_concurrent(bar_index, events).replace(0, np.nan)
    starts = bar_index.searchsorted(events.index)
    ends = bar_index.searchsorted(events["t1"].to_numpy())
    inv = (1.0 / conc).to_numpy()
    u = np.array(
        [
            np.nanmean(inv[s : e + 1]) if e >= s else np.nan
            for s, e in zip(starts, ends, strict=True)
        ]
    )
    return pd.Series(u, index=events.index, name="uniqueness").fillna(0.0)


def return_weights(
    bar_index: pd.DatetimeIndex, events: pd.DataFrame, close: pd.Series
) -> pd.Series:
    """Weight samples by |attributed return| as well as uniqueness, so the model cares more about the observations that actually moved money."""
    conc = count_concurrent(bar_index, events).replace(0, np.nan)
    logret = np.log(close).diff().to_numpy()
    inv = (1.0 / conc).to_numpy()
    starts = bar_index.searchsorted(events.index)
    ends = bar_index.searchsorted(events["t1"].to_numpy())
    w = np.array(
        [
            np.abs(np.nansum(logret[s : e + 1] * inv[s : e + 1])) if e >= s else 0.0
            for s, e in zip(starts, ends, strict=True)
        ]
    )
    w = pd.Series(w, index=events.index, name="weight")
    return w / w.mean() if w.mean() > 0 else w


def time_decay(uniqueness: pd.Series, last_weight: float = 0.5) -> pd.Series:
    """Linearly decay the weight of old observations. last_weight=1 disables decay; 0 means the oldest sample gets zero weight. Markets change; recent data deserves more say, but do not set this too aggressively or you throw away the only out-of-regime data you have."""
    cum = uniqueness.sort_index().cumsum()
    if cum.iloc[-1] == 0:
        return uniqueness
    slope = (
        (1.0 - last_weight) / cum.iloc[-1]
        if last_weight >= 0
        else 1.0 / ((last_weight + 1) * cum.iloc[-1])
    )
    const = 1.0 - slope * cum.iloc[-1]
    w = const + slope * cum
    return w.clip(lower=0)


def cusum_events(close: pd.Series, threshold: float | pd.Series) -> pd.DatetimeIndex:
    """CUSUM filter: sample a bet only when cumulative log-return since the last event exceeds a threshold. This does two useful things - it removes the huge redundancy of labelling every single bar, and it concentrates your training set on moments where something is actually happening.

    Pass a Series threshold (e.g. 1.0 * daily_vol) to make it adaptive.
    """
    logret = np.log(close).diff().fillna(0.0)
    if isinstance(threshold, pd.Series):
        thr = threshold.reindex(close.index).ffill()
    else:
        thr = pd.Series(float(threshold), index=close.index)
    s_pos = s_neg = 0.0
    events = []
    for t, r in logret.items():
        h = thr.get(t, np.nan)
        if not np.isfinite(h) or h <= 0:
            continue
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_pos > h:
            s_pos = 0.0
            events.append(t)
        elif s_neg < -h:
            s_neg = 0.0
            events.append(t)
    return pd.DatetimeIndex(events)
