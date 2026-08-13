"""Validation - the part that decides whether you make money.

Everything else in this repo is plumbing. This module exists because the
default outcome of quantitative trading research is a beautiful backtest that
loses money live, and there are only a handful of techniques that reliably
detect that before you fund the account.

What breaks naive validation on financial data:

  * Leakage through overlapping labels. A label spanning bars 100-124 shares
    almost all its information with one spanning 101-125. Put one in train and
    one in test and your test score is contaminated. Fix: purging + embargo.
  * Non-stationarity. Random k-fold trains on 2024 and tests on 2021. Fix:
    walk-forward, always forward in time.
  * Multiple testing. If you try 200 variants and keep the best, the best
    Sharpe is mostly luck. A Sharpe of 2.0 selected from 200 trials is roughly
    as impressive as a Sharpe of 0.6 from one. Fix: deflated Sharpe ratio and
    probability of backtest overfitting.
  * Small effective sample size. Five years of hourly bars is 43,800 rows but
    only ~5 independent market regimes. Fix: block bootstrap, not iid.

References: Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio";
Bailey et al. (2017) "The Probability of Backtest Overfitting";
Lopez de Prado, Advances in Financial Machine Learning, ch. 7, 11-12.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Cross-validation that respects time
# ---------------------------------------------------------------------------
class PurgedKFold:
    """K-fold CV with purging and embargo.

    Purging: drop any training observation whose label lifespan overlaps the
    test window. Embargo: additionally drop training observations immediately
    *after* the test window, because serial correlation leaks backwards
    through overlapping labels too.

    t1 : Series mapping each sample's start timestamp to its label end
         timestamp (the `t1` column from triple_barrier).
    """

    def __init__(self, n_splits: int = 5, t1: pd.Series | None = None, embargo_pct: float = 0.01):
        if t1 is None:
            raise ValueError("PurgedKFold needs t1 (label end times)")
        self.n_splits = n_splits
        self.t1 = t1.sort_index()
        self.embargo_pct = embargo_pct

    @staticmethod
    def _i8(values) -> np.ndarray:
        """Timestamps -> int64 nanoseconds, tz-safe."""
        idx = pd.DatetimeIndex(values)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        return idx.asi8

    def split(
        self, X: pd.DataFrame, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate purged, embargoed train/test index pairs.

        Args:
            X: Feature matrix whose index must equal ``t1``'s index.
            y: Ignored; present for scikit-learn splitter compatibility.
            groups: Ignored; present for scikit-learn splitter compatibility.

        Yields:
            ``(train_indices, test_indices)`` positional index arrays.

        Raises:
            ValueError: If ``X`` and ``t1`` do not share an index.
        """
        if not X.index.equals(self.t1.index):
            raise ValueError("X and t1 must share the same index")
        n = len(X)
        indices = np.arange(n)
        embargo = int(n * self.embargo_pct)
        test_ranges = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]

        start_i8 = self._i8(self.t1.index)
        end_i8 = self._i8(self.t1.to_numpy())

        for start, end in test_ranges:
            test_idx = indices[start:end]
            t0_test = start_i8[start]
            t1_test_max = end_i8[start:end].max()

            # purge: training labels that end after the test window starts and
            # begin before the test window ends
            train_mask = np.ones(n, dtype=bool)
            train_mask[start:end] = False
            overlap = (end_i8 > t0_test) & (start_i8 < t1_test_max)
            train_mask &= ~overlap

            # embargo the bars just after the test block
            if embargo > 0:
                lo, hi = end, min(end + embargo, n)
                train_mask[lo:hi] = False

            yield indices[train_mask], test_idx


def walk_forward_splits(
    n: int, n_splits: int = 6, min_train: float = 0.3, expanding: bool = True, embargo: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Anchored (expanding) or rolling walk-forward. Always trains on the past and tests on the future. This is the only validation scheme whose result resembles what live trading will feel like."""
    idx = np.arange(n)
    start_test = int(n * min_train)
    bounds = np.linspace(start_test, n, n_splits + 1).astype(int)
    out = []
    for i in range(n_splits):
        te_lo, te_hi = bounds[i], bounds[i + 1]
        if te_hi - te_lo < 2:
            continue
        tr_hi = max(0, te_lo - embargo)
        tr_lo = 0 if expanding else max(0, tr_hi - start_test)
        if tr_hi - tr_lo < 2:
            continue
        out.append((idx[tr_lo:tr_hi], idx[te_lo:te_hi]))
    return out


# ---------------------------------------------------------------------------
# Multiple-testing-aware Sharpe statistics
# ---------------------------------------------------------------------------
def probabilistic_sharpe(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true Sharpe > benchmark), correcting for skew, fat tails and sample length. Returns are per-bar; benchmark_sr is per-bar too."""
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 30 or r.std(ddof=1) == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    if denom <= 0:
        return np.nan
    z = (sr - benchmark_sr) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe achievable by pure luck across `n_trials` independent backtests whose Sharpes have variance `sr_variance`.

    This is the bar your strategy must clear, and it is higher than people
    expect: with 100 trials it is around 2.5 standard deviations.
    """
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1 - gamma) * z1 + gamma * z2))


def deflated_sharpe(
    returns: pd.Series,
    n_trials: int,
    sr_variance: float | None = None,
    all_trial_sharpes: Sequence[float] | None = None,
) -> dict:
    """Deflated Sharpe Ratio: the probability that the observed Sharpe is real once you account for how many strategies you tried before finding it.

    Pass either sr_variance directly, or the Sharpes of every variant you
    tested (per-bar, not annualised) and it will be computed for you.

    Rule of thumb: DSR < 0.90 means you have not demonstrated anything.
    And be honest about n_trials - it includes every parameter you nudged,
    not just the ones you wrote down.
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 30 or r.std(ddof=1) == 0:
        return {"dsr": np.nan, "sr": np.nan, "sr0": np.nan, "n_trials": n_trials}
    if sr_variance is None:
        if all_trial_sharpes is not None and len(all_trial_sharpes) > 1:
            sr_variance = float(np.var(np.asarray(all_trial_sharpes), ddof=1))
        else:
            sr_variance = float(np.var(r) / n)  # weak fallback
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    dsr = probabilistic_sharpe(returns, benchmark_sr=sr0)
    sr = float(r.mean() / r.std(ddof=1))
    return {"dsr": dsr, "sr": sr, "sr0": sr0, "n_trials": n_trials, "sr_variance": sr_variance}


def min_track_record_length(
    returns: pd.Series, target_sr: float = 0.0, confidence: float = 0.95
) -> float:
    """How many observations you need before a Sharpe this size is statistically distinguishable from `target_sr`. Frequently a humbling number."""
    r = returns.dropna().to_numpy()
    if len(r) < 30 or r.std(ddof=1) == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    if sr <= target_sr:
        return np.inf
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    z = stats.norm.ppf(confidence)
    return float(1 + (1 - skew * sr + (kurt - 1) / 4 * sr**2) * (z / (sr - target_sr)) ** 2)


# ---------------------------------------------------------------------------
# Probability of backtest overfitting (CSCV)
# ---------------------------------------------------------------------------
def probability_of_backtest_overfitting(perf_matrix: pd.DataFrame, s: int = 12) -> dict:
    """Combinatorially Symmetric Cross-Validation.

    perf_matrix : rows = time (per-bar returns), columns = candidate strategy
                  variants. Every variant you considered goes in here, not
                  just the winner.

    Splits time into `s` blocks, forms every way of choosing s/2 blocks as
    in-sample, picks the best variant in-sample, and records where it ranks
    out-of-sample. PBO is the fraction of splits where the in-sample winner
    lands in the bottom half out-of-sample.

    PBO above ~0.5 means your selection procedure is worse than random. Below
    ~0.2 is what you want. Most retail strategies score 0.7+.
    """
    M = perf_matrix.dropna(how="all").fillna(0.0)
    n_rows, n_cols = M.shape
    if n_cols < 2:
        return {"pbo": np.nan, "n_splits": 0}
    s = min(s, max(2, (n_rows // 20) * 2))
    if s % 2:
        s -= 1
    blocks = np.array_split(np.arange(n_rows), s)

    logit_values: list[float] = []
    is_values: list[float] = []
    oos_values: list[float] = []
    for combo in itertools.combinations(range(s), s // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(s) if i not in combo])
        is_sr = _sharpe_cols(M.iloc[is_idx])
        oos_sr = _sharpe_cols(M.iloc[oos_idx])
        best = int(np.nanargmax(is_sr))
        rank = stats.rankdata(oos_sr)[best] / (n_cols + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logit_values.append(float(np.log(rank / (1 - rank))))
        is_values.append(float(is_sr[best]))
        oos_values.append(float(oos_sr[best]))

    logits = np.asarray(logit_values, dtype="float64")
    is_perfs = np.asarray(is_values, dtype="float64")
    oos_perfs = np.asarray(oos_values, dtype="float64")
    ok = np.isfinite(is_perfs) & np.isfinite(oos_perfs)
    slope = float(np.polyfit(is_perfs[ok], oos_perfs[ok], 1)[0]) if ok.sum() > 2 else float("nan")
    return {
        "pbo": float((logits <= 0).mean()),
        "n_splits": len(logits),
        "median_logit": float(np.median(logits)),
        "oos_sharpe_mean": float(np.nanmean(oos_perfs)),
        "is_sharpe_mean": float(np.nanmean(is_perfs)),
        "degradation_slope": float(slope),  # <1 means OOS decays vs IS
        "prob_oos_loss": float((oos_perfs < 0).mean()),
    }


def _sharpe_cols(df: pd.DataFrame) -> np.ndarray:
    mu = df.mean(axis=0).to_numpy()
    sd = df.std(axis=0, ddof=1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, mu / sd, np.nan)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def stationary_bootstrap(
    returns: pd.Series, n_boot: int = 1000, mean_block: int = 48, seed: int = 0
) -> np.ndarray:
    """Politis-Romano stationary bootstrap: resamples blocks of geometrically distributed length, preserving serial dependence. Use this instead of iid resampling for anything time-series."""
    rng = np.random.default_rng(seed)
    r = returns.dropna().to_numpy()
    n = len(r)
    p = 1.0 / mean_block
    out = np.empty((n_boot, n))
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        out[b] = r[idx]
    return out


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_boot: int = 1000,
    mean_block: int = 48,
    alpha: float = 0.05,
    bars_per_year: int = 8760,
    seed: int = 0,
) -> dict:
    """Block-bootstrap confidence interval for the annualised Sharpe."""
    samples = stationary_bootstrap(returns, n_boot, mean_block, seed)
    mu, sd = samples.mean(axis=1), samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        srs = np.where(sd > 0, mu / sd, np.nan) * np.sqrt(bars_per_year)
    srs = srs[np.isfinite(srs)]
    r = returns.dropna()
    point = (
        float(r.mean() / r.std(ddof=1) * np.sqrt(bars_per_year)) if r.std(ddof=1) > 0 else np.nan
    )
    if srs.size == 0:
        return {"sharpe": point, "ci_low": np.nan, "ci_high": np.nan, "p_sharpe_le_0": np.nan}
    return {
        "sharpe": point,
        "ci_low": float(np.quantile(srs, alpha / 2)),
        "ci_high": float(np.quantile(srs, 1 - alpha / 2)),
        "p_sharpe_le_0": float((srs <= 0).mean()),
    }


# ---------------------------------------------------------------------------
# Sanity checks that catch the stupid mistakes
# ---------------------------------------------------------------------------
def leakage_check(
    features: pd.DataFrame, future_return: pd.Series, threshold: float = 0.30
) -> pd.DataFrame:
    """Correlate every feature with the *contemporaneous* future return. Any feature with suspiciously high correlation is almost certainly leaking.

    Run this every time you add a feature. It has saved more projects than
    any model improvement.
    """
    aligned = features.align(future_return, axis=0, join="inner")
    X, y = aligned[0], aligned[1]
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    return corr[corr > threshold].to_frame("abs_corr_with_future_return")


def shuffle_test(
    returns_fn: Callable[[pd.Series], pd.Series], signal: pd.Series, n: int = 200, seed: int = 0
) -> dict:
    """Randomise the signal (preserving its distribution) and re-run. If a shuffled signal produces a similar Sharpe, your strategy is measuring something structural (drift, vol harvesting) rather than predicting."""
    rng = np.random.default_rng(seed)
    real = returns_fn(signal)
    real_sr = real.mean() / real.std(ddof=1) if real.std(ddof=1) > 0 else np.nan
    null_values: list[float] = []
    vals = signal.to_numpy().copy()
    for _ in range(n):
        rng.shuffle(vals)
        r = returns_fn(pd.Series(vals, index=signal.index))
        null_values.append(float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else float("nan"))
    null = np.asarray(null_values, dtype="float64")
    null = null[np.isfinite(null)]
    return {
        "real_sharpe": float(real_sr),
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "p_value": float((null >= real_sr).mean()),
    }


def validation_report(
    returns: pd.Series,
    n_trials: int = 1,
    all_trial_sharpes: Sequence[float] | None = None,
    bars_per_year: int = 8760,
) -> pd.Series:
    """One call that produces the numbers you should look at before deploying."""
    r = returns.dropna()
    if len(r) < 30 or r.std(ddof=1) == 0:
        return pd.Series({"sharpe_ann": np.nan, "note": "degenerate return series"})
    dsr = deflated_sharpe(r, n_trials=n_trials, all_trial_sharpes=all_trial_sharpes)
    boot = bootstrap_sharpe_ci(r, n_boot=500, bars_per_year=bars_per_year)
    out = {
        "sharpe_ann": boot["sharpe"],
        "sharpe_ci_low": boot["ci_low"],
        "sharpe_ci_high": boot["ci_high"],
        "p_sharpe_le_0": boot["p_sharpe_le_0"],
        "psr_vs_0": probabilistic_sharpe(r, 0.0),
        "deflated_sharpe": dsr["dsr"],
        "haircut_sr0_ann": dsr["sr0"] * np.sqrt(bars_per_year),
        "n_trials_assumed": n_trials,
        "min_track_record_bars": min_track_record_length(r),
    }
    return pd.Series(out)
