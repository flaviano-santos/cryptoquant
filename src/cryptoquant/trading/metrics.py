"""Performance and reporting statistics.

A deliberate note on the Sharpe ratio, because getting it wrong is common and
expensive. The annualised Sharpe is::

    mean(returns) / std(returns) * sqrt(periods_per_year)

Compounding the mean return in the numerator while scaling volatility by the
square root of time in the denominator is *not* equivalent - the numerator then
grows super-linearly in the mean and the ratio inflates whenever returns are
large. On daily Bitcoin data that mistake overstates the Sharpe by roughly a
factor of two.

Every statistic here is computed on per-bar returns and annualised with an
explicit ``bars_per_year``, so hourly and daily results stay comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "drawdown_series",
    "monthly_table",
    "performance_stats",
    "rolling_sharpe",
]


def performance_stats(
    returns: pd.Series,
    bars_per_year: int = 8760,
    turnover: pd.Series | None = None,
    gross: pd.Series | None = None,
    rf: float = 0.0,
) -> pd.Series:
    """Summarise a return series.

    Args:
        returns: Net per-bar simple returns.
        bars_per_year: Bars per year, used for annualisation.
        turnover: Optional per-bar turnover, used to report annual turnover.
        gross: Optional pre-cost returns, used to report the Sharpe lost to
            transaction costs.
        rf: Annual risk-free rate.

    Returns:
        A ``Series`` of named statistics, empty if the input is degenerate
        (fewer than two observations or zero variance).
    """
    r = returns.dropna()
    if r.empty or r.std() == 0:
        return pd.Series(dtype="float64")

    n_years = len(r) / bars_per_year
    total = float((1 + r).prod())
    cagr = total ** (1 / n_years) - 1 if n_years > 0 and total > 0 else np.nan
    ann_vol = float(r.std() * np.sqrt(bars_per_year))
    sharpe = float((r.mean() - rf / bars_per_year) / r.std() * np.sqrt(bars_per_year))

    downside = r[r < 0].std()
    sortino = float(r.mean() / downside * np.sqrt(bars_per_year)) if downside > 0 else np.nan

    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan

    wins, losses = r[r > 0], r[r < 0]
    profit_factor = (
        float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.nan
    )

    out = {
        "total_return": total - 1,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "hit_rate": float((r > 0).mean()),
        "profit_factor": profit_factor,
        "skew": float(r.skew()),
        "kurtosis": float(r.kurt()),
        "tail_ratio": float(np.percentile(r, 95) / abs(np.percentile(r, 5)))
        if np.percentile(r, 5) != 0
        else np.nan,
        "worst_bar": float(r.min()),
        "n_bars": len(r),
        "years": n_years,
    }
    if turnover is not None:
        out["ann_turnover"] = float(turnover.mean() * bars_per_year)
    if gross is not None:
        g = gross.dropna()
        if g.std() > 0:
            out["gross_sharpe"] = float(g.mean() / g.std() * np.sqrt(bars_per_year))
            out["cost_drag_sharpe"] = out["gross_sharpe"] - sharpe
    return pd.Series(out)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute the running drawdown from a series of per-bar returns.

    Args:
        returns: Per-bar simple returns.

    Returns:
        Drawdown as a non-positive fraction of the running peak.
    """
    eq = (1 + returns.fillna(0)).cumprod()
    return eq / eq.cummax() - 1


def rolling_sharpe(returns: pd.Series, window: int, bars_per_year: int = 8760) -> pd.Series:
    """Compute an annualised Sharpe ratio over a rolling window.

    Useful for spotting a strategy whose edge is concentrated in one regime
    rather than persistent.

    Args:
        returns: Per-bar simple returns.
        window: Rolling window length in bars.
        bars_per_year: Bars per year, used to annualise.

    Returns:
        The rolling annualised Sharpe ratio, ``NaN`` where undefined.
    """
    m = returns.rolling(window).mean()
    s = returns.rolling(window).std()
    return m / s.replace(0, np.nan) * np.sqrt(bars_per_year)


def monthly_table(returns: pd.Series) -> pd.DataFrame:
    """Pivot returns into a year-by-month grid.

    The fastest way to see whether a strategy worked throughout the sample or
    only during one favourable stretch.

    Args:
        returns: Per-bar simple returns with a ``DatetimeIndex``.

    Returns:
        A frame indexed by year with one column per calendar month.
    """
    m = (1 + returns.fillna(0)).resample("ME").prod() - 1
    df = m.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    return df.pivot_table(index="year", columns="month", values="ret")
