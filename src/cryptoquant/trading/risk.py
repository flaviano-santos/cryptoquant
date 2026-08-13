"""Position sizing and risk control.

Sizing matters more than signal quality. A mediocre signal sized well beats a
good signal sized badly, because ruin is absorbing: you cannot recover from
-100%, no matter how good your edge was.

Three layers, applied in order:
  1. Volatility targeting  - make each position contribute equal risk, and the
     portfolio hit a constant risk level, so your Sharpe is not an artefact of
     leverage drifting with the vol cycle.
  2. Fractional Kelly      - scale by estimated edge, but never above ~1/4
     Kelly. Full Kelly is the growth-optimal *and* wildly volatile solution,
     and it assumes you know your edge exactly, which you never do.
  3. Hard constraints      - per-asset caps, gross leverage cap, drawdown
     throttle, kill switches. These are not optimisations. They are the part
     that keeps you solvent when the model is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskConfig:
    """Risk limits and sizing parameters.

    Attributes:
        target_annual_vol: Volatility the book is scaled to target.
        max_gross_leverage: Cap on the sum of absolute position weights.
        max_position_per_asset: Cap on any single position weight.
        vol_lookback: Half-life in bars for the volatility estimate.
        kelly_fraction: Multiplier applied to the estimated Kelly fraction.
            Never set this to 1.0; the edge estimate is far too noisy.
        max_daily_loss: Daily loss that trips the kill switch.
        max_drawdown: Peak-to-trough loss that trips the kill switch.
        bars_per_year: Bars per year, used to annualise volatility.
    """

    target_annual_vol: float = 0.20
    max_gross_leverage: float = 2.0
    max_position_per_asset: float = 0.5
    vol_lookback: int = 72
    kelly_fraction: float = 0.25
    max_daily_loss: float = 0.03
    max_drawdown: float = 0.20
    bars_per_year: int = 8760


def vol_target_weights(
    signal: pd.DataFrame, prices: pd.DataFrame, cfg: RiskConfig, max_scale: float = 5.0
) -> pd.DataFrame:
    """Convert a signal in roughly [-1, 1] into position weights.

    Two stages:
      1. Inverse-volatility scaling per asset, so a unit of signal buys the
         same amount of risk in BTC as in a 200%-vol altcoin.
      2. Portfolio-level scaling so that the *realised* book volatility tracks
         the target. This is estimated causally from the trailing volatility of
         the un-scaled book, so there is no lookahead - and it is what actually
         makes the target bind, because assets in crypto are ~0.8 correlated
         and naive per-asset budgeting undershoots badly.
    """
    ret = np.log(prices).diff()
    ann_vol = ret.ewm(halflife=cfg.vol_lookback, min_periods=cfg.vol_lookback // 2).std() * np.sqrt(
        cfg.bars_per_year
    )

    raw = (signal / ann_vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # trailing vol of the un-scaled book -> scale factor to hit the target
    book_ret = (raw.shift(1).fillna(0.0) * ret.fillna(0.0)).sum(axis=1)
    book_vol = book_ret.ewm(
        halflife=cfg.vol_lookback * 4, min_periods=cfg.vol_lookback
    ).std() * np.sqrt(cfg.bars_per_year)
    scale = (cfg.target_annual_vol / book_vol.replace(0, np.nan)).clip(upper=max_scale)
    scale = scale.shift(1).ffill().fillna(0.0)  # strictly causal

    w = raw.mul(scale, axis=0)

    # hard constraints last: per-asset cap, then gross leverage cap
    w = w.clip(-cfg.max_position_per_asset, cfg.max_position_per_asset)
    gross = w.abs().sum(axis=1)
    gross_scale = (cfg.max_gross_leverage / gross.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
    return w.mul(gross_scale, axis=0).fillna(0.0)


def kelly_fraction_from_returns(
    returns: pd.Series, window: int = 720, cap: float = 1.0
) -> pd.Series:
    """Rolling Kelly fraction f* = mu / sigma^2 for continuous returns, estimated out-of-sample on a trailing window. Multiply by cfg.kelly_fraction before use. The estimate is noisy - that is the point of the fraction."""
    mu = returns.rolling(window, min_periods=window // 2).mean()
    var = returns.rolling(window, min_periods=window // 2).var()
    f = (mu / var.replace(0, np.nan)).clip(-cap, cap)
    return f.fillna(0.0)


def drawdown_throttle(equity: pd.Series, start: float = 0.08, stop: float = 0.20) -> pd.Series:
    """Continuously de-risk as drawdown deepens: full size above `start`, linearly to zero at `stop`. Smoother and less path-dependent than a binary switch, and it avoids the classic failure of turning off at the bottom."""
    dd = (equity / equity.cummax() - 1).abs()
    scale = 1 - (dd - start) / (stop - start)
    return scale.clip(0.0, 1.0)


def apply_turnover_buffer(
    target: pd.DataFrame, buffer: float = 0.1, min_band: float = 0.01
) -> pd.DataFrame:
    """Proportional no-trade band. Only move to the new target when it differs from the current position by more than `buffer` times the larger of the two (floored at `min_band` so tiny positions do not dither).

    Typically cuts turnover 40-70% with almost no loss of signal, and is often
    the single biggest net-return improvement available in a strategy that
    trades intraday. Tune it: too wide and you lag the signal, too narrow and
    you donate the edge to the exchange.
    """
    vals = target.to_numpy(dtype="float64")
    prev = np.zeros(vals.shape[1])
    res = np.empty_like(vals)
    for i in range(len(vals)):
        row = vals[i]
        band = buffer * np.maximum.reduce([np.abs(row), np.abs(prev), np.full_like(row, min_band)])
        prev = np.where(np.abs(row - prev) > band, row, prev)
        res[i] = prev
    return pd.DataFrame(res, index=target.index, columns=target.columns)


class KillSwitch:
    """Runtime circuit breaker for live trading. Checked before every order.

    Once tripped it stays tripped until you reset it by hand - deliberately.
    """

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.peak_equity: float | None = None
        self.day_start_equity: float | None = None
        self.day: str | None = None
        self.tripped: str | None = None

    def update(self, equity: float, now: pd.Timestamp) -> str | None:
        """Record the latest equity and test every limit.

        Args:
            equity: Current account equity.
            now: Current time, used to detect the start of a new trading day.

        Returns:
            A description of the breach if one has occurred at any point, or
            ``None`` while the switch remains untripped. Once tripped, the
            reason is returned on every subsequent call until reset by hand.
        """
        day = now.strftime("%Y-%m-%d")
        if self.day != day:
            self.day, self.day_start_equity = day, equity
        self.peak_equity = equity if self.peak_equity is None else max(self.peak_equity, equity)

        if self.day_start_equity and equity / self.day_start_equity - 1 <= -self.cfg.max_daily_loss:
            self.tripped = f"daily loss limit hit: {equity / self.day_start_equity - 1:.2%}"
        if self.peak_equity and equity / self.peak_equity - 1 <= -self.cfg.max_drawdown:
            self.tripped = f"max drawdown hit: {equity / self.peak_equity - 1:.2%}"
        return self.tripped

    @property
    def ok(self) -> bool:
        """Whether trading is still permitted."""
        return self.tripped is None
