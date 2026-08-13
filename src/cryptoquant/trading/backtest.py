"""Backtest engine.

Deliberately vectorised and deliberately pessimistic. The purpose of a
backtest is not to show you a nice equity curve; it is to *fail* strategies
cheaply. Everything here is biased against the strategy:

  * Signals computed on bar t are executed at bar t+1 (configurable lag).
    No same-bar fills, ever.
  * Costs are charged on turnover at taker rates plus slippage.
  * Perpetual funding is charged on the actual historical rate when available.
  * Positions are capped before, not after, the performance is measured.

Known limitations you should keep in mind (no free backtester escapes these):
  * No order book. Fills are assumed at close +/- slippage. If your strategy
    trades faster than ~5 minutes or sizes above a small fraction of bar
    volume, this understates cost badly.
  * Survivorship: if you select today's top-10 coins and backtest to 2020,
    you have baked in the answer. Use a point-in-time universe.
  * Liquidations and margin mechanics are not simulated. Keep leverage low.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import performance_stats


# ---------------------------------------------------------------------------
@dataclass
class Costs:
    """Transaction cost model.

    Attributes:
        taker_fee_bps: Fee in basis points when crossing the spread.
        maker_fee_bps: Fee in basis points when providing liquidity.
        slippage_bps: Additional cost in basis points covering half-spread and
            market impact. Deliberately pessimistic by default.
        maker_share: Fraction of volume assumed to be filled passively. The
            default of ``0.0`` assumes you always cross, which is the safe
            assumption unless you have measured otherwise.
    """

    taker_fee_bps: float = 4.5
    maker_fee_bps: float = 2.0
    slippage_bps: float = 2.0
    maker_share: float = 0.0  # 0.0 = assume you always cross the spread

    @property
    def per_turnover(self) -> float:
        """Total cost charged per unit of turnover, as a fraction of notional."""
        fee = self.maker_share * self.maker_fee_bps + (1 - self.maker_share) * self.taker_fee_bps
        return (fee + self.slippage_bps) / 1e4


@dataclass
class BacktestResult:
    """Everything a backtest produces.

    Attributes:
        equity: Compounded equity curve.
        returns: Net per-bar returns after costs and funding.
        gross_returns: Per-bar returns before costs.
        positions: Realised position weights after the execution lag.
        turnover: Per-bar absolute change in gross position.
        costs: Per-bar transaction costs as a fraction of equity.
        funding_cost: Per-bar perpetual funding paid (positive) or received.
        bars_per_year: Bars per year, used to annualise statistics.
    """

    equity: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    funding_cost: pd.Series
    bars_per_year: int = 8760

    @property
    def stats(self) -> pd.Series:
        """Summary performance statistics for this backtest."""
        return performance_stats(
            self.returns, self.bars_per_year, turnover=self.turnover, gross=self.gross_returns
        )

    def summary(self) -> str:
        """Render the statistics as an aligned, printable block of text."""
        s = self.stats
        lines = [f"{k:<22}{v: >12.4f}" for k, v in s.items()]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    costs: Costs | None = None,
    funding: pd.DataFrame | None = None,
    lag: int = 1,
    bars_per_year: int = 8760,
    initial_equity: float = 10_000.0,
    max_gross: float | None = None,
) -> BacktestResult:
    """Prices  : wide close prices, index = ts, columns = symbols weights : target position as a signed fraction of equity, same shape.

              +0.5 = long 50% of equity notional. Computed from information
              available at that bar's close.
    funding : wide funding rates per bar (already forward-filled), same shape.
              Positive rate = longs pay shorts.
    lag     : bars between signal and execution. 1 is the minimum honest value.
    """
    costs = costs or Costs()
    prices = prices.sort_index()
    weights = weights.reindex(prices.index).reindex(columns=prices.columns).fillna(0.0)

    # --- enforce the execution lag -----------------------------------------
    pos = weights.shift(lag).fillna(0.0)

    if max_gross is not None:
        gross = pos.abs().sum(axis=1)
        scale = (max_gross / gross.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
        pos = pos.mul(scale, axis=0)

    asset_ret = prices.pct_change().fillna(0.0)

    gross_ret = (pos.shift(1).fillna(0.0) * asset_ret).sum(axis=1)

    turnover = (pos - pos.shift(1).fillna(0.0)).abs().sum(axis=1)
    trade_cost = turnover * costs.per_turnover

    if funding is not None and not funding.empty:
        f = funding.reindex(prices.index).reindex(columns=prices.columns).fillna(0.0)
        fund_cost = (pos.shift(1).fillna(0.0) * f).sum(axis=1)  # long pays positive funding
    else:
        fund_cost = pd.Series(0.0, index=prices.index)

    net_ret = gross_ret - trade_cost - fund_cost
    equity = initial_equity * (1 + net_ret).cumprod()

    return BacktestResult(
        equity=equity,
        returns=net_ret,
        gross_returns=gross_ret,
        positions=pos,
        turnover=turnover,
        costs=trade_cost,
        funding_cost=fund_cost,
        bars_per_year=bars_per_year,
    )


# ---------------------------------------------------------------------------
