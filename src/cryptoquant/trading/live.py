"""Live execution.

Read this before you use it.

The gap between a validated backtest and a profitable live system is mostly
engineering, and the failures are mundane: a websocket drops and you trade on
stale prices; a partial fill leaves you with a position the model does not
know about; a restart loses state and you double up; the exchange rejects an
order for a precision rule and your code silently swallows the exception.

So this module is built around three principles:

1. **State is derived from the exchange, never from memory.** Before every
   decision the runner asks the exchange what it actually holds. If local and
   remote disagree, remote wins.
2. **Everything is off by default.** `dry_run` logs the orders it would send.
   `testnet` points at Binance's free futures testnet. You must switch both
   off deliberately, in config, to touch real money.
3. **The kill switch is checked before every order**, and once tripped it
   requires a manual reset.

Progression, and do not skip steps:
    backtest -> purged walk-forward -> paper on testnet (>=1 month)
    -> live with money you can lose entirely -> scale slowly

Nothing in this file constitutes financial advice, and no backtest result
implies a live result.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .risk import KillSwitch, RiskConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
@dataclass
class Broker:
    """Thin CCXT wrapper. Public data needs no keys; trading reads them from env."""

    exchange_id: str = "binanceusdm"
    testnet: bool = True
    dry_run: bool = True
    default_type: str = "future"
    _ex: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import ccxt

        params = {
            "enableRateLimit": True,
            "options": {"defaultType": self.default_type, "adjustForTimeDifference": True},
        }
        key, secret = os.environ.get("CQ_API_KEY"), os.environ.get("CQ_API_SECRET")
        if key and secret:
            params |= {"apiKey": key, "secret": secret}
        elif not self.dry_run:
            raise RuntimeError(
                "Live trading requested but CQ_API_KEY / CQ_API_SECRET are not set. "
                "Never hard-code keys; export them in your shell. "
                "Use API keys restricted to trading, with withdrawals disabled "
                "and an IP allowlist."
            )
        self._ex = getattr(ccxt, self.exchange_id)(params)
        if self.testnet:
            self._ex.set_sandbox_mode(True)
        self._ex.load_markets()

    # -- market data --------------------------------------------------------
    def bars(self, symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
        """Fetch the most recent OHLCV bars for a symbol.

        Args:
            symbol: Unified CCXT symbol.
            timeframe: Bar size, e.g. ``"1h"``.
            limit: Number of bars to request.

        Returns:
            A frame with a UTC ``ts`` column and OHLCV values.
        """
        raw = self._ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df.drop(columns="open_time")

    def mark_price(self, symbol: str) -> float:
        """Return the last traded price for a symbol.

        Args:
            symbol: Unified CCXT symbol.

        Returns:
            The last price, used to convert target weights into quantities.
        """
        t = self._ex.fetch_ticker(symbol)
        return float(t.get("last") or t.get("close"))

    def funding_rate(self, symbol: str) -> float:
        """Return the current perpetual funding rate.

        Args:
            symbol: Unified CCXT symbol.

        Returns:
            The funding rate, or ``0.0`` if the venue does not report one.
        """
        try:
            return float(self._ex.fetch_funding_rate(symbol).get("fundingRate", 0.0) or 0.0)
        except Exception:
            return 0.0

    # -- account ------------------------------------------------------------
    def equity(self) -> float:
        """Return current account equity in quote currency.

        When running dry without credentials, a notional paper balance is used
        so that sizing logic can be exercised end to end. Override it with the
        ``CQ_PAPER_EQUITY`` environment variable.

        Returns:
            Account equity.
        """
        if self.dry_run and not os.environ.get("CQ_API_KEY"):
            return float(os.environ.get("CQ_PAPER_EQUITY", 10_000))
        bal = self._ex.fetch_balance()
        total = bal.get("total", {})
        return float(total.get("USDT") or total.get("USD") or 0.0)

    def positions(self) -> dict[str, float]:
        """Signed base-asset quantity per symbol, straight from the exchange."""
        if self.dry_run and not os.environ.get("CQ_API_KEY"):
            return {}
        out: dict[str, float] = {}
        for p in self._ex.fetch_positions():
            amt = float(p.get("contracts") or 0.0)
            if amt == 0:
                continue
            side = (p.get("side") or "").lower()
            out[p["symbol"]] = amt if side == "long" else -amt
        return out

    # -- orders -------------------------------------------------------------
    def market_order(self, symbol: str, qty: float, reduce_only: bool = False) -> dict | None:
        """Qty > 0 buys, qty < 0 sells. Rounded to the exchange's precision."""
        if qty == 0:
            return None
        market = self._ex.market(symbol)
        amount = abs(float(self._ex.amount_to_precision(symbol, abs(qty))))
        min_amt = (market.get("limits", {}).get("amount", {}) or {}).get("min") or 0
        if amount <= 0 or amount < min_amt:
            log.info("skip %s: size %.8f below exchange minimum %.8f", symbol, amount, min_amt)
            return None
        side = "buy" if qty > 0 else "sell"
        params = {"reduceOnly": True} if reduce_only else {}

        if self.dry_run:
            log.info("[DRY RUN] %s %s %s", side, amount, symbol)
            return {"dry_run": True, "symbol": symbol, "side": side, "amount": amount}
        order = self._ex.create_order(symbol, "market", side, amount, params=params)
        log.info("SENT %s %s %s -> id=%s", side, amount, symbol, order.get("id"))
        return order

    def close_all(self) -> None:
        """Flatten every open position with reduce-only market orders."""
        for sym, qty in self.positions().items():
            self.market_order(sym, -qty, reduce_only=True)


# ---------------------------------------------------------------------------
@dataclass
class LiveRunner:
    """The trading loop.

    Every cycle:

    1. Pull fresh bars and verify they are not stale.
    2. Rebuild features and the target signal.
    3. Read actual positions from the exchange, not from memory.
    4. Check the kill switch.
    5. Trade only the difference, and only when it exceeds the no-trade band.

    Attributes:
        broker: Exchange connection.
        symbols: Instruments to trade, as unified CCXT symbols.
        signal_fn: Maps ``{symbol: bars}`` onto target weights.
        risk: Risk limits, including the kill-switch thresholds.
        timeframe: Bar size the strategy operates on.
        poll_seconds: Delay between cycles.
        rebalance_band: No-trade band as a fraction of the target notional.
        max_bar_staleness_min: Refuse to trade if the newest bar is older than
            one bar plus this many minutes.
        state_path: Where each cycle's decision record is written.
    """

    broker: Broker
    symbols: list[str]
    signal_fn: Callable[[dict[str, pd.DataFrame]], pd.DataFrame]
    risk: RiskConfig
    timeframe: str = "1h"
    poll_seconds: int = 60
    rebalance_band: float = 0.10
    max_bar_staleness_min: int = 15
    state_path: Path = Path("./data/live_state.json")

    def __post_init__(self) -> None:
        self.kill = KillSwitch(self.risk)
        self.state_path = Path(self.state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    def _fetch(self) -> dict[str, pd.DataFrame]:
        out = {}
        now = pd.Timestamp.now(tz="UTC")
        for s in self.symbols:
            df = self.broker.bars(s, self.timeframe, limit=1500)
            if df.empty:
                raise RuntimeError(f"no bars for {s}")
            age = (now - df["ts"].iloc[-1]).total_seconds() / 60
            bar_min = pd.Timedelta(self.timeframe).total_seconds() / 60
            if age > bar_min + self.max_bar_staleness_min:
                raise RuntimeError(f"stale data for {s}: last bar {age:.0f} min old")
            out[s] = df
        return out

    def _target_qty(
        self, weights: pd.Series, equity: float, prices: dict[str, float]
    ) -> dict[str, float]:
        return {
            s: (weights.get(s, 0.0) * equity) / prices[s] for s in self.symbols if prices.get(s)
        }

    def _save_state(self, payload: dict) -> None:
        payload["ts"] = datetime.now(timezone.utc).isoformat()
        self.state_path.write_text(json.dumps(payload, indent=2, default=str))

    # -----------------------------------------------------------------------
    def step(self) -> dict:
        """Run one decision cycle.

        Returns:
            A record of the cycle: equity, target weights, target and current
            quantities, and the orders that were placed.

        Raises:
            SystemExit: If a risk limit is breached. Positions are flattened
                first, and the switch must be reset by hand before restarting.
            RuntimeError: If market data is missing or stale.
        """
        bars = self._fetch()
        weights = self.signal_fn(bars).iloc[-1]

        equity = self.broker.equity()
        breach = self.kill.update(equity, pd.Timestamp.now(tz="UTC"))
        if breach:
            log.error("KILL SWITCH: %s - flattening and halting", breach)
            self.broker.close_all()
            self._save_state({"halted": True, "reason": breach, "equity": equity})
            raise SystemExit(1)

        prices = {s: self.broker.mark_price(s) for s in self.symbols}
        target = self._target_qty(weights, equity, prices)
        current = self.broker.positions()

        actions = []
        for sym, tgt in target.items():
            cur = current.get(sym, 0.0)
            notional_gap = abs(tgt - cur) * prices[sym]
            band = self.rebalance_band * equity * abs(weights.get(sym, 0.0) or self.rebalance_band)
            if notional_gap < max(band, 0.002 * equity):
                continue
            # An order is reduce-only when it shrinks the position without
            # crossing through zero. Closing to flat counts: np.sign(0) is 0,
            # so comparing signs alone would wrongly mark a full close as an
            # opening trade and let the exchange flip the position on overshoot.
            reduce_only = bool(
                abs(tgt) < abs(cur) and (tgt == 0.0 or float(np.sign(tgt)) == float(np.sign(cur)))
            )
            self.broker.market_order(sym, tgt - cur, reduce_only=reduce_only)
            actions.append({"symbol": sym, "delta": tgt - cur, "target": tgt, "current": cur})

        state = {
            "equity": equity,
            "weights": weights.to_dict(),
            "target_qty": target,
            "current_qty": current,
            "actions": actions,
        }
        self._save_state(state)
        return state

    def run(self) -> None:
        """Run cycles forever, sleeping between them.

        Transient errors are logged and skipped rather than allowed to kill the
        process, but a failed cycle never trades: the loop simply waits for the
        next one. A risk breach is the one condition that stops the loop.
        """
        log.info(
            "live runner starting | testnet=%s dry_run=%s symbols=%s",
            self.broker.testnet,
            self.broker.dry_run,
            self.symbols,
        )
        if not self.broker.testnet and not self.broker.dry_run:
            log.warning("REAL MONEY MODE. Positions will be opened with live capital.")
        while True:
            try:
                st = self.step()
                log.info("equity=%.2f actions=%d", st["equity"], len(st["actions"]))
            except SystemExit:
                raise
            except Exception as exc:
                # never let a transient error take the process down; never
                # trade on a failed cycle either
                log.exception("cycle failed, skipping: %s", exc)
            time.sleep(self.poll_seconds)
