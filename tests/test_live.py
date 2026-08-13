"""Tests for the live trading loop, using a fake exchange.

Live execution is where real money is lost, and the failures are mundane:
stale data, positions drifting out of sync with the exchange, a risk limit that
does not actually stop anything. None of that needs a network connection to
test, so none of these tests use one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cryptoquant.data.synthetic import make_ohlcv
from cryptoquant.trading.live import LiveRunner
from cryptoquant.trading.risk import RiskConfig


class FakeBroker:
    """An in-memory stand-in for :class:`~cryptoquant.trading.live.Broker`.

    Records the orders it is asked to place so tests can assert on them, and
    lets a test drive equity, prices and reported positions directly.
    """

    def __init__(self, bars: pd.DataFrame, equity_value: float = 10_000.0) -> None:
        self._bars = bars
        self._equity = equity_value
        self._positions: dict[str, float] = {}
        self.orders: list[tuple[str, float, bool]] = []
        self.testnet = True
        self.dry_run = True
        self.closed_all = False

    def bars(self, symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
        return self._bars.tail(limit).copy()

    def mark_price(self, symbol: str) -> float:
        return float(self._bars["close"].iloc[-1])

    def equity(self) -> float:
        return self._equity

    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    def market_order(self, symbol: str, qty: float, reduce_only: bool = False):
        self.orders.append((symbol, qty, reduce_only))
        self._positions[symbol] = self._positions.get(symbol, 0.0) + qty
        return {"symbol": symbol, "qty": qty}

    def close_all(self) -> None:
        self.closed_all = True
        self._positions.clear()


@pytest.fixture
def fresh_bars() -> pd.DataFrame:
    """Synthetic bars whose most recent timestamp is 'now'."""
    frame = make_ohlcv(n=600, seed=5, freq="1h")
    now = pd.Timestamp.now(tz="UTC").floor("h")
    frame["ts"] = pd.date_range(end=now, periods=len(frame), freq="1h", tz="UTC")
    return frame


def _runner(
    broker: FakeBroker, tmp_path: Path, weight: float = 0.5, risk: RiskConfig | None = None
) -> LiveRunner:
    """Build a runner whose signal is a fixed target weight."""

    def signal_fn(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        index = pd.DatetimeIndex(next(iter(bars.values()))["ts"])
        return pd.DataFrame({"X/USDT:USDT": weight}, index=index)

    return LiveRunner(
        broker=broker,
        symbols=["X/USDT:USDT"],
        signal_fn=signal_fn,
        risk=risk or RiskConfig(),
        timeframe="1h",
        state_path=tmp_path / "live_state.json",
    )


class TestStaleness:
    """The runner must refuse to trade on old data."""

    def test_stale_bars_raise(self, tmp_path):
        frame = make_ohlcv(n=300, seed=1, freq="1h")
        frame["ts"] = pd.date_range("2020-01-01", periods=len(frame), freq="1h", tz="UTC")
        runner = _runner(FakeBroker(frame), tmp_path)
        with pytest.raises(RuntimeError, match="stale"):
            runner.step()

    def test_fresh_bars_are_accepted(self, tmp_path, fresh_bars):
        runner = _runner(FakeBroker(fresh_bars), tmp_path)
        assert runner.step()["equity"] == 10_000.0


class TestOrderGeneration:
    """Orders must close the gap between target and actual position."""

    def test_opens_a_position_from_flat(self, tmp_path, fresh_bars):
        broker = FakeBroker(fresh_bars)
        state = _runner(broker, tmp_path, weight=0.5).step()
        assert len(broker.orders) == 1
        assert broker.orders[0][1] > 0
        assert state["actions"]

    def test_no_order_inside_the_no_trade_band(self, tmp_path, fresh_bars):
        broker = FakeBroker(fresh_bars)
        price = broker.mark_price("X/USDT:USDT")
        broker._positions["X/USDT:USDT"] = 0.5 * broker.equity() / price
        _runner(broker, tmp_path, weight=0.5).step()
        assert broker.orders == []

    def test_position_state_comes_from_the_exchange(self, tmp_path, fresh_bars):
        """A position the runner never opened must still be reconciled."""
        broker = FakeBroker(fresh_bars)
        price = broker.mark_price("X/USDT:USDT")
        broker._positions["X/USDT:USDT"] = -2.0 * broker.equity() / price
        _runner(broker, tmp_path, weight=0.5).step()
        assert len(broker.orders) == 1
        assert broker.orders[0][1] > 0  # buys back through flat to the target

    def test_zero_target_closes_the_position(self, tmp_path, fresh_bars):
        broker = FakeBroker(fresh_bars)
        price = broker.mark_price("X/USDT:USDT")
        broker._positions["X/USDT:USDT"] = 0.5 * broker.equity() / price
        _runner(broker, tmp_path, weight=0.0).step()
        assert len(broker.orders) == 1
        assert broker.orders[0][1] < 0
        assert broker.orders[0][2] is True  # reduce-only


class TestKillSwitch:
    """A risk breach must flatten and halt."""

    def test_drawdown_breach_flattens_and_exits(self, tmp_path, fresh_bars):
        broker = FakeBroker(fresh_bars, equity_value=10_000)
        runner = _runner(broker, tmp_path, risk=RiskConfig(max_daily_loss=0.02))
        runner.step()

        broker._equity = 9_000  # a 10% intraday loss
        with pytest.raises(SystemExit):
            runner.step()
        assert broker.closed_all
        assert not runner.kill.ok

    def test_state_file_records_the_halt(self, tmp_path, fresh_bars):
        broker = FakeBroker(fresh_bars, equity_value=10_000)
        runner = _runner(broker, tmp_path, risk=RiskConfig(max_daily_loss=0.02))
        runner.step()
        broker._equity = 9_000
        with pytest.raises(SystemExit):
            runner.step()

        import json

        state = json.loads((tmp_path / "live_state.json").read_text())
        assert state["halted"] is True
        assert "daily loss" in state["reason"]


def test_state_is_written_every_cycle(tmp_path, fresh_bars):
    """Every decision must leave an audit trail on disk."""
    runner = _runner(FakeBroker(fresh_bars), tmp_path)
    runner.step()
    path = tmp_path / "live_state.json"
    assert path.is_file()

    import json

    state = json.loads(path.read_text())
    assert {"equity", "weights", "target_qty", "current_qty", "actions", "ts"} <= set(state)
