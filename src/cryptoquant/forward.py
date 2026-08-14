"""Tamper-evident forward-test evidence.

The cutoff and research implementation are frozen before observations arrive.
Each subsequent snapshot is chained to the previous one by SHA-256 so edits are
detectable. This is research bookkeeping, not an execution or custody system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from .config import Config, load_config
from .data.store import Store, ingest
from .pipeline import evaluate, load_feature_data
from .trading.strategies import AdaptiveMultiFactor

ROOT = Path(__file__).resolve().parents[2]
STRATEGY_FILE = ROOT / "src/cryptoquant/trading/strategies/adaptive_multifactor.py"
CONFIG_FILE = ROOT / "config.yaml"

_REST_BASES = {
    "spot": ("https://api.binance.com/api/v3",),
    "futures_um": tuple(f"https://fapi{i or ''}.binance.com/fapi/v1" for i in range(4)),
    "futures_cm": tuple(f"https://dapi{i or ''}.binance.com/dapi/v1" for i in range(4)),
}


def ingest_recent_completed(cfg: Config, store: Store, limit: int = 1_000) -> pd.Timestamp | None:
    """Merge a synchronized tail of completed public Binance bars into the store.

    Binance Vision remains the bulk source, but its daily archives can appear
    late. The public REST endpoint fills that publication gap without an API
    key. Every symbol is trimmed to the latest timestamp common to the entire
    universe, and still-open candles are rejected before anything is written.
    """
    market = cfg.get("data.market", "futures_um")
    if market not in _REST_BASES:
        raise ValueError(f"recent collection is unsupported for market {market!r}")
    bases = _REST_BASES[market]
    interval = cfg.get("data.interval", "1h")
    symbols = list(cfg.get("data.symbols"))
    now_ms = int(datetime.now(UTC).timestamp() * 1_000)
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}

    for symbol in symbols:
        response = None
        for base in bases:
            try:
                candidate = requests.get(
                    f"{base}/klines",
                    params={"symbol": symbol, "interval": interval, "limit": limit},
                    timeout=8,
                )
                candidate.raise_for_status()
                rows = candidate.json()
            except requests.RequestException:
                continue
            if not isinstance(rows, list):
                continue
            response = candidate
            sources[symbol] = base
            break
        if response is None:
            print("Binance REST is region-blocked; using completed Vision archives only.")
            return None
        frame = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )
        frame = frame[pd.to_numeric(frame["close_time"]) < now_ms].copy()
        if frame.empty:
            raise RuntimeError(f"Binance returned no completed bars for {symbol}")
        frame["ts"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.drop(columns=["open_time", "close_time", "ignore"])
        frame.insert(0, "symbol", symbol)
        frames[symbol] = frame.dropna(subset=["ts", "close"])

    common_last = min(frame["ts"].max() for frame in frames.values())
    for symbol, frame in frames.items():
        store.write(frame[frame["ts"] <= common_last], "klines", symbol, interval)

    if market.startswith("futures"):
        start_ms = int((common_last - pd.Timedelta(days=45)).timestamp() * 1_000)
        for symbol in symbols:
            base = sources[symbol]
            response = requests.get(
                f"{base}/fundingRate",
                params={"symbol": symbol, "startTime": start_ms, "limit": limit},
                timeout=8,
            )
            response.raise_for_status()
            funding = pd.DataFrame(response.json())
            if funding.empty:
                continue
            funding = pd.DataFrame(
                {
                    "symbol": symbol,
                    "ts": pd.to_datetime(funding["fundingTime"], unit="ms", utc=True),
                    "funding_rate": pd.to_numeric(funding["fundingRate"], errors="coerce"),
                }
            ).dropna()
            store.write(funding[funding["ts"] <= common_last], "funding", symbol)
    return common_last


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def initialise(cfg: Config, directory: Path, cutoff: pd.Timestamp) -> dict:
    """Freeze a forward test before any bar later than ``cutoff`` is evaluated."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"forward test already initialised: {manifest_path}")
    cutoff = pd.Timestamp(cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    manifest = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "cutoff": cutoff.isoformat(),
        "strategy": "AdaptiveMultiFactor",
        "strategy_sha256": _sha256(STRATEGY_FILE),
        "config_sha256": _sha256(CONFIG_FILE),
        "symbols": list(cfg.get("data.symbols")),
        "interval": cfg.get("data.interval"),
        "status": "frozen",
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    (directory / "ledger.jsonl").touch(exist_ok=False)
    return manifest


def verify(directory: Path) -> dict:
    """Verify the frozen files and the complete hash chain."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    stored = manifest.pop("manifest_sha256")
    if hashlib.sha256(_canonical(manifest)).hexdigest() != stored:
        raise RuntimeError("forward manifest was modified")
    manifest["manifest_sha256"] = stored
    if _sha256(STRATEGY_FILE) != manifest["strategy_sha256"]:
        raise RuntimeError("strategy changed after the forward cutoff")
    if _sha256(CONFIG_FILE) != manifest["config_sha256"]:
        raise RuntimeError("configuration changed after the forward cutoff")

    previous = manifest["manifest_sha256"]
    for line in (directory / "ledger.jsonl").read_text(encoding="utf8").splitlines():
        entry = json.loads(line)
        digest = entry.pop("entry_sha256")
        if (
            entry["previous_sha256"] != previous
            or hashlib.sha256(_canonical(entry)).hexdigest() != digest
        ):
            raise RuntimeError("forward ledger hash chain is invalid")
        previous = digest
    return manifest


def append_snapshot(cfg: Config, directory: Path) -> dict:
    """Evaluate only bars published after the frozen cutoff and append a snapshot."""
    manifest = verify(directory)
    feats, prices, funding = load_feature_data(cfg)
    result, weights = evaluate(AdaptiveMultiFactor(), feats, prices, funding, cfg)
    cutoff = pd.Timestamp(manifest["cutoff"])
    returns = result.returns[result.returns.index > cutoff]
    last_bar = prices.dropna(how="all").index.max()

    lines = (directory / "ledger.jsonl").read_text(encoding="utf8").splitlines()
    if lines:
        latest = json.loads(lines[-1])
        if pd.Timestamp(latest["last_bar"]) >= last_bar:
            return {**latest, "status": "no_new_completed_bar"}
    previous = manifest["manifest_sha256"] if not lines else json.loads(lines[-1])["entry_sha256"]
    entry = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "last_bar": last_bar.isoformat(),
        "cutoff": manifest["cutoff"],
        "n_forward_bars": len(returns),
        "cumulative_return": float((1 + returns).prod() - 1) if len(returns) else 0.0,
        "latest_weights": {k: float(v) for k, v in weights.loc[last_bar].items()},
        "previous_sha256": previous,
    }
    entry["entry_sha256"] = hashlib.sha256(_canonical(entry)).hexdigest()
    with (directory / "ledger.jsonl").open("a", encoding="utf8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return {**entry, "status": "appended"}


def main() -> int:
    """Initialize or update a frozen forward ledger from the command line."""
    parser = argparse.ArgumentParser(prog="python -m cryptoquant.forward")
    parser.add_argument("action", choices=("init", "update", "verify"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--directory", default="forward_evidence")
    args = parser.parse_args()
    cfg = load_config(args.config)
    directory = Path(args.directory)
    if args.action == "init":
        _, prices, _ = load_feature_data(cfg)
        value = initialise(cfg, directory, prices.dropna(how="all").index.max())
    elif args.action == "update":
        store = ingest(cfg, with_funding=True, with_metrics=False)
        ingest_recent_completed(cfg, store)
        value = append_snapshot(cfg, directory)
    else:
        value = verify(directory)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
