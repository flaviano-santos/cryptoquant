"""Tests for the tamper-evident forward ledger."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cryptoquant.config import Config
from cryptoquant.forward import ingest_recent_completed, initialise, verify


@pytest.fixture
def forward_dir(tmp_path):
    cfg = Config(
        raw={"data": {"symbols": ["BTCUSDT"], "interval": "1h"}},
        path=tmp_path / "config.yaml",
    )
    directory = tmp_path / "evidence"
    initialise(cfg, directory, pd.Timestamp("2026-08-12 23:00", tz="UTC"))
    return directory


def test_initial_manifest_verifies(forward_dir):
    assert verify(forward_dir)["status"] == "frozen"


def test_manifest_tampering_is_detected(forward_dir):
    path = forward_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf8"))
    manifest["cutoff"] = "2020-01-01T00:00:00+00:00"
    path.write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(RuntimeError, match="manifest was modified"):
        verify(forward_dir)


def test_ledger_tampering_is_detected(forward_dir):
    manifest = verify(forward_dir)
    entry = {
        "previous_sha256": manifest["manifest_sha256"],
        "entry_sha256": "not-a-real-digest",
    }
    (forward_dir / "ledger.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf8")
    with pytest.raises(RuntimeError, match="hash chain"):
        verify(forward_dir)


def test_recent_collection_rejects_open_bars_and_synchronizes(monkeypatch, tmp_path):
    cfg = Config(
        raw={
            "data": {
                "root": str(tmp_path / "data"),
                "market": "spot",
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "interval": "1h",
            }
        },
        path=tmp_path / "config.yaml",
    )

    class Response:
        def __init__(self, rows):
            self._rows = rows
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._rows

    def row(open_time, close):
        return [open_time, close, close, close, close, 1, open_time + 3_599_999, 1, 1, 1, 1, 0]

    completed = 1_700_000_000_000
    open_bar = 9_000_000_000_000
    payloads = {
        "BTCUSDT": [row(completed, 100), row(completed + 3_600_000, 101), row(open_bar, 999)],
        "ETHUSDT": [row(completed, 50), row(open_bar, 999)],
    }

    def fake_get(_url, params, timeout):
        assert timeout == 8
        return Response(payloads[params["symbol"]])

    monkeypatch.setattr("cryptoquant.forward.requests.get", fake_get)
    from cryptoquant.data.store import Store

    store = Store(cfg.data_root)
    common_last = ingest_recent_completed(cfg, store)
    assert common_last == pd.Timestamp(completed, unit="ms", tz="UTC")
    assert len(store.read("klines", "BTCUSDT", "1h")) == 1
    assert len(store.read("klines", "ETHUSDT", "1h")) == 1
