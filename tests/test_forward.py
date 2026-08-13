"""Tests for the tamper-evident forward ledger."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cryptoquant.config import Config
from cryptoquant.forward import initialise, verify


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
