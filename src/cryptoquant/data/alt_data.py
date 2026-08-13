"""Free alternative data sources.

None of these require an API key or a paid plan. They are slow-moving series and
will not drive a high-turnover strategy on their own, but they are genuinely
orthogonal to price-derived features, which is exactly what a factor set usually
lacks.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

__all__ = ["defillama_tvl", "fear_greed", "stablecoin_supply"]


def fear_greed(limit: int = 0) -> pd.DataFrame:
    """Crypto Fear & Greed index, daily, back to 2018. alternative.me, free."""
    r = requests.get(
        "https://api.alternative.me/fng/",
        params={"limit": str(limit or 0), "format": "json"},
        timeout=30,
    )
    r.raise_for_status()
    d = pd.DataFrame(r.json()["data"])
    d["ts"] = pd.to_datetime(pd.to_numeric(d["timestamp"]), unit="s", utc=True)
    d["fng"] = pd.to_numeric(d["value"])
    return d[["ts", "fng"]].sort_values("ts").reset_index(drop=True)


def defillama_tvl(protocol: str | None = None) -> pd.DataFrame:
    """Total value locked. Whole-market if protocol is None, else a single protocol. DefiLlama's API is free and unauthenticated."""
    url = (
        "https://api.llama.fi/v2/historicalChainTvl"
        if protocol is None
        else f"https://api.llama.fi/protocol/{protocol}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    rows = js if isinstance(js, list) else js.get("tvl", [])
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["ts"] = pd.to_datetime(d["date"], unit="s", utc=True)
    d = d.rename(columns={"totalLiquidityUSD": "tvl"})
    keep = [c for c in ("ts", "tvl") if c in d]
    return d[keep].sort_values("ts").reset_index(drop=True)


def stablecoin_supply() -> pd.DataFrame:
    """Fetch the aggregate stablecoin market capitalisation over time.

    A slow but genuine macro signal: expansion in stablecoin supply is dry
    powder entering the system, and it leads risk appetite rather than
    following it.

    Returns:
        A frame with ``ts`` and ``stable_mcap`` columns, or an empty frame if
        the API returned nothing.

    Raises:
        requests.HTTPError: If the request fails.
    """
    r = requests.get("https://stablecoins.llama.fi/stablecoincharts/all", timeout=30)
    r.raise_for_status()
    d = pd.DataFrame(r.json())
    if d.empty:
        return d
    d["ts"] = pd.to_datetime(pd.to_numeric(d["date"]), unit="s", utc=True)
    d["stable_mcap"] = d["totalCirculatingUSD"].apply(
        lambda x: sum(x.values()) if isinstance(x, dict) else np.nan
    )
    return d[["ts", "stable_mcap"]].sort_values("ts").reset_index(drop=True)
