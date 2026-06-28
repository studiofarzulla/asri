#!/usr/bin/env python3
"""
Build historical daily stablecoin peg/price dataset for the ASRI backtest.

Why this exists
---------------
The ASRI backtester (``src/asri/backtest/backtest.py``) hardcoded every
stablecoin at par:

    price=1.0, peg_deviation=0.0          # _snapshot_to_inputs
    peg_volatility=10.0                   # StablecoinRiskInputs

This blinded the Stablecoin Concentration Risk (SCR) sub-index to every
historical depeg (UST May 2022; USDC/DAI SVB March 2023). This script
acquires REAL daily price data so the backtest can be re-run with a live
``peg_volatility`` channel.

How SCR consumes the data (see src/asri/pipeline/transform.py
``transform_stablecoin_risk``):

    peg_deviation_i = |1 - price_i|
    weighted_deviation = sum(peg_deviation_i * circulating_i) / sum(circulating_i)
    peg_volatility = normalize_to_100(weighted_deviation * 100, 0, 5)   # 0%->0, 5%+->100
    # peg_volatility enters SCR with weight 0.1 (STABLECOIN_WEIGHTS).

Source
------
Primary daily price series: DeFiLlama "coins" API
(https://coins.llama.fi/chart/coingecko:<id>) which proxies CoinGecko
prices. Keyless, reproducible, and the same provider the backtest already
uses for BTC. Daily aggregates SMOOTH intraday troughs (e.g. USDC SVB shows
~0.96 daily vs ~0.88 intraday); the intraday lows are added separately from
CoinGecko OHLC by ``scripts/enrich_peg_intraday.py`` into the ``price_low``
column. Validation magnitudes are printed by the builder / enrich scripts.

Output: data/peg_history.csv  (long format, one row per date x symbol)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "data" / "peg_history.csv"

DEFILLAMA_COINS = "https://coins.llama.fi"

# symbol (as used by backtest MAJOR_STABLES) -> CoinGecko id (DeFiLlama key)
# Core five required + three the backtest also tracks.
COINS: dict[str, str] = {
    "USDT": "tether",
    "USDC": "usd-coin",
    "DAI": "dai",
    "BUSD": "binance-usd",
    "UST": "terrausd",       # native UST pre-collapse, TerraClassicUSD (USTC) after
    "FRAX": "frax",
    "TUSD": "true-usd",
    "USDP": "paxos-standard",
}

START = datetime(2021, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 1, 16, tzinfo=timezone.utc)   # match asri_history.parquet end


def pull_coin(client: httpx.Client, cgid: str) -> list[tuple[int, float]]:
    """Pull daily prices for one coin across the full window in yearly chunks."""
    points: dict[int, float] = {}
    year_starts = []
    y = START
    while y < END:
        year_starts.append(y)
        y = datetime(y.year + 1, 1, 1, tzinfo=timezone.utc)
    for ys in year_starts:
        start_ts = int(ys.timestamp())
        url = f"{DEFILLAMA_COINS}/chart/coingecko:{cgid}"
        params = {"start": start_ts, "span": 370, "period": "1d"}
        for attempt in range(4):
            try:
                r = client.get(url, params=params, timeout=40.0)
                if r.status_code != 200:
                    break
                j = r.json()
                series = j.get("coins", {}).get(f"coingecko:{cgid}", {}).get("prices", [])
                for p in series:
                    ts = int(p["timestamp"])
                    if start_ts <= ts <= int(END.timestamp()):
                        points[ts] = float(p["price"])
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        time.sleep(0.25)
    return sorted(points.items())


def to_daily(symbol: str, raw: list[tuple[int, float]]) -> pd.DataFrame:
    """Collapse raw (ts, price) to one row per UTC date (median of intra-date points)."""
    if not raw:
        return pd.DataFrame(columns=["date", "symbol", "price"])
    recs = []
    for ts, px in raw:
        d = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        recs.append((d, px))
    df = pd.DataFrame(recs, columns=["date", "price"])
    df = df.groupby("date", as_index=False)["price"].median()
    df["symbol"] = symbol
    return df[["date", "symbol", "price"]]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    coverage = {}
    with httpx.Client() as client:
        for symbol, cgid in COINS.items():
            raw = pull_coin(client, cgid)
            df = to_daily(symbol, raw)
            coverage[symbol] = (len(df), (df["date"].min() if len(df) else None),
                                (df["date"].max() if len(df) else None))
            print(f"{symbol:5s} ({cgid:16s}): {len(df):5d} daily rows "
                  f"{coverage[symbol][1]} -> {coverage[symbol][2]}")
            if len(df):
                frames.append(df)

    alldf = pd.concat(frames, ignore_index=True)
    alldf["date"] = pd.to_datetime(alldf["date"])
    alldf = alldf.sort_values(["symbol", "date"]).reset_index(drop=True)

    # peg deviation
    alldf["peg_deviation"] = (1.0 - alldf["price"]).abs()

    # rolling peg volatility (std of daily price), per coin, on the observed series
    def roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        g["peg_volatility_7d"] = g["price"].rolling(7, min_periods=3).std()
        g["peg_volatility_30d"] = g["price"].rolling(30, min_periods=7).std()
        return g

    alldf = alldf.groupby("symbol", group_keys=False).apply(roll)

    # price_low enriched later from CoinGecko OHLC; default = daily price
    alldf["price_low"] = alldf["price"]
    alldf["peg_deviation_low"] = alldf["peg_deviation"]
    alldf["low_source"] = "defillama_daily"

    alldf["source"] = "defillama_coins_coingecko_daily"
    cols = ["date", "symbol", "price", "peg_deviation",
            "peg_volatility_7d", "peg_volatility_30d",
            "price_low", "peg_deviation_low",
            "low_source", "source"]
    alldf = alldf[cols].sort_values(["date", "symbol"]).reset_index(drop=True)
    alldf.to_csv(OUT, index=False)
    print(f"\nWrote {len(alldf)} rows -> {OUT}")
    print("\nCoverage summary:")
    for s, (n, lo, hi) in coverage.items():
        print(f"  {s:5s}: {n:5d} rows  {lo} .. {hi}")


if __name__ == "__main__":
    main()
