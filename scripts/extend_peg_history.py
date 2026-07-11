#!/usr/bin/env python3
"""
Additively extend data/peg_history.csv to the present.

``build_peg_history.py`` rebuilds the whole 2021->2026-01-15 dataset from
scratch and overwrites the CSV; that file is a committed input to the frozen
canonical series and must not be regenerated wholesale. This script instead
pulls ONLY dates after the current CSV maximum (same DeFiLlama coins source,
same daily-median collapse), computes the 7d/30d rolling peg volatilities on
the stitched per-coin series so the windows carry over the boundary, and
appends the new rows. Existing rows are never touched.

New rows get price_low = price (low_source=defillama_daily) exactly like the
un-enriched builder output; intraday-low enrichment only ever covered
2022-2023 (scripts/enrich_peg_intraday.py).

Usage:
    python scripts/extend_peg_history.py            # extend to yesterday (UTC)
    python scripts/extend_peg_history.py --end 2026-07-10
    python scripts/extend_peg_history.py --dry-run
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV = PROJECT_ROOT / "data" / "peg_history.csv"

DEFILLAMA_COINS = "https://coins.llama.fi"

# Same symbol -> CoinGecko id map as build_peg_history.py
COINS: dict[str, str] = {
    "USDT": "tether",
    "USDC": "usd-coin",
    "DAI": "dai",
    "BUSD": "binance-usd",
    "UST": "terrausd",
    "FRAX": "frax",
    "TUSD": "true-usd",
    "USDP": "paxos-standard",
}


def pull_window(client: httpx.Client, cgid: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Pull daily prices for one coin over [start, end], collapsed to one row/UTC date."""
    points: dict[int, float] = {}
    span = (end - start).days + 2
    params = {"start": int(start.timestamp()), "span": span, "period": "1d"}
    url = f"{DEFILLAMA_COINS}/chart/coingecko:{cgid}"
    for attempt in range(4):
        try:
            r = client.get(url, params=params, timeout=40.0)
            if r.status_code != 200:
                break
            series = r.json().get("coins", {}).get(f"coingecko:{cgid}", {}).get("prices", [])
            for p in series:
                ts = int(p["timestamp"])
                if int(start.timestamp()) <= ts <= int(end.timestamp()) + 86400:
                    points[ts] = float(p["price"])
            break
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    if not points:
        return pd.DataFrame(columns=["date", "price"])
    recs = [(datetime.fromtimestamp(ts, tz=timezone.utc).date(), px) for ts, px in sorted(points.items())]
    df = pd.DataFrame(recs, columns=["date", "price"])
    return df.groupby("date", as_index=False)["price"].median()


def main() -> None:
    ap = argparse.ArgumentParser(description="Additively extend data/peg_history.csv")
    ap.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                    default=(datetime.now(timezone.utc) - timedelta(days=1)).date(),
                    help="last date to add (default: yesterday UTC)")
    ap.add_argument("--dry-run", action="store_true", help="report what would be appended, write nothing")
    args = ap.parse_args()

    existing = pd.read_csv(CSV, parse_dates=["date"])
    last = existing["date"].max().date()
    if args.end <= last:
        print(f"Nothing to do: CSV already covers through {last} (requested end {args.end}).")
        return
    first_new = last + timedelta(days=1)
    print(f"Extending {CSV.name}: {first_new} .. {args.end} "
          f"(existing max {last}, {len(existing)} rows)")

    # Pull with a 40-day lead so 30d rolling windows have context, but only
    # rows strictly after `last` are appended.
    pull_start = datetime.combine(last - timedelta(days=40), datetime.min.time(), tzinfo=timezone.utc)
    pull_end = datetime.combine(args.end, datetime.min.time(), tzinfo=timezone.utc)

    frames = []
    with httpx.Client() as client:
        for symbol, cgid in COINS.items():
            df = pull_window(client, cgid, pull_start, pull_end)
            df["symbol"] = symbol
            n_new = (df["date"] > last).sum()
            print(f"  {symbol:5s} ({cgid:16s}): {len(df):3d} pulled, {n_new:3d} new")
            frames.append(df)
            time.sleep(0.25)

    pulled = pd.concat(frames, ignore_index=True)
    pulled["date"] = pd.to_datetime(pulled["date"])

    new_rows = []
    for symbol, grp in pulled.groupby("symbol"):
        # Stitch: existing tail (authoritative for overlap dates) + newly pulled dates.
        old = existing[existing["symbol"] == symbol][["date", "price"]]
        add = grp[grp["date"] > existing["date"].max()][["date", "price"]]
        combined = pd.concat([old, add], ignore_index=True).sort_values("date").reset_index(drop=True)
        combined["peg_volatility_7d"] = combined["price"].rolling(7, min_periods=3).std()
        combined["peg_volatility_30d"] = combined["price"].rolling(30, min_periods=7).std()
        out = combined[combined["date"] > existing["date"].max()].copy()
        out["symbol"] = symbol
        new_rows.append(out)

    new = pd.concat(new_rows, ignore_index=True)
    if new.empty:
        print("No new rows pulled (source not yet updated for the requested window).")
        return
    new["peg_deviation"] = (1.0 - new["price"]).abs()
    new["price_low"] = new["price"]
    new["peg_deviation_low"] = new["peg_deviation"]
    new["low_source"] = "defillama_daily"
    new["source"] = "defillama_coins_coingecko_daily"
    cols = ["date", "symbol", "price", "peg_deviation",
            "peg_volatility_7d", "peg_volatility_30d",
            "price_low", "peg_deviation_low", "low_source", "source"]
    new = new[cols].sort_values(["date", "symbol"]).reset_index(drop=True)
    new["date"] = new["date"].dt.strftime("%Y-%m-%d")

    print(f"\nAppending {len(new)} rows ({new['date'].min()} .. {new['date'].max()})")
    if args.dry_run:
        print(new.head(10).to_string(index=False))
        print("[dry-run] wrote nothing")
        return
    new.to_csv(CSV, mode="a", header=False, index=False)
    total = sum(1 for _ in open(CSV)) - 1
    print(f"Done: {CSV} now has {total} data rows.")


if __name__ == "__main__":
    main()
