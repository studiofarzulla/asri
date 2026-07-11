#!/usr/bin/env python3
"""
Load a generated ASRI parquet (scripts/generate_asri_series.py output) into the
live Cloudflare D1 database behind api.dissensus.ai -- ADDITIVELY.

Guarantees:
  * INSERT OR IGNORE only -- existing rows (the frozen canonical series,
    2021-01-01..2026-01-15) are never modified or overwritten.
  * Only rows strictly AFTER the current D1 max(date) are considered.
  * Stored values follow the reconciled-equation convention already in D1
    (scripts/reconcile_d1_equation.py): sub-indices rounded to 1 dp, asri =
    round(weighted sum of the rounded subs, 1), alert level from that asri.
    The worker recomputes asri from subs at read time, so stored asri is
    display-consistent by construction.
  * asri_30d_avg / trend are derived from the trailing 30 stored asri values
    (fetched from D1), matching the worker's /asri/current semantics
    (trend: rising if asri - avg > 2, falling if < -2, else stable).

Credentials: ~/.env.cloudflare must export CLOUDFLARE_API_KEY (global key,
used with X-Auth-Email like scripts/backfill_d1_standalone.py).

Usage:
    python scripts/load_d1_backfill.py results/data/asri_backfill_2026H1.parquet --dry-run
    python scripts/load_d1_backfill.py results/data/asri_backfill_2026H1.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import pandas as pd

CF_EMAIL = "contact@farzulla.com"
CF_ACCOUNT_ID = "917877c07fa2e1c9f223db31d3fc52d6"
D1_DATABASE_ID = "f8c08c15-a596-4d7f-8c00-0f7592da26f4"

WEIGHTS = {
    "stablecoin_risk": 0.30,
    "defi_liquidity_risk": 0.25,
    "contagion_risk": 0.25,
    "arbitrage_opacity": 0.20,
}
SUBS = list(WEIGHTS)
BATCH = 50


def load_api_key() -> str:
    for line in open(Path.home() / ".env.cloudflare"):
        if line.startswith("export CLOUDFLARE_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("CLOUDFLARE_API_KEY not found in ~/.env.cloudflare")


def alert_level(asri: float) -> str:
    if asri < 30:
        return "low"
    if asri < 50:
        return "moderate"
    if asri < 70:
        return "elevated"
    return "critical"


def trend(asri: float, avg: float) -> str:
    d = asri - avg
    return "rising" if d > 2 else ("falling" if d < -2 else "stable")


class D1:
    def __init__(self, key: str):
        self.url = (f"https://api.cloudflare.com/client/v4/accounts/"
                    f"{CF_ACCOUNT_ID}/d1/database/{D1_DATABASE_ID}/query")
        self.headers = {"X-Auth-Email": CF_EMAIL, "X-Auth-Key": key,
                        "Content-Type": "application/json"}
        self.client = httpx.Client(timeout=60.0)

    def query(self, sql: str, params: list | None = None) -> dict:
        payload: dict = {"sql": sql}
        if params:
            payload["params"] = params
        r = self.client.post(self.url, headers=self.headers, json=payload)
        r.raise_for_status()
        j = r.json()
        if not j.get("success"):
            raise RuntimeError(f"D1 error: {j.get('errors')}")
        return j


def main() -> None:
    ap = argparse.ArgumentParser(description="Additively load ASRI parquet into live D1")
    ap.add_argument("parquet", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--table", default="asri_daily",
                    help="target D1 table (default asri_daily; asri_daily_open for the full-recompute series)")
    ap.add_argument("--profile", default="open_pipeline_continuation",
                    help="methodology_profile written on inserted rows")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet).sort_index()
    d1 = D1(load_api_key())

    state = d1.query(f"SELECT MAX(date) AS latest, COUNT(*) AS n FROM {args.table}")
    latest, n = (state["result"][0]["results"][0][k] for k in ("latest", "n"))
    print(f"D1 {args.table} before: {n} rows, latest {latest}")

    # Empty table -> load the whole parquet; otherwise append-only past max(date).
    new = df if latest is None else df[df.index > pd.Timestamp(latest)]
    if new.empty:
        print("Nothing to load: parquet has no rows after D1 latest.")
        return
    print(f"Loading {len(new)} rows: {new.index.min().date()} .. {new.index.max().date()}")

    # Trailing stored-asri context for the 30d average.
    window: list[float] = []
    if latest is not None:
        tail = d1.query(f"SELECT asri FROM {args.table} ORDER BY date DESC LIMIT 29")
        window = [r["asri"] for r in tail["result"][0]["results"]][::-1]

    rows = []
    for date, r in new.iterrows():
        subs = {k: round(float(r[k]), 1) for k in SUBS}
        asri = round(sum(subs[k] * w for k, w in WEIGHTS.items()), 1)
        window.append(asri)
        window[:] = window[-30:]
        avg = round(sum(window) / len(window), 1)
        rows.append((date.strftime("%Y-%m-%d"), asri, avg, trend(asri, avg),
                     alert_level(asri), subs["stablecoin_risk"],
                     subs["defi_liquidity_risk"], subs["contagion_risk"],
                     subs["arbitrage_opacity"]))

    print(f"First: {rows[0]}")
    print(f"Last:  {rows[-1]}")
    if args.dry_run:
        print("[dry-run] no writes")
        return

    inserted = 0
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        values = ",".join(
            f"('{d}',{a},{av},'{t}','{al}',{s},{de},{c},{ar},'{args.profile}')"
            for d, a, av, t, al, s, de, c, ar in chunk
        )
        sql = (f"INSERT OR IGNORE INTO {args.table} (date, asri, asri_30d_avg, trend, "
               "alert_level, stablecoin_risk, defi_liquidity_risk, contagion_risk, "
               "arbitrage_opacity, methodology_profile) VALUES " + values)
        res = d1.query(sql)
        changes = res["result"][0]["meta"].get("changes", 0)
        inserted += changes
        print(f"  batch {i // BATCH + 1}: {changes} inserted")

    state = d1.query(f"SELECT MAX(date) AS latest, COUNT(*) AS n FROM {args.table}")
    latest, n = (state["result"][0]["results"][0][k] for k in ("latest", "n"))
    print(f"D1 {args.table} after: {n} rows, latest {latest} ({inserted} inserted)")


if __name__ == "__main__":
    main()
