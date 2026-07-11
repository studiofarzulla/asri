#!/usr/bin/env python3
"""
Daily/weekly ASRI dashboard refresh -- keeps the live D1 series current.

Idempotent end-to-end refresh:
  1. Ask live D1 for max(date). If already >= --end (default: yesterday UTC),
     exit 0 without touching anything.
  2. Extend data/peg_history.csv to --end (scripts/extend_peg_history.py,
     additive, no-op when current).
  3. Run the committed generation pipeline (scripts/generate_asri_series.py,
     ASRIBacktester) for exactly the missing window, under the pinned frozen
     universe snapshot (PINNED_UNIVERSE below).
  4. Load the new rows additively into D1 (scripts/load_d1_backfill.py,
     INSERT OR IGNORE -- frozen rows can never be overwritten).
  5. Append the loaded rows to results/data/asri_live_appends.csv (audit trail)
     and verify https://api.dissensus.ai/asri/current serves the new date.

Run it from the repo root with the pinned venv:
    .venv/bin/python scripts/refresh_asri.py

Provenance note: rows after 2026-01-15 are produced by this (Jun-2026 fixed)
pipeline; rows up to 2026-01-15 are the frozen canonical paper series. There is
a documented level discontinuity between the two -- see RUNBOOK_DASHBOARD.md.
Do not "fix" the seam by rescaling either side.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python"
TMP_PARQUET = ROOT / "results" / "data" / "asri_refresh_tmp.parquet"
AUDIT_CSV = ROOT / "results" / "data" / "asri_live_appends.csv"

# Frozen protocols/bridges universe used for all live-refresh rows. Bridges
# universe is pinned permanently (bridges.llama.fi went behind the paid API in
# 2026; the snapshot holds its last public output). Re-dump protocols with
# scripts/dump_universe_snapshot.py only as a deliberate, documented decision --
# it shifts the DeFi/Contagion/Arbitrage channel levels.
PINNED_UNIVERSE = "2026-07-11"

sys.path.insert(0, str(ROOT / "scripts"))
from load_d1_backfill import D1, load_api_key  # noqa: E402


def run(cmd: list[str], env: dict | None = None) -> None:
    import os
    full_env = {**os.environ, **(env or {})}
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, env=full_env)


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh the live ASRI D1 series")
    ap.add_argument("--end", default=None,
                    help="last date to compute, YYYY-MM-DD (default: yesterday UTC)")
    args = ap.parse_args()
    end = args.end or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    d1 = D1(load_api_key())
    latest = d1.query("SELECT MAX(date) AS latest FROM asri_daily")["result"][0]["results"][0]["latest"]
    if latest >= end:
        print(f"Up to date: D1 latest {latest} >= requested end {end}. Nothing to do.")
        return 0
    start = (datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Refreshing {start} .. {end} (D1 latest: {latest})")

    run([PYTHON, "scripts/extend_peg_history.py", "--end", end])
    run([PYTHON, "scripts/generate_asri_series.py",
         "--start", start, "--end", end, "--out", str(TMP_PARQUET)],
        env={"ASRI_SNAPSHOT_AS_OF": PINNED_UNIVERSE})
    run([PYTHON, "scripts/load_d1_backfill.py", str(TMP_PARQUET)])
    # Same rows also extend the alternate single-methodology series (identical
    # pipeline for new dates; the loader's own max(date) guard keeps it safe).
    run([PYTHON, "scripts/load_d1_backfill.py", str(TMP_PARQUET),
         "--table", "asri_daily_open", "--profile", "open_pipeline_full"])

    # Audit trail of every live-appended row.
    df = pd.read_parquet(TMP_PARQUET).sort_index()
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    new_file = not AUDIT_CSV.exists()
    with open(AUDIT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "asri", "stablecoin_risk", "defi_liquidity_risk",
                        "contagion_risk", "arbitrage_opacity", "loaded_at_utc"])
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for date, r in df.iterrows():
            w.writerow([date.strftime("%Y-%m-%d"),
                        round(float(r["asri"]), 2),
                        round(float(r["stablecoin_risk"]), 2),
                        round(float(r["defi_liquidity_risk"]), 2),
                        round(float(r["contagion_risk"]), 2),
                        round(float(r["arbitrage_opacity"]), 2), now])
    TMP_PARQUET.unlink(missing_ok=True)

    # Live verification.
    resp = httpx.get("https://api.dissensus.ai/asri/current", timeout=30.0)
    resp.raise_for_status()
    served = resp.json().get("last_update")
    if served != end:
        print(f"VERIFY FAILED: API serves last_update={served}, expected {end}")
        return 1
    print(f"OK: api.dissensus.ai serves last_update={served}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
