#!/usr/bin/env python3
"""One-time dumper: freeze the CURRENT DeFiLlama protocols + bridges universe to
``data/snapshots/{protocols,bridges}_<as_of>.json``.

Why: the ASRI generation pipeline (src/asri/backtest/historical.py) projects a
single protocols/bridges universe onto every historical date (documented
look-ahead). Reading that universe from a frozen on-disk snapshot -- instead of
re-pulling the live, drifting universe on every run -- makes a full regen
DETERMINISTIC and auditable (Bug 2 fix, Jun 2026). With a snapshot present,
``HistoricalDataFetcher(as_of="<date>")`` (or env ``ASRI_SNAPSHOT_AS_OF``) reads
it; without one, the fetcher falls back to a LIVE pull and loudly flags the
non-determinism in ``data_quality``.

HONESTY NOTE -- this does NOT recover the published series. The ORIGINAL
(~Jan-2026) generation-time universe was never snapshotted, so freezing now pins
a CURRENT (post-publication) universe. That guarantees reproducibility of
*future* regens only; it does not bit-reproduce the DeFi/Contagion/Arbitrage
channels of the frozen canonical parquet (results/data/asri_history.parquet),
which remains the dataset of record. See DATA_PROVENANCE.md / REPRODUCIBILITY.md.

Usage:
    python scripts/dump_universe_snapshot.py            # as_of = today
    python scripts/dump_universe_snapshot.py 2026-06-28 # explicit as-of date
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
SNAP_DIR = ROOT / "data" / "snapshots"

DEFILLAMA_PROTOCOLS = "https://api.llama.fi/protocols"
DEFILLAMA_BRIDGES = "https://bridges.llama.fi/bridges"


async def main() -> int:
    as_of = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    SNAP_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=60.0) as client:
        pr = await client.get(DEFILLAMA_PROTOCOLS)
        protocols = pr.json() if pr.status_code == 200 else []
        br = await client.get(DEFILLAMA_BRIDGES)
        bridges = br.json().get("bridges", []) if br.status_code == 200 else []

    if not protocols and not bridges:
        print("ERROR: both pulls returned empty (network blocker?). Nothing written.")
        return 2

    p_path = SNAP_DIR / f"protocols_{as_of}.json"
    b_path = SNAP_DIR / f"bridges_{as_of}.json"
    p_path.write_text(json.dumps(protocols))
    b_path.write_text(json.dumps(bridges))

    print(f"Wrote {len(protocols)} protocols -> {p_path}")
    print(f"Wrote {len(bridges)} bridges  -> {b_path}")
    print(
        f"\nUse with: HistoricalDataFetcher(as_of='{as_of}')  "
        f"or  ASRI_SNAPSHOT_AS_OF={as_of}\n"
        "NOTE: this is a CURRENT universe (post-publication); it does not "
        "reproduce the frozen canonical series."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
