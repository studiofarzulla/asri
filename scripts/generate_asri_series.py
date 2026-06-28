#!/usr/bin/env python3
"""
Generate a code-consistent ASRI history series (DB-free).

This is the canonical *generator* for ``results/data/asri_history.parquet``:
it runs the real backtest pipeline -- ``ASRIBacktester.calculate_for_date`` ->
``HistoricalDataFetcher.fetch_snapshot`` + ``_snapshot_to_inputs`` +
``signals.calculator`` -- over a date range and writes the resulting daily
sub-indices + aggregate ASRI to parquet. Unlike ``backfill.py`` it never touches
Postgres (it does not call ``backfill_database``), so it runs anywhere the
scientific stack is installed under a pinned interpreter (py3.11-3.13).

It exercises the June-2026 construction fixes that are now live in the pipeline:
  * D2  -- corrected DeFiLlama stablecoin ids in HistoricalDataFetcher.MAJOR_STABLES
           (id=3 is USTC/Terra, mapped to "UST"; id=11 is USDP, not Terra UST).
  * peg -- real supply-weighted peg_volatility from data/peg_history.csv, wired
           into ASRIBacktester._snapshot_to_inputs (replaces the flat 10.0).
  * D5  -- trailing 365-day rolling-max TVL normalisation (replaces expanding max).

Data sources are LIVE (DeFiLlama TVL/stablecoins, FRED DGS10/VIXCLS/T10Y2Y/SP500,
DeFiLlama coins for BTC). FRED requires FRED_API_KEY in ./.env (loaded by
src/asri/config.py); run this script from the repo root (the code/ dir).

Determinism (Bug 2 fix, Jun 2026): protocol/bridge lists in fetch_snapshot are a
*current* universe projected onto historical dates (documented look-ahead). The
fetcher now reads that universe from a frozen on-disk snapshot
(data/snapshots/<name>_<as_of>.json -- create with scripts/dump_universe_snapshot.py)
when present, so a regen is deterministic; absent a snapshot it pulls LIVE and
loudly flags non-determinism. The SCR (stablecoin) channel -- where the D2/peg
fixes act -- is point-in-time and re-derivable.

THE CANONICAL SERIES IS FROZEN, NOT REGENERATED. The published, paper-of-record
ASRI daily series (max 84.70 on 2022-11-08, mean 39.20) is the frozen parquet at
results/data/asri_history.parquet (Zenodo 10.5281/zenodo.17918239); it is NOT
bit-reproducible from this pipeline (original live inputs + universe snapshot
were never archived). This generator produces a CODE-CONSISTENT series for
auditing the fixes -- it does not, and must not, overwrite the frozen canonical
file. Output therefore always goes to a NON-canonical path (--overwrite-canonical
is refused; see DATA_PROVENANCE.md / REPRODUCIBILITY.md).

Usage:
    # smoke run -- a few crisis dates, prints + writes a small non-canonical parquet
    python scripts/generate_asri_series.py --smoke

    # explicit window (deterministic if a frozen snapshot exists)
    ASRI_SNAPSHOT_AS_OF=2026-06-28 \
    python scripts/generate_asri_series.py --start 2022-05-01 --end 2022-05-20 \
        --out results/data/asri_smoke.parquet

    # full code-consistent regen of the canonical span (slow; non-canonical out)
    python scripts/generate_asri_series.py --start 2021-01-01 --end 2026-01-15 \
        --out results/data/asri_regen_full.parquet
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from asri.backtest.backtest import ASRIBacktester  # noqa: E402

CANON = ROOT / "results" / "data" / "asri_history.parquet"
COLS = ["asri", "stablecoin_risk", "defi_liquidity_risk", "contagion_risk", "arbitrage_opacity"]


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


async def generate(dates: list[datetime]) -> pd.DataFrame:
    bt = ASRIBacktester()
    records: list[dict] = []
    try:
        for i, d in enumerate(dates, 1):
            try:
                r = await bt.calculate_for_date(d)
                records.append({
                    "date": pd.Timestamp(d.date()),
                    "asri": r["asri"],
                    "stablecoin_risk": r["stablecoin_risk"],
                    "defi_liquidity_risk": r["defi_liquidity_risk"],
                    "contagion_risk": r["contagion_risk"],
                    "arbitrage_opacity": r["arbitrage_opacity"],
                    "alert_level": r["alert_level"],
                })
                print(f"[{i}/{len(dates)}] {d.date()}  ASRI={r['asri']:.2f} "
                      f"({r['alert_level']})  SCR={r['stablecoin_risk']:.2f} "
                      f"DeFi={r['defi_liquidity_risk']:.2f} "
                      f"Cont={r['contagion_risk']:.2f} "
                      f"Arb={r['arbitrage_opacity']:.2f}")
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(dates)}] {d.date()}  ERROR: {e}")
    finally:
        await bt.close()
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.set_index("date").sort_index()
    return df


def daterange(start: datetime, end: datetime, step: int = 1) -> list[datetime]:
    out, cur = [], start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=step)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a code-consistent ASRI series (DB-free)")
    p.add_argument("--start", type=parse_date)
    p.add_argument("--end", type=parse_date)
    p.add_argument("--step", type=int, default=1, help="days between snapshots")
    p.add_argument("--smoke", action="store_true",
                   help="run a handful of crisis dates as a connectivity/fix smoke test")
    p.add_argument("--out", type=str, default=None,
                   help="output parquet path (default: results/data/asri_generated.parquet)")
    p.add_argument("--overwrite-canonical", action="store_true",
                   help="REFUSED: the canonical parquet is a frozen dataset of record "
                        "(see DATA_PROVENANCE.md). Output is redirected to a "
                        "non-canonical path instead.")
    args = p.parse_args()

    if args.smoke:
        # One pre-crisis baseline + the four crisis peaks + the UST depeg trough.
        dates = [
            datetime(2022, 1, 15),   # calm baseline
            datetime(2022, 5, 9),    # Terra/UST depeg onset
            datetime(2022, 5, 12),   # UST trough (~$0.06)
            datetime(2022, 6, 17),   # 3AC / Celsius
            datetime(2022, 11, 8),   # FTX (published ASRI max ~84.7)
            datetime(2023, 3, 11),   # SVB / USDC depeg
        ]
    elif args.start and args.end:
        dates = daterange(args.start, args.end, args.step)
    else:
        p.print_help()
        return 1

    print(f"Generating ASRI for {len(dates)} date(s) via the fixed backtest pipeline "
          f"(D2 coin-ids + real peg + D5 rolling-365 TVL)\n")
    df = asyncio.run(generate(dates))

    if df.empty:
        print("\nNo rows produced (all dates errored -- likely a live-data/network blocker).")
        return 2

    if args.overwrite_canonical:
        # HARD REFUSAL. The canonical parquet is the frozen, published dataset of
        # record (max 84.70; Zenodo 10.5281/zenodo.17918239). This pipeline is
        # code-consistent but NOT bit-reproducing, so overwriting canon would
        # silently replace the paper's numbers with a drifted regen. Refuse and
        # redirect to a timestamped non-canonical path so no work is lost.
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = ROOT / "results" / "data" / f"asri_regen_{ts}.parquet"
        print(
            "\n" + "=" * 72 +
            "\nREFUSED: --overwrite-canonical will NOT clobber "
            f"{CANON.relative_to(ROOT)}." +
            "\nThat file is the FROZEN, published series of record (see "
            "DATA_PROVENANCE.md)." +
            f"\nWriting the code-consistent regen to a non-canonical path instead:" +
            f"\n  {out.relative_to(ROOT)}" +
            "\n" + "=" * 72
        )
    else:
        out = Path(args.out) if args.out else (ROOT / "results" / "data" / "asri_generated.parquet")
    # Defence in depth: never write to the canonical path from this generator.
    if out.resolve() == CANON.resolve():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        redirected = out.with_name(f"asri_regen_{ts}.parquet")
        print(f"\nRefusing to write canonical path; redirecting -> {redirected}")
        out = redirected
    out.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_parquet(out)
    print(f"\nWrote {len(df)} rows x {len(COLS)} cols -> {out}")
    print(f"ASRI range [{df['asri'].min():.2f}, {df['asri'].max():.2f}], "
          f"max on {df['asri'].idxmax().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
