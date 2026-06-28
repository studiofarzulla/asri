# Frozen universe snapshots (`data/snapshots/`)

The ASRI generation pipeline (`src/asri/backtest/historical.py`) needs a list of
DeFiLlama **protocols** and **bridges** to compute the DeFi-Liquidity,
Contagion, and Arbitrage-Opacity channels. DeFiLlama only exposes the *current*
universe, so the pipeline projects one universe snapshot onto every historical
date (a documented look-ahead).

To make a regeneration **deterministic and auditable**, the fetcher reads that
universe from a frozen JSON snapshot here when one is present:

```
protocols_<as_of>.json   # output of GET https://api.llama.fi/protocols
bridges_<as_of>.json     # output of GET https://bridges.llama.fi/bridges (["bridges"])
```

Create one with:

```bash
python scripts/dump_universe_snapshot.py 2026-06-28
```

Then run the generator against it:

```bash
ASRI_SNAPSHOT_AS_OF=2026-06-28 python scripts/generate_asri_series.py --smoke
# or in code: HistoricalDataFetcher(as_of="2026-06-28")
```

When no snapshot exists, the fetcher falls back to a **live** pull and loudly
records `LIVE non-deterministic (as-of <today>)` in `data_quality`.

## Honest reproducibility caveat

The **original (~Jan-2026) generation-time universe was never snapshotted.**
Freezing a snapshot now pins a *current, post-publication* universe. That makes
*future* regens reproducible, but it does **not** bit-reproduce the published,
frozen canonical series at `results/data/asri_history.parquet` (max 84.70 ASRI,
2022-11-08). The published DeFi/Contagion/Arbitrage channels exist only inside
that frozen parquet. See `../../DATA_PROVENANCE.md` and `../../REPRODUCIBILITY.md`.

This directory is intentionally left empty in version control until a snapshot
is dumped locally.
