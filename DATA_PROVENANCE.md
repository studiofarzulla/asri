# DATA PROVENANCE — ASRI canonical series

This file pins the provenance of the **published ASRI daily series** and documents,
honestly, what is and is not reproducible from the generation pipeline.

## 1. The frozen canonical dataset (dataset of record)

The series reported in the paper (arXiv:2602.03874) and exposed by the analysis
scripts in `scripts/` is the frozen parquet:

```
results/data/asri_history.parquet
```

| Property        | Value |
|-----------------|-------|
| sha256          | `f0fc15020635901c175b5e1e4e21f723c9b41a59e82b7068bbf1e847628a827d` |
| md5             | `3e4237890fd534fe0ee07c5a2f88cea3` |
| rows            | 1841 |
| date span       | 2021-01-01 → 2026-01-15 (daily, index name `date`) |
| columns         | `asri`, `stablecoin_risk`, `defi_liquidity_risk`, `contagion_risk`, `arbitrage_opacity` |
| max ASRI        | **84.7000** on **2022-11-08** (FTX) |
| mean ASRI       | **39.2022** |
| min ASRI        | 25.8000 |
| Archived at     | Zenodo concept DOI `10.5281/zenodo.17918238` (parquet deposited from v2.0.0; earlier version records were paper-only) |

**This series is treated as a frozen, released dataset — not a re-derived
artefact.** Every headline number in the paper recomputes deterministically from
it via the analysis scripts (see `REPRODUCIBILITY.md`). Do not regenerate or
overwrite it.

Byte-identical backups are kept in `_repro_backups_jun2026/`:
`asri_history_OLD_hardcoded.parquet`, `asri_history_OLD_prerepro.parquet`,
`asri_history.parquet.bak-20260628-204402` (all sha256 above).

To verify your copy:

```bash
sha256sum results/data/asri_history.parquet
# expect f0fc15020635901c175b5e1e4e21f723c9b41a59e82b7068bbf1e847628a827d
python -c "import pandas as pd; d=pd.read_parquet('results/data/asri_history.parquet'); \
print(len(d), round(d['asri'].max(),4), round(d['asri'].mean(),4), d['asri'].idxmax().date())"
# expect 1841 84.7 39.2022 2022-11-08
```

### 1.1 The stored `asri` column is the series of record — recomposition is NOT an identity

The parquet's `asri` column is **not** the Eq-6 weighted recomposition of its own
sub-index columns. Recomposing (0.30·SCR + 0.25·DLR + 0.25·CR + 0.20·AO) from the
stored sub-indices reproduces `asri` only approximately: mean gap +0.84 pts,
MAE ≈ 1.06, max 6.34 on 2022-11-25; at the headline peak the stored value is
**84.70** (2022-11-08) vs **81.06** recomposed. Cause: the sub-index columns were
repaired *after* the aggregate `asri` column was generated (the post-generation
sub-index repair), and the released aggregate was deliberately left untouched.
The rule is:

- **The stored `asri` column is the released series of record** — the paper's
  numbers, the Zenodo deposit, and every published headline trace to it.
- Recomposition from the stored sub-indices is a *different* (internally
  consistent) series. If you want a series where Eq-6 holds row-by-row, use the
  `open_pipeline_full` recompute (`asri_daily_open` in D1;
  `results/data/asri_open_full_20260711.parquet`), not a recomposed canon.

**Dashboard/API history:** the original one-shot D1 load recomputed `asri` from
the sub-indices at load time and discarded the released column, so
api.dissensus.ai had *never* served the published canon series (it served the
recomposed one, peak 81.1 on 2022-11-08). Fixed 11 Jul 2026: the released `asri`
values were backfilled onto all 1,841 `paper_canon` rows in D1 (1,495 rows
changed), and as of worker **v2.3.0** the API serves the **stored** value for
`paper_canon` rows while keeping read-time Eq-6 recomposition for
pipeline-generated profiles (`open_pipeline_continuation`, `open_pipeline_full`),
where it is an identity by construction (verified on every row of both). The
canon rows' `asri_30d_avg`, `trend`, and `alert_level` columns were left as
originally loaded (recomposition-derived): no endpoint serves them for canon
rows, and the worker derives alert/trend from the served `asri` at read time.

## 2. Why the series is frozen and NOT regenerated from the pipeline

The generation pipeline (`src/asri/backtest/{historical,backtest}.py`,
entrypoint `scripts/generate_asri_series.py`) pulls its inputs **live** from
DeFiLlama (TVL, stablecoins, BTC), FRED (DGS10/VIXCLS/T10Y2Y/SP500), and the
DeFiLlama protocols/bridges universe. No raw inputs were ever archived
(`data/raw/`, `data/cache/` do not exist). Two consequences:

1. **The original point-in-time inputs are gone.** A re-pull returns *today's*
   live values, not the snapshot used at generation time (~Jan 2026).
2. **The protocols/bridges universe was never snapshotted.** That universe drives
   the DeFi/Contagion/Arbitrage channels and only exists, frozen, inside the
   parquet.

The repo's own forensic check
(`results/tvl_respec_jun2026/provenance_parquet_vs_code.json`) confirms the
released SCR is **not reproducible from current code + current DeFiLlama/FRED data
under any TVL normalisation window**. A full-pipeline regen therefore produces a
*drifted* series (an attempted regen on 2026-06-28 collapsed max 84.70 → 52.42 and
contagion 88 → 41 purely from universe/input drift; that broken artefact is kept,
clearly labelled, at `_repro_backups_jun2026/asri_history_REGEN_BROKEN.parquet`).

**Reproducibility status:** dataset-level **YES** (frozen parquet + deterministic
analysis); generation-from-live-APIs **NO/partial** (SCR/peg + FRED + TVL/BTC
channels are point-in-time and re-derivable; the protocols/bridges look-ahead
channels are not, absent the original snapshot).

## 3. Known generation-code bugs the published series carries

These are documented **for honesty, not re-numbering.** The published series was
produced by an earlier code state that carried all four items below. The current
code **fixes** each (so the public repo is correct going forward), but the fixes
are *not* applied to the frozen series — they would shift it slightly (peg) or
not at all (coin-id), and the published numbers remain those in the parquet.

### D2 — DeFiLlama stablecoin id mislabelling  *(fixed; ZERO impact on published series)*
`HistoricalDataFetcher.MAJOR_STABLES` previously mapped id=3 → "DAI" (id=3 is
actually USTC/TerraClassicUSD, the coin that collapsed May 2022) and id=11 → "UST"
(id=11 is actually USDP/Pax Dollar, never depegged). Corrected map now:
`{1:USDT, 2:USDC, 3:UST, 4:BUSD, 5:DAI, 6:FRAX, 7:TUSD, 8:LUSD, 9:FEI, 10:MIM,
11:USDP, 12:USDN}` (mirror kept in sync in `scripts/backfill_d1_standalone.py`;
`LagAwareHistoricalFetcher` inherits it). **Impact on the published series: none**
— under the hardcoded-peg pipeline (below) labels never reach a price; the only
consumers (HHI/concentration/top-2 share) use supply *values*, not labels.

### Hardcoded stablecoin peg  *(fixed; SMALL documented sensitivity)*
`_snapshot_to_inputs` previously hardcoded `price=1.0`, `peg_deviation=0.0`,
`peg_volatility=10.0` for every stablecoin on every date ("can't easily get
historical peg data"), so the Stablecoin-Concentration-Risk peg term was inert
straight through every depeg. The current code reads real supply-weighted peg
volatility from `scripts/peg_loader.py` (`data/peg_history.csv`); set
`ASRI_PEG_HARDCODE=1` to restore the exact legacy behaviour. **Impact:** small and
documented — peg-on (corrected map) shifts max 84.70 → ~84.42, mean 39.20 → ~39.00;
headline is robust. The peg-sensitivity variants live at
`results/data/asri_history_pegfix*.parquet` (see `results/data/README.md`); they
are perturbations *of* the published series, not independent regens.

### D5 — expanding-max TVL saturation  *(fixed; affects 2024+ only)*
`_get_max_tvl_before` previously used an **expanding** running max over all
history, which froze at the Nov-2021 all-chain TVL ATH and saturated the TVL term
at ~98–100 from mid-2022 onward (a ~12 ASRI-pt offset). The current code uses a
trailing **365-day rolling max**. At the four 2022–2023 crises the Nov-2021 ATH is
still inside the trailing year, so rolling == expanding there; the specs diverge
only in 2024+. A 30-day rate-of-change spec exists as a sensitivity in
`results/tvl_respec_jun2026/` (not the default).

### Look-ahead — current protocols/bridges universe  *(fixed for future runs; irreducible for the published series)*
`fetch_snapshot` projects a single **current** DeFiLlama protocols/bridges universe
onto every historical date. The current code reads that universe from a frozen
on-disk snapshot (`data/snapshots/<name>_<as_of>.json`, dumped by
`scripts/dump_universe_snapshot.py`) when present — making future regens
deterministic — and otherwise pulls live and loudly flags
`data_quality[...]="LIVE non-deterministic"`. **The original generation-time
universe was never snapshotted**, so this guarantees determinism only *going
forward*; it cannot recover the published DeFi/Contagion/Arbitrage channels, which
remain reproducible **only** from the frozen parquet.

## 4. Other inputs on disk

- `data/peg_history.csv` (sha256 `fe9b98211ec734da6bd9ad9bf93f7d9f4958cf3675c8fe79b72227e52ae33d0b`,
  12,895 rows) — a **post-publication (Jun 2026) reconstruction** built by
  `scripts/build_peg_history.py` from a fresh live DeFiLlama pull. It drives the
  peg-fix sensitivity, **not** the published series (which used the hardcoded peg).
- `data/_ohlc_intraday_lows_2022_2023.json` — intraday daily-low prices for the
  `ASRI_PEG_INTRADAY=1` stress-sensitive peg variant (sensitivity only).
- `data/snapshots/` — frozen universe snapshots (empty until you run the dumper).

### Analysis inputs/outputs with generators (provenance closers)

- `results/data/treasury_dgs10.csv` (**FRED DGS10**, 10Y Treasury constant
  maturity) — a manual FRED input that is **not committed**. It is required only
  by `scripts/duration_sensitivity.py` (Reviewer-Q6 duration sensitivity). That
  script now **fetches it on demand** from the FRED public CSV endpoint
  (`fredgraph.csv?id=DGS10`, no API key) and caches it to this path; DGS10 is a
  **non-revised** public series, so a live pull reproduces the historical values.
  If FRED is unreachable the script aborts cleanly without synthesising a yield.
- `results/data/dy_rolling_connectedness_daily.csv` (rolling Diebold--Yilmaz total
  connectedness; 60-day rolling VAR(1), generalized FEVD H=10, on the four
  sub-indices; paper window, mean ≈ 28.74%) — **frozen released artefact**. Its
  generator is now in the repo: `scripts/build_dy_daily.py` calls the same
  `rolling_connectedness()` routine on the frozen parquet sub-indices and writes
  the CSV. **Caveat (same class as sec. 2):** the released CSV was built from the
  *pre-repro* sub-index series; the re-frozen parquet now carries more valid
  sub-index days, so a fresh regen **drifts** (≈44%) and does **not** bit-reproduce
  the released figure. The generator therefore refuses to overwrite the frozen CSV
  unless `ASRI_DY_FORCE=1` (it writes a `*.regen.csv` sibling by default). The
  released CSV remains the artefact of record.
- Bybit specificity readings (supplement Table `tab:bybit`) are recomputed from
  the frozen parquet by `scripts/verify_deferred_validation.py` (now
  cross-referenced in the table caption).

## 5. One-line summary

> The published ASRI series is a **frozen released dataset** (sha256 above; Zenodo
> concept 10.5281/zenodo.17918238). All paper numbers reproduce from it deterministically.
> The generation pipeline is provided and corrected, but does **not** bit-reproduce
> the series — its original live inputs and universe snapshot were never archived.
