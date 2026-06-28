# `results/data/` — file manifest

Disambiguates the canonical series from sensitivity variants. Full provenance:
`../../DATA_PROVENANCE.md`; reviewer steps: `../../REPRODUCIBILITY.md`.

## Canonical (the paper's dataset of record)

| File | What it is |
|------|------------|
| **`asri_history.parquet`** | **THE published ASRI series.** 1841 rows, 2021-01-01→2026-01-15, max **84.70** on 2022-11-08, mean **39.20**. sha256 `f0fc1502…628a827d`, md5 `3e423789…2f88cea3`. Zenodo `10.5281/zenodo.17918239`. Every paper headline recomputes from this file. **Frozen — do not regenerate or overwrite.** |

## Peg-fix sensitivity variants (NOT the published series, NOT independent regens)

These are documented sensitivities of the Bug-3 peg fix (`DATA_PROVENANCE.md` §3).
They are perturbations *of* the published series (e.g. SCR_new = SCR_old +
0.1·(pv_new−10)), produced by `scripts/regenerate_asri_with_peg.py` — they
presuppose the published numbers, they do not reconstruct them. Headline is robust
(max stays ~84.4–84.8).

| File | Variant |
|------|---------|
| `asri_history_pegfix.parquet` | real peg, as-published coin-id map |
| `asri_history_pegfix_corrected.parquet` | real peg + corrected (D2) coin-id map → max ~84.42 |
| `asri_history_pegfix_intraday.parquet` | real peg, intraday daily-low prices (`ASRI_PEG_INTRADAY=1`) |
| `asri_history_pegfix_corrected_intraday.parquet` | real peg + D2 map + intraday |

## Other artefacts

| File | What it is |
|------|------------|
| `asri_generated.parquet` | Non-canonical smoke/scratch output of `scripts/generate_asri_series.py`. Regenerable; not used by the paper. |
| `dy_rolling_connectedness.csv`, `dy_rolling_connectedness_daily.csv` | Diebold-Yilmaz rolling connectedness benchmark series (input to `baseline_comparison.py`). |

## Relocated (kept out of this dir to keep the canonical path unambiguous)

In `../../_repro_backups_jun2026/`:
- `asri_history_OLD_hardcoded.parquet`, `asri_history_OLD_prerepro.parquet`,
  `asri_history.parquet.bak-20260628-204402` — byte-identical backups of the
  canonical series (same sha256 as `asri_history.parquet`).
- `asri_history_REGEN_BROKEN.parquet` — a **broken** full-pipeline regen
  (max 52.42; recomputed sub-indices off drifted current snapshots). Kept only as
  evidence of why the series must stay frozen. **Never** use it as canon.
