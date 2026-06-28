# REPRODUCIBILITY — ASRI (arXiv:2602.03874)

This document gives a reviewer exact steps to reproduce every headline number in
the paper, and states honestly which level of reproduction works.

There are **two** levels:

1. **Dataset-level (works, deterministic).** Every reported number recomputes from
   the frozen canonical series `results/data/asri_history.parquet` via the
   `scripts/` analysis layer. This is the reproduction of record.
2. **Generation-from-raw-APIs (provided, partial).** The pipeline that built the
   series is provided (`src/asri/backtest/`, `scripts/generate_asri_series.py`).
   With a frozen universe snapshot it runs deterministically, but it does **not**
   bit-reproduce the published series — see `DATA_PROVENANCE.md` §2–3.

---

## 0. Environment

The scientific stack must run on **Python 3.11–3.13** (3.14 segfaults pandas; this
repo's lesson). Using `uv`:

```bash
cd asri/code
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install pandas numpy scipy statsmodels scikit-learn hmmlearn pyarrow structlog
# (generation pipeline also needs: httpx sqlalchemy asyncpg python-dotenv pydantic pydantic-settings)
```

Confirmed reproduction stack: pandas 2.3.3, numpy 2.1.3, scipy 1.18, statsmodels
0.14.6, scikit-learn 1.9.0, hmmlearn 0.3.3, Python 3.12.12.

## 1. Get the frozen series

The canonical parquet ships in the repo (`results/data/asri_history.parquet`) and
is mirrored on Zenodo (`10.5281/zenodo.17918239`). Verify it before running
anything:

```bash
sha256sum results/data/asri_history.parquet
# f0fc15020635901c175b5e1e4e21f723c9b41a59e82b7068bbf1e847628a827d
python -c "import pandas as pd; d=pd.read_parquet('results/data/asri_history.parquet'); \
print(len(d), round(d['asri'].max(),4), round(d['asri'].mean(),4), d['asri'].idxmax().date())"
# 1841 84.7 39.2022 2022-11-08
```

If this fails, you are not running against the published series — restore it from
`_repro_backups_jun2026/asri_history.parquet.bak-20260628-204402` (same sha256).

## 2. Reproduce the headline numbers (dataset-level — WORKS)

Run from `asri/code`. Each script reads the frozen parquet and prints (and saves)
its numbers. **Actual outputs below were reproduced on the frozen series and match
the paper to reported precision.**

| # | Command | Paper claim | Reproduced (actual) | Status |
|---|---------|-------------|---------------------|--------|
| 1 | `python scripts/event_study_hac.py` | HAC t(L=20): Terra 1.72 (p=0.086, n.s.), Celsius 3.49, FTX 3.28, SVB 3.86; **3/4** survive at 5% | Terra **1.72** (p=0.086), Celsius **3.49** (p=0.001), FTX **3.28** (p=0.001), SVB **3.86** (p<0.001); survivors = Celsius/FTX/SVB (**3/4**) | PASS |
| 2 | `python scripts/baseline_comparison.py` | AUROC ASRI **0.866**, Contagion **0.851**, PC1 **0.858**, D-Y **0.670**, Fear&Greed **0.789**; aggregation +~0.008 over best channel | ASRI **0.8657**, Contagion **0.8505**, PC1(cov) **0.8576**, D-Y **0.6696**, F&G **0.7886**; ASRI − best-simple = **+0.0081** | PASS |
| 3 | `python scripts/moving_block_bootstrap_roc.py` | ASRI AUROC 0.866 / AUPRC 0.298 vs D-Y 0.670 / 0.121; ASRI beats D-Y, diff CI excludes 0 | ASRI AUROC **0.8657** / AUPRC **0.2977**; D-Y **0.6696** / **0.1215**; DIFF AUROC +0.194, 95% CI [+0.10,+0.29], p≈0 | PASS |
| 4 | `python scripts/generate_detection_table.py` | peaks Terra 48.7 / Celsius 71.4 / **FTX 84.7** / SVB 68.7; fixed τ=50 → **3/4** (Terra<50); event-study 4/4; walk-forward OOS **4/4** | identical: FTX peak **84.7**, τ=50 recall **3/4** (75%), event-sig 4/4, WF-OOS **4/4** (100%) | PASS |
| 5 | `python scripts/extract_hmm_diagnostics.py` | 3 regimes, persistence **0.997 / 0.992 / 0.980**; Crisis mean 48.2, freq 23.8% | persistence **0.997 / 0.992 / 0.980**; Crisis mean **48.2**, freq **23.8%** | PASS |

Seeds are fixed (e.g. bootstrap seed=42) so confidence intervals are reproducible.

### Out-of-scope of the five core scripts (documentation accuracy)

- **VIX/10Y composite AUROC 0.875** is produced by a *separate* script,
  `scripts/baseline_extension.py` (a standalone TradFi-only composite that fetches
  live FRED data), not by `baseline_comparison.py`.
- **Lead-time figures (19 d / 26 d)** come from the lag / real-D-Y analysis
  (`scripts/real_dy_hmm_analysis.py`), not from the detection table.
- `event_study_hac.py` reports **asymptotic-normal (z) HAC** p-values; the
  load-bearing t-statistics (1.72 / 3.49 / 3.28 / 3.86) are what the paper cites and
  they match exactly.

## 3. Generation from raw APIs (PROVIDED — partial, NOT bit-reproducing)

```bash
# 1. Freeze the current DeFiLlama protocols/bridges universe (deterministic regens)
python scripts/dump_universe_snapshot.py 2026-06-28

# 2. Run the corrected, code-consistent generator (live FRED/DeFiLlama; needs FRED_API_KEY in .env)
ASRI_SNAPSHOT_AS_OF=2026-06-28 \
python scripts/generate_asri_series.py --smoke           # a few crisis dates
# full span -> writes a NON-canonical parquet (canonical path is refused):
ASRI_SNAPSHOT_AS_OF=2026-06-28 \
python scripts/generate_asri_series.py --start 2021-01-01 --end 2026-01-15 \
    --out results/data/asri_regen_full.parquet
```

This exercises the Jun-2026 fixes (D2 coin-ids, real peg, D5 rolling-365 TVL,
frozen-snapshot determinism). **It will not equal the published series** — the
original live inputs and universe snapshot were never archived (`DATA_PROVENANCE.md`).
The generator **refuses** `--overwrite-canonical` and never writes the canonical
path, by design. To reproduce the exact legacy (pre-peg-fix) construction set
`ASRI_PEG_HARDCODE=1`.

## 4. What is and is not reproducible — summary

| Level | Reproducible? |
|-------|---------------|
| Every paper headline from the frozen series (scripts above) | **YES**, deterministic |
| SCR/peg, FRED (DGS10/VIXCLS/T10Y2Y/SP500), TVL/BTC channels — point-in-time | **YES**, re-derivable |
| DeFi/Contagion/Arbitrage channels — depend on the current protocols/bridges universe | **NO** (universe never snapshotted; frozen only inside the parquet) |
| Bit-exact regen of the published daily series from live APIs | **NO** (inputs gone; see `DATA_PROVENANCE.md`) |

The frozen parquet is the reproducibility artefact of record. The generation
pipeline is provided for transparency and correctness, with the limits above
stated honestly.
