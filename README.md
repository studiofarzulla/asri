# ASRI: Aggregated Systemic Risk Index

**Interpretable, channel-decomposed monitoring of crypto-native systemic stress**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17918238-blue.svg)](https://doi.org/10.5281/zenodo.17918238)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.03874-b31b1b.svg)](https://arxiv.org/abs/2602.03874)
[![Status](https://img.shields.io/badge/Status-Under_Review-blue.svg)](https://arxiv.org/abs/2602.03874)

**Working Paper DAI-2509** | [Dissensus](https://dissensus.ai) | [Dashboard](https://asri.dissensus.ai)

## Abstract

The Aggregated Systemic Risk Index (ASRI) is a composite measure for **retrospective, interpretable monitoring** of systemic stress arising from DeFi-TradFi interconnection. It aggregates four sub-indices -- Stablecoin Concentration Risk, DeFi Liquidity Risk, Contagion Risk, and Regulatory Opacity Risk -- into a daily composite, decomposable back to its channels. It is **not** a validated real-time early-warning system.

Against four major crypto crises (Terra/Luna, Celsius/3AC, FTX, SVB), an HAC-robust event study finds statistically significant abnormal stress for **three of four** events (Terra/Luna non-significant under serial-correlation-robust inference, t = 1.72, p = 0.086); fixed-threshold detection (ASRI >= 50) likewise identifies three of four, with Terra/Luna's pre-window peak (46.0) below the threshold. A Hidden Markov Model labels three regimes (Low/Moderate/Crisis) with >97% persistence, restating the index's own serial correlation rather than statistically distinct generating processes.

On a fair-baseline comparison (identical labels and protocol), ASRI's discrimination (AUROC **0.866**) is **not statistically distinguishable** from its single strongest sub-index (Contagion, 0.851) or from PC1 of its sub-indices (0.858) -- with only four crisis episodes, those gaps are within overlapping bootstrap intervals. ASRI's measured advantage is confined to the Diebold-Yilmaz connectedness benchmark (0.670), which is itself constructed from ASRI's sub-indices and is outperformed by an off-the-shelf Fear & Greed index (0.789). **The value of aggregation here is interpretive -- channel decomposition, a single auditable composite, regime structure -- not discriminative.** ASRI proxies DeFi-TradFi transmission via TradFi-stress indicators (Treasury/VIX/yield), rather than measuring a realised exposure directly. An open-source implementation with live dashboard is provided.

Paper links:
- arXiv: https://arxiv.org/abs/2602.03874
- DOI: https://doi.org/10.48550/arXiv.2602.03874

## Key Findings

| Finding | Result |
|---------|--------|
| Crisis detection (HAC-robust) | Significant abnormal stress for **3/4** crises (Celsius/3AC, FTX, SVB; t = 3.28--3.86, p < 0.01); Terra/Luna non-significant (t = 1.72, p = 0.086) |
| Threshold detection | Fixed ASRI >= 50 identifies 3/4 at ~19-day first-crossing lead; Terra/Luna pre-window peak 46.0 < 50. Walk-forward (training-calibrated thresholds) recovers 4/4 out-of-sample, at a 47--59% false-alarm cost |
| Regime structure | HMM labels 3 regimes (Low/Moderate/Crisis), persistence >97% (0.997 / 0.992 / 0.980) -- interpretive structure over an autocorrelated trend |
| Fair-baseline discrimination | ASRI AUROC 0.866 **not distinguishable** from Contagion channel alone (0.851) or PC1 (0.858); aggregation adds ~+0.008 AUROC over the best single channel |
| Benchmark caveat | The Diebold-Yilmaz benchmark (0.670) is built from ASRI's own sub-indices (circular) and is beaten by an off-the-shelf Fear & Greed index (0.789) -- outperforming it is not evidence of value |
| Out-of-sample behaviour | No sustained false alarms on 2024--2025 holdout (single non-systemic Aug-2024 elevation, peak 58.8) |

Full numbers: `results/baseline_comparison.json`, `results/event_study_hac.json`, `results/moving_block_bootstrap_roc.json`, `results/regenerated_tables_jun2026.json`.

## Keywords

systemic risk, cryptocurrency, decentralised finance, stablecoin stability, contagion risk, DeFi-TradFi interconnection, interpretable monitoring, channel decomposition

## Architecture

ASRI comprises four weighted sub-indices aggregated into a daily composite score:

| Sub-Index | Weight | Coverage |
|-----------|--------|----------|
| Stablecoin Concentration Risk | 30% | Peg deviation, dominance, reserve opacity |
| DeFi Liquidity Risk | 25% | TVL drawdowns, protocol concentration, composability |
| Contagion Risk | 25% | Cross-market correlation, exchange flow, cascade metrics |
| Regulatory Opacity Risk | 20% | Classification uncertainty, enforcement patterns |

> Note: in discrimination terms the Contagion channel carries most of the signal (single-channel AUROC 0.851); the "stablecoin dominance" of some weight specifications is about loading, not predictive power.

### Project Structure

```
asri/
├── src/asri/
│   ├── api/          # FastAPI endpoints
│   ├── ingestion/    # Data source connectors
│   ├── signals/      # Sub-index calculations
│   └── models/       # Database models
├── tests/            # Test suite
├── scripts/          # Analysis + verification scripts (event study, bootstrap, baselines)
├── config/           # Configuration files
└── docs/             # Documentation
```

### API Endpoints

```
GET /asri/current          # Current ASRI value + sub-indices
GET /asri/timeseries       # Historical data
GET /asri/subindex/{name}  # Individual sub-index
GET /asri/regime           # Regime classification
GET /asri/validation       # Validation summary
GET /asri/methodology      # Documentation
```

### Production Routing

- Canonical dashboard: `https://asri.dissensus.ai/`
- Canonical API base: `https://asri.dissensus.ai/api`
- Canonical docs: `https://asri.dissensus.ai/docs`
- Legacy API compatibility: `https://api.dissensus.ai/*`

### Data Sources

| Source | Type | Connector |
|--------|------|-----------|
| CoinGecko | Prices, market data (cross-market correlations) | `src/asri/ingestion/coingecko.py` |
| DeFi Llama | TVL, stablecoin supply and peg data | `src/asri/ingestion/defillama.py` |
| FRED | Macro indicators (Treasury, VIX, yields) | `src/asri/ingestion/fred.py` |
| News (CryptoPanic / NewsAPI) | Regulatory news sentiment (VADER) | `src/asri/ingestion/news.py` |

## Getting Started

```bash
# Clone
git clone https://github.com/studiofarzulla/asri.git
cd asri

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run
uvicorn asri.api.main:app --reload
```

## Reproducing the paper results

All reported numbers reproduce **deterministically** from the frozen canonical
ASRI series `results/data/asri_history.parquet` (max **84.70** on 2022-11-08,
mean **39.20**, 1841 rows; sha256 `f0fc1502…628a827d`; Zenodo concept
`10.5281/zenodo.17918238`, parquet deposited from v2.0.0 onwards). This series is a **released, frozen dataset of
record**, not a re-derived artefact: the generation pipeline's original live
inputs and protocols/bridges universe were never archived, so the daily series is
**not** bit-reproducible from the live ingestion pipeline. The analysis scripts
read the frozen parquet directly.

> Use Python **3.11–3.13** (3.14 segfaults pandas). See `REPRODUCIBILITY.md` for
> the exact env, the script→number map (verified actuals), and provenance; see
> `DATA_PROVENANCE.md` for the hash freeze and the documented generation-code bugs.

```bash
# verify the frozen series first
sha256sum results/data/asri_history.parquet   # f0fc1502…628a827d

python scripts/event_study_hac.py            # HAC event study: t 1.72/3.49/3.28/3.86, 3/4 survive
python scripts/moving_block_bootstrap_roc.py # ASRI 0.866 vs Diebold-Yilmaz 0.670, block-bootstrap CIs
python scripts/baseline_comparison.py        # AUROC ASRI 0.866 / Contagion 0.851 / PC1 0.858 / F&G 0.789
python scripts/generate_detection_table.py   # peaks (FTX 84.7), τ=50 3/4, walk-forward OOS 4/4
python scripts/extract_hmm_diagnostics.py    # 3 regimes, persistence 0.997/0.992/0.980
```

The generation pipeline (`src/asri/backtest/`, `scripts/generate_asri_series.py`)
is provided and corrected (D2 coin-ids, real peg, D5 rolling-365 TVL,
frozen-universe-snapshot determinism). With a frozen snapshot
(`scripts/dump_universe_snapshot.py`) it runs deterministically, but it produces a
*code-consistent* series — it does not, and is hard-refused from, overwriting the
frozen canonical parquet.

## Citation

```bibtex
@article{farzulla2025asri,
  author    = {Farzulla, Murad and Maksakov, Andrew},
  title     = {ASRI: An Aggregated Systemic Risk Index for Cryptocurrency Markets},
  year      = {2025},
  eprint    = {2602.03874},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.RM},
  doi       = {10.5281/zenodo.17918238}
}
```

## Authors

- **Murad Farzulla** -- [Dissensus](https://dissensus.ai) & King's College London
  - ORCID: [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)
  - Email: murad@dissensus.ai
- **Andrew Maksakov** -- [Dissensus](https://dissensus.ai)

## License

Paper content: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | Code: [MIT](LICENSE)
