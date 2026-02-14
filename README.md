# ASRI: Aggregated Systemic Risk Index

**Real-time systemic risk monitoring for cryptocurrency markets**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17918239-blue.svg)](https://doi.org/10.5281/zenodo.17918239)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.03874-b31b1b.svg)](https://arxiv.org/abs/2602.03874)
[![Status](https://img.shields.io/badge/Status-arXiv_Preprint-b31b1b.svg)](https://arxiv.org/abs/2602.03874)

**Working Paper DAI-2509** | [Dissensus AI](https://dissensus.ai) | [Dashboard](https://asri.dissensus.ai)

## Abstract

This paper introduces the Aggregated Systemic Risk Index (ASRI), the first composite measure designed to monitor systemic risks arising from DeFi-TradFi interconnection. ASRI aggregates four sub-indices -- Stablecoin Concentration Risk, DeFi Liquidity Risk, Contagion Risk, and Regulatory Opacity Risk -- into a daily composite score. Validated against four major crypto crises (Terra/Luna, Celsius/3AC, FTX, SVB), event study analysis detects statistically significant abnormal stress for all four events (t-statistics 5.47--32.64, all p < 0.01), with threshold-based detection identifying three of four at an average 30-day lead time. A Hidden Markov Model identifies three risk regimes with persistence exceeding 97%. Out-of-sample testing on 2024--2025 data confirms zero false positives. ASRI captures DeFi-specific vulnerabilities -- composability risk, flash loan exposure, and RWA linkages -- that traditional measures such as SRISK and CoVaR cannot accommodate. An open-source implementation with live dashboard is provided.

Paper links:
- arXiv: https://arxiv.org/abs/2602.03874
- DOI: https://doi.org/10.48550/arXiv.2602.03874

## Key Findings

| Finding | Result |
|---------|--------|
| Crisis detection | Statistically significant abnormal stress for all 4 major crises (t-stats 5.47--32.64, p < 0.01) |
| Early warning | Threshold-based detection identifies 3/4 crises at ~30-day lead time |
| Regime persistence | HMM identifies 3 risk regimes with >97% persistence |
| Out-of-sample validation | Zero false positives on 2024--2025 holdout data |
| DeFi-specific coverage | Captures composability risk, flash loan exposure, and RWA linkages |

## Keywords

systemic risk, cryptocurrency, decentralized finance, stablecoin stability, contagion risk, DeFi-TradFi interconnection, risk monitoring

## Architecture

ASRI comprises four weighted sub-indices aggregated into a daily composite score:

| Sub-Index | Weight | Coverage |
|-----------|--------|----------|
| Stablecoin Concentration Risk | 30% | Peg deviation, dominance, reserve opacity |
| DeFi Liquidity Risk | 25% | TVL drawdowns, protocol concentration, composability |
| Contagion Risk | 25% | Cross-market correlation, exchange flow, cascade metrics |
| Regulatory Opacity Risk | 20% | Classification uncertainty, enforcement patterns |

### Project Structure

```
asri/
├── src/asri/
│   ├── api/          # FastAPI endpoints
│   ├── ingestion/    # Data source connectors
│   ├── signals/      # Sub-index calculations
│   └── models/       # Database models
├── tests/            # Test suite
├── scripts/          # Utility scripts
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

| Source | Type | Status |
|--------|------|--------|
| DeFi Llama | TVL, volumes | Planned |
| Token Terminal | Protocol metrics | Planned |
| FRED | Macro indicators | Planned |
| Messari | On-chain data | Conditional |
| Chainalysis | Risk reports | Crawler |

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

## Citation

```bibtex
@article{farzulla2025asri,
  author    = {Farzulla, Murad and Maksakov, Andrew},
  title     = {ASRI: An Aggregated Systemic Risk Index for Cryptocurrency Markets},
  year      = {2025},
  eprint    = {2602.03874},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.RM},
  doi       = {10.5281/zenodo.17918239}
}
```

## Authors

- **Murad Farzulla** -- [Dissensus AI](https://dissensus.ai) & King's College London
  - ORCID: [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)
  - Email: murad@dissensus.ai
- **Andrew Maksakov** -- [Dissensus AI](https://dissensus.ai)

## License

Paper content: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | Code: [MIT](LICENSE)
