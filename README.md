# ASRI - Aggregated Systemic Risk Index

> Unified, quantified systemic risk monitoring for crypto/DeFi markets

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

ASRI aggregates fragmented crypto risk indicators into a single coherent systemic risk index, providing daily monitoring from 2020 to present via REST API and web dashboard.

### Key Features

- **4 Risk Sub-Indices**: Stablecoin exposure, DeFi liquidity, cross-market contagion, regulatory opacity
- **Daily Updates**: Automated data ingestion from 10+ sources
- **Backtested**: Validated against major crypto crises (2020-2025)
- **REST API**: Public and premium access tiers
- **Stress Testing**: Scenario analysis for risk modeling

## Quick Start

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

## Project Structure

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

## Data Sources

| Source | Type | Status |
|--------|------|--------|
| DeFi Llama | TVL, volumes | 🟢 Planned |
| Token Terminal | Protocol metrics | 🟢 Planned |
| FRED | Macro indicators | 🟢 Planned |
| Messari | On-chain data | 🟡 Conditional |
| Chainalysis | Risk reports | 🟡 Crawler |

## API Endpoints

```
GET /asri/current          # Current ASRI value + sub-indices
GET /asri/timeseries       # Historical data
GET /asri/subindex/{name}  # Individual sub-index
GET /asri/stress-test      # Scenario analysis
GET /asri/methodology      # Documentation
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Resurrexi Labs** | [resurrexi.io](https://resurrexi.io)
