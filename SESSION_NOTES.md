# ASRI Implementation Session Notes
**Date:** 2025-12-16 / 2025-12-17
**Status:** Production Ready - Fully Automated with Daily Updates

---

## What We Built

### Fully Automated Systemic Risk Index
ASRI calculates from **4 live data sources** with **zero manual inputs**, persists to PostgreSQL, generates a static dashboard, and auto-deploys daily:

```
ASRI: 46.8 | Alert Level: LOW

Sub-Indices:
  stablecoin_risk           59.1  (USDT dominance, HHI=4308)
  defi_liquidity_risk       49.6  (protocol concentration, audit coverage)
  contagion_risk            27.9  (low BTC-S&P500 correlation)
  arbitrage_opacity         48.5  (news-based regulatory sentiment)
```

**Live Dashboard:** https://resurrexi.io/asri/

---

## Complete Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DAILY AUTOMATION (01:00 UTC)                    │
│                     systemd: asri-scheduler.service                 │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                               │
│  DeFiLlama → FRED → CoinGecko → Google News                        │
│     (concurrent asyncio.gather)                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        TRANSFORM LAYER                              │
│  src/asri/pipeline/transform.py                                    │
│  - Raw data → normalized 0-100 inputs                              │
│  - HHI calculations, correlation, sentiment conversion              │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        CALCULATOR                                   │
│  src/asri/signals/calculator.py                                    │
│  - Sub-index formulas (30/25/25/20 weights)                        │
│  - Alert level determination                                        │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                                 │
│  src/asri/pipeline/orchestrator.py                                 │
│  - Coordinates data fetching                                        │
│  - calculate_and_save() → PostgreSQL                               │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        DATABASE                                     │
│  PostgreSQL (Docker: asri-postgres)                                │
│  Table: asri_daily (id, date, asri, sub-indices, trend, etc.)      │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STATIC SITE GENERATOR                            │
│  src/asri/scheduler/static_generator.py                            │
│  - Generates static_site/index.html                                │
│  - Dark theme, Chart.js, responsive                                │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT                                   │
│  scripts/deploy.sh                                                  │
│  ├── rsync → K3s cluster (immediate)                               │
│  ├── git push → resurrexi-io (Cloudflare Pages)                    │
│  └── git push → asri repo (backup)                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

### 1. DeFiLlama (No Auth)
**File:** `src/asri/ingestion/defillama.py`
- `/protocols` - 6826 protocols with TVL, category, audits
- `/v2/historicalChainTvl` - Historical TVL for volatility
- `stablecoins.llama.fi/stablecoins` - 316 stablecoins
- `bridges.llama.fi/bridges` - 88 bridges

### 2. FRED (Free API Key)
**File:** `src/asri/ingestion/fred.py`
- `DGS10` - 10-Year Treasury Rate
- `VIXCLS` - VIX Volatility Index
- `T10Y2Y` - Yield Curve Spread
- `SP500` - S&P 500 for correlation

### 3. CoinGecko (Free Demo Key)
**File:** `src/asri/ingestion/coingecko.py`
- `/coins/bitcoin/market_chart` - 90-day BTC prices
- Used for BTC-S&P500 correlation calculation

### 4. Google News RSS + VADER NLP (No Auth)
**File:** `src/asri/ingestion/news.py`
- RSS feeds for crypto regulation news
- VADER sentiment analysis
- Converts sentiment to regulatory risk score

---

## Database

### PostgreSQL (Docker)
```bash
docker run -d \
  --name asri-postgres \
  -e POSTGRES_USER=asri \
  -e POSTGRES_PASSWORD=asri \
  -e POSTGRES_DB=asri \
  -p 5432:5432 \
  -v asri-pgdata:/var/lib/postgresql/data \
  postgres:16-alpine
```

### Schema: `asri_daily`
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| date | TIMESTAMP | Calculation date (unique) |
| asri | FLOAT | Aggregate score (0-100) |
| asri_normalized | FLOAT | Same as asri |
| asri_30d_avg | FLOAT | 30-day moving average |
| trend | VARCHAR(20) | increasing/decreasing/stable |
| alert_level | VARCHAR(20) | low/moderate/elevated/high/critical |
| stablecoin_risk | FLOAT | Sub-index (30% weight) |
| defi_liquidity_risk | FLOAT | Sub-index (25% weight) |
| contagion_risk | FLOAT | Sub-index (25% weight) |
| arbitrage_opacity | FLOAT | Sub-index (20% weight) |
| created_at | TIMESTAMP | Record creation time |
| updated_at | TIMESTAMP | Last update time |

---

## API Endpoints

### FastAPI Server
```bash
uvicorn asri.api.main:app --reload --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/asri/current` | GET | Latest ASRI with `?calculate_if_missing=true` |
| `/asri/calculate` | POST | Trigger live calculation with `?save=true` |
| `/asri/timeseries` | GET | Historical data `?start=YYYY-MM-DD&end=YYYY-MM-DD` |
| `/asri/subindex/{name}` | GET | Individual sub-index history |
| `/asri/stress-test` | GET | Stress test scenarios (TODO) |
| `/asri/methodology` | GET | Methodology documentation |
| `/docs` | GET | Swagger UI |

---

## Scheduler

### Systemd Service
**File:** `scripts/asri-scheduler.service`

```bash
# Install
sudo cp scripts/asri-scheduler.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable asri-scheduler
sudo systemctl start asri-scheduler

# Check status
sudo systemctl status asri-scheduler
sudo journalctl -u asri-scheduler -f
```

### Daily Job (01:00 UTC)
**File:** `src/asri/scheduler/daemon.py`

1. Calculate ASRI from live data
2. Save to PostgreSQL
3. Generate static HTML dashboard
4. Deploy via rsync + git push

---

## Deployment

### Deploy Script
**File:** `scripts/deploy.sh`

```bash
./scripts/deploy.sh
```

**Actions:**
1. **rsync to K3s** - Immediate update to `sudosenpai:/mnt/storage/resurrexi-io/public/asri/`
2. **git push resurrexi-io** - Triggers Cloudflare Pages deploy
3. **git push asri** - Commits code changes to asri repo

### Targets
- **K3s:** https://resurrexi.io/asri/ (via Cloudflare Tunnel)
- **Cloudflare Pages:** Auto-deploys from `studiofarzulla/resurrexi-io`
- **Source:** `studiofarzulla/asri`

---

## Project Structure

```
asri/
├── .env                          # API keys (not committed)
├── .gitignore                    # Ignores static_site/, .venv/, etc.
├── pyproject.toml                # Python package config
├── SESSION_NOTES.md              # This file
│
├── scripts/
│   ├── deploy.sh                 # Full deployment script
│   ├── asri-scheduler.service    # Systemd unit file
│   └── install-service.sh        # Service installer
│
├── static_site/                  # Generated (not committed)
│   └── index.html                # Static dashboard
│
├── src/asri/
│   ├── __init__.py
│   ├── config.py                 # Pydantic settings
│   │
│   ├── api/
│   │   └── main.py               # FastAPI application
│   │
│   ├── db/
│   │   └── init_db.py            # Database initialization
│   │
│   ├── ingestion/
│   │   ├── base.py               # Base client class
│   │   ├── defillama.py          # DeFiLlama client
│   │   ├── fred.py               # FRED client
│   │   ├── coingecko.py          # CoinGecko client
│   │   └── news.py               # Google News + VADER
│   │
│   ├── models/
│   │   ├── base.py               # SQLAlchemy base + engine
│   │   ├── asri.py               # ASRIDaily model
│   │   └── raw_data.py           # Raw data models
│   │
│   ├── pipeline/
│   │   ├── calculate.py          # Calculation helpers
│   │   ├── transform.py          # Data transformation
│   │   └── orchestrator.py       # Main pipeline coordinator
│   │
│   ├── scheduler/
│   │   ├── __init__.py           # Package exports
│   │   ├── daemon.py             # APScheduler daemon
│   │   ├── jobs.py               # Daily job logic
│   │   └── static_generator.py   # HTML generator
│   │
│   └── signals/
│       └── calculator.py         # ASRI calculation formulas
```

---

## Quick Commands

```bash
cd /home/purrpower/Resurrexi/projects/resurrexi-projects/asri
source .venv/bin/activate

# Manual calculation + deploy
./scripts/deploy.sh

# Just calculate (no deploy)
python -m asri.pipeline.orchestrator

# Run scheduler daemon (foreground)
python -m asri.scheduler.daemon

# Initialize database
python -m asri.db.init_db

# Start API server
uvicorn asri.api.main:app --reload --port 8000

# Test components
python -m asri.ingestion.news          # News sentiment
python -m asri.ingestion.coingecko     # BTC prices
python -m asri.scheduler.static_generator  # Generate HTML
```

---

## Environment Variables

```bash
# .env
DATABASE_URL=postgresql+asyncpg://asri:asri@localhost:5432/asri
DATABASE_SYNC_URL=postgresql://asri:asri@localhost:5432/asri
FRED_API_KEY=<your-key>
COINGECKO_API_KEY=<your-key>
ENVIRONMENT=development
DEBUG=true
SCHEDULER_ENABLED=true
DAILY_UPDATE_HOUR=1
```

---

## Sub-Index Formulas

### Stablecoin Risk (30% weight)
```python
WEIGHTS = {
    'tvl_ratio': 0.4,           # current_tvl / max_historical
    'treasury_stress': 0.3,      # normalized 10Y treasury rate
    'concentration_hhi': 0.2,    # stablecoin HHI → risk score
    'peg_volatility': 0.1,       # weighted peg deviation
}
```

### DeFi Liquidity Risk (25% weight)
```python
WEIGHTS = {
    'top10_concentration': 0.35,  # protocol HHI
    'tvl_volatility': 0.25,       # std/mean of TVL history
    'smart_contract_risk': 0.20,  # inverse of audit coverage
    'flash_loan_proxy': 0.10,     # daily TVL changes
    'leverage_change': 0.10,      # lending protocol ratio
}
```

### Contagion Risk (25% weight)
```python
WEIGHTS = {
    'rwa_growth_rate': 0.30,            # RWA category TVL share
    'bank_exposure': 0.25,              # treasury + VIX composite
    'tradfi_linkage': 0.20,             # yield curve signal
    'crypto_equity_correlation': 0.15,  # BTC-S&P500 correlation
    'bridge_exploit_frequency': 0.10,   # number of bridges
}
```

### Arbitrage/Opacity Risk (20% weight)
```python
WEIGHTS = {
    'unregulated_exposure': 0.25,   # chain distribution estimate
    'multi_issuer_risk': 0.25,      # stablecoin issuer count
    'custody_concentration': 0.20,  # top 2 stablecoin share
    'regulatory_sentiment': 0.15,   # NEWS-BASED (live!)
    'transparency_score': 0.15,     # audit coverage (inverted)
}
```

---

## Key Findings (2025-12-16)

1. **Stablecoin Concentration: HIGH**
   - USDT: 60.4% ($186B), USDC: 25.3% ($78B)
   - HHI: 4308 (>2500 = highly concentrated)

2. **BTC-S&P500 Correlation: NEAR ZERO (0.056)**
   - Crypto currently decoupled from equities
   - Lower contagion risk than typical

3. **Regulatory Sentiment: LOW RISK (25/100)**
   - News about constructive frameworks
   - Not crackdown/enforcement tone

4. **TVL at 67% of Historical Max**
   - Current: $119.5B, Max: $177.5B
   - Not at peak froth levels

5. **Yield Curve: NOT Inverted (+0.67%)**
   - No recession signal from bond markets

---

## Future Improvements

### Short Term
- [ ] Backtest against historical crises (Luna, FTX, SVB)
- [ ] Add email/webhook alerts for threshold breaches
- [ ] Implement stress test scenarios

### Medium Term
- [ ] CryptoPanic integration (needs auth token)
- [ ] LLM-based regulatory analysis
- [ ] Interactive frontend with filters

### Long Term
- [ ] Multi-chain analysis
- [ ] Prediction model for risk spikes
- [ ] Integration with trading signals

---

## References

- **Paper:** [ASRI: Aggregated Systemic Risk Index](https://doi.org/10.5281/zenodo.17918239)
- **Live Dashboard:** https://resurrexi.io/asri/
- **API Docs:** http://localhost:8000/docs (when running)
- **Source Code:** https://github.com/studiofarzulla/asri
