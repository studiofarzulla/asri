# ASRI Implementation Session Notes
**Date:** 2025-12-16
**Status:** MVP Complete - Fully Automated

---

## What We Built

### Fully Automated Systemic Risk Index
ASRI now calculates from **4 live data sources** with **zero manual inputs**:

```
ASRI: 46.9 | Alert Level: LOW

Sub-Indices:
  stablecoin_risk           59.1  (USDT dominance, HHI=4308)
  defi_liquidity_risk       49.6  (protocol concentration, audit coverage)
  contagion_risk            27.9  (low BTC-S&P500 correlation)
  arbitrage_opacity         48.7  (news-based regulatory sentiment)
```

---

## Data Sources Implemented

### 1. DeFiLlama (No Auth Required)
**File:** `src/asri/ingestion/defillama.py`
**Endpoints:**
- `/protocols` - 6826 protocols with TVL, category, audits
- `/v2/historicalChainTvl` - Historical TVL for volatility
- `stablecoins.llama.fi/stablecoins` - 316 stablecoins with supply, price, chains
- `bridges.llama.fi/bridges` - 88 bridges

**Used For:**
- TVL ratio (current vs historical max)
- Protocol concentration (HHI of top 10)
- Stablecoin concentration (HHI = 4308, highly concentrated)
- Audit coverage (~44%)
- RWA exposure (2.7% of TVL)
- Bridge count (contagion vectors)

### 2. FRED (Free API Key Required)
**File:** `src/asri/ingestion/fred.py`
**API Key:** Stored in `.env` as `FRED_API_KEY`
**Series:**
- `DGS10` - 10-Year Treasury Rate (4.18%)
- `VIXCLS` - VIX Volatility Index (16.5)
- `T10Y2Y` - Yield Curve Spread (0.67%, not inverted)
- `SP500` - S&P 500 for correlation calculation

**Used For:**
- Treasury stress indicator
- Market volatility (VIX)
- Recession signal (yield curve inversion)
- Crypto-equity correlation baseline

### 3. CoinGecko (Free Demo Key)
**File:** `src/asri/ingestion/coingecko.py`
**API Key:** Stored in `.env` as `COINGECKO_API_KEY`
**Endpoints:**
- `/coins/bitcoin/market_chart` - 90-day BTC price history

**Used For:**
- BTC-S&P500 correlation calculation (currently 0.056 - near zero!)
- Crypto is decoupled from equities right now

### 4. Google News RSS + VADER NLP (No Auth)
**File:** `src/asri/ingestion/news.py`
**RSS Feeds:**
- cryptocurrency+regulation
- bitcoin+SEC
- crypto+law+policy
- stablecoin+regulation

**Used For:**
- Regulatory sentiment score (25/100 = low risk)
- Automated NLP analysis of 50+ articles
- Detects constructive vs restrictive regulation tone

---

## Architecture

### Pipeline Flow
```
┌─────────────────────────────────────────────────────────────┐
│                    DATA INGESTION                           │
│  DeFiLlama → FRED → CoinGecko → Google News                │
│     (concurrent asyncio.gather)                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFORM LAYER                          │
│  src/asri/pipeline/transform.py                            │
│  - Raw data → normalized 0-100 inputs                      │
│  - HHI calculations                                         │
│  - Correlation calculations                                 │
│  - Sentiment → risk score conversion                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    CALCULATOR                               │
│  src/asri/signals/calculator.py                            │
│  - Sub-index formulas from paper                           │
│  - Weighted aggregation (30/25/25/20)                      │
│  - Alert level determination                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                             │
│  src/asri/pipeline/orchestrator.py                         │
│  - Coordinates all data fetching                           │
│  - Handles errors gracefully                               │
│  - Returns complete ASRI result                            │
└─────────────────────────────────────────────────────────────┘
```

### Key Files Created/Modified

**New Files:**
- `src/asri/ingestion/coingecko.py` - CoinGecko client
- `src/asri/ingestion/news.py` - Google News + VADER sentiment
- `src/asri/pipeline/transform.py` - Data transformation layer
- `src/asri/pipeline/orchestrator.py` - Pipeline coordinator

**Modified Files:**
- `src/asri/ingestion/defillama.py` - Fixed null handling, added protocols endpoint
- `.env` - Added FRED_API_KEY, COINGECKO_API_KEY

**Existing (Unchanged):**
- `src/asri/signals/calculator.py` - Core ASRI formulas
- `src/asri/models/` - Database models
- `src/asri/api/main.py` - FastAPI endpoints

---

## Sub-Index Calculations

### Stablecoin Risk (30% weight)
```python
WEIGHTS = {
    'tvl_ratio': 0.4,        # current_tvl / max_historical
    'treasury_stress': 0.3,   # normalized 10Y treasury rate
    'concentration_hhi': 0.2, # stablecoin HHI → risk score
    'peg_volatility': 0.1,    # weighted peg deviation
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
    'rwa_growth_rate': 0.30,           # RWA category TVL share
    'bank_exposure': 0.25,             # treasury + VIX composite
    'tradfi_linkage': 0.20,            # yield curve signal
    'crypto_equity_correlation': 0.15, # BTC-S&P500 correlation
    'bridge_exploit_frequency': 0.10,  # number of bridges
}
```

### Arbitrage/Opacity Risk (20% weight)
```python
WEIGHTS = {
    'unregulated_exposure': 0.25,  # chain distribution estimate
    'multi_issuer_risk': 0.25,     # stablecoin issuer count
    'custody_concentration': 0.20, # top 2 stablecoin share
    'regulatory_sentiment': 0.15,  # NEWS-BASED (live!)
    'transparency_score': 0.15,    # audit coverage (inverted)
}
```

---

## Key Findings from Live Data

1. **Stablecoin Concentration is HIGH**
   - USDT: 60.4% ($186B)
   - USDC: 25.3% ($78B)
   - HHI: 4308 (>2500 = highly concentrated)

2. **BTC-S&P500 Correlation is NEAR ZERO (0.056)**
   - Crypto currently decoupled from equities
   - Lower contagion risk than typical

3. **Regulatory Sentiment is LOW RISK (25/100)**
   - Current news about constructive frameworks (UK FCA)
   - Not crackdown/enforcement tone

4. **TVL at 67% of Historical Max**
   - Current: $119.5B
   - Max: $177.5B
   - Not at peak froth levels

5. **Yield Curve NOT Inverted (+0.67%)**
   - No recession signal from bond markets

---

## To Do Next

### Immediate
- [ ] Wire up PostgreSQL storage for historical tracking
- [ ] Add APScheduler for daily automated runs
- [ ] Implement `/asri/timeseries` endpoint with real DB queries

### Short Term
- [ ] Frontend dashboard (React scaffold exists)
- [ ] Deploy to K3s cluster
- [ ] Backtest against historical crises (Luna, FTX, SVB)

### Nice to Have
- [ ] CryptoPanic integration (needs auth token)
- [ ] Add more news sources
- [ ] LLM-based regulatory analysis for deeper insight

---

## How to Run

```bash
cd /home/purrpower/Resurrexi/projects/resurrexi-projects/asri
source .venv/bin/activate

# Full ASRI calculation
python -m asri.pipeline.orchestrator

# Test individual components
python -m asri.ingestion.news      # Regulatory sentiment
python -m asri.ingestion.coingecko # BTC prices

# Start API
uvicorn asri.api.main:app --reload
```

---

## Environment Variables (.env)

```
DATABASE_URL=postgresql+asyncpg://asri:asri@localhost:5432/asri
FRED_API_KEY=c0763ee3bcbb5756b4ceb33a21c82fdc
COINGECKO_API_KEY=CG-j2whS6an7sVUDqJU4jEkoHvo
```

---

## Dependencies Added

```bash
pip install vaderSentiment  # For news sentiment analysis
```

All other deps were already in pyproject.toml.
