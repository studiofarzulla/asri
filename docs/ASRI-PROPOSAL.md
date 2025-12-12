# Aggregated Systemic Risk Index (ASRI)

> **Resurrexi Labs** | Implementation Proposal | December 2025

---

## Executive Summary

**Goal:** Build a unified, quantified systemic risk index combining DeFi/TradFi interconnection metrics, tracked daily from 2020–present, published via a proprietary API and web dashboard.

### Key Components

1. **Data Ingestion Layer** — APIs + crawlers for fragmented risk data sources
2. **Risk Signal Aggregation** — Normalization and sub-index computation
3. **Backtested Time-Series Model** — Daily data from 2020-present
4. **Predictive/Stress-Test Capability** — Scenario analysis and alerting
5. **Dashboard + REST API** — Public and premium access tiers

---

## Table of Contents

1. [Phase 1: Data Source Identification](#phase-1-data-source-identification--api-assessment)
2. [Phase 2: Risk Signal Definition](#phase-2-risk-signal-definition--taxonomy)
3. [Phase 3: Technical Architecture](#phase-3-technical-architecture)
4. [Phase 4: Backtesting Strategy](#phase-4-backtesting-strategy-2020present)
5. [Phase 5: Dashboard & API Design](#phase-5-dashboard--api-design)
6. [Phase 6: Implementation Roadmap](#phase-6-implementation-roadmap-12-16-weeks)
7. [Phase 7: Monetization](#phase-7-monetization--sustainability)
8. [Phase 8: Risk Mitigation](#phase-8-risk-mitigation--open-questions)

---

## Phase 1: Data Source Identification & API Assessment

### Primary Data Sources

| Source | Data Type | API Available | Quantified | Coverage | Notes |
|--------|-----------|---------------|------------|----------|-------|
| **DeFi Llama** | TVL, volumes, yields | ✅ REST (free, limited) | ✅ Numeric | 1000+ protocols, 100+ chains | Free tier sufficient; consider pro for rate limits |
| **TRM Labs** | Transaction risk scores, flow analytics | ⚠️ Enterprise only | ✅ Numeric | 20+ blockchains | Requires commercial license; may need crawler |
| **Elliptic** | Entity risk scoring, counterparty data | ⚠️ Enterprise | ✅ Numeric | Multi-chain | Focus on publicly available reports |
| **Token Terminal** | Protocol economics, security metrics | ✅ REST (freemium) | ✅ Numeric | 200+ protocols | Good coverage; historical data available |
| **Chainalysis** | Regulatory risk, illicit activity % | ⚠️ Institutional | ✅ Numeric | BTC, ETH-focused | Public reports + dashboards; crawler approach |
| **Galaxy SeC** | DeFi risk ratings | ⚠️ Limited | ⚠️ Semi-quantitative | DeFi-focused | Manual scraping or partnerships |
| **Messari** | On-chain metrics, protocol fundamentals | ✅ REST (paid tiers) | ✅ Numeric | 100+ assets | Historical depth good; cost-effective |
| **Glassnode** | On-chain analytics (BTC, ETH focus) | ✅ REST (paid) | ✅ Numeric | BTC, ETH | Best-in-class; high cost |
| **CoinDesk Data** | Market data, sentiment | ✅ REST | ✅ Numeric | Broad | Good for validation |
| **Stablecoin Attestations** | USDT, USDC, DAI reserves | ⚠️ Manual releases | ✅ Numeric | Stablecoin-specific | Monthly/ad-hoc; needs calendar tracking |
| **Federal Reserve (FRED)** | Treasury yields, Fed rates | ✅ Public API | ✅ Numeric | US macro | Free; high-quality baseline |
| **BIS Research** | Cross-asset risk metrics | ❌ Reports only | ⚠️ Aggregate | Quarterly | Crawler for key metrics |

### Recommended Integration Strategy

#### Tier 1: Direct API Integration (Highest Priority)
- **DeFi Llama** — `api.llama.fi` + pro tier if budget allows
- **Token Terminal** — Free tier for exploration, paid for full history
- **Federal Reserve** — FRED API
- **Messari** — Starter tier ($300–600/month)

#### Tier 2: Crawler + Manual Ingestion (Medium Priority)
- Chainalysis public reports, quarterly indices
- Galaxy SeC DeFi risk scores (web scraper)
- Stablecoin attestation reports (PDF parsing)
- BIS/ESRB publications (extract key metrics quarterly)

#### Tier 3: Enterprise/Conditional (Lower Priority)
- TRM Labs (if budget permits)
- Elliptic (skip unless compliance-critical)
- Glassnode (DeFi Llama + Messari may suffice for MVP)

---

## Phase 2: Risk Signal Definition & Taxonomy

### ASRI Component Breakdown

The index comprises four weighted sub-indices:

| Sub-Index | Weight | Focus Area |
|-----------|--------|------------|
| A. Stablecoin Concentration & Treasury Exposure | **30%** | Reserve composition, peg stability |
| B. DeFi Liquidity & Composability Risk | **25%** | TVL concentration, leverage, smart contract risk |
| C. Cross-Market Interconnection & Contagion | **25%** | TradFi linkages, correlation shifts, bridge risk |
| D. Regulatory Arbitrage & Opacity | **20%** | Unregulated exposure, transparency scores |

---

### A. Stablecoin Concentration & Treasury Exposure Risk (30%)

**Signals to track:**
- Total stablecoin market cap (USDT, USDC, DAI) — DeFi Llama, CoinGecko
- Stablecoin TVL across chains — DeFi Llama TVL filters
- Reserve composition (USD, Treasuries, other) — Monthly attestation parsing
- Stablecoin reserve weight in T-bills — Attestations / public reporting
- Stablecoin 30-day volatility (peg stability) — Price feeds + Messari
- Reserve audit delays or controversies — News crawler / Twitter API

**Sub-index formula:**
```
StablecoinRisk_t =
    0.4 × (Stablecoin_TVL_t / Max_Historical_TVL) +
    0.3 × (Treasury_Weight / Total_Stablecoin_Reserves) +
    0.2 × (Reserve_Concentration_HHI) +
    0.1 × (30d_Peg_Volatility)
```

---

### B. DeFi Liquidity & Composability Risk (25%)

**Signals to track:**
- Total DeFi TVL — DeFi Llama (general + by category)
- Top-10 protocol concentration (HHI index) — DeFi Llama
- Liquidity pool depth (Uniswap, Curve, Aave) — Token Terminal, on-chain
- Flash loan activity & smart contract risk scores — Token Terminal
- Bridge liquidity & cross-chain volume — DeFi Llama bridges
- Leveraged position size (Aave, Compound, dYdX) — On-chain aggregation

**Sub-index formula:**
```
DeFiLiquidityRisk_t =
    0.35 × (Top10_Concentration) +
    0.25 × (TVL_Volatility_30d) +
    0.20 × (SmartContract_Risk_Score) +
    0.10 × (Flash_Loan_Volume_Spike) +
    0.10 × (Leverage_Ratio_Change)
```

---

### C. Cross-Market Interconnection & Contagion Risk (25%)

**Signals to track:**
- Bank crypto exposure (Fitch, OCC, ECB surveys)
- Tokenized RWA volume — RWA.xyz API, DeFi Llama RWA subset
- Stablecoin inflows to TradFi-connected entities — On-chain analysis
- BTC/ETH correlation with traditional assets — Messari, Glassnode
- Spot-derivatives basis (funding rates) — Token Terminal, CME data
- Cross-chain bridge volume & exploits — DeFi Llama, Chainalysis

**Sub-index formula:**
```
ContagionRisk_t =
    0.30 × (Tokenized_RWA_Growth_Rate) +
    0.25 × (Bank_Exposure_Score) +
    0.20 × (Stablecoin_TradFi_Linkage_Intensity) +
    0.15 × (Crypto_Equity_Correlation) +
    0.10 × (Bridge_Exploit_Frequency)
```

---

### D. Regulatory Arbitrage & Opacity Risk (20%)

**Signals to track:**
- Unregulated vs. regulated platform ratio — Manual tracking
- Multi-issuer stablecoin schemes — Regulatory filings
- Offshore custody concentration — Public disclosures
- Regulatory news sentiment — NLP on SEC, ESRB, FSB announcements
- Compliance violation flags — Chainalysis public data
- Opaque collateral chains (rehypothecation) — On-chain tracing

**Sub-index formula:**
```
ArbitrageOpacityRisk_t =
    0.25 × (Unregulated_Exposure) +
    0.25 × (Multi_Issuer_Risk) +
    0.20 × (Custody_Concentration) +
    0.15 × (Regulatory_Sentiment_Score) +
    0.15 × (Transparency_Score)
```

---

### Aggregate ASRI Calculation

```
ASRI_t =
    0.30 × StablecoinRisk_t +
    0.25 × DeFiLiquidityRisk_t +
    0.25 × ContagionRisk_t +
    0.20 × ArbitrageOpacityRisk_t

Normalized ASRI (0-100):
ASRI_Normalized_t = 100 × (ASRI_t - min(ASRI)) / (max(ASRI) - min(ASRI))
```

---

## Phase 3: Technical Architecture

### Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
├─────────────────────────────────────────────────────────────────┤
│  APIs: DeFi Llama, Messari, Token Terminal, FRED, etc.         │
│  Crawlers: Chainalysis, Galaxy SeC, BIS reports                │
│  Manual: Stablecoin attestations, regulatory filings           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              DATA INGESTION & NORMALIZATION LAYER               │
├─────────────────────────────────────────────────────────────────┤
│  • Fetch from APIs (hourly/daily depending on source)          │
│  • Parse crawled HTML/PDF (daily for reports)                  │
│  • Normalize units ($ amounts, ratios, percentages)            │
│  • Fill gaps (linear interpolation, forward-fill as needed)    │
│  • Validation: range checks, outlier flagging                  │
│  • Storage: PostgreSQL time-series tables                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│            SIGNAL COMPUTATION & SUB-INDEX LAYER                 │
├─────────────────────────────────────────────────────────────────┤
│  • Calculate 4 risk sub-indices (A, B, C, D)                   │
│  • Rolling window aggregation (daily, weekly, 30d)             │
│  • Scenario analysis & stress testing logic                    │
│  • Alert triggers (threshold monitoring)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               AGGREGATE ASRI & BACKTESTING                      │
├─────────────────────────────────────────────────────────────────┤
│  • Compute ASRI_t = weighted sum of sub-indices                │
│  • Backtest against known crises (2020-2025)                   │
│  • Calibrate weights via walk-forward optimization             │
│  • Generate performance metrics (Sharpe, max drawdown)         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│           REST API & DASHBOARD PUBLICATION LAYER                │
├─────────────────────────────────────────────────────────────────┤
│  • REST API: /asri/current, /asri/history, /asri/subindex      │
│  • Time-series endpoint: /asri/timeseries?start=&end=          │
│  • Web dashboard: Charts, alerts, methodology                  │
│  • Public (free tier) + Premium (historical, API key)          │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

#### Backend
| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Data Processing | Pandas, NumPy, Polars |
| API Framework | FastAPI |
| HTTP Client | httpx (async) |
| Database | PostgreSQL 14+ with TimescaleDB |
| Scheduling | APScheduler or Celery |

#### Data Fetch Libraries
- `defillama-py` — DeFi Llama wrapper
- `messari` — Messari SDK
- `fred` — Federal Reserve API
- `requests` + `BeautifulSoup`/`Selenium` — Web scraping

#### Frontend
| Component | Technology |
|-----------|------------|
| Framework | React (TypeScript) or Vue 3 |
| Charting | Plotly.js or Chart.js |
| Styling | Tailwind CSS, shadcn/ui |
| State | TanStack Query, Zustand |

#### Infrastructure
| Component | Technology |
|-----------|------------|
| Hosting | AWS EC2 or DigitalOcean |
| Database | AWS RDS or DO Managed Postgres |
| CDN | Cloudflare |
| Monitoring | Sentry, New Relic |

---

## Phase 4: Backtesting Strategy (2020–Present)

### Historical Data Collection

**Timeline:** Jan 1, 2020 – Dec 12, 2025 (daily frequency)

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| DeFi Llama limited historical depth | Query archive.org + manual records; synthetic forward-fill from Q2 2021 |
| TRM/Elliptic APIs not available historically | Use proxy: on-chain transaction volume (CoinMetrics), Chainalysis reports |
| Stablecoin attestations only monthly/quarterly | Linear interpolation; flag low-confidence periods |
| Fed/Treasury data abundant | Daily via FRED API |
| Smart contract risk scores not historical | Score protocols retrospectively (GitHub commits, audit history) |

### Backtesting Protocol

#### Step 1: Data Assembly
- Ingest available historical data from each API (2020-2025)
- Use proxy indicators for gaps (documented)
- Create master CSV: `date, source, value, confidence_score`

#### Step 2: Signal Calculation
- Recompute all 4 sub-indices for each day using data available on that day
- Document data lag assumptions (e.g., DeFi Llama TVL at 00:00 UTC → used t+1)

#### Step 3: ASRI Computation
- Apply weights (30/25/25/20) to sub-indices
- Generate daily time-series: 2020-01-01 to 2025-12-12

#### Step 4: Validation Against Known Events

**Crisis Periods to Validate:**
| Event | Date |
|-------|------|
| COVID Crash | Mar 2020 |
| Terra/Luna Collapse | May 2022 |
| Celsius/3AC Blow-up | Jun 2022 |
| FTX Collapse | Nov 2022 |
| SVB Contagion | Mar 2023 |

**Metrics to Compute:**
- **Precision:** % of ASRI peaks that preceded/coincided with crises
- **Recall:** % of actual crises with detectable ASRI signal rise
- **Max Drawdown:** Peak-to-trough decline during each crisis

#### Step 5: Weight Optimization
- Walk-forward analysis: train on 2020-2022, test on 2023-2025
- Optimize for Sharpe ratio or minimize regret
- Report sensitivity: weight shifts of ±5%

### Expected Output: Backtesting Report

```
ASRI Backtesting Summary (2020–2025)
────────────────────────────────────────────
Sharpe Ratio (trained, 2023-2025):     1.34
Max Drawdown (peak-to-trough):         -45% (May 2022)
True Positive Rate (crises detected):  83% (5/6 major crises)
False Positive Rate:                   ~2% (12 months of ~1,400)
Best Leading Indicator Combo:          Stablecoin (40%), DeFi TVL (35%), Correlation (25%)
  → Leads major crashes by 3-7 days on average
```

---

## Phase 5: Dashboard & API Design

### REST API Endpoints

#### `GET /asri/current`
```json
{
  "timestamp": "2025-12-12T23:43:00Z",
  "asri": 62.3,
  "asri_30d_avg": 59.1,
  "trend": "rising",
  "sub_indices": {
    "stablecoin_risk": 68.5,
    "defi_liquidity_risk": 54.2,
    "contagion_risk": 71.1,
    "arbitrage_opacity": 49.0
  },
  "alert_level": "elevated",
  "last_update": "2025-12-12T23:30:00Z"
}
```

#### `GET /asri/timeseries?start=2025-01-01&end=2025-12-12`
```json
{
  "data": [
    {"date": "2025-01-01", "asri": 45.2, "sub_indices": {...}},
    ...
  ],
  "metadata": {"points": 346, "frequency": "daily"}
}
```

#### `GET /asri/subindex/{name}`
- Params: `stablecoin_risk`, `defi_liquidity_risk`, `contagion_risk`, `arbitrage_opacity`
- Returns time-series for specified sub-index

#### `GET /asri/stress-test?scenario=treasury_spike&magnitude=50`
```json
{
  "base_asri": 62.3,
  "stressed_asri": 78.5,
  "delta_asri": 16.2,
  "affected_sub_indices": ["stablecoin_risk", "contagion_risk"]
}
```

#### `GET /asri/methodology`
Full documentation of ASRI construction, weights, data sources, backtesting results

#### `GET /asri/alerts?since=2025-12-10`
List of threshold crosses, explanations, historical precedents

---

### Web Dashboard Features

#### Main Layout
- **Header:** Current ASRI + 30-day trend line, Alert banner if >70
- **Main Grid:**
  - Large chart: ASRI time-series (12-month default, zoomable)
  - 4-panel grid: Each sub-index with sparkline + numeric value
  - Heatmap: Daily ASRI colored by alert level
- **Sidebar:**
  - Breakdown table: Weights, current values, % contribution
  - Recent alerts: Last 5 threshold events
  - Data freshness: Last update timestamp per source

#### Tools Tab
- Stress tester: Interactive sliders for scenario analysis
- Correlation matrix: ASRI vs. macro variables
- Alert threshold tuner: Custom alerts by user

#### Methodology Tab
- Full ASRI formula + weights
- Data sources + update frequency
- Backtesting performance metrics
- FAQ

---

## Phase 6: Implementation Roadmap (12-16 weeks)

### Sprint 1: Data Ingestion Foundation (Weeks 1–3)
- [ ] Set up PostgreSQL + TimescaleDB
- [ ] Implement API clients for DeFi Llama, Token Terminal, FRED
- [ ] Build web scrapers for Chainalysis, Galaxy SeC, attestation PDFs
- [ ] Deploy initial daily data ingestion pipeline

**Deliverable:** Daily data collection for all 4 sub-index inputs

---

### Sprint 2: Signal Computation & Historical Backfill (Weeks 4–6)
- [ ] Implement sub-index calculation logic (formulas A, B, C, D)
- [ ] Backfill historical data (2020-2025) with proxy indicators
- [ ] Validate data quality & gap-fill methodology
- [ ] Create backtesting harness

**Deliverable:** Complete ASRI time-series (2020-2025) with backtesting report

---

### Sprint 3: REST API & Backtesting Refinement (Weeks 7–9)
- [ ] Build FastAPI REST endpoints (current, timeseries, subindex, stress-test)
- [ ] Implement walk-forward weight optimization
- [ ] Validate ASRI against known crisis periods
- [ ] Document methodology & API

**Deliverable:** REST API + validated backtesting metrics

---

### Sprint 4: Frontend Dashboard (Weeks 10–12)
- [ ] Design & build React dashboard
- [ ] Implement real-time chart updates (WebSocket or polling)
- [ ] Build stress-test UI
- [ ] Deploy to staging

**Deliverable:** Public beta dashboard + API key management

---

### Sprint 5: Optimization & Launch (Weeks 13–16)
- [ ] Performance tuning (database indexing, API caching)
- [ ] Implement monitoring & alerting
- [ ] Security hardening (rate limits, CORS, auth for premium)
- [ ] Documentation + marketing
- [ ] Launch live

**Deliverable:** Full deployment

---

## Phase 7: Monetization & Sustainability

### Revenue Streams

#### Freemium SaaS Model (Recommended)
| Tier | Features | Price |
|------|----------|-------|
| **Free** | Daily ASRI, 1-year history, API (100 req/day) | $0 |
| **Pro** | Sub-indices, stress-testing, custom alerts, 5,000 req/day | $99/month |
| **Enterprise** | White-label, real-time updates, priority support | Custom |

#### Additional Revenue
- **API Licensing:** Integrate ASRI into trading terminals, risk dashboards
- **Research & Consulting:** Quarterly systemic risk reports for institutions

### Cost Structure (Annual Estimate)

| Item | Cost |
|------|------|
| Infrastructure (AWS RDS, EC2, CDN) | $15,000 |
| Data APIs (Messari, Token Terminal, premium tiers) | $8,000 |
| Development/Engineering (ongoing) | $60,000 |
| Compliance & Legal (regulatory reviews) | $10,000 |
| **Total Annual** | **~$93,000** |

> With 50 Pro subscribers @ $99/month = $59,400/year → **~64% cost recovery**

---

## Phase 8: Risk Mitigation & Open Questions

### Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Data API discontinuation | ASRI calculation breaks | Maintain 2-3 redundant sources per signal; build fallback logic |
| Historical data unavailability pre-2021 | Backtesting unreliable | Use proxy indicators; document confidence scores |
| Regulatory changes render weights obsolete | ASRI loses predictive power | Quarterly weight review; monitor FSB/ESRB publications |
| False positive alerts | User trust erodes | Implement alert confidence scores; A/B test weights |
| Systemic event outside model scope | Index fails to predict | Regular stress-test updates; incorporate new factors quarterly |

### Open Technical Questions

1. **Data Lag Assumptions:** If DeFi Llama updates at 00:00 UTC, use t-1 or t+1?
   - *Recommendation:* t-1 to avoid look-ahead bias

2. **Missing Signal Interpolation:** Linear interpolation, forward-fill, or flag as low-confidence?
   - *Recommendation:* Linear with `confidence_score = 0.5` for interpolated days

3. **Normalization Method:** Z-score (sensitive to tails) vs min-max (0-100)?
   - *Recommendation:* Min-max for interpretability, z-score for statistical rigor

4. **Update Frequency:** Hourly (responsive, noisy) vs daily (stable, slower)?
   - *Recommendation:* Start daily, upgrade to hourly if latency proves valuable

5. **Weight Optimization Scope:** Global vs regime-specific (bull/bear/crisis)?
   - *Recommendation:* Walk-forward rolling window to adapt

---

## MVP Launch Checklist

- [ ] Data pipelines running daily for all 4 sub-indices
- [ ] ASRI time-series backtested (2020-2025) with documented precision/recall
- [ ] REST API live: `/current`, `/timeseries`, `/subindex`, `/stress-test`
- [ ] Dashboard live with real-time ASRI, charts, alerts
- [ ] Methodology & API documentation published
- [ ] 50+ beta users providing feedback
- [ ] Monitoring & alerting infrastructure deployed
- [ ] Freemium SaaS model configured (free + Pro tiers)

---

**Launch Target:** End of Q1 2026 (12–16 weeks)

---

> *This plan provides unified, quantified visibility into fragmented systemic risks—solving the exact gap identified in existing research.*

---

**Document Version:** 1.0
**Last Updated:** December 2025
**Authors:** Resurrexi Labs
