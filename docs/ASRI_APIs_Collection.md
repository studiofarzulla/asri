# ASRI INDEX - COMPREHENSIVE API COLLECTION & INTEGRATION GUIDE
## All Data Sources for Systemic Risk Index

---

## TABLE OF CONTENTS
1. [Tier 1: Priority APIs (MVP Foundation)](#tier-1)
2. [Tier 2: Secondary APIs & Data Sources](#tier-2)
3. [Tier 3: Enterprise/Optional APIs](#tier-3)
4. [API Comparison Matrix](#matrix)
5. [Integration Code Examples](#code-examples)
6. [Rate Limits & Pricing Summary](#pricing)
7. [Data Collection Schedule](#schedule)

---

## TIER 1: PRIORITY APIs (MVP Foundation) {#tier-1}

### 1. DeFi Llama API

**Overview**: Industry-standard TVL aggregator tracking 1,000+ protocols across 100+ blockchains

**Base URL**: `https://api.llama.fi`  
**Pro API Base URL**: `https://pro-api.llama.fi`

**Key Endpoints**:

```
1. GET /protocols
   Description: List all protocols with TVL data
   Response: Array of protocols with TVL, changes, yields
   Example: https://api.llama.fi/protocols
   Rate Limit: Free tier unlimited, Pro tier: unlimited

2. GET /protocol/{protocol_name}
   Description: Detailed protocol data
   Path Param: protocol_name (e.g., "aave", "uniswap")
   Response: TVL by chain, historical data, fees
   Example: https://api.llama.fi/protocol/aave

3. GET /tvl/{protocol}
   Description: Historical TVL time-series
   Path Param: protocol name
   Response: Array of {date, tvl} pairs (daily)
   Example: https://api.llama.fi/tvl/uniswap

4. GET /charts/{protocol}
   Description: Charts data for frontend rendering
   Response: Historical TVL, volumes, etc.

5. GET /bridges
   Description: Bridge volume and liquidity data
   Response: All bridges with volumes, destinations
   Example: https://bridges.llama.fi/bridges

6. GET /bridge/{bridge_id}
   Description: Detailed bridge analytics
   Response: Daily deposits, withdrawals, transaction counts

7. GET /stablecoins
   Description: Stablecoin TVL across protocols
   Response: Stablecoin-specific metrics

8. GET /stablecoins/chains
   Description: Stablecoin TVL by chain
   Response: TVL breakdown by blockchain

9. GET /yields
   Description: DeFi yields across protocols
   Response: Current yields, historical APY data
```

**Authentication**: None required (free tier), API key for pro tier

**Rate Limits**:
- Free: 10 calls/second (unlimited monthly)
- Pro: Higher limits for commercial use

**Data Update Frequency**: Real-time (updates hourly)

**Historical Coverage**: 2021-Q2 to present (~4 years)

**Python Integration**:
```python
import requests
import pandas as pd

# Get all protocols
response = requests.get('https://api.llama.fi/protocols')
protocols = response.json()

# Get specific protocol TVL
protocol = 'aave'
tvl_data = requests.get(f'https://api.llama.fi/tvl/{protocol}').json()

# Convert to DataFrame
df = pd.DataFrame(tvl_data)
df['date'] = pd.to_datetime(df['date'], unit='s')
df = df.set_index('date')
```

**Documentation**: https://api-docs.defillama.com

**Cost**: Free (with limits), Pro tier pricing available

---

### 2. Token Terminal API

**Overview**: Standardized protocol economics and on-chain metrics

**Base URL**: `https://api.tokenterminal.com/v2`

**Authentication**: Bearer token (API key required)

**Key Endpoints**:

```
1. GET /financial-metrics
   Description: Protocol revenue, volume, active users
   Query Params:
     - metric (revenue, tvl, volume, active_users, etc.)
     - metric_type (aggregate, current_value)
     - interval (hourly, daily, weekly, monthly)
     - limit (max 10,000)
   Response: Time-series financial metrics

2. GET /projects
   Description: List all tracked projects
   Response: Project list with metadata

3. GET /projects/{project_id}/metrics
   Description: Metrics for specific project
   Query Params:
     - start_date (ISO 8601)
     - end_date (ISO 8601)
     - frequency (1h, 1d, 1w, 1mo)

4. GET /projects/{project_id}/fees
   Description: Protocol fee data
   Response: Fee type, amount, denominated asset

5. GET /projects/{project_id}/revenue
   Description: Daily revenue for protocol
   Query Params: start_date, end_date, frequency

6. GET /projects/{project_id}/tvl
   Description: Total Value Locked
   Response: Historical TVL time-series

7. GET /security/audits
   Description: Smart contract audit history
   Response: Audit firm, date, scope, findings
```

**Authentication Header**:
```
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

**Rate Limits**:
- Free tier: 100 requests/month
- Pro tier: 10,000 requests/month (~$400/year)
- Enterprise: Custom limits

**Data Update Frequency**: Daily

**Historical Coverage**: Protocol-dependent (1-5 years typically)

**Python Integration**:
```python
import requests
import json

API_KEY = "your_api_key_here"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Fetch revenue metrics
url = "https://api.tokenterminal.com/v2/financial-metrics"
params = {
    "metric": "revenue",
    "project_id": "aave",
    "interval": "daily",
    "limit": 365
}

response = requests.get(url, headers=headers, params=params)
data = response.json()

# Parse and store
for record in data['data']:
    print(f"{record['date']}: {record['revenue_usd']}")
```

**Documentation**: https://tokenterminal.com/docs/api

**Cost**: Freemium ($400–$10,000+/year depending on tier)

---

### 3. Messari API

**Overview**: Comprehensive on-chain metrics, asset fundamentals, and market data

**Base URL**: `https://data.messari.io/api/v1`

**Authentication**: API key in headers

**Key Endpoints**:

```
1. GET /assets
   Description: List all tracked assets
   Query Params:
     - with_profile (true/false)
     - limit (max 500)
     - page
   Response: Asset metadata

2. GET /assets/{asset_key}/metrics
   Description: On-chain metrics for asset
   Path Params: asset_key (e.g., bitcoin, ethereum, aave)
   Response: 50+ on-chain metrics (supply, flow, addresses, etc.)

3. GET /assets/{asset_key}/timeseries
   Description: Historical time-series data
   Query Params:
     - metric (metric_name)
     - start (ISO 8601)
     - end (ISO 8601)
     - interval (1d, 1w, 1mo)
     - limit (max 10,000)

4. GET /assets/{asset_key}/profile
   Description: Asset profile and metadata
   Response: Description, team, links, categories

5. GET /assets/{asset_key}/market-data
   Description: Price, volume, market cap
   Response: Current and historical price data

6. GET /assets/{asset_key}/risk-metrics
   Description: Liquidity-adjusted metrics
   Response: MVRV ratio, Sharpe ratio, leverage metrics

7. GET /assets/{asset_key}/governance
   Description: Governance token details
   Response: Holder distribution, voting power

8. GET /assets/{asset_key}/technicals
   Description: Technical analysis indicators
   Response: RSI, MACD, Bollinger Bands, etc.
```

**Authentication Header**:
```
X-Messari-API-Key: YOUR_API_KEY
```

**Rate Limits**:
- Starter: 50 requests/minute
- Pro: 500 requests/minute ($300–$600/month)
- Enterprise: Custom

**Data Update Frequency**: Real-time for prices, daily for metrics

**Historical Coverage**: Full history since launch (protocol-dependent)

**Python Integration**:
```python
import requests
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "your_messari_api_key"
headers = {"X-Messari-API-Key": API_KEY}

# Get Bitcoin metrics
asset_key = "bitcoin"
url = f"https://data.messari.io/api/v1/assets/{asset_key}/metrics"

response = requests.get(url, headers=headers)
metrics = response.json()['data']

print(f"Bitcoin Circulating Supply: {metrics['supply']['circulating']}")
print(f"Active Addresses: {metrics['on_chain']['active_addresses']}")

# Get historical data
start_date = (datetime.now() - timedelta(days=365)).isoformat()
end_date = datetime.now().isoformat()

timeseries_url = f"https://data.messari.io/api/v1/assets/{asset_key}/timeseries"
params = {
    "metric": "on_chain_transaction_volume",
    "start": start_date,
    "end": end_date,
    "interval": "1d"
}

response = requests.get(timeseries_url, headers=headers, params=params)
data = response.json()['data']['values']

df = pd.DataFrame(data, columns=['timestamp', 'value'])
df['date'] = pd.to_datetime(df['timestamp'], unit='s')
```

**Documentation**: https://docs.messari.io

**Cost**: Starter tier free ($300/month pro, enterprise custom)

---

### 4. Federal Reserve FRED API

**Overview**: US macroeconomic data (Treasury yields, Fed rates, inflation, etc.)

**Base URL**: `https://api.stlouisfed.org/fred`

**Authentication**: API key in query parameters

**Key Series for ASRI**:

```
Economic Indicators:
- FEDFUNDS: Federal Funds Effective Rate
- DFF: Effective Fed Funds Rate (real-time)
- DGS3MO: 3-Month Treasury Constant Maturity Rate
- DGS10: 10-Year Treasury Constant Maturity Rate
- DGS30: 30-Year Treasury Constant Maturity Rate
- T10Y3M: 10-Year minus 3-Month Treasury Spread
- CPIAUCSL: Consumer Price Index (CPI)
- UNRATE: Unemployment Rate
- TOTALSA: Total Non-Farm Employment
- BOGMBASE: Monetary Base
- WIMFSL: Weekly Imports of Merchandise (goods)
```

**Key Endpoints**:

```
1. GET /series/search
   Description: Search for series by text
   Query Params:
     - search_text (required)
     - limit (max 10,000)
   Response: Array of matching series

2. GET /series/{series_id}/observations
   Description: Retrieve observations for a series
   Path Params: series_id (e.g., FEDFUNDS, DGS3MO)
   Query Params:
     - units (lin, chg, ch1, pch, pch1, log)
     - frequency (d, w, bw, m, q, sa, a, wef, weth, wew, wemw)
     - sort_order (asc, desc)
     - observation_start (YYYY-MM-DD)
     - observation_end (YYYY-MM-DD)
     - limit (max 120,000)
   Response: Array of {date, value} pairs

3. GET /series/{series_id}
   Description: Get series metadata
   Response: Title, units, frequency, last update

4. GET /series/popular
   Description: Retrieve popular series
   Query Params:
     - limit
   Response: List of commonly used series
```

**Authentication**:
```
Query Parameter: api_key=YOUR_FRED_API_KEY
Example: https://api.stlouisfed.org/fred/series/FEDFUNDS/observations?api_key=XXX
```

**Rate Limits**: 120 requests per minute (generous for institutional use)

**Data Update Frequency**: Daily (most series updated 1-2 business days behind)

**Historical Coverage**: Depends on series (FEDFUNDS since 1954)

**Python Integration**:
```python
import requests
import pandas as pd

API_KEY = "your_fred_api_key"
BASE_URL = "https://api.stlouisfed.org/fred"

def fetch_fred_series(series_id, start_date, end_date):
    url = f"{BASE_URL}/series/{series_id}/observations"
    params = {
        "api_key": API_KEY,
        "observation_start": start_date,
        "observation_end": end_date,
        "units": "lin"
    }
    
    response = requests.get(url, params=params)
    data = response.json()['observations']
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    return df.set_index('date')

# Example: Fetch 3-month Treasury yield
yield_3m = fetch_fred_series("DGS3MO", "2020-01-01", "2025-12-12")
print(yield_3m.head())
```

**Documentation**: https://fred.stlouisfed.org/docs/api/

**Cost**: Free (no API key required, but limited to 120 requests/minute)

---

### 5. CoinGecko API

**Overview**: Free cryptocurrency price data, market caps, and on-chain metrics

**Base URL**: `https://api.coingecko.com/api/v3`

**Authentication**: Free tier (no key), Demo/Paid tiers (API key)

**Key Endpoints**:

```
1. GET /simple/price
   Description: Get current prices for cryptocurrencies
   Query Params:
     - ids (comma-separated: bitcoin,ethereum)
     - vs_currencies (usd, eur, gbp, etc.)
     - include_market_cap (true/false)
     - include_24hr_vol (true/false)
     - include_last_updated_at (true/false)
   Response: Current price data

2. GET /simple/token_price/{id}
   Description: Token prices by contract address
   Path Params: id (platform, e.g., ethereum)
   Query Params:
     - contract_addresses
     - vs_currencies

3. GET /coins/{id}
   Description: Detailed coin data
   Path Params: id (e.g., bitcoin, ethereum)
   Query Params:
     - localization (true/false)
     - tickers (true/false)
     - market_data (true/false)
     - community_data (true/false)
     - developer_data (true/false)

4. GET /coins/{id}/market_chart
   Description: Historical price and volume
   Query Params:
     - vs_currency (usd)
     - days (1, 7, 30, 365, max)
     - interval (daily)
   Response: OHLCV data

5. GET /coins/{id}/ohlc
   Description: OHLC candlestick data
   Query Params:
     - vs_currency
     - days (1, 7, 30, 90, 180, 365, max)

6. GET /exchanges
   Description: List cryptocurrency exchanges
   Query Params:
     - per_page
     - page

7. GET /global
   Description: Global cryptocurrency market data
   Response: Total market cap, Bitcoin dominance, Ethereum dominance
```

**Rate Limits**:
- Free (Demo): 10–30 calls/minute, 10,000 calls/month
- Starter: 50 calls/minute, 500,000 calls/month
- Pro: 500 calls/minute, unlimited monthly

**Data Update Frequency**: Real-time (prices), hourly (aggregates)

**Historical Coverage**: Full history since token launch

**Python Integration**:
```python
import requests
import pandas as pd

# Free tier (no API key)
url = "https://api.coingecko.com/api/v3/simple/price"
params = {
    "ids": "bitcoin,ethereum,uniswap",
    "vs_currencies": "usd",
    "include_market_cap": True,
    "include_24hr_vol": True
}

response = requests.get(url, params=params)
prices = response.json()

print(prices)

# Fetch historical data (1 year)
coin_id = "ethereum"
url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
params = {
    "vs_currency": "usd",
    "days": "365",
    "interval": "daily"
}

response = requests.get(url, params=params)
data = response.json()

df = pd.DataFrame({
    'date': pd.to_datetime([x[0] for x in data['prices']], unit='ms'),
    'price': [x[1] for x in data['prices']],
    'market_cap': [x[1] for x in data['market_caps']],
    'volume': [x[1] for x in data['volumes']]
})

print(df.head())
```

**Documentation**: https://www.coingecko.com/en/api

**Cost**: Free tier available; premium plans start at $129/month

---

### 6. Stablecoin Reserve Attestations (Manual Collection)

**Overview**: Monthly/quarterly attestation reports from stablecoin issuers (Tether, Circle, etc.)

**Sources**:

1. **Tether (USDT)**
   - Monthly Attestation: https://tether.to/en/transparency/
   - Report Type: PDF attestation
   - Data: USD reserves, Treasury holdings, other assets
   - Update Frequency: Monthly
   - Format: Manual PDF parsing required

2. **Circle (USDC)**
   - Monthly Attestation: https://www.circle.com/en/usdc/attestation
   - Report Type: PDF + Dashboard
   - Data: USD reserves, Treasury bills, cash
   - Update Frequency: Monthly
   - Format: JSON API available on some reports

3. **MakerDAO (DAI)**
   - Collateral Dashboard: https://makerdao.com/en/
   - Data: On-chain collateral composition
   - Update Frequency: Real-time (on-chain)
   - Format: Smart contract queries

**Data Collection Strategy**:

```python
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re

# Example: Scrape Tether attestation page
def collect_tether_attestation():
    url = "https://tether.to/en/transparency/"
    response = requests.get(url)
    
    # Parse and extract PDF links
    soup = BeautifulSoup(response.content, 'html.parser')
    attestations = []
    
    for link in soup.find_all('a', href=re.compile(r'.*\.pdf')):
        attestations.append({
            'url': link['href'],
            'date': link.text,
            'issuer': 'Tether'
        })
    
    return attestations

# Example: Parse USDC dashboard
def collect_usdc_reserves():
    url = "https://www.circle.com/en/usdc/attestation"
    # Circle provides structured data; manual extraction or API if available
    # Typical data:
    # {
    #   "date": "2025-12-01",
    #   "total_usdc_issued": 33500000000,
    #   "usd_cash": 15000000000,
    #   "treasury_bills": 18500000000
    # }
    pass

# Store in database
def store_attestation(issuer, date, reserves_data):
    # INSERT INTO stablecoin_attestations 
    # (issuer, date, total_reserves, usd_cash, treasury_bills)
    pass
```

**Update Schedule**: 1st of each month (manual trigger)

**Historical Coverage**: Last 12 attestations (typically)

**Data Quality**: High (officially audited)

---

## TIER 2: SECONDARY APIs & DATA SOURCES {#tier-2}

### 7. Chainalysis Investigations API

**Overview**: Blockchain risk screening, entity mapping, transaction tracing

**Base URL**: `https://api.chainalysis.com/api/kyt`

**Authentication**: Bearer token (enterprise only)

**Key Endpoints**:

```
1. POST /addresses/{address}
   Description: Screen individual address
   Path Param: address (BTC, ETH, etc.)
   Request Body: {network, asset}
   Response: Risk level, linked entities, transaction history

2. POST /transactions/{network}/{tx_hash}
   Description: Analyze specific transaction
   Response: Input/output details, associated risks

3. GET /alerts
   Description: Retrieve security alerts
   Query Params:
     - created_after
     - created_before
     - severity_level
   Response: Array of alerts with risk assessment

4. POST /entities/search
   Description: Search entity database
   Request Body: {address, network}
   Response: Entity name, category, risk profile
```

**Rate Limits**: Custom (enterprise only)

**Data Update Frequency**: Real-time

**Cost**: Enterprise pricing (custom quote, typically $50k+/year)

**Access**: Requires institutional relationship

**Status for ASRI**: Consider for Phase 2; use proxy indicators in MVP

---

### 8. TRM Labs BLOCKINT API

**Overview**: Transaction monitoring, entity intelligence, AML screening

**Base URL**: `https://api.trmlabs.com/api/v1`

**Authentication**: API key (enterprise only)

**Key Features**:

```
- Real-time transaction monitoring
- Entity enrichment (80+ blockchains)
- Risk scoring
- Sanctions screening
- AML/CFT compliance checks
- Wallet analysis
```

**Cost**: Enterprise pricing (custom quote)

**Access**: Limited to institutional clients with onboarding

**Status for ASRI**: For Phase 2+; not for MVP due to cost/access

---

### 9. Glassnode API

**Overview**: Enterprise on-chain analytics (primarily BTC, ETH)

**Base URL**: `https://api.glassnode.com/v1`

**Authentication**: API key (professional tier+ required)

**Key Metrics**:

```
Network Metrics:
- Active addresses
- Transaction count
- Transaction volume
- Transaction fees

Supply Metrics:
- Circulating supply
- Locked supply
- Staking amount

Holder Metrics:
- Whale transactions
- Exchange flows
- In/out of exchange

Valuation Metrics:
- Realized price
- MVRV ratio
- NVT ratio
```

**Rate Limits**: Professional plan required ($400–$2,000/month)

**Data Update Frequency**: Daily/hourly depending on metric

**Cost**: $400/month (professional), $2,000/month (institutional)

**Status for ASRI**: Consider for BTC/ETH specific metrics in Phase 2

---

### 10. RWA.xyz API

**Overview**: Tokenized real-world assets tracking and analytics

**Base URL**: `https://api.rwa.xyz` (if available)

**Authentication**: API key for premium tiers

**Data Available**:

```
- Tokenized asset volumes
- Protocol rankings
- Yield aggregation
- Asset class breakdown (Treasuries, bonds, credit, real estate)
```

**Current Status**: Primarily dashboard-based; API status unclear

**Recommendation**: Monitor for API availability; use web scraping as interim

---

### 11. News Feeds & Regulatory Announcements (NLP Sentiment)

**Sources**:

1. **RSS Feeds**:
   - ECB announcements: https://www.ecb.europa.eu/feeds/
   - Federal Reserve: https://www.federalreserve.gov/feeds/
   - FSB: https://www.fsb.org/news/
   - SEC: https://www.sec.gov/cgi-bin/browse-edgar

2. **Twitter/Social Media API**:
   - Search for regulatory keywords (@SEC_News, @federalreserve, @ECB)
   - Sentiment analysis on mentions

3. **News Aggregation APIs**:
   - NewsAPI: https://newsapi.org/ (crypto news)
   - CryptoPanic: https://cryptopanic.com/api (crypto-specific)

**Python Implementation**:

```python
import feedparser
from textblob import TextBlob
import re

def fetch_regulatory_news():
    feeds = [
        "https://www.ecb.europa.eu/feeds/",
        "https://www.federalreserve.gov/feeds/"
    ]
    
    news = []
    for feed_url in feeds:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:10]:  # Last 10 entries
            sentiment = TextBlob(entry.summary).sentiment.polarity
            news.append({
                'source': feed_url,
                'title': entry.title,
                'date': entry.published_parsed,
                'sentiment': sentiment,  # -1 to +1
                'text': entry.summary
            })
    
    return news

regulatory_sentiment = fetch_regulatory_news()
```

---

## TIER 3: ENTERPRISE/OPTIONAL APIs {#tier-3}

### 12. Elliptic API (Compliance)
- **Status**: Enterprise only, high cost
- **Use Case**: Entity risk scoring, compliance checks
- **Recommendation**: Defer to Phase 2

### 13. Sentora/IntoTheBlock DeFi Analytics
- **Status**: Premium subscription
- **Use Case**: DeFi-specific protocol analytics
- **Cost**: $1,000+/month

---

## API COMPARISON MATRIX {#matrix}

| Source | Type | Cost | Data Depth | Update Freq | Coverage | Priority |
|--------|------|------|-----------|-------------|----------|----------|
| **DeFi Llama** | TVL Aggregator | Free | High | Hourly | 1000+ protocols | T1 ✓ |
| **Token Terminal** | Metrics | $400/yr | Very High | Daily | 200+ protocols | T1 ✓ |
| **Messari** | On-chain | $300/mo | Very High | Real-time | 100+ assets | T1 ✓ |
| **FRED** | Macro | Free | High | Daily | US economy | T1 ✓ |
| **CoinGecko** | Price Data | Free | Medium | Real-time | 10M+ coins | T1 ✓ |
| **Attestations** | Manual | Free | Very High | Monthly | 10+ stablecoins | T1 ✓ |
| **Chainalysis** | KYT | Enterprise | Very High | Real-time | All chains | T2 ⏳ |
| **TRM Labs** | Transaction Mon. | Enterprise | Very High | Real-time | 90+ chains | T2 ⏳ |
| **Glassnode** | On-chain (BTC/ETH) | $400+/mo | Very High | Daily | BTC, ETH focus | T2 ⏳ |
| **RWA.xyz** | RWA Data | TBD | High | Daily | Tokenized RWAs | T2 ⏳ |

---

## INTEGRATION CODE EXAMPLES {#code-examples}

### Example 1: Daily ASRI Data Collection Pipeline

```python
"""
ASRI Daily Data Ingestion Script
Runs at 22:00 UTC daily via APScheduler
"""

import requests
import pandas as pd
from datetime import datetime
import logging
from sqlalchemy import create_engine
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)

class ASRIDataCollector:
    def __init__(self):
        self.defillama_base = "https://api.llama.fi"
        self.messari_base = "https://data.messari.io/api/v1"
        self.fred_base = "https://api.stlouisfed.org/fred"
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        
        self.messari_key = os.getenv("MESSARI_API_KEY")
        self.fred_key = os.getenv("FRED_API_KEY")
        
    def collect_tvl_data(self):
        """Fetch DeFi TVL metrics"""
        logger.info("Collecting DeFi TVL data...")
        
        try:
            # Get all protocols
            response = requests.get(f"{self.defillama_base}/protocols")
            data = response.json()
            
            tvl_records = []
            for protocol in data:
                tvl_records.append({
                    'date': datetime.utcnow(),
                    'protocol': protocol['name'],
                    'tvl': protocol.get('tvl', 0),
                    'tvl_change_24h': protocol.get('change_1d', 0),
                    'chains': ','.join(protocol.get('chains', [])),
                    'category': protocol.get('category', 'Other')
                })
            
            df = pd.DataFrame(tvl_records)
            df.to_sql('defi_tvl_daily', engine, if_exists='append', index=False)
            logger.info(f"✓ Stored {len(df)} protocol TVL records")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Error collecting TVL data: {e}")
            return None
    
    def collect_stablecoin_data(self):
        """Fetch stablecoin metrics"""
        logger.info("Collecting stablecoin data...")
        
        try:
            # Get stablecoin TVL from DeFi Llama
            response = requests.get(f"{self.defillama_base}/stablecoins")
            stablecoin_data = response.json()
            
            stablecoin_records = []
            for sc in stablecoin_data.get('stablecoins', []):
                stablecoin_records.append({
                    'date': datetime.utcnow(),
                    'name': sc['name'],
                    'symbol': sc['symbol'],
                    'tvl': sc.get('tvl', 0),
                    'tvl_change_24h': sc.get('change_1d', 0)
                })
            
            df = pd.DataFrame(stablecoin_records)
            df.to_sql('stablecoin_metrics_daily', engine, if_exists='append', index=False)
            logger.info(f"✓ Stored {len(df)} stablecoin records")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Error collecting stablecoin data: {e}")
            return None
    
    def collect_treasury_yields(self):
        """Fetch US Treasury yields from FRED"""
        logger.info("Collecting Treasury yield data...")
        
        try:
            yield_series = ['DGS3MO', 'DGS10', 'DGS30', 'T10Y3M', 'DFF']
            
            for series_id in yield_series:
                url = f"{self.fred_base}/series/{series_id}/observations"
                params = {
                    'api_key': self.fred_key,
                    'limit': 1  # Just get latest
                }
                
                response = requests.get(url, params=params)
                obs = response.json()['observations'][0]
                
                record = {
                    'date': datetime.utcnow(),
                    'series_id': series_id,
                    'value': float(obs['value']),
                    'fred_date': obs['date']
                }
                
                pd.DataFrame([record]).to_sql(
                    'treasury_yields_daily',
                    engine,
                    if_exists='append',
                    index=False
                )
            
            logger.info(f"✓ Stored {len(yield_series)} Treasury yield series")
            
        except Exception as e:
            logger.error(f"✗ Error collecting Treasury data: {e}")
    
    def collect_crypto_metrics(self):
        """Fetch Bitcoin/Ethereum metrics"""
        logger.info("Collecting crypto price metrics...")
        
        try:
            assets = {'bitcoin': 'BTC', 'ethereum': 'ETH'}
            
            for asset_id, symbol in assets.items():
                headers = {'X-Messari-API-Key': self.messari_key}
                url = f"{self.messari_base}/assets/{asset_id}/metrics"
                
                response = requests.get(url, headers=headers)
                metrics = response.json()['data']
                
                record = {
                    'date': datetime.utcnow(),
                    'symbol': symbol,
                    'price_usd': metrics['market_data']['price_usd'],
                    'market_cap': metrics['market_data']['market_cap_usd'],
                    'active_addresses_1d': metrics['on_chain']['active_addresses']['v1']['one_day_active_count'],
                    'transaction_volume_24h': metrics['on_chain']['transaction_volume']['v1']['one_day_usd']
                }
                
                pd.DataFrame([record]).to_sql(
                    'crypto_metrics_daily',
                    engine,
                    if_exists='append',
                    index=False
                )
            
            logger.info(f"✓ Stored crypto metrics for {len(assets)} assets")
            
        except Exception as e:
            logger.error(f"✗ Error collecting crypto data: {e}")
    
    def run_daily_collection(self):
        """Execute complete daily data collection"""
        logger.info("=" * 60)
        logger.info(f"Starting ASRI daily data collection at {datetime.utcnow()}")
        logger.info("=" * 60)
        
        self.collect_tvl_data()
        self.collect_stablecoin_data()
        self.collect_treasury_yields()
        self.collect_crypto_metrics()
        
        logger.info("=" * 60)
        logger.info("Daily data collection complete")
        logger.info("=" * 60)

# APScheduler configuration
from apscheduler.schedulers.background import BackgroundScheduler

if __name__ == "__main__":
    collector = ASRIDataCollector()
    
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        collector.run_daily_collection,
        'cron',
        hour=22,
        minute=0,
        timezone='UTC'
    )
    scheduler.start()
    
    # Keep scheduler running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.shutdown()
```

### Example 2: Multi-Source Data Normalization

```python
"""
Data normalization layer
Standardizes data from different sources to common schema
"""

import pandas as pd
import numpy as np
from datetime import datetime

class DataNormalizer:
    @staticmethod
    def normalize_tvl(raw_tvl):
        """
        Input: {protocol: str, tvl: float, chains: list}
        Output: normalized 0-1 scale vs historical max
        """
        max_tvl = raw_tvl['protocol_history']['max_tvl']
        normalized = raw_tvl['tvl'] / max_tvl if max_tvl > 0 else 0
        return min(normalized, 1.0)  # Cap at 1.0
    
    @staticmethod
    def normalize_yields(treasury_yields):
        """
        Input: {series_id: str, value: float, date: datetime}
        Output: {basis_point_change_from_avg: float}
        """
        # Calculate 30-day rolling average
        # Use as baseline (value around 0 = stable, value > 0 = rising, < 0 = falling)
        pass
    
    @staticmethod
    def normalize_active_addresses(on_chain_metrics):
        """
        Input: on-chain metric value
        Output: % change from 30-day avg
        """
        pass
    
    @staticmethod
    def confidence_score(data_quality_metadata):
        """
        Input: {source: str, lag_hours: int, is_interpolated: bool}
        Output: confidence_score 0.0-1.0
        """
        base_confidence = 1.0
        
        # Deduct for data lag
        if data_quality_metadata['lag_hours'] > 24:
            base_confidence -= 0.2
        elif data_quality_metadata['lag_hours'] > 12:
            base_confidence -= 0.1
        
        # Deduct for interpolation
        if data_quality_metadata['is_interpolated']:
            base_confidence -= 0.15
        
        return max(base_confidence, 0.0)
```

---

## RATE LIMITS & PRICING SUMMARY {#pricing}

### Free Tier APIs
| Service | Rate Limit | Monthly Requests | Cost |
|---------|-----------|------------------|------|
| DeFi Llama | 10 req/sec | Unlimited | Free |
| FRED | 120 req/min | Unlimited | Free |
| CoinGecko (Demo) | 30 req/min | 10,000 | Free |

### Paid Tier APIs (Recommended for MVP)
| Service | Tier | Cost/Month | Rate Limit | Monthly Cap |
|---------|------|-----------|-----------|------------|
| Messari | Starter | $25 | 50 req/min | 50,000 |
| Token Terminal | Pro | $33 | 100 req/min | 10,000 |
| Glassnode | Professional | $400 | 500 req/min | Unlimited |
| CoinGecko | Lite | $400 | 500 req/min | 500,000 |

### **Total Estimated Monthly Cost (MVP)**:
- **Free tier APIs**: $0
- **Paid tier APIs**: $25 + $33 + $0 (or $400 if Glassnode) = **$58/month minimum, $433/month with Glassnode**
- **Annual**: ~$700–$5,000 depending on tier selection

---

## DATA COLLECTION SCHEDULE {#schedule}

| Source | Frequency | Best Time (UTC) | Retry Logic |
|--------|-----------|-----------------|-------------|
| DeFi Llama TVL | Hourly | Every hour | 3 retries, 60s backoff |
| Token Terminal | Daily | 06:00 UTC | 2 retries, 5min backoff |
| Messari Metrics | Daily | 08:00 UTC | 2 retries, 5min backoff |
| FRED Data | Daily | 16:00 UTC | 1 retry, 10min backoff (data released afternoon ET) |
| CoinGecko Prices | Hourly | Every hour | 3 retries, 60s backoff |
| Stablecoin Attestations | Monthly | 1st day 10:00 UTC | Manual trigger + automatic check |
| Regulatory News | 2x daily | 06:00 & 18:00 UTC | 2 retries, 5min backoff |

---

## INTEGRATION CHECKLIST

- [ ] Set up API keys for all Tier 1 sources
- [ ] Test endpoints for rate limits and response format
- [ ] Build database schema for raw data storage
- [ ] Implement retry logic and circuit breakers
- [ ] Set up error logging and monitoring
- [ ] Create data validation/quality checks
- [ ] Implement normalization layer
- [ ] Schedule daily ingestion jobs
- [ ] Create backfill scripts for 2020-2025 historical data
- [ ] Validate data against known crisis periods

---

**Document Version**: 1.0  
**Last Updated**: December 13, 2025  
**Status**: Ready for Implementation