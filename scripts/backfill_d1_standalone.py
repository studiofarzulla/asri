#!/usr/bin/env python3
"""
ASRI D1 Backfill - Standalone Version

Calculates historical ASRI values and pushes directly to Cloudflare D1.
Self-contained with no local ASRI package dependencies.

Requirements: pip install httpx python-dotenv

Usage:
    python scripts/backfill_d1_standalone.py --check
    python scripts/backfill_d1_standalone.py --start 2021-01-01 --end 2021-12-31 --dry-run
    python scripts/backfill_d1_standalone.py --start 2021-01-01 --end 2021-12-31
"""

import argparse
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

# =============================================================================
# Configuration
# =============================================================================

# Cloudflare credentials - load from env or hardcode
CF_EMAIL = "contact@farzulla.com"
CF_ACCOUNT_ID = "917877c07fa2e1c9f223db31d3fc52d6"
D1_DATABASE_ID = "f8c08c15-a596-4d7f-8c00-0f7592da26f4"

# Load API key from ~/.env.cloudflare
CF_API_KEY = None
cf_env = Path.home() / ".env.cloudflare"
if cf_env.exists():
    with open(cf_env) as f:
        for line in f:
            if line.startswith("export CLOUDFLARE_API_KEY="):
                CF_API_KEY = line.split("=", 1)[1].strip().strip('"')

# Load FRED API key from project .env
FRED_API_KEY = None
project_env = Path(__file__).parent.parent / ".env"
if project_env.exists():
    with open(project_env) as f:
        for line in f:
            if line.startswith("FRED_API_KEY="):
                FRED_API_KEY = line.split("=", 1)[1].strip()

# ASRI Weights (from paper)
ASRI_WEIGHTS = {
    "stablecoin_risk": 0.30,
    "defi_liquidity_risk": 0.25,
    "contagion_risk": 0.25,
    "arbitrage_opacity": 0.20,
}

# Sub-index weights
STABLECOIN_WEIGHTS = {
    'tvl_ratio': 0.4,
    'treasury_stress': 0.3,
    'concentration_hhi': 0.2,
    'peg_volatility': 0.1,
}
DEFI_WEIGHTS = {
    'top10_concentration': 0.35,
    'tvl_volatility': 0.25,
    'smart_contract_risk': 0.20,
    'flash_loan_proxy': 0.10,
    'leverage_change': 0.10,
}
CONTAGION_WEIGHTS = {
    'rwa_growth_rate': 0.30,
    'bank_exposure': 0.25,
    'tradfi_linkage': 0.20,
    'crypto_equity_correlation': 0.15,
    'bridge_exploit_frequency': 0.10,
}
ARBITRAGE_WEIGHTS = {
    'unregulated_exposure': 0.25,
    'multi_issuer_risk': 0.25,
    'custody_concentration': 0.20,
    'regulatory_sentiment': 0.15,
    'transparency_score': 0.15,
}

# Major stablecoins to track
MAJOR_STABLES = {
    1: "USDT", 2: "USDC", 3: "DAI", 4: "BUSD", 5: "FRAX",
    6: "TUSD", 7: "USDP", 8: "GUSD", 9: "LUSD", 10: "sUSD",
    11: "UST", 12: "MIM",
}


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_to_100(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to 0-100 scale."""
    if max_val == min_val:
        return 50.0
    normalized = (value - min_val) / (max_val - min_val) * 100
    return max(0, min(100, normalized))


def calculate_hhi(values: list[float]) -> float:
    """Calculate Herfindahl-Hirschman Index."""
    if not values:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    shares = [v / total for v in values]
    return sum(s ** 2 for s in shares) * 10000


def normalize_hhi_to_risk(hhi: float) -> float:
    """Convert HHI to risk score (0-100)."""
    # HHI ranges from 0 to 10000
    # Above 2500 is highly concentrated
    return normalize_to_100(hhi, 0, 5000)


def determine_alert_level(asri: float) -> str:
    """Determine alert level from ASRI value."""
    if asri < 30:
        return "low"
    elif asri < 50:
        return "moderate"
    elif asri < 70:
        return "elevated"
    return "critical"


def compute_weighted_asri(
    stablecoin_risk: float,
    defi_liquidity_risk: float,
    contagion_risk: float,
    arbitrage_opacity: float,
) -> float:
    """Compute ASRI from stored sub-index columns using canonical weights."""
    asri = (
        stablecoin_risk * ASRI_WEIGHTS["stablecoin_risk"]
        + defi_liquidity_risk * ASRI_WEIGHTS["defi_liquidity_risk"]
        + contagion_risk * ASRI_WEIGHTS["contagion_risk"]
        + arbitrage_opacity * ASRI_WEIGHTS["arbitrage_opacity"]
    )
    return round(asri, 1)


# =============================================================================
# Cloudflare D1 Client
# =============================================================================

class D1Client:
    """Cloudflare D1 client for direct database operations."""

    def __init__(self):
        if not CF_API_KEY:
            raise ValueError("CLOUDFLARE_API_KEY not found in ~/.env.cloudflare")

        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_DATABASE_ID}"
        self.headers = {
            "X-Auth-Email": CF_EMAIL,
            "X-Auth-Key": CF_API_KEY,
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(timeout=60.0)

    async def close(self):
        await self.client.aclose()

    async def query(self, sql: str, params: list = None) -> dict:
        """Execute a SQL query against D1."""
        payload = {"sql": sql}
        if params:
            payload["params"] = params

        resp = await self.client.post(
            f"{self.base_url}/query",
            headers=self.headers,
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    async def get_date_range(self) -> dict:
        """Get the current date range in D1."""
        result = await self.query(
            "SELECT MIN(date) as earliest, MAX(date) as latest, COUNT(*) as total FROM asri_daily"
        )
        return result["result"][0]["results"][0]

    async def check_date_exists(self, date: str) -> bool:
        """Check if a date already exists in D1."""
        result = await self.query(
            "SELECT 1 FROM asri_daily WHERE date = ? LIMIT 1",
            [date]
        )
        return len(result["result"][0]["results"]) > 0

    async def insert_record(self, record: dict) -> bool:
        """Insert a single ASRI record into D1."""
        sql = """
            INSERT INTO asri_daily
            (date, asri, asri_30d_avg, trend, alert_level,
             stablecoin_risk, defi_liquidity_risk, contagion_risk, arbitrage_opacity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = [
            record["date"],
            record["asri"],
            record.get("asri_30d_avg", record["asri"]),
            record.get("trend", "stable"),
            record["alert_level"],
            record["stablecoin_risk"],
            record["defi_liquidity_risk"],
            record["contagion_risk"],
            record["arbitrage_opacity"],
        ]

        try:
            await self.query(sql, params)
            return True
        except Exception as e:
            print(f"  Error inserting {record['date']}: {e}")
            return False


# =============================================================================
# Historical Data Fetcher
# =============================================================================

@dataclass
class HistoricalSnapshot:
    """All data needed for ASRI calculation at a specific date."""
    date: datetime
    current_tvl: float
    max_historical_tvl: float
    tvl_30d_history: list[float]
    stablecoin_market_caps: dict[str, float]
    total_stablecoin_supply: float
    treasury_10y_rate: float
    vix: float
    yield_curve_spread: float
    btc_prices: list[float]
    sp500_prices: list[float]
    num_protocols: int
    num_bridges: int


class HistoricalDataFetcher:
    """Fetches historical data from DeFi Llama, FRED, etc."""

    DEFILLAMA_BASE = "https://api.llama.fi"
    DEFILLAMA_STABLES = "https://stablecoins.llama.fi"
    DEFILLAMA_COINS = "https://coins.llama.fi"
    FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
        self._tvl_cache: list[dict] | None = None
        self._stablecoin_cache: dict[int, list[dict]] = {}
        self._btc_cache: list[dict] | None = None
        self._fred_cache: dict[str, list[dict]] = {}
        self._sp500_cache: list[dict] | None = None

    async def close(self):
        await self.client.aclose()

    async def _ensure_tvl_cache(self):
        """Load and cache full TVL history."""
        if self._tvl_cache is None:
            print("    Fetching full TVL history...", end=" ", flush=True)
            resp = await self.client.get(f"{self.DEFILLAMA_BASE}/v2/historicalChainTvl")
            resp.raise_for_status()
            self._tvl_cache = resp.json()
            print(f"({len(self._tvl_cache)} points)")

    async def _ensure_stablecoin_cache(self, stablecoin_id: int):
        """Load and cache stablecoin history."""
        if stablecoin_id not in self._stablecoin_cache:
            resp = await self.client.get(
                f"{self.DEFILLAMA_STABLES}/stablecoincharts/all?stablecoin={stablecoin_id}"
            )
            if resp.status_code == 200:
                self._stablecoin_cache[stablecoin_id] = resp.json()
            else:
                self._stablecoin_cache[stablecoin_id] = []

    async def _ensure_btc_cache(self):
        """Load and cache BTC price history."""
        if self._btc_cache is None:
            print("    Fetching BTC price history...", end=" ", flush=True)
            all_prices = []

            for year in [2021, 2022, 2023, 2024]:
                try:
                    start_ts = int(datetime(year, 1, 1).timestamp())
                    resp = await self.client.get(
                        f"{self.DEFILLAMA_COINS}/chart/coingecko:bitcoin?start={start_ts}&span=365",
                        timeout=30.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if 'coins' in data and 'coingecko:bitcoin' in data['coins']:
                            prices = data['coins']['coingecko:bitcoin'].get('prices', [])
                            all_prices.extend(prices)
                except Exception:
                    pass

            if all_prices:
                all_prices.sort(key=lambda x: x['timestamp'])
                seen = set()
                self._btc_cache = []
                for p in all_prices:
                    ts = p['timestamp']
                    if ts not in seen:
                        seen.add(ts)
                        self._btc_cache.append(p)
            else:
                self._btc_cache = []

            print(f"({len(self._btc_cache)} points)")

    async def _ensure_fred_cache(self, series_id: str):
        """Load and cache FRED series history."""
        if series_id not in self._fred_cache:
            params = {
                "series_id": series_id,
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "observation_start": "2020-01-01",  # Extra buffer
            }

            try:
                resp = await self.client.get(self.FRED_BASE, params=params, timeout=30.0)
                resp.raise_for_status()
                data = resp.json()
                self._fred_cache[series_id] = data.get("observations", [])
            except Exception:
                self._fred_cache[series_id] = []

    async def _ensure_sp500_cache(self):
        """Load and cache S&P 500 history."""
        if self._sp500_cache is None:
            print("    Fetching S&P500 history...", end=" ", flush=True)
            await self._ensure_fred_cache("SP500")
            self._sp500_cache = self._fred_cache.get("SP500", [])
            print(f"({len(self._sp500_cache)} points)")

    def _find_closest_tvl(self, target_date: datetime) -> tuple[float, datetime]:
        """Find TVL closest to target date."""
        if not self._tvl_cache:
            return 0.0, target_date
        target_ts = target_date.timestamp()
        closest = min(self._tvl_cache, key=lambda x: abs(x['date'] - target_ts))
        return closest['tvl'], datetime.fromtimestamp(closest['date'])

    def _get_tvl_range(self, end_date: datetime, days: int) -> list[float]:
        """Get TVL values for a date range."""
        if not self._tvl_cache:
            return [0.0]
        end_ts = end_date.timestamp()
        start_ts = (end_date - timedelta(days=days)).timestamp()
        values = [p['tvl'] for p in self._tvl_cache if start_ts <= p['date'] <= end_ts]
        return values if values else [0.0]

    def _get_max_tvl_before(self, date: datetime) -> float:
        """Get max TVL up to a specific date."""
        if not self._tvl_cache:
            return 0.0
        target_ts = date.timestamp()
        values = [p['tvl'] for p in self._tvl_cache if p['date'] <= target_ts]
        return max(values) if values else 0.0

    async def _fetch_stablecoin_at_date(self, stablecoin_id: int, target_date: datetime) -> float:
        """Get stablecoin market cap at a specific date."""
        await self._ensure_stablecoin_cache(stablecoin_id)
        data = self._stablecoin_cache.get(stablecoin_id, [])

        if not data:
            return 0.0

        target_ts = target_date.timestamp()
        closest = None
        min_diff = float('inf')

        for point in data:
            ts = int(point.get('date', 0))
            diff = abs(ts - target_ts)
            if diff < min_diff:
                min_diff = diff
                closest = point

        if closest:
            circulating = closest.get('totalCirculating', {}).get('peggedUSD', 0)
            return float(circulating) if circulating else 0.0

        return 0.0

    def _get_fred_value_at_date(self, series_id: str, target_date: datetime) -> float | None:
        """Get FRED value from cache for a specific date."""
        observations = self._fred_cache.get(series_id, [])
        if not observations:
            return None

        target_str = target_date.strftime("%Y-%m-%d")
        closest_value = None

        for obs in observations:
            obs_date = obs.get("date", "")
            if obs_date <= target_str:
                value = obs.get("value")
                if value not in [".", None, ""]:
                    closest_value = float(value)
            else:
                break

        return closest_value

    def _get_price_range(self, cache: list[dict], end_date: datetime, days: int, key: str = 'price') -> list[float]:
        """Get prices for a date range from cache."""
        if not cache:
            return []
        end_ts = end_date.timestamp()
        start_ts = (end_date - timedelta(days=days)).timestamp()

        if 'timestamp' in cache[0]:
            return [p[key] for p in cache if start_ts <= p['timestamp'] <= end_ts]
        else:
            # FRED format
            end_str = end_date.strftime("%Y-%m-%d")
            start_str = (end_date - timedelta(days=days)).strftime("%Y-%m-%d")
            prices = []
            for obs in cache:
                obs_date = obs.get("date", "")
                if start_str <= obs_date <= end_str:
                    value = obs.get("value")
                    if value not in [".", None, ""]:
                        prices.append(float(value))
            return prices

    async def fetch_snapshot(self, target_date: datetime) -> HistoricalSnapshot:
        """Fetch all data needed for ASRI calculation at a specific date."""
        # TVL Data
        await self._ensure_tvl_cache()
        current_tvl, _ = self._find_closest_tvl(target_date)
        max_historical_tvl = self._get_max_tvl_before(target_date)
        tvl_30d_history = self._get_tvl_range(target_date, 30)

        # Stablecoin Data
        print("    Fetching stablecoins...", end=" ", flush=True)
        stablecoin_caps: dict[str, float] = {}
        for stable_id, symbol in MAJOR_STABLES.items():
            cap = await self._fetch_stablecoin_at_date(stable_id, target_date)
            if cap > 0:
                stablecoin_caps[symbol] = cap
        total_stablecoin = sum(stablecoin_caps.values())
        print(f"({len(stablecoin_caps)} found)")

        # FRED Data
        print("    Fetching FRED data...", end=" ", flush=True)
        await self._ensure_fred_cache("DGS10")
        await self._ensure_fred_cache("VIXCLS")
        await self._ensure_fred_cache("T10Y2Y")

        treasury_10y = self._get_fred_value_at_date("DGS10", target_date) or 3.5
        vix = self._get_fred_value_at_date("VIXCLS", target_date) or 20.0
        yield_spread = self._get_fred_value_at_date("T10Y2Y", target_date) or 0.5
        print("done")

        # BTC and S&P500 prices
        await self._ensure_btc_cache()
        await self._ensure_sp500_cache()

        btc_prices = self._get_price_range(self._btc_cache, target_date, 90, 'price')
        sp500_prices = self._get_price_range(self._sp500_cache, target_date, 90, 'value')

        return HistoricalSnapshot(
            date=target_date,
            current_tvl=current_tvl,
            max_historical_tvl=max_historical_tvl,
            tvl_30d_history=tvl_30d_history,
            stablecoin_market_caps=stablecoin_caps,
            total_stablecoin_supply=total_stablecoin,
            treasury_10y_rate=treasury_10y,
            vix=vix,
            yield_curve_spread=yield_spread,
            btc_prices=btc_prices,
            sp500_prices=sp500_prices,
            num_protocols=100,  # Approximation for historical
            num_bridges=50,  # Approximation for historical
        )


# =============================================================================
# ASRI Calculator
# =============================================================================

def calculate_correlation(series1: list[float], series2: list[float]) -> float:
    """Calculate Pearson correlation between two series."""
    if len(series1) < 5 or len(series2) < 5:
        return 0.3

    # Align lengths
    min_len = min(len(series1), len(series2))
    s1 = series1[-min_len:]
    s2 = series2[-min_len:]

    # Calculate returns
    r1 = [(s1[i] - s1[i-1]) / s1[i-1] if s1[i-1] != 0 else 0 for i in range(1, len(s1))]
    r2 = [(s2[i] - s2[i-1]) / s2[i-1] if s2[i-1] != 0 else 0 for i in range(1, len(s2))]

    if len(r1) < 5:
        return 0.3

    # Correlation
    mean1 = sum(r1) / len(r1)
    mean2 = sum(r2) / len(r2)

    num = sum((r1[i] - mean1) * (r2[i] - mean2) for i in range(len(r1)))
    den1 = sum((r1[i] - mean1) ** 2 for i in range(len(r1))) ** 0.5
    den2 = sum((r2[i] - mean2) ** 2 for i in range(len(r2))) ** 0.5

    if den1 == 0 or den2 == 0:
        return 0.3

    corr = num / (den1 * den2)
    return max(-1, min(1, corr))


def calculate_sub_index(inputs: dict[str, float], weights: dict[str, float]) -> float:
    """Calculate a sub-index from inputs using weights."""
    total = 0.0
    for field, weight in weights.items():
        value = inputs.get(field, 50.0)
        total += value * weight
    return max(0, min(100, total))


def calculate_asri_from_snapshot(snapshot: HistoricalSnapshot) -> dict[str, Any]:
    """Calculate ASRI from a historical snapshot."""

    # BTC-S&P500 correlation
    crypto_equity_corr = calculate_correlation(snapshot.btc_prices, snapshot.sp500_prices)

    # Stablecoin Risk Inputs
    tvl_ratio = snapshot.current_tvl / snapshot.max_historical_tvl if snapshot.max_historical_tvl > 0 else 0.5
    tvl_risk = normalize_to_100(1 - tvl_ratio, 0, 0.5)
    treasury_stress = normalize_to_100(snapshot.treasury_10y_rate, 2.0, 6.0)

    circulating_values = list(snapshot.stablecoin_market_caps.values())
    hhi = calculate_hhi(circulating_values)
    concentration_risk = normalize_hhi_to_risk(hhi)

    stablecoin_inputs = {
        'tvl_ratio': tvl_risk,
        'treasury_stress': treasury_stress,
        'concentration_hhi': concentration_risk,
        'peg_volatility': 10.0,  # Default
    }

    # DeFi Liquidity Risk Inputs
    if len(snapshot.tvl_30d_history) > 1:
        mean_tvl = sum(snapshot.tvl_30d_history) / len(snapshot.tvl_30d_history)
        std_tvl = (sum((x - mean_tvl) ** 2 for x in snapshot.tvl_30d_history) / len(snapshot.tvl_30d_history)) ** 0.5
        volatility = std_tvl / mean_tvl if mean_tvl > 0 else 0
        tvl_volatility = normalize_to_100(volatility * 100, 0, 20)
    else:
        tvl_volatility = 30.0

    defi_inputs = {
        'top10_concentration': 45.0,  # Approximation
        'tvl_volatility': tvl_volatility,
        'smart_contract_risk': 35.0,  # Approximation
        'flash_loan_proxy': 30.0,
        'leverage_change': 40.0,
    }

    # Contagion Risk Inputs
    vix_stress = normalize_to_100(snapshot.vix, 12.0, 40.0)
    bank_exposure = (treasury_stress * 0.6 + vix_stress * 0.4)

    if snapshot.yield_curve_spread < 0:
        tradfi_linkage = normalize_to_100(abs(snapshot.yield_curve_spread), 0, 2) + 50
    else:
        tradfi_linkage = max(0, 50 - normalize_to_100(snapshot.yield_curve_spread, 0, 2))

    correlation_risk = normalize_to_100(abs(crypto_equity_corr), 0, 1)

    contagion_inputs = {
        'rwa_growth_rate': 25.0,
        'bank_exposure': bank_exposure,
        'tradfi_linkage': tradfi_linkage,
        'crypto_equity_correlation': correlation_risk,
        'bridge_exploit_frequency': normalize_to_100(snapshot.num_bridges, 0, 150),
    }

    # Arbitrage Opacity Risk Inputs
    stablecoins = snapshot.stablecoin_market_caps
    num_issuers = len([s for s, cap in stablecoins.items() if cap > 1e9])

    if num_issuers < 3:
        multi_issuer_risk = 70.0
    elif num_issuers < 10:
        multi_issuer_risk = 30.0
    else:
        multi_issuer_risk = 50.0 + (num_issuers - 10) * 2

    total_supply = sum(stablecoins.values())
    sorted_caps = sorted(stablecoins.values(), reverse=True)
    top2_share = sum(sorted_caps[:2]) / total_supply * 100 if total_supply > 0 else 85

    arbitrage_inputs = {
        'unregulated_exposure': 35.0,
        'multi_issuer_risk': multi_issuer_risk,
        'custody_concentration': normalize_to_100(top2_share, 50, 100),
        'regulatory_sentiment': 40.0,  # Neutral
        'transparency_score': 100 - 30.0,  # Inverted
    }

    # Calculate sub-indices
    stablecoin_risk = calculate_sub_index(stablecoin_inputs, STABLECOIN_WEIGHTS)
    defi_risk = calculate_sub_index(defi_inputs, DEFI_WEIGHTS)
    contagion_risk = calculate_sub_index(contagion_inputs, CONTAGION_WEIGHTS)
    arbitrage_risk = calculate_sub_index(arbitrage_inputs, ARBITRAGE_WEIGHTS)

    # Calculate aggregate ASRI
    asri = (
        stablecoin_risk * ASRI_WEIGHTS["stablecoin_risk"] +
        defi_risk * ASRI_WEIGHTS["defi_liquidity_risk"] +
        contagion_risk * ASRI_WEIGHTS["contagion_risk"] +
        arbitrage_risk * ASRI_WEIGHTS["arbitrage_opacity"]
    )

    return {
        "date": snapshot.date,
        "asri": round(asri, 1),
        "alert_level": determine_alert_level(asri),
        "stablecoin_risk": round(stablecoin_risk, 1),
        "defi_liquidity_risk": round(defi_risk, 1),
        "contagion_risk": round(contagion_risk, 1),
        "arbitrage_opacity": round(arbitrage_risk, 1),
    }


# =============================================================================
# Main Functions
# =============================================================================

async def run_backfill(
    start_date: datetime,
    end_date: datetime,
    dry_run: bool = False,
    skip_existing: bool = True,
):
    """Calculate ASRI for date range and push to D1."""

    print(f"\n{'=' * 70}")
    print("ASRI D1 BACKFILL - STANDALONE")
    print(f"{'=' * 70}")
    print(f"Range: {start_date.date()} to {end_date.date()}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()

    # Initialize clients
    fetcher = HistoricalDataFetcher()
    d1 = D1Client()

    try:
        # Check current D1 state
        print("Checking current D1 database state...")
        current_range = await d1.get_date_range()
        print(f"  Current range: {current_range['earliest']} to {current_range['latest']}")
        print(f"  Total records: {current_range['total']}")
        print()

        # Calculate and insert
        records_calculated = 0
        records_inserted = 0
        records_skipped = 0
        errors = []

        current = start_date
        total_days = (end_date - start_date).days + 1

        print(f"Processing {total_days} days...")
        print()

        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            progress = (current - start_date).days + 1

            # Check if exists
            if skip_existing:
                exists = await d1.check_date_exists(date_str)
                if exists:
                    print(f"[{progress}/{total_days}] {date_str}: SKIPPED (exists)")
                    records_skipped += 1
                    current += timedelta(days=1)
                    continue

            # Calculate ASRI
            print(f"[{progress}/{total_days}] {date_str}: Calculating...")
            try:
                snapshot = await fetcher.fetch_snapshot(current)
                result = calculate_asri_from_snapshot(snapshot)
                records_calculated += 1

                stablecoin_risk = round(result["stablecoin_risk"], 1)
                defi_liquidity_risk = round(result["defi_liquidity_risk"], 1)
                contagion_risk = round(result["contagion_risk"], 1)
                arbitrage_opacity = round(result["arbitrage_opacity"], 1)
                asri = compute_weighted_asri(
                    stablecoin_risk=stablecoin_risk,
                    defi_liquidity_risk=defi_liquidity_risk,
                    contagion_risk=contagion_risk,
                    arbitrage_opacity=arbitrage_opacity,
                )

                record = {
                    "date": date_str,
                    # Keep aggregate and component columns exactly consistent.
                    "asri": asri,
                    "asri_30d_avg": asri,
                    "trend": "stable",
                    "alert_level": determine_alert_level(asri),
                    "stablecoin_risk": stablecoin_risk,
                    "defi_liquidity_risk": defi_liquidity_risk,
                    "contagion_risk": contagion_risk,
                    "arbitrage_opacity": arbitrage_opacity,
                }

                if dry_run:
                    print(f"    -> ASRI={record['asri']:.1f} ({record['alert_level']}) [DRY RUN]")
                else:
                    success = await d1.insert_record(record)
                    if success:
                        records_inserted += 1
                        print(f"    -> ASRI={record['asri']:.1f} ({record['alert_level']}) [INSERTED]")
                    else:
                        errors.append(date_str)
                        print(f"    -> ASRI={record['asri']:.1f} [FAILED]")

            except Exception as e:
                errors.append(f"{date_str}: {str(e)}")
                print(f"    -> ERROR: {e}")

            current += timedelta(days=1)

            # Small delay to avoid rate limiting
            if not dry_run and records_inserted > 0 and records_inserted % 10 == 0:
                await asyncio.sleep(0.5)

        # Summary
        print()
        print(f"{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Days processed:  {total_days}")
        print(f"  Calculated:      {records_calculated}")
        print(f"  Inserted:        {records_inserted}")
        print(f"  Skipped:         {records_skipped}")
        print(f"  Errors:          {len(errors)}")

        if errors:
            print()
            print("Errors:")
            for err in errors[:10]:
                print(f"    - {err}")

        if not dry_run and records_inserted > 0:
            print()
            print("Verifying D1 state...")
            new_range = await d1.get_date_range()
            print(f"  New range: {new_range['earliest']} to {new_range['latest']}")
            print(f"  Total records: {new_range['total']}")

    finally:
        await fetcher.close()
        await d1.close()


async def check_d1_state():
    """Check current D1 database state."""
    print("\n=== D1 Database Status ===\n")

    d1 = D1Client()

    try:
        range_info = await d1.get_date_range()
        print(f"Earliest date: {range_info['earliest']}")
        print(f"Latest date:   {range_info['latest']}")
        print(f"Total records: {range_info['total']}")

        if range_info['earliest'] and range_info['latest']:
            earliest = datetime.strptime(range_info['earliest'], "%Y-%m-%d")
            latest = datetime.strptime(range_info['latest'], "%Y-%m-%d")
            expected = (latest - earliest).days + 1
            coverage = range_info['total'] / expected * 100 if expected > 0 else 0
            print(f"Expected days: {expected}")
            print(f"Coverage:      {coverage:.1f}%")

        print("\nChecking 2021 data...")
        result = await d1.query(
            "SELECT COUNT(*) as count FROM asri_daily WHERE date >= '2021-01-01' AND date <= '2021-12-31'"
        )
        count_2021 = result["result"][0]["results"][0]["count"]
        print(f"2021 records: {count_2021} / 365 expected")

    finally:
        await d1.close()


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def main():
    parser = argparse.ArgumentParser(description="ASRI D1 Backfill - Standalone")

    parser.add_argument("--start", type=parse_date, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=parse_date, help="End date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Calculate but don't insert into D1")
    parser.add_argument("--force", action="store_true", help="Don't skip existing dates")
    parser.add_argument("--check", action="store_true", help="Check current D1 state")

    args = parser.parse_args()

    if args.check:
        asyncio.run(check_d1_state())
    elif args.start and args.end:
        asyncio.run(run_backfill(
            args.start,
            args.end,
            dry_run=args.dry_run,
            skip_existing=not args.force,
        ))
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/backfill_d1_standalone.py --check")
        print("  python scripts/backfill_d1_standalone.py --start 2021-01-01 --end 2021-12-31 --dry-run")
        print("  python scripts/backfill_d1_standalone.py --start 2021-01-01 --end 2021-12-31")


if __name__ == "__main__":
    main()
