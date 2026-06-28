"""
Historical Data Fetcher for ASRI Backtesting

Fetches historical data from DeFiLlama, FRED, and CoinGecko
for a specific date to enable backtesting against historical crises.
"""

import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import structlog

from asri.config import settings as app_settings

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Canonical series provenance (read this before regenerating anything).
#
# The published, paper-of-record ASRI daily series (max 84.70 on 2022-11-08,
# mean 39.20, 1841 rows, 2021-01-01..2026-01-15) is the FROZEN parquet at
# results/data/asri_history.parquet (sha256 in DATA_PROVENANCE.md; Zenodo
# 10.5281/zenodo.17918239). Every paper headline is recomputed from that file
# by the scripts/ analysis layer -- the series is a released dataset, not a
# re-derived artefact.
#
# This module is the *generation* pipeline. After the Jun-2026 honesty fixes
# (D2 coin-ids, real peg loader, D5 rolling-365 TVL, and the frozen
# protocols/bridges snapshot below) it produces a CODE-CONSISTENT series, but it
# does NOT bit-reproduce the published parquet: the original generation-time
# DeFiLlama/FRED point-in-time inputs were pulled live and never archived, and
# the original protocols/bridges universe was never snapshotted. To make a regen
# DETERMINISTIC, fetch_snapshot now reads the protocols/bridges universe from a
# frozen on-disk snapshot (data/snapshots/<name>_<as_of>.json) when present,
# falling back to a loudly-flagged LIVE pull only when no snapshot exists.
# Do NOT overwrite the frozen canonical parquet from this pipeline.
# ---------------------------------------------------------------------------


@dataclass
class HistoricalSnapshot:
    """All data needed for ASRI calculation at a specific date."""
    date: datetime

    # TVL data
    current_tvl: float
    max_historical_tvl: float  # Max up to this date
    tvl_30d_history: list[float]

    # Stablecoin data
    stablecoin_market_caps: dict[str, float]  # symbol -> market cap
    total_stablecoin_supply: float

    # FRED data
    treasury_10y_rate: float
    vix: float
    yield_curve_spread: float
    sp500_prices: list[float]  # 90 days for correlation

    # BTC data
    btc_prices: list[float]  # 90 days for correlation

    # Protocol data (current, can't backtest protocol list easily)
    protocols: list[dict]
    bridges: list[dict]

    # Metadata
    data_quality: dict[str, str]  # source -> status


class HistoricalDataFetcher:
    """Fetches historical data for ASRI backtesting."""

    DEFILLAMA_BASE = "https://api.llama.fi"
    DEFILLAMA_STABLES = "https://stablecoins.llama.fi"
    DEFILLAMA_COINS = "https://coins.llama.fi"
    FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

    # DeFiLlama stablecoin id -> symbol.
    #
    # D2 fix (Jun 2026): corrected against the live DeFiLlama /stablecoins
    # listing (28 Jun 2026). The prior map used WRONG ids: it labelled id=3
    # "DAI" when id=3 is USTC/TerraClassicUSD -- the coin that actually
    # collapsed in May 2022 (supply ~$13.7B, price to ~$0.006) -- and labelled
    # id=11 "UST" when id=11 is USDP/Pax Dollar (~$1B, never depegged). With the
    # peg loader wired in, that mispairing fed the genuinely-collapsing supply a
    # ~par price and fed USDP's stable supply the UST depeg price, so the SCR
    # peg term stayed inert right through the Terra crisis. id=3 is mapped to
    # "UST" (not the literal DeFiLlama symbol "USTC") so it pairs with the Terra
    # depeg series stored under symbol "UST" in data/peg_history.csv.
    # ids 8/9/10/12 (LUSD/FEI/MIM/USDN) carry no peg-history rows, so the loader
    # par-treats them; their supplies still count toward HHI/concentration.
    MAJOR_STABLES = {
        1: "USDT",
        2: "USDC",
        3: "UST",   # id=3 = USTC/TerraClassicUSD (collapsed May 2022); paired
                    #   with the "UST" depeg series in data/peg_history.csv
        4: "BUSD",
        5: "DAI",
        6: "FRAX",
        7: "TUSD",
        8: "LUSD",
        9: "FEI",
        10: "MIM",
        11: "USDP",  # id=11 = USDP/Pax Dollar (never depegged) -- NOT Terra UST
        12: "USDN",
    }

    # Default location for frozen universe snapshots (Bug 2 fix, Jun 2026).
    SNAPSHOT_DIR = Path(__file__).resolve().parents[3] / "data" / "snapshots"

    def __init__(
        self,
        timeout: float = 60.0,
        snapshot_dir: str | Path | None = None,
        as_of: str | None = None,
    ):
        self.client = httpx.AsyncClient(timeout=timeout)
        self.settings = app_settings
        self._tvl_cache: list[dict] | None = None
        self._stablecoin_cache: dict[int, list[dict]] = {}
        self._btc_cache: list[dict] | None = None
        self._fred_cache: dict[str, list[dict]] = {}  # series_id -> observations
        self._sp500_cache: list[dict] | None = None
        # Protocol/bridge lists are *current* snapshots (documented look-ahead):
        # the same current state is projected onto every historical date, so the
        # value is identical for all dates. Cache once per fetcher instance to
        # avoid re-pulling the large /protocols payload on every date of a long
        # backfill (~thousands of redundant calls) and to make a full regen
        # deterministic within a run.
        self._protocols_cache: list[dict] | None = None
        self._bridges_cache: list[dict] | None = None
        self._protocols_quality: str = ""
        self._bridges_quality: str = ""
        # Bug 2 (look-ahead determinism) fix, Jun 2026: read the protocols/bridges
        # universe from a frozen on-disk snapshot pinned to an as-of-date when
        # available, instead of re-pulling the live (drifting) universe each run.
        # ``as_of`` selects data/snapshots/<name>_<as_of>.json; when None, the
        # newest snapshot on disk is used (or a loud LIVE fallback if none exist).
        # Override ASRI_SNAPSHOT_AS_OF / ASRI_SNAPSHOT_DIR via env for batch runs.
        self.snapshot_dir = Path(
            snapshot_dir
            or os.environ.get("ASRI_SNAPSHOT_DIR")
            or self.SNAPSHOT_DIR
        )
        self.as_of = as_of or os.environ.get("ASRI_SNAPSHOT_AS_OF") or None

    async def close(self):
        await self.client.aclose()

    def _frozen_snapshot_path(self, name: str) -> Path | None:
        """Resolve the frozen universe-snapshot file for ``name`` (Bug 2 fix).

        Returns data/snapshots/<name>_<as_of>.json when ``as_of`` is set, else
        the newest data/snapshots/<name>_*.json on disk. Returns None when no
        matching frozen snapshot exists (caller then falls back to a live pull).
        """
        if self.as_of:
            cand = self.snapshot_dir / f"{name}_{self.as_of}.json"
            return cand if cand.exists() else None
        matches = sorted(glob.glob(str(self.snapshot_dir / f"{name}_*.json")))
        return Path(matches[-1]) if matches else None

    async def _load_frozen_or_live(
        self, name: str, fetch_live
    ) -> tuple[list[dict], str]:
        """Load a universe list (protocols/bridges) from a frozen snapshot when
        present (deterministic), else pull LIVE and flag the non-determinism.

        ``fetch_live`` is an async callable returning the parsed live list.
        Returns ``(data, data_quality_string)``.
        """
        path = self._frozen_snapshot_path(name)
        if path is not None:
            try:
                with open(path) as f:
                    data = json.load(f)
                return data, f"frozen snapshot {path.name} ({len(data)} {name})"
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    f"Failed to read frozen {name} snapshot; falling back to live",
                    path=str(path), error=str(e),
                )
        # No frozen snapshot -> LIVE pull. This is the documented look-ahead AND
        # is non-deterministic across runs: the current universe drifts, so a
        # full regen will NOT match the published frozen series. Flag loudly.
        data = await fetch_live()
        today = datetime.now().strftime("%Y-%m-%d")
        logger.warning(
            f"No frozen {name} snapshot in {self.snapshot_dir}; pulled LIVE "
            f"(non-deterministic current universe). Run "
            f"scripts/dump_universe_snapshot.py to freeze for reproducible regens.",
            as_of=today, count=len(data),
        )
        return data, f"LIVE non-deterministic (as-of {today}, {len(data)} {name})"

    async def _ensure_tvl_cache(self):
        """Load and cache full TVL history."""
        if self._tvl_cache is None:
            logger.info("Fetching full TVL history...")
            resp = await self.client.get(f"{self.DEFILLAMA_BASE}/v2/historicalChainTvl")
            resp.raise_for_status()
            self._tvl_cache = resp.json()
            logger.info(f"Cached {len(self._tvl_cache)} TVL data points")

    async def _ensure_stablecoin_cache(self, stablecoin_id: int):
        """Load and cache stablecoin history."""
        if stablecoin_id not in self._stablecoin_cache:
            logger.info(f"Fetching stablecoin {stablecoin_id} history...")
            resp = await self.client.get(
                f"{self.DEFILLAMA_STABLES}/stablecoincharts/all?stablecoin={stablecoin_id}"
            )
            if resp.status_code == 200:
                self._stablecoin_cache[stablecoin_id] = resp.json()
            else:
                self._stablecoin_cache[stablecoin_id] = []

    async def _ensure_btc_cache(self):
        """Load and cache full BTC price history."""
        if self._btc_cache is None:
            logger.info("Fetching full BTC price history...")
            # Fetch in chunks to avoid timeout
            all_prices = []

            # 2021 onwards (covers all crises)
            chunks = [
                (datetime(2021, 1, 1), 365),
                (datetime(2022, 1, 1), 365),
                (datetime(2023, 1, 1), 365),
                (datetime(2024, 1, 1), 365),
            ]

            for start_date, span in chunks:
                try:
                    start_ts = int(start_date.timestamp())
                    resp = await self.client.get(
                        f"{self.DEFILLAMA_COINS}/chart/coingecko:bitcoin?start={start_ts}&span={span}",
                        timeout=30.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if 'coins' in data and 'coingecko:bitcoin' in data['coins']:
                            prices = data['coins']['coingecko:bitcoin'].get('prices', [])
                            all_prices.extend(prices)
                except Exception as e:
                    logger.warning(f"Failed to fetch BTC chunk {start_date.year}", error=str(e))

            # Sort by timestamp and deduplicate
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

            logger.info(f"Cached {len(self._btc_cache)} BTC price points")

    def _find_closest_tvl(self, target_date: datetime, data: list[dict]) -> tuple[float, datetime]:
        """Find TVL closest to target date."""
        target_ts = target_date.timestamp()
        closest = min(data, key=lambda x: abs(x['date'] - target_ts))
        return closest['tvl'], datetime.fromtimestamp(closest['date'])

    def _get_tvl_range(self, end_date: datetime, days: int, data: list[dict]) -> list[float]:
        """Get TVL values for a date range."""
        end_ts = end_date.timestamp()
        start_ts = (end_date - timedelta(days=days)).timestamp()

        values = [
            point['tvl'] for point in data
            if start_ts <= point['date'] <= end_ts
        ]
        return values if values else [0.0]

    def _get_max_tvl_before(
        self, date: datetime, data: list[dict], window_days: int = 365
    ) -> float:
        """Rolling maximum TVL over the trailing ``window_days`` window.

        D5 fix (Jun 2026): the prior implementation used an *expanding* running
        max over all history, which froze at the Nov-2021 all-chain TVL ATH
        (~$177.5B) and saturated ``tvl_ratio`` at ~98-100 from Terra (May 2022)
        through 2024 -- a ~12 ASRI-pt saturation artefact (the TVL term enters
        ASRI as 0.30*0.40*tvl_risk). A trailing 365-day rolling max keeps the
        denominator economically meaningful (a drawdown is measured against the
        prior-year peak, not a once-touched ATH). At the four 2022-2023 crises
        the Nov-2021 ATH is still inside the trailing year, so rolling ==
        expanding there; the two specs only diverge in 2024+. A 30-day
        rate-of-change spec exists as a sensitivity (results/tvl_respec_jun2026/)
        but is NOT the default.
        """
        target_ts = date.timestamp()
        start_ts = (date - timedelta(days=window_days)).timestamp()
        values = [
            point['tvl'] for point in data
            if start_ts <= point['date'] <= target_ts
        ]
        return max(values) if values else 0.0

    async def _fetch_stablecoin_at_date(
        self, stablecoin_id: int, target_date: datetime
    ) -> float:
        """Get stablecoin market cap at a specific date."""
        await self._ensure_stablecoin_cache(stablecoin_id)
        data = self._stablecoin_cache.get(stablecoin_id, [])

        if not data:
            return 0.0

        target_ts = target_date.timestamp()

        # Find closest data point
        closest = None
        min_diff = float('inf')

        for point in data:
            # Date might be string or int
            ts = int(point.get('date', 0))
            diff = abs(ts - target_ts)
            if diff < min_diff:
                min_diff = diff
                closest = point

        if closest:
            circulating = closest.get('totalCirculating', {}).get('peggedUSD', 0)
            return float(circulating) if circulating else 0.0

        return 0.0

    async def _fetch_btc_prices(
        self, end_date: datetime, days: int = 90
    ) -> list[float]:
        """Get BTC prices for correlation calculation."""
        await self._ensure_btc_cache()

        if not self._btc_cache:
            return []

        end_ts = end_date.timestamp()
        start_ts = (end_date - timedelta(days=days)).timestamp()

        prices = [
            point['price'] for point in self._btc_cache
            if start_ts <= point['timestamp'] <= end_ts
        ]

        return prices

    async def _ensure_fred_cache(self, series_id: str):
        """Load and cache full FRED series history."""
        if series_id not in self._fred_cache:
            logger.info(f"Fetching full FRED {series_id} history...")
            # Fetch from 2021 to present to cover all crisis periods
            params = {
                "series_id": series_id,
                "api_key": self.settings.fred_api_key,
                "file_type": "json",
                "observation_start": "2021-01-01",
            }

            try:
                resp = await self.client.get(self.FRED_BASE, params=params, timeout=30.0)
                resp.raise_for_status()
                data = resp.json()
                self._fred_cache[series_id] = data.get("observations", [])
                logger.info(f"Cached {len(self._fred_cache[series_id])} {series_id} observations")
            except Exception as e:
                logger.warning(f"FRED cache failed for {series_id}", error=str(e))
                self._fred_cache[series_id] = []

    def _get_fred_value_at_date(self, series_id: str, target_date: datetime) -> float | None:
        """Get FRED value from cache for a specific date."""
        observations = self._fred_cache.get(series_id, [])
        if not observations:
            return None

        target_str = target_date.strftime("%Y-%m-%d")

        # Find closest observation at or before target date
        closest_value = None
        for obs in observations:
            obs_date = obs.get("date", "")
            if obs_date <= target_str:
                value = obs.get("value")
                if value not in [".", None, ""]:
                    closest_value = float(value)
            else:
                break  # Observations are sorted by date

        return closest_value

    async def _fetch_fred_series(
        self, series_id: str, target_date: datetime, lookback_days: int = 7
    ) -> float | None:
        """Fetch FRED data closest to target date (using cache)."""
        await self._ensure_fred_cache(series_id)
        return self._get_fred_value_at_date(series_id, target_date)

    async def _ensure_sp500_cache(self):
        """Load and cache full S&P 500 history."""
        if self._sp500_cache is None:
            logger.info("Fetching full S&P500 history...")
            params = {
                "series_id": "SP500",
                "api_key": self.settings.fred_api_key,
                "file_type": "json",
                "observation_start": "2021-01-01",
            }

            try:
                resp = await self.client.get(self.FRED_BASE, params=params, timeout=30.0)
                resp.raise_for_status()
                data = resp.json()
                self._sp500_cache = data.get("observations", [])
                logger.info(f"Cached {len(self._sp500_cache)} S&P500 observations")
            except Exception as e:
                logger.warning("S&P500 cache failed", error=str(e))
                self._sp500_cache = []

    async def _fetch_sp500_prices(
        self, end_date: datetime, days: int = 90
    ) -> list[float]:
        """Fetch S&P 500 prices from FRED (using cache)."""
        await self._ensure_sp500_cache()

        if not self._sp500_cache:
            return []

        end_str = end_date.strftime("%Y-%m-%d")
        start_str = (end_date - timedelta(days=days)).strftime("%Y-%m-%d")

        prices = []
        for obs in self._sp500_cache:
            obs_date = obs.get("date", "")
            if start_str <= obs_date <= end_str:
                value = obs.get("value")
                if value not in [".", None, ""]:
                    prices.append(float(value))

        return prices

    async def fetch_snapshot(self, target_date: datetime) -> HistoricalSnapshot:
        """
        Fetch all data needed for ASRI calculation at a specific date.

        Args:
            target_date: The historical date to fetch data for

        Returns:
            HistoricalSnapshot with all data needed for calculation
        """
        logger.info(f"Fetching historical snapshot for {target_date.date()}")

        data_quality: dict[str, str] = {}

        # 1. TVL Data
        await self._ensure_tvl_cache()
        current_tvl, actual_date = self._find_closest_tvl(target_date, self._tvl_cache)
        max_historical_tvl = self._get_max_tvl_before(target_date, self._tvl_cache)
        tvl_30d_history = self._get_tvl_range(target_date, 30, self._tvl_cache)
        data_quality['tvl'] = f"ok (closest: {actual_date.date()})"

        # 2. Stablecoin Data
        stablecoin_caps: dict[str, float] = {}
        for stable_id, symbol in self.MAJOR_STABLES.items():
            cap = await self._fetch_stablecoin_at_date(stable_id, target_date)
            if cap > 0:
                stablecoin_caps[symbol] = cap

        total_stablecoin = sum(stablecoin_caps.values())
        data_quality['stablecoins'] = f"ok ({len(stablecoin_caps)} tracked)"

        # 3. FRED Data
        treasury_10y = await self._fetch_fred_series("DGS10", target_date)
        vix = await self._fetch_fred_series("VIXCLS", target_date)
        yield_spread = await self._fetch_fred_series("T10Y2Y", target_date)
        sp500_prices = await self._fetch_sp500_prices(target_date, days=90)

        # Use defaults for missing data
        if treasury_10y is None:
            treasury_10y = 3.5
            data_quality['treasury'] = "missing (using default 3.5%)"
        else:
            data_quality['treasury'] = "ok"

        if vix is None:
            vix = 20.0
            data_quality['vix'] = "missing (using default 20)"
        else:
            data_quality['vix'] = "ok"

        if yield_spread is None:
            yield_spread = 0.5
            data_quality['yield_curve'] = "missing (using default 0.5)"
        else:
            data_quality['yield_curve'] = "ok"

        data_quality['sp500'] = f"ok ({len(sp500_prices)} days)" if sp500_prices else "missing"

        # 4. BTC Prices
        btc_prices = await self._fetch_btc_prices(target_date, days=90)
        data_quality['btc'] = f"ok ({len(btc_prices)} days)" if btc_prices else "missing"

        # 5. Protocol/Bridge data -- CURRENT-universe look-ahead (documented).
        # Bug 2 fix (Jun 2026): prefer a frozen on-disk snapshot
        # (data/snapshots/<name>_<as_of>.json) so a full historical regen is
        # DETERMINISTIC and auditable; fall back to a LIVE pull (loudly flagged
        # non-deterministic in data_quality) only when no frozen snapshot exists.
        # Either way the same universe is projected onto every historical date,
        # so it is cached once per fetcher instance (see __init__).
        async def _live_protocols() -> list[dict]:
            try:
                resp = await self.client.get(f"{self.DEFILLAMA_BASE}/protocols")
                return resp.json() if resp.status_code == 200 else []
            except Exception:
                return []

        async def _live_bridges() -> list[dict]:
            try:
                resp = await self.client.get("https://bridges.llama.fi/bridges")
                return resp.json().get("bridges", []) if resp.status_code == 200 else []
            except Exception:
                return []

        if self._protocols_cache is None:
            self._protocols_cache, self._protocols_quality = \
                await self._load_frozen_or_live("protocols", _live_protocols)
        protocols = self._protocols_cache
        data_quality['protocols'] = self._protocols_quality or "failed"

        if self._bridges_cache is None:
            self._bridges_cache, self._bridges_quality = \
                await self._load_frozen_or_live("bridges", _live_bridges)
        bridges = self._bridges_cache
        data_quality['bridges'] = self._bridges_quality or "failed"

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
            sp500_prices=sp500_prices,
            btc_prices=btc_prices,
            protocols=protocols,
            bridges=bridges,
            data_quality=data_quality,
        )

    async def fetch_date_range(
        self, start_date: datetime, end_date: datetime, step_days: int = 1
    ) -> list[HistoricalSnapshot]:
        """
        Fetch snapshots for a date range.

        Args:
            start_date: Start of range
            end_date: End of range
            step_days: Days between snapshots

        Returns:
            List of HistoricalSnapshots
        """
        snapshots = []
        current = start_date

        while current <= end_date:
            snapshot = await self.fetch_snapshot(current)
            snapshots.append(snapshot)
            current += timedelta(days=step_days)

        return snapshots


async def main():
    """Test historical data fetching."""
    import asyncio

    fetcher = HistoricalDataFetcher()

    try:
        # Test fetching data for Luna crash (May 9, 2022)
        luna_date = datetime(2022, 5, 9)
        print(f"\n=== Fetching snapshot for Luna crash: {luna_date.date()} ===\n")

        snapshot = await fetcher.fetch_snapshot(luna_date)

        print(f"Date: {snapshot.date.date()}")
        print(f"TVL: ${snapshot.current_tvl / 1e9:.2f}B")
        print(f"Max Historical TVL: ${snapshot.max_historical_tvl / 1e9:.2f}B")
        print(f"TVL Ratio: {snapshot.current_tvl / snapshot.max_historical_tvl:.1%}")
        print()

        print("Stablecoins:")
        for symbol, cap in sorted(snapshot.stablecoin_market_caps.items(), key=lambda x: -x[1]):
            if cap > 1e9:
                print(f"  {symbol}: ${cap / 1e9:.2f}B")
        print(f"  Total: ${snapshot.total_stablecoin_supply / 1e9:.2f}B")
        print()

        print("FRED Data:")
        print(f"  10Y Treasury: {snapshot.treasury_10y_rate:.2f}%")
        print(f"  VIX: {snapshot.vix:.1f}")
        print(f"  Yield Spread: {snapshot.yield_curve_spread:.2f}%")
        print(f"  S&P500 data points: {len(snapshot.sp500_prices)}")
        print()

        print(f"BTC price data points: {len(snapshot.btc_prices)}")
        if snapshot.btc_prices:
            print(f"  Latest BTC: ${snapshot.btc_prices[-1]:,.0f}")
        print()

        print("Data Quality:")
        for source, status in snapshot.data_quality.items():
            print(f"  {source}: {status}")

    finally:
        await fetcher.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
