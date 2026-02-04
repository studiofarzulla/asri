"""
DeFi Llama API Client

Fetches TVL, stablecoin data, and protocol metrics from DeFi Llama.
API Docs: https://defillama.com/docs/api
"""

from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel


class TVLData(BaseModel):
    """Total Value Locked data point."""

    date: datetime
    tvl: float


class StablecoinData(BaseModel):
    """Stablecoin market data."""

    name: str
    symbol: str
    circulating: float
    price: float
    peg_deviation: float

    # Extended fields for algorithmic stablecoin risk (v2.1+)
    peg_type: str | None = None  # "fiat-collateral" | "crypto-collateral" | "algorithmic"
    backing_assets: dict[str, float] | None = None  # e.g., {"LUNA": 0.5, "USD": 0.5}
    backing_ratio: float | None = None  # reserves / circulating (if available)


class DeFiLlamaClient:
    """Client for DeFi Llama API."""

    BASE_URL = "https://api.llama.fi"
    STABLECOINS_URL = "https://stablecoins.llama.fi"

    def __init__(self, timeout: float = 30.0):
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def get_total_tvl(self) -> float:
        """Get current total DeFi TVL across all chains."""
        response = await self.client.get(f"{self.BASE_URL}/v2/historicalChainTvl")
        response.raise_for_status()
        data = response.json()
        # Latest data point
        if data:
            return data[-1].get("tvl", 0)
        return 0

    async def get_tvl_history(self) -> list[TVLData]:
        """Get historical total TVL."""
        response = await self.client.get(f"{self.BASE_URL}/v2/historicalChainTvl")
        response.raise_for_status()
        data = response.json()
        return [
            TVLData(date=datetime.fromtimestamp(point["date"]), tvl=point["tvl"])
            for point in data
        ]

    async def get_protocol_tvl(self, protocol: str) -> dict[str, Any]:
        """Get TVL for a specific protocol."""
        response = await self.client.get(f"{self.BASE_URL}/protocol/{protocol}")
        response.raise_for_status()
        return response.json()

    async def get_stablecoins(self) -> list[StablecoinData]:
        """Get stablecoin circulating supply and peg data."""
        response = await self.client.get(f"{self.STABLECOINS_URL}/stablecoins")
        response.raise_for_status()
        data = response.json()

        stablecoins = []
        for coin in data.get("peggedAssets", []):
            circulating = coin.get("circulating", {}).get("peggedUSD", 0)
            raw_price = coin.get("price")
            try:
                price = float(raw_price) if raw_price is not None else 1.0
            except (ValueError, TypeError):
                price = 1.0

            # Extract peg type if available (DeFi Llama API field)
            peg_type = coin.get("pegType")  # e.g., "peggedUSD", "peggedETH"
            peg_mechanism = coin.get("pegMechanism")  # e.g., "fiat", "algorithmic"

            # Classify based on available fields
            if peg_mechanism:
                classified_type = peg_mechanism
            elif peg_type and "algo" in str(peg_type).lower():
                classified_type = "algorithmic"
            else:
                classified_type = None

            stablecoins.append(
                StablecoinData(
                    name=coin.get("name", ""),
                    symbol=coin.get("symbol", ""),
                    circulating=float(circulating) if circulating else 0.0,
                    price=price,
                    peg_deviation=abs(1.0 - price),
                    peg_type=classified_type,
                    backing_assets=coin.get("backing"),  # May not be available
                    backing_ratio=coin.get("backingRatio"),  # May not be available
                )
            )
        return stablecoins

    async def get_stablecoin_history(self, stablecoin_id: int) -> list[dict]:
        """Get historical data for a specific stablecoin."""
        response = await self.client.get(
            f"{self.STABLECOINS_URL}/stablecoin/{stablecoin_id}"
        )
        response.raise_for_status()
        return response.json()

    async def get_bridges(self) -> list[dict]:
        """Get bridge TVL data."""
        response = await self.client.get("https://bridges.llama.fi/bridges")
        response.raise_for_status()
        return response.json().get("bridges", [])

    async def get_protocols(self) -> list[dict]:
        """Get all protocols with TVL, category, audits, etc."""
        response = await self.client.get(f"{self.BASE_URL}/protocols")
        response.raise_for_status()
        return response.json()

    async def get_yields(self) -> list[dict]:
        """Get yield/APY data across protocols."""
        response = await self.client.get(f"{self.BASE_URL}/pools")
        response.raise_for_status()
        return response.json().get("data", [])


# Convenience function for one-off calls
async def fetch_current_tvl() -> float:
    """Fetch current total DeFi TVL."""
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.llama.fi/v2/historicalChainTvl")
        response.raise_for_status()
        data = response.json()
        return data[-1].get("tvl", 0) if data else 0
