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
            price = coin.get("price", 1.0)
            stablecoins.append(
                StablecoinData(
                    name=coin.get("name", ""),
                    symbol=coin.get("symbol", ""),
                    circulating=circulating,
                    price=price,
                    peg_deviation=abs(1.0 - price) if price else 0,
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
        response = await self.client.get(f"{self.BASE_URL}/bridges")
        response.raise_for_status()
        return response.json().get("bridges", [])

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
