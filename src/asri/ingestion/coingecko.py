"""
CoinGecko API Client

Fetches cryptocurrency price data for correlation calculations.
API Docs: https://docs.coingecko.com/
"""

import os
from datetime import datetime, timedelta
from typing import Any

import httpx
import numpy as np
import structlog
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger()


class CoinGeckoClient:
    """Client for CoinGecko API."""

    BASE_URL = "https://api.coingecko.com/api/v3"
    PRO_URL = "https://pro-api.coingecko.com/api/v3"

    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        self.api_key = api_key or os.getenv('COINGECKO_API_KEY')
        self.timeout = timeout

        # Determine API type from key format
        if self.api_key:
            if self.api_key.startswith("CG-"):
                # Demo API key
                self.base_url = self.BASE_URL
                self.headers = {"x-cg-demo-api-key": self.api_key}
            else:
                # Pro API key
                self.base_url = self.PRO_URL
                self.headers = {"x-cg-pro-api-key": self.api_key}
        else:
            self.base_url = self.BASE_URL
            self.headers = {}

        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers=self.headers
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def get_price_history(
        self,
        coin_id: str = "bitcoin",
        vs_currency: str = "usd",
        days: int = 90,
    ) -> list[tuple[datetime, float]]:
        """
        Get historical price data for a coin.

        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
            vs_currency: Currency to get prices in
            days: Number of days of history

        Returns:
            List of (datetime, price) tuples
        """
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": days,
            "interval": "daily",
        }

        logger.info("Fetching CoinGecko price history", coin=coin_id, days=days)

        response = await self.client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        prices = []
        for timestamp_ms, price in data.get("prices", []):
            dt = datetime.fromtimestamp(timestamp_ms / 1000)
            prices.append((dt, price))

        return prices

    async def get_current_price(
        self,
        coin_ids: list[str],
        vs_currencies: list[str] = ["usd"],
    ) -> dict[str, dict[str, float]]:
        """Get current prices for multiple coins."""
        url = f"{self.base_url}/simple/price"
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": ",".join(vs_currencies),
        }

        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def calculate_correlation(
    series1: list[float],
    series2: list[float],
) -> float:
    """
    Calculate Pearson correlation between two price series.

    Args:
        series1: First price series
        series2: Second price series (must be same length)

    Returns:
        Correlation coefficient (-1 to 1)
    """
    if len(series1) != len(series2):
        # Trim to same length
        min_len = min(len(series1), len(series2))
        series1 = series1[-min_len:]
        series2 = series2[-min_len:]

    if len(series1) < 2:
        return 0.0

    # Calculate returns (percentage change)
    returns1 = np.diff(series1) / series1[:-1]
    returns2 = np.diff(series2) / series2[:-1]

    # Handle any NaN or inf values
    mask = np.isfinite(returns1) & np.isfinite(returns2)
    returns1 = returns1[mask]
    returns2 = returns2[mask]

    if len(returns1) < 2:
        return 0.0

    # Calculate correlation
    correlation = np.corrcoef(returns1, returns2)[0, 1]

    return float(correlation) if np.isfinite(correlation) else 0.0


async def get_crypto_equity_correlation(
    coingecko_client: CoinGeckoClient,
    fred_sp500_prices: list[float],
    days: int = 90,
) -> float:
    """
    Calculate correlation between BTC and S&P500.

    Args:
        coingecko_client: CoinGecko client instance
        fred_sp500_prices: S&P500 prices from FRED
        days: Number of days for correlation window

    Returns:
        Correlation coefficient (-1 to 1)
    """
    try:
        # Get BTC price history
        btc_history = await coingecko_client.get_price_history(
            coin_id="bitcoin",
            days=days
        )

        btc_prices = [price for _, price in btc_history]

        if not btc_prices or not fred_sp500_prices:
            logger.warning("Missing price data for correlation")
            return 0.5  # Default moderate correlation

        correlation = calculate_correlation(btc_prices, fred_sp500_prices)

        logger.info(
            "Calculated crypto-equity correlation",
            correlation=correlation,
            btc_points=len(btc_prices),
            sp500_points=len(fred_sp500_prices),
        )

        return correlation

    except Exception as e:
        logger.error("Failed to calculate correlation", error=str(e))
        return 0.5  # Default on error


# Convenience function for testing
async def test_coingecko():
    """Test CoinGecko client."""
    client = CoinGeckoClient()

    try:
        # Test current price
        print("--- Current Prices ---")
        prices = await client.get_current_price(["bitcoin", "ethereum"])
        print(f"BTC: ${prices['bitcoin']['usd']:,.2f}")
        print(f"ETH: ${prices['ethereum']['usd']:,.2f}")

        # Test historical data
        print("\n--- BTC 30-Day History ---")
        history = await client.get_price_history("bitcoin", days=30)
        print(f"Got {len(history)} data points")
        if history:
            print(f"First: {history[0][0].date()} = ${history[0][1]:,.2f}")
            print(f"Last:  {history[-1][0].date()} = ${history[-1][1]:,.2f}")

        # Calculate simple volatility
        prices_only = [p for _, p in history]
        if prices_only:
            returns = np.diff(prices_only) / prices_only[:-1]
            volatility = np.std(returns) * np.sqrt(365) * 100
            print(f"Annualized volatility: {volatility:.1f}%")

        print("\n✅ CoinGecko client working!")

    finally:
        await client.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_coingecko())
