"""FRED (Federal Reserve Economic Data) connector."""

from datetime import datetime
from typing import Any

import structlog

from .base import BaseConnector

logger = structlog.get_logger()


class FREDConnector(BaseConnector):
    """Connector for Federal Reserve Economic Data API."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str):
        super().__init__(api_key=api_key)
        if not api_key:
            raise ValueError("FRED API key is required")

    async def fetch_series(
        self, series_id: str, start_date: str | None = None, end_date: str | None = None
    ) -> dict[str, Any]:
        """
        Fetch a data series from FRED.

        Args:
            series_id: FRED series ID (e.g., 'DGS10' for 10-year Treasury rate)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dictionary with series data
        """
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }

        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date

        logger.info("Fetching FRED series", series_id=series_id, start_date=start_date)

        data = await self._get_with_retry(self.BASE_URL, params)
        return data

    async def fetch_data(self, series_ids: list[str] | None = None, **kwargs) -> dict[str, Any]:
        """
        Fetch multiple series from FRED.

        Args:
            series_ids: List of FRED series IDs to fetch
            **kwargs: Additional parameters (start_date, end_date)

        Returns:
            Dictionary mapping series IDs to their data
        """
        if series_ids is None:
            series_ids = self.get_default_series()

        results = {}
        for series_id in series_ids:
            try:
                data = await self.fetch_series(series_id, **kwargs)
                results[series_id] = data
            except Exception as e:
                logger.error("Failed to fetch series", series_id=series_id, error=str(e))
                results[series_id] = {"error": str(e)}

        return results

    @staticmethod
    def get_default_series() -> list[str]:
        """Get default FRED series for ASRI calculation."""
        return [
            "DGS10",  # 10-Year Treasury Constant Maturity Rate
            "DGS2",  # 2-Year Treasury Constant Maturity Rate
            "VIXCLS",  # CBOE Volatility Index: VIX
            "DEXUSEU",  # U.S. / Euro Foreign Exchange Rate
            "T10Y2Y",  # 10-Year Treasury Minus 2-Year Treasury
        ]

    async def get_treasury_rates(self, start_date: str) -> dict[str, Any]:
        """Get Treasury yield curve data."""
        return await self.fetch_series("DGS10", start_date=start_date)

    async def get_vix(self, start_date: str) -> dict[str, Any]:
        """Get VIX (market volatility) data."""
        return await self.fetch_series("VIXCLS", start_date=start_date)

    async def get_yield_curve_spread(self, start_date: str) -> dict[str, Any]:
        """Get 10-Year minus 2-Year Treasury spread."""
        return await self.fetch_series("T10Y2Y", start_date=start_date)
