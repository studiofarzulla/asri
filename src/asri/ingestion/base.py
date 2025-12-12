"""Base connector class for data ingestion."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class BaseConnector(ABC):
    """Abstract base class for data connectors."""

    def __init__(self, api_key: str | None = None, rate_limit: int = 5):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.client = httpx.AsyncClient(timeout=30.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    @abstractmethod
    async def fetch_data(self, **kwargs) -> dict[str, Any]:
        """Fetch data from the source."""
        pass

    async def _get_with_retry(
        self, url: str, params: dict[str, Any] | None = None, max_retries: int = 3
    ) -> dict[str, Any]:
        """Make GET request with retry logic."""
        for attempt in range(max_retries):
            try:
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.warning(
                    "HTTP error",
                    status_code=e.response.status_code,
                    url=url,
                    attempt=attempt + 1,
                )
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)
            except Exception as e:
                logger.error("Request failed", error=str(e), url=url, attempt=attempt + 1)
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)

        raise Exception(f"Failed to fetch data from {url} after {max_retries} attempts")
