"""
ASRI FastAPI Application
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from asri import __version__


class SubIndices(BaseModel):
    """Sub-index values."""

    stablecoin_risk: float
    defi_liquidity_risk: float
    contagion_risk: float
    arbitrage_opacity: float


class ASRIResponse(BaseModel):
    """Current ASRI response."""

    timestamp: datetime
    asri: float
    asri_30d_avg: float
    trend: str
    sub_indices: SubIndices
    alert_level: str
    last_update: datetime


class TimeSeriesPoint(BaseModel):
    """Single time series data point."""

    date: str
    asri: float
    sub_indices: SubIndices


class TimeSeriesResponse(BaseModel):
    """Time series response."""

    data: list[TimeSeriesPoint]
    metadata: dict[str, Any]


class StressTestResponse(BaseModel):
    """Stress test response."""

    base_asri: float
    stressed_asri: float
    delta_asri: float
    affected_sub_indices: list[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: datetime


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print(f"Starting ASRI API v{__version__}")
    # TODO: Initialize database connection
    # TODO: Start scheduler for daily updates
    yield
    # Shutdown
    print("Shutting down ASRI API")


app = FastAPI(
    title="ASRI API",
    description="Aggregated Systemic Risk Index - Unified crypto/DeFi systemic risk monitoring",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.utcnow(),
    )


@app.get("/asri/current", response_model=ASRIResponse, tags=["ASRI"])
async def get_current_asri():
    """
    Get current ASRI value and sub-indices.

    Returns the latest calculated ASRI along with all sub-index values,
    trend direction, and alert level.
    """
    # TODO: Fetch from database
    # Placeholder response
    return ASRIResponse(
        timestamp=datetime.utcnow(),
        asri=62.3,
        asri_30d_avg=59.1,
        trend="rising",
        sub_indices=SubIndices(
            stablecoin_risk=68.5,
            defi_liquidity_risk=54.2,
            contagion_risk=71.1,
            arbitrage_opacity=49.0,
        ),
        alert_level="elevated",
        last_update=datetime.utcnow(),
    )


@app.get("/asri/timeseries", response_model=TimeSeriesResponse, tags=["ASRI"])
async def get_timeseries(
    start: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end: str = Query(..., description="End date (YYYY-MM-DD)"),
):
    """
    Get historical ASRI time series.

    Returns daily ASRI values between the specified dates.
    """
    # TODO: Fetch from database
    # Placeholder response
    return TimeSeriesResponse(
        data=[
            TimeSeriesPoint(
                date=start,
                asri=45.2,
                sub_indices=SubIndices(
                    stablecoin_risk=50.0,
                    defi_liquidity_risk=40.0,
                    contagion_risk=45.0,
                    arbitrage_opacity=42.0,
                ),
            )
        ],
        metadata={"points": 1, "frequency": "daily"},
    )


@app.get("/asri/subindex/{name}", tags=["ASRI"])
async def get_subindex(
    name: str,
    start: str = Query(None, description="Start date (YYYY-MM-DD)"),
    end: str = Query(None, description="End date (YYYY-MM-DD)"),
):
    """
    Get individual sub-index time series.

    Valid names: stablecoin_risk, defi_liquidity_risk, contagion_risk, arbitrage_opacity
    """
    valid_names = [
        "stablecoin_risk",
        "defi_liquidity_risk",
        "contagion_risk",
        "arbitrage_opacity",
    ]
    if name not in valid_names:
        return {"error": f"Invalid sub-index. Valid options: {valid_names}"}

    # TODO: Fetch from database
    return {"subindex": name, "data": [], "metadata": {"points": 0}}


@app.get("/asri/stress-test", response_model=StressTestResponse, tags=["ASRI"])
async def stress_test(
    scenario: str = Query(..., description="Scenario type (e.g., treasury_spike, defi_crash)"),
    magnitude: float = Query(50, description="Scenario magnitude (0-100)"),
):
    """
    Run stress test scenario.

    Calculates hypothetical ASRI under specified stress conditions.
    """
    # TODO: Implement actual stress test logic
    return StressTestResponse(
        base_asri=62.3,
        stressed_asri=78.5,
        delta_asri=16.2,
        affected_sub_indices=["stablecoin_risk", "contagion_risk"],
    )


@app.get("/asri/methodology", tags=["ASRI"])
async def get_methodology():
    """
    Get ASRI methodology documentation.

    Returns full documentation of ASRI construction, weights,
    data sources, and backtesting results.
    """
    return {
        "version": "1.0",
        "weights": {
            "stablecoin_risk": 0.30,
            "defi_liquidity_risk": 0.25,
            "contagion_risk": 0.25,
            "arbitrage_opacity": 0.20,
        },
        "data_sources": [
            "DeFi Llama",
            "Token Terminal",
            "FRED",
            "Messari",
            "Chainalysis (reports)",
        ],
        "update_frequency": "daily",
        "backtest_period": "2020-01-01 to present",
        "documentation_url": "https://resurrexi.dev/asri/methodology",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
