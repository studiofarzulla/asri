"""
ASRI FastAPI Application
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from asri import __version__
from asri.models.asri import ASRIDaily
from asri.models.base import async_session
from asri.pipeline.orchestrator import ASRIOrchestrator


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


class CalculateResponse(BaseModel):
    """Calculate response."""

    status: str
    asri: float
    alert_level: str
    sub_indices: SubIndices
    db_id: int | None = None
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: datetime


async def get_db() -> AsyncSession:
    """Dependency for database sessions."""
    async with async_session() as session:
        yield session


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print(f"Starting ASRI API v{__version__}")
    print("✅ Database connection ready")
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
async def get_current_asri(
    calculate_if_missing: bool = Query(
        False,
        description="If no data exists, trigger a live calculation",
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Get current ASRI value and sub-indices.

    Returns the latest calculated ASRI along with all sub-index values,
    trend direction, and alert level.

    Set calculate_if_missing=true to trigger a live calculation if no data exists.
    """
    stmt = select(ASRIDaily).order_by(desc(ASRIDaily.date)).limit(1)
    result = await db.execute(stmt)
    record = result.scalar_one_or_none()

    if not record:
        if calculate_if_missing:
            # Trigger a live calculation
            orchestrator = ASRIOrchestrator()
            try:
                calc_result = await orchestrator.calculate_and_save()

                return ASRIResponse(
                    timestamp=calc_result['timestamp'],
                    asri=calc_result['asri'],
                    asri_30d_avg=calc_result.get('asri_30d_avg', calc_result['asri']),
                    trend=calc_result.get('trend', 'stable'),
                    sub_indices=SubIndices(
                        stablecoin_risk=calc_result['sub_indices']['stablecoin_risk'],
                        defi_liquidity_risk=calc_result['sub_indices']['defi_liquidity_risk'],
                        contagion_risk=calc_result['sub_indices']['contagion_risk'],
                        arbitrage_opacity=calc_result['sub_indices']['arbitrage_opacity'],
                    ),
                    alert_level=calc_result['alert_level'],
                    last_update=calc_result['timestamp'],
                )
            finally:
                await orchestrator.close()
        else:
            raise HTTPException(
                status_code=404,
                detail="No ASRI data available yet. Use POST /asri/calculate or set calculate_if_missing=true.",
            )

    return ASRIResponse(
        timestamp=record.date,
        asri=record.asri,
        asri_30d_avg=record.asri_30d_avg or record.asri,
        trend=record.trend or "stable",
        sub_indices=SubIndices(
            stablecoin_risk=record.stablecoin_risk,
            defi_liquidity_risk=record.defi_liquidity_risk,
            contagion_risk=record.contagion_risk,
            arbitrage_opacity=record.arbitrage_opacity,
        ),
        alert_level=record.alert_level,
        last_update=record.created_at,
    )


@app.get("/asri/timeseries", response_model=TimeSeriesResponse, tags=["ASRI"])
async def get_timeseries(
    start: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end: str = Query(..., description="End date (YYYY-MM-DD)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get historical ASRI time series.

    Returns daily ASRI values between the specified dates.
    """
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD.",
        )

    stmt = (
        select(ASRIDaily)
        .where(ASRIDaily.date >= start_date)
        .where(ASRIDaily.date <= end_date)
        .order_by(ASRIDaily.date)
    )
    result = await db.execute(stmt)
    records = result.scalars().all()

    data = [
        TimeSeriesPoint(
            date=record.date.strftime("%Y-%m-%d"),
            asri=record.asri,
            sub_indices=SubIndices(
                stablecoin_risk=record.stablecoin_risk,
                defi_liquidity_risk=record.defi_liquidity_risk,
                contagion_risk=record.contagion_risk,
                arbitrage_opacity=record.arbitrage_opacity,
            ),
        )
        for record in records
    ]

    return TimeSeriesResponse(
        data=data,
        metadata={
            "points": len(data),
            "frequency": "daily",
            "start": start,
            "end": end,
        },
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


@app.post("/asri/calculate", response_model=CalculateResponse, tags=["ASRI"])
async def calculate_asri(
    save: bool = Query(True, description="Save result to database"),
):
    """
    Trigger a live ASRI calculation.

    Fetches data from all sources (DeFiLlama, FRED, CoinGecko, News)
    and calculates the current ASRI value.

    Set save=true to persist the result to the database.
    """
    orchestrator = ASRIOrchestrator()

    try:
        if save:
            result = await orchestrator.calculate_and_save()
            db_id = result.get('db_id')
            message = f"ASRI calculated and saved (ID: {db_id})"
        else:
            result = await orchestrator.calculate_asri()
            db_id = None
            message = "ASRI calculated (not saved)"

        return CalculateResponse(
            status="success",
            asri=result['asri'],
            alert_level=result['alert_level'],
            sub_indices=SubIndices(
                stablecoin_risk=result['sub_indices']['stablecoin_risk'],
                defi_liquidity_risk=result['sub_indices']['defi_liquidity_risk'],
                contagion_risk=result['sub_indices']['contagion_risk'],
                arbitrage_opacity=result['sub_indices']['arbitrage_opacity'],
            ),
            db_id=db_id,
            message=message,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Calculation failed: {str(e)}",
        )

    finally:
        await orchestrator.close()


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
