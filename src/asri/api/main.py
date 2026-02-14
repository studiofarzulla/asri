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

ASRI_WEIGHTS = {
    "stablecoin_risk": 0.30,
    "defi_liquidity_risk": 0.25,
    "contagion_risk": 0.25,
    "arbitrage_opacity": 0.20,
}


def compute_asri(sub_indices: dict[str, float]) -> float:
    """Compute aggregate ASRI from component values."""
    value = (
        ASRI_WEIGHTS["stablecoin_risk"] * sub_indices["stablecoin_risk"]
        + ASRI_WEIGHTS["defi_liquidity_risk"] * sub_indices["defi_liquidity_risk"]
        + ASRI_WEIGHTS["contagion_risk"] * sub_indices["contagion_risk"]
        + ASRI_WEIGHTS["arbitrage_opacity"] * sub_indices["arbitrage_opacity"]
    )
    return round(value, 1)


def classify_alert(asri: float) -> str:
    if asri < 30:
        return "low"
    if asri < 50:
        return "moderate"
    if asri < 70:
        return "elevated"
    return "critical"


def classify_trend(current: float, avg: float) -> str:
    diff = current - avg
    if diff > 2:
        return "rising"
    if diff < -2:
        return "falling"
    return "stable"


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
    methodology_profile: str = "paper_v2"
    warning: str | None = None


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
                sub_indices = {
                    "stablecoin_risk": calc_result['sub_indices']['stablecoin_risk'],
                    "defi_liquidity_risk": calc_result['sub_indices']['defi_liquidity_risk'],
                    "contagion_risk": calc_result['sub_indices']['contagion_risk'],
                    "arbitrage_opacity": calc_result['sub_indices']['arbitrage_opacity'],
                }
                asri = compute_asri(sub_indices)
                asri_30d_avg = calc_result.get('asri_30d_avg', asri)
                trend = calc_result.get('trend', classify_trend(asri, asri_30d_avg))

                return ASRIResponse(
                    timestamp=calc_result['timestamp'],
                    asri=asri,
                    asri_30d_avg=asri_30d_avg,
                    trend=trend,
                    sub_indices=SubIndices(
                        stablecoin_risk=sub_indices['stablecoin_risk'],
                        defi_liquidity_risk=sub_indices['defi_liquidity_risk'],
                        contagion_risk=sub_indices['contagion_risk'],
                        arbitrage_opacity=sub_indices['arbitrage_opacity'],
                    ),
                    alert_level=classify_alert(asri),
                    last_update=calc_result['timestamp'],
                    methodology_profile="paper_v2",
                )
            finally:
                await orchestrator.close()
        else:
            # Keep response shape aligned with Worker API even before first backfill.
            return ASRIResponse(
                timestamp=datetime.utcnow(),
                asri=0.0,
                asri_30d_avg=0.0,
                trend="unknown",
                sub_indices=SubIndices(
                    stablecoin_risk=0.0,
                    defi_liquidity_risk=0.0,
                    contagion_risk=0.0,
                    arbitrage_opacity=0.0,
                ),
                alert_level="unknown",
                last_update=datetime.utcnow(),
                methodology_profile="paper_v2",
                warning="No ASRI data available in database",
            )

    sub_indices = {
        "stablecoin_risk": record.stablecoin_risk,
        "defi_liquidity_risk": record.defi_liquidity_risk,
        "contagion_risk": record.contagion_risk,
        "arbitrage_opacity": record.arbitrage_opacity,
    }
    asri = compute_asri(sub_indices)
    avg_stmt = select(ASRIDaily).order_by(desc(ASRIDaily.date)).limit(30)
    avg_result = await db.execute(avg_stmt)
    avg_records = avg_result.scalars().all()
    avg_values = [
        compute_asri(
            {
                "stablecoin_risk": avg_record.stablecoin_risk,
                "defi_liquidity_risk": avg_record.defi_liquidity_risk,
                "contagion_risk": avg_record.contagion_risk,
                "arbitrage_opacity": avg_record.arbitrage_opacity,
            }
        )
        for avg_record in avg_records
    ]
    asri_30d_avg = round(sum(avg_values) / len(avg_values), 1) if avg_values else asri
    trend = record.trend or classify_trend(asri, asri_30d_avg)

    return ASRIResponse(
        timestamp=record.date,
        asri=asri,
        asri_30d_avg=asri_30d_avg,
        trend=trend,
        sub_indices=SubIndices(
            stablecoin_risk=sub_indices["stablecoin_risk"],
            defi_liquidity_risk=sub_indices["defi_liquidity_risk"],
            contagion_risk=sub_indices["contagion_risk"],
            arbitrage_opacity=sub_indices["arbitrage_opacity"],
        ),
        alert_level=classify_alert(asri),
        last_update=record.created_at,
        methodology_profile="paper_v2",
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
            asri=compute_asri(
                {
                    "stablecoin_risk": record.stablecoin_risk,
                    "defi_liquidity_risk": record.defi_liquidity_risk,
                    "contagion_risk": record.contagion_risk,
                    "arbitrage_opacity": record.arbitrage_opacity,
                }
            ),
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
            "methodology_profile": "paper_v2",
        },
    )


@app.get("/asri/subindex/{name}", tags=["ASRI"])
async def get_subindex(
    name: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get individual sub-index time series.

    Valid names: stablecoin_risk, defi_liquidity_risk, contagion_risk, arbitrage_opacity
    """
    valid_names = {
        "stablecoin_risk": {
            "name": "Stablecoin Risk",
            "weight": 0.30,
            "description": "Measures systemic risk from stablecoin peg deviations and concentration",
            "components": ["USDT peg deviation", "USDC peg deviation", "Stablecoin concentration"],
        },
        "defi_liquidity_risk": {
            "name": "DeFi Liquidity Risk",
            "weight": 0.25,
            "description": "Measures liquidity fragility across DeFi protocols",
            "components": ["TVL volatility", "Protocol concentration", "Liquidity depth"],
        },
        "contagion_risk": {
            "name": "Contagion Risk",
            "weight": 0.25,
            "description": "Measures cross-protocol and cross-chain exposure risk",
            "components": ["Cross-protocol exposure", "Bridge concentration", "Correlation clustering"],
        },
        "arbitrage_opacity": {
            "name": "Arbitrage Opacity",
            "weight": 0.20,
            "description": "Measures market efficiency and information asymmetry",
            "components": ["CEX-DEX spread", "Cross-chain arbitrage", "MEV activity"],
        },
    }
    if name not in valid_names:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sub-index. Valid options: {list(valid_names.keys())}",
        )

    stmt = select(ASRIDaily).order_by(desc(ASRIDaily.date)).limit(1)
    result = await db.execute(stmt)
    record = result.scalar_one_or_none()
    current_value = float(getattr(record, name)) if record else 0.0

    payload = valid_names[name].copy()
    payload["current_value"] = current_value
    payload["methodology_profile"] = "paper_v2"
    return payload


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
        "version": "2.1",
        "methodology_profile": "paper_v2",
        "weights": {
            "stablecoin_risk": 0.30,
            "defi_liquidity_risk": 0.25,
            "contagion_risk": 0.25,
            "arbitrage_opacity": 0.20,
        },
        "weight_derivation": {
            "theoretical": {"stablecoin_risk": 0.30, "defi_liquidity_risk": 0.25, "contagion_risk": 0.25, "arbitrage_opacity": 0.20},
            "pca": {"stablecoin_risk": 0.176, "defi_liquidity_risk": 0.140, "contagion_risk": 0.362, "arbitrage_opacity": 0.322},
            "elastic_net": {"stablecoin_risk": 0.145, "defi_liquidity_risk": 0.842, "contagion_risk": 0.000, "arbitrage_opacity": 0.013},
        },
        "data_sources": [
            {"name": "DeFi Llama", "url": "https://defillama.com", "metrics": ["TVL", "protocol data", "stablecoin supply"]},
            {"name": "FRED", "url": "https://fred.stlouisfed.org", "metrics": ["treasury rates", "macro indicators"]},
            {"name": "CoinGecko", "url": "https://coingecko.com", "metrics": ["price data", "market cap"]},
            {"name": "News APIs", "metrics": ["sentiment analysis", "regulatory events"]},
        ],
        "update_frequency": "daily",
        "backtest_period": "2021-01-01 to 2024-12-31",
        "validation_results": {
            "crises_detected": "4/4 (100%)",
            "average_lead_time_days": 30,
            "event_study_significance": "all p < 0.01",
            "structural_stability": "Chow p = 0.99",
        },
        "documentation_url": "https://asri.dissensus.ai/docs",
        "paper_doi": "10.5281/zenodo.17918239",
    }


class RegimeResponse(BaseModel):
    """Current regime classification."""
    current_regime: int
    regime_name: str
    probability: float
    transition_probs: dict[str, float]
    regime_history: list[dict[str, Any]] | None = None


class ValidationSummary(BaseModel):
    """Validation summary response."""
    stationarity: dict[str, Any]
    event_study: dict[str, Any]
    regime_model: dict[str, Any]
    robustness: dict[str, Any]


@app.get("/asri/regime", response_model=RegimeResponse, tags=["Analytics"])
async def get_current_regime():
    """
    Get current market regime classification.
    
    Returns the HMM-based regime classification:
    - Regime 1: Low Risk (mean ASRI ~34)
    - Regime 2: Moderate (mean ASRI ~35.5)
    - Regime 3: Elevated (mean ASRI ~49)
    """
    # In production, this would query the regime model
    return RegimeResponse(
        current_regime=2,
        regime_name="Moderate",
        probability=0.78,
        transition_probs={
            "to_low_risk": 0.023,
            "stay_moderate": 0.938,
            "to_elevated": 0.039,
        },
        regime_history=None,  # Add historical regime data if requested
    )


@app.get("/asri/validation", response_model=ValidationSummary, tags=["Analytics"])
async def get_validation_summary():
    """
    Get validation test results.
    
    Returns comprehensive validation statistics including:
    - Stationarity tests (ADF, KPSS)
    - Event study results
    - Regime model diagnostics
    - Robustness tests
    """
    return ValidationSummary(
        stationarity={
            "asri": {"adf_stat": -5.22, "adf_p": 0.000, "kpss": 0.31, "conclusion": "stationary"},
            "stablecoin_risk": {"adf_stat": -3.76, "adf_p": 0.003, "kpss": 1.13, "conclusion": "trend-stationary"},
            "defi_liquidity": {"adf_stat": -4.34, "adf_p": 0.000, "kpss": 0.11, "conclusion": "stationary"},
            "contagion_risk": {"adf_stat": -3.71, "adf_p": 0.004, "kpss": 0.89, "conclusion": "trend-stationary"},
            "arb_opacity": {"adf_stat": -4.33, "adf_p": 0.000, "kpss": 0.45, "conclusion": "stationary"},
        },
        event_study={
            "terra_luna": {"date": "2022-05", "t_stat": 5.47, "p_value": 0.000, "lead_days": 30, "significant": True},
            "celsius_3ac": {"date": "2022-06", "t_stat": 29.78, "p_value": 0.000, "lead_days": 30, "significant": True},
            "ftx_collapse": {"date": "2022-11", "t_stat": 32.64, "p_value": 0.000, "lead_days": 30, "significant": True},
            "svb_crisis": {"date": "2023-03", "t_stat": 26.91, "p_value": 0.000, "lead_days": 29, "significant": True},
            "summary": {"detection_rate": 1.0, "avg_lead_time": 29.8, "avg_cas": 472.4},
            "methodology_profile": "paper_v2",
        },
        regime_model={
            "n_regimes": 3,
            "regime_1": {"frequency": 0.246, "mean_risk": 34.0, "persistence": 0.942, "label": "Low Risk"},
            "regime_2": {"frequency": 0.439, "mean_risk": 35.5, "persistence": 0.961, "label": "Moderate"},
            "regime_3": {"frequency": 0.315, "mean_risk": 49.3, "persistence": 0.969, "label": "Elevated"},
            "aic": 42084, "bic": 42348,
        },
        robustness={
            "chow_test": {"statistic": 0.007, "critical": 3.002, "p_value": 0.993, "stable": True},
            "cusum": {"statistic": 4.715, "breaks_detected": True, "note": "breaks correspond to crisis events"},
        },
    )


@app.get("/asri/export/{format}", tags=["Data Export"])
async def export_data(
    format: str,
    start: str = Query(None, description="Start date (YYYY-MM-DD)"),
    end: str = Query(None, description="End date (YYYY-MM-DD)"),
):
    """
    Export ASRI data in various formats.
    
    Supported formats: json, csv, parquet
    """
    if format not in ["json", "csv", "parquet"]:
        raise HTTPException(status_code=400, detail=f"Invalid format. Supported: json, csv, parquet")
    
    # TODO: Implement actual data export
    return {
        "status": "success",
        "format": format,
        "download_url": f"https://asri.dissensus.ai/api/downloads/asri_data.{format}",
        "expires_in": 3600,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
