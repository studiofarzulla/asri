"""ASRI calculation pipeline."""

import asyncio
from datetime import datetime

import structlog

from asri.models.asri import ASRIDaily
from asri.models.base import async_session
from asri.signals.calculator import compute_asri

logger = structlog.get_logger()


async def calculate_and_store_asri(
    stablecoin_risk: float,
    defi_liquidity_risk: float,
    contagion_risk: float,
    arbitrage_opacity: float,
    date: datetime | None = None,
) -> ASRIDaily:
    """
    Calculate ASRI from sub-indices and store in database.

    Args:
        stablecoin_risk: Stablecoin risk sub-index (0-100)
        defi_liquidity_risk: DeFi liquidity risk sub-index (0-100)
        contagion_risk: Contagion risk sub-index (0-100)
        arbitrage_opacity: Arbitrage opacity risk sub-index (0-100)
        date: Date for this calculation (default: now)

    Returns:
        ASRIDaily database record
    """
    if date is None:
        date = datetime.utcnow()

    logger.info(
        "Calculating ASRI",
        date=date,
        stablecoin_risk=stablecoin_risk,
        defi_liquidity_risk=defi_liquidity_risk,
        contagion_risk=contagion_risk,
        arbitrage_opacity=arbitrage_opacity,
    )

    # Calculate ASRI using the calculator
    result = compute_asri(
        stablecoin_risk=stablecoin_risk,
        defi_liquidity_risk=defi_liquidity_risk,
        contagion_risk=contagion_risk,
        arbitrage_opacity=arbitrage_opacity,
    )

    # Store in database
    async with async_session() as session:
        asri_record = ASRIDaily(
            date=date,
            asri=result.asri,
            asri_normalized=result.asri_normalized,
            alert_level=result.alert_level,
            stablecoin_risk=result.sub_indices.stablecoin_risk,
            defi_liquidity_risk=result.sub_indices.defi_liquidity_risk,
            contagion_risk=result.sub_indices.contagion_risk,
            arbitrage_opacity=result.sub_indices.arbitrage_opacity,
        )

        session.add(asri_record)
        await session.commit()
        await session.refresh(asri_record)

        logger.info(
            "ASRI calculated and stored",
            asri=asri_record.asri,
            alert_level=asri_record.alert_level,
            record_id=asri_record.id,
        )

        return asri_record


async def run_daily_calculation():
    """
    Run the full daily ASRI calculation pipeline.

    This is a simplified version. In production, this would:
    1. Fetch data from all sources (FRED, DeFi Llama, etc.)
    2. Transform raw data into sub-index inputs
    3. Calculate each sub-index
    4. Calculate aggregate ASRI
    5. Store results
    """
    logger.info("Starting daily ASRI calculation")

    # For now, use mock sub-index values
    # TODO: Replace with real data fetching and transformation
    result = await calculate_and_store_asri(
        stablecoin_risk=65.0,
        defi_liquidity_risk=58.0,
        contagion_risk=72.0,
        arbitrage_opacity=48.0,
    )

    logger.info("Daily calculation complete", asri=result.asri)
    return result


if __name__ == "__main__":
    asyncio.run(run_daily_calculation())
