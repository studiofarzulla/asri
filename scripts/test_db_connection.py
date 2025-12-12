"""Test database connection and insert sample data."""

import asyncio
from datetime import datetime

from sqlalchemy import select

from asri.models.asri import ASRIDaily
from asri.models.base import async_session, engine, Base


async def test_connection():
    """Test database connection."""
    print("Testing database connection...")

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    print("✅ Tables created/verified")

    # Insert sample data
    async with async_session() as session:
        sample_asri = ASRIDaily(
            date=datetime(2025, 12, 12),
            asri=62.5,
            asri_normalized=65.0,
            asri_30d_avg=60.0,
            trend="rising",
            alert_level="elevated",
            stablecoin_risk=68.5,
            defi_liquidity_risk=54.2,
            contagion_risk=71.1,
            arbitrage_opacity=49.0,
        )
        session.add(sample_asri)
        await session.commit()
        print(f"✅ Inserted sample ASRI record: {sample_asri}")

    # Query data back
    async with async_session() as session:
        stmt = select(ASRIDaily).order_by(ASRIDaily.date.desc()).limit(1)
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()

        if record:
            print(f"✅ Retrieved record: date={record.date}, asri={record.asri}, alert={record.alert_level}")
        else:
            print("❌ No records found")

    print("\n✅ Database connection test successful!")


if __name__ == "__main__":
    asyncio.run(test_connection())
