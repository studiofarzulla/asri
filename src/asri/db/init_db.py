"""Database initialization script."""

import asyncio

from sqlalchemy import text

# Import models to register them with Base
from asri.models.asri import ASRIDaily  # noqa: F401
from asri.models.base import Base, engine


async def init_db():
    """Create all database tables."""
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        print("✅ Database tables created")

        # Verify tables exist
        result = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
        )
        tables = [row[0] for row in result.fetchall()]
        print(f"📋 Tables: {tables}")


async def drop_db():
    """Drop all database tables (use with caution!)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        print("⚠️  All tables dropped")


if __name__ == "__main__":
    print("Initializing ASRI database...")
    asyncio.run(init_db())
