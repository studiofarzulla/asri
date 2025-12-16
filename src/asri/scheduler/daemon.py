"""
ASRI Scheduler Daemon

Runs daily ASRI calculations and static site generation.
Can be run as a standalone daemon or integrated into the API.
"""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from asri.config import settings
from asri.scheduler.jobs import run_daily_job

load_dotenv()
logger = structlog.get_logger()


class ASRIScheduler:
    """ASRI calculation scheduler."""

    def __init__(self, hour: int = 1, minute: int = 0):
        """
        Initialize scheduler.

        Args:
            hour: Hour to run daily job (0-23, UTC)
            minute: Minute to run daily job (0-59)
        """
        self.hour = hour
        self.minute = minute
        self.scheduler = AsyncIOScheduler()
        self._running = False

    def _setup_jobs(self):
        """Set up scheduled jobs."""
        # Daily ASRI calculation at specified time (UTC)
        self.scheduler.add_job(
            run_daily_job,
            CronTrigger(hour=self.hour, minute=self.minute),
            id="daily_asri",
            name="Daily ASRI Calculation",
            replace_existing=True,
        )

        logger.info(
            "Scheduled daily ASRI job",
            hour=self.hour,
            minute=self.minute,
            timezone="UTC",
        )

    async def start(self):
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._setup_jobs()
        self.scheduler.start()
        self._running = True

        logger.info("ASRI Scheduler started")

        # Print next run time
        job = self.scheduler.get_job("daily_asri")
        if job:
            logger.info(f"Next run: {job.next_run_time}")

    async def stop(self):
        """Stop the scheduler."""
        if not self._running:
            return

        self.scheduler.shutdown(wait=False)
        self._running = False
        logger.info("ASRI Scheduler stopped")

    async def run_now(self):
        """Trigger immediate calculation (for testing/manual runs)."""
        logger.info("Triggering immediate ASRI calculation")
        return await run_daily_job()

    def get_status(self) -> dict:
        """Get scheduler status."""
        job = self.scheduler.get_job("daily_asri") if self._running else None

        return {
            "running": self._running,
            "scheduled_hour": self.hour,
            "scheduled_minute": self.minute,
            "next_run": str(job.next_run_time) if job else None,
        }


async def run_daemon():
    """Run the scheduler as a standalone daemon."""
    print("=" * 60)
    print("ASRI SCHEDULER DAEMON")
    print(f"Daily calculation at {settings.daily_update_hour:02d}:00 UTC")
    print("=" * 60)

    scheduler = ASRIScheduler(hour=settings.daily_update_hour)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(scheduler)))

    try:
        await scheduler.start()

        # Run initial calculation if no recent data
        from asri.models.asri import ASRIDaily
        from asri.models.base import async_session
        from sqlalchemy import desc, select
        from datetime import timedelta

        async with async_session() as db:
            # Check for data from today
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            stmt = select(ASRIDaily).where(ASRIDaily.date >= today)
            result = await db.execute(stmt)
            recent = result.scalar_one_or_none()

            if not recent:
                print("\n📊 No data for today, running initial calculation...")
                await scheduler.run_now()

        print("\n✅ Scheduler running. Press Ctrl+C to stop.")
        print(f"📅 Next scheduled run: {scheduler.get_status()['next_run']}")

        # Keep running
        while True:
            await asyncio.sleep(60)

    except asyncio.CancelledError:
        pass
    finally:
        await scheduler.stop()


async def shutdown(scheduler: ASRIScheduler):
    """Graceful shutdown handler."""
    print("\n🛑 Shutting down scheduler...")
    await scheduler.stop()
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(run_daemon())
