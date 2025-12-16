"""
ASRI Scheduled Jobs

Daily calculation and static site generation.
"""

import asyncio
import subprocess
from datetime import datetime
from pathlib import Path

import structlog

from asri.pipeline.orchestrator import ASRIOrchestrator
from asri.scheduler.static_generator import generate_static_site

logger = structlog.get_logger()

# Output directory for static files
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
STATIC_OUTPUT_DIR = PROJECT_ROOT / "static_site"
DEPLOY_SCRIPT = PROJECT_ROOT / "scripts" / "deploy.sh"


async def run_daily_calculation() -> dict:
    """
    Run daily ASRI calculation and save to database.

    Returns:
        Result dictionary from orchestrator
    """
    logger.info("Starting daily ASRI calculation", timestamp=datetime.utcnow())

    orchestrator = ASRIOrchestrator()

    try:
        result = await orchestrator.calculate_and_save()

        logger.info(
            "Daily calculation complete",
            asri=result['asri'],
            alert_level=result['alert_level'],
            db_id=result.get('db_id'),
        )

        return result

    except Exception as e:
        logger.error("Daily calculation failed", error=str(e))
        raise

    finally:
        await orchestrator.close()


def deploy_static_site() -> bool:
    """
    Deploy static site to resurrexi.io via rsync.

    Returns:
        True if deployment succeeded
    """
    if not DEPLOY_SCRIPT.exists():
        logger.warning("Deploy script not found", path=str(DEPLOY_SCRIPT))
        return False

    try:
        logger.info("Deploying to resurrexi.io...")
        result = subprocess.run(
            [str(DEPLOY_SCRIPT), str(STATIC_OUTPUT_DIR)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            logger.info("Deployment successful")
            return True
        else:
            logger.error("Deployment failed", stderr=result.stderr)
            return False

    except subprocess.TimeoutExpired:
        logger.error("Deployment timed out")
        return False
    except Exception as e:
        logger.error("Deployment error", error=str(e))
        return False


async def run_daily_job(deploy: bool = True):
    """
    Full daily job: calculate ASRI, regenerate static site, and deploy.

    Args:
        deploy: Whether to deploy to resurrexi.io after generation
    """
    logger.info("=" * 60)
    logger.info("DAILY ASRI JOB STARTED", timestamp=datetime.utcnow())
    logger.info("=" * 60)

    try:
        # Step 1: Calculate ASRI
        result = await run_daily_calculation()

        # Step 2: Generate static site
        output_dir = await generate_static_site(STATIC_OUTPUT_DIR)

        # Step 3: Deploy (optional)
        deployed = False
        if deploy:
            deployed = deploy_static_site()

        logger.info(
            "Daily job complete",
            asri=result['asri'],
            output_dir=str(output_dir),
            deployed=deployed,
        )

        return {
            'calculation': result,
            'static_site': str(output_dir),
            'deployed': deployed,
            'timestamp': datetime.utcnow(),
        }

    except Exception as e:
        logger.error("Daily job failed", error=str(e))
        raise


def sync_daily_job():
    """Synchronous wrapper for APScheduler."""
    return asyncio.run(run_daily_job())


if __name__ == "__main__":
    # Test run
    result = asyncio.run(run_daily_job())
    print(f"\n✅ Job complete! ASRI: {result['calculation']['asri']:.1f}")
    print(f"📁 Static site: {result['static_site']}")
