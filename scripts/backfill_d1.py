#!/usr/bin/env python3
"""
ASRI D1 Backfill Script

Calculates historical ASRI values and pushes directly to Cloudflare D1.
This bypasses the local PostgreSQL and syncs directly to the production API.

Usage:
    python scripts/backfill_d1.py --start 2021-01-01 --end 2021-12-31
    python scripts/backfill_d1.py --start 2021-01-01 --end 2021-12-31 --dry-run
    python scripts/backfill_d1.py --check  # Check current D1 date range
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment
load_dotenv(Path(__file__).parent.parent / ".env")

# Cloudflare credentials
CF_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
CF_EMAIL = os.getenv("CLOUDFLARE_EMAIL", "contact@farzulla.com")
CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "917877c07fa2e1c9f223db31d3fc52d6")
D1_DATABASE_ID = "f8c08c15-a596-4d7f-8c00-0f7592da26f4"

ASRI_WEIGHTS = {
    "stablecoin_risk": 0.30,
    "defi_liquidity_risk": 0.25,
    "contagion_risk": 0.25,
    "arbitrage_opacity": 0.20,
}

# Try to load from cloudflare env if not in main .env
if not CF_API_KEY:
    cf_env = Path.home() / ".env.cloudflare"
    if cf_env.exists():
        with open(cf_env) as f:
            for line in f:
                if line.startswith("export CLOUDFLARE_API_KEY="):
                    CF_API_KEY = line.split("=", 1)[1].strip().strip('"')
                elif line.startswith("export CLOUDFLARE_ACCOUNT_ID="):
                    CF_ACCOUNT_ID = line.split("=", 1)[1].strip().strip('"')


class D1Client:
    """Cloudflare D1 client for direct database operations."""

    def __init__(self):
        if not CF_API_KEY:
            raise ValueError("CLOUDFLARE_API_KEY not found. Set in .env or ~/.env.cloudflare")

        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_DATABASE_ID}"
        self.headers = {
            "X-Auth-Email": CF_EMAIL,
            "X-Auth-Key": CF_API_KEY,
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(timeout=60.0)

    async def close(self):
        await self.client.aclose()

    async def query(self, sql: str, params: list = None) -> dict:
        """Execute a SQL query against D1."""
        payload = {"sql": sql}
        if params:
            payload["params"] = params

        resp = await self.client.post(
            f"{self.base_url}/query",
            headers=self.headers,
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    async def get_date_range(self) -> dict:
        """Get the current date range in D1."""
        result = await self.query(
            "SELECT MIN(date) as earliest, MAX(date) as latest, COUNT(*) as total FROM asri_daily"
        )
        return result["result"][0]["results"][0]

    async def check_date_exists(self, date: str) -> bool:
        """Check if a date already exists in D1."""
        result = await self.query(
            "SELECT 1 FROM asri_daily WHERE date = ? LIMIT 1",
            [date]
        )
        return len(result["result"][0]["results"]) > 0

    async def insert_record(self, record: dict) -> bool:
        """Insert a single ASRI record into D1."""
        sql = """
            INSERT INTO asri_daily
            (date, asri, asri_30d_avg, trend, alert_level,
             stablecoin_risk, defi_liquidity_risk, contagion_risk, arbitrage_opacity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = [
            record["date"],
            record["asri"],
            record.get("asri_30d_avg", record["asri"]),
            record.get("trend", "stable"),
            record["alert_level"],
            record["stablecoin_risk"],
            record["defi_liquidity_risk"],
            record["contagion_risk"],
            record["arbitrage_opacity"],
        ]

        try:
            await self.query(sql, params)
            return True
        except Exception as e:
            print(f"  Error inserting {record['date']}: {e}")
            return False

    async def batch_insert(self, records: list[dict]) -> int:
        """Insert multiple records. Returns count of successful inserts."""
        success = 0
        for record in records:
            if await self.insert_record(record):
                success += 1
        return success


def compute_weighted_asri(
    stablecoin_risk: float,
    defi_liquidity_risk: float,
    contagion_risk: float,
    arbitrage_opacity: float,
) -> float:
    """Compute ASRI using canonical weighted aggregation."""
    asri = (
        ASRI_WEIGHTS["stablecoin_risk"] * stablecoin_risk
        + ASRI_WEIGHTS["defi_liquidity_risk"] * defi_liquidity_risk
        + ASRI_WEIGHTS["contagion_risk"] * contagion_risk
        + ASRI_WEIGHTS["arbitrage_opacity"] * arbitrage_opacity
    )
    return round(asri, 1)


def determine_alert_level(asri: float) -> str:
    """Map ASRI score to alert level with deterministic boundaries."""
    if asri < 30:
        return "low"
    if asri < 50:
        return "moderate"
    if asri < 70:
        return "elevated"
    return "critical"


async def run_backfill(
    start_date: datetime,
    end_date: datetime,
    dry_run: bool = False,
    skip_existing: bool = True,
):
    """
    Calculate ASRI for date range and push to D1.

    Args:
        start_date: Start of backfill range
        end_date: End of backfill range
        dry_run: If True, calculate but don't insert
        skip_existing: If True, skip dates that already exist in D1
    """
    from asri.backtest import ASRIBacktester

    print(f"\n{'=' * 70}")
    print("ASRI D1 BACKFILL")
    print(f"{'=' * 70}")
    print(f"Range: {start_date.date()} to {end_date.date()}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()

    # Initialize clients
    backtester = ASRIBacktester()
    d1 = D1Client()

    try:
        # Check current D1 state
        print("Checking current D1 database state...")
        current_range = await d1.get_date_range()
        print(f"  Current range: {current_range['earliest']} to {current_range['latest']}")
        print(f"  Total records: {current_range['total']}")
        print()

        # Calculate and insert
        records_calculated = 0
        records_inserted = 0
        records_skipped = 0
        errors = []

        current = start_date
        total_days = (end_date - start_date).days + 1

        print(f"Processing {total_days} days...")
        print()

        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            progress = (current - start_date).days + 1

            # Check if exists
            if skip_existing:
                exists = await d1.check_date_exists(date_str)
                if exists:
                    print(f"  [{progress}/{total_days}] {date_str}: SKIPPED (exists)")
                    records_skipped += 1
                    current += timedelta(days=1)
                    continue

            # Calculate ASRI
            try:
                result = await backtester.calculate_for_date(current)
                records_calculated += 1

                stablecoin_risk = round(result["stablecoin_risk"], 1)
                defi_liquidity_risk = round(result["defi_liquidity_risk"], 1)
                contagion_risk = round(result["contagion_risk"], 1)
                arbitrage_opacity = round(result["arbitrage_opacity"], 1)
                asri = compute_weighted_asri(
                    stablecoin_risk=stablecoin_risk,
                    defi_liquidity_risk=defi_liquidity_risk,
                    contagion_risk=contagion_risk,
                    arbitrage_opacity=arbitrage_opacity,
                )

                record = {
                    "date": date_str,
                    # Keep ASRI mathematically consistent with stored sub-index columns.
                    "asri": asri,
                    "asri_30d_avg": asri,  # Will be recalculated later
                    "trend": "stable",
                    "alert_level": determine_alert_level(asri),
                    "stablecoin_risk": stablecoin_risk,
                    "defi_liquidity_risk": defi_liquidity_risk,
                    "contagion_risk": contagion_risk,
                    "arbitrage_opacity": arbitrage_opacity,
                }

                if dry_run:
                    print(f"  [{progress}/{total_days}] {date_str}: ASRI={record['asri']:.1f} ({record['alert_level']}) [DRY RUN]")
                else:
                    success = await d1.insert_record(record)
                    if success:
                        records_inserted += 1
                        print(f"  [{progress}/{total_days}] {date_str}: ASRI={record['asri']:.1f} ({record['alert_level']}) [INSERTED]")
                    else:
                        errors.append(date_str)
                        print(f"  [{progress}/{total_days}] {date_str}: ASRI={record['asri']:.1f} [FAILED]")

            except Exception as e:
                errors.append(f"{date_str}: {str(e)}")
                print(f"  [{progress}/{total_days}] {date_str}: ERROR - {e}")

            current += timedelta(days=1)

            # Small delay to avoid rate limiting
            if not dry_run and records_inserted > 0 and records_inserted % 10 == 0:
                await asyncio.sleep(0.5)

        # Summary
        print()
        print(f"{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Days processed:  {total_days}")
        print(f"  Calculated:      {records_calculated}")
        print(f"  Inserted:        {records_inserted}")
        print(f"  Skipped:         {records_skipped}")
        print(f"  Errors:          {len(errors)}")

        if errors:
            print()
            print("Errors:")
            for err in errors[:10]:
                print(f"    - {err}")
            if len(errors) > 10:
                print(f"    ... and {len(errors) - 10} more")

        if not dry_run:
            # Verify new state
            print()
            print("Verifying D1 state...")
            new_range = await d1.get_date_range()
            print(f"  New range: {new_range['earliest']} to {new_range['latest']}")
            print(f"  Total records: {new_range['total']}")

    finally:
        await backtester.close()
        await d1.close()


async def check_d1_state():
    """Check current D1 database state."""
    print("\n=== D1 Database Status ===\n")

    d1 = D1Client()

    try:
        range_info = await d1.get_date_range()
        print(f"Earliest date: {range_info['earliest']}")
        print(f"Latest date:   {range_info['latest']}")
        print(f"Total records: {range_info['total']}")

        # Calculate expected
        if range_info['earliest'] and range_info['latest']:
            earliest = datetime.strptime(range_info['earliest'], "%Y-%m-%d")
            latest = datetime.strptime(range_info['latest'], "%Y-%m-%d")
            expected = (latest - earliest).days + 1
            coverage = range_info['total'] / expected * 100 if expected > 0 else 0
            print(f"Expected days: {expected}")
            print(f"Coverage:      {coverage:.1f}%")

        # Check for gaps in 2021
        print("\nChecking 2021 data...")
        result = await d1.query(
            "SELECT COUNT(*) as count FROM asri_daily WHERE date >= '2021-01-01' AND date <= '2021-12-31'"
        )
        count_2021 = result["result"][0]["results"][0]["count"]
        print(f"2021 records: {count_2021} / 365 expected")

    finally:
        await d1.close()


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def main():
    parser = argparse.ArgumentParser(description="ASRI D1 Backfill")

    parser.add_argument(
        "--start", type=parse_date,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=parse_date,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Calculate but don't insert into D1"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Don't skip existing dates"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check current D1 state without modifying"
    )

    args = parser.parse_args()

    if args.check:
        asyncio.run(check_d1_state())
    elif args.start and args.end:
        asyncio.run(run_backfill(
            args.start,
            args.end,
            dry_run=args.dry_run,
            skip_existing=not args.force,
        ))
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/backfill_d1.py --check")
        print("  python scripts/backfill_d1.py --start 2021-01-01 --end 2021-12-31 --dry-run")
        print("  python scripts/backfill_d1.py --start 2021-01-01 --end 2021-12-31")


if __name__ == "__main__":
    main()
