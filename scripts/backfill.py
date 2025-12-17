#!/usr/bin/env python3
"""
ASRI Historical Backfill Script

Populates the database with historical ASRI calculations
to extend the dashboard chart backwards in time.

Usage:
    python scripts/backfill.py --start 2022-01-01 --end 2024-12-31
    python scripts/backfill.py --days 365  # Last 365 days
    python scripts/backfill.py --validate  # Run crisis validation
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asri.backtest import ASRIBacktester, BacktestResult
from asri.backtest.backtest import CRISIS_EVENTS, print_backtest_report


async def run_backfill(start_date: datetime, end_date: datetime, step_days: int = 1):
    """Run historical backfill."""
    print(f"\n=== ASRI Historical Backfill ===")
    print(f"Range: {start_date.date()} to {end_date.date()}")
    print(f"Step: every {step_days} day(s)")
    print()

    backtester = ASRIBacktester()

    try:
        records = await backtester.backfill_database(
            start_date=start_date,
            end_date=end_date,
            step_days=step_days,
            clear_existing=False,
        )
        print(f"\nBackfill complete: {records} records created")

    finally:
        await backtester.close()


async def run_validation():
    """Run crisis validation backtest."""
    print("\n=== ASRI Crisis Validation ===")
    print("Testing against historical crises...\n")

    backtester = ASRIBacktester()

    try:
        results = await backtester.run_full_backtest()
        print_backtest_report(results)

    finally:
        await backtester.close()


async def run_single_date(target_date: datetime):
    """Calculate ASRI for a single date."""
    print(f"\n=== ASRI for {target_date.date()} ===\n")

    backtester = ASRIBacktester()

    try:
        result = await backtester.calculate_for_date(target_date)

        print(f"ASRI: {result['asri']:.1f}")
        print(f"Alert Level: {result['alert_level'].upper()}")
        print()
        print("Sub-Indices:")
        print(f"  Stablecoin Risk:     {result['stablecoin_risk']:.1f}")
        print(f"  DeFi Liquidity Risk: {result['defi_liquidity_risk']:.1f}")
        print(f"  Contagion Risk:      {result['contagion_risk']:.1f}")
        print(f"  Arbitrage Opacity:   {result['arbitrage_opacity']:.1f}")
        print()
        print("Raw Data:")
        for key, value in result['raw_snapshot'].items():
            if isinstance(value, float):
                if value > 1e9:
                    print(f"  {key}: ${value / 1e9:.2f}B")
                else:
                    print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

    finally:
        await backtester.close()


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def main():
    parser = argparse.ArgumentParser(description="ASRI Historical Backfill")

    parser.add_argument(
        "--start", type=parse_date,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=parse_date,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--days", type=int,
        help="Number of days to backfill from today"
    )
    parser.add_argument(
        "--step", type=int, default=1,
        help="Days between calculations (default: 1)"
    )
    parser.add_argument(
        "--date", type=parse_date,
        help="Calculate ASRI for a single date"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run crisis validation backtest"
    )

    args = parser.parse_args()

    if args.validate:
        asyncio.run(run_validation())
    elif args.date:
        asyncio.run(run_single_date(args.date))
    elif args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        asyncio.run(run_backfill(start_date, end_date, args.step))
    elif args.start and args.end:
        asyncio.run(run_backfill(args.start, args.end, args.step))
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/backfill.py --start 2022-01-01 --end 2024-12-31")
        print("  python scripts/backfill.py --days 365")
        print("  python scripts/backfill.py --date 2022-05-09  # Luna crash")
        print("  python scripts/backfill.py --validate")


if __name__ == "__main__":
    main()
