#!/usr/bin/env python3
"""
Quick script to investigate ablation discrepancy.

Runs ablation analysis on actual data to determine ground truth.
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asri.validation.ablation import (
    run_ablation_analysis,
    format_ablation_table,
    CRISIS_EVENTS,
)

def main():
    # Load actual data
    data_path = PROJECT_ROOT / "results" / "data" / "asri_history.parquet"
    print(f"Loading data from {data_path}...")

    df = pd.read_parquet(data_path)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)

    print(f"Loaded {len(df)} observations from {df.index.min().date()} to {df.index.max().date()}")
    print()

    # Check columns
    print("Available columns:", df.columns.tolist())
    print()

    # Sub-indices needed
    sub_cols = ["stablecoin_risk", "defi_liquidity_risk", "contagion_risk", "arbitrage_opacity"]

    missing = [c for c in sub_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return

    sub_indices = df[sub_cols]

    # Print crisis dates for reference
    print("Crisis Events:")
    for e in CRISIS_EVENTS:
        print(f"  {e['name']}: {e['date'].strftime('%Y-%m-%d')}")
    print()

    # Run ablation on actual data
    print("="*70)
    print("RUNNING ABLATION ANALYSIS ON ACTUAL DATA")
    print("="*70)

    results = run_ablation_analysis(
        sub_indices=sub_indices,
        threshold=50.0,
        pre_window_days=30,
    )

    # Print results
    print()
    print("ABLATION RESULTS (ACTUAL DATA):")
    print("-"*70)
    print(f"{'Excluded':<20} {'Weights':<15} {'Det.':<8} {'Lead':<10} {'Δ Lead':<10}")
    print("-"*70)

    for r in results:
        delta_str = f"{r.lead_time_delta:+.0f}" if r.excluded_short != "None" else "---"
        print(f"{r.excluded_short:<20} {r.weights_str:<15} {r.detection_rate:<8} {r.avg_lead_time:<10.0f} {delta_str:<10}")

    print()
    print("="*70)
    print()

    # Print per-crisis details
    print("PER-CRISIS DETECTION DETAILS:")
    print("-"*70)

    for r in results:
        print(f"\n{r.excluded_short}:")
        for cr in r.crisis_results:
            status = "✓" if cr.detected else "✗"
            print(f"  {status} {cr.crisis_name}: peak={cr.peak_asri:.1f}, lead={cr.lead_time_days}d")

    print()
    print("="*70)

    # Generate LaTeX
    print("\nLaTeX table (actual data):")
    print(format_ablation_table(results))

    # Compare to paper values
    print("\n" + "="*70)
    print("COMPARISON: PAPER vs ACTUAL DATA")
    print("="*70)

    paper_values = {
        "None": {"detection": "4/4", "lead": 40},
        "SCR": {"detection": "3/4", "lead": 35},
        "DLR": {"detection": "4/4", "lead": 32},
        "CR": {"detection": "3/4", "lead": 28},
        "OR": {"detection": "4/4", "lead": 36},
    }

    print(f"{'Component':<12} {'Paper Det.':<12} {'Actual Det.':<12} {'Paper Lead':<12} {'Actual Lead':<12} {'MATCH?'}")
    print("-"*80)

    for r in results:
        short = r.excluded_short
        if short in paper_values:
            paper = paper_values[short]
            det_match = paper["detection"] == r.detection_rate
            lead_diff = abs(paper["lead"] - r.avg_lead_time)
            match = "✓" if det_match and lead_diff < 5 else "✗ MISMATCH"

            print(f"{short:<12} {paper['detection']:<12} {r.detection_rate:<12} {paper['lead']:<12} {r.avg_lead_time:<12.0f} {match}")

    print()
    print("="*70)
    print("CONCLUSION: Paper uses SYNTHETIC data (hardcoded expected values),")
    print("while results/tables/ablation.tex uses ACTUAL data from run_ablation_analysis().")
    print()
    print("The ACTUAL DATA is ground truth. Paper narrative needs updating.")
    print("="*70)


if __name__ == "__main__":
    main()
