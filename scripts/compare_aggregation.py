#!/usr/bin/env python3
"""
Compare Aggregation Methods for ASRI

Addresses Reviewer Q7: Non-linear aggregation (CES/geometric) comparison.
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asri.aggregation.nonlinear import (
    linear_aggregate,
    ces_aggregate,
    geometric_aggregate,
    compare_aggregation_methods,
    fit_ces_rho,
    format_aggregation_comparison_latex,
)


CRISIS_EVENTS = [
    datetime(2022, 5, 12),   # Terra/Luna
    datetime(2022, 6, 17),   # Celsius/3AC
    datetime(2022, 11, 11),  # FTX Collapse
    datetime(2023, 3, 11),   # SVB Crisis
]


def main():
    # Load data
    df = pd.read_parquet(PROJECT_ROOT / "results" / "data" / "asri_history.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)

    # Extract sub-indices
    sub_indices = df[["stablecoin_risk", "defi_liquidity_risk",
                      "contagion_risk", "arbitrage_opacity"]].dropna()

    print("=" * 70)
    print("AGGREGATION METHOD COMPARISON")
    print("=" * 70)
    print(f"Sample: {len(sub_indices)} observations")
    print(f"Period: {sub_indices.index.min()} to {sub_indices.index.max()}")
    print()

    # Compare methods
    results = compare_aggregation_methods(sub_indices, CRISIS_EVENTS, threshold=50.0)

    print("Results:")
    print("-" * 80)
    print(f"{'Method':<20} {'Mean':>8} {'Std':>8} {'Max':>8} {'Skew':>8} {'Det':>8} {'Lead':>8}")
    print("-" * 80)

    for r in results:
        print(f"{r.method:<20} {r.mean:>8.1f} {r.std:>8.1f} {r.max_value:>8.1f} "
              f"{r.skewness:>8.2f} {r.n_detected}/{r.n_crises:>5} {r.avg_lead_days:>8.1f}")

    print()

    # Fit optimal CES rho
    print("Fitting optimal CES ρ parameter...")
    optimal_rho = fit_ces_rho(sub_indices, CRISIS_EVENTS, threshold=50.0)
    print(f"Optimal ρ: {optimal_rho:.3f}")
    print()

    # Test with optimal rho
    optimal_asri = ces_aggregate(sub_indices, rho=optimal_rho)
    print(f"CES (ρ={optimal_rho:.2f}): mean={optimal_asri.mean():.1f}, "
          f"max={optimal_asri.max():.1f}")

    # Save table
    tables_dir = PROJECT_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    latex = format_aggregation_comparison_latex(results)
    (tables_dir / "aggregation_nonlinear.tex").write_text(latex)
    print(f"\nSaved: {tables_dir / 'aggregation_nonlinear.tex'}")

    # Generate interpretation
    linear_result = next(r for r in results if r.method == "Linear")
    ces_neg_result = next(r for r in results if "ρ=-0.5" in r.method)
    geom_result = next(r for r in results if r.method == "Geometric")

    print()
    print("=" * 70)
    print("KEY FINDINGS:")
    print("=" * 70)
    print(f"1. Linear baseline: {linear_result.n_detected}/4 detected, "
          f"avg lead {linear_result.avg_lead_days:.0f} days")
    print(f"2. CES (ρ=-0.5): {ces_neg_result.n_detected}/4 detected, "
          f"avg lead {ces_neg_result.avg_lead_days:.0f} days")
    print(f"3. Geometric: {geom_result.n_detected}/4 detected, "
          f"avg lead {geom_result.avg_lead_days:.0f} days")
    print()

    if ces_neg_result.n_detected > linear_result.n_detected:
        print("✓ CES with ρ < 0 (complementary risks) improves detection!")
        print("  This validates the intuition that systemic risks compound.")
    elif ces_neg_result.n_detected == linear_result.n_detected:
        if ces_neg_result.avg_lead_days > linear_result.avg_lead_days:
            print("✓ CES provides equivalent detection with longer lead times.")
        else:
            print("→ Detection rates equivalent across methods.")


if __name__ == "__main__":
    main()
