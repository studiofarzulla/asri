#!/usr/bin/env python3
"""
Run Publication Lag Analysis for ASRI

This script compares ASRI detection performance with and without
publication lag simulation, addressing JFS reviewer concern about
pseudo-real-time evaluation.

Usage:
    python scripts/run_lag_analysis.py

Output:
    - results/lag_comparison_results.json
    - results/tables/lag_comparison.tex
    - Console summary report
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asri.backtest.publication_lag import (
    compare_lag_impact,
    format_lag_comparison_table,
    get_lag_summary,
    DATA_LAGS,
)


async def main():
    """Run the full lag analysis."""
    print("=" * 70)
    print("ASRI PUBLICATION LAG ANALYSIS")
    print("Pseudo-Real-Time Evaluation for JFS Reviewer")
    print("=" * 70)
    print()

    # Output paths
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    # 1. Document lag configuration
    print("Publication Lag Configuration")
    print("-" * 40)
    lag_summary = get_lag_summary()
    for name, info in lag_summary.items():
        days = info['lag_days']
        hours = info['lag_hours']
        if days >= 1:
            print(f"  {name}: {days:.1f} days")
        else:
            print(f"  {name}: {hours:.0f} hours")
    print()

    # 2. Run comparison analysis
    print("Running comparison analysis (this may take a few minutes)...")
    print("-" * 40)

    try:
        results = await compare_lag_impact(threshold=50.0)
    except Exception as e:
        print(f"Error running analysis: {e}")
        print("\nNote: This analysis requires API access to DeFi Llama and FRED.")
        print("Make sure you have a valid FRED API key in your .env file.")
        raise

    # 3. Print detailed results
    print()
    print("=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)

    for r in results:
        status = "PASS" if r.detection_maintained else "FAIL"
        print(f"\n{r.crisis_name} [{status}]")
        print("-" * 50)
        print(f"  Date: {r.crisis_date.date()}")
        print()
        print("  Perfect Foresight (Baseline):")
        print(f"    Peak ASRI: {r.baseline_peak_asri:.1f}")
        print(f"    Lead Time: {r.baseline_lead_time} days")
        print(f"    Alert Level: {r.baseline_alert_level}")
        print(f"    Detected: {'Yes' if r.baseline_detected else 'No'}")
        print()
        print("  Lag-Simulated (Pseudo-Real-Time):")
        print(f"    Peak ASRI: {r.lagged_peak_asri:.1f}")
        print(f"    Lead Time: {r.lagged_lead_time} days")
        print(f"    Alert Level: {r.lagged_alert_level}")
        print(f"    Detected: {'Yes' if r.lagged_detected else 'No'}")
        print()
        print(f"  Impact Analysis:")
        print(f"    ASRI Degradation: {r.asri_degradation:.1f}%")
        print(f"    Lead Time Change: {r.lead_time_change:+d} days")
        print(f"    Limiting Source: {r.limiting_source}")

        if r.notes:
            print("  Notes:")
            for note in r.notes:
                print(f"    - {note}")

    # 4. Summary statistics
    print()
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    total = len(results)
    baseline_detected = sum(1 for r in results if r.baseline_detected)
    lagged_detected = sum(1 for r in results if r.lagged_detected)
    all_maintained = sum(1 for r in results if r.detection_maintained)

    import numpy as np
    avg_degradation = np.mean([r.asri_degradation for r in results])
    max_degradation = max(r.asri_degradation for r in results)
    avg_lead_change = np.mean([r.lead_time_change for r in results])
    min_lead_change = min(r.lead_time_change for r in results)

    print(f"  Baseline Detection Rate: {baseline_detected}/{total} "
          f"({baseline_detected/total*100:.0f}%)")
    print(f"  Lag-Simulated Detection Rate: {lagged_detected}/{total} "
          f"({lagged_detected/total*100:.0f}%)")
    print(f"  Detection Maintained: {all_maintained}/{total} "
          f"({all_maintained/total*100:.0f}%)")
    print()
    print(f"  Average ASRI Degradation: {avg_degradation:.2f}%")
    print(f"  Maximum ASRI Degradation: {max_degradation:.2f}%")
    print(f"  Average Lead Time Change: {avg_lead_change:+.1f} days")
    print(f"  Worst Lead Time Loss: {min_lead_change:+d} days")

    # 5. Conclusion
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if all_maintained == total:
        print("  RESULT: All crises remain detectable under lag simulation.")
        print("  The ASRI framework provides robust early warning even with")
        print("  realistic publication delays. Average ASRI degradation is")
        print(f"  minimal ({avg_degradation:.1f}%), and lead times remain")
        print("  sufficient for risk management purposes.")
    else:
        failed = [r.crisis_name for r in results if not r.detection_maintained]
        print(f"  WARNING: {len(failed)} crisis(es) not detected under lag simulation:")
        for name in failed:
            print(f"    - {name}")
        print()
        print("  Further investigation needed to understand which data sources")
        print("  are most critical for timely detection.")

    # 6. Save results
    print()
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # JSON results
    results_dict = {
        'metadata': {
            'analysis_date': str(asyncio.get_event_loop().time()),
            'threshold': 50.0,
            'num_crises': total,
        },
        'lag_configuration': lag_summary,
        'summary': {
            'baseline_detection_rate': baseline_detected / total,
            'lagged_detection_rate': lagged_detected / total,
            'detection_maintained_rate': all_maintained / total,
            'avg_asri_degradation_pct': avg_degradation,
            'max_asri_degradation_pct': max_degradation,
            'avg_lead_time_change_days': avg_lead_change,
            'worst_lead_time_loss_days': min_lead_change,
        },
        'results': [
            {
                'crisis': r.crisis_name,
                'crisis_date': r.crisis_date.isoformat(),
                'baseline': {
                    'peak_asri': r.baseline_peak_asri,
                    'lead_time_days': r.baseline_lead_time,
                    'alert_level': r.baseline_alert_level,
                    'detected': r.baseline_detected,
                },
                'lagged': {
                    'peak_asri': r.lagged_peak_asri,
                    'lead_time_days': r.lagged_lead_time,
                    'alert_level': r.lagged_alert_level,
                    'detected': r.lagged_detected,
                },
                'impact': {
                    'asri_degradation_pct': r.asri_degradation,
                    'lead_time_change_days': r.lead_time_change,
                    'detection_maintained': r.detection_maintained,
                    'limiting_source': r.limiting_source,
                },
                'notes': r.notes,
            }
            for r in results
        ],
    }

    json_path = results_dir / "lag_comparison_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"  Saved JSON results to: {json_path}")

    # LaTeX table
    latex_table = format_lag_comparison_table(results)
    tex_path = tables_dir / "lag_comparison.tex"
    with open(tex_path, 'w') as f:
        f.write(latex_table)
    print(f"  Saved LaTeX table to: {tex_path}")

    print()
    print("Analysis complete.")

    return results


if __name__ == "__main__":
    asyncio.run(main())
