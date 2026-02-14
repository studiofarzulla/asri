#!/usr/bin/env python3
"""Run ASRI event-study analysis with explicit methodology profiles."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asri.validation.event_study import (  # noqa: E402
    CrisisEvent,
    METHODOLOGY_PROFILES,
    format_event_study_table,
    run_event_study,
)

CRISIS_EVENTS = [
    CrisisEvent(name="Terra/Luna", event_date=datetime(2022, 5, 12)),
    CrisisEvent(name="Celsius/3AC", event_date=datetime(2022, 6, 17)),
    CrisisEvent(name="FTX Collapse", event_date=datetime(2022, 11, 11)),
    CrisisEvent(name="SVB Crisis", event_date=datetime(2023, 3, 11)),
]


def load_asri_series() -> pd.Series:
    """Load ASRI series from local parquet, fallback to public API."""
    parquet_path = PROJECT_ROOT / "results" / "data" / "asri_history.parquet"
    try:
        df = pd.read_parquet(parquet_path)
        source = f"parquet:{parquet_path}"
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)
        asri = df["asri"].sort_index()
        print(f"Loaded ASRI data from {source} ({len(asri)} rows)")
        return asri
    except Exception:
        url = "https://api.dissensus.ai/asri/timeseries?start=2021-01-01&end=2026-12-31"
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ASRI-Validation/1.0 (+https://asri.dissensus.ai)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("data", [])
        if not rows:
            raise RuntimeError("No ASRI rows returned from fallback API")
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        asri = df.set_index("date")["asri"].sort_index()
        print(f"Loaded ASRI data from api:{url} ({len(asri)} rows)")
        return asri


def summarize(results: list, profile: str) -> dict:
    """Build compact summary dict for reporting and serialization."""
    return {
        "profile": profile,
        "n_events": len(results),
        "n_significant": sum(1 for r in results if r.is_significant),
        "avg_lead_days": float(pd.Series([r.lead_days for r in results]).mean()),
        "avg_cas": float(pd.Series([r.cumulative_abnormal_signal for r in results]).mean()),
        "events": [
            {
                "name": r.event.name,
                "date": r.event.event_date.strftime("%Y-%m"),
                "pre_mean": round(float(r.pre_event_mean), 3),
                "peak": round(float(r.peak_asri), 3),
                "cas": round(float(r.cumulative_abnormal_signal), 3),
                "t_stat": round(float(r.t_statistic), 3),
                "p_value": round(float(r.p_value), 6),
                "lead_days": int(r.lead_days),
                "significant": bool(r.is_significant),
                "lead_method": r.lead_method,
            }
            for r in results
        ],
    }


def format_comparison_table(profile_to_summary: dict[str, dict]) -> str:
    """Create LaTeX table comparing methodology profiles event-by-event."""
    events = [e.name for e in CRISIS_EVENTS]
    profiles = list(profile_to_summary.keys())

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Event Study Reconciliation Across Methodology Profiles}",
        r"\label{tab:event_study_reconciliation}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Profile & Event & Peak & CAS & $t$-stat & Lead \\",
        r"\midrule",
    ]

    for profile in profiles:
        event_rows = profile_to_summary[profile]["events"]
        for i, event_name in enumerate(events):
            row = next(r for r in event_rows if r["name"] == event_name)
            prefix = profile if i == 0 else ""
            lines.append(
                f"{prefix} & {row['name']} & {row['peak']:.1f} & {row['cas']:.1f} & "
                f"{row['t_stat']:.2f} & {row['lead_days']} \\\\"
            )
        lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"\item Profiles are computed on the same ASRI series to isolate methodological effects.",
            r"\end{tablenotes}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ASRI event study with profile support.")
    parser.add_argument(
        "--profile",
        choices=sorted(METHODOLOGY_PROFILES.keys()),
        default="paper_v2",
        help="Methodology profile for primary event study output.",
    )
    parser.add_argument(
        "--compare-profiles",
        action="store_true",
        help="Generate side-by-side profile comparison artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asri = load_asri_series()

    results = run_event_study(asri=asri, events=CRISIS_EVENTS, profile=args.profile)
    summary = summarize(results, args.profile)

    print("=" * 80)
    print(f"ASRI EVENT STUDY ({args.profile})")
    print("=" * 80)
    print(
        f"Events={summary['n_events']} | Significant={summary['n_significant']} | "
        f"AvgLead={summary['avg_lead_days']:.1f}d | AvgCAS={summary['avg_cas']:.1f}"
    )
    print()
    print(f"{'Event':<15} {'Date':<8} {'Peak':>7} {'CAS':>10} {'t-stat':>9} {'Lead':>6}")
    print("-" * 80)
    for row in summary["events"]:
        print(
            f"{row['name']:<15} {row['date']:<8} "
            f"{row['peak']:>7.1f} {row['cas']:>10.1f} {row['t_stat']:>9.2f} {row['lead_days']:>6d}"
        )

    tables_dir = PROJECT_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "event_study.tex", "w", encoding="utf-8") as handle:
        handle.write(format_event_study_table(results))
    print(f"\nSaved table: {tables_dir / 'event_study.tex'}")

    recon_dir = PROJECT_ROOT / "results" / "reconciliation"
    recon_dir.mkdir(parents=True, exist_ok=True)
    with open(recon_dir / "event_study_primary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary: {recon_dir / 'event_study_primary.json'}")

    if args.compare_profiles:
        profile_to_summary: dict[str, dict] = {}
        for profile in sorted(METHODOLOGY_PROFILES.keys()):
            prof_results = run_event_study(asri=asri, events=CRISIS_EVENTS, profile=profile)
            profile_to_summary[profile] = summarize(prof_results, profile)

        with open(recon_dir / "event_study_method_comparison.json", "w", encoding="utf-8") as handle:
            json.dump(profile_to_summary, handle, indent=2)
        with open(tables_dir / "event_study_comparison.tex", "w", encoding="utf-8") as handle:
            handle.write(format_comparison_table(profile_to_summary))

        print(f"Saved comparison JSON: {recon_dir / 'event_study_method_comparison.json'}")
        print(f"Saved comparison table: {tables_dir / 'event_study_comparison.tex'}")


if __name__ == "__main__":
    main()
