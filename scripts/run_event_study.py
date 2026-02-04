#!/usr/bin/env python3
"""Run event study analysis on actual data."""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CRISIS_EVENTS = [
    {"name": "Terra/Luna", "date": datetime(2022, 5, 12)},
    {"name": "Celsius/3AC", "date": datetime(2022, 6, 17)},
    {"name": "FTX Collapse", "date": datetime(2022, 11, 11)},
    {"name": "SVB Crisis", "date": datetime(2023, 3, 11)},
]

def run_event_study(asri, event_date, estimation_window=(-90, -31), event_window=(-30, 10)):
    """Run event study for a single event."""
    event_ts = pd.Timestamp(event_date)

    # Estimation window
    est_start = event_ts + pd.Timedelta(days=estimation_window[0])
    est_end = event_ts + pd.Timedelta(days=estimation_window[1])
    est_data = asri[(asri.index >= est_start) & (asri.index <= est_end)]

    pre_mean = est_data.mean()
    pre_std = est_data.std()

    # Event window
    evt_start = event_ts + pd.Timedelta(days=event_window[0])
    evt_end = event_ts + pd.Timedelta(days=event_window[1])
    evt_data = asri[(asri.index >= evt_start) & (asri.index <= evt_end)]

    # Peak in event window
    peak = evt_data.max()
    peak_date = evt_data.idxmax()

    # Abnormal signal
    abnormal = evt_data - pre_mean
    cas = abnormal.sum()

    # t-test
    n = len(abnormal)
    se = pre_std * np.sqrt(n)
    t_stat = cas / se if se > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))

    # Lead time (days before event with elevated ASRI)
    # Using 1.5 sigma threshold
    threshold = pre_mean + 1.5 * pre_std
    pre_event_data = asri[(asri.index >= evt_start) & (asri.index < event_ts)]
    elevated = pre_event_data[pre_event_data > threshold]
    if len(elevated) > 0:
        first_elevated = elevated.index.min()
        lead_days = (event_ts - first_elevated).days
    else:
        lead_days = 0

    return {
        "pre_mean": pre_mean,
        "pre_std": pre_std,
        "peak": peak,
        "cas": cas,
        "t_stat": t_stat,
        "p_value": p_value,
        "lead_days": lead_days,
        "significant": p_value < 0.01,
    }


def main():
    # Load data
    df = pd.read_parquet(PROJECT_ROOT / "results" / "data" / "asri_history.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)

    asri = df["asri"]

    print("="*70)
    print("EVENT STUDY RESULTS (ACTUAL DATA)")
    print("="*70)
    print()

    print(f"{'Event':<15} {'Date':<10} {'Pre-Mean':<10} {'Peak':<8} {'CAS':<12} {'t-stat':<10} {'Lead':<8} {'Sig'}")
    print("-"*90)

    total_lead = 0
    total_cas = 0
    sig_count = 0

    for event in CRISIS_EVENTS:
        result = run_event_study(asri, event["date"])
        sig = "***" if result["p_value"] < 0.01 else ("**" if result["p_value"] < 0.05 else "*" if result["p_value"] < 0.1 else "")

        print(f"{event['name']:<15} {event['date'].strftime('%Y-%m'):<10} {result['pre_mean']:<10.1f} {result['peak']:<8.1f} {result['cas']:<12.1f}{sig:<3} {result['t_stat']:<10.2f} {result['lead_days']:<8} {'Yes' if result['significant'] else 'No'}")

        total_lead += result["lead_days"]
        total_cas += result["cas"]
        if result["significant"]:
            sig_count += 1

    print("-"*90)
    print(f"{'Summary':<15} {'':<10} {'':<10} {'':<8} {total_cas/4:<12.1f} {'':<10} {total_lead/4:<8.0f} {sig_count}/4")
    print()

    print("\nLaTeX table format:")
    print("="*70)
    print(r"\begin{tabular}{lccccccc}")
    print(r"\toprule")
    print(r"Event & Date & Pre-Event & Peak & CAS & $t$-stat & Lead Days & Sig. \\")
    print(r"\midrule")

    for event in CRISIS_EVENTS:
        result = run_event_study(asri, event["date"])
        sig = "***" if result["p_value"] < 0.01 else ("**" if result["p_value"] < 0.05 else "*" if result["p_value"] < 0.1 else "")
        print(f"{event['name']} & {event['date'].strftime('%Y-%m')} & {result['pre_mean']:.1f} & {result['peak']:.1f} & {result['cas']:.1f}{sig} & {result['t_stat']:.2f} & {result['lead_days']} & {'Yes' if result['significant'] else 'No'} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
