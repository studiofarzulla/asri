#!/usr/bin/env python3
"""Check ASRI values across the full crisis period."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
df = pd.read_parquet(PROJECT_ROOT / "results" / "data" / "asri_history.parquet")

if not isinstance(df.index, pd.DatetimeIndex):
    if "date" in df.columns:
        df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)

crises = [
    ("Terra/Luna", datetime(2022, 5, 12)),
    ("Celsius/3AC", datetime(2022, 6, 17)),
    ("FTX Collapse", datetime(2022, 11, 11)),
    ("SVB Crisis", datetime(2023, 3, 11)),
]

for name, date in crises:
    print("="*70)
    print(f"{name} ({date.strftime('%Y-%m-%d')})")
    print("="*70)

    # Check wider window
    start = date - timedelta(days=60)
    end = date + timedelta(days=30)

    window = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    # Find max ASRI in this period
    max_row = window['asri'].idxmax()
    max_val = window['asri'].max()

    # Check 30-day pre-crisis
    pre_start = date - timedelta(days=30)
    pre_window = df[(df.index >= pd.Timestamp(pre_start)) & (df.index < pd.Timestamp(date))]
    pre_max = pre_window['asri'].max()
    pre_detected = pre_max >= 50

    # Find first breach
    breaches = pre_window[pre_window['asri'] >= 50]
    if len(breaches) > 0:
        first_breach = breaches.index.min()
        lead_days = (pd.Timestamp(date) - first_breach).days
    else:
        first_breach = None
        lead_days = 0

    print(f"Pre-crisis window (30d): max ASRI = {pre_max:.1f}")
    print(f"Detected (threshold 50)? {pre_detected}")
    print(f"First breach: {first_breach}")
    print(f"Lead time: {lead_days} days")
    print()
    print(f"Overall period max: {max_val:.1f} on {max_row.strftime('%Y-%m-%d')}")
    print()

    # Show ASRI trajectory
    print("Daily ASRI around crisis:")
    crisis_range = df[(df.index >= pd.Timestamp(date - timedelta(days=10))) &
                      (df.index <= pd.Timestamp(date + timedelta(days=10)))]
    for idx, row in crisis_range.iterrows():
        marker = " <-- CRISIS" if idx.date() == date.date() else ""
        thresh_mark = " [ABOVE 50]" if row['asri'] >= 50 else ""
        print(f"  {idx.strftime('%Y-%m-%d')}: {row['asri']:.1f}{thresh_mark}{marker}")
    print()
