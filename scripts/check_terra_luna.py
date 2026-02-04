#!/usr/bin/env python3
"""Check what's happening with Terra/Luna detection."""

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

# Terra/Luna: May 12, 2022
terra_date = datetime(2022, 5, 12)
window_start = terra_date - timedelta(days=30)

print("="*70)
print("TERRA/LUNA PRE-CRISIS WINDOW (30 days before May 12, 2022)")
print("="*70)

window = df[(df.index >= pd.Timestamp(window_start)) & (df.index < pd.Timestamp(terra_date))]

print(f"\nWindow: {window_start.strftime('%Y-%m-%d')} to {terra_date.strftime('%Y-%m-%d')}")
print(f"Data points: {len(window)}")
print()

print("ASRI values in pre-crisis window:")
print(window[['asri', 'stablecoin_risk', 'defi_liquidity_risk', 'contagion_risk', 'arbitrage_opacity']])

print()
print(f"ASRI Max: {window['asri'].max():.1f}")
print(f"ASRI Mean: {window['asri'].mean():.1f}")
print(f"Threshold: 50")
print(f"Detected (ASRI >= 50)? {window['asri'].max() >= 50}")

print()
print("="*70)
print("EVENT WINDOW (around May 12, 2022)")
print("="*70)

event_window = df[(df.index >= pd.Timestamp(terra_date - timedelta(days=5))) &
                   (df.index <= pd.Timestamp(terra_date + timedelta(days=10)))]

print(event_window[['asri', 'stablecoin_risk']])

print()
print("="*70)
print("WHAT HAPPENED TO STABLECOIN RISK?")
print("="*70)

# Check if stablecoin risk data exists for this period
print(f"\nStablecoin Risk in pre-crisis window:")
print(f"  Min: {window['stablecoin_risk'].min():.1f}")
print(f"  Max: {window['stablecoin_risk'].max():.1f}")
print(f"  Mean: {window['stablecoin_risk'].mean():.1f}")

# The Terra/Luna event should have caused a MASSIVE spike in stablecoin risk
# If it didn't, there's a data quality issue
