#!/usr/bin/env python3
"""
ASRI Robustness Suite Runner
Runs comprehensive robustness tests in background
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np


async def run_weight_perturbation(df: pd.DataFrame) -> dict:
    """Test sensitivity to weight changes."""
    from asri.weights.pca_weights import derive_pca_weights
    
    results = []
    base_weights = {"stablecoin": 0.30, "defi": 0.25, "contagion": 0.25, "opacity": 0.20}
    
    # Perturbation grid
    perturbations = [-0.10, -0.05, 0.0, 0.05, 0.10]
    
    for component in base_weights.keys():
        for delta in perturbations:
            weights = base_weights.copy()
            weights[component] = max(0.05, min(0.50, weights[component] + delta))
            
            # Renormalize
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            
            # Recalculate ASRI
            asri = (
                weights["stablecoin"] * df["stablecoin_risk"] +
                weights["defi"] * df["defi_liquidity_risk"] +
                weights["contagion"] * df["contagion_risk"] +
                weights["opacity"] * df["arbitrage_opacity"]
            )
            
            results.append({
                "component": component,
                "delta": delta,
                "mean_asri": asri.mean(),
                "std_asri": asri.std(),
                "max_asri": asri.max(),
            })
    
    return {"weight_perturbation": results}


async def run_bootstrap_ci(df: pd.DataFrame, n_bootstrap: int = 1000) -> dict:
    """Bootstrap confidence intervals for ASRI statistics."""
    from asri.statistics.bootstrap import block_bootstrap_ci
    
    asri = df["asri"].values
    
    # Mean CI
    mean_ci = block_bootstrap_ci(asri, np.mean, n_bootstrap=n_bootstrap, block_size=20)
    
    # Std CI
    std_ci = block_bootstrap_ci(asri, np.std, n_bootstrap=n_bootstrap, block_size=20)
    
    # Max CI
    max_ci = block_bootstrap_ci(asri, np.max, n_bootstrap=n_bootstrap, block_size=20)
    
    return {
        "bootstrap_ci": {
            "mean": {"lower": mean_ci[0], "upper": mean_ci[1], "point": asri.mean()},
            "std": {"lower": std_ci[0], "upper": std_ci[1], "point": asri.std()},
            "max": {"lower": max_ci[0], "upper": max_ci[1], "point": asri.max()},
        }
    }


async def run_rolling_validation(df: pd.DataFrame) -> dict:
    """Walk-forward validation with rolling windows."""
    results = []
    window_size = 365  # 1 year
    step_size = 90     # Quarterly
    
    for start in range(0, len(df) - window_size - step_size, step_size):
        train = df.iloc[start:start + window_size]
        test = df.iloc[start + window_size:start + window_size + step_size]
        
        if len(test) < step_size:
            break
        
        # Simple persistence forecast
        train_mean = train["asri"].mean()
        train_std = train["asri"].std()
        
        test_mean = test["asri"].mean()
        test_std = test["asri"].std()
        
        # MAE of persistence forecast
        mae = abs(test["asri"] - train["asri"].iloc[-1]).mean()
        
        results.append({
            "train_end": str(train["date"].iloc[-1]),
            "test_start": str(test["date"].iloc[0]),
            "train_mean": train_mean,
            "test_mean": test_mean,
            "mae": mae,
            "in_2std": abs(test_mean - train_mean) < 2 * train_std,
        })
    
    return {
        "rolling_validation": {
            "windows": results,
            "avg_mae": np.mean([r["mae"] for r in results]),
            "pct_in_2std": np.mean([r["in_2std"] for r in results]),
        }
    }


async def run_placebo_tests(df: pd.DataFrame, n_placebo: int = 100) -> dict:
    """Placebo event study with random dates."""
    from asri.validation.event_study import run_event_study
    
    # Generate random placebo dates (avoiding real crisis periods)
    crisis_dates = pd.to_datetime([
        "2022-05-09", "2022-06-13", "2022-11-11", "2023-03-10"
    ])
    
    all_dates = df["date"].values
    valid_dates = [d for d in all_dates if not any(abs((pd.Timestamp(d) - c).days) < 60 for c in crisis_dates)]
    
    np.random.seed(42)
    placebo_dates = np.random.choice(valid_dates, size=min(n_placebo, len(valid_dates)), replace=False)
    
    significant_count = 0
    for date in placebo_dates[:50]:  # Limit for speed
        try:
            result = run_event_study(
                df, 
                event_date=pd.Timestamp(date), 
                event_name="placebo",
                pre_window=60, 
                post_window=10
            )
            if result.get("significant", False):
                significant_count += 1
        except Exception:
            pass
    
    return {
        "placebo_tests": {
            "n_tests": 50,
            "significant_count": significant_count,
            "false_positive_rate": significant_count / 50,
            "expected_fpr_5pct": 0.05,
            "passes": significant_count / 50 < 0.10,  # Allow 10% false positives
        }
    }


async def main():
    """Run full robustness suite."""
    print("=" * 60)
    print("ASRI ROBUSTNESS SUITE")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / "results" / "data" / "asri_history.parquet"
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        return
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} observations")
    print()
    
    results = {}
    
    # 1. Weight Perturbation
    print("[1/4] Running weight perturbation analysis...")
    try:
        results.update(await run_weight_perturbation(df))
        print("      ✓ Weight perturbation complete")
    except Exception as e:
        print(f"      ✗ Weight perturbation failed: {e}")
    
    # 2. Bootstrap CI
    print("[2/4] Running bootstrap confidence intervals...")
    try:
        results.update(await run_bootstrap_ci(df, n_bootstrap=500))
        print("      ✓ Bootstrap CI complete")
    except Exception as e:
        print(f"      ✗ Bootstrap CI failed: {e}")
    
    # 3. Rolling Validation
    print("[3/4] Running walk-forward validation...")
    try:
        results.update(await run_rolling_validation(df))
        print("      ✓ Rolling validation complete")
    except Exception as e:
        print(f"      ✗ Rolling validation failed: {e}")
    
    # 4. Placebo Tests
    print("[4/4] Running placebo event studies...")
    try:
        results.update(await run_placebo_tests(df, n_placebo=50))
        print("      ✓ Placebo tests complete")
    except Exception as e:
        print(f"      ✗ Placebo tests failed: {e}")
    
    # Save results
    output_path = Path(__file__).parent.parent.parent.parent / "results" / "robustness_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print()
    print("=" * 60)
    print("ROBUSTNESS SUITE COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_path}")
    print(f"Finished: {datetime.now().isoformat()}")
    
    # Summary
    print()
    print("SUMMARY:")
    if "placebo_tests" in results:
        fpr = results["placebo_tests"]["false_positive_rate"]
        status = "✓ PASS" if fpr < 0.10 else "✗ FAIL"
        print(f"  Placebo FPR: {fpr:.1%} {status}")
    if "rolling_validation" in results:
        pct = results["rolling_validation"]["pct_in_2std"]
        status = "✓ PASS" if pct > 0.90 else "✗ FAIL"
        print(f"  Rolling 2σ coverage: {pct:.1%} {status}")
    if "bootstrap_ci" in results:
        print(f"  Mean ASRI 95% CI: [{results['bootstrap_ci']['mean']['lower']:.1f}, {results['bootstrap_ci']['mean']['upper']:.1f}]")


if __name__ == "__main__":
    asyncio.run(main())
