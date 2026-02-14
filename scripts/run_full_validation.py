#!/usr/bin/env python3
"""
ASRI Full Validation Pipeline

Complete orchestration script that runs the entire validation suite for the
ASRI paper, generating all statistical tables and publication figures.

This script implements:
1. Data loading from parquet or fresh fetch
2. Stationarity tests (ADF, KPSS)
3. Event study around historical crises
4. Walk-forward validation with purged gaps
5. Out-of-sample testing (train 2021-2023, test 2024)
6. Sensitivity analysis (weight, threshold, window)
7. SVB anomaly investigation
8. Publication figure generation

Usage:
    python scripts/run_full_validation.py
    python scripts/run_full_validation.py --output-dir results/validation
    python scripts/run_full_validation.py --skip-figures
    python scripts/run_full_validation.py --fresh-data

Author: Farzulla Research
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Constants and Configuration
# =============================================================================

CRISIS_EVENTS = [
    {
        "name": "Terra/Luna",
        "date": datetime(2022, 5, 12),
        "description": "UST depeg and Luna death spiral",
        "severity": "extreme",
    },
    {
        "name": "Celsius/3AC",
        "date": datetime(2022, 6, 17),
        "description": "Celsius freeze and 3AC insolvency",
        "severity": "severe",
    },
    {
        "name": "FTX Collapse",
        "date": datetime(2022, 11, 11),
        "description": "FTX/Alameda fraud and bankruptcy",
        "severity": "extreme",
    },
    {
        "name": "SVB Crisis",
        "date": datetime(2023, 3, 11),
        "description": "Silicon Valley Bank failure, USDC depeg",
        "severity": "moderate",
    },
]

# Theoretical weights from the paper
THEORETICAL_WEIGHTS = {
    "stablecoin_risk": 0.30,
    "defi_liquidity_risk": 0.25,
    "contagion_risk": 0.25,
    "arbitrage_opacity": 0.20,
}

SUB_INDEX_COLUMNS = [
    "stablecoin_risk",
    "defi_liquidity_risk",
    "contagion_risk",
    "arbitrage_opacity",
]

EVENT_STUDY_PROFILE = "paper_v2"


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class ValidationSummary:
    """Complete validation results summary."""

    # Metadata
    timestamp: str
    n_observations: int
    date_range: tuple[str, str]

    # Stationarity
    stationarity_conclusions: dict[str, str]
    all_stationary: bool

    # Event study
    event_study_profile: str
    n_events: int
    n_significant_events: int
    avg_lead_days: float
    avg_t_statistic: float
    svb_t_statistic: float

    # Walk-forward
    wf_mean_r2: float
    wf_std_r2: float
    wf_n_folds: int

    # Out-of-sample
    oos_r2: float
    oos_mse: float
    oos_train_period: str
    oos_test_period: str

    # Sensitivity
    n_robust_components: int
    optimal_threshold: int
    optimal_window: int

    # Overall assessment
    validation_passed: bool
    notes: list[str] = field(default_factory=list)


# =============================================================================
# Full Validation Pipeline
# =============================================================================

class FullValidationPipeline:
    """
    Complete validation orchestrator for ASRI paper.

    Runs all validation tests and generates publication outputs.
    """

    def __init__(
        self,
        output_dir: Path | str = "results",
        paper_dir: Path | str = "paper",
        verbose: bool = True,
    ):
        """
        Initialize the validation pipeline.

        Args:
            output_dir: Directory for results, tables, and intermediate data
            paper_dir: Directory for publication figures
            verbose: Print progress messages
        """
        self.output_dir = Path(output_dir)
        self.paper_dir = Path(paper_dir)
        self.verbose = verbose

        # Create subdirectories
        self.data_dir = self.output_dir / "data"
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.paper_dir / "figures"

        for d in [self.data_dir, self.tables_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.data: pd.DataFrame | None = None
        self.results: dict[str, Any] = {}

    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(msg)

    # =========================================================================
    # Data Loading
    # =========================================================================

    def load_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load ASRI historical data from parquet or fetch fresh.

        Args:
            force_refresh: If True, fetch fresh data even if cache exists

        Returns:
            DataFrame with ASRI and sub-index columns
        """
        cache_path = self.data_dir / "asri_history.parquet"

        if cache_path.exists() and not force_refresh:
            self._log(f"Loading cached data from {cache_path}")
            self.data = pd.read_parquet(cache_path)
        else:
            self._log("Fetching fresh data (this may take a while)...")
            self.data = self._fetch_fresh_data()
            self.data.to_parquet(cache_path)
            self._log(f"Saved fresh data to {cache_path}")

        # Ensure datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if "date" in self.data.columns:
                self.data.set_index("date", inplace=True)
            self.data.index = pd.to_datetime(self.data.index)

        self._log(f"Loaded {len(self.data)} observations from {self.data.index.min().date()} to {self.data.index.max().date()}")

        return self.data

    def _fetch_fresh_data(self) -> pd.DataFrame:
        """Fetch fresh historical data using backtester."""
        import asyncio

        async def _fetch():
            from asri.backtest.backtest import ASRIBacktester

            backtester = ASRIBacktester()
            records = []

            start_date = datetime(2021, 1, 1)
            end_date = datetime.now()
            current = start_date

            while current <= end_date:
                try:
                    result = await backtester.calculate_for_date(current)
                    records.append({
                        "date": current,
                        "asri": result["asri"],
                        "stablecoin_risk": result["stablecoin_risk"],
                        "defi_liquidity_risk": result["defi_liquidity_risk"],
                        "contagion_risk": result["contagion_risk"],
                        "arbitrage_opacity": result["arbitrage_opacity"],
                    })
                except Exception as e:
                    self._log(f"  Warning: Failed for {current.date()}: {e}")

                current += timedelta(days=1)

            await backtester.close()

            df = pd.DataFrame(records)
            df.set_index("date", inplace=True)
            return df

        return asyncio.run(_fetch())

    # =========================================================================
    # Stationarity Tests
    # =========================================================================

    def run_stationarity_tests(self) -> dict:
        """
        Run ADF and KPSS stationarity tests on all sub-indices.

        Returns:
            Dictionary of StationarityResult objects
        """
        from asri.statistics.stationarity import (
            test_stationarity_suite,
            format_stationarity_table,
        )

        self._log("\n" + "="*60)
        self._log("STATIONARITY TESTS")
        self._log("="*60)

        sub_indices = self.data[SUB_INDEX_COLUMNS]

        # Add ASRI itself
        test_data = sub_indices.copy()
        test_data["asri"] = self.data["asri"]

        results = test_stationarity_suite(test_data)

        # Generate LaTeX table
        table = format_stationarity_table(results)
        table_path = self.tables_dir / "stationarity.tex"
        with open(table_path, "w") as f:
            f.write(table)
        self._log(f"Saved table: {table_path}")

        # Print summary
        for name, result in results.items():
            if result:
                self._log(f"  {name}: {result.conclusion.value} (ADF p={result.adf_pvalue:.3f}, KPSS p={result.kpss_pvalue:.3f})")

        self.results["stationarity"] = results
        return results

    # =========================================================================
    # Event Study
    # =========================================================================

    def run_event_study(self) -> list:
        """
        Run event study analysis around historical crises.

        Returns:
            List of EventStudyResult objects
        """
        from asri.validation.event_study import (
            CrisisEvent,
            run_event_study,
            format_event_study_table,
        )

        self._log("\n" + "="*60)
        self._log("EVENT STUDY ANALYSIS")
        self._log("="*60)

        events = [
            CrisisEvent(
                name=e["name"],
                event_date=e["date"],
                description=e["description"],
                severity=e["severity"],
            )
            for e in CRISIS_EVENTS
        ]

        asri = self.data["asri"]

        results = run_event_study(
            asri=asri,
            events=events,
            profile=EVENT_STUDY_PROFILE,
        )

        # Generate LaTeX table
        table = format_event_study_table(results)
        table_path = self.tables_dir / "event_study.tex"
        with open(table_path, "w") as f:
            f.write(table)
        self._log(f"Saved table: {table_path}")

        # Print summary
        for r in results:
            sig = "***" if r.p_value < 0.01 else ("**" if r.p_value < 0.05 else ("*" if r.p_value < 0.1 else ""))
            self._log(f"  {r.event.name}: t={r.t_statistic:.2f}{sig}, lead={r.lead_days}d, CAS={r.cumulative_abnormal_signal:.1f}")

        self.results["event_study"] = results
        return results

    # =========================================================================
    # Walk-Forward Validation
    # =========================================================================

    def run_walk_forward_validation(self) -> dict:
        """
        Run walk-forward validation with purged time series split.

        Returns:
            Dictionary with 'fixed' and 'optimized' WalkForwardResult
        """
        from asri.validation.walk_forward import (
            purged_walk_forward_cv,
            walk_forward_optimization,
            format_walk_forward_table,
        )

        self._log("\n" + "="*60)
        self._log("WALK-FORWARD VALIDATION")
        self._log("="*60)

        sub_indices = self.data[SUB_INDEX_COLUMNS]

        # Create target: forward "stress" (negative ASRI change)
        target = -self.data["asri"].pct_change(30).shift(-30)

        # Fixed weights validation
        self._log("Running fixed-weight validation...")
        fixed_result = purged_walk_forward_cv(
            sub_indices=sub_indices,
            target=target,
            weights=THEORETICAL_WEIGHTS,
            n_splits=5,
            purge_days=30,
        )

        # Optimized weights validation
        self._log("Running optimized-weight validation...")
        optimized_result = walk_forward_optimization(
            sub_indices=sub_indices,
            target=target,
            n_splits=5,
            purge_days=30,
            optimization_method="elastic_net",
        )

        # Generate LaTeX tables
        with open(self.tables_dir / "walk_forward_fixed.tex", "w") as f:
            f.write(format_walk_forward_table(fixed_result))

        with open(self.tables_dir / "walk_forward_optimized.tex", "w") as f:
            f.write(format_walk_forward_table(optimized_result))

        self._log(f"  Fixed weights: Mean R2={fixed_result.mean_test_r2:.3f} +/- {fixed_result.std_test_r2:.3f}")
        self._log(f"  Optimized:     Mean R2={optimized_result.mean_test_r2:.3f} +/- {optimized_result.std_test_r2:.3f}")
        self._log(f"  Weight stability: {optimized_result.weight_stability:.3f}")

        results = {"fixed": fixed_result, "optimized": optimized_result}
        self.results["walk_forward"] = results
        return results

    # =========================================================================
    # Out-of-Sample Test
    # =========================================================================

    def run_out_of_sample_test(self) -> dict:
        """
        Out-of-sample test: train on 2021-2023, test on 2024.

        Returns:
            Dictionary with OOS metrics
        """
        self._log("\n" + "="*60)
        self._log("OUT-OF-SAMPLE TEST (Train: 2021-2023, Test: 2024)")
        self._log("="*60)

        # Split data
        train_end = datetime(2023, 12, 31)
        test_start = datetime(2024, 1, 1)

        train_data = self.data[self.data.index <= train_end]
        test_data = self.data[self.data.index >= test_start]

        if len(test_data) < 30:
            self._log("  WARNING: Insufficient test data for OOS validation")
            results = {
                "train_period": f"2021-01 to {train_end.strftime('%Y-%m')}",
                "test_period": "Insufficient data",
                "train_n": len(train_data),
                "test_n": len(test_data),
                "test_r2": np.nan,
                "test_mse": np.nan,
            }
            self.results["oos"] = results
            return results

        sub_cols = SUB_INDEX_COLUMNS

        # Compute ASRI with theoretical weights
        def compute_asri(df):
            return sum(THEORETICAL_WEIGHTS[c] * df[c] for c in sub_cols)

        # Target: forward stress
        target_train = -train_data["asri"].pct_change(30).shift(-30).dropna()
        target_test = -test_data["asri"].pct_change(30).shift(-30).dropna()

        asri_train = compute_asri(train_data).loc[target_train.index]
        asri_test = compute_asri(test_data).loc[target_test.index]

        # Metrics
        def r2(y_pred, y_true):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - ss_res / ss_tot if ss_tot > 0 else 0

        train_r2 = r2(asri_train.values, target_train.values)
        test_r2 = r2(asri_test.values, target_test.values)
        test_mse = np.mean((asri_test.values - target_test.values) ** 2)

        # Spearman correlation
        from scipy import stats
        train_corr, _ = stats.spearmanr(asri_train.values, target_train.values)
        test_corr, _ = stats.spearmanr(asri_test.values, target_test.values)

        results = {
            "train_period": f"2021-01 to {train_end.strftime('%Y-%m')}",
            "test_period": f"{test_start.strftime('%Y-%m')} to {test_data.index.max().strftime('%Y-%m')}",
            "train_n": len(target_train),
            "test_n": len(target_test),
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_mse": test_mse,
            "train_spearman": train_corr,
            "test_spearman": test_corr,
        }

        self._log(f"  Train: n={len(target_train)}, R2={train_r2:.3f}, Spearman={train_corr:.3f}")
        self._log(f"  Test:  n={len(target_test)}, R2={test_r2:.3f}, Spearman={test_corr:.3f}")

        # Generate LaTeX table
        table = self._format_oos_table(results)
        with open(self.tables_dir / "out_of_sample.tex", "w") as f:
            f.write(table)

        self.results["oos"] = results
        return results

    def _format_oos_table(self, results: dict) -> str:
        """Format OOS results as LaTeX table."""
        return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Out-of-Sample Validation Results}}
\\label{{tab:oos}}
\\small
\\begin{{tabular}}{{lcc}}
\\toprule
Metric & Training & Test \\\\
\\midrule
Period & {results['train_period']} & {results['test_period']} \\\\
Observations & {results['train_n']} & {results['test_n']} \\\\
$R^2$ & {results['train_r2']:.3f} & {results['test_r2']:.3f} \\\\
Spearman $\\rho$ & {results['train_spearman']:.3f} & {results['test_spearman']:.3f} \\\\
MSE & -- & {results['test_mse']:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item Training on 2021--2023, testing on 2024 data.
\\item No data leakage: 30-day purge gap enforced.
\\end{{tablenotes}}
\\end{{table}}
"""

    # =========================================================================
    # Sensitivity Analysis
    # =========================================================================

    def run_sensitivity_analysis(self) -> dict:
        """
        Run comprehensive sensitivity analysis.

        Returns:
            Dictionary with weight, threshold, and window results
        """
        from asri.validation.sensitivity import (
            run_full_sensitivity_analysis,
            format_sensitivity_table,
            format_threshold_table,
            format_window_table,
        )

        self._log("\n" + "="*60)
        self._log("SENSITIVITY ANALYSIS")
        self._log("="*60)

        sub_indices = self.data[SUB_INDEX_COLUMNS]
        asri = self.data["asri"]

        # Create returns proxy for window sensitivity
        returns = asri.pct_change().dropna()

        crisis_dates = [e["date"] for e in CRISIS_EVENTS]

        weight_results, threshold_result, window_result = run_full_sensitivity_analysis(
            sub_indices=sub_indices,
            weights=THEORETICAL_WEIGHTS,
            returns=returns,
            crisis_dates=crisis_dates,
            asri=asri,
        )

        # Generate LaTeX tables
        with open(self.tables_dir / "weight_sensitivity.tex", "w") as f:
            f.write(format_sensitivity_table(weight_results))

        with open(self.tables_dir / "threshold_sensitivity.tex", "w") as f:
            f.write(format_threshold_table(threshold_result))

        with open(self.tables_dir / "window_sensitivity.tex", "w") as f:
            f.write(format_window_table(window_result))

        # Summary
        n_robust = sum(1 for r in weight_results if r.is_robust)
        self._log(f"  Weight robustness: {n_robust}/{len(weight_results)} components robust")
        self._log(f"  Optimal threshold: {threshold_result.optimal_threshold} (F1={threshold_result.f1_at_threshold[threshold_result.optimal_threshold]:.3f})")
        self._log(f"  Optimal window: {window_result.optimal_window} days (AUC={window_result.auc_roc_by_window[window_result.optimal_window]:.3f})")

        results = {
            "weight": weight_results,
            "threshold": threshold_result,
            "window": window_result,
        }
        self.results["sensitivity"] = results
        return results

    # =========================================================================
    # SVB Anomaly Investigation
    # =========================================================================

    def investigate_svb_anomaly(self) -> dict:
        """
        Investigate why SVB has t-stat of 34.54 (5x higher than others).

        This is a critical diagnostic to understand if the result is
        genuine or an artifact of data/methodology issues.

        Returns:
            Dictionary with investigation findings
        """
        self._log("\n" + "="*60)
        self._log("SVB ANOMALY INVESTIGATION")
        self._log("="*60)

        svb_date = datetime(2023, 3, 11)
        svb_ts = pd.Timestamp(svb_date)

        asri = self.data["asri"]

        findings = {
            "event_date": svb_date.isoformat(),
            "hypothesis_tests": [],
        }

        # 1. Check data availability around SVB
        window_60 = asri[(asri.index >= svb_ts - pd.Timedelta(days=90)) &
                         (asri.index <= svb_ts + pd.Timedelta(days=30))]

        findings["data_points_in_window"] = len(window_60)
        findings["missing_data_pct"] = window_60.isna().mean() * 100

        self._log(f"  Data points in [-90, +30] window: {len(window_60)}")
        self._log(f"  Missing data: {findings['missing_data_pct']:.1f}%")

        # 2. Compare estimation window variance to other events
        for event in CRISIS_EVENTS:
            evt_ts = pd.Timestamp(event["date"])
            est_start = evt_ts - pd.Timedelta(days=90)
            est_end = evt_ts - pd.Timedelta(days=31)

            est_data = asri[(asri.index >= est_start) & (asri.index <= est_end)]

            if len(est_data) > 10:
                findings[f"{event['name']}_est_std"] = est_data.std()
                self._log(f"  {event['name']} estimation std: {est_data.std():.2f}")

        # 3. Check if SVB estimation window is unusually quiet
        svb_est_start = svb_ts - pd.Timedelta(days=90)
        svb_est_end = svb_ts - pd.Timedelta(days=31)
        svb_est = asri[(asri.index >= svb_est_start) & (asri.index <= svb_est_end)]

        if len(svb_est) > 10:
            svb_std = svb_est.std()
            svb_mean = svb_est.mean()

            # Compare to overall ASRI variance
            overall_std = asri.std()

            findings["svb_estimation_std"] = svb_std
            findings["overall_asri_std"] = overall_std
            findings["std_ratio"] = svb_std / overall_std if overall_std > 0 else np.nan

            self._log(f"  SVB estimation std: {svb_std:.2f} (overall: {overall_std:.2f})")
            self._log(f"  Ratio: {findings['std_ratio']:.2f}")

        # 4. Check ASRI spike magnitude
        svb_window = asri[(asri.index >= svb_ts - pd.Timedelta(days=30)) &
                         (asri.index <= svb_ts)]

        if len(svb_window) > 0:
            svb_peak = svb_window.max()
            svb_change = svb_peak - svb_mean if len(svb_est) > 10 else svb_peak

            findings["svb_peak_asri"] = svb_peak
            findings["svb_asri_change"] = svb_change
            findings["svb_z_score"] = svb_change / svb_std if svb_std > 0 else np.nan

            self._log(f"  SVB peak ASRI: {svb_peak:.1f}")
            self._log(f"  Change from baseline: {svb_change:.1f}")
            self._log(f"  Z-score: {findings['svb_z_score']:.2f}")

        # 5. Hypothesis: USDC depeg contributed to extreme signal
        # Check stablecoin risk component specifically
        if "stablecoin_risk" in self.data.columns:
            stab_window = self.data["stablecoin_risk"][(self.data.index >= svb_ts - pd.Timedelta(days=30)) &
                                                        (self.data.index <= svb_ts)]
            if len(stab_window) > 0:
                findings["svb_stablecoin_peak"] = stab_window.max()
                self._log(f"  Stablecoin risk peak at SVB: {stab_window.max():.1f}")

        # 6. Conclusion
        hypotheses = []

        if findings.get("std_ratio", 1) < 0.7:
            hypotheses.append("Low estimation window variance inflates t-statistic (classic statistical artifact)")

        if findings.get("svb_z_score", 0) > 5:
            hypotheses.append("Genuine extreme signal: SVB caused outsized ASRI spike due to USDC depeg cascading through stablecoin component")

        if findings.get("svb_stablecoin_peak", 0) > 80:
            hypotheses.append("Stablecoin risk component dominated during SVB (expected given USDC depeg)")

        findings["hypotheses"] = hypotheses

        self._log("\n  HYPOTHESES:")
        for h in hypotheses:
            self._log(f"    - {h}")

        # Save detailed findings
        with open(self.output_dir / "svb_investigation.json", "w") as f:
            json.dump(findings, f, indent=2, default=str)

        self.results["svb_investigation"] = findings
        return findings

    # =========================================================================
    # Figure Generation
    # =========================================================================

    def generate_all_figures(self) -> None:
        """Generate all publication-quality figures."""
        from asri.publication.figures import (
            plot_asri_time_series,
            plot_sub_index_decomposition,
            plot_event_study_panels,
            plot_sensitivity_heatmaps,
        )

        self._log("\n" + "="*60)
        self._log("GENERATING FIGURES")
        self._log("="*60)

        asri = self.data["asri"]
        sub_indices = self.data[SUB_INDEX_COLUMNS]

        crisis_dates = [e["date"] for e in CRISIS_EVENTS]
        crisis_labels = [e["name"] for e in CRISIS_EVENTS]

        # 1. ASRI Time Series
        self._log("  Generating ASRI time series plot...")
        plot_asri_time_series(
            asri=asri,
            crisis_dates=crisis_dates,
            crisis_labels=crisis_labels,
            output_path=str(self.figures_dir / "asri_timeseries.pdf"),
        )

        # 2. Sub-index Decomposition
        self._log("  Generating decomposition plot...")
        plot_sub_index_decomposition(
            sub_indices=sub_indices,
            weights=THEORETICAL_WEIGHTS,
            output_path=str(self.figures_dir / "decomposition.pdf"),
        )

        # 3. Event Study Panels
        if "event_study" in self.results:
            self._log("  Generating event study panels...")
            event_results = []
            for r in self.results["event_study"]:
                event_results.append({
                    "name": r.event.name,
                    "date": r.event.event_date,
                    "asri_window": r.asri_trajectory,
                    "pre_mean": r.pre_event_mean,
                    "post_mean": r.peak_asri,
                })

            plot_event_study_panels(
                event_results=event_results,
                output_path=str(self.figures_dir / "event_study.pdf"),
            )

        # 4. Sensitivity Heatmaps
        if "sensitivity" in self.results:
            self._log("  Generating sensitivity heatmaps...")

            # Convert WeightSensitivityResult to dict format expected by plot function
            weight_results = []
            for r in self.results["sensitivity"]["weight"]:
                for i, delta in enumerate(r.perturbation_grid):
                    weight_results.append({
                        "sub_index": r.component,
                        "perturbation": delta,
                        "mean_asri": r.asri_means[i],
                        "std_asri": r.asri_stds[i],
                        "max_asri": r.asri_means[i] + r.asri_stds[i],
                    })

            plot_sensitivity_heatmaps(
                weight_results=weight_results,
                output_path=str(self.figures_dir / "sensitivity.pdf"),
            )

        self._log(f"  Figures saved to {self.figures_dir}/")

    # =========================================================================
    # Main Orchestration
    # =========================================================================

    def run(self, skip_figures: bool = False, fresh_data: bool = False) -> ValidationSummary:
        """
        Execute the full validation pipeline.

        Args:
            skip_figures: Skip figure generation
            fresh_data: Force fresh data fetch

        Returns:
            ValidationSummary with all results
        """
        start_time = datetime.now()

        self._log("\n" + "="*70)
        self._log("ASRI FULL VALIDATION PIPELINE")
        self._log("="*70)
        self._log(f"Started at: {start_time.isoformat()}")
        self._log(f"Output directory: {self.output_dir}")
        self._log(f"Paper figures: {self.figures_dir}")

        # 1. Load data
        self.load_data(force_refresh=fresh_data)

        # 2. Run all validation tests
        self.run_stationarity_tests()
        event_results = self.run_event_study()
        wf_results = self.run_walk_forward_validation()
        oos_results = self.run_out_of_sample_test()
        sens_results = self.run_sensitivity_analysis()
        svb_results = self.investigate_svb_anomaly()

        # 3. Generate figures
        if not skip_figures:
            self.generate_all_figures()

        # 4. Create summary
        summary = self._create_summary()

        # 5. Save summary
        summary_path = self.output_dir / "validation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(asdict(summary), f, indent=2, default=str)

        # 6. Print final report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self._log("\n" + "="*70)
        self._log("VALIDATION COMPLETE")
        self._log("="*70)
        self._log(f"Duration: {duration:.1f} seconds")
        self._log(f"Observations: {summary.n_observations}")
        self._log(f"Date range: {summary.date_range[0]} to {summary.date_range[1]}")
        self._log("")
        self._log("KEY METRICS:")
        self._log(f"  Event study: {summary.n_significant_events}/{summary.n_events} significant")
        self._log(f"  Average lead time: {summary.avg_lead_days:.1f} days")
        self._log(f"  Walk-forward R2: {summary.wf_mean_r2:.3f} +/- {summary.wf_std_r2:.3f}")
        self._log(f"  Out-of-sample R2: {summary.oos_r2:.3f}")
        self._log(f"  Robust components: {summary.n_robust_components}/4")
        self._log(f"  SVB t-statistic: {summary.svb_t_statistic:.2f}")
        self._log("")
        self._log(f"VALIDATION {'PASSED' if summary.validation_passed else 'NEEDS ATTENTION'}!")

        if summary.notes:
            self._log("\nNOTES:")
            for note in summary.notes:
                self._log(f"  - {note}")

        self._log(f"\nResults saved to: {summary_path}")

        return summary

    def _create_summary(self) -> ValidationSummary:
        """Create validation summary from results."""
        notes = []

        # Stationarity
        stat_results = self.results.get("stationarity", {})
        stat_conclusions = {}
        all_stationary = True
        for name, result in stat_results.items():
            if result:
                stat_conclusions[name] = result.conclusion.value
                if result.conclusion.value != "stationary":
                    all_stationary = False

        # Event study
        event_results = self.results.get("event_study", [])
        n_significant = sum(1 for r in event_results if r.is_significant)
        avg_lead = np.mean([r.lead_days for r in event_results]) if event_results else 0
        avg_t = np.mean([r.t_statistic for r in event_results]) if event_results else 0

        svb_t = 0
        for r in event_results:
            if "SVB" in r.event.name:
                svb_t = r.t_statistic
                if svb_t > 30:
                    notes.append(f"SVB t-statistic ({svb_t:.2f}) is unusually high - see investigation")

        # Walk-forward
        wf = self.results.get("walk_forward", {})
        wf_fixed = wf.get("fixed")
        wf_mean_r2 = wf_fixed.mean_test_r2 if wf_fixed else 0
        wf_std_r2 = wf_fixed.std_test_r2 if wf_fixed else 0
        wf_n_folds = wf_fixed.n_folds if wf_fixed else 0

        # OOS
        oos = self.results.get("oos", {})
        oos_r2 = oos.get("test_r2", np.nan)
        oos_mse = oos.get("test_mse", np.nan)

        # Sensitivity
        sens = self.results.get("sensitivity", {})
        weight_results = sens.get("weight", [])
        n_robust = sum(1 for r in weight_results if r.is_robust)

        threshold_result = sens.get("threshold")
        optimal_threshold = threshold_result.optimal_threshold if threshold_result else 70

        window_result = sens.get("window")
        optimal_window = window_result.optimal_window if window_result else 30

        # Validation passed?
        validation_passed = (
            n_significant >= 3 and  # At least 3/4 events significant
            wf_mean_r2 > 0 and  # Positive OOS R2
            n_robust >= 2  # At least 2/4 components robust
        )

        if not all_stationary:
            notes.append("Some series may be non-stationary - consider differencing")

        if n_robust < 3:
            notes.append(f"Only {n_robust}/4 components are robust to weight perturbations")

        return ValidationSummary(
            timestamp=datetime.now().isoformat(),
            n_observations=len(self.data) if self.data is not None else 0,
            date_range=(
                self.data.index.min().strftime("%Y-%m-%d") if self.data is not None else "",
                self.data.index.max().strftime("%Y-%m-%d") if self.data is not None else "",
            ),
            stationarity_conclusions=stat_conclusions,
            all_stationary=all_stationary,
            event_study_profile=EVENT_STUDY_PROFILE,
            n_events=len(event_results),
            n_significant_events=n_significant,
            avg_lead_days=avg_lead,
            avg_t_statistic=avg_t,
            svb_t_statistic=svb_t,
            wf_mean_r2=wf_mean_r2,
            wf_std_r2=wf_std_r2,
            wf_n_folds=wf_n_folds,
            oos_r2=oos_r2,
            oos_mse=oos_mse,
            oos_train_period=oos.get("train_period", ""),
            oos_test_period=oos.get("test_period", ""),
            n_robust_components=n_robust,
            optimal_threshold=optimal_threshold,
            optimal_window=optimal_window,
            validation_passed=validation_passed,
            notes=notes,
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="ASRI Full Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_full_validation.py
    python scripts/run_full_validation.py --output-dir results/validation
    python scripts/run_full_validation.py --skip-figures
    python scripts/run_full_validation.py --fresh-data
    python scripts/run_full_validation.py --quiet
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Output directory for results and tables (default: results/)",
    )

    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=PROJECT_ROOT / "paper",
        help="Paper directory for figures (default: paper/)",
    )

    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation",
    )

    parser.add_argument(
        "--fresh-data",
        action="store_true",
        help="Force fresh data fetch (ignore cache)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    pipeline = FullValidationPipeline(
        output_dir=args.output_dir,
        paper_dir=args.paper_dir,
        verbose=not args.quiet,
    )

    summary = pipeline.run(
        skip_figures=args.skip_figures,
        fresh_data=args.fresh_data,
    )

    # Exit with error code if validation failed
    sys.exit(0 if summary.validation_passed else 1)


if __name__ == "__main__":
    main()
