#!/usr/bin/env python3
"""
ASRI Rigorous Validation Pipeline

Master script that runs the complete statistical validation:
1. Stationarity tests on all sub-indices
2. Granger causality analysis
3. Weight derivation (PCA, Elastic Net, Granger)
4. Walk-forward out-of-sample validation
5. Event study around historical crises
6. ROC/AUC crisis prediction metrics
7. Benchmark comparison
8. Robustness tests (placebo, structural breaks)
9. Regime detection
10. Generate publication-ready tables and figures

Usage:
    python -m asri.scripts.run_validation --output-dir results/
"""

import argparse
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


# Crisis events for validation
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


class ValidationPipeline:
    """Master validation pipeline for ASRI."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tables_dir = output_dir / "tables"
        self.figures_dir = output_dir / "figures"
        self.data_dir = output_dir / "data"
        
        for d in [self.tables_dir, self.figures_dir, self.data_dir]:
            d.mkdir(exist_ok=True)
        
        self.results = {}
    
    async def load_data(self) -> pd.DataFrame:
        """Load historical ASRI data."""
        from asri.backtest.historical import HistoricalDataFetcher
        from asri.backtest.backtest import ASRIBacktester
        
        logger.info("Loading historical data...")
        
        # Try to load from cache first
        cache_path = self.data_dir / "asri_history.parquet"
        
        if cache_path.exists():
            logger.info("Loading from cache", path=str(cache_path))
            df = pd.read_parquet(cache_path)
        else:
            logger.info("Fetching historical data (this may take a while)...")
            
            backtester = ASRIBacktester()
            
            # Fetch data for validation period
            start_date = datetime(2021, 1, 1)
            end_date = datetime(2025, 12, 31)
            
            records = []
            current = start_date
            
            while current <= end_date:
                try:
                    result = await backtester.calculate_for_date(current)
                    records.append({
                        'date': current,
                        'asri': result['asri'],
                        'stablecoin_risk': result['stablecoin_risk'],
                        'defi_liquidity_risk': result['defi_liquidity_risk'],
                        'contagion_risk': result['contagion_risk'],
                        'arbitrage_opacity': result['arbitrage_opacity'],
                    })
                except Exception as e:
                    logger.warning(f"Failed for {current.date()}", error=str(e))
                
                current += timedelta(days=7)  # Weekly for speed
            
            await backtester.close()
            
            df = pd.DataFrame(records)
            df.set_index('date', inplace=True)
            df.to_parquet(cache_path)
        
        return df
    
    def run_stationarity_tests(self, data: pd.DataFrame) -> dict:
        """Run stationarity tests on all sub-indices."""
        from asri.statistics.stationarity import (
            test_stationarity_suite,
            format_stationarity_table,
        )
        
        logger.info("Running stationarity tests...")
        
        sub_indices = data[['stablecoin_risk', 'defi_liquidity_risk', 
                           'contagion_risk', 'arbitrage_opacity']]
        
        results = test_stationarity_suite(sub_indices)
        
        # Generate LaTeX table
        table = format_stationarity_table(results)
        with open(self.tables_dir / "stationarity.tex", "w") as f:
            f.write(table)
        
        return results
    
    def run_granger_causality(self, data: pd.DataFrame) -> dict:
        """Run Granger causality tests."""
        from asri.statistics.causality import (
            granger_causality_matrix,
            format_granger_table,
            compute_predictive_weights,
        )
        
        logger.info("Running Granger causality tests...")
        
        # Create forward drawdown target
        asri = data['asri']
        forward_return = asri.pct_change(30).shift(-30)
        data_with_target = data.copy()
        data_with_target['forward_drawdown'] = -forward_return  # Negative = worse
        
        sub_indices = data[['stablecoin_risk', 'defi_liquidity_risk',
                           'contagion_risk', 'arbitrage_opacity']]
        
        matrix = granger_causality_matrix(
            sub_indices,
            target_column=None,  # Test all pairs
            lags=[1, 7, 14, 30],
        )
        
        # Derive weights from Granger F-stats
        granger_weights = compute_predictive_weights(
            matrix, 
            target='forward_drawdown' if 'forward_drawdown' in data_with_target.columns else list(sub_indices.columns)[0],
            lag=7,
        )
        
        # Generate table
        table = format_granger_table(matrix, target='asri')
        with open(self.tables_dir / "granger_causality.tex", "w") as f:
            f.write(table)
        
        return {
            'matrix': matrix,
            'granger_weights': granger_weights,
            'significant_pairs': matrix.significant_at(0.05),
        }
    
    def derive_weights(self, data: pd.DataFrame) -> dict:
        """Derive weights using all methods."""
        from asri.weights.pca import PCAWeightDeriver, format_pca_table
        from asri.weights.elastic_net import ElasticNetWeightDeriver, format_elastic_net_table
        from asri.weights.comparison import compare_weight_methods, format_weight_comparison_table
        
        logger.info("Deriving weights via PCA and Elastic Net...")
        
        sub_indices = data[['stablecoin_risk', 'defi_liquidity_risk',
                           'contagion_risk', 'arbitrage_opacity']]
        
        # Create target
        asri = data['asri']
        target = -asri.pct_change(30).shift(-30)  # Forward "stress"
        
        # PCA weights
        pca_deriver = PCAWeightDeriver()
        pca_deriver.fit(sub_indices)
        
        # Elastic Net weights
        en_deriver = ElasticNetWeightDeriver(forward_window=30)
        en_deriver.fit(sub_indices, target)
        
        # Comparison
        comparison = compare_weight_methods(sub_indices, market_returns=target)
        
        # Generate tables
        with open(self.tables_dir / "pca_weights.tex", "w") as f:
            f.write(format_pca_table(pca_deriver.result))
        
        with open(self.tables_dir / "elastic_net_weights.tex", "w") as f:
            f.write(format_elastic_net_table(en_deriver.result))
        
        with open(self.tables_dir / "weight_comparison.tex", "w") as f:
            f.write(format_weight_comparison_table(comparison))
        
        return {
            'pca': pca_deriver.result,
            'elastic_net': en_deriver.result,
            'comparison': comparison,
        }
    
    def run_walk_forward(self, data: pd.DataFrame, weights: dict) -> dict:
        """Run walk-forward validation."""
        from asri.validation.walk_forward import (
            purged_walk_forward_cv,
            walk_forward_optimization,
            format_walk_forward_table,
        )
        
        logger.info("Running walk-forward validation...")
        
        sub_indices = data[['stablecoin_risk', 'defi_liquidity_risk',
                           'contagion_risk', 'arbitrage_opacity']]
        target = -data['asri'].pct_change(30).shift(-30)
        
        # Fixed weights validation
        theoretical_weights = {
            'stablecoin_risk': 0.30,
            'defi_liquidity_risk': 0.25,
            'contagion_risk': 0.25,
            'arbitrage_opacity': 0.20,
        }
        
        fixed_result = purged_walk_forward_cv(
            sub_indices, target, theoretical_weights,
            n_splits=5, purge_days=30,
        )
        
        # Optimized weights validation
        optimized_result = walk_forward_optimization(
            sub_indices, target,
            n_splits=5, purge_days=30,
            optimization_method="elastic_net",
        )
        
        # Generate tables
        with open(self.tables_dir / "walk_forward_fixed.tex", "w") as f:
            f.write(format_walk_forward_table(fixed_result))
        
        with open(self.tables_dir / "walk_forward_optimized.tex", "w") as f:
            f.write(format_walk_forward_table(optimized_result))
        
        return {
            'fixed': fixed_result,
            'optimized': optimized_result,
        }
    
    def run_event_study(self, data: pd.DataFrame) -> list:
        """Run event study around crises."""
        from asri.validation.event_study import (
            CrisisEvent,
            run_event_study,
            format_event_study_table,
        )
        
        logger.info("Running event study...")
        
        events = [
            CrisisEvent(e['name'], e['date'], e['description'], e['severity'])
            for e in CRISIS_EVENTS
        ]
        
        asri = data['asri']
        
        results = run_event_study(
            asri, events,
            estimation_window=(-90, -31),
            event_window=(-30, 10),
        )
        
        # Generate table
        with open(self.tables_dir / "event_study.tex", "w") as f:
            f.write(format_event_study_table(results))
        
        return results
    
    def run_roc_analysis(self, data: pd.DataFrame):
        """Run ROC analysis for crisis prediction."""
        from asri.validation.roc_analysis import (
            compute_roc_metrics,
            format_roc_table,
        )
        
        logger.info("Running ROC analysis...")
        
        asri = data['asri']
        
        # Create synthetic returns from ASRI changes
        returns = -asri.pct_change()  # Negative ASRI change = positive "return"
        
        metrics = compute_roc_metrics(
            asri, returns,
            forward_window=30,
            drawdown_threshold=-0.20,
        )
        
        # Generate table
        with open(self.tables_dir / "roc_metrics.tex", "w") as f:
            f.write(format_roc_table(metrics))
        
        return metrics
    
    def run_benchmark_comparison(self, data: pd.DataFrame):
        """Compare ASRI against benchmarks."""
        from asri.validation.benchmark import (
            compare_with_benchmarks,
            create_naive_benchmarks,
            format_benchmark_table,
        )
        
        logger.info("Running benchmark comparison...")
        
        asri = data['asri']
        target = -asri.pct_change(30).shift(-30)
        
        # Create naive benchmarks
        defi_tvl = data['stablecoin_risk'] + data['defi_liquidity_risk']  # Proxy
        benchmarks = create_naive_benchmarks(defi_tvl)
        
        comparison = compare_with_benchmarks(asri, benchmarks, target)
        
        # Generate table
        with open(self.tables_dir / "benchmarks.tex", "w") as f:
            f.write(format_benchmark_table(comparison))
        
        return comparison
    
    def run_robustness_tests(self, data: pd.DataFrame) -> dict:
        """Run robustness and placebo tests."""
        from asri.validation.robustness import (
            run_placebo_tests,
            structural_break_test,
            format_robustness_table,
        )
        
        logger.info("Running robustness tests...")
        
        asri = data['asri']
        sub_indices = data[['stablecoin_risk', 'defi_liquidity_risk',
                           'contagion_risk', 'arbitrage_opacity']]
        target = -asri.pct_change(30).shift(-30)
        
        weights = {
            'stablecoin_risk': 0.30,
            'defi_liquidity_risk': 0.25,
            'contagion_risk': 0.25,
            'arbitrage_opacity': 0.20,
        }
        
        crisis_dates = [e['date'] for e in CRISIS_EVENTS]
        
        placebo_results = run_placebo_tests(
            asri, sub_indices, weights, target, crisis_dates,
            n_permutations=500,
        )
        
        break_result = structural_break_test(asri, method="cusum")
        
        # Generate table
        with open(self.tables_dir / "robustness.tex", "w") as f:
            f.write(format_robustness_table(placebo_results, break_result))
        
        return {
            'placebo': placebo_results,
            'structural_break': break_result,
        }
    
    def run_regime_detection(self, data: pd.DataFrame):
        """Detect market regimes."""
        from asri.regime.hmm import RegimeDetector, format_regime_table
        
        logger.info("Detecting regimes...")
        
        sub_indices = data[['stablecoin_risk', 'defi_liquidity_risk',
                           'contagion_risk', 'arbitrage_opacity']]
        
        detector = RegimeDetector(n_regimes=3)
        detector.fit(sub_indices)
        
        # Compute regime-specific weights
        target = -data['asri'].pct_change(30).shift(-30)
        detector.compute_regime_weights(sub_indices, target)
        
        # Generate table
        with open(self.tables_dir / "regimes.tex", "w") as f:
            f.write(format_regime_table(detector.result))
        
        return detector.result
    
    def generate_descriptive_stats(self, data: pd.DataFrame) -> dict:
        """Generate descriptive statistics."""
        from asri.statistics.descriptive import (
            compute_descriptive_table,
            correlation_matrix_with_significance,
            format_descriptive_table,
            format_correlation_table,
        )
        
        logger.info("Computing descriptive statistics...")
        
        stats_df = compute_descriptive_table(data)
        corr, pval = correlation_matrix_with_significance(data)
        
        # Generate tables
        with open(self.tables_dir / "descriptive.tex", "w") as f:
            f.write(format_descriptive_table(stats_df))
        
        with open(self.tables_dir / "correlation.tex", "w") as f:
            f.write(format_correlation_table(corr, pval))
        
        return {
            'stats': stats_df,
            'correlation': corr,
            'pvalues': pval,
        }
    
    async def run_full_pipeline(self) -> dict:
        """Run the complete validation pipeline."""
        logger.info("="*60)
        logger.info("ASRI RIGOROUS VALIDATION PIPELINE")
        logger.info("="*60)
        
        # Load data
        data = await self.load_data()
        logger.info(f"Loaded {len(data)} observations")
        
        # Run all validation steps
        self.results['descriptive'] = self.generate_descriptive_stats(data)
        self.results['stationarity'] = self.run_stationarity_tests(data)
        self.results['granger'] = self.run_granger_causality(data)
        self.results['weights'] = self.derive_weights(data)
        self.results['walk_forward'] = self.run_walk_forward(data, self.results['weights'])
        self.results['event_study'] = self.run_event_study(data)
        self.results['roc'] = self.run_roc_analysis(data)
        self.results['benchmarks'] = self.run_benchmark_comparison(data)
        self.results['robustness'] = self.run_robustness_tests(data)
        self.results['regimes'] = self.run_regime_detection(data)
        
        # Generate summary
        self.generate_summary()
        
        logger.info("="*60)
        logger.info("VALIDATION COMPLETE")
        logger.info(f"Results saved to {self.output_dir}")
        logger.info("="*60)
        
        return self.results
    
    def generate_summary(self):
        """Generate summary report."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_observations': len(self.results.get('descriptive', {}).get('stats', [])),
            'weight_comparison': {
                'pca_vs_theoretical_corr': float(self.results['weights']['comparison'].pca_vs_theoretical),
                'elastic_net_vs_theoretical_corr': float(self.results['weights']['comparison'].elastic_net_vs_theoretical),
            },
            'walk_forward': {
                'fixed_mean_r2': float(self.results['walk_forward']['fixed'].mean_test_r2),
                'optimized_mean_r2': float(self.results['walk_forward']['optimized'].mean_test_r2),
            },
            'event_study': {
                'n_significant': sum(1 for r in self.results['event_study'] if r.is_significant),
                'avg_lead_days': float(np.mean([r.lead_days for r in self.results['event_study']])),
            },
            'roc': {
                'auc_roc': float(self.results['roc'].auc_roc),
                'auc_pr': float(self.results['roc'].auc_pr),
            },
            'benchmarks': {
                'best_model': self.results['benchmarks'].best_model,
            },
            'robustness': {
                'placebo_passed': all(r.is_genuine for r in self.results['robustness']['placebo']),
                'structurally_stable': self.results['robustness']['structural_break'].is_stable,
            },
        }
        
        with open(self.output_dir / "validation_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        logger.info(f"AUC-ROC: {summary['roc']['auc_roc']:.3f}")
        logger.info(f"Walk-Forward R²: {summary['walk_forward']['fixed_mean_r2']:.3f}")
        logger.info(f"Event Study Significant: {summary['event_study']['n_significant']}/{len(self.results['event_study'])}")
        logger.info(f"Average Lead Time: {summary['event_study']['avg_lead_days']:.1f} days")
        logger.info(f"Best Model: {summary['benchmarks']['best_model']}")
        logger.info(f"Robustness Passed: {summary['robustness']['placebo_passed']}")


async def main():
    parser = argparse.ArgumentParser(description="ASRI Validation Pipeline")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )
    args = parser.parse_args()
    
    pipeline = ValidationPipeline(args.output_dir)
    await pipeline.run_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
