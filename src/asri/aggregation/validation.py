"""
Out-of-Sample Validation and Bootstrap Confidence Intervals for ASRI Aggregation Methods.

Addresses reviewer concern: "What's the out-of-sample detection rate?"

This module implements temporal validation where models are trained on historical data
(2021-2023) and tested on future data (2024), including the August 2024 crisis event.
This provides a genuine test of predictive ability rather than in-sample fitting.

Methodology:
    1. Split data at 2023-12-31 (train: 2021-2023, test: 2024)
    2. For each aggregation method:
       - Fit/calibrate parameters on training data only
       - Apply to test data without refitting
       - Check if August 2024 crisis was detected (ASRI > threshold before event)
    3. Compute bootstrap confidence intervals for uncertainty quantification

The key insight is that many backtesting studies suffer from look-ahead bias where
models are calibrated on the entire sample including crisis periods. Out-of-sample
validation ensures the early warning system would have worked in real-time.

References:
    - Berkowitz & O'Brien (2002). How Accurate are Value-at-Risk Models at
      Commercial Banks? Journal of Finance.
    - Christoffersen, P. (1998). Evaluating Interval Forecasts. International
      Economic Review.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Any, Tuple, List

import numpy as np
import pandas as pd

# Import aggregators
from .ciss_aggregation import CISSAggregator
from .copula_aggregation import CopulaAggregator
from .regime_aggregation import RegimeAggregator

# Canonical sub-index column names
CANONICAL_COLUMNS = ['stablecoin_risk', 'defi_liquidity_risk', 'contagion_risk', 'arbitrage_opacity']

# August 2024 crisis event for out-of-sample testing
OOS_CRISIS = {
    'name': 'August 2024 Crash',
    'date': '2024-08-05',
    'window_start': '2024-06-05',
    'window_end': '2024-08-15',
}


@dataclass
class ValidationResult:
    """Results from out-of-sample validation of a single aggregation method."""

    method: str  # Aggregation method name
    detected: bool  # Whether crisis was detected (ASRI > threshold before event)
    lead_time_days: Optional[int]  # Days between first detection and crisis (None if not detected)
    peak_asri: float  # Maximum ASRI in pre-crisis window
    threshold: float  # Detection threshold used
    train_period: str  # Training period description
    test_period: str  # Test period description
    mean_asri_train: float  # Mean ASRI in training period
    mean_asri_test: float  # Mean ASRI in test period
    std_asri_train: float  # Std ASRI in training period
    std_asri_test: float  # Std ASRI in test period


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval results."""

    method: str
    mean_asri: pd.Series  # Point estimate (mean across bootstrap samples)
    lower_ci: pd.Series  # Lower confidence bound
    upper_ci: pd.Series  # Upper confidence bound
    confidence_level: float  # e.g., 0.90 for 90% CI
    n_bootstrap: int  # Number of bootstrap samples used


@dataclass
class BootstrapDetectionResult:
    """Results from bootstrap detection analysis.

    Provides confidence intervals for crisis detection metrics by resampling
    the estimation window used to establish "expected" ASRI levels.
    """

    detection_rate: float  # Proportion of bootstraps detecting crisis
    detection_rate_ci: Tuple[float, float]  # 95% CI for detection rate
    lead_time_mean: float  # Mean lead time across detected bootstraps
    lead_time_ci: Tuple[float, float]  # 95% CI for lead time
    n_bootstrap: int  # Number of bootstrap samples
    n_detected: int  # Number of bootstraps that detected crisis


class OutOfSampleValidator:
    """
    Out-of-sample validation for ASRI aggregation methods.

    Implements temporal train/test split to evaluate genuine predictive ability.
    Models are calibrated on training data (2021-2023) and evaluated on test data
    (2024) without refitting, simulating real-world deployment.

    Example usage:
        >>> validator = OutOfSampleValidator(train_end='2023-12-31', threshold=70.0)
        >>> train, test = validator.split_data(subindices)
        >>> result = validator.validate_linear(train, test)
        >>> print(f"Detected: {result.detected}, Lead time: {result.lead_time_days} days")

        >>> # Run all methods and compare
        >>> results_df = validator.run_all_methods(subindices)
        >>> print(results_df)

    Attributes:
        train_end: End date of training period (test starts next day)
        threshold: ASRI threshold for crisis detection (default: 70.0)
        crisis_date: Date of the out-of-sample crisis event
        crisis_window_start: Start of pre-crisis detection window
    """

    def __init__(
        self,
        train_end: str = '2023-12-31',
        threshold: float = 70.0,
        crisis_config: Optional[dict] = None,
    ) -> None:
        """
        Initialize the out-of-sample validator.

        Args:
            train_end: End date of training period (YYYY-MM-DD format).
                Test period starts the next day.
            threshold: ASRI threshold for crisis detection. Values above this
                are considered "detected" crisis signals.
            crisis_config: Optional dict overriding OOS_CRISIS defaults.
                Must contain 'name', 'date', 'window_start', 'window_end'.
        """
        self.train_end = pd.Timestamp(train_end)
        self.threshold = threshold

        # Use provided crisis config or default
        crisis = crisis_config or OOS_CRISIS
        self.crisis_name = crisis['name']
        self.crisis_date = pd.Timestamp(crisis['date'])
        self.crisis_window_start = pd.Timestamp(crisis['window_start'])
        self.crisis_window_end = pd.Timestamp(crisis['window_end'])

    def split_data(
        self,
        subindices: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split sub-indices into train (2021-2023) and test (2024) sets.

        The split is temporal: training data ends at train_end, test data
        starts immediately after. This ensures no information leakage from
        future to past.

        Args:
            subindices: DataFrame with datetime index and sub-index columns.
                Expected columns: CANONICAL_COLUMNS or similar.

        Returns:
            Tuple of (train_df, test_df) where:
                - train_df: Data from start to train_end (inclusive)
                - test_df: Data after train_end to present

        Raises:
            ValueError: If data doesn't span both train and test periods.
        """
        # Validate and clean data
        df = subindices.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Split
        train = df[df.index <= self.train_end].copy()
        test = df[df.index > self.train_end].copy()

        if len(train) == 0:
            raise ValueError(
                f"No training data before {self.train_end}. "
                f"Data range: {df.index.min()} to {df.index.max()}"
            )

        if len(test) == 0:
            raise ValueError(
                f"No test data after {self.train_end}. "
                f"Data range: {df.index.min()} to {df.index.max()}"
            )

        return train, test

    def _check_detection(
        self,
        asri_series: pd.Series,
        threshold: Optional[float] = None,
    ) -> tuple[bool, Optional[int], float]:
        """
        Check if crisis was detected in the pre-crisis window.

        Args:
            asri_series: ASRI time series for test period
            threshold: Detection threshold (uses self.threshold if None)

        Returns:
            Tuple of (detected, lead_time_days, peak_asri):
                - detected: True if ASRI exceeded threshold before crisis date
                - lead_time_days: Days between first detection and crisis (None if not detected)
                - peak_asri: Maximum ASRI value in pre-crisis window
        """
        if threshold is None:
            threshold = self.threshold

        # Pre-crisis window: from window_start to crisis_date
        pre_crisis = asri_series[
            (asri_series.index >= self.crisis_window_start) &
            (asri_series.index < self.crisis_date)
        ]

        if len(pre_crisis) == 0:
            return False, None, np.nan

        peak_asri = float(pre_crisis.max())

        # Check for threshold breach
        above_threshold = pre_crisis[pre_crisis > threshold]

        if len(above_threshold) == 0:
            return False, None, peak_asri

        # Lead time: days from first breach to crisis
        first_breach = above_threshold.index[0]
        lead_time = (self.crisis_date - first_breach).days

        return True, lead_time, peak_asri

    def _compute_linear_asri(
        self,
        subindices: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
    ) -> pd.Series:
        """
        Compute linear weighted sum ASRI.

        Args:
            subindices: DataFrame with sub-index columns
            weights: Weight vector (defaults to [0.30, 0.25, 0.25, 0.20])

        Returns:
            ASRI time series
        """
        if weights is None:
            weights = np.array([0.30, 0.25, 0.25, 0.20])

        # Get columns in canonical order (or use what's available)
        cols = [c for c in CANONICAL_COLUMNS if c in subindices.columns]
        if len(cols) == 0:
            cols = subindices.columns[:4].tolist()

        if len(cols) != len(weights):
            # Adjust weights to match available columns
            weights = np.ones(len(cols)) / len(cols)

        asri = subindices[cols].values @ weights
        return pd.Series(asri, index=subindices.index, name='asri_linear')

    def validate_linear(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
    ) -> ValidationResult:
        """
        Validate linear aggregation (no training needed, weights are fixed).

        Linear aggregation uses predetermined weights and requires no parameter
        estimation, so it trivially satisfies the out-of-sample requirement.
        We still compute statistics on train/test for comparison.

        Args:
            train: Training period sub-indices (for statistics only)
            test: Test period sub-indices
            weights: Weight vector (defaults to [0.30, 0.25, 0.25, 0.20])

        Returns:
            ValidationResult with detection metrics
        """
        if weights is None:
            weights = np.array([0.30, 0.25, 0.25, 0.20])

        # Compute ASRI for both periods
        asri_train = self._compute_linear_asri(train, weights)
        asri_test = self._compute_linear_asri(test, weights)

        # Check detection
        detected, lead_time, peak = self._check_detection(asri_test)

        return ValidationResult(
            method='linear',
            detected=detected,
            lead_time_days=lead_time,
            peak_asri=peak,
            threshold=self.threshold,
            train_period=f"{train.index.min().date()} to {train.index.max().date()}",
            test_period=f"{test.index.min().date()} to {test.index.max().date()}",
            mean_asri_train=float(asri_train.mean()),
            mean_asri_test=float(asri_test.mean()),
            std_asri_train=float(asri_train.std()),
            std_asri_test=float(asri_test.std()),
        )

    def validate_ciss(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        decay_factor: float = 0.94,
    ) -> ValidationResult:
        """
        Validate CISS aggregation with out-of-sample evaluation.

        CISS uses EWMA correlation estimation. We initialize the EWMA on
        training data, then continue updating on test data. The key is that
        no test data information is used to calibrate the decay parameter.

        Args:
            train: Training period sub-indices (for EWMA initialization)
            test: Test period sub-indices
            decay_factor: EWMA decay parameter (calibrated on train, but
                typically fixed at 0.94 following RiskMetrics standard)

        Returns:
            ValidationResult with detection metrics
        """
        # Initialize aggregator
        aggregator = CISSAggregator(decay_factor=decay_factor, use_equal_weights=False)

        # Compute on full series (EWMA initialized on train, continues on test)
        full_data = pd.concat([train, test]).sort_index()

        try:
            asri_full = aggregator.compute_asri_series(
                full_data,
                normalize=True,
                normalization_method='expanding',  # Avoids look-ahead bias
            )
        except Exception as e:
            warnings.warn(f"CISS aggregation failed: {e}")
            return self._empty_result('ciss', train, test)

        # Split results
        asri_train = asri_full[asri_full.index <= self.train_end]
        asri_test = asri_full[asri_full.index > self.train_end]

        # Check detection
        detected, lead_time, peak = self._check_detection(asri_test)

        return ValidationResult(
            method='ciss',
            detected=detected,
            lead_time_days=lead_time,
            peak_asri=peak,
            threshold=self.threshold,
            train_period=f"{train.index.min().date()} to {train.index.max().date()}",
            test_period=f"{test.index.min().date()} to {test.index.max().date()}",
            mean_asri_train=float(asri_train.mean()) if len(asri_train) > 0 else np.nan,
            mean_asri_test=float(asri_test.mean()) if len(asri_test) > 0 else np.nan,
            std_asri_train=float(asri_train.std()) if len(asri_train) > 0 else np.nan,
            std_asri_test=float(asri_test.std()) if len(asri_test) > 0 else np.nan,
        )

    def validate_copula(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        copula_family: str = 'clayton',
        tail_threshold: float = 0.90,
        boost_factor: float = 1.5,
    ) -> ValidationResult:
        """
        Validate copula aggregation with out-of-sample evaluation.

        The copula is fitted on training data to estimate tail dependence
        parameters. These parameters are then held fixed while computing
        ASRI on test data.

        Args:
            train: Training period sub-indices (for copula fitting)
            test: Test period sub-indices
            copula_family: Copula type ('clayton', 'gumbel', 'student', 'frank')
            tail_threshold: Quantile for tail event detection
            boost_factor: Weight amplification during tail events

        Returns:
            ValidationResult with detection metrics
        """
        # Fit copula on training data only
        aggregator = CopulaAggregator(copula_family=copula_family)

        try:
            # Fit on train
            aggregator.fit_copula(train)

            # Apply to train and test (using parameters from train)
            asri_train = aggregator.aggregate_with_tail_boost(
                train,
                tail_threshold=tail_threshold,
                boost_factor=boost_factor,
            )

            # For test, we need to apply the same copula parameters
            # The fit_copula step is already done, so aggregate_with_tail_boost
            # will use the stored parameters
            asri_test = aggregator.aggregate_with_tail_boost(
                test,
                tail_threshold=tail_threshold,
                boost_factor=boost_factor,
            )

        except Exception as e:
            warnings.warn(f"Copula aggregation failed: {e}")
            return self._empty_result('copula', train, test)

        # Check detection
        detected, lead_time, peak = self._check_detection(asri_test)

        return ValidationResult(
            method='copula',
            detected=detected,
            lead_time_days=lead_time,
            peak_asri=peak,
            threshold=self.threshold,
            train_period=f"{train.index.min().date()} to {train.index.max().date()}",
            test_period=f"{test.index.min().date()} to {test.index.max().date()}",
            mean_asri_train=float(asri_train.mean()) if len(asri_train) > 0 else np.nan,
            mean_asri_test=float(asri_test.mean()) if len(asri_test) > 0 else np.nan,
            std_asri_train=float(asri_train.std()) if len(asri_train) > 0 else np.nan,
            std_asri_test=float(asri_test.std()) if len(asri_test) > 0 else np.nan,
        )

    def validate_regime(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        n_regimes: int = 3,
        use_hmmlearn: bool = True,
    ) -> ValidationResult:
        """
        Validate regime-switching aggregation with out-of-sample evaluation.

        The HMM is fitted on training data to learn regime dynamics. These
        learned parameters (transition matrix, emission distributions) are
        then applied to test data for regime classification and aggregation.

        Args:
            train: Training period sub-indices (for HMM fitting)
            test: Test period sub-indices
            n_regimes: Number of hidden states (default: 3)
            use_hmmlearn: Whether to use hmmlearn library (recommended)

        Returns:
            ValidationResult with detection metrics
        """
        # Fit HMM on training data
        aggregator = RegimeAggregator(
            n_regimes=n_regimes,
            use_hmmlearn=use_hmmlearn,
            random_state=42,
        )

        try:
            aggregator.fit_hmm(train)

            # Aggregate train and test using fitted model
            asri_train = aggregator.aggregate(train, method='probability_weighted')

            # For test, the HMM uses its learned parameters to infer regime
            # probabilities and compute weighted aggregation
            asri_test = aggregator.aggregate(test, method='probability_weighted')

        except Exception as e:
            warnings.warn(f"Regime aggregation failed: {e}")
            return self._empty_result('regime', train, test)

        # Check detection
        detected, lead_time, peak = self._check_detection(asri_test)

        return ValidationResult(
            method='regime',
            detected=detected,
            lead_time_days=lead_time,
            peak_asri=peak,
            threshold=self.threshold,
            train_period=f"{train.index.min().date()} to {train.index.max().date()}",
            test_period=f"{test.index.min().date()} to {test.index.max().date()}",
            mean_asri_train=float(asri_train.mean()) if len(asri_train) > 0 else np.nan,
            mean_asri_test=float(asri_test.mean()) if len(asri_test) > 0 else np.nan,
            std_asri_train=float(asri_train.std()) if len(asri_train) > 0 else np.nan,
            std_asri_test=float(asri_test.std()) if len(asri_test) > 0 else np.nan,
        )

    def _empty_result(
        self,
        method: str,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> ValidationResult:
        """Create empty result when method fails."""
        return ValidationResult(
            method=method,
            detected=False,
            lead_time_days=None,
            peak_asri=np.nan,
            threshold=self.threshold,
            train_period=f"{train.index.min().date()} to {train.index.max().date()}",
            test_period=f"{test.index.min().date()} to {test.index.max().date()}",
            mean_asri_train=np.nan,
            mean_asri_test=np.nan,
            std_asri_train=np.nan,
            std_asri_test=np.nan,
        )

    def run_all_methods(
        self,
        subindices: pd.DataFrame,
        include_failed: bool = True,
    ) -> pd.DataFrame:
        """
        Compare all 4 aggregation methods out-of-sample.

        Runs linear, CISS, copula, and regime validation on the same
        train/test split and returns a comparison DataFrame.

        Args:
            subindices: Full sub-index DataFrame (will be split internally)
            include_failed: Whether to include methods that failed

        Returns:
            DataFrame with one row per method, columns:
                - method: Aggregation method name
                - detected: Whether crisis was detected
                - lead_time_days: Days of lead time (None if not detected)
                - peak_asri: Maximum ASRI in pre-crisis window
                - threshold: Detection threshold used
                - train_period, test_period: Period descriptions
                - mean/std statistics for train and test
        """
        train, test = self.split_data(subindices)

        results = []

        # Linear (always succeeds)
        results.append(self.validate_linear(train, test))

        # CISS
        ciss_result = self.validate_ciss(train, test)
        if include_failed or not np.isnan(ciss_result.peak_asri):
            results.append(ciss_result)

        # Copula
        copula_result = self.validate_copula(train, test)
        if include_failed or not np.isnan(copula_result.peak_asri):
            results.append(copula_result)

        # Regime
        regime_result = self.validate_regime(train, test)
        if include_failed or not np.isnan(regime_result.peak_asri):
            results.append(regime_result)

        # Convert to DataFrame
        records = []
        for r in results:
            records.append({
                'method': r.method,
                'detected': r.detected,
                'lead_time_days': r.lead_time_days,
                'peak_asri': r.peak_asri,
                'threshold': r.threshold,
                'train_period': r.train_period,
                'test_period': r.test_period,
                'mean_asri_train': r.mean_asri_train,
                'mean_asri_test': r.mean_asri_test,
                'std_asri_train': r.std_asri_train,
                'std_asri_test': r.std_asri_test,
            })

        return pd.DataFrame(records)

    def bootstrap_confidence_intervals(
        self,
        subindices: pd.DataFrame,
        method: str = 'linear',
        n_bootstrap: int = 500,
        confidence: float = 0.90,
        block_size: int = 20,
        random_state: int = 42,
    ) -> BootstrapResult:
        """
        Compute bootstrap confidence intervals for ASRI estimates.

        Uses block bootstrap to preserve temporal dependence in the
        sub-index time series. Returns point estimates and confidence
        bounds for each time step.

        Args:
            subindices: Sub-index DataFrame
            method: Aggregation method ('linear', 'ciss', 'copula', 'regime')
            n_bootstrap: Number of bootstrap replications
            confidence: Confidence level (e.g., 0.90 for 90% CI)
            block_size: Size of blocks for block bootstrap
            random_state: Random seed for reproducibility

        Returns:
            BootstrapResult with mean, lower CI, and upper CI series
        """
        rng = np.random.default_rng(random_state)
        n = len(subindices)

        # Determine how to compute ASRI for this method
        if method == 'linear':
            compute_asri = lambda df: self._compute_linear_asri(df)
        elif method == 'ciss':
            def compute_asri(df):
                agg = CISSAggregator(decay_factor=0.94)
                return agg.compute_asri_series(df, normalize=True, normalization_method='expanding')
        elif method == 'copula':
            def compute_asri(df):
                agg = CopulaAggregator(copula_family='clayton')
                agg.fit_copula(df)
                return agg.aggregate_with_tail_boost(df)
        elif method == 'regime':
            def compute_asri(df):
                agg = RegimeAggregator(n_regimes=3)
                agg.fit_hmm(df)
                return agg.aggregate(df)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Store bootstrap replications
        bootstrap_asri = []

        for b in range(n_bootstrap):
            # Block bootstrap: sample blocks with replacement
            n_blocks = int(np.ceil(n / block_size))
            block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)

            # Build bootstrap sample
            indices = []
            for start in block_starts:
                indices.extend(range(start, min(start + block_size, n)))

            indices = indices[:n]  # Trim to original length
            boot_df = subindices.iloc[indices].copy()
            boot_df.index = subindices.index  # Keep original index

            try:
                asri = compute_asri(boot_df)
                bootstrap_asri.append(asri.values)
            except Exception:
                # Skip failed bootstrap samples
                continue

        if len(bootstrap_asri) == 0:
            raise ValueError(f"All bootstrap samples failed for method {method}")

        # Convert to array
        bootstrap_array = np.array(bootstrap_asri)

        # Compute statistics
        mean_asri = np.nanmean(bootstrap_array, axis=0)
        alpha = 1 - confidence
        lower_ci = np.nanpercentile(bootstrap_array, 100 * alpha / 2, axis=0)
        upper_ci = np.nanpercentile(bootstrap_array, 100 * (1 - alpha / 2), axis=0)

        return BootstrapResult(
            method=method,
            mean_asri=pd.Series(mean_asri, index=subindices.index, name=f'asri_{method}_mean'),
            lower_ci=pd.Series(lower_ci, index=subindices.index, name=f'asri_{method}_lower'),
            upper_ci=pd.Series(upper_ci, index=subindices.index, name=f'asri_{method}_upper'),
            confidence_level=confidence,
            n_bootstrap=len(bootstrap_asri),
        )

    def bootstrap_detection_metrics(
        self,
        subindices: pd.DataFrame,
        crisis_date: str = '2024-08-05',
        n_bootstrap: int = 500,
        block_size: int = 20,
        confidence_level: float = 0.95,
        estimation_window: int = 60,
        pre_crisis_window: int = 90,
        random_state: int = 42,
    ) -> BootstrapDetectionResult:
        """
        Bootstrap confidence intervals for detection rate and lead time.

        Uses block bootstrap to preserve autocorrelation in time series.
        For each bootstrap sample:
        1. Resample the pre-crisis estimation window with replacement (block bootstrap)
        2. Compute expected ASRI from resampled estimation window
        3. Check if crisis was detected (ASRI > threshold before crisis_date)
        4. Record lead time if detected

        Args:
            subindices: DataFrame with sub-index columns (datetime index)
            crisis_date: Date of crisis event (YYYY-MM-DD)
            n_bootstrap: Number of bootstrap replications (default: 500)
            block_size: Size of blocks for block bootstrap (default: 20 days)
            confidence_level: Confidence level for CIs (default: 0.95)
            estimation_window: Days before pre-crisis window to estimate "normal" (default: 60)
            pre_crisis_window: Days before crisis to look for detection (default: 90)
            random_state: Random seed for reproducibility

        Returns:
            BootstrapDetectionResult with CIs for detection rate and lead time.
        """
        rng = np.random.default_rng(random_state)

        # Parse crisis date
        crisis_dt = pd.Timestamp(crisis_date)

        # Ensure datetime index
        df = subindices.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Define windows
        # Pre-crisis detection window: [crisis_date - pre_crisis_window, crisis_date)
        detection_start = crisis_dt - pd.Timedelta(days=pre_crisis_window)
        # Estimation window: [detection_start - estimation_window, detection_start)
        estimation_start = detection_start - pd.Timedelta(days=estimation_window)

        # Extract estimation period data
        estimation_data = df[(df.index >= estimation_start) & (df.index < detection_start)]
        # Extract pre-crisis period data
        pre_crisis_data = df[(df.index >= detection_start) & (df.index < crisis_dt)]

        if len(estimation_data) < block_size:
            raise ValueError(
                f"Estimation window too short: {len(estimation_data)} days, "
                f"need at least {block_size} for block bootstrap"
            )

        if len(pre_crisis_data) == 0:
            raise ValueError(f"No data in pre-crisis window before {crisis_date}")

        # Compute baseline ASRI for pre-crisis period (point estimate)
        baseline_asri = self._compute_linear_asri(pre_crisis_data)

        # Store detection results across bootstrap samples
        detection_flags: List[bool] = []
        lead_times: List[float] = []

        n_est = len(estimation_data)

        for b in range(n_bootstrap):
            # Block bootstrap the estimation window to perturb "expected" level
            n_blocks = int(np.ceil(n_est / block_size))
            block_starts = rng.integers(0, max(1, n_est - block_size + 1), size=n_blocks)

            # Build bootstrap sample indices
            boot_indices = []
            for start in block_starts:
                boot_indices.extend(range(start, min(start + block_size, n_est)))
            boot_indices = boot_indices[:n_est]

            # Create bootstrap estimation sample
            boot_estimation = estimation_data.iloc[boot_indices].copy()
            boot_estimation.index = estimation_data.index  # Keep original dates

            # Compute bootstrap "expected" ASRI from perturbed estimation window
            boot_expected_asri = self._compute_linear_asri(boot_estimation)
            expected_mean = float(boot_expected_asri.mean())
            expected_std = float(boot_expected_asri.std())

            # Use dynamic threshold: expected_mean + 2*std, capped at self.threshold
            # This simulates uncertainty in what "elevated" means given estimation noise
            if expected_std > 0:
                dynamic_threshold = min(self.threshold, expected_mean + 2 * expected_std)
            else:
                dynamic_threshold = self.threshold

            # Check detection: ASRI exceeds threshold before crisis
            above_threshold = baseline_asri[baseline_asri > dynamic_threshold]

            if len(above_threshold) > 0:
                detection_flags.append(True)
                first_breach = above_threshold.index[0]
                lead_time = (crisis_dt - first_breach).days
                lead_times.append(lead_time)
            else:
                detection_flags.append(False)

        # Compute detection rate and CI
        n_detected = sum(detection_flags)
        detection_rate = n_detected / n_bootstrap

        # Bootstrap CI for detection rate using Wilson score interval
        # (better for proportions near 0 or 1)
        alpha = 1 - confidence_level
        z = 1.96 if confidence_level == 0.95 else 1.645  # approximate

        if n_bootstrap > 0:
            p = detection_rate
            # Wilson score interval
            denom = 1 + z**2 / n_bootstrap
            center = (p + z**2 / (2 * n_bootstrap)) / denom
            margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_bootstrap)) / n_bootstrap) / denom
            detection_ci = (max(0, center - margin), min(1, center + margin))
        else:
            detection_ci = (0.0, 0.0)

        # Compute lead time stats (only for detected samples)
        if len(lead_times) > 0:
            lead_time_mean = float(np.mean(lead_times))
            # Percentile CI for lead time
            lower_pct = 100 * alpha / 2
            upper_pct = 100 * (1 - alpha / 2)
            lead_time_ci = (
                float(np.percentile(lead_times, lower_pct)),
                float(np.percentile(lead_times, upper_pct)),
            )
        else:
            lead_time_mean = np.nan
            lead_time_ci = (np.nan, np.nan)

        return BootstrapDetectionResult(
            detection_rate=detection_rate,
            detection_rate_ci=detection_ci,
            lead_time_mean=lead_time_mean,
            lead_time_ci=lead_time_ci,
            n_bootstrap=n_bootstrap,
            n_detected=n_detected,
        )

    def format_validation_table(
        self,
        results: pd.DataFrame,
        caption: str = "Out-of-Sample Validation: August 2024 Crisis Detection",
        label: str = "tab:oos-validation",
    ) -> str:
        """
        Generate LaTeX table of out-of-sample validation results.

        Args:
            results: DataFrame from run_all_methods()
            caption: Table caption
            label: LaTeX label for cross-referencing

        Returns:
            LaTeX table code
        """
        method_names = {
            'linear': 'Linear',
            'ciss': 'CISS',
            'copula': 'Copula',
            'regime': 'Regime-Switching',
        }

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\small",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Method & Detected & Lead Time & Peak ASRI & Train Mean & Test Mean \\",
            r"\midrule",
        ]

        for _, row in results.iterrows():
            method = method_names.get(row['method'], row['method'])
            detected = "Yes" if row['detected'] else "No"
            lead_time = f"{row['lead_time_days']} days" if row['lead_time_days'] else "N/A"
            peak = f"{row['peak_asri']:.1f}" if not np.isnan(row['peak_asri']) else "N/A"
            train_mean = f"{row['mean_asri_train']:.1f}" if not np.isnan(row['mean_asri_train']) else "N/A"
            test_mean = f"{row['mean_asri_test']:.1f}" if not np.isnan(row['mean_asri_test']) else "N/A"

            lines.append(
                f"{method} & {detected} & {lead_time} & {peak} & {train_mean} & {test_mean} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            f"\\item Training period: 2021-01-01 to {self.train_end.date()}. "
            f"Test period: {self.train_end.date()} to 2024-12-31.",
            f"\\item Crisis event: {self.crisis_name} ({self.crisis_date.date()}). "
            f"Detection threshold: ASRI $>$ {self.threshold:.0f}.",
            r"\item Lead Time = days between first threshold breach and crisis onset.",
            r"\end{tablenotes}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def summarize_findings(
        self,
        results: pd.DataFrame,
    ) -> str:
        """
        Generate prose summary of validation findings for paper discussion.

        Args:
            results: DataFrame from run_all_methods()

        Returns:
            Summary text suitable for results section
        """
        detected_methods = results[results['detected']]['method'].tolist()
        n_detected = len(detected_methods)
        n_total = len(results)

        if n_detected == 0:
            return (
                f"Out-of-sample validation against the {self.crisis_name} reveals that none of "
                f"the four aggregation methods produced a threshold breach prior to the event, "
                f"suggesting potential limitations in real-time crisis detection."
            )

        # Best lead time
        detected_df = results[results['detected']].copy()
        best_method = detected_df.loc[detected_df['lead_time_days'].idxmax(), 'method']
        best_lead = detected_df['lead_time_days'].max()

        # Peak values
        max_peak_method = results.loc[results['peak_asri'].idxmax(), 'method']
        max_peak = results['peak_asri'].max()

        method_names = {
            'linear': 'linear aggregation',
            'ciss': 'CISS (portfolio variance)',
            'copula': 'copula-based tail dependence',
            'regime': 'regime-switching',
        }

        summary = (
            f"Out-of-sample validation against the {self.crisis_name} demonstrates that "
            f"{n_detected} of {n_total} aggregation methods successfully detected the crisis "
            f"before its onset. "
        )

        if n_detected > 1:
            method_list = ", ".join([method_names.get(m, m) for m in detected_methods[:-1]])
            method_list += f", and {method_names.get(detected_methods[-1], detected_methods[-1])}"
            summary += f"These include {method_list}. "
        else:
            summary += f"The {method_names.get(detected_methods[0], detected_methods[0])} method was the only successful detector. "

        summary += (
            f"The {method_names.get(best_method, best_method)} method achieved the longest "
            f"lead time of {best_lead} days, while {method_names.get(max_peak_method, max_peak_method)} "
            f"produced the highest peak signal ({max_peak:.1f}). "
        )

        # Compare train vs test means
        mean_train = results['mean_asri_train'].mean()
        mean_test = results['mean_asri_test'].mean()

        if mean_test > mean_train * 1.2:
            summary += (
                "The elevated test-period means across all methods suggest genuine structural "
                "change in systemic risk conditions during 2024, rather than in-sample overfitting."
            )
        elif mean_test < mean_train * 0.8:
            summary += (
                "The lower test-period means suggest 2024 was generally less stressed than the "
                "training period, making the August crisis detection particularly meaningful."
            )

        return summary


# -----------------------------------------------------------------------------
# Demonstration with synthetic data
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Out-of-Sample Validation - Synthetic Data Demonstration")
    print("=" * 70)

    # Generate synthetic sub-index data spanning 2021-2024
    np.random.seed(42)

    dates = pd.date_range("2021-01-01", "2024-12-31", freq="D")
    n = len(dates)

    # Base stress levels with trend and seasonality
    t = np.arange(n)
    trend = 35 + 0.005 * t  # Slight upward trend
    seasonality = 5 * np.sin(2 * np.pi * t / 365)  # Annual cycle

    # Generate correlated sub-indices
    common_factor = np.random.normal(0, 1, n)

    stablecoin = trend + seasonality + 15 * common_factor + np.random.normal(0, 5, n)
    defi = trend * 0.9 + seasonality * 0.8 + 12 * common_factor + np.random.normal(0, 6, n)
    contagion = trend * 1.1 + seasonality * 1.2 + 18 * common_factor + np.random.normal(0, 4, n)
    arbitrage = trend * 0.8 + seasonality * 0.9 + 10 * common_factor + np.random.normal(0, 7, n)

    # Inject August 2024 crisis
    crisis_start = pd.Timestamp("2024-07-15")
    crisis_peak = pd.Timestamp("2024-08-05")
    crisis_end = pd.Timestamp("2024-08-20")

    for date, col_name, col_data in [
        (crisis_peak, 'stablecoin', stablecoin),
        (crisis_peak, 'defi', defi),
        (crisis_peak, 'contagion', contagion),
        (crisis_peak, 'arbitrage', arbitrage),
    ]:
        crisis_mask = (dates >= crisis_start) & (dates <= crisis_end)
        days_to_peak = (crisis_peak - crisis_start).days
        days_after_peak = (crisis_end - crisis_peak).days

        # Build up before peak, decay after
        ramp_up = np.linspace(0, 35, days_to_peak)
        ramp_down = np.linspace(35, 10, days_after_peak + 1)
        spike = np.concatenate([ramp_up, ramp_down])[:crisis_mask.sum()]

        if col_name == 'stablecoin':
            stablecoin[crisis_mask] += spike
        elif col_name == 'defi':
            defi[crisis_mask] += spike * 0.9
        elif col_name == 'contagion':
            contagion[crisis_mask] += spike * 1.1
        else:
            arbitrage[crisis_mask] += spike * 0.7

    # Clip to valid range
    stablecoin = np.clip(stablecoin, 0, 100)
    defi = np.clip(defi, 0, 100)
    contagion = np.clip(contagion, 0, 100)
    arbitrage = np.clip(arbitrage, 0, 100)

    # Create DataFrame
    subindices = pd.DataFrame({
        'stablecoin_risk': stablecoin,
        'defi_liquidity_risk': defi,
        'contagion_risk': contagion,
        'arbitrage_opacity': arbitrage,
    }, index=dates)

    print(f"\nGenerated {n} days of synthetic data")
    print(f"Date range: {dates.min().date()} to {dates.max().date()}")
    print(f"\nSub-index summary:")
    print(subindices.describe().round(2))

    # Run validation
    print("\n" + "-" * 70)
    print("Running out-of-sample validation...")
    print("-" * 70)

    validator = OutOfSampleValidator(
        train_end='2023-12-31',
        threshold=70.0,
    )

    # Split data
    train, test = validator.split_data(subindices)
    print(f"\nTrain period: {train.index.min().date()} to {train.index.max().date()} ({len(train)} obs)")
    print(f"Test period:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} obs)")

    # Run all methods
    results = validator.run_all_methods(subindices)

    print("\n" + "-" * 70)
    print("Validation Results:")
    print("-" * 70)
    print(results.to_string(index=False))

    # Summary
    print("\n" + "-" * 70)
    print("Summary:")
    print("-" * 70)
    print(validator.summarize_findings(results))

    # LaTeX table
    print("\n" + "-" * 70)
    print("LaTeX Table:")
    print("-" * 70)
    print(validator.format_validation_table(results))

    # Bootstrap CI (just for linear as demo - faster)
    print("\n" + "-" * 70)
    print("Computing bootstrap confidence intervals (linear method)...")
    print("-" * 70)

    try:
        boot_result = validator.bootstrap_confidence_intervals(
            subindices,
            method='linear',
            n_bootstrap=100,  # Reduced for demo speed
            confidence=0.90,
            block_size=20,
        )

        print(f"Bootstrap samples used: {boot_result.n_bootstrap}")
        print(f"Confidence level: {boot_result.confidence_level:.0%}")
        print(f"\nSample CI values (last 5 days):")
        ci_sample = pd.DataFrame({
            'Mean': boot_result.mean_asri.tail(5),
            'Lower 90% CI': boot_result.lower_ci.tail(5),
            'Upper 90% CI': boot_result.upper_ci.tail(5),
        }).round(2)
        print(ci_sample)

    except Exception as e:
        print(f"Bootstrap failed: {e}")

    print("\n" + "=" * 70)
    print("Demonstration complete.")
    print("=" * 70)
