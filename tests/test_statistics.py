"""
Tests for ASRI Statistical Module

Validates stationarity tests, Granger causality, confidence intervals,
and descriptive statistics.
"""

import numpy as np
import pandas as pd
import pytest

from asri.statistics.stationarity import (
    test_stationarity,
    test_stationarity_suite,
    StationarityConclusion,
)
from asri.statistics.causality import (
    granger_causality_test,
    granger_causality_matrix,
    GrangerResult,
)
from asri.statistics.confidence import (
    block_bootstrap_ci,
    ConfidenceInterval,
)
from asri.statistics.descriptive import (
    compute_descriptive_stats,
    correlation_matrix_with_significance,
)


class TestStationarity:
    """Tests for stationarity testing module."""
    
    def test_stationary_series(self):
        """Stationary series should be identified as such."""
        np.random.seed(42)
        # White noise is stationary
        series = pd.Series(np.random.randn(500))
        
        result = test_stationarity(series, name="white_noise")
        
        assert result.n_observations == 500
        assert result.adf_pvalue < 0.10  # Should reject unit root
        # Conclusion should be STATIONARY or TREND_STATIONARY
        assert result.conclusion in [
            StationarityConclusion.STATIONARY,
            StationarityConclusion.TREND_STATIONARY,
        ]
    
    def test_nonstationary_series(self):
        """Random walk should be identified as non-stationary."""
        np.random.seed(42)
        # Random walk: y_t = y_{t-1} + e_t
        innovations = np.random.randn(500)
        series = pd.Series(np.cumsum(innovations))
        
        result = test_stationarity(series, name="random_walk")
        
        # ADF should fail to reject (p > 0.05)
        # KPSS should reject (p < 0.05)
        assert result.adf_pvalue > 0.05 or result.kpss_pvalue < 0.05
        assert result.conclusion in [
            StationarityConclusion.NON_STATIONARY,
            StationarityConclusion.INCONCLUSIVE,
        ]
    
    def test_insufficient_data(self):
        """Should raise error with too few observations."""
        series = pd.Series([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError, match="at least 20"):
            test_stationarity(series)
    
    def test_suite_multiple_series(self):
        """Test suite on multiple series."""
        np.random.seed(42)
        data = pd.DataFrame({
            'stationary': np.random.randn(200),
            'trend': np.arange(200) + np.random.randn(200),
        })
        
        results = test_stationarity_suite(data)
        
        assert 'stationary' in results
        assert 'trend' in results


class TestGrangerCausality:
    """Tests for Granger causality module."""
    
    def test_causal_relationship(self):
        """Series with true causal relationship should be detected."""
        np.random.seed(42)
        n = 500
        
        # X causes Y: Y_t = 0.5 * X_{t-1} + e_t
        x = np.random.randn(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * x[t-1] + 0.5 * np.random.randn()
        
        result = granger_causality_test(
            pd.Series(x), pd.Series(y), lag=1,
            cause_name="X", effect_name="Y"
        )
        
        assert isinstance(result, GrangerResult)
        assert result.cause_variable == "X"
        assert result.effect_variable == "Y"
        assert result.f_statistic > 0
        # Should be significant
        assert result.p_value < 0.05
        assert result.is_significant
    
    def test_no_causality(self):
        """Independent series should show no causality."""
        np.random.seed(42)
        n = 500
        
        x = np.random.randn(n)
        y = np.random.randn(n)
        
        result = granger_causality_test(
            pd.Series(x), pd.Series(y), lag=1
        )
        
        # Should not be significant at 5%
        assert result.p_value > 0.05 or result.f_statistic < 4
    
    def test_causality_matrix(self):
        """Test full causality matrix computation."""
        np.random.seed(42)
        data = pd.DataFrame({
            'A': np.random.randn(200),
            'B': np.random.randn(200),
            'C': np.random.randn(200),
        })
        
        matrix = granger_causality_matrix(data, lags=[1, 5])
        
        assert len(matrix.variables) == 3
        assert 1 in matrix.lags_tested
        assert 5 in matrix.lags_tested
        assert len(matrix.results) > 0


class TestConfidenceIntervals:
    """Tests for bootstrap confidence intervals."""
    
    def test_basic_ci(self):
        """Basic CI should work."""
        np.random.seed(42)
        series = pd.Series(np.random.randn(200) + 50)
        
        ci = block_bootstrap_ci(
            series,
            statistic=np.mean,
            n_bootstrap=500,
            confidence_level=0.95,
        )
        
        assert isinstance(ci, ConfidenceInterval)
        assert ci.lower < ci.point_estimate < ci.upper
        assert ci.confidence_level == 0.95
        assert ci.n_bootstrap == 500
        # True mean is 50, CI should contain it
        assert ci.lower < 50 < ci.upper
    
    def test_ci_width(self):
        """Higher confidence should give wider interval."""
        np.random.seed(42)
        series = pd.Series(np.random.randn(200))
        
        ci_90 = block_bootstrap_ci(series, np.mean, confidence_level=0.90, n_bootstrap=300)
        ci_99 = block_bootstrap_ci(series, np.mean, confidence_level=0.99, n_bootstrap=300)
        
        assert ci_99.width > ci_90.width


class TestDescriptiveStats:
    """Tests for descriptive statistics."""
    
    def test_basic_stats(self):
        """Basic stats should be computed correctly."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        stats = compute_descriptive_stats(series, name="test")
        
        assert stats.name == "test"
        assert stats.n == 10
        assert stats.mean == 5.5
        assert stats.min == 1
        assert stats.max == 10
        assert stats.median == 5.5
    
    def test_correlation_matrix(self):
        """Correlation matrix should be symmetric with ones on diagonal."""
        np.random.seed(42)
        data = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.randn(100),
        })
        
        corr, pval = correlation_matrix_with_significance(data)
        
        assert corr.shape == (3, 3)
        assert pval.shape == (3, 3)
        # Diagonal should be 1
        assert corr.loc['A', 'A'] == 1.0
        assert corr.loc['B', 'B'] == 1.0
        # Should be symmetric
        assert corr.loc['A', 'B'] == corr.loc['B', 'A']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
