"""
Tests for ASRI Validation Module

Validates event study, ROC analysis, walk-forward validation,
benchmarks, and robustness tests.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from asri.validation.event_study import (
    CrisisEvent,
    EventStudyResult,
    run_event_study,
    compute_cumulative_abnormal_signal,
)
from asri.validation.roc_analysis import (
    compute_roc_metrics,
    compute_forward_crisis_labels,
    optimal_threshold,
)
from asri.validation.walk_forward import (
    purged_walk_forward_cv,
    PurgedTimeSeriesSplit,
)
from asri.validation.robustness import (
    run_placebo_date_shuffle,
    structural_break_test,
)


class TestEventStudy:
    """Tests for event study methodology."""
    
    def test_cumulative_abnormal_signal(self):
        """CAS should be computed correctly."""
        np.random.seed(42)
        
        # Create ASRI that spikes around event
        dates = pd.date_range('2022-01-01', periods=200, freq='D')
        values = np.ones(200) * 50  # Baseline 50
        
        # Event at day 150
        event_date = dates[150]
        
        # Spike in event window
        values[130:160] = 70
        
        asri = pd.Series(values, index=dates)
        
        abnormal, cas, t_stat, p_value = compute_cumulative_abnormal_signal(
            asri, event_date,
            estimation_window=(-90, -31),
            event_window=(-30, 0),
        )
        
        assert cas > 0  # Should be positive (elevated ASRI)
        assert len(abnormal) > 0
    
    def test_event_study_multiple(self):
        """Run event study on multiple events."""
        np.random.seed(42)
        
        dates = pd.date_range('2021-01-01', periods=730, freq='D')
        values = np.random.randn(730) * 5 + 50
        
        # Two events
        events = [
            CrisisEvent("Event 1", dates[200], "Test event 1"),
            CrisisEvent("Event 2", dates[500], "Test event 2"),
        ]
        
        asri = pd.Series(values, index=dates)
        
        results = run_event_study(asri, events)
        
        assert len(results) == 2
        assert all(isinstance(r, EventStudyResult) for r in results)


class TestROCAnalysis:
    """Tests for ROC analysis."""
    
    def test_forward_crisis_labels(self):
        """Crisis labels should be created correctly."""
        np.random.seed(42)
        
        # Random returns
        returns = pd.Series(np.random.randn(200) * 0.02)
        
        labels = compute_forward_crisis_labels(
            returns,
            forward_window=30,
            drawdown_threshold=-0.10,
        )
        
        assert len(labels) == 200
        assert set(labels.unique()).issubset({0, 1})
    
    def test_optimal_threshold(self):
        """Optimal threshold should be found."""
        np.random.seed(42)
        
        # Create data where high scores predict positive class
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = np.concatenate([
            np.random.randn(50) * 10 + 40,  # Low scores for negatives
            np.random.randn(50) * 10 + 60,  # High scores for positives
        ])
        
        thresh, metrics = optimal_threshold(y_true, y_score, criterion="f1")
        
        assert 40 < thresh < 60
        assert metrics['precision'] > 0.5
        assert metrics['recall'] > 0.5


class TestWalkForward:
    """Tests for walk-forward validation."""
    
    def test_purged_split(self):
        """Purged split should have proper gaps."""
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        data = pd.DataFrame({'A': np.random.randn(1000)}, index=dates)
        
        splitter = PurgedTimeSeriesSplit(
            n_splits=3,
            purge_days=30,
            min_train_days=200,
        )
        
        splits = list(splitter.split(data))
        
        assert len(splits) > 0
        
        for train_idx, test_idx, meta in splits:
            # Check purge gap
            train_end = data.index[train_idx[-1]]
            test_start = data.index[test_idx[0]]
            gap = (test_start - train_end).days
            
            assert gap >= 30  # Purge gap should be at least 30 days
    
    def test_walk_forward_cv(self):
        """Walk-forward CV should run."""
        np.random.seed(42)
        
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        sub_indices = pd.DataFrame({
            'A': np.random.randn(1000),
            'B': np.random.randn(1000),
        }, index=dates)
        
        target = pd.Series(np.random.randn(1000), index=dates)
        weights = {'A': 0.5, 'B': 0.5}
        
        result = purged_walk_forward_cv(
            sub_indices, target, weights,
            n_splits=3,
            purge_days=30,
        )
        
        assert result.n_folds > 0
        assert result.mean_test_r2 is not None


class TestRobustness:
    """Tests for robustness module."""
    
    def test_structural_break_cusum(self):
        """CUSUM test should run."""
        np.random.seed(42)
        
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        values = np.random.randn(500) + 50
        
        asri = pd.Series(values, index=dates)
        
        result = structural_break_test(asri, method="cusum")
        
        assert result.test_type == "cusum"
        assert result.test_statistic is not None
        assert result.stability_score >= 0
        assert result.stability_score <= 1
    
    def test_structural_break_with_break(self):
        """Should detect obvious structural break."""
        np.random.seed(42)
        
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        # Clear break at midpoint
        values = np.concatenate([
            np.random.randn(250) + 30,
            np.random.randn(250) + 70,
        ])
        
        asri = pd.Series(values, index=dates)
        
        result = structural_break_test(asri, method="cusum")
        
        # Should detect the break
        assert result.n_breaks_detected >= 1 or not result.is_stable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
