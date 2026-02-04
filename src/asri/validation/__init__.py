"""
ASRI Advanced Validation Module

Rigorous validation framework for crisis prediction performance.
Implements event studies, ROC analysis, walk-forward validation,
and benchmark comparisons.
"""

from .event_study import (
    EventStudyResult,
    run_event_study,
    compute_cumulative_abnormal_signal,
)
from .roc_analysis import (
    CrisisClassificationMetrics,
    compute_roc_metrics,
    compute_precision_recall,
    optimal_threshold,
)
from .walk_forward import (
    WalkForwardResult,
    purged_walk_forward_cv,
    walk_forward_optimization,
)
from .benchmark import (
    BenchmarkComparison,
    compare_with_benchmarks,
    compute_benchmark_metrics,
)
from .robustness import (
    PlaceboTestResult,
    run_placebo_tests,
    structural_break_test,
)
from .ablation import (
    AblationResult,
    CrisisDetectionResult,
    run_ablation_analysis,
    compute_ablated_weights,
    compute_component_importance,
    format_ablation_table,
)

__all__ = [
    # Event study
    "EventStudyResult",
    "run_event_study",
    "compute_cumulative_abnormal_signal",
    # ROC analysis
    "CrisisClassificationMetrics",
    "compute_roc_metrics",
    "compute_precision_recall",
    "optimal_threshold",
    # Walk-forward
    "WalkForwardResult",
    "purged_walk_forward_cv",
    "walk_forward_optimization",
    # Benchmark
    "BenchmarkComparison",
    "compare_with_benchmarks",
    "compute_benchmark_metrics",
    # Robustness
    "PlaceboTestResult",
    "run_placebo_tests",
    "structural_break_test",
    # Ablation
    "AblationResult",
    "CrisisDetectionResult",
    "run_ablation_analysis",
    "compute_ablated_weights",
    "compute_component_importance",
    "format_ablation_table",
]
