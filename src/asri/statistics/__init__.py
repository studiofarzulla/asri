"""
ASRI Statistical Testing Module

Rigorous statistical foundations for the Aggregated Systemic Risk Index.
Implements stationarity tests, Granger causality, cointegration analysis,
and bootstrap confidence intervals.
"""

from .stationarity import (
    test_stationarity,
    test_stationarity_suite,
    recommend_transformation,
    StationarityResult,
)
from .causality import (
    granger_causality_test,
    granger_causality_matrix,
    var_lag_selection,
    GrangerResult,
)
from .cointegration import (
    johansen_test,
    engle_granger_test,
    CointegrationResult,
)
from .confidence import (
    block_bootstrap_ci,
    bootstrap_asri_distribution,
    ConfidenceInterval,
)
from .descriptive import (
    compute_descriptive_stats,
    correlation_matrix_with_significance,
    DescriptiveStats,
)

__all__ = [
    # Stationarity
    "test_stationarity",
    "test_stationarity_suite",
    "recommend_transformation",
    "StationarityResult",
    # Causality
    "granger_causality_test",
    "granger_causality_matrix",
    "var_lag_selection",
    "GrangerResult",
    # Cointegration
    "johansen_test",
    "engle_granger_test",
    "CointegrationResult",
    # Confidence
    "block_bootstrap_ci",
    "bootstrap_asri_distribution",
    "ConfidenceInterval",
    # Descriptive
    "compute_descriptive_stats",
    "correlation_matrix_with_significance",
    "DescriptiveStats",
]
