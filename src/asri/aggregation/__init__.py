"""
ASRI Aggregation Methods

Provides different approaches for combining sub-indices into the aggregate
ASRI (Aggregated Systemic Risk Index).

Methods:
    - Linear: Simple weighted sum (assumes independence)
    - CISS: Portfolio-theoretic aggregation (captures correlation amplification)
    - Copula: Tail-dependence amplified aggregation (asymmetric crash clustering)
    - Regime: HMM posterior-weighted regime-specific weights

Additional modules:
    - Comparison: Tools for evaluating aggregation method performance across crises
    - Validation: Out-of-sample validation and bootstrap confidence intervals
"""

from .ciss_aggregation import (
    CISSAggregator,
    CISSResult,
    format_ciss_comparison_table,
)
from .copula_aggregation import CANONICAL_COLUMNS
from .comparison import (
    AggregationComparison,
    CrisisEvent,
    CRISIS_EVENTS,
)
from .copula_aggregation import (
    CopulaAggregator,
    CopulaFitResult,
    TailDependenceResult,
    compute_copula_weighted_asri,
)
from .regime_aggregation import (
    RegimeAggregator,
    RegimeAggregationResult,
    RegimeWeights,
    TransitionAnalysis,
    format_regime_weights_table,
    format_transition_analysis_table,
)
from .validation import (
    OutOfSampleValidator,
    ValidationResult,
    BootstrapResult,
    BootstrapDetectionResult,
    OOS_CRISIS,
)

__all__ = [
    # Canonical column names
    "CANONICAL_COLUMNS",
    # CISS aggregation
    "CISSAggregator",
    "CISSResult",
    "format_ciss_comparison_table",
    # Comparison tools
    "AggregationComparison",
    "CrisisEvent",
    "CRISIS_EVENTS",
    # Copula-based tail aggregation
    "CopulaAggregator",
    "CopulaFitResult",
    "TailDependenceResult",
    "compute_copula_weighted_asri",
    # Regime-conditional aggregation
    "RegimeAggregator",
    "RegimeAggregationResult",
    "RegimeWeights",
    "TransitionAnalysis",
    "format_regime_weights_table",
    "format_transition_analysis_table",
    # Out-of-sample validation
    "OutOfSampleValidator",
    "ValidationResult",
    "BootstrapResult",
    "BootstrapDetectionResult",
    "OOS_CRISIS",
]
