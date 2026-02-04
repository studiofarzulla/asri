"""Signal computation module - sub-index calculations."""

from asri.signals.algorithmic_stablecoin import (
    AlgorithmicStablecoinMetrics,
    AlgorithmicStablecoinRiskResult,
    StablecoinType,
    calculate_algorithmic_stablecoin_risk,
    adjust_scr_for_algorithmic_risk,
    classify_stablecoin,
)

__all__ = [
    "AlgorithmicStablecoinMetrics",
    "AlgorithmicStablecoinRiskResult",
    "StablecoinType",
    "calculate_algorithmic_stablecoin_risk",
    "adjust_scr_for_algorithmic_risk",
    "classify_stablecoin",
]
