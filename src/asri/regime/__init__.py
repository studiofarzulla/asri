"""
ASRI Regime Detection Module

Hidden Markov Models to detect market regimes and enable
regime-conditional weight switching.
"""

from .hmm import (
    RegimeDetector,
    RegimeResult,
    detect_regimes,
)

__all__ = [
    "RegimeDetector",
    "RegimeResult",
    "detect_regimes",
]
