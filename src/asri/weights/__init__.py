"""
ASRI Weight Derivation Module

Empirically derive sub-index weights instead of using ad-hoc theoretical values.
This is the core innovation that transforms ASRI from "a nice idea" into
"a rigorous, data-driven risk measure."

Five approaches:
1. PCA: Weights from first principal component loadings
2. Elastic Net: Weights from predictive regression on forward returns
3. CRITIC: Weights from contrast intensity × information content
4. Entropy: Weights from Shannon entropy (inverse of uniformity)
5. Granger: Weights proportional to predictive power (F-statistics)
"""

from .pca import (
    PCAWeightDeriver,
    derive_pca_weights,
)
from .elastic_net import (
    ElasticNetWeightDeriver,
    derive_elastic_net_weights,
)
from .critic import (
    CRITICWeightDeriver,
    derive_critic_weights,
)
from .entropy import (
    EntropyWeightDeriver,
    derive_entropy_weights,
)
from .comparison import (
    compare_weight_methods,
    weight_sensitivity_analysis,
    WeightComparison,
)

__all__ = [
    "PCAWeightDeriver",
    "derive_pca_weights",
    "ElasticNetWeightDeriver",
    "derive_elastic_net_weights",
    "CRITICWeightDeriver",
    "derive_critic_weights",
    "EntropyWeightDeriver",
    "derive_entropy_weights",
    "compare_weight_methods",
    "weight_sensitivity_analysis",
    "WeightComparison",
]
