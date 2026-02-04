"""
CRITIC Weight Derivation for ASRI

CRITIC (Criteria Importance Through Intercriteria Correlation) is an
objective weighting method that accounts for both contrast intensity
(standard deviation) and information content (correlation structure).

Key insight: Variables that are highly correlated with others contain
redundant information and should receive lower weight. Variables that
are uncorrelated AND have high variance are most informative.

Formula:
    C_j = σ_j × Σ_k(1 - |corr_jk|)
    w_j = C_j / Σ C_j

Where:
    σ_j = standard deviation of variable j (contrast intensity)
    corr_jk = correlation between variables j and k
    Σ(1 - |corr|) = information content (higher when uncorrelated)

Reference:
    Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995).
    Determining objective weights in multiple criteria problems:
    The critic method. Computers & Operations Research, 22(7), 763-770.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd


@dataclass
class CRITICResult:
    """Results from CRITIC weight derivation."""
    weights: dict[str, float]
    standard_deviations: dict[str, float]
    correlation_matrix: pd.DataFrame
    contrast_intensity: dict[str, float]  # σ_j (same as std_dev for normalized data)
    information_content: dict[str, float]  # Σ(1 - |corr|)
    critic_scores: dict[str, float]  # C_j = σ_j × info_j


class CRITICWeightDeriver:
    """
    CRITIC objective weighting method.

    CRITIC balances two considerations:
    1. Contrast intensity: Variables with higher variance contain more information
    2. Conflicting criteria: Variables uncorrelated with others are more informative

    This makes it particularly suitable for ASRI where sub-indices may
    capture overlapping aspects of systemic risk.
    """

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: Whether to normalize data to [0,1] before analysis.
                       Recommended for fair comparison across different scales.
        """
        self.normalize = normalize
        self._result: CRITICResult | None = None

    def fit(self, data: pd.DataFrame) -> Self:
        """
        Derive weights using CRITIC method.

        Args:
            data: DataFrame with sub-indices as columns

        Returns:
            Self (for method chaining)
        """
        # Remove NaN rows
        clean_data = data.dropna()

        if len(clean_data) < 10:
            raise ValueError(f"Insufficient observations ({len(clean_data)}) for CRITIC")

        # Normalize data to [0, 1] for fair comparison across scales
        if self.normalize:
            data_range = clean_data.max() - clean_data.min()
            # Handle constant columns
            data_range[data_range == 0] = 1
            normalized = (clean_data - clean_data.min()) / data_range
        else:
            normalized = clean_data

        # Step 1: Compute standard deviations (contrast intensity)
        std_devs = normalized.std()

        # Step 2: Compute correlation matrix
        corr_matrix = normalized.corr()

        # Step 3: Compute information content for each variable
        # info_j = Σ_k (1 - |corr_jk|) for k ≠ j
        info_content = {}
        for col in clean_data.columns:
            info_content[col] = sum(
                1 - abs(corr_matrix.loc[col, other])
                for other in clean_data.columns
                if other != col
            )

        # Step 4: Compute CRITIC scores C_j = σ_j × info_j
        critic_scores = {
            col: std_devs[col] * info_content[col]
            for col in clean_data.columns
        }

        # Step 5: Normalize to weights (sum to 1)
        total = sum(critic_scores.values())
        if total == 0:
            # Edge case: all variables are identical
            n_vars = len(clean_data.columns)
            weights = {col: 1.0 / n_vars for col in clean_data.columns}
        else:
            weights = {col: score / total for col, score in critic_scores.items()}

        self._result = CRITICResult(
            weights=weights,
            standard_deviations=std_devs.to_dict(),
            correlation_matrix=corr_matrix,
            contrast_intensity=std_devs.to_dict(),
            information_content=info_content,
            critic_scores=critic_scores,
        )

        return self

    @property
    def weights(self) -> dict[str, float]:
        """Get derived weights."""
        if self._result is None:
            raise ValueError("Must call fit() first")
        return self._result.weights

    @property
    def result(self) -> CRITICResult:
        """Get full results."""
        if self._result is None:
            raise ValueError("Must call fit() first")
        return self._result


def derive_critic_weights(
    data: pd.DataFrame,
    normalize: bool = True,
) -> dict[str, float]:
    """
    Convenience function to derive CRITIC weights.

    Args:
        data: DataFrame with sub-indices
        normalize: Whether to normalize data first

    Returns:
        Dictionary of variable -> weight
    """
    deriver = CRITICWeightDeriver(normalize=normalize)
    deriver.fit(data)
    return deriver.weights


def format_critic_table(result: CRITICResult) -> str:
    """Format CRITIC results as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{CRITIC Method: Objective Weights from Contrast and Correlation}",
        r"\label{tab:critic}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Sub-Index & $\sigma_j$ & Info Content & CRITIC Score & Weight \\",
        r"\midrule",
    ]

    for col in sorted(result.weights.keys()):
        std = result.standard_deviations[col]
        info = result.information_content[col]
        score = result.critic_scores[col]
        weight = result.weights[col]

        lines.append(f"{col} & {std:.3f} & {info:.3f} & {score:.3f} & {weight:.3f} \\\\")

    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Note: Higher info content $\Rightarrow$ lower correlation with others}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item CRITIC weights: $w_j = C_j / \sum C_j$ where $C_j = \sigma_j \times \sum_k(1 - |\rho_{jk}|)$",
        r"\item Data normalized to $[0,1]$ before analysis.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    return "\n".join(lines)
