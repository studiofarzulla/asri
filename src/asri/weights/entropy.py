"""
Entropy-Based Weight Derivation for ASRI

Shannon entropy measures the uncertainty/disorder in a probability distribution.
For MCDM (Multi-Criteria Decision Making), entropy weighting assigns lower
weights to criteria with more uniform distributions (high entropy = less
discriminating power) and higher weights to criteria with more varied
distributions (low entropy = more discriminating power).

Key insight: If a sub-index takes similar values across all observations,
it cannot differentiate between high-risk and low-risk states, so it
should receive lower weight.

Formula:
    p_ij = x_ij / Σ_i x_ij  (proportion for each observation)
    E_j = -Σ_i p_ij × ln(p_ij) / ln(n)  (normalized entropy, 0 to 1)
    d_j = 1 - E_j  (divergence/information)
    w_j = d_j / Σ d_j  (normalize to sum to 1)

Reference:
    Shannon, C. E. (1948). A mathematical theory of communication.
    Bell System Technical Journal, 27(3), 379-423.

    Wang, T. C., & Lee, H. D. (2009). Developing a fuzzy TOPSIS approach
    based on subjective weights and objective weights. Expert Systems
    with Applications, 36(5), 8980-8985.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd


@dataclass
class EntropyResult:
    """Results from entropy weight derivation."""
    weights: dict[str, float]
    entropy: dict[str, float]  # E_j (normalized, 0-1)
    divergence: dict[str, float]  # d_j = 1 - E_j
    normalized_matrix: pd.DataFrame  # p_ij values


class EntropyWeightDeriver:
    """
    Shannon entropy-based objective weighting.

    Variables with more uniform distributions have higher entropy and
    receive lower weights (less discriminating power). Variables with
    more skewed/varied distributions receive higher weights.

    This is particularly relevant for ASRI where we want sub-indices
    that can discriminate between crisis and non-crisis periods.
    """

    def __init__(self, epsilon: float = 1e-10):
        """
        Args:
            epsilon: Small constant to avoid log(0). Default 1e-10.
        """
        self.epsilon = epsilon
        self._result: EntropyResult | None = None

    def fit(self, data: pd.DataFrame) -> Self:
        """
        Derive weights using entropy method.

        Args:
            data: DataFrame with sub-indices as columns.
                  Values should be positive (will be shifted if needed).

        Returns:
            Self (for method chaining)
        """
        # Remove NaN rows
        clean_data = data.dropna()
        n = len(clean_data)

        if n < 10:
            raise ValueError(f"Insufficient observations ({n}) for entropy weighting")

        # Step 1: Ensure all values are positive
        # Shift data so minimum is slightly above zero
        shifted_data = clean_data.copy()
        for col in shifted_data.columns:
            col_min = shifted_data[col].min()
            if col_min <= 0:
                shifted_data[col] = shifted_data[col] - col_min + self.epsilon

        # Step 2: Normalize to proportions (p_ij = x_ij / Σ_i x_ij)
        # Each column sums to 1
        col_sums = shifted_data.sum()
        normalized = shifted_data / col_sums

        # Step 3: Compute entropy for each variable
        # E_j = -Σ_i p_ij × ln(p_ij) / ln(n)
        entropy = {}
        k = np.log(n)  # Normalization constant

        for col in clean_data.columns:
            p = normalized[col].values
            # Add epsilon to avoid log(0)
            p = np.clip(p, self.epsilon, 1.0)
            # Shannon entropy, normalized to [0, 1]
            e_j = -np.sum(p * np.log(p)) / k
            entropy[col] = e_j

        # Step 4: Compute divergence (information content)
        # d_j = 1 - E_j
        divergence = {col: 1 - e for col, e in entropy.items()}

        # Step 5: Normalize to weights
        total = sum(divergence.values())
        if total == 0:
            # Edge case: all variables have maximum entropy
            n_vars = len(clean_data.columns)
            weights = {col: 1.0 / n_vars for col in clean_data.columns}
        else:
            weights = {col: d / total for col, d in divergence.items()}

        self._result = EntropyResult(
            weights=weights,
            entropy=entropy,
            divergence=divergence,
            normalized_matrix=normalized,
        )

        return self

    @property
    def weights(self) -> dict[str, float]:
        """Get derived weights."""
        if self._result is None:
            raise ValueError("Must call fit() first")
        return self._result.weights

    @property
    def result(self) -> EntropyResult:
        """Get full results."""
        if self._result is None:
            raise ValueError("Must call fit() first")
        return self._result


def derive_entropy_weights(data: pd.DataFrame) -> dict[str, float]:
    """
    Convenience function to derive entropy weights.

    Args:
        data: DataFrame with sub-indices

    Returns:
        Dictionary of variable -> weight
    """
    deriver = EntropyWeightDeriver()
    deriver.fit(data)
    return deriver.weights


def format_entropy_table(result: EntropyResult) -> str:
    """Format entropy results as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Entropy Method: Objective Weights from Information Content}",
        r"\label{tab:entropy}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Sub-Index & Entropy $E_j$ & Divergence $d_j$ & Weight \\",
        r"\midrule",
    ]

    for col in sorted(result.weights.keys()):
        e = result.entropy[col]
        d = result.divergence[col]
        w = result.weights[col]

        lines.append(f"{col} & {e:.4f} & {d:.4f} & {w:.3f} \\\\")

    lines.extend([
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Note: Higher entropy $\Rightarrow$ more uniform $\Rightarrow$ lower weight}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Entropy: $E_j = -\sum_i p_{ij} \ln(p_{ij}) / \ln(n)$ (normalized to $[0,1]$)",
        r"\item Divergence: $d_j = 1 - E_j$; Weights: $w_j = d_j / \sum d_j$",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    return "\n".join(lines)
