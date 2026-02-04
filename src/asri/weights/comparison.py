"""
Weight Method Comparison and Sensitivity Analysis

Compare theoretical weights vs. data-driven methods (PCA, Elastic Net, CRITIC, Entropy).
This is the key analysis that either validates the theoretical framework
or exposes it as ad-hoc speculation.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pca import PCAWeightDeriver, PCAResult
from .elastic_net import ElasticNetWeightDeriver, ElasticNetResult
from .critic import CRITICWeightDeriver, CRITICResult
from .entropy import EntropyWeightDeriver, EntropyResult


@dataclass
class WeightComparison:
    """Comparison of weight derivation methods."""
    theoretical_weights: dict[str, float]
    pca_weights: dict[str, float]
    elastic_net_weights: dict[str, float]
    critic_weights: dict[str, float]
    entropy_weights: dict[str, float]
    granger_weights: dict[str, float] | None

    # Correlation between weight vectors
    pca_vs_theoretical: float
    elastic_net_vs_theoretical: float
    critic_vs_theoretical: float
    entropy_vs_theoretical: float
    pca_vs_elastic_net: float

    # Performance comparison (if available)
    theoretical_auc: float | None = None
    pca_auc: float | None = None
    elastic_net_auc: float | None = None
    critic_auc: float | None = None
    entropy_auc: float | None = None

    # Detailed results
    pca_result: PCAResult | None = None
    elastic_net_result: ElasticNetResult | None = None
    critic_result: CRITICResult | None = None
    entropy_result: EntropyResult | None = None


def _weight_correlation(w1: dict[str, float], w2: dict[str, float]) -> float:
    """Compute correlation between two weight vectors."""
    common_keys = sorted(set(w1.keys()) & set(w2.keys()))
    if len(common_keys) < 2:
        return np.nan
    
    v1 = np.array([w1[k] for k in common_keys])
    v2 = np.array([w2[k] for k in common_keys])
    
    corr = np.corrcoef(v1, v2)[0, 1]
    return corr


def compare_weight_methods(
    sub_indices: pd.DataFrame,
    theoretical_weights: dict[str, float] | None = None,
    market_returns: pd.Series | None = None,
    granger_weights: dict[str, float] | None = None,
) -> WeightComparison:
    """
    Compare different weight derivation methods.

    Args:
        sub_indices: DataFrame with sub-index time series
        theoretical_weights: Original theoretical weights
        market_returns: Optional market returns for Elastic Net target
        granger_weights: Optional pre-computed Granger-based weights

    Returns:
        WeightComparison with all weight vectors and correlations
    """
    # Default theoretical weights
    if theoretical_weights is None:
        theoretical_weights = {
            "stablecoin_risk": 0.30,
            "defi_liquidity_risk": 0.25,
            "contagion_risk": 0.25,
            "arbitrage_opacity": 0.20,
        }

    # Derive PCA weights
    pca_deriver = PCAWeightDeriver()
    pca_deriver.fit(sub_indices)
    pca_weights = pca_deriver.weights

    # Derive Elastic Net weights
    en_deriver = ElasticNetWeightDeriver()
    en_deriver.fit(sub_indices, market_returns)
    en_weights = en_deriver.weights

    # Derive CRITIC weights
    critic_deriver = CRITICWeightDeriver()
    critic_deriver.fit(sub_indices)
    critic_weights = critic_deriver.weights

    # Derive Entropy weights
    entropy_deriver = EntropyWeightDeriver()
    entropy_deriver.fit(sub_indices)
    entropy_weights = entropy_deriver.weights

    # Compute correlations
    pca_vs_theo = _weight_correlation(pca_weights, theoretical_weights)
    en_vs_theo = _weight_correlation(en_weights, theoretical_weights)
    critic_vs_theo = _weight_correlation(critic_weights, theoretical_weights)
    entropy_vs_theo = _weight_correlation(entropy_weights, theoretical_weights)
    pca_vs_en = _weight_correlation(pca_weights, en_weights)

    return WeightComparison(
        theoretical_weights=theoretical_weights,
        pca_weights=pca_weights,
        elastic_net_weights=en_weights,
        critic_weights=critic_weights,
        entropy_weights=entropy_weights,
        granger_weights=granger_weights,
        pca_vs_theoretical=pca_vs_theo,
        elastic_net_vs_theoretical=en_vs_theo,
        critic_vs_theoretical=critic_vs_theo,
        entropy_vs_theoretical=entropy_vs_theo,
        pca_vs_elastic_net=pca_vs_en,
        pca_result=pca_deriver.result,
        elastic_net_result=en_deriver.result,
        critic_result=critic_deriver.result,
        entropy_result=entropy_deriver.result,
    )


@dataclass
class SensitivityResult:
    """Result from weight sensitivity analysis."""
    base_weights: dict[str, float]
    perturbation_range: tuple[float, float]
    
    # Performance at each perturbation level
    performance_matrix: pd.DataFrame  # rows = variable, cols = perturbation
    
    # Which weights are most sensitive?
    sensitivity_ranking: list[tuple[str, float]]  # (variable, sensitivity)
    
    # Optimal weights found during sensitivity analysis
    optimal_weights: dict[str, float]
    optimal_performance: float


def weight_sensitivity_analysis(
    sub_indices: pd.DataFrame,
    base_weights: dict[str, float],
    target: pd.Series,
    perturbation_range: tuple[float, float] = (-0.10, 0.10),
    n_steps: int = 21,
    metric: str = "auc",
) -> SensitivityResult:
    """
    Analyze sensitivity of performance to weight perturbations.
    
    For each sub-index, perturb its weight while re-normalizing others,
    and measure impact on predictive performance.
    
    Args:
        sub_indices: DataFrame with sub-index time series
        base_weights: Starting weight vector
        target: Target variable (e.g., forward drawdown, crisis indicator)
        perturbation_range: Range of perturbations to test
        n_steps: Number of perturbation levels
        metric: Performance metric ('auc', 'r2', 'mse')
        
    Returns:
        SensitivityResult with performance matrix and rankings
    """
    variables = list(base_weights.keys())
    perturbations = np.linspace(perturbation_range[0], perturbation_range[1], n_steps)
    
    # Align data
    data = sub_indices[variables].dropna()
    min_len = min(len(data), len(target))
    X = data.iloc[:min_len]
    y = target.iloc[:min_len].values
    
    # Performance matrix
    perf_matrix = pd.DataFrame(index=variables, columns=perturbations, dtype=float)
    
    for var in variables:
        for delta in perturbations:
            # Create perturbed weights
            perturbed = base_weights.copy()
            perturbed[var] = max(0, perturbed[var] + delta)
            
            # Renormalize
            total = sum(perturbed.values())
            perturbed = {k: v / total for k, v in perturbed.items()}
            
            # Compute ASRI with perturbed weights
            asri = sum(perturbed[c] * X[c] for c in variables)
            
            # Compute performance
            perf = _compute_metric(asri.values, y, metric)
            perf_matrix.loc[var, delta] = perf
    
    # Compute sensitivity (range of performance across perturbations)
    sensitivity = {
        var: perf_matrix.loc[var].max() - perf_matrix.loc[var].min()
        for var in variables
    }
    sensitivity_ranking = sorted(sensitivity.items(), key=lambda x: -x[1])
    
    # Find optimal weights
    best_perf = -np.inf
    best_weights = base_weights.copy()
    
    for var in variables:
        for delta in perturbations:
            perf = perf_matrix.loc[var, delta]
            if perf > best_perf:
                best_perf = perf
                best_weights = base_weights.copy()
                best_weights[var] = max(0, best_weights[var] + delta)
                total = sum(best_weights.values())
                best_weights = {k: v / total for k, v in best_weights.items()}
    
    return SensitivityResult(
        base_weights=base_weights,
        perturbation_range=perturbation_range,
        performance_matrix=perf_matrix,
        sensitivity_ranking=sensitivity_ranking,
        optimal_weights=best_weights,
        optimal_performance=best_perf,
    )


def _compute_metric(
    predictions: np.ndarray,
    target: np.ndarray,
    metric: str,
) -> float:
    """Compute performance metric."""
    if metric == "r2":
        ss_res = np.sum((target - predictions) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    elif metric == "mse":
        return -np.mean((target - predictions) ** 2)  # Negative so higher is better
    
    elif metric == "auc":
        # Convert predictions to binary classification
        # Assume higher ASRI should predict worse outcomes (higher target)
        from scipy import stats
        
        # Rank correlation as proxy for AUC
        corr, _ = stats.spearmanr(predictions, target)
        return abs(corr)  # Use absolute since direction may be inverted
    
    elif metric == "correlation":
        return np.corrcoef(predictions, target)[0, 1]
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def format_weight_comparison_table(comparison: WeightComparison) -> str:
    """Format weight comparison as LaTeX table."""
    variables = sorted(comparison.theoretical_weights.keys())

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison of Weight Derivation Methods}",
        r"\label{tab:weight_comparison}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Sub-Index & Theoretical & PCA & Elastic Net & CRITIC & Entropy & Granger \\",
        r"\midrule",
    ]

    for var in variables:
        theo = comparison.theoretical_weights.get(var, 0)
        pca = comparison.pca_weights.get(var, 0)
        en = comparison.elastic_net_weights.get(var, 0)
        critic = comparison.critic_weights.get(var, 0)
        entropy = comparison.entropy_weights.get(var, 0)
        granger = comparison.granger_weights.get(var, 0) if comparison.granger_weights else None

        granger_str = f"{granger:.3f}" if granger is not None else "--"
        lines.append(
            f"{var} & {theo:.3f} & {pca:.3f} & {en:.3f} & {critic:.3f} & {entropy:.3f} & {granger_str} \\\\"
        )

    lines.extend([
        r"\midrule",
        r"\multicolumn{7}{l}{\textbf{Correlation with Theoretical Weights}} \\",
        f"PCA & -- & {comparison.pca_vs_theoretical:.3f} & -- & -- & -- & -- \\\\",
        f"Elastic Net & -- & -- & {comparison.elastic_net_vs_theoretical:.3f} & -- & -- & -- \\\\",
        f"CRITIC & -- & -- & -- & {comparison.critic_vs_theoretical:.3f} & -- & -- \\\\",
        f"Entropy & -- & -- & -- & -- & {comparison.entropy_vs_theoretical:.3f} & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Theoretical: domain expertise; PCA: PC1 loadings; Elastic Net: predictive regression;",
        r"\item CRITIC: contrast $\times$ information; Entropy: inverse uniformity; Granger: causality F-stats.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_sensitivity_heatmap(result: SensitivityResult) -> str:
    """Generate code for sensitivity heatmap figure."""
    return f"""
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))

# Performance matrix heatmap
sns.heatmap(
    result.performance_matrix.astype(float),
    cmap='RdYlGn',
    center=result.performance_matrix.values.mean(),
    annot=True,
    fmt='.3f',
    ax=ax,
)

ax.set_xlabel('Weight Perturbation')
ax.set_ylabel('Sub-Index')
ax.set_title('ASRI Performance Sensitivity to Weight Perturbations')

plt.tight_layout()
plt.savefig('figures/weight_sensitivity.pdf')
"""
