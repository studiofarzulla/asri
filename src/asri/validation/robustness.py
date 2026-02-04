"""
Robustness Tests for ASRI

Placebo tests and structural break analysis to ensure ASRI
signal is genuine and not spurious.

Key tests:
1. Placebo: Shuffle crisis dates → ASRI should NOT predict
2. Permutation: Randomize sub-index assignment → performance should degrade
3. Structural breaks: Test if model parameters are stable over time
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PlaceboTestResult:
    """Results from placebo testing."""
    test_type: Literal["date_shuffle", "feature_permutation", "random_labels"]
    
    # Original performance
    original_auc: float
    
    # Placebo distribution
    placebo_aucs: np.ndarray
    placebo_mean: float
    placebo_std: float
    
    # Statistical test
    z_score: float
    p_value: float
    is_genuine: bool  # True if original significantly beats placebo
    
    # Interpretation
    conclusion: str


@dataclass
class StructuralBreakResult:
    """Results from structural break testing."""
    test_type: Literal["cusum", "chow", "bai_perron"]
    
    # Break detection
    n_breaks_detected: int
    break_dates: list[pd.Timestamp]
    
    # Test statistics
    test_statistic: float
    critical_value: float
    p_value: float | None
    
    # Stability assessment
    is_stable: bool
    stability_score: float  # 0-1, higher = more stable
    
    # Interpretation
    conclusion: str


def run_placebo_date_shuffle(
    asri: pd.Series,
    crisis_dates: list,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> PlaceboTestResult:
    """
    Placebo test: shuffle crisis dates randomly.
    
    If ASRI genuinely predicts crises, shuffling dates should
    destroy its predictive power.
    
    Args:
        asri: ASRI time series
        crisis_dates: List of actual crisis dates
        n_permutations: Number of random shuffles
        random_state: Random seed
        
    Returns:
        PlaceboTestResult
    """
    from .roc_analysis import compute_precision_recall
    
    # Original performance
    orig_precision, orig_recall = compute_precision_recall(asri, crisis_dates)
    orig_f1 = 2 * orig_precision * orig_recall / (orig_precision + orig_recall) if (orig_precision + orig_recall) > 0 else 0
    
    # Placebo: shuffle dates
    rng = np.random.default_rng(random_state)
    placebo_f1s = []
    
    # Available dates in ASRI series
    available_dates = asri.index.tolist()
    
    for _ in range(n_permutations):
        # Random crisis dates
        fake_dates = rng.choice(available_dates, size=len(crisis_dates), replace=False)
        fake_dates = [pd.Timestamp(d) for d in fake_dates]
        
        prec, rec = compute_precision_recall(asri, fake_dates)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        placebo_f1s.append(f1)
    
    placebo_f1s = np.array(placebo_f1s)
    
    # Statistical test
    placebo_mean = np.mean(placebo_f1s)
    placebo_std = np.std(placebo_f1s)
    
    z_score = (orig_f1 - placebo_mean) / placebo_std if placebo_std > 0 else 0
    p_value = 1 - stats.norm.cdf(z_score)
    
    is_genuine = p_value < 0.05
    
    if is_genuine:
        conclusion = f"ASRI signal is genuine: original F1 ({orig_f1:.3f}) significantly exceeds placebo ({placebo_mean:.3f} ± {placebo_std:.3f}), p={p_value:.4f}"
    else:
        conclusion = f"WARNING: ASRI may be spurious. Original F1 ({orig_f1:.3f}) not significantly different from placebo ({placebo_mean:.3f}), p={p_value:.4f}"
    
    return PlaceboTestResult(
        test_type="date_shuffle",
        original_auc=orig_f1,
        placebo_aucs=placebo_f1s,
        placebo_mean=placebo_mean,
        placebo_std=placebo_std,
        z_score=z_score,
        p_value=p_value,
        is_genuine=is_genuine,
        conclusion=conclusion,
    )


def run_placebo_feature_permutation(
    sub_indices: pd.DataFrame,
    weights: dict[str, float],
    target: pd.Series,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> PlaceboTestResult:
    """
    Placebo test: permute sub-index assignments.
    
    If specific sub-indices matter, randomly reassigning them
    should degrade performance.
    
    Args:
        sub_indices: DataFrame with sub-index time series
        weights: Original weights
        target: Target variable
        n_permutations: Number of permutations
        random_state: Random seed
        
    Returns:
        PlaceboTestResult
    """
    from scipy import stats as sp_stats
    
    # Align data
    common_idx = sub_indices.index.intersection(target.index)
    X = sub_indices.loc[common_idx]
    y = target.loc[common_idx].values
    
    # Original ASRI
    orig_asri = sum(weights.get(c, 0) * X[c] for c in X.columns)
    orig_corr = abs(sp_stats.spearmanr(orig_asri.values, y)[0])
    
    # Permutation test
    rng = np.random.default_rng(random_state)
    placebo_corrs = []
    
    columns = list(X.columns)
    
    for _ in range(n_permutations):
        # Shuffle column assignments
        shuffled_cols = rng.permutation(columns)
        shuffled_weights = {orig: weights.get(shuffled, 0) 
                          for orig, shuffled in zip(columns, shuffled_cols)}
        
        permuted_asri = sum(shuffled_weights.get(c, 0) * X[c] for c in X.columns)
        corr = abs(sp_stats.spearmanr(permuted_asri.values, y)[0])
        placebo_corrs.append(corr)
    
    placebo_corrs = np.array(placebo_corrs)
    
    placebo_mean = np.mean(placebo_corrs)
    placebo_std = np.std(placebo_corrs)
    
    z_score = (orig_corr - placebo_mean) / placebo_std if placebo_std > 0 else 0
    p_value = 1 - stats.norm.cdf(z_score)
    
    is_genuine = p_value < 0.05
    
    if is_genuine:
        conclusion = f"Sub-index assignments matter: correlation ({orig_corr:.3f}) significantly exceeds permuted ({placebo_mean:.3f}), p={p_value:.4f}"
    else:
        conclusion = f"WARNING: Sub-index assignments may be arbitrary. p={p_value:.4f}"
    
    return PlaceboTestResult(
        test_type="feature_permutation",
        original_auc=orig_corr,
        placebo_aucs=placebo_corrs,
        placebo_mean=placebo_mean,
        placebo_std=placebo_std,
        z_score=z_score,
        p_value=p_value,
        is_genuine=is_genuine,
        conclusion=conclusion,
    )


def run_placebo_tests(
    asri: pd.Series,
    sub_indices: pd.DataFrame,
    weights: dict[str, float],
    target: pd.Series,
    crisis_dates: list,
    n_permutations: int = 1000,
) -> list[PlaceboTestResult]:
    """
    Run all placebo tests.
    
    Returns:
        List of PlaceboTestResult
    """
    results = []
    
    # Date shuffle test
    results.append(run_placebo_date_shuffle(asri, crisis_dates, n_permutations))
    
    # Feature permutation test
    results.append(run_placebo_feature_permutation(
        sub_indices, weights, target, n_permutations
    ))
    
    return results


def structural_break_test(
    asri: pd.Series,
    method: Literal["cusum", "chow"] = "cusum",
    significance: float = 0.05,
) -> StructuralBreakResult:
    """
    Test for structural breaks in ASRI dynamics.
    
    If breaks exist, the model may need regime-specific parameters.
    
    Args:
        asri: ASRI time series
        method: Test method ('cusum' or 'chow')
        significance: Significance level
        
    Returns:
        StructuralBreakResult
    """
    values = asri.dropna().values
    n = len(values)
    
    if method == "cusum":
        # CUSUM test
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Recursive residuals (simplified: use demeaned values)
        residuals = values - mean_val
        
        # Cumulative sum
        cusum = np.cumsum(residuals) / (std_val * np.sqrt(n))
        
        # Maximum absolute CUSUM
        max_cusum = np.max(np.abs(cusum))
        
        # Critical value (Brownian bridge)
        # Approximate: 1.36 for 5%, 1.63 for 1%
        if significance <= 0.01:
            critical = 1.63
        elif significance <= 0.05:
            critical = 1.36
        else:
            critical = 1.22
        
        # Detect breaks where CUSUM exceeds critical
        break_points = np.where(np.abs(cusum) > critical)[0]
        
        # Cluster break points
        if len(break_points) > 0:
            breaks = [break_points[0]]
            for bp in break_points[1:]:
                if bp - breaks[-1] > 30:  # At least 30 days apart
                    breaks.append(bp)
            break_dates = [asri.index[bp] for bp in breaks if bp < len(asri.index)]
        else:
            break_dates = []
        
        is_stable = max_cusum < critical
        stability_score = 1 - min(max_cusum / (2 * critical), 1)
        
        if is_stable:
            conclusion = f"No structural breaks detected (CUSUM={max_cusum:.3f} < {critical:.3f})"
        else:
            conclusion = f"Structural break(s) detected at {[d.strftime('%Y-%m') for d in break_dates]}"
        
        return StructuralBreakResult(
            test_type="cusum",
            n_breaks_detected=len(break_dates),
            break_dates=break_dates,
            test_statistic=max_cusum,
            critical_value=critical,
            p_value=None,
            is_stable=is_stable,
            stability_score=stability_score,
            conclusion=conclusion,
        )
    
    elif method == "chow":
        # Chow test (single break at midpoint)
        mid = n // 2
        
        # Full sample regression (AR(1))
        y = values[1:]
        X = np.column_stack([np.ones(n-1), values[:-1]])
        
        beta_full = np.linalg.lstsq(X, y, rcond=None)[0]
        ssr_full = np.sum((y - X @ beta_full) ** 2)
        
        # Split samples
        y1, y2 = y[:mid-1], y[mid-1:]
        X1, X2 = X[:mid-1], X[mid-1:]
        
        beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
        ssr1 = np.sum((y1 - X1 @ beta1) ** 2)
        
        beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
        ssr2 = np.sum((y2 - X2 @ beta2) ** 2)
        
        # Chow statistic
        k = X.shape[1]
        chow_stat = ((ssr_full - ssr1 - ssr2) / k) / ((ssr1 + ssr2) / (n - 2*k))
        
        # F critical value
        critical = stats.f.ppf(1 - significance, k, n - 2*k)
        p_value = 1 - stats.f.cdf(chow_stat, k, n - 2*k)
        
        is_stable = chow_stat < critical
        stability_score = 1 - min(chow_stat / (2 * critical), 1)
        
        if is_stable:
            conclusion = f"No structural break at midpoint (Chow={chow_stat:.3f}, p={p_value:.3f})"
            break_dates = []
        else:
            break_dates = [asri.index[mid]]
            conclusion = f"Structural break detected at {break_dates[0].strftime('%Y-%m')} (p={p_value:.3f})"
        
        return StructuralBreakResult(
            test_type="chow",
            n_breaks_detected=len(break_dates),
            break_dates=break_dates,
            test_statistic=chow_stat,
            critical_value=critical,
            p_value=p_value,
            is_stable=is_stable,
            stability_score=stability_score,
            conclusion=conclusion,
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")


def format_robustness_table(
    placebo_results: list[PlaceboTestResult],
    break_result: StructuralBreakResult,
) -> str:
    """Format robustness test results as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Robustness Tests}",
        r"\label{tab:robustness}",
        r"\small",
        r"\begin{tabular}{llccl}",
        r"\toprule",
        r"Test & Original & Placebo Mean & $p$-value & Result \\",
        r"\midrule",
    ]
    
    for r in placebo_results:
        result_str = r"\checkmark" if r.is_genuine else r"$\times$"
        lines.append(
            f"{r.test_type.replace('_', ' ').title()} & {r.original_auc:.3f} & "
            f"{r.placebo_mean:.3f} $\\pm$ {r.placebo_std:.3f} & "
            f"{r.p_value:.4f} & {result_str} \\\\"
        )
    
    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{l}{\textbf{Structural Break Test}} \\",
        f"Method: {break_result.test_type.upper()} & "
        f"Stat: {break_result.test_statistic:.3f} & "
        f"Crit: {break_result.critical_value:.3f} & "
        f"{'--' if break_result.p_value is None else f'{break_result.p_value:.4f}'} & "
        f"{'Stable' if break_result.is_stable else f'{break_result.n_breaks_detected} break(s)'} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item \checkmark = ASRI signal is genuine; $\times$ = potential concern",
        r"\item Placebo tests use 1000 permutations.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)
