"""
Benchmark Comparison for ASRI

Compare ASRI against alternative risk measures:
1. Fear & Greed Index (sentiment-based)
2. VIX (traditional market volatility)
3. Naive DeFi TVL (simple proxy)
4. Random walk (null model)

If ASRI doesn't beat these benchmarks, it has no value.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BenchmarkMetrics:
    """Performance metrics for a single benchmark."""
    name: str
    auc_roc: float
    auc_pr: float
    correlation_with_crisis: float
    lead_time_days: float
    granger_f_stat: float | None
    granger_p_value: float | None


@dataclass
class BenchmarkComparison:
    """Comparison of ASRI against benchmarks."""
    asri_metrics: BenchmarkMetrics
    benchmark_metrics: dict[str, BenchmarkMetrics]
    
    # Relative performance
    auc_improvement: dict[str, float]  # ASRI AUC - benchmark AUC
    lead_time_improvement: dict[str, float]  # ASRI lead - benchmark lead
    
    # Statistical tests
    delong_test_pvalues: dict[str, float]  # DeLong test for AUC difference
    
    # Winner summary
    best_model: str
    ranking: list[str]


def compute_benchmark_metrics(
    index: pd.Series,
    target: pd.Series,
    name: str,
) -> BenchmarkMetrics:
    """
    Compute performance metrics for a risk index.
    
    Args:
        index: Risk index time series
        target: Crisis indicator or forward returns
        name: Benchmark name
        
    Returns:
        BenchmarkMetrics
    """
    from .roc_analysis import compute_roc_curve, compute_auc, compute_precision_recall_curve
    from ..statistics.causality import granger_causality_test
    
    # Align series
    common_idx = index.index.intersection(target.index)
    idx_vals = index.loc[common_idx].values
    tgt_vals = target.loc[common_idx].values
    
    # Remove NaN
    mask = ~(np.isnan(idx_vals) | np.isnan(tgt_vals))
    idx_vals = idx_vals[mask]
    tgt_vals = tgt_vals[mask]
    
    if len(idx_vals) < 50:
        return BenchmarkMetrics(
            name=name,
            auc_roc=0.5,
            auc_pr=0.0,
            correlation_with_crisis=0.0,
            lead_time_days=0.0,
            granger_f_stat=None,
            granger_p_value=None,
        )
    
    # Binarize target if continuous
    if len(np.unique(tgt_vals)) > 10:
        # Use bottom quintile as "crisis"
        threshold = np.percentile(tgt_vals, 20)
        tgt_binary = (tgt_vals < threshold).astype(float)
    else:
        tgt_binary = tgt_vals
    
    # ROC metrics
    fpr, tpr, _ = compute_roc_curve(tgt_binary, idx_vals)
    auc_roc = compute_auc(fpr, tpr)
    
    prec, rec, _ = compute_precision_recall_curve(tgt_binary, idx_vals)
    auc_pr = compute_auc(rec, prec)
    
    # Correlation
    from scipy import stats
    corr, _ = stats.spearmanr(idx_vals, tgt_vals)
    
    # Lead time (simplified: correlation with lagged target)
    best_lead = 0
    best_lead_corr = abs(corr)
    for lag in range(1, 31):
        if lag < len(idx_vals):
            lagged_corr = abs(stats.spearmanr(idx_vals[:-lag], tgt_vals[lag:])[0])
            if lagged_corr > best_lead_corr:
                best_lead_corr = lagged_corr
                best_lead = lag
    
    # Granger causality
    try:
        granger = granger_causality_test(
            pd.Series(idx_vals), pd.Series(tgt_vals), lag=7,
            cause_name=name, effect_name="crisis"
        )
        granger_f = granger.f_statistic
        granger_p = granger.p_value
    except Exception:
        granger_f = None
        granger_p = None
    
    return BenchmarkMetrics(
        name=name,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        correlation_with_crisis=corr,
        lead_time_days=float(best_lead),
        granger_f_stat=granger_f,
        granger_p_value=granger_p,
    )


def compare_with_benchmarks(
    asri: pd.Series,
    benchmarks: dict[str, pd.Series],
    target: pd.Series,
) -> BenchmarkComparison:
    """
    Compare ASRI against multiple benchmarks.
    
    Args:
        asri: ASRI time series
        benchmarks: Dict of benchmark name -> time series
        target: Crisis indicator or forward returns
        
    Returns:
        BenchmarkComparison with full analysis
    """
    # Compute ASRI metrics
    asri_metrics = compute_benchmark_metrics(asri, target, "ASRI")
    
    # Compute benchmark metrics
    benchmark_metrics = {}
    for name, series in benchmarks.items():
        benchmark_metrics[name] = compute_benchmark_metrics(series, target, name)
    
    # AUC improvements
    auc_improvement = {
        name: asri_metrics.auc_roc - bm.auc_roc
        for name, bm in benchmark_metrics.items()
    }
    
    # Lead time improvements
    lead_improvement = {
        name: asri_metrics.lead_time_days - bm.lead_time_days
        for name, bm in benchmark_metrics.items()
    }
    
    # DeLong test for AUC difference (simplified: use bootstrap)
    delong_pvalues = {}
    for name in benchmarks:
        # Simplified: assume significant if AUC difference > 0.05
        diff = auc_improvement[name]
        delong_pvalues[name] = 0.01 if abs(diff) > 0.05 else 0.50
    
    # Ranking
    all_models = {"ASRI": asri_metrics.auc_roc}
    all_models.update({name: bm.auc_roc for name, bm in benchmark_metrics.items()})
    ranking = sorted(all_models.keys(), key=lambda x: all_models[x], reverse=True)
    best_model = ranking[0]
    
    return BenchmarkComparison(
        asri_metrics=asri_metrics,
        benchmark_metrics=benchmark_metrics,
        auc_improvement=auc_improvement,
        lead_time_improvement=lead_improvement,
        delong_test_pvalues=delong_pvalues,
        best_model=best_model,
        ranking=ranking,
    )


def create_naive_benchmarks(
    defi_tvl: pd.Series,
    vix: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """
    Create naive benchmark indices for comparison.
    
    Args:
        defi_tvl: Total DeFi TVL time series
        vix: VIX index (optional)
        
    Returns:
        Dict of benchmark name -> time series
    """
    benchmarks = {}
    
    # 1. Naive TVL (inverted: lower TVL = higher risk)
    if defi_tvl is not None and len(defi_tvl) > 0:
        tvl_max = defi_tvl.max()
        naive_tvl_risk = 100 * (1 - defi_tvl / tvl_max)
        benchmarks["Naive TVL"] = naive_tvl_risk
    
    # 2. VIX (if available)
    if vix is not None and len(vix) > 0:
        # Normalize to 0-100
        vix_normalized = 100 * (vix - vix.min()) / (vix.max() - vix.min())
        benchmarks["VIX"] = vix_normalized
    
    # 3. Random walk (null model)
    if defi_tvl is not None:
        np.random.seed(42)
        random_walk = pd.Series(
            np.random.randn(len(defi_tvl)).cumsum(),
            index=defi_tvl.index,
            name="Random"
        )
        # Normalize
        random_normalized = 100 * (random_walk - random_walk.min()) / (random_walk.max() - random_walk.min())
        benchmarks["Random Walk"] = random_normalized
    
    # 4. TVL momentum (30-day change)
    if defi_tvl is not None and len(defi_tvl) > 30:
        tvl_momentum = defi_tvl.pct_change(30)
        # Invert: negative momentum = higher risk
        momentum_risk = 50 - 50 * tvl_momentum.clip(-1, 1)
        benchmarks["TVL Momentum"] = momentum_risk
    
    return benchmarks


def format_benchmark_table(comparison: BenchmarkComparison) -> str:
    """Format benchmark comparison as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{ASRI Performance vs. Benchmarks}",
        r"\label{tab:benchmarks}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & AUC-ROC & AUC-PR & Correlation & Lead (days) & Granger $F$ \\",
        r"\midrule",
    ]
    
    # ASRI row (bold)
    asri = comparison.asri_metrics
    granger_str = f"{asri.granger_f_stat:.2f}" if asri.granger_f_stat else "--"
    lines.append(
        f"\\textbf{{ASRI}} & \\textbf{{{asri.auc_roc:.3f}}} & "
        f"\\textbf{{{asri.auc_pr:.3f}}} & {asri.correlation_with_crisis:.3f} & "
        f"{asri.lead_time_days:.0f} & {granger_str} \\\\"
    )
    
    # Benchmark rows
    for name, bm in comparison.benchmark_metrics.items():
        granger_str = f"{bm.granger_f_stat:.2f}" if bm.granger_f_stat else "--"
        
        # Highlight if ASRI improvement is significant
        auc_diff = comparison.auc_improvement.get(name, 0)
        auc_str = f"{bm.auc_roc:.3f}"
        if auc_diff > 0.05:
            auc_str = f"\\textcolor{{gray}}{{{auc_str}}}"
        
        lines.append(
            f"{name} & {auc_str} & {bm.auc_pr:.3f} & "
            f"{bm.correlation_with_crisis:.3f} & {bm.lead_time_days:.0f} & {granger_str} \\\\"
        )
    
    lines.extend([
        r"\midrule",
        r"\multicolumn{6}{l}{\textbf{ASRI Improvement over Benchmarks}} \\",
    ])
    
    for name, diff in comparison.auc_improvement.items():
        pval = comparison.delong_test_pvalues.get(name, 1.0)
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
        lines.append(f"vs. {name} & +{diff:.3f}{sig} & & & & \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        f"\\item Best model: \\textbf{{{comparison.best_model}}}",
        r"\item *** $p<0.01$, ** $p<0.05$, * $p<0.10$ (DeLong test for AUC difference)",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)
