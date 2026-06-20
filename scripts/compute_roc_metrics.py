#!/usr/bin/env python3
"""
Compute AUROC/AUPRC with Bootstrap Confidence Intervals

Generates ROC metrics for ASRI crisis prediction with:
- Point estimates for AUROC and AUPRC
- 95% bootstrap confidence intervals (BCa method)
- Comparison against D-Y Connectedness benchmark
- LaTeX table output for paper

Usage:
    python scripts/compute_roc_metrics.py --output results/tables/roc_metrics.tex
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ROCMetricsWithCI:
    """ROC metrics with bootstrap confidence intervals."""
    auroc: float
    auroc_ci_lower: float
    auroc_ci_upper: float

    auprc: float
    auprc_ci_lower: float
    auprc_ci_upper: float

    optimal_threshold: float
    precision_at_optimal: float
    recall_at_optimal: float
    f1_at_optimal: float

    n_positive: int
    n_negative: int
    n_bootstrap: int


def compute_auc_trapezoidal(x: np.ndarray, y: np.ndarray) -> float:
    """Compute area under curve using trapezoidal rule."""
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    return np.trapezoid(y, x)


def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """Compute ROC curve points."""
    thresholds = np.unique(y_score)[::-1]

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    tpr_list = [0.0]
    fpr_list = [0.0]

    for thresh in thresholds:
        predicted_pos = y_score >= thresh
        tp = np.sum((predicted_pos) & (y_true == 1))
        fp = np.sum((predicted_pos) & (y_true == 0))

        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    tpr_list.append(1.0)
    fpr_list.append(1.0)

    return np.array(fpr_list), np.array(tpr_list), thresholds


def compute_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """Compute precision-recall curve points."""
    thresholds = np.unique(y_score)[::-1]
    n_pos = np.sum(y_true)

    precision_list = []
    recall_list = []

    for thresh in thresholds:
        predicted_pos = y_score >= thresh
        tp = np.sum((predicted_pos) & (y_true == 1))
        fp = np.sum((predicted_pos) & (y_true == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / n_pos if n_pos > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    return np.array(precision_list), np.array(recall_list), thresholds


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC."""
    fpr, tpr, _ = compute_roc_curve(y_true, y_score)
    return compute_auc_trapezoidal(fpr, tpr)


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUPRC (average precision)."""
    precision, recall, _ = compute_pr_curve(y_true, y_score)
    # Sort by recall for proper AUC calculation
    idx = np.argsort(recall)
    return compute_auc_trapezoidal(recall[idx], precision[idx])


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """Find optimal threshold using Youden's J statistic."""
    thresholds = np.unique(y_score)
    best_j = -np.inf
    best_thresh = thresholds[0]
    best_metrics = {}

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    for thresh in thresholds:
        predicted_pos = y_score >= thresh

        tp = np.sum((predicted_pos) & (y_true == 1))
        fp = np.sum((predicted_pos) & (y_true == 0))
        tn = np.sum((~predicted_pos) & (y_true == 0))
        fn = np.sum((~predicted_pos) & (y_true == 1))

        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        j = tpr - fpr  # Youden's J

        if j > best_j:
            best_j = j
            best_thresh = thresh
            recall = tpr
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }

    return best_thresh, best_metrics


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Uses BCa (bias-corrected and accelerated) method.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    # Point estimate
    point_estimate = metric_fn(y_true, y_score)

    # Bootstrap replicates
    boot_estimates = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_estimates[i] = metric_fn(y_true[idx], y_score[idx])

    # BCa correction
    # Bias correction factor
    z0 = stats_norm_ppf(np.mean(boot_estimates < point_estimate))

    # Acceleration factor (jackknife)
    jackknife = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jackknife[i] = metric_fn(y_true[mask], y_score[mask])

    jack_mean = np.mean(jackknife)
    num = np.sum((jack_mean - jackknife) ** 3)
    denom = 6 * (np.sum((jack_mean - jackknife) ** 2) ** 1.5)
    a = num / denom if denom != 0 else 0

    # Adjusted percentiles
    z_alpha = stats_norm_ppf(alpha / 2)
    z_1_alpha = stats_norm_ppf(1 - alpha / 2)

    alpha_1 = stats_norm_cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    alpha_2 = stats_norm_cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))

    # Fallback to percentile method if BCa fails
    if np.isnan(alpha_1) or np.isnan(alpha_2):
        alpha_1 = alpha / 2
        alpha_2 = 1 - alpha / 2

    ci_lower = np.percentile(boot_estimates, alpha_1 * 100)
    ci_upper = np.percentile(boot_estimates, alpha_2 * 100)

    return point_estimate, ci_lower, ci_upper


def stats_norm_ppf(p: float) -> float:
    """Inverse normal CDF (percent point function) - simple approximation."""
    # Approximation using Abramowitz & Stegun formula
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf

    p = min(max(p, 1e-10), 1 - 1e-10)

    if p < 0.5:
        t = np.sqrt(-2 * np.log(p))
    else:
        t = np.sqrt(-2 * np.log(1 - p))

    # Coefficients for approximation
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)

    return z if p >= 0.5 else -z


def stats_norm_cdf(z: float) -> float:
    """Normal CDF - simple approximation."""
    # Approximation using error function
    return 0.5 * (1 + np.tanh(z * 0.7978845608))  # sqrt(2/pi) approximation


def compute_roc_with_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> ROCMetricsWithCI:
    """
    Compute full ROC analysis with bootstrap confidence intervals.

    Args:
        y_true: Binary labels (0/1)
        y_score: Prediction scores
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level for CI (default 0.05 = 95% CI)
        seed: Random seed for reproducibility

    Returns:
        ROCMetricsWithCI with all metrics and confidence intervals
    """
    # AUROC with CI
    auroc, auroc_lower, auroc_upper = bootstrap_ci(
        y_true, y_score, compute_auroc, n_bootstrap, alpha, seed
    )

    # AUPRC with CI
    auprc, auprc_lower, auprc_upper = bootstrap_ci(
        y_true, y_score, compute_auprc, n_bootstrap, alpha, seed
    )

    # Optimal threshold metrics
    opt_thresh, opt_metrics = find_optimal_threshold(y_true, y_score)

    return ROCMetricsWithCI(
        auroc=auroc,
        auroc_ci_lower=auroc_lower,
        auroc_ci_upper=auroc_upper,
        auprc=auprc,
        auprc_ci_lower=auprc_lower,
        auprc_ci_upper=auprc_upper,
        optimal_threshold=opt_thresh,
        precision_at_optimal=opt_metrics['precision'],
        recall_at_optimal=opt_metrics['recall'],
        f1_at_optimal=opt_metrics['f1'],
        n_positive=int(np.sum(y_true)),
        n_negative=int(len(y_true) - np.sum(y_true)),
        n_bootstrap=n_bootstrap,
    )


def load_real_data(
    parquet_path: Path,
    window_start: str = "2021-01-01",
    window_end: str = "2024-12-31",
    roll_window: int = 60,
    var_lags: int = 1,
    fevd_horizon: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load REAL ASRI series and compute a REAL rolling Diebold-Yilmaz connectedness
    benchmark from the actual sub-indices, then build crisis labels from the four
    historical events. No synthetic data.

    Crisis labels: 1 if a historical crisis onset falls within the 30-day forward
    (pre-crisis) window of a given date -- identical to generate_roc_figure.py
    (the labeling that produces the reproducible ASRI ROC).

    D-Y benchmark: generalized (Pesaran-Shin) FEVD total connectedness from a
    `roll_window`-day rolling VAR(`var_lags`), forecast horizon `fevd_horizon`,
    computed in scripts/real_dy_hmm_analysis.py (ordering-invariant).

    Returns:
        (crisis_labels, asri_scores, dy_scores) aligned on a common date index.
    """
    from datetime import datetime

    # Reuse the real D-Y implementation rather than re-deriving it here.
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "real_dy_hmm_analysis", Path(__file__).parent / "real_dy_hmm_analysis.py"
    )
    _rd = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_rd)

    crisis_events = [
        datetime(2022, 5, 12),   # Terra/Luna
        datetime(2022, 6, 17),   # Celsius/3AC
        datetime(2022, 11, 11),  # FTX
        datetime(2023, 3, 11),   # SVB
    ]
    sub_cols = ["stablecoin_risk", "defi_liquidity_risk",
                "contagion_risk", "arbitrage_opacity"]

    df = pd.read_parquet(parquet_path).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)

    win = df.loc[window_start:window_end]

    # Real rolling D-Y connectedness (computed on full sample for warm-up, then sliced)
    roll = _rd.rolling_connectedness(
        df[sub_cols].dropna(), window=roll_window, lags=var_lags, horizon=fevd_horizon
    ).loc[window_start:window_end].dropna()

    asri_win = win["asri"].dropna()
    common = asri_win.index.intersection(roll.index)

    asri = asri_win.loc[common].values
    dy = roll.loc[common].values
    labels = _rd.create_crisis_labels(common, crisis_events, window_days=30)

    return labels, asri, dy


def format_latex_table(
    asri_metrics: ROCMetricsWithCI,
    benchmark_metrics: ROCMetricsWithCI | None = None,
    benchmark_name: str = "D-Y Connectedness",
) -> str:
    """Format ROC metrics as LaTeX table with confidence intervals."""

    def fmt_ci(val: float, lower: float, upper: float) -> str:
        return f"{val:.3f} [{lower:.3f}, {upper:.3f}]"

    if benchmark_metrics:
        # Comparison table
        auroc_diff = asri_metrics.auroc - benchmark_metrics.auroc
        auprc_diff = asri_metrics.auprc - benchmark_metrics.auprc

        lines = [
            r"\begin{table}[H]",
            r"\begin{threeparttable}",
            r"\centering",
            r"\caption{Crisis Prediction Classification Metrics with 95\% Bootstrap Confidence Intervals}",
            r"\label{tab:roc_metrics}",
            r"\small",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            f"Metric & ASRI & {benchmark_name} & Difference \\\\",
            r"\midrule",
            f"AUROC & {fmt_ci(asri_metrics.auroc, asri_metrics.auroc_ci_lower, asri_metrics.auroc_ci_upper)} & "
            f"{fmt_ci(benchmark_metrics.auroc, benchmark_metrics.auroc_ci_lower, benchmark_metrics.auroc_ci_upper)} & "
            f"{auroc_diff:+.3f} \\\\",
            f"AUPRC & {fmt_ci(asri_metrics.auprc, asri_metrics.auprc_ci_lower, asri_metrics.auprc_ci_upper)} & "
            f"{fmt_ci(benchmark_metrics.auprc, benchmark_metrics.auprc_ci_lower, benchmark_metrics.auprc_ci_upper)} & "
            f"{auprc_diff:+.3f} \\\\",
            r"\midrule",
            f"Optimal Threshold & {asri_metrics.optimal_threshold:.1f} & {benchmark_metrics.optimal_threshold:.2f} & --- \\\\",
            f"Precision @ Optimal & {asri_metrics.precision_at_optimal:.3f} & {benchmark_metrics.precision_at_optimal:.3f} & "
            f"{asri_metrics.precision_at_optimal - benchmark_metrics.precision_at_optimal:+.3f} \\\\",
            f"Recall @ Optimal & {asri_metrics.recall_at_optimal:.3f} & {benchmark_metrics.recall_at_optimal:.3f} & "
            f"{asri_metrics.recall_at_optimal - benchmark_metrics.recall_at_optimal:+.3f} \\\\",
            f"F1 @ Optimal & {asri_metrics.f1_at_optimal:.3f} & {benchmark_metrics.f1_at_optimal:.3f} & "
            f"{asri_metrics.f1_at_optimal - benchmark_metrics.f1_at_optimal:+.3f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            f"\\item $n = {asri_metrics.n_positive + asri_metrics.n_negative}$ observations; "
            f"{asri_metrics.n_positive} crisis-imminent days, {asri_metrics.n_negative} non-crisis days.",
            f"\\item Bootstrap confidence intervals computed with $B = {asri_metrics.n_bootstrap}$ resamples (BCa method).",
            r"\item Crisis defined as 30-day forward period preceding historical crisis onset.",
            r"\item Optimal threshold selected by Youden's J statistic (maximizes TPR $-$ FPR).",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    else:
        # Single model table
        lines = [
            r"\begin{table}[H]",
            r"\begin{threeparttable}",
            r"\centering",
            r"\caption{ASRI Crisis Prediction Performance with 95\% Bootstrap Confidence Intervals}",
            r"\label{tab:roc_metrics}",
            r"\small",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Metric & Value \\",
            r"\midrule",
            f"AUROC & {fmt_ci(asri_metrics.auroc, asri_metrics.auroc_ci_lower, asri_metrics.auroc_ci_upper)} \\\\",
            f"AUPRC & {fmt_ci(asri_metrics.auprc, asri_metrics.auprc_ci_lower, asri_metrics.auprc_ci_upper)} \\\\",
            r"\midrule",
            f"Optimal Threshold & {asri_metrics.optimal_threshold:.1f} \\\\",
            f"Precision @ Optimal & {asri_metrics.precision_at_optimal:.3f} \\\\",
            f"Recall @ Optimal & {asri_metrics.recall_at_optimal:.3f} \\\\",
            f"F1 @ Optimal & {asri_metrics.f1_at_optimal:.3f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            f"\\item $n = {asri_metrics.n_positive + asri_metrics.n_negative}$ observations.",
            f"\\item Bootstrap CI computed with $B = {asri_metrics.n_bootstrap}$ resamples.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute ROC metrics with bootstrap CIs")
    parser.add_argument(
        "--output",
        type=str,
        default="results/tables/roc_metrics.tex",
        help="Output path for LaTeX table",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (bootstrap resampling only)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).parent.parent / "results" / "data" / "asri_history.parquet"),
        help="Path to asri_history.parquet (REAL data)",
    )
    args = parser.parse_args()

    print("Computing ROC metrics with bootstrap confidence intervals...")

    # Load REAL ASRI series and compute a REAL rolling Diebold-Yilmaz benchmark.
    # (The previous synthetic generator was removed -- it fabricated both series.)
    labels, asri_scores, dy_scores = load_real_data(Path(args.data))

    print(f"  Data: {len(labels)} observations, {int(np.sum(labels))} crisis-imminent days")

    # Compute metrics for ASRI
    print("  Computing ASRI metrics...")
    asri_metrics = compute_roc_with_bootstrap_ci(
        labels, asri_scores, args.n_bootstrap, seed=args.seed
    )
    print(f"    AUROC: {asri_metrics.auroc:.3f} [{asri_metrics.auroc_ci_lower:.3f}, {asri_metrics.auroc_ci_upper:.3f}]")
    print(f"    AUPRC: {asri_metrics.auprc:.3f} [{asri_metrics.auprc_ci_lower:.3f}, {asri_metrics.auprc_ci_upper:.3f}]")

    # Compute metrics for benchmark
    print("  Computing D-Y Connectedness metrics...")
    dy_metrics = compute_roc_with_bootstrap_ci(
        labels, dy_scores, args.n_bootstrap, seed=args.seed
    )
    print(f"    AUROC: {dy_metrics.auroc:.3f} [{dy_metrics.auroc_ci_lower:.3f}, {dy_metrics.auroc_ci_upper:.3f}]")
    print(f"    AUPRC: {dy_metrics.auprc:.3f} [{dy_metrics.auprc_ci_lower:.3f}, {dy_metrics.auprc_ci_upper:.3f}]")

    # Generate LaTeX table
    latex_table = format_latex_table(asri_metrics, dy_metrics)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_table)

    print(f"\nLaTeX table written to: {output_path}")


if __name__ == "__main__":
    main()
