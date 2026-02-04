"""
Sensitivity Analysis for ASRI Systemic Risk Index

Comprehensive sensitivity testing to validate robustness of ASRI
to parameter choices. This addresses a common critique of composite
indices: that results are sensitive to arbitrary weight/threshold choices.

Key analyses:
1. Weight perturbation: Does ASRI performance degrade when weights shift?
2. Threshold sensitivity: Are alert thresholds well-calibrated?
3. Window sensitivity: Is forward horizon choice robust?

These tests either validate the theoretical framework or expose it
as dependent on specific parameter tuning.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class WeightSensitivityResult:
    """
    Results from weight perturbation analysis for a single component.

    Attributes:
        component: Name of the sub-index being perturbed
        base_weight: Original weight for this component
        perturbation_grid: Array of perturbation deltas tested
        asri_means: Mean ASRI value at each perturbation level
        asri_stds: Standard deviation of ASRI at each perturbation level
        event_detection_rate: Fraction of crises detected at each level
        is_robust: True if performance remains stable across perturbations
    """
    component: str
    base_weight: float
    perturbation_grid: np.ndarray  # [-15%, -10%, -5%, 0%, +5%, +10%, +15%]
    asri_means: np.ndarray
    asri_stds: np.ndarray
    event_detection_rate: np.ndarray  # % of crises still detected
    is_robust: bool

    # Additional diagnostics
    performance_range: float = 0.0  # Max - min detection rate
    optimal_delta: float = 0.0  # Best perturbation found
    optimal_detection: float = 0.0  # Detection rate at optimal


@dataclass
class ThresholdSensitivityResult:
    """
    Results from alert threshold sensitivity analysis.

    Attributes:
        threshold_levels: List of ASRI thresholds tested
        precision_at_threshold: Precision for each threshold
        recall_at_threshold: Recall for each threshold
        f1_at_threshold: F1 score for each threshold
        optimal_threshold: Threshold that maximizes F1
    """
    threshold_levels: list[int]
    precision_at_threshold: dict[int, float]
    recall_at_threshold: dict[int, float]
    f1_at_threshold: dict[int, float]
    optimal_threshold: int

    # Additional metrics
    auc_at_threshold: dict[int, float] = field(default_factory=dict)
    specificity_at_threshold: dict[int, float] = field(default_factory=dict)


@dataclass
class WindowSensitivityResult:
    """
    Results from forward window length sensitivity analysis.

    Attributes:
        windows: List of window lengths tested (in days)
        auc_roc_by_window: AUC-ROC for each window length
        lead_time_by_window: Average lead time for each window
        optimal_window: Window length that maximizes AUC-ROC
    """
    windows: list[int]
    auc_roc_by_window: dict[int, float]
    lead_time_by_window: dict[int, float]
    optimal_window: int

    # Additional metrics
    f1_by_window: dict[int, float] = field(default_factory=dict)
    precision_by_window: dict[int, float] = field(default_factory=dict)
    recall_by_window: dict[int, float] = field(default_factory=dict)


# =============================================================================
# Helper Functions
# =============================================================================

def _compute_asri_with_weights(
    sub_indices: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Compute ASRI given sub-indices and weights.

    Args:
        sub_indices: DataFrame with sub-index time series
        weights: Dictionary mapping component names to weights

    Returns:
        ASRI time series
    """
    asri = pd.Series(0.0, index=sub_indices.index)

    for component, weight in weights.items():
        if component in sub_indices.columns:
            asri += weight * sub_indices[component]

    return asri


def _detect_crisis_events(
    asri: pd.Series,
    crisis_dates: list[datetime],
    threshold: float = 70.0,
    window_days: int = 30,
) -> tuple[int, int]:
    """
    Count crisis events detected by ASRI exceeding threshold.

    Args:
        asri: ASRI time series
        crisis_dates: Known crisis dates
        threshold: ASRI level considered "alert"
        window_days: Days before crisis that count as detection

    Returns:
        (detected_count, total_count)
    """
    detected = 0

    for crisis_date in crisis_dates:
        crisis_ts = pd.Timestamp(crisis_date)
        window_start = crisis_ts - pd.Timedelta(days=window_days)

        # Check if ASRI exceeded threshold before crisis
        window_asri = asri[(asri.index >= window_start) & (asri.index < crisis_ts)]

        if len(window_asri) > 0 and window_asri.max() >= threshold:
            detected += 1

    return detected, len(crisis_dates)


def _compute_forward_drawdown(
    returns: pd.Series,
    forward_window: int,
) -> pd.Series:
    """
    Compute maximum forward drawdown for each date.

    Args:
        returns: Daily returns series
        forward_window: Days to look ahead

    Returns:
        Series of forward drawdowns (negative values)
    """
    n = len(returns)
    drawdowns = np.zeros(n)

    for i in range(n - forward_window):
        window = returns.iloc[i:i + forward_window]
        cumulative = (1 + window).cumprod()
        peak = cumulative.cummax()
        dd = ((cumulative - peak) / peak).min()
        drawdowns[i] = dd

    # Fill end with NaN
    drawdowns[n - forward_window:] = np.nan

    return pd.Series(drawdowns, index=returns.index, name='forward_drawdown')


def _compute_crisis_labels(
    returns: pd.Series,
    forward_window: int = 30,
    drawdown_threshold: float = -0.20,
) -> pd.Series:
    """
    Create binary crisis labels from forward drawdowns.

    Args:
        returns: Daily returns series
        forward_window: Days to look ahead
        drawdown_threshold: Drawdown level defining crisis

    Returns:
        Binary series (1 = crisis ahead, 0 = no crisis)
    """
    drawdowns = _compute_forward_drawdown(returns, forward_window)
    labels = (drawdowns < drawdown_threshold).astype(int)
    return labels


def _compute_auc_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """
    Compute AUC-ROC without sklearn dependency.

    Args:
        y_true: Binary ground truth
        y_score: Prediction scores

    Returns:
        AUC-ROC value
    """
    # Remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5

    # Sort by score descending
    desc_idx = np.argsort(y_score)[::-1]
    y_score = y_score[desc_idx]
    y_true = y_true[desc_idx]

    # Compute ROC curve
    thresholds = np.unique(y_score)[::-1]
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr_list = [0.0]
    fpr_list = [0.0]

    for thresh in thresholds:
        predicted_pos = y_score >= thresh
        tp = np.sum(predicted_pos & (y_true == 1))
        fp = np.sum(predicted_pos & (y_true == 0))
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    tpr_list.append(1.0)
    fpr_list.append(1.0)

    # AUC via trapezoidal rule
    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    idx = np.argsort(fpr)

    return float(np.trapezoid(tpr[idx], fpr[idx]))


def _compute_lead_time(
    asri: pd.Series,
    crisis_date: datetime,
    threshold: float = 70.0,
    max_lookback: int = 60,
) -> int:
    """
    Compute how many days before crisis ASRI first exceeded threshold.

    Args:
        asri: ASRI time series
        crisis_date: Date of crisis
        threshold: Alert threshold
        max_lookback: Maximum days to look back

    Returns:
        Lead time in days (0 if no early warning)
    """
    crisis_ts = pd.Timestamp(crisis_date)
    lookback_start = crisis_ts - pd.Timedelta(days=max_lookback)

    window = asri[(asri.index >= lookback_start) & (asri.index < crisis_ts)]

    # Find first day exceeding threshold (going backwards)
    alert_days = window[window >= threshold]

    if len(alert_days) == 0:
        return 0

    first_alert = alert_days.index.min()
    lead_time = (crisis_ts - first_alert).days

    return lead_time


# =============================================================================
# Main Analysis Functions
# =============================================================================

def run_weight_perturbation_analysis(
    sub_indices: pd.DataFrame,
    weights: dict[str, float],
    crisis_dates: list[datetime],
    perturbation_range: tuple[float, float] = (-0.15, 0.15),
    n_steps: int = 7,
    detection_threshold: float = 70.0,
    detection_window: int = 30,
) -> list[WeightSensitivityResult]:
    """
    Grid search weight perturbations for each component.

    For each sub-index, perturb its weight across a range while
    re-normalizing other weights. Measure impact on crisis detection.

    This tests whether ASRI's performance is robust to weight choices
    or critically dependent on specific values.

    Args:
        sub_indices: DataFrame with sub-index time series
        weights: Base weights for each component
        crisis_dates: List of known crisis dates
        perturbation_range: (min, max) perturbation as fraction
        n_steps: Number of perturbation levels to test
        detection_threshold: ASRI level for crisis alert
        detection_window: Days before crisis for detection

    Returns:
        List of WeightSensitivityResult, one per component
    """
    results = []
    perturbation_grid = np.linspace(
        perturbation_range[0],
        perturbation_range[1],
        n_steps
    )

    components = list(weights.keys())

    for component in components:
        if component not in sub_indices.columns:
            continue

        base_weight = weights[component]
        asri_means = []
        asri_stds = []
        detection_rates = []

        for delta in perturbation_grid:
            # Create perturbed weights
            perturbed_weights = weights.copy()
            new_weight = base_weight + delta

            # Clamp to [0, 1]
            new_weight = max(0.01, min(0.99, new_weight))
            perturbed_weights[component] = new_weight

            # Re-normalize to sum to 1
            total = sum(perturbed_weights.values())
            perturbed_weights = {k: v / total for k, v in perturbed_weights.items()}

            # Compute ASRI with perturbed weights
            asri = _compute_asri_with_weights(sub_indices, perturbed_weights)

            # Statistics
            asri_means.append(asri.mean())
            asri_stds.append(asri.std())

            # Crisis detection
            detected, total_crises = _detect_crisis_events(
                asri, crisis_dates, detection_threshold, detection_window
            )
            detection_rate = detected / total_crises if total_crises > 0 else 0
            detection_rates.append(detection_rate)

        asri_means = np.array(asri_means)
        asri_stds = np.array(asri_stds)
        detection_rates = np.array(detection_rates)

        # Assess robustness
        performance_range = detection_rates.max() - detection_rates.min()
        is_robust = performance_range < 0.15  # <15% variation is robust

        # Find optimal perturbation
        optimal_idx = np.argmax(detection_rates)
        optimal_delta = perturbation_grid[optimal_idx]
        optimal_detection = detection_rates[optimal_idx]

        results.append(WeightSensitivityResult(
            component=component,
            base_weight=base_weight,
            perturbation_grid=perturbation_grid,
            asri_means=asri_means,
            asri_stds=asri_stds,
            event_detection_rate=detection_rates,
            is_robust=is_robust,
            performance_range=performance_range,
            optimal_delta=optimal_delta,
            optimal_detection=optimal_detection,
        ))

    return results


def run_threshold_sensitivity(
    asri: pd.Series,
    crisis_dates: list[datetime],
    thresholds: list[int] = [60, 65, 70, 75, 80],
    window_days: int = 30,
) -> ThresholdSensitivityResult:
    """
    Test different alert threshold levels.

    Higher thresholds reduce false positives but may miss crises.
    Lower thresholds catch more crises but generate more noise.
    Find the optimal trade-off.

    Args:
        asri: ASRI time series
        crisis_dates: List of known crisis dates
        thresholds: ASRI threshold levels to test
        window_days: Days before crisis for detection

    Returns:
        ThresholdSensitivityResult with metrics at each level
    """
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    specificity_dict = {}

    for threshold in thresholds:
        # Count alerts
        alerts = asri[asri > threshold]
        n_alerts = len(alerts)

        # True positives: alerts that preceded a crisis
        true_positives = 0
        for alert_date in alerts.index:
            for crisis_date in crisis_dates:
                crisis_ts = pd.Timestamp(crisis_date)
                if 0 < (crisis_ts - alert_date).days <= window_days:
                    true_positives += 1
                    break

        # False positives: alerts that didn't precede crisis
        false_positives = n_alerts - true_positives

        # Detected crises: crises with preceding alert
        detected_crises = 0
        for crisis_date in crisis_dates:
            crisis_ts = pd.Timestamp(crisis_date)
            window_start = crisis_ts - pd.Timedelta(days=window_days)
            window_alerts = alerts[(alerts.index >= window_start) &
                                   (alerts.index < crisis_ts)]
            if len(window_alerts) > 0:
                detected_crises += 1

        # Metrics
        precision = true_positives / n_alerts if n_alerts > 0 else 0
        recall = detected_crises / len(crisis_dates) if len(crisis_dates) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Specificity: of non-alert days, what fraction were actually safe?
        non_alert_days = asri[asri <= threshold]
        n_non_alerts = len(non_alert_days)

        # True negatives: non-alert days that weren't before a crisis
        true_negatives = 0
        for date in non_alert_days.index:
            is_before_crisis = False
            for crisis_date in crisis_dates:
                crisis_ts = pd.Timestamp(crisis_date)
                if 0 < (crisis_ts - date).days <= window_days:
                    is_before_crisis = True
                    break
            if not is_before_crisis:
                true_negatives += 1

        specificity = true_negatives / n_non_alerts if n_non_alerts > 0 else 0

        precision_dict[threshold] = precision
        recall_dict[threshold] = recall
        f1_dict[threshold] = f1
        specificity_dict[threshold] = specificity

    # Find optimal threshold (maximize F1)
    optimal_threshold = max(thresholds, key=lambda t: f1_dict[t])

    return ThresholdSensitivityResult(
        threshold_levels=thresholds,
        precision_at_threshold=precision_dict,
        recall_at_threshold=recall_dict,
        f1_at_threshold=f1_dict,
        optimal_threshold=optimal_threshold,
        specificity_at_threshold=specificity_dict,
    )


def run_window_sensitivity(
    asri: pd.Series,
    returns: pd.Series,
    windows: list[int] = [14, 30, 60, 90],
    drawdown_threshold: float = -0.20,
) -> WindowSensitivityResult:
    """
    Test different forward window lengths.

    Shorter windows test near-term prediction; longer windows
    test if ASRI provides early warning. Optimal window depends
    on use case (trading vs. risk management).

    Args:
        asri: ASRI time series
        returns: Market returns series
        windows: Forward window lengths to test (days)
        drawdown_threshold: What constitutes a "crisis"

    Returns:
        WindowSensitivityResult with AUC and lead time by window
    """
    auc_dict = {}
    lead_time_dict = {}
    f1_dict = {}
    precision_dict = {}
    recall_dict = {}

    # Align ASRI and returns
    common_idx = asri.index.intersection(returns.index)
    asri_aligned = asri.loc[common_idx]
    returns_aligned = returns.loc[common_idx]

    for window in windows:
        # Create crisis labels for this window
        labels = _compute_crisis_labels(
            returns_aligned,
            forward_window=window,
            drawdown_threshold=drawdown_threshold
        )

        # Align with ASRI
        valid_idx = labels.dropna().index
        y_true = labels.loc[valid_idx].values
        y_score = asri_aligned.loc[valid_idx].values

        # AUC-ROC
        auc = _compute_auc_roc(y_true, y_score)
        auc_dict[window] = auc

        # Find crisis dates from labels
        crisis_starts = []
        in_crisis = False
        for i, (date, label) in enumerate(labels.items()):
            if label == 1 and not in_crisis:
                crisis_starts.append(date)
                in_crisis = True
            elif label == 0:
                in_crisis = False

        # Average lead time
        if len(crisis_starts) > 0:
            lead_times = []
            for crisis_date in crisis_starts:
                lt = _compute_lead_time(asri_aligned, crisis_date,
                                        threshold=70.0, max_lookback=window)
                lead_times.append(lt)
            lead_time_dict[window] = np.mean(lead_times)
        else:
            lead_time_dict[window] = 0

        # Precision/Recall/F1 at default threshold
        threshold = 70.0
        predicted_pos = (y_score >= threshold)

        tp = np.sum(predicted_pos & (y_true == 1))
        fp = np.sum(predicted_pos & (y_true == 0))
        fn = np.sum((~predicted_pos) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_dict[window] = precision
        recall_dict[window] = recall
        f1_dict[window] = f1

    # Optimal window maximizes AUC-ROC
    optimal_window = max(windows, key=lambda w: auc_dict[w])

    return WindowSensitivityResult(
        windows=windows,
        auc_roc_by_window=auc_dict,
        lead_time_by_window=lead_time_dict,
        optimal_window=optimal_window,
        f1_by_window=f1_dict,
        precision_by_window=precision_dict,
        recall_by_window=recall_dict,
    )


# =============================================================================
# LaTeX Output Formatting
# =============================================================================

def format_sensitivity_table(results: list[WeightSensitivityResult]) -> str:
    """
    Generate LaTeX table for weight sensitivity analysis.

    Args:
        results: List of WeightSensitivityResult from perturbation analysis

    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Weight Sensitivity Analysis: ASRI Component Perturbations}",
        r"\label{tab:weight_sensitivity}",
        r"\small",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Component & Base $w$ & $-15\%$ & $-5\%$ & Base & $+5\%$ & $+15\%$ & Robust \\",
        r"\midrule",
    ]

    for r in results:
        # Get detection rates at key perturbation levels
        grid = r.perturbation_grid
        rates = r.event_detection_rate

        # Find indices for specific perturbation levels
        def get_rate(delta: float) -> str:
            idx = np.argmin(np.abs(grid - delta))
            return f"{rates[idx]:.2f}"

        robust_str = r"\checkmark" if r.is_robust else r"$\times$"

        lines.append(
            f"{r.component.replace('_', ' ').title()} & "
            f"{r.base_weight:.2f} & "
            f"{get_rate(-0.15)} & "
            f"{get_rate(-0.05)} & "
            f"{get_rate(0.0)} & "
            f"{get_rate(0.05)} & "
            f"{get_rate(0.15)} & "
            f"{robust_str} \\\\"
        )

    # Summary statistics
    n_robust = sum(1 for r in results if r.is_robust)
    avg_range = np.mean([r.performance_range for r in results])

    lines.extend([
        r"\midrule",
        f"\\multicolumn{{8}}{{l}}{{Robust components: {n_robust}/{len(results)}}} \\\\",
        f"\\multicolumn{{8}}{{l}}{{Average detection rate variation: {avg_range:.3f}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Values show crisis detection rate at each perturbation level.",
        r"\item Robust = detection rate varies $<15\%$ across perturbations.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_threshold_table(result: ThresholdSensitivityResult) -> str:
    """
    Generate LaTeX table for threshold sensitivity analysis.

    Args:
        result: ThresholdSensitivityResult

    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Alert Threshold Sensitivity Analysis}",
        r"\label{tab:threshold_sensitivity}",
        r"\small",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Threshold & Precision & Recall & F1 Score & Specificity \\",
        r"\midrule",
    ]

    for thresh in result.threshold_levels:
        prec = result.precision_at_threshold[thresh]
        rec = result.recall_at_threshold[thresh]
        f1 = result.f1_at_threshold[thresh]
        spec = result.specificity_at_threshold.get(thresh, 0)

        # Mark optimal with asterisk
        opt_mark = "*" if thresh == result.optimal_threshold else ""

        lines.append(
            f"{thresh}{opt_mark} & {prec:.3f} & {rec:.3f} & {f1:.3f} & {spec:.3f} \\\\"
        )

    lines.extend([
        r"\midrule",
        f"\\multicolumn{{5}}{{l}}{{Optimal threshold: {result.optimal_threshold} "
        f"(F1 = {result.f1_at_threshold[result.optimal_threshold]:.3f})}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item * indicates optimal threshold maximizing F1 score.",
        r"\item Window: 30 days before crisis for detection.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_window_table(result: WindowSensitivityResult) -> str:
    """
    Generate LaTeX table for window sensitivity analysis.

    Args:
        result: WindowSensitivityResult

    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Forward Window Sensitivity Analysis}",
        r"\label{tab:window_sensitivity}",
        r"\small",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"Window (days) & AUC-ROC & Lead Time & Precision & Recall & F1 \\",
        r"\midrule",
    ]

    for window in result.windows:
        auc = result.auc_roc_by_window[window]
        lead = result.lead_time_by_window[window]
        prec = result.precision_by_window.get(window, 0)
        rec = result.recall_by_window.get(window, 0)
        f1 = result.f1_by_window.get(window, 0)

        opt_mark = "*" if window == result.optimal_window else ""

        lines.append(
            f"{window}{opt_mark} & {auc:.3f} & {lead:.1f} & {prec:.3f} & {rec:.3f} & {f1:.3f} \\\\"
        )

    lines.extend([
        r"\midrule",
        f"\\multicolumn{{6}}{{l}}{{Optimal window: {result.optimal_window} days "
        f"(AUC = {result.auc_roc_by_window[result.optimal_window]:.3f})}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item * indicates optimal window maximizing AUC-ROC.",
        r"\item Lead time = average days before crisis ASRI exceeded threshold.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_combined_sensitivity_table(
    weight_results: list[WeightSensitivityResult],
    threshold_result: ThresholdSensitivityResult,
    window_result: WindowSensitivityResult,
) -> str:
    """
    Generate combined LaTeX table summarizing all sensitivity analyses.

    Args:
        weight_results: Weight perturbation results
        threshold_result: Threshold sensitivity results
        window_result: Window sensitivity results

    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{ASRI Sensitivity Analysis Summary}",
        r"\label{tab:sensitivity_summary}",
        r"\small",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Analysis & Parameter & Range Tested & Conclusion \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textbf{Weight Perturbations}} \\",
    ]

    for r in weight_results:
        status = "Robust" if r.is_robust else f"Sensitive ($\\Delta$={r.performance_range:.2f})"
        lines.append(
            f"  {r.component.replace('_', ' ').title()} & "
            f"$w \\pm 15\\%$ & "
            f"[{r.base_weight - 0.15:.2f}, {r.base_weight + 0.15:.2f}] & "
            f"{status} \\\\"
        )

    lines.extend([
        r"\midrule",
        r"\multicolumn{4}{l}{\textbf{Alert Threshold}} \\",
        f"  Optimal threshold & ASRI level & "
        f"[{min(threshold_result.threshold_levels)}, {max(threshold_result.threshold_levels)}] & "
        f"{threshold_result.optimal_threshold} (F1={threshold_result.f1_at_threshold[threshold_result.optimal_threshold]:.2f}) \\\\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textbf{Forward Window}} \\",
        f"  Optimal window & Days & "
        f"[{min(window_result.windows)}, {max(window_result.windows)}] & "
        f"{window_result.optimal_window}d (AUC={window_result.auc_roc_by_window[window_result.optimal_window]:.2f}) \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Robust: performance varies $<15\%$ across parameter range.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# =============================================================================
# Figure Generation Code
# =============================================================================

def plot_weight_sensitivity_heatmap(
    results: list[WeightSensitivityResult],
    output_path: str = "figures/weight_sensitivity.pdf",
) -> str:
    """
    Generate matplotlib code for weight sensitivity heatmap.

    Args:
        results: Weight sensitivity results
        output_path: Output file path

    Returns:
        Python code string for generating the figure
    """
    return f'''
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Build data matrix
components = [r.component for r in results]
perturbations = results[0].perturbation_grid

data = np.array([r.event_detection_rate for r in results])

fig, ax = plt.subplots(figsize=(10, 6))

# Format perturbation labels as percentages
pct_labels = [f"{{p*100:+.0f}}%" for p in perturbations]

sns.heatmap(
    data,
    xticklabels=pct_labels,
    yticklabels=[c.replace('_', ' ').title() for c in components],
    cmap='RdYlGn',
    center=data.mean(),
    annot=True,
    fmt='.2f',
    ax=ax,
    cbar_kws={{'label': 'Crisis Detection Rate'}}
)

ax.set_xlabel('Weight Perturbation')
ax.set_ylabel('Component')
ax.set_title('ASRI Weight Sensitivity: Crisis Detection Rate')

plt.tight_layout()
plt.savefig('{output_path}', dpi=300, bbox_inches='tight')
plt.close()
'''


def plot_threshold_sensitivity(
    result: ThresholdSensitivityResult,
    output_path: str = "figures/threshold_sensitivity.pdf",
) -> str:
    """
    Generate matplotlib code for threshold sensitivity plot.

    Args:
        result: Threshold sensitivity result
        output_path: Output file path

    Returns:
        Python code string for generating the figure
    """
    return f'''
import matplotlib.pyplot as plt
import numpy as np

thresholds = {result.threshold_levels}
precision = [result.precision_at_threshold[t] for t in thresholds]
recall = [result.recall_at_threshold[t] for t in thresholds]
f1 = [result.f1_at_threshold[t] for t in thresholds]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(thresholds, precision, 'b-o', label='Precision', linewidth=2)
ax.plot(thresholds, recall, 'g-s', label='Recall', linewidth=2)
ax.plot(thresholds, f1, 'r-^', label='F1 Score', linewidth=2)

# Mark optimal threshold
opt_thresh = {result.optimal_threshold}
opt_f1 = result.f1_at_threshold[opt_thresh]
ax.axvline(opt_thresh, color='gray', linestyle='--', alpha=0.7,
           label=f'Optimal ({{opt_thresh}})')
ax.scatter([opt_thresh], [opt_f1], color='red', s=100, zorder=5, marker='*')

ax.set_xlabel('ASRI Alert Threshold')
ax.set_ylabel('Score')
ax.set_title('Alert Threshold Sensitivity')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('{output_path}', dpi=300, bbox_inches='tight')
plt.close()
'''


def plot_window_sensitivity(
    result: WindowSensitivityResult,
    output_path: str = "figures/window_sensitivity.pdf",
) -> str:
    """
    Generate matplotlib code for window sensitivity plot.

    Args:
        result: Window sensitivity result
        output_path: Output file path

    Returns:
        Python code string for generating the figure
    """
    return f'''
import matplotlib.pyplot as plt
import numpy as np

windows = {result.windows}
auc = [result.auc_roc_by_window[w] for w in windows]
lead_time = [result.lead_time_by_window[w] for w in windows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# AUC-ROC by window
ax1.bar(windows, auc, color='steelblue', alpha=0.8)
ax1.axhline(0.5, color='gray', linestyle='--', label='Random (0.5)')
ax1.set_xlabel('Forward Window (days)')
ax1.set_ylabel('AUC-ROC')
ax1.set_title('Predictive Power by Horizon')
ax1.legend()
ax1.set_ylim(0, 1)

# Mark optimal
opt_window = {result.optimal_window}
opt_idx = windows.index(opt_window)
ax1.bar(opt_window, auc[opt_idx], color='darkorange', alpha=0.8,
        label=f'Optimal ({{opt_window}}d)')

# Lead time by window
ax2.bar(windows, lead_time, color='seagreen', alpha=0.8)
ax2.set_xlabel('Forward Window (days)')
ax2.set_ylabel('Average Lead Time (days)')
ax2.set_title('Early Warning Lead Time')

plt.tight_layout()
plt.savefig('{output_path}', dpi=300, bbox_inches='tight')
plt.close()
'''


# =============================================================================
# Convenience Runner
# =============================================================================

def run_full_sensitivity_analysis(
    sub_indices: pd.DataFrame,
    weights: dict[str, float],
    returns: pd.Series,
    crisis_dates: list[datetime],
    asri: pd.Series | None = None,
) -> tuple[
    list[WeightSensitivityResult],
    ThresholdSensitivityResult,
    WindowSensitivityResult,
]:
    """
    Run all sensitivity analyses and return results.

    Convenience function that runs weight, threshold, and window
    sensitivity analyses in sequence.

    Args:
        sub_indices: DataFrame with sub-index time series
        weights: Base weights for ASRI computation
        returns: Market returns for crisis labeling
        crisis_dates: Known crisis dates
        asri: Pre-computed ASRI (optional, will compute if None)

    Returns:
        Tuple of (weight_results, threshold_result, window_result)
    """
    # Compute ASRI if not provided
    if asri is None:
        asri = _compute_asri_with_weights(sub_indices, weights)

    # Weight perturbation analysis
    weight_results = run_weight_perturbation_analysis(
        sub_indices=sub_indices,
        weights=weights,
        crisis_dates=crisis_dates,
    )

    # Threshold sensitivity
    threshold_result = run_threshold_sensitivity(
        asri=asri,
        crisis_dates=crisis_dates,
    )

    # Window sensitivity
    window_result = run_window_sensitivity(
        asri=asri,
        returns=returns,
    )

    return weight_results, threshold_result, window_result
