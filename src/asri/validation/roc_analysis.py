"""
ROC Analysis for ASRI Crisis Prediction

Treat ASRI as a binary classifier:
- Positive class: Crisis occurs in next N days
- Negative class: No crisis

This allows standard classification metrics (AUC-ROC, precision, recall)
which are interpretable and comparable across different risk indices.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CrisisClassificationMetrics:
    """Classification metrics for ASRI crisis prediction."""
    # Core metrics
    auc_roc: float
    auc_pr: float  # Precision-recall AUC
    
    # At optimal threshold
    optimal_threshold: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    
    # Confusion matrix
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Additional metrics
    balanced_accuracy: float
    matthews_corrcoef: float
    
    # Curves for plotting
    fpr_curve: np.ndarray
    tpr_curve: np.ndarray
    precision_curve: np.ndarray
    recall_curve: np.ndarray
    thresholds: np.ndarray


def compute_forward_crisis_labels(
    returns: pd.Series,
    forward_window: int = 30,
    drawdown_threshold: float = -0.20,
) -> pd.Series:
    """
    Create binary crisis labels from forward returns.
    
    A "crisis" is defined as a drawdown exceeding the threshold
    within the forward window.
    
    Args:
        returns: Daily returns series
        forward_window: Days to look ahead
        drawdown_threshold: Drawdown level that defines crisis (e.g., -0.20)
        
    Returns:
        Binary series (1 = crisis ahead, 0 = no crisis)
    """
    n = len(returns)
    labels = np.zeros(n)
    
    for i in range(n - forward_window):
        # Compute forward returns
        window = returns.iloc[i:i + forward_window]
        
        # Compute drawdown
        cumulative = (1 + window).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        
        # Check if crisis occurs
        if drawdown.min() < drawdown_threshold:
            labels[i] = 1
    
    return pd.Series(labels, index=returns.index, name='crisis_label')


def compute_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.
    
    Args:
        y_true: Binary ground truth (0 or 1)
        y_score: Prediction scores (higher = more positive)
        
    Returns:
        (fpr, tpr, thresholds)
    """
    # Sort by score descending
    desc_idx = np.argsort(y_score)[::-1]
    y_score = y_score[desc_idx]
    y_true = y_true[desc_idx]
    
    # Get unique thresholds
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


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute area under curve using trapezoidal rule."""
    # Sort by fpr
    idx = np.argsort(fpr)
    fpr = fpr[idx]
    tpr = tpr[idx]
    
    return np.trapezoid(tpr, fpr)


def compute_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve.
    
    Args:
        y_true: Binary ground truth
        y_score: Prediction scores
        
    Returns:
        (precision, recall, thresholds)
    """
    desc_idx = np.argsort(y_score)[::-1]
    y_score = y_score[desc_idx]
    y_true = y_true[desc_idx]
    
    thresholds = np.unique(y_score)[::-1]
    
    precision_list = []
    recall_list = []
    
    n_pos = np.sum(y_true)
    
    for thresh in thresholds:
        predicted_pos = y_score >= thresh
        
        tp = np.sum((predicted_pos) & (y_true == 1))
        fp = np.sum((predicted_pos) & (y_true == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / n_pos if n_pos > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
    
    return np.array(precision_list), np.array(recall_list), thresholds


def optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    criterion: Literal["f1", "youden", "balanced"] = "f1",
) -> tuple[float, dict]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: Binary ground truth
        y_score: Prediction scores
        criterion: Optimization criterion
            - 'f1': Maximize F1 score
            - 'youden': Maximize TPR - FPR (Youden's J)
            - 'balanced': Maximize balanced accuracy
            
    Returns:
        (optimal_threshold, metrics_at_threshold)
    """
    thresholds = np.unique(y_score)
    
    best_thresh = thresholds[0]
    best_score = -np.inf
    best_metrics = {}
    
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    
    for thresh in thresholds:
        predicted_pos = y_score >= thresh
        
        tp = np.sum((predicted_pos) & (y_true == 1))
        fp = np.sum((predicted_pos) & (y_true == 0))
        tn = np.sum((~predicted_pos) & (y_true == 0))
        fn = np.sum((~predicted_pos) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / n_pos if n_pos > 0 else 0
        specificity = tn / n_neg if n_neg > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        youden_j = recall + specificity - 1
        balanced_acc = (recall + specificity) / 2
        
        if criterion == "f1":
            score = f1
        elif criterion == "youden":
            score = youden_j
        elif criterion == "balanced":
            score = balanced_acc
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            }
    
    return best_thresh, best_metrics


def compute_roc_metrics(
    asri: pd.Series,
    returns: pd.Series,
    forward_window: int = 30,
    drawdown_threshold: float = -0.20,
) -> CrisisClassificationMetrics:
    """
    Compute full ROC analysis for ASRI crisis prediction.
    
    Args:
        asri: ASRI time series
        returns: Market returns (for computing crisis labels)
        forward_window: Days ahead for crisis definition
        drawdown_threshold: What counts as a "crisis"
        
    Returns:
        CrisisClassificationMetrics with all metrics
    """
    # Create crisis labels
    labels = compute_forward_crisis_labels(
        returns, forward_window, drawdown_threshold
    )
    
    # Align ASRI and labels
    common_idx = asri.index.intersection(labels.index)
    y_score = asri.loc[common_idx].values
    y_true = labels.loc[common_idx].values
    
    # Remove NaN
    mask = ~(np.isnan(y_score) | np.isnan(y_true))
    y_score = y_score[mask]
    y_true = y_true[mask]
    
    # Compute ROC curve
    fpr, tpr, roc_thresholds = compute_roc_curve(y_true, y_score)
    auc_roc = compute_auc(fpr, tpr)
    
    # Compute precision-recall curve
    precision, recall, pr_thresholds = compute_precision_recall_curve(y_true, y_score)
    auc_pr = compute_auc(recall, precision)
    
    # Find optimal threshold
    opt_thresh, metrics = optimal_threshold(y_true, y_score, criterion="f1")
    
    # Matthews correlation coefficient
    tp, fp, tn, fn = metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']
    mcc_num = tp * tn - fp * fn
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_denom if mcc_denom > 0 else 0
    
    return CrisisClassificationMetrics(
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        optimal_threshold=opt_thresh,
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1_score=metrics['f1'],
        specificity=metrics['specificity'],
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        balanced_accuracy=(metrics['recall'] + metrics['specificity']) / 2,
        matthews_corrcoef=mcc,
        fpr_curve=fpr,
        tpr_curve=tpr,
        precision_curve=precision,
        recall_curve=recall,
        thresholds=roc_thresholds,
    )


def compute_precision_recall(
    asri: pd.Series,
    crisis_dates: list,
    window_days: int = 30,
) -> tuple[float, float]:
    """
    Compute precision and recall using known crisis dates.
    
    Precision: Of all ASRI alerts, what fraction preceded actual crises?
    Recall: Of all crises, what fraction did ASRI detect?
    
    Args:
        asri: ASRI time series
        crisis_dates: List of known crisis dates
        window_days: Days before crisis that count as "detection"
        
    Returns:
        (precision, recall)
    """
    # Define "alert" as ASRI > 70 (high risk)
    alert_threshold = 70
    
    alerts = asri[asri > alert_threshold]
    
    # For each alert, check if crisis followed within window
    true_positives = 0
    for alert_date in alerts.index:
        for crisis_date in crisis_dates:
            crisis_ts = pd.Timestamp(crisis_date)
            if 0 < (crisis_ts - alert_date).days <= window_days:
                true_positives += 1
                break
    
    # For each crisis, check if alert preceded it
    detected_crises = 0
    for crisis_date in crisis_dates:
        crisis_ts = pd.Timestamp(crisis_date)
        window_start = crisis_ts - pd.Timedelta(days=window_days)
        window_alerts = alerts[(alerts.index >= window_start) & (alerts.index < crisis_ts)]
        if len(window_alerts) > 0:
            detected_crises += 1
    
    precision = true_positives / len(alerts) if len(alerts) > 0 else 0
    recall = detected_crises / len(crisis_dates) if len(crisis_dates) > 0 else 0
    
    return precision, recall


def format_roc_table(metrics: CrisisClassificationMetrics, model_name: str = "ASRI") -> str:
    """Format ROC metrics as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Crisis Prediction Performance: " + model_name + "}",
        r"\label{tab:roc_metrics}",
        r"\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"AUC-ROC & {metrics.auc_roc:.3f} \\\\",
        f"AUC-PR & {metrics.auc_pr:.3f} \\\\",
        f"Optimal Threshold & {metrics.optimal_threshold:.1f} \\\\",
        r"\midrule",
        f"Precision & {metrics.precision:.3f} \\\\",
        f"Recall (Sensitivity) & {metrics.recall:.3f} \\\\",
        f"Specificity & {metrics.specificity:.3f} \\\\",
        f"F1 Score & {metrics.f1_score:.3f} \\\\",
        f"Balanced Accuracy & {metrics.balanced_accuracy:.3f} \\\\",
        f"Matthews Correlation & {metrics.matthews_corrcoef:.3f} \\\\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textbf{Confusion Matrix}} \\",
        f"True Positives & {metrics.true_positives} \\\\",
        f"False Positives & {metrics.false_positives} \\\\",
        f"True Negatives & {metrics.true_negatives} \\\\",
        f"False Negatives & {metrics.false_negatives} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Crisis defined as 30-day forward drawdown $>$ 20\%.",
        r"\end{tablenotes}",
        r"\end{table}",
    ]
    
    return "\n".join(lines)


def plot_roc_curves(
    metrics_dict: dict[str, CrisisClassificationMetrics],
    output_path: str = "figures/roc_curves.pdf",
) -> str:
    """Generate matplotlib code for ROC curve comparison."""
    return f"""
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ROC curves
for name, metrics in metrics_dict.items():
    ax1.plot(metrics.fpr_curve, metrics.tpr_curve, 
             label=f'{{name}} (AUC={{metrics.auc_roc:.3f}})')

ax1.plot([0, 1], [0, 1], 'k--', label='Random')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Precision-Recall curves
for name, metrics in metrics_dict.items():
    ax2.plot(metrics.recall_curve, metrics.precision_curve,
             label=f'{{name}} (AUC={{metrics.auc_pr:.3f}})')

ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curves')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('{output_path}')
"""
