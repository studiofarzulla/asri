#!/usr/bin/env python3
"""
Generate ROC and Precision-Recall curves figure for the ASRI paper.
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asri.validation.roc_analysis import (
    compute_roc_curve,
    compute_precision_recall_curve,
    compute_auc,
    optimal_threshold,
)

# Crisis dates for defining "positive" class
CRISIS_EVENTS = [
    datetime(2022, 5, 12),   # Terra/Luna
    datetime(2022, 6, 17),   # Celsius/3AC
    datetime(2022, 11, 11),  # FTX
    datetime(2023, 3, 11),   # SVB
]


def create_crisis_labels(index, crisis_dates, window_days=30):
    """Create binary labels: 1 if crisis occurs within window_days."""
    labels = np.zeros(len(index))

    for i, date in enumerate(index):
        for crisis in crisis_dates:
            crisis_ts = pd.Timestamp(crisis)
            days_to_crisis = (crisis_ts - date).days
            if 0 <= days_to_crisis <= window_days:
                labels[i] = 1
                break

    return labels


def main():
    # Set publication style
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['legend.fontsize'] = 9

    # Load data
    df = pd.read_parquet(PROJECT_ROOT / "results" / "data" / "asri_history.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)

    asri = df["asri"]

    # Create labels for 30-day forward crisis prediction
    y_true = create_crisis_labels(asri.index, CRISIS_EVENTS, window_days=30)
    y_score = asri.values

    # Remove NaN
    mask = ~np.isnan(y_score)
    y_score = y_score[mask]
    y_true = y_true[mask]

    # Compute curves
    fpr, tpr, roc_thresholds = compute_roc_curve(y_true, y_score)
    auc_roc = compute_auc(fpr, tpr)

    precision, recall, pr_thresholds = compute_precision_recall_curve(y_true, y_score)
    # Fix: sort by recall for proper AUC computation
    sort_idx = np.argsort(recall)
    auc_pr = np.trapezoid(precision[sort_idx], recall[sort_idx])

    # Find optimal threshold
    opt_thresh, metrics = optimal_threshold(y_true, y_score, criterion="f1")

    print(f"AUC-ROC: {auc_roc:.3f}")
    print(f"AUC-PR: {auc_pr:.3f}")
    print(f"Optimal threshold: {opt_thresh:.1f}")
    print(f"At optimal: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ROC curve
    ax1 = axes[0]
    ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'ASRI (AUC = {auc_roc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random classifier')
    ax1.fill_between(fpr, tpr, alpha=0.1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('(a) ROC Curve')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Mark optimal point
    opt_fpr = metrics['fp'] / (metrics['fp'] + metrics['tn']) if (metrics['fp'] + metrics['tn']) > 0 else 0
    opt_tpr = metrics['recall']
    ax1.scatter([opt_fpr], [opt_tpr], c='red', s=100, zorder=5, marker='o')
    ax1.annotate(f'Threshold={opt_thresh:.0f}',
                 xy=(opt_fpr, opt_tpr),
                 xytext=(opt_fpr + 0.1, opt_tpr - 0.1),
                 fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    # Precision-Recall curve
    ax2 = axes[1]
    ax2.plot(recall, precision, 'b-', linewidth=2, label=f'ASRI (AUC = {auc_pr:.3f})')

    # Baseline: proportion of positive class
    baseline = np.mean(y_true)
    ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=1, alpha=0.5,
                label=f'Baseline (prevalence = {baseline:.3f})')

    ax2.fill_between(recall, precision, alpha=0.1)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('(b) Precision-Recall Curve')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    # Mark optimal point
    ax2.scatter([metrics['recall']], [metrics['precision']], c='red', s=100, zorder=5, marker='o')

    plt.tight_layout()

    # Save figure
    output_path = PROJECT_ROOT / "paper" / "figures" / "roc_pr_curves.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved figure to: {output_path}")

    # Also save as PNG for quick preview
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=150)

    plt.close()

    # Generate LaTeX inclusion code
    print("\n" + "="*60)
    print("LaTeX inclusion code:")
    print("="*60)
    print(r"""
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/roc_pr_curves.pdf}
\caption{ASRI Classification Performance for 30-Day Crisis Prediction.
(a) ROC curve showing trade-off between true positive rate and false positive rate;
AUC = """ + f"{auc_roc:.3f}" + r""".
(b) Precision-Recall curve accounting for class imbalance; AUC = """ + f"{auc_pr:.3f}" + r""".
Red markers indicate F1-optimal threshold (""" + f"{opt_thresh:.0f}" + r""").
Crisis defined as ASRI threshold breach within 30-day pre-crisis window for four historical events (Terra/Luna, Celsius/3AC, FTX, SVB).}
\label{fig:roc_pr}
\end{figure}
""")


if __name__ == "__main__":
    main()
