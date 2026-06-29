#!/usr/bin/env python3
r"""
Regenerate ALL ASRI paper figures into arxiv-submission/figures/ from the SINGLE
canonical data file (results/data/asri_history.parquet) and the same code paths
that produce the canon tables. This pins figures and tables to one source so they
cannot diverge again.

Figures written (the five the paper \includegraphics):
  - asri_timeseries.pdf        plot_asri_time_series        (parquet 'asri')
  - decomposition.pdf          plot_sub_index_decomposition (parquet sub-indices + THEORETICAL_WEIGHTS)
  - event_study.pdf            plot_event_study_panels      (run_event_study, profile paper_v2; post=peak)
  - sensitivity.pdf            plot_sensitivity_heatmaps    (run_weight_perturbation_analysis)
  - roc_pr_curves.pdf          (canon ROC: aligned n=1402 sample, Youden's J, trapezoidal AUC)

Canon targets (must match tab:event_study / tab:roc_metrics):
  Event study peaks: Terra/Luna 48.7 (<50), Celsius/3AC 71.4, FTX 84.7, SVB 68.7
  ROC: AUROC 0.866, AUPRC 0.298, threshold 45.3, prevalence 0.088
"""
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

DATA = PROJECT_ROOT / "results" / "data" / "asri_history.parquet"
# Default to an in-tree output dir so the repo is self-contained; override with
# ASRI_FIGURES_OUT (e.g. point it at ../arxiv-submission/figures when refreshing
# the manuscript figures).
OUT = Path(os.environ.get("ASRI_FIGURES_OUT", str(PROJECT_ROOT / "results" / "figures")))
OUT.mkdir(parents=True, exist_ok=True)

# --- canonical constants (mirror scripts/run_full_validation.py) -------------
THEORETICAL_WEIGHTS = {
    "stablecoin_risk": 0.30,
    "defi_liquidity_risk": 0.25,
    "contagion_risk": 0.25,
    "arbitrage_opacity": 0.20,
}
SUB_INDEX_COLUMNS = list(THEORETICAL_WEIGHTS.keys())
CRISIS_EVENTS = [
    {"name": "Terra/Luna", "date": datetime(2022, 5, 12)},
    {"name": "Celsius/3AC", "date": datetime(2022, 6, 17)},
    {"name": "FTX Collapse", "date": datetime(2022, 11, 11)},
    {"name": "SVB Crisis", "date": datetime(2023, 3, 11)},
]
EVENT_STUDY_PROFILE = "paper_v2"


def load_df() -> pd.DataFrame:
    df = pd.read_parquet(DATA).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
    return df


def gen_timeseries(df):
    from asri.publication.figures import plot_asri_time_series

    plot_asri_time_series(
        asri=df["asri"],
        crisis_dates=[e["date"] for e in CRISIS_EVENTS],
        crisis_labels=[e["name"] for e in CRISIS_EVENTS],
        output_path=str(OUT / "asri_timeseries.pdf"),
    )


def gen_decomposition(df):
    from asri.publication.figures import plot_sub_index_decomposition

    plot_sub_index_decomposition(
        sub_indices=df[SUB_INDEX_COLUMNS],
        weights=THEORETICAL_WEIGHTS,
        output_path=str(OUT / "decomposition.pdf"),
    )


def gen_event_study(df):
    from asri.publication.figures import plot_event_study_panels
    from asri.validation.event_study import run_event_study, CrisisEvent

    asri = df["asri"]
    events = [CrisisEvent(e["name"], e["date"]) for e in CRISIS_EVENTS]
    res = run_event_study(asri, events, profile=EVENT_STUDY_PROFILE)

    event_results = []
    print("\n  event-study (profile=%s):" % EVENT_STUDY_PROFILE)
    for r in res:
        traj = r.asri_trajectory
        print(
            f"    {r.event.name:<14} pre={r.pre_event_mean:5.1f}  peak={r.peak_asri:5.1f}  "
            f"CAS={r.cumulative_abnormal_signal:6.1f}  traj[min,max]=[{traj.min():.1f},{traj.max():.1f}]"
        )
        event_results.append(
            {
                "name": r.event.name,
                "date": r.event.event_date,
                "asri_window": traj,
                "pre_mean": r.pre_event_mean,
                "post_mean": r.peak_asri,  # caption "Post" = peak (canon)
            }
        )
    plot_event_study_panels(event_results=event_results, output_path=str(OUT / "event_study.pdf"))


def gen_sensitivity(df):
    from asri.publication.figures import plot_sensitivity_heatmaps
    from asri.validation.sensitivity import run_weight_perturbation_analysis

    weight_res = run_weight_perturbation_analysis(
        sub_indices=df[SUB_INDEX_COLUMNS],
        weights=THEORETICAL_WEIGHTS,
        crisis_dates=[e["date"] for e in CRISIS_EVENTS],
    )
    weight_results = []
    for r in weight_res:
        for i, delta in enumerate(r.perturbation_grid):
            weight_results.append(
                {
                    "sub_index": r.component,
                    "perturbation": round(float(delta), 2),
                    "mean_asri": float(r.asri_means[i]),
                    "std_asri": float(r.asri_stds[i]),
                    "max_asri": float(r.asri_means[i] + r.asri_stds[i]),
                }
            )
    plot_sensitivity_heatmaps(weight_results=weight_results, output_path=str(OUT / "sensitivity.pdf"))


def gen_roc():
    """Canon ROC/PR on the aligned n=1402 sample with Youden's J threshold."""
    _spec = importlib.util.spec_from_file_location(
        "compute_roc_metrics", SCRIPTS / "compute_roc_metrics.py"
    )
    C = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(C)

    labels, asri, dy = C.load_real_data(DATA)
    n = len(labels)
    prevalence = float(labels.mean())

    fpr, tpr, _ = C.compute_roc_curve(labels, asri)
    auroc = C.compute_auc_trapezoidal(fpr, tpr)
    precision, recall, _ = C.compute_pr_curve(labels, asri)
    auprc = C.compute_auprc(labels, asri)
    thr, m = C.find_optimal_threshold(labels, asri)

    # FPR/TPR at the Youden-optimal threshold (for the ROC marker)
    pred = asri >= thr
    tp = float(np.sum(pred & (labels == 1)))
    fp = float(np.sum(pred & (labels == 0)))
    tn = float(np.sum(~pred & (labels == 0)))
    opt_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    opt_tpr = m["recall"]

    print("\n  ROC (aligned n=%d, prevalence=%.4f):" % (n, prevalence))
    print(
        f"    ASRI AUROC={auroc:.4f}  AUPRC={auprc:.4f}  thr(Youden)={float(thr):.1f}  "
        f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
    )

    import matplotlib as mpl

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 11
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["legend.fontsize"] = 9

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1 = axes[0]
    ax1.plot(fpr, tpr, "b-", linewidth=2, label=f"ASRI (AUC = {auroc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random classifier")
    ax1.fill_between(fpr, tpr, alpha=0.1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("(a) ROC Curve")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.scatter([opt_fpr], [opt_tpr], c="red", s=100, zorder=5, marker="o")
    ax1.annotate(
        f"Threshold = {float(thr):.1f}",
        xy=(opt_fpr, opt_tpr),
        xytext=(opt_fpr + 0.12, opt_tpr - 0.12),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
    )

    ax2 = axes[1]
    ax2.plot(recall, precision, "b-", linewidth=2, label=f"ASRI (AUC = {auprc:.3f})")
    ax2.axhline(
        y=prevalence,
        color="k",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Baseline (prevalence = {prevalence:.3f})",
    )
    ax2.fill_between(recall, precision, alpha=0.1)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("(b) Precision-Recall Curve")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.scatter([m["recall"]], [m["precision"]], c="red", s=100, zorder=5, marker="o")

    plt.tight_layout()
    fig.savefig(OUT / "roc_pr_curves.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {OUT / 'roc_pr_curves.pdf'}")


def main():
    print(f"Data: {DATA}")
    print(f"Out : {OUT}")
    df = load_df()
    print(f"Loaded {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
    gen_timeseries(df)
    gen_decomposition(df)
    gen_event_study(df)
    gen_sensitivity(df)
    gen_roc()
    print("\nAll five arxiv figures regenerated.")


if __name__ == "__main__":
    main()
