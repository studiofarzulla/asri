#!/usr/bin/env python3
"""
Moving-Block Bootstrap CIs for ASRI vs D-Y AUROC / AUPRC.

WHY: The existing bootstrap (compute_roc_metrics.py:bootstrap_ci and
real_dy_hmm_analysis.py:percentile_bootstrap_ci) resamples INDIVIDUAL DAYS
i.i.d. with replacement. That is invalid here: the crisis labels are four
contiguous ~30-day blocks (only ~4 independent events), and both the ASRI
score and the D-Y connectedness series are strongly autocorrelated. I.i.d.
day resampling treats 1,402 days as 1,402 independent draws, so it massively
understates sampling variability -> CIs ~4x too narrow.

This script replaces day-level resampling with a MOVING-BLOCK BOOTSTRAP (MBB,
Kunsch 1989 / Liu-Singh 1992): overlapping blocks of length L (~20-30 days)
are drawn with replacement and concatenated to length n, so within-block serial
dependence (and the contiguous crisis runs) is preserved. The SAME resampled
block indices are applied to ASRI and D-Y on each replicate, giving a PAIRED
bootstrap difference test for AUROC and AUPRC.

Point estimates are unchanged (only the CIs / inference change):
  AUROC via sklearn.metrics.roc_auc_score (identical to the paper's
  trapezoidal AUROC: ASRI 0.8657, D-Y 0.6696).
  AUPRC via the paper's trapezoidal average-precision (compute_auprc), which
  reproduces the reported 0.298 / 0.121; sklearn average_precision_score is
  also reported as a cross-check (a different, step-wise AP estimator).

Local methodology fix only. No data fabrication. Reuses the REAL data loader
and REAL rolling D-Y series from compute_roc_metrics.load_real_data.

Run: python3 scripts/moving_block_bootstrap_roc.py
Output: results/moving_block_bootstrap_roc.json
"""

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the paper's real data loader + trapezoidal AUPRC (keeps point estimates
# identical to the .tex table; no re-derivation).
_cr = _load_module("compute_roc_metrics", HERE / "compute_roc_metrics.py")
load_real_data = _cr.load_real_data
trapz_auprc = _cr.compute_auprc          # reproduces reported 0.298 / 0.121
trapz_auroc = _cr.compute_auroc          # == sklearn roc_auc_score here


# ---------------------------------------------------------------------------
# Metric wrappers (return np.nan on degenerate resamples)
# ---------------------------------------------------------------------------
def m_auroc(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, s)


def m_auprc(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return trapz_auprc(y, s)        # paper-consistent trapezoidal AP


def m_auprc_sklearn(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return average_precision_score(y, s)


# ---------------------------------------------------------------------------
# Moving-block bootstrap index generator
# ---------------------------------------------------------------------------
def mbb_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    One moving-block-bootstrap index vector of length n.

    Overlapping blocks: a block is the L consecutive positions [s, s+L) for any
    start s in 0..n-L. ceil(n/L) blocks are drawn with replacement and
    concatenated, then truncated to exactly n positions.
    """
    n_blocks = int(np.ceil(n / block_len))
    max_start = n - block_len  # inclusive
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
    return idx[:n]


def mbb_bootstrap(
    labels: np.ndarray,
    scores: dict,
    metric_fns: dict,
    block_len: int,
    n_boot: int = 2000,
    seed: int = 42,
):
    """
    Paired moving-block bootstrap.

    scores:     {model_name: score_array}
    metric_fns: {metric_name: callable(y, s)}
    Returns nested dict: replicates[model][metric] = np.ndarray(n_boot) and the
    SAME resampled block indices are used across models on each replicate (so
    differences are paired). Degenerate replicates -> np.nan (filtered later).
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    models = list(scores.keys())
    metrics = list(metric_fns.keys())
    reps = {mdl: {met: np.empty(n_boot) for met in metrics} for mdl in models}

    for b in range(n_boot):
        idx = mbb_indices(n, block_len, rng)
        yb = labels[idx]
        for mdl in models:
            sb = scores[mdl][idx]
            for met in metrics:
                reps[mdl][met][b] = metric_fns[met](yb, sb)
    return reps


def pct_ci(arr, alpha=0.05):
    a = arr[~np.isnan(arr)]
    lo = float(np.percentile(a, alpha / 2 * 100))
    hi = float(np.percentile(a, (1 - alpha / 2) * 100))
    return lo, hi, len(a)


def paired_diff_test(rep_a, rep_b, alpha=0.05):
    """
    Bootstrap test for diff = metric(A) - metric(B) using paired replicates.
    Returns point-free summary: diff CI, two-sided bootstrap p, frac<=0.
    """
    valid = ~np.isnan(rep_a) & ~np.isnan(rep_b)
    d = rep_a[valid] - rep_b[valid]
    lo = float(np.percentile(d, alpha / 2 * 100))
    hi = float(np.percentile(d, (1 - alpha / 2) * 100))
    frac_le0 = float(np.mean(d <= 0))
    frac_ge0 = float(np.mean(d >= 0))
    p_two = float(min(1.0, 2 * min(frac_le0, frac_ge0)))
    return dict(
        diff_mean=float(np.mean(d)),
        diff_ci_lower=lo,
        diff_ci_upper=hi,
        ci_excludes_zero=bool(lo > 0 or hi < 0),
        frac_le0=frac_le0,
        p_two_sided=p_two,
        n_valid=int(valid.sum()),
    )


def main():
    n_boot = 2000
    seed = 42
    block_lens = [20, 25, 30]
    main_L = 25

    print("=" * 78)
    print("Moving-block bootstrap CIs for ASRI vs D-Y (AUROC / AUPRC)")
    print("=" * 78)

    labels, asri, dy = load_real_data(PROJECT_ROOT / "results" / "data" / "asri_history.parquet")
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    print(f"n={n}  crisis-imminent={n_pos}  non-crisis={n_neg}  prevalence={n_pos/n:.1%}")

    # Count contiguous positive runs (the "true" number of independent events)
    runs = int(np.sum(np.diff(np.concatenate([[0], labels, [0]])) == 1))
    print(f"contiguous positive runs (independent crisis blocks) = {runs}")

    # Point estimates (unchanged from paper)
    point = {
        "ASRI": {
            "auroc": float(roc_auc_score(labels, asri)),
            "auprc_trapz": float(trapz_auprc(labels, asri)),
            "auprc_sklearn": float(average_precision_score(labels, asri)),
        },
        "DY": {
            "auroc": float(roc_auc_score(labels, dy)),
            "auprc_trapz": float(trapz_auprc(labels, dy)),
            "auprc_sklearn": float(average_precision_score(labels, dy)),
        },
    }
    print("\nPoint estimates (unchanged):")
    print(f"  ASRI AUROC={point['ASRI']['auroc']:.4f}  AUPRC(trapz)={point['ASRI']['auprc_trapz']:.4f}"
          f"  AUPRC(sklearn)={point['ASRI']['auprc_sklearn']:.4f}")
    print(f"  D-Y  AUROC={point['DY']['auroc']:.4f}  AUPRC(trapz)={point['DY']['auprc_trapz']:.4f}"
          f"  AUPRC(sklearn)={point['DY']['auprc_sklearn']:.4f}")

    scores = {"ASRI": asri, "DY": dy}
    metric_fns = {"auroc": m_auroc, "auprc_trapz": m_auprc, "auprc_sklearn": m_auprc_sklearn}

    results = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "method": "moving_block_bootstrap (overlapping blocks, Kunsch 1989)",
            "n_observations": n,
            "n_crisis_imminent": n_pos,
            "n_non_crisis": n_neg,
            "n_independent_crisis_blocks": runs,
            "n_bootstrap": n_boot,
            "seed": seed,
            "block_lengths_days": block_lens,
            "main_block_length_days": main_L,
            "point_estimates": point,
            "note": ("Replaces invalid i.i.d. day-level resampling. AUROC via "
                     "sklearn.roc_auc_score (== paper trapezoidal). Primary AUPRC "
                     "is the paper's trapezoidal AP (reproduces 0.298/0.121); "
                     "sklearn average_precision_score reported as cross-check."),
        },
        "by_block_length": {},
    }

    for L in block_lens:
        print("\n" + "-" * 78)
        print(f"[block length L={L} days]  B={n_boot} resamples")
        print("-" * 78)
        reps = mbb_bootstrap(labels, scores, metric_fns, block_len=L,
                             n_boot=n_boot, seed=seed)

        block_out = {"marginal": {}, "difference": {}}
        for mdl in ("ASRI", "DY"):
            block_out["marginal"][mdl] = {}
            for met in ("auroc", "auprc_trapz", "auprc_sklearn"):
                lo, hi, nv = pct_ci(reps[mdl][met])
                pe = point["ASRI" if mdl == "ASRI" else "DY"][met]
                block_out["marginal"][mdl][met] = dict(
                    point=float(pe), ci_lower=lo, ci_upper=hi, n_valid=nv)
                print(f"  {mdl:4s} {met:14s} point={pe:.4f}  95% MBB CI=[{lo:.4f}, {hi:.4f}]")

        # Paired difference tests (ASRI - DY), same blocks
        for met in ("auroc", "auprc_trapz", "auprc_sklearn"):
            d = paired_diff_test(reps["ASRI"][met], reps["DY"][met])
            block_out["difference"][met] = d
            # Marginal-CI separation
            asri_lo = block_out["marginal"]["ASRI"][met]["ci_lower"]
            dy_hi = block_out["marginal"]["DY"][met]["ci_upper"]
            separates = bool(asri_lo > dy_hi)
            block_out["difference"][met]["marginal_cis_separate"] = separates
            print(f"  DIFF {met:14s} mean={d['diff_mean']:+.4f}  "
                  f"95% CI=[{d['diff_ci_lower']:+.4f}, {d['diff_ci_upper']:+.4f}]  "
                  f"p2={d['p_two_sided']:.4f}  CI>0={d['ci_excludes_zero']}  "
                  f"marg-sep={separates}")

        results["by_block_length"][str(L)] = block_out

    # Headline summary at main_L
    main = results["by_block_length"][str(main_L)]
    results["headline"] = {
        "block_length_days": main_L,
        "ASRI_AUROC": main["marginal"]["ASRI"]["auroc"],
        "DY_AUROC": main["marginal"]["DY"]["auroc"],
        "ASRI_AUPRC_trapz": main["marginal"]["ASRI"]["auprc_trapz"],
        "DY_AUPRC_trapz": main["marginal"]["DY"]["auprc_trapz"],
        "AUROC_difference": main["difference"]["auroc"],
        "AUPRC_trapz_difference": main["difference"]["auprc_trapz"],
        "asri_significantly_beats_dy_auroc": bool(
            main["difference"]["auroc"]["ci_excludes_zero"]),
        "asri_significantly_beats_dy_auprc": bool(
            main["difference"]["auprc_trapz"]["ci_excludes_zero"]),
    }

    out_path = PROJECT_ROOT / "results" / "moving_block_bootstrap_roc.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")

    # Final human-readable verdict
    print("\n" + "=" * 78)
    print("VERDICT (main L=25)")
    print("=" * 78)
    a = main["marginal"]["ASRI"]
    d = main["marginal"]["DY"]
    da = main["difference"]["auroc"]
    dp = main["difference"]["auprc_trapz"]
    print(f"ASRI AUROC 0.866, MBB 95% CI [{a['auroc']['ci_lower']:.3f}, {a['auroc']['ci_upper']:.3f}]")
    print(f"D-Y  AUROC 0.670, MBB 95% CI [{d['auroc']['ci_lower']:.3f}, {d['auroc']['ci_upper']:.3f}]")
    print(f"ASRI AUPRC 0.298, MBB 95% CI [{a['auprc_trapz']['ci_lower']:.3f}, {a['auprc_trapz']['ci_upper']:.3f}]")
    print(f"D-Y  AUPRC 0.121, MBB 95% CI [{d['auprc_trapz']['ci_lower']:.3f}, {d['auprc_trapz']['ci_upper']:.3f}]")
    print(f"AUROC diff +{da['diff_mean']:.3f}  95% CI [{da['diff_ci_lower']:+.3f},{da['diff_ci_upper']:+.3f}]  "
          f"p={da['p_two_sided']:.4f}  -> ASRI beats D-Y: {da['ci_excludes_zero']}")
    print(f"AUPRC diff +{dp['diff_mean']:.3f}  95% CI [{dp['diff_ci_lower']:+.3f},{dp['diff_ci_upper']:+.3f}]  "
          f"p={dp['p_two_sided']:.4f}  -> ASRI beats D-Y: {dp['ci_excludes_zero']}")


if __name__ == "__main__":
    main()
