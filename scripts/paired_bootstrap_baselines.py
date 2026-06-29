#!/usr/bin/env python3
"""
Paired moving-block bootstrap for ASRI vs its fair baselines (PC1 and best
single channel), backing the demotion claim with the SAME test used for
ASRI-vs-D-Y (moving_block_bootstrap_roc.py).

WHY: The fair-baseline section (Table~\\ref{tab:fair_baselines}) currently
rests the "ASRI is statistically indistinguishable from PC1 (0.858) and from
its best single channel, Contagion (0.851)" claim on marginal-CI OVERLAP
(each baseline falls inside ASRI's i.i.d.-day bootstrap interval). Marginal-CI
overlap is a weak, indirect argument. The correct test is a PAIRED block
bootstrap of the AUROC DIFFERENCE (ASRI - baseline), with the SAME resampled
blocks applied to both series on each replicate, so the difference is evaluated
on identical days and the strong day-to-day autocorrelation cancels in the
contrast. This is exactly the test moving_block_bootstrap_roc.py runs for
ASRI - D-Y; here we run it for ASRI - PC1 and ASRI - Contagion.

PROTOCOL (identical to baseline_comparison.py, which reproduces 0.866/0.858/
0.851 on the canonical parquet):
  parquet  = results/data/asri_history.parquet (frozen canonical series)
  window   = 2021-01-01 .. 2024-12-31
  common   = asri_win.index INTERSECT rolling-D-Y.index  (same 1,402-day sample)
  labels   = create_crisis_labels(common, {Terra,Celsius/3AC,FTX,SVB}, 30d fwd)
  PC1      = first principal component of the 4 sub-indices, COVARIANCE form
             (the paper's PC1 baseline, AUROC 0.858, 74.9% var, loads ~0.81 on
             Contagion); standardized/correlation-form PC1 reported as a check.
  Contagion= contagion_risk sub-index (the best single channel, AUROC 0.851).
  AUROC    = sklearn.roc_auc_score (== paper trapezoidal on this sample).

Bootstrap = paired moving-block (Kunsch 1989), block L in {20,25,30}, B=2000,
seed=42 -- IDENTICAL machinery (mbb_bootstrap / paired_diff_test) to
moving_block_bootstrap_roc.py. Point estimates are unchanged; only inference.

No data fabrication. Reuses real loaders. Writes
results/paired_bootstrap_baselines.json.

Run: python3 scripts/paired_bootstrap_baselines.py
"""
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Real loaders / sample construction (same modules baseline_comparison.py uses)
_rd = _load("real_dy_hmm_analysis", HERE / "real_dy_hmm_analysis.py")
_cr = _load("compute_roc_metrics", HERE / "compute_roc_metrics.py")
# MBB machinery (same resampling + paired test as the ASRI-vs-D-Y script)
_mbb = _load("moving_block_bootstrap_roc", HERE / "moving_block_bootstrap_roc.py")

create_crisis_labels = _rd.create_crisis_labels
rolling_connectedness = _rd.rolling_connectedness
compute_auroc = _cr.compute_auroc
compute_auprc = _cr.compute_auprc

mbb_bootstrap = _mbb.mbb_bootstrap
paired_diff_test = _mbb.paired_diff_test
pct_ci = _mbb.pct_ci
m_auroc = _mbb.m_auroc
m_auprc = _mbb.m_auprc

from sklearn.metrics import roc_auc_score  # noqa: E402

SUB = ["stablecoin_risk", "defi_liquidity_risk", "contagion_risk", "arbitrage_opacity"]
CRISIS = [datetime(2022, 5, 12), datetime(2022, 6, 17),
          datetime(2022, 11, 11), datetime(2023, 3, 11)]
WIN_START, WIN_END = "2021-01-01", "2024-12-31"


def orient(labels, score):
    """Orient a raw score toward 'higher = more crisis-imminent' (AUROC>=0.5),
    matching baseline_comparison.auroc_oriented. Returns (oriented_score, sign)."""
    if compute_auroc(labels, score) >= 0.5:
        return score.astype(float), +1
    return (-score).astype(float), -1


def build_sample():
    df = pd.read_parquet(PROJECT_ROOT / "results" / "data" / "asri_history.parquet").sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)

    win = df.loc[WIN_START:WIN_END]
    roll = rolling_connectedness(df[SUB].dropna(), window=60, lags=1, horizon=10) \
        .loc[WIN_START:WIN_END].dropna()
    asri_win = win["asri"].dropna()
    common = asri_win.index.intersection(roll.index)

    labels = create_crisis_labels(common, CRISIS, window_days=30).astype(int)
    sub_df = win.loc[common, SUB].astype(float)
    asri = asri_win.loc[common].values.astype(float)

    # Contagion channel (best single channel; paper AUROC 0.851)
    contagion = sub_df["contagion_risk"].values.astype(float)

    # PC1 covariance form (the paper's PC1 baseline; 0.858, 74.9% var)
    X = sub_df.values.astype(float)
    Xc = X - X.mean(0)
    _, S2, Vt2 = np.linalg.svd(Xc, full_matrices=False)
    pc1_cov = Xc @ Vt2[0]
    cov_evr = float(S2[0] ** 2 / np.sum(S2 ** 2))
    cov_load = {c: float(v) for c, v in zip(SUB, Vt2[0])}

    # PC1 correlation/standardized form (robustness)
    Xz = (X - X.mean(0)) / X.std(0, ddof=0)
    _, S1, Vt1 = np.linalg.svd(Xz, full_matrices=False)
    pc1_std = Xz @ Vt1[0]
    std_evr = float(S1[0] ** 2 / np.sum(S1 ** 2))

    return dict(common=common, labels=np.asarray(labels), asri=asri,
                contagion=contagion, pc1_cov=pc1_cov, pc1_std=pc1_std,
                cov_evr=cov_evr, cov_load=cov_load, std_evr=std_evr)


def main():
    n_boot = 2000
    seed = 42
    block_lens = [20, 25, 30]
    main_L = 25

    print("=" * 80)
    print("Paired moving-block bootstrap: ASRI vs PC1 and vs Contagion (best channel)")
    print("=" * 80)

    S = build_sample()
    labels = S["labels"]
    n = len(labels)
    n_pos = int(labels.sum())
    runs = int(np.sum(np.diff(np.concatenate([[0], labels, [0]])) == 1))
    print(f"common sample n={n}  crisis-imminent={n_pos}  prevalence={n_pos/n:.1%}  "
          f"independent crisis blocks={runs}")
    print(f"span {S['common'].min().date()} .. {S['common'].max().date()}")

    # Orient each series toward higher=worse (all expected +1 on this sample)
    asri, s_asri = orient(labels, S["asri"])
    contagion, s_con = orient(labels, S["contagion"])
    pc1_cov, s_pc = orient(labels, S["pc1_cov"])
    pc1_std, s_pcs = orient(labels, S["pc1_std"])

    point = {
        "ASRI": {"auroc": float(roc_auc_score(labels, asri)),
                 "auprc_trapz": float(compute_auprc(labels, asri))},
        "PC1_cov": {"auroc": float(roc_auc_score(labels, pc1_cov)),
                    "auprc_trapz": float(compute_auprc(labels, pc1_cov)),
                    "explained_var_ratio": S["cov_evr"], "loadings": S["cov_load"],
                    "orientation_sign": s_pc},
        "Contagion": {"auroc": float(roc_auc_score(labels, contagion)),
                      "auprc_trapz": float(compute_auprc(labels, contagion)),
                      "orientation_sign": s_con},
        "PC1_std": {"auroc": float(roc_auc_score(labels, pc1_std)),
                    "auprc_trapz": float(compute_auprc(labels, pc1_std)),
                    "explained_var_ratio": S["std_evr"], "orientation_sign": s_pcs},
    }
    print("\nPoint estimates (unchanged; should match canon 0.866 / 0.858 / 0.851):")
    print(f"  ASRI      AUROC={point['ASRI']['auroc']:.4f}")
    print(f"  PC1 (cov) AUROC={point['PC1_cov']['auroc']:.4f}  "
          f"var={point['PC1_cov']['explained_var_ratio']:.3f}  "
          f"contagion_loading={S['cov_load']['contagion_risk']:.3f}")
    print(f"  Contagion AUROC={point['Contagion']['auroc']:.4f}")
    print(f"  PC1 (std) AUROC={point['PC1_std']['auroc']:.4f}  "
          f"var={point['PC1_std']['explained_var_ratio']:.3f}")

    scores = {"ASRI": asri, "PC1_cov": pc1_cov, "Contagion": contagion, "PC1_std": pc1_std}
    metric_fns = {"auroc": m_auroc, "auprc_trapz": m_auprc}

    results = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "method": ("paired moving-block bootstrap (overlapping blocks, Kunsch "
                       "1989); SAME resampled block indices applied to ASRI and each "
                       "baseline on every replicate (paired difference of AUROC/AUPRC)."),
            "purpose": ("Back the demotion claim with the proper test: is ASRI's "
                        "AUROC difference vs PC1 / vs best single channel (Contagion) "
                        "distinguishable from 0 under autocorrelation-robust inference?"),
            "protocol": ("IDENTICAL sample/labels to baseline_comparison.py "
                         "(parquet=results/data/asri_history.parquet, window "
                         "2021-01-01..2024-12-31, common=asri_win INTERSECT rolling-D-Y, "
                         "labels=30d forward pre-windows of {Terra,Celsius/3AC,FTX,SVB}). "
                         "PC1=covariance form (paper baseline); Contagion=contagion_risk."),
            "n_observations": n, "n_crisis_imminent": n_pos,
            "n_independent_crisis_blocks": runs,
            "n_bootstrap": n_boot, "seed": seed,
            "block_lengths_days": block_lens, "main_block_length_days": main_L,
            "point_estimates": point,
        },
        "by_block_length": {},
    }

    contrasts = [("ASRI", "PC1_cov"), ("ASRI", "Contagion"), ("ASRI", "PC1_std")]

    for L in block_lens:
        print("\n" + "-" * 80)
        print(f"[block length L={L} days]  B={n_boot} resamples (same blocks across all series)")
        print("-" * 80)
        reps = mbb_bootstrap(labels, scores, metric_fns, block_len=L,
                             n_boot=n_boot, seed=seed)

        block_out = {"marginal": {}, "difference": {}}
        for mdl in scores:
            block_out["marginal"][mdl] = {}
            for met in ("auroc", "auprc_trapz"):
                lo, hi, nv = pct_ci(reps[mdl][met])
                pe = point[mdl][met]
                block_out["marginal"][mdl][met] = dict(point=float(pe), ci_lower=lo,
                                                        ci_upper=hi, n_valid=nv)

        for (a, b) in contrasts:
            block_out["difference"][f"{a}_minus_{b}"] = {}
            for met in ("auroc", "auprc_trapz"):
                d = paired_diff_test(reps[a][met], reps[b][met])
                block_out["difference"][f"{a}_minus_{b}"][met] = d
                print(f"  {a}-{b:9s} {met:12s} mean={d['diff_mean']:+.4f}  "
                      f"95% CI=[{d['diff_ci_lower']:+.4f}, {d['diff_ci_upper']:+.4f}]  "
                      f"p2={d['p_two_sided']:.3f}  CI-excludes-0={d['ci_excludes_zero']}")
        results["by_block_length"][str(L)] = block_out

    # Headline at main_L
    main = results["by_block_length"][str(main_L)]
    headline = {"block_length_days": main_L}
    for (a, b) in contrasts:
        key = f"{a}_minus_{b}"
        dd = main["difference"][key]["auroc"]
        headline[key] = {
            "auroc_diff_mean": dd["diff_mean"],
            "auroc_diff_ci": [dd["diff_ci_lower"], dd["diff_ci_upper"]],
            "ci_excludes_zero": dd["ci_excludes_zero"],
            "p_two_sided": dd["p_two_sided"],
        }
    # Robustness: does ANY block length give an excludes-0 AUROC contrast?
    headline["any_blocklen_excludes_zero_auroc"] = {
        f"{a}_minus_{b}": bool(any(
            results["by_block_length"][str(L)]["difference"][f"{a}_minus_{b}"]["auroc"]["ci_excludes_zero"]
            for L in block_lens))
        for (a, b) in contrasts
    }
    results["headline"] = headline

    out_path = PROJECT_ROOT / "results" / "paired_bootstrap_baselines.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 80)
    print(f"VERDICT (main L={main_L})")
    print("=" * 80)
    for (a, b) in contrasts:
        h = headline[f"{a}_minus_{b}"]
        any0 = headline["any_blocklen_excludes_zero_auroc"][f"{a}_minus_{b}"]
        verdict = ("DISTINGUISHABLE (CI excludes 0)" if h["ci_excludes_zero"]
                   else "INDISTINGUISHABLE (CI includes 0)")
        print(f"  {a} - {b}:  AUROC diff {h['auroc_diff_mean']:+.4f}  "
              f"95% CI [{h['auroc_diff_ci'][0]:+.4f}, {h['auroc_diff_ci'][1]:+.4f}]  "
              f"p={h['p_two_sided']:.3f}  -> {verdict}  "
              f"(any L excludes 0: {any0})")


if __name__ == "__main__":
    main()
