#!/usr/bin/env python3
"""
REAL Diebold-Yilmaz connectedness + HMM reconciliation for the ASRI paper.

Replaces the fabricated `generate_synthetic_data()` path in
`compute_roc_metrics.py` (which used np.random "calibrated to paper results")
with a genuine VAR + generalized (Pesaran-Shin) FEVD connectedness measure on
the REAL ASRI sub-indices, and recomputes AUROC / AUPRC / precision for both
ASRI and the real D-Y connectedness series against the REAL crisis labels.

Also runs the HMM 2/3/4-state selection sweep with 10 random restarts
(best-by-log-likelihood) using the released RegimeDetector, to collapse the
three contradictory HMM number-sets to one reproducible fit, and verifies the
"24.6% zeros in Contagion Risk" footnote against the released data.

Local proposal only. No data fabrication in either direction.
Run: python3 scripts/real_dy_hmm_analysis.py
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from statsmodels.tsa.api import VAR

from asri.regime.hmm import RegimeDetector

SUB_INDICES = ["stablecoin_risk", "defi_liquidity_risk", "contagion_risk", "arbitrage_opacity"]

# Canonical crisis dates (match run_event_study.py / generate_roc_figure.py)
CRISIS_EVENTS = [
    datetime(2022, 5, 12),   # Terra/Luna
    datetime(2022, 6, 17),   # Celsius/3AC
    datetime(2022, 11, 11),  # FTX
    datetime(2023, 3, 11),   # SVB
]

# Paper's stated analysis window for the day-level precision/recall (1,461 days)
WINDOW_START = "2021-01-01"
WINDOW_END = "2024-12-31"


# ---------------------------------------------------------------------------
# Generalized (Pesaran-Shin 1998) FEVD total connectedness on a fitted VAR
# ---------------------------------------------------------------------------
def generalized_fevd_connectedness(resid_cov: np.ndarray, ma_coefs: np.ndarray) -> float:
    """
    Diebold-Yilmaz (2012) TOTAL connectedness using a generalized (KPPS / Pesaran-Shin)
    forecast-error variance decomposition, which is invariant to variable ordering.

    Args:
        resid_cov: K x K residual covariance matrix (Sigma) of the VAR.
        ma_coefs:  (H+1) x K x K array of VMA(infinity) coefficient matrices A_0..A_H,
                   where A_0 = I (as returned by statsmodels VARResults.ma_rep(H)).

    Returns:
        Total connectedness in percent (0-100).
    """
    Sigma = resid_cov
    K = Sigma.shape[0]
    H1 = ma_coefs.shape[0]  # = H + 1 horizons (includes step 0)

    sigma_jj = np.diag(Sigma)  # K

    # theta[i, j] = generalized FEVD: share of variable i's H-step forecast-error
    #   variance attributable to shocks in variable j.
    theta = np.zeros((K, K))
    for i in range(K):
        # denominator: sum_h (e_i' A_h Sigma A_h' e_i)
        denom = 0.0
        for h in range(H1):
            Ah = ma_coefs[h]
            row = Ah[i, :]  # e_i' A_h
            denom += row @ Sigma @ row.T
        for j in range(K):
            num = 0.0
            for h in range(H1):
                Ah = ma_coefs[h]
                # (e_i' A_h Sigma e_j)^2
                val = Ah[i, :] @ Sigma[:, j]
                num += val ** 2
            theta[i, j] = (1.0 / sigma_jj[j]) * num / denom

    # Row-normalize (KPPS shares do not sum to 1 across j because shocks are correlated)
    row_sums = theta.sum(axis=1, keepdims=True)
    theta_norm = theta / row_sums

    # Total connectedness = (sum of off-diagonal) / K * 100
    off_diag = theta_norm.sum() - np.trace(theta_norm)
    total_connectedness = off_diag / K * 100.0
    return total_connectedness


def static_connectedness(data: pd.DataFrame, lags: int, horizon: int = 10) -> float:
    """Full-sample static D-Y total connectedness at the given VAR lag and FEVD horizon."""
    model = VAR(data.values)
    res = model.fit(lags)
    ma = res.ma_rep(horizon)  # (horizon+1) x K x K, A_0 = I
    return generalized_fevd_connectedness(res.sigma_u, ma)


def rolling_connectedness(
    data: pd.DataFrame,
    window: int = 60,
    lags: int = 1,
    horizon: int = 10,
) -> pd.Series:
    """
    Rolling D-Y total connectedness, replicating the paper's stated spec
    (60-day rolling VAR(1), generalized FEVD at H=10).
    """
    idx = data.index
    vals = data.values
    out_dates = []
    out_conn = []
    for end in range(window, len(vals) + 1):
        sub = vals[end - window:end]
        try:
            model = VAR(sub)
            res = model.fit(lags)
            ma = res.ma_rep(horizon)
            c = generalized_fevd_connectedness(res.sigma_u, ma)
        except Exception:
            c = np.nan
        out_dates.append(idx[end - 1])
        out_conn.append(c)
    return pd.Series(out_conn, index=pd.DatetimeIndex(out_dates), name="connectedness")


# ---------------------------------------------------------------------------
# Classification metrics (self-contained; mirrors roc_analysis.py logic)
# ---------------------------------------------------------------------------
def create_crisis_labels(index, crisis_dates, window_days: int = 30) -> np.ndarray:
    """1 if a crisis date falls within `window_days` AHEAD of this date (pre-crisis window)."""
    labels = np.zeros(len(index))
    for i, date in enumerate(index):
        for crisis in crisis_dates:
            days_to_crisis = (pd.Timestamp(crisis) - pd.Timestamp(date)).days
            if 0 <= days_to_crisis <= window_days:
                labels[i] = 1
                break
    return labels


def roc_curve(y_true, y_score):
    thresholds = np.unique(y_score)[::-1]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    tpr_list, fpr_list = [0.0], [0.0]
    for t in thresholds:
        pred = y_score >= t
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        tpr_list.append(tp / n_pos if n_pos > 0 else 0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0)
    tpr_list.append(1.0)
    fpr_list.append(1.0)
    return np.array(fpr_list), np.array(tpr_list)


def auc_trapz(x, y):
    idx = np.argsort(x)
    return np.trapezoid(y[idx], x[idx])


def auroc(y_true, y_score):
    fpr, tpr = roc_curve(y_true, y_score)
    return auc_trapz(fpr, tpr)


def auprc(y_true, y_score):
    thresholds = np.unique(y_score)[::-1]
    n_pos = y_true.sum()
    prec, rec = [], []
    for t in thresholds:
        pred = y_score >= t
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        prec.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        rec.append(tp / n_pos if n_pos > 0 else 0)
    prec, rec = np.array(prec), np.array(rec)
    idx = np.argsort(rec)
    return auc_trapz(rec[idx], prec[idx])


def youden_optimal(y_true, y_score):
    """Optimal threshold by Youden's J (matches the synthetic table's stated criterion)."""
    thresholds = np.unique(y_score)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    best_j = -np.inf
    best = {}
    for t in thresholds:
        pred = y_score >= t
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        tn = np.sum((~pred) & (y_true == 0))
        fn = np.sum((~pred) & (y_true == 1))
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best = dict(threshold=t, precision=prec, recall=tpr, fpr=fpr,
                        f1=(2 * prec * tpr / (prec + tpr)) if (prec + tpr) > 0 else 0,
                        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))
    return best


def percentile_bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_score)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots[b] = metric_fn(y_true[idx], y_score[idx])
    lo = np.percentile(boots, alpha / 2 * 100)
    hi = np.percentile(boots, (1 - alpha / 2) * 100)
    return point, lo, hi


# ---------------------------------------------------------------------------
# HMM sweep
# ---------------------------------------------------------------------------
def hmm_sweep(sub_indices: pd.DataFrame, ks=(2, 3, 4), n_restarts=10):
    """10 random restarts per K, keep best-by-LL. Returns dict K -> best result + summary."""
    out = {}
    for K in ks:
        best_ll = -np.inf
        best_result = None
        best_seed = None
        for seed in range(n_restarts):
            det = RegimeDetector(n_regimes=K, n_iterations=1000,
                                 convergence_threshold=1e-4, random_state=seed)
            det.fit(sub_indices)
            r = det.result
            if r.log_likelihood > best_ll:
                best_ll = r.log_likelihood
                best_result = r
                best_seed = seed
        out[K] = dict(result=best_result, seed=best_seed, ll=best_ll)
    return out


def ergodic_distribution(A: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(A.T)
    idx = np.argmin(np.abs(eigvals - 1))
    erg = np.real(eigvecs[:, idx])
    return erg / erg.sum()


def main():
    print("=" * 78)
    print("REAL Diebold-Yilmaz + HMM analysis for ASRI (no synthetic data)")
    print("=" * 78)

    df = pd.read_parquet(PROJECT_ROOT / "results" / "data" / "asri_history.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    full = df.copy()
    win = df.loc[WINDOW_START:WINDOW_END].copy()
    print(f"\nFull sample: {full.index.min().date()} .. {full.index.max().date()} "
          f"({len(full)} rows)")
    print(f"Paper window {WINDOW_START}..{WINDOW_END}: {len(win)} rows")

    # -------------------------------------------------------------------
    # 1. STATIC full-sample D-Y total connectedness (vs paper's claimed 0.3%)
    # -------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[1] STATIC full-sample D-Y total connectedness (generalized FEVD, H=10)")
    print("-" * 78)
    for label, dd in [("FULL 2021-2026", full[SUB_INDICES].dropna()),
                      ("WINDOW 2021-2024", win[SUB_INDICES].dropna())]:
        for lags in (1, 2):
            c = static_connectedness(dd, lags=lags, horizon=10)
            print(f"  {label:18s}  VAR({lags})  static total connectedness = {c:6.2f}%")

    # -------------------------------------------------------------------
    # 2. ROLLING D-Y connectedness (60-day VAR(1), GFEVD H=10) on real data
    # -------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[2] ROLLING D-Y total connectedness (60-day window, VAR(1), GFEVD H=10)")
    print("-" * 78)
    roll_full = rolling_connectedness(full[SUB_INDICES].dropna(), window=60, lags=1, horizon=10)
    roll = roll_full.loc[WINDOW_START:WINDOW_END].dropna()
    print(f"  Rolling series (window): n={len(roll)}  "
          f"mean={roll.mean():.2f}%  std={roll.std():.2f}%  "
          f"min={roll.min():.2f}%  max={roll.max():.2f}%")
    print(f"  mean+1sd = {roll.mean() + roll.std():.2f}%")
    # Compare to the released CSV (no generator in repo)
    csv = pd.read_csv(PROJECT_ROOT / "results" / "data" / "dy_rolling_connectedness_daily.csv",
                      parse_dates=["date"]).set_index("date")["connectedness"]
    print(f"  [released CSV] mean={csv.mean():.2f}% min={csv.min():.2f}% max={csv.max():.2f}% "
          f"(n={len(csv)})")

    # -------------------------------------------------------------------
    # 3. CLASSIFICATION: ASRI vs real rolling D-Y on real crisis labels
    #    Use the paper window and the SAME labeling as generate_roc_figure.py
    # -------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[3] CLASSIFICATION on REAL crisis labels (30-day pre-crisis windows)")
    print("-" * 78)

    asri_win = win["asri"].dropna()

    # Align ASRI and rolling D-Y to a COMMON index so both are scored on the same days/labels
    common = asri_win.index.intersection(roll.index)
    asri_c = asri_win.loc[common].values
    dy_c = roll.loc[common].values
    labels = create_crisis_labels(common, CRISIS_EVENTS, window_days=30)

    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    print(f"  Common scoring sample: n={len(common)} "
          f"({common.min().date()} .. {common.max().date()}); "
          f"{n_pos} crisis-imminent, {n_neg} non-crisis "
          f"(prevalence {n_pos / len(common):.1%})")

    results = {}
    for name, score in [("ASRI", asri_c), ("D-Y (real)", dy_c)]:
        au, au_lo, au_hi = percentile_bootstrap_ci(labels, score, auroc)
        ap, ap_lo, ap_hi = percentile_bootstrap_ci(labels, score, auprc)
        opt = youden_optimal(labels, score)
        results[name] = dict(auroc=au, auroc_ci=(au_lo, au_hi),
                             auprc=ap, auprc_ci=(ap_lo, ap_hi), opt=opt)
        print(f"\n  {name}:")
        print(f"    AUROC = {au:.3f}  [{au_lo:.3f}, {au_hi:.3f}]")
        print(f"    AUPRC = {ap:.3f}  [{ap_lo:.3f}, {ap_hi:.3f}]")
        print(f"    Youden-opt threshold = {opt['threshold']:.3f}")
        print(f"      precision={opt['precision']:.3f} recall={opt['recall']:.3f} "
              f"f1={opt['f1']:.3f}  (TP={opt['tp']} FP={opt['fp']} TN={opt['tn']} FN={opt['fn']})")

    # ALSO: ASRI scored on its FULL native index (matches the reproducible AUC 0.890 figure)
    asri_full_score = asri_win.values
    asri_full_labels = create_crisis_labels(asri_win.index, CRISIS_EVENTS, window_days=30)
    print(f"\n  [cross-check] ASRI on full native window index (n={len(asri_win)}): "
          f"AUROC={auroc(asri_full_labels, asri_full_score):.3f}, "
          f"AUPRC={auprc(asri_full_labels, asri_full_score):.3f}")

    # -------------------------------------------------------------------
    # 4. HMM: 10-restart sweep, best-by-LL, 2/3/4 states
    # -------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[4] HMM 10-restart sweep (best-by-LL) on sub-indices")
    print("-" * 78)
    # The released extract script fits on the FULL sample; report both.
    for label, dd in [("FULL 2021-2026", full[SUB_INDICES].dropna()),
                      ("WINDOW 2021-2024", win[SUB_INDICES].dropna())]:
        print(f"\n  === HMM sweep on {label} (n={len(dd)}) ===")
        sweep = hmm_sweep(dd, ks=(2, 3, 4), n_restarts=10)
        print(f"  {'K':>2} {'best_seed':>9} {'LL':>14} {'AIC':>14} {'BIC':>14}")
        for K in (2, 3, 4):
            r = sweep[K]["result"]
            print(f"  {K:>2} {sweep[K]['seed']:>9} {r.log_likelihood:>14.1f} "
                  f"{r.aic:>14.1f} {r.bic:>14.1f}")
        # Detailed 3-state structure
        r3 = sweep[3]["result"]
        T = len(r3.regime_labels)
        freqs = [np.mean(r3.regime_labels == i) for i in range(3)]
        means = [np.mean(r3.regime_means[i]) for i in range(3)]
        persist = [r3.transition_matrix[i, i] for i in range(3)]
        erg = ergodic_distribution(r3.transition_matrix)
        print(f"  3-state (best seed {sweep[3]['seed']}):")
        print(f"    freqs       = {[f'{f:.1%}' for f in freqs]}")
        print(f"    mean risk   = {[f'{m:.1f}' for m in means]}")
        print(f"    persistence = {[f'{p:.3f}' for p in persist]}")
        print(f"    ergodic     = {[f'{e:.3f}' for e in erg]}")

    # -------------------------------------------------------------------
    # 5. Verify the "24.6% zeros in Contagion Risk" footnote
    # -------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[5] VERIFY footnote: '24.6% zeros in Contagion Risk'")
    print("-" * 78)
    for label, dd in [("FULL 2021-2026", full), ("WINDOW 2021-2024", win)]:
        cr = dd["contagion_risk"].dropna()
        pct_zero = float((cr == 0).mean() * 100)
        pct_le_1 = float((cr <= 1).mean() * 100)
        print(f"  {label:18s} contagion_risk: min={cr.min():.2f} max={cr.max():.2f} "
              f"%==0 = {pct_zero:.1f}%  %<=1 = {pct_le_1:.1f}%")

    print("\nDONE.")


if __name__ == "__main__":
    main()
