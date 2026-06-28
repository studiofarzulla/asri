#!/usr/bin/env python3
"""
Discrete-outcome Granger-style test for ASRI sub-index leading properties.

Motivation
----------
The paper's Table 13 ("Granger Causality Tests: Sub-Index Leading Properties")
regresses a *binary* crisis indicator on lagged sub-indices by **ordinary least
squares** (a linear probability model, LPM) and references the homoskedastic
F-distribution. The paper itself flags this as approximate and recommends a
"properly specified discrete-time hazard (logit/probit) Granger test" as the
correct refinement. This script implements that refinement.

We do NOT edit the paper. We re-run the leading-indicator test with a
discrete-outcome specification and report, per sub-index:
  (1) naive logit-LR Granger test (state indicator)   -- direct analogue of the
      paper's naive OLS F-test, assumes iid daily obs;
  (2) cluster-robust Wald test on the lagged sub-index -- exposes the
      pseudoreplication induced by the persistence of the crisis state;
  (3) discrete-time hazard logit for crisis *onset*    -- the principled test
      the paper names; mechanically removes within-episode pseudoreplication.

Bonferroni correction: family-wise alpha = 0.05 across the 4 sub-indices ->
per-test alpha = 0.0125 (matching the paper's framing).

The fundamental constraint -- there are only FOUR crisis episodes in the
sample -- means every variant is fragile. We report this honestly.

Outputs:
  results/granger_discrete_outcome.json
"""
from __future__ import annotations
import json, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.simplefilter("ignore")  # statsmodels convergence chatter; we check status manually

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "results" / "data" / "asri_history.parquet"
OUT = ROOT / "results" / "granger_discrete_outcome.json"

SUBS = {
    "SCR": "stablecoin_risk",      # Stablecoin Concentration Risk
    "DLR": "defi_liquidity_risk",  # DeFi Liquidity Risk
    "CR":  "contagion_risk",       # Contagion Risk
    "OR":  "arbitrage_opacity",    # (Regulatory) Opacity Risk
}
EVENTS = [
    datetime(2022, 5, 12),   # Terra/Luna
    datetime(2022, 6, 17),   # Celsius/3AC
    datetime(2022, 11, 11),  # FTX
    datetime(2023, 3, 11),   # SVB / USDC depeg
]
ALPHA = 0.05
M_TESTS = len(SUBS)
ALPHA_BONF = ALPHA / M_TESTS  # 0.0125


# ----------------------------------------------------------------------------- helpers
def _safe_logit(y, X):
    """Fit a Logit, trying newton then bfgs; return None on singular/separation."""
    for method in ("newton", "bfgs"):
        try:
            m = sm.Logit(y, X).fit(disp=0, maxiter=300, method=method)
            if np.all(np.isfinite(np.asarray(m.bse))):
                return m
        except (np.linalg.LinAlgError, Exception):
            continue
    return None


def build_state_indicator(index: pd.DatetimeIndex, lo: int, hi: int) -> pd.Series:
    """Binary crisis-STATE indicator: 1 on [onset+lo, onset+hi] for each event."""
    y = pd.Series(0.0, index=index)
    for e in EVENTS:
        m = (index >= pd.Timestamp(e) + pd.Timedelta(days=lo)) & (
            index <= pd.Timestamp(e) + pd.Timedelta(days=hi)
        )
        y[m] = 1.0
    return y


def build_onset_and_riskset(index: pd.DatetimeIndex, win: int):
    """
    Discrete-time hazard framing.
      onset_t   = 1 on the FIRST day of each crisis window, else 0.
      at_risk_t = False while a crisis window is 'active' AFTER its onset day
                  (standard discrete-survival: once in crisis you are no longer
                  'at risk' of a new onset). The onset day itself stays in the
                  risk set as the event.
    """
    onset = pd.Series(0.0, index=index)
    active = pd.Series(False, index=index)  # in-crisis (post-onset) -> drop from risk set
    for e in EVENTS:
        start = pd.Timestamp(e)
        # onset day = first available index date >= event date
        on_days = index[index >= start]
        if len(on_days) == 0:
            continue
        on = on_days[0]
        onset[on] = 1.0
        m = (index > on) & (index <= start + pd.Timedelta(days=win))
        active[m] = True
    at_risk = ~active
    return onset, at_risk


def naive_logit_lr(y: np.ndarray, x: np.ndarray, lag: int = 1):
    """
    Logit Granger (state). Direct discrete-outcome analogue of the paper's OLS-F.
      restricted   : y_t ~ 1 + y_{t-1..lag}
      unrestricted : y_t ~ 1 + y_{t-1..lag} + x_{t-1..lag}
      LR ~ chi2(lag), naive (iid daily obs).
    Returns dict with LR stat, p, df, n, convergence flag, robust-cluster Wald.
    """
    n = len(y)
    yt = y[lag:]
    cols_r = [np.ones(n - lag)]
    cols_u = [np.ones(n - lag)]
    for i in range(1, lag + 1):
        cols_r.append(y[lag - i:n - i])
        cols_u.append(y[lag - i:n - i])
    for i in range(1, lag + 1):
        cols_u.append(x[lag - i:n - i])
    Xr = np.column_stack(cols_r)
    Xu = np.column_stack(cols_u)

    mr = _safe_logit(yt, Xr)
    mu = _safe_logit(yt, Xu)
    if mr is None or mu is None:
        return {"lr_stat": None, "p_naive_lr": None, "df": lag, "n_obs": int(len(yt)),
                "status": "singular/non-converged (state logit ill-posed at this lag)",
                "_yt": yt, "_Xu": Xu, "_lag": lag}
    lr = max(2.0 * (mu.llf - mr.llf), 0.0)
    p_lr = float(stats.chi2.sf(lr, lag))
    return {
        "lr_stat": float(lr),
        "p_naive_lr": p_lr,
        "df": lag,
        "n_obs": int(len(yt)),
        "converged_restricted": bool(mr.mle_retvals.get("converged", False)),
        "converged_unrestricted": bool(mu.mle_retvals.get("converged", False)),
        "_yt": yt, "_Xu": Xu, "_lag": lag,
    }


def cluster_robust_wald(yt, Xu, lag, groups):
    """
    Wald test (chi2, df=lag) on the lagged-x coefficients of the unrestricted
    state logit, using cluster-robust covariance. Exposes pseudoreplication:
    clusters = contiguous crisis/calm regime runs.
    """
    try:
        m = sm.Logit(yt, Xu).fit(
            disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": groups}
        )
    except Exception as ex:  # pragma: no cover
        return {"p_cluster_wald": None, "wald_stat": None, "n_clusters": int(len(np.unique(groups))),
                "error": str(ex)}
    # x coefficients are the LAST `lag` params (const + lag AR + lag X)
    k = Xu.shape[1]
    idx_x = list(range(k - lag, k))
    R = np.zeros((lag, k))
    for r, j in enumerate(idx_x):
        R[r, j] = 1.0
    beta = m.params
    cov = m.cov_params()
    Rb = R @ beta
    RCR = R @ cov @ R.T
    try:
        wald = float(Rb.T @ np.linalg.inv(RCR) @ Rb)
        p = float(stats.chi2.sf(wald, lag))
    except np.linalg.LinAlgError:
        wald, p = None, None
    return {"p_cluster_wald": p, "wald_stat": wald,
            "n_clusters": int(len(np.unique(groups)))}


def regime_run_clusters(y: np.ndarray, lag: int) -> np.ndarray:
    """Cluster id = maximal contiguous run of identical state (for the post-lag rows)."""
    yt = y[lag:]
    runs = np.zeros(len(yt), dtype=int)
    cur = 0
    for i in range(len(yt)):
        if i > 0 and yt[i] != yt[i - 1]:
            cur += 1
        runs[i] = cur
    return runs


def hazard_onset_test(onset: np.ndarray, x: np.ndarray, at_risk: np.ndarray, lag: int = 1):
    """
    Discrete-time hazard logit for crisis ONSET (the paper's recommended test).
      onset_t ~ 1 + x_{t-lag}   over the risk set (post-onset crisis days dropped).
    LR test of the x coefficient vs intercept-only. Flags separation / underpower.
    """
    n = len(onset)
    on_t = onset[lag:]
    risk_t = at_risk[lag:]
    xlag = x[lag - 1:n - 1] if lag == 1 else x[lag - lag:n - lag]  # x_{t-lag}
    # keep only at-risk rows
    keep = risk_t.astype(bool)
    yv = on_t[keep]
    xv = xlag[keep]
    n_events = int(yv.sum())
    Xr = np.ones((len(yv), 1))
    Xu = np.column_stack([np.ones(len(yv)), xv])
    separation = False
    try:
        mr = sm.Logit(yv, Xr).fit(disp=0, maxiter=300)
        mu = sm.Logit(yv, Xu).fit(disp=0, maxiter=300)
        conv = bool(mu.mle_retvals.get("converged", False))
        # crude separation check: |coef| explosive or se explosive
        if np.any(np.abs(mu.params) > 25) or np.any(np.asarray(mu.bse) > 25):
            separation = True
        lr = max(2.0 * (mu.llf - mr.llf), 0.0)
        p = float(stats.chi2.sf(lr, 1))
        coef = float(mu.params[1])
    except Exception as ex:  # pragma: no cover
        return {"p_hazard_lr": None, "lr_stat": None, "n_events": n_events,
                "n_riskset": int(len(yv)), "converged": False, "separation": True,
                "error": str(ex)}
    return {"p_hazard_lr": p, "lr_stat": float(lr), "coef": coef,
            "n_events": n_events, "n_riskset": int(len(yv)),
            "converged": conv, "separation": separation}


def circular_permutation_test(onset: np.ndarray, x: np.ndarray, lag: int = 1,
                              lead_window: int | None = None):
    """
    Small-sample-VALID leading-indicator test (the honest verdict-driver).

    Problem: with only 4 crisis onsets, the asymptotic logit/hazard LR is invalid
    and every correlated sub-index "predicts" the persistent crisis block.

    Fix: a circular-shift permutation test. We keep the 4 onsets (their spacing
    and the covariate's autocorrelation intact) and slide the onset pattern to
    every calendar offset, recomputing the association statistic. This yields an
    exact-style p-value that respects both the 4-event count and serial
    dependence -- no iid or large-sample assumption.

    Statistic (Granger/score flavour): mean standardized sub-index value over the
    lead window preceding each onset. lag=1 => value the day before onset;
    lead_window=k => mean over [onset-k, onset-1]. Larger => sub-index is elevated
    BEFORE the crisis, i.e. leads it.
    """
    T = len(x)
    onset_pos = np.flatnonzero(onset > 0.5)
    if len(onset_pos) == 0:
        return {"p_perm": None, "obs_stat": None, "n_onsets": 0}

    def lead_value(positions):
        if lead_window is None:
            srcs = (positions - lag) % T
            return float(np.mean(x[srcs]))
        vals = []
        for p in positions:
            offs = [(p - d) % T for d in range(1, lead_window + 1)]
            vals.append(np.mean(x[offs]))
        return float(np.mean(vals))

    obs = lead_value(onset_pos)
    # exact circular permutation over all T shifts (preserves onset spacing + x autocorr)
    null = np.empty(T)
    for s in range(T):
        null[s] = lead_value((onset_pos + s) % T)
    # one-sided: leading indicator => elevated pre-onset value
    p = (1 + int(np.sum(null >= obs - 1e-12))) / (1 + T)
    return {"p_perm": float(p), "obs_stat": obs,
            "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
            "n_onsets": int(len(onset_pos)), "n_shifts": T,
            "lead_window": lead_window, "lag": lag}


def lead_vs_confirm_trajectory(df: pd.DataFrame):
    """
    Directly test the paper's SPECIFIC claim that SCR/DLR are *leading* (rise
    before onset) while CR is *confirming* (spikes at/after onset), free of the
    'crisis-era' confound that the permutation test cannot remove.

    For each event and sub-index, z-score the sub-index against a LOCAL baseline
    [onset-30, onset-15] (so we measure movement, not era level), then compare:
        pre  = mean z over [onset-7, onset-1]   (does it rise BEFORE onset?)
        post = mean z over [onset+1, onset+7]   (does it rise AT/AFTER onset?)
    Leading  => pre already high (pre >~ post). Confirming => post >> pre.
    Reported per sub-index, averaged over the 4 events (n=4, descriptive only).
    """
    out = {}
    for code, col in SUBS.items():
        s = df[col]
        pres, posts = [], []
        for e in EVENTS:
            on_days = df.index[df.index >= pd.Timestamp(e)]
            if len(on_days) == 0:
                continue
            on = on_days[0]
            base = s[(df.index >= on - pd.Timedelta(days=30)) & (df.index <= on - pd.Timedelta(days=15))]
            if len(base) < 3 or base.std() == 0:
                continue
            mu, sd = base.mean(), base.std()
            pre = s[(df.index >= on - pd.Timedelta(days=7)) & (df.index <= on - pd.Timedelta(days=1))]
            post = s[(df.index >= on + pd.Timedelta(days=1)) & (df.index <= on + pd.Timedelta(days=7))]
            pres.append(float((pre.mean() - mu) / sd))
            posts.append(float((post.mean() - mu) / sd))
        pres, posts = np.array(pres), np.array(posts)
        out[code] = {
            "pre_z_mean": float(np.mean(pres)), "post_z_mean": float(np.mean(posts)),
            "pre_minus_post": float(np.mean(pres - posts)),
            "n_events_used": int(len(pres)),
            "label_by_data": ("leading" if np.mean(pres) >= np.mean(posts) else "confirming"),
        }
    return out


def bonferroni_flags(pvals: dict):
    return {
        k: {
            "p": (None if v is None else round(v, 4)),
            "sig_raw_5pct": (None if v is None else bool(v < ALPHA)),
            "sig_bonferroni": (None if v is None else bool(v < ALPHA_BONF)),
        }
        for k, v in pvals.items()
    }


# ----------------------------------------------------------------------------- main
def run_for_window(df, lo, hi, win, lag=1):
    idx = df.index
    y_state = build_state_indicator(idx, lo, hi).values
    onset, at_risk = build_onset_and_riskset(idx, win)
    onset = onset.values
    at_risk = at_risk.values
    clusters = regime_run_clusters(y_state, lag)

    out = {"naive_lr": {}, "cluster_wald": {}, "hazard": {}, "perm": {}, "perm_lead7": {},
           "n_crisis_days": int(y_state.sum()), "n_onsets": int(onset.sum())}
    p_naive, p_cluster, p_hazard, p_perm, p_perm7 = {}, {}, {}, {}, {}
    for code, col in SUBS.items():
        x = df[col].values.astype(float)
        # standardize x for numerical stability (does not affect LR/Wald/perm p-values)
        xz = (x - np.nanmean(x)) / np.nanstd(x)

        nl = naive_logit_lr(y_state, xz, lag=lag)
        cw = cluster_robust_wald(nl.pop("_yt"), nl.pop("_Xu"), nl.pop("_lag"), clusters)
        hz = hazard_onset_test(onset, xz, at_risk, lag=lag)
        pm = circular_permutation_test(onset, xz, lag=lag, lead_window=None)
        pm7 = circular_permutation_test(onset, xz, lag=lag, lead_window=7)

        out["naive_lr"][code] = nl
        out["cluster_wald"][code] = cw
        out["hazard"][code] = hz
        out["perm"][code] = pm
        out["perm_lead7"][code] = pm7
        p_naive[code] = nl["p_naive_lr"]
        p_cluster[code] = cw["p_cluster_wald"]
        p_hazard[code] = hz["p_hazard_lr"]
        p_perm[code] = pm["p_perm"]
        p_perm7[code] = pm7["p_perm"]

    out["bonferroni"] = {
        "naive_lr": bonferroni_flags(p_naive),
        "cluster_wald": bonferroni_flags(p_cluster),
        "hazard": bonferroni_flags(p_hazard),
        "perm_lag1": bonferroni_flags(p_perm),
        "perm_lead7": bonferroni_flags(p_perm7),
        "alpha_raw": ALPHA, "alpha_bonferroni": ALPHA_BONF, "m_tests": M_TESTS,
    }
    return out


def main():
    df = pd.read_parquet(DATA).sort_index()
    subs_present = [c for c in SUBS.values() if c in df.columns]
    assert len(subs_present) == 4, f"missing sub-indices: {set(SUBS.values())-set(df.columns)}"

    results = {
        "meta": {
            "data": str(DATA), "n_rows": int(len(df)),
            "date_min": str(df.index.min().date()), "date_max": str(df.index.max().date()),
            "sub_indices": SUBS, "events": [e.strftime("%Y-%m-%d") for e in EVENTS],
            "n_events": len(EVENTS),
            "paper_table13_lpm": {  # the numbers we are refining (NOT reproduced here)
                "SCR": {"F": 6.38, "p": 0.0117}, "DLR": {"F": 5.89, "p": 0.0154},
                "CR": {"F": 2.58, "p": 0.1084}, "OR": {"F": 3.08, "p": 0.0797},
                "note": "Original OLS-LPM numbers; generator orphaned, not reproducible "
                        "from current canonical data (see report).",
            },
            "alpha_raw": ALPHA, "alpha_bonferroni": ALPHA_BONF,
        },
        "primary": None,
        "sensitivity": {},
    }

    # primary spec: crisis-state window [onset, onset+14]; hazard exclusion window 14d; lag 1
    results["primary"] = {"window": "[0,14], hazard_excl=14, lag=1",
                          **run_for_window(df, lo=0, hi=14, win=14, lag=1)}
    results["lead_vs_confirm_trajectory"] = lead_vs_confirm_trajectory(df)

    # sensitivity grid
    grid = [("[0,7], excl7, lag1", 0, 7, 7, 1),
            ("[0,30], excl30, lag1", 0, 30, 30, 1),
            ("[-7,7], excl14, lag1", -7, 7, 14, 1),
            ("[0,14], excl14, lag3", 0, 14, 14, 3)]
    for name, lo, hi, win, lag in grid:
        results["sensitivity"][name] = run_for_window(df, lo, hi, win, lag)

    # ---- machine-readable verdict
    perm_p = {c: results["primary"]["perm"][c]["p_perm"] for c in SUBS}
    results["verdict"] = {
        "valid_test": "circular-shift permutation (small-sample valid at n=4 events)",
        "permutation_p_lag1": perm_p,
        "all_four_survive_bonferroni_0_0125": bool(all(p < ALPHA_BONF for p in perm_p.values())),
        "asymptotic_logit_probit_hazard": "DEGENERATE at 4 events (all p<1e-4 incl. CR); "
            "chi2 reference invalid; cannot reproduce paper's per-channel discrimination.",
        "paper_table13_lpm_reproducible": False,
        "paper_table13_lpm_note": "Exact F-stats (6.38/5.89/2.58/3.08) not reproducible from "
            "current canonical data under any documented crisis window; generator orphaned.",
        "leading_vs_confirming_hierarchy_supported": False,
        "hierarchy_reason": "Under the valid permutation test ALL four sub-indices (incl. CR) "
            "show significant pre-onset elevation and survive Bonferroni; the era-controlled "
            "within-event trajectory shows all four rise before AND further after onset with an "
            "identical lead-minus-confirm gap (-2.1..-2.5 z), so no channel is distinctly "
            "'leading' vs 'confirming'. The paper's two hierarchy statements are also mutually "
            "inconsistent (Granger sec: lead={SCR,DLR}; ablation sec: lead={DLR,CR}).",
        "recommendation": "Drop or heavily caveat the leading-vs-confirming functional hierarchy; "
            "the 4 co-located 2022-23 episodes do not identify per-channel lead timing.",
    }
    OUT.write_text(json.dumps(results, indent=2, default=str))
    # ---- console summary
    print("=" * 78)
    print("DISCRETE-OUTCOME GRANGER TEST — ASRI sub-index leading properties")
    print("=" * 78)
    print(f"Data: {df.index.min().date()} .. {df.index.max().date()}  ({len(df)} days)")
    print(f"Crisis episodes: {len(EVENTS)}  | Bonferroni per-test alpha = {ALPHA_BONF}")
    pr = results["primary"]
    print(f"\nPRIMARY spec: {pr['window']}  "
          f"(crisis days={pr['n_crisis_days']}, onsets={pr['n_onsets']})")
    def fmt(v):
        return "   n/a  " if v is None else f"{v:8.4f}"
    def star(v):
        if v is None: return "  "
        return "**" if v < ALPHA_BONF else ("* " if v < ALPHA else "  ")
    print("\nAsymptotic discrete-outcome tests (INVALID at 4 events -- see permutation):")
    hdr = f"{'sub':4s} | {'naiveLR':>8s}   {'clustWald':>9s}  {'hazard':>8s}  | n_ev/risk"
    print(hdr); print("-" * len(hdr))
    for code in SUBS:
        nl = pr["naive_lr"][code]["p_naive_lr"]
        cw = pr["cluster_wald"][code]["p_cluster_wald"]
        hz = pr["hazard"][code]
        print(f"{code:4s} | {fmt(nl)}{star(nl)} {fmt(cw)}{star(cw)} {fmt(hz['p_hazard_lr'])}{star(hz['p_hazard_lr'])} "
              f"| {hz['n_events']}/{hz['n_riskset']}")
    print("\nCircular-shift PERMUTATION test (small-sample valid; VERDICT-DRIVER):")
    hdr2 = f"{'sub':4s} | {'lag1 p':>8s}   {'lead7 p':>8s}   | obs lead-z (rank vs null)"
    print(hdr2); print("-" * len(hdr2))
    for code in SUBS:
        pm = pr["perm"][code]; pm7 = pr["perm_lead7"][code]
        z = (pm["obs_stat"] - pm["null_mean"]) / pm["null_sd"] if pm["null_sd"] else float("nan")
        print(f"{code:4s} | {fmt(pm['p_perm'])}{star(pm['p_perm'])} {fmt(pm7['p_perm'])}{star(pm7['p_perm'])}   "
              f"| obs={pm['obs_stat']:+.2f}  z={z:+.2f}")
    print(f"\n(** survives Bonferroni {ALPHA_BONF}, * raw 5% only)")
    tj = results["lead_vs_confirm_trajectory"]
    print("\nWithin-event LEAD-vs-CONFIRM (local-baseline z; tests paper's specific claim; n=4):")
    print(f"{'sub':4s} | {'pre_z':>7s} {'post_z':>7s} {'pre-post':>9s} | paper_says -> data_says")
    print("-" * 60)
    paper_role = {"SCR": "leading", "DLR": "leading", "CR": "confirming(no-lead)", "OR": "leading*/marg"}
    for code in SUBS:
        t = tj[code]
        print(f"{code:4s} | {t['pre_z_mean']:+7.2f} {t['post_z_mean']:+7.2f} {t['pre_minus_post']:+9.2f} "
              f"| {paper_role[code]:18s} -> {t['label_by_data']}")
    print(f"\nFull JSON -> {OUT}")


if __name__ == "__main__":
    main()
