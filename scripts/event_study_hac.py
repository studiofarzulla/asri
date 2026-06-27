#!/usr/bin/env python3
"""
Recompute ASRI event-study CAS significance with Newey-West (HAC) standard
errors, replacing the paper's false independence assumption.

Reproduces the paper's naive (IID) t/p, then:
  (1) Ljung-Box on the abnormal-signal (residual) series -- estimation window,
      event window, and combined -- to report the TRUTH that replaces the
      non-existent "Ljung-Box p>0.10 for all events" claim.
  (2) Recomputes each event's CAS t/p via Newey-West HAC SEs (OLS of the
      event-window abnormal series on a constant; the intercept HAC t-stat
      IS the HAC t-stat for CAS, since CAS = n * mean_AS).

Faithful to paper_v2 profile: estimation_window=(-90,-31), event_window=(-30,10).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asri.validation.event_study import (  # noqa: E402
    compute_cumulative_abnormal_signal,
    get_event_study_config,
)

EVENTS = [
    ("Terra/Luna", datetime(2022, 5, 12)),
    ("Celsius/3AC", datetime(2022, 6, 17)),
    ("FTX Collapse", datetime(2022, 11, 11)),
    ("SVB Crisis", datetime(2023, 3, 11)),
]

HAC_LAGS = [3, 10, 20, 30]   # 3 ~ Newey-West auto floor(4*(T/100)^(2/9)); 20 = primary
PRIMARY_LAG = 20
LB_LAGS = [5, 10, 20]


def load_asri() -> pd.Series:
    p = PROJECT_ROOT / "results" / "data" / "asri_history.parquet"
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
    return df["asri"].sort_index()


def ar1(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return float("nan")
    return float(np.sum(x[1:] * x[:-1]) / denom)


def ljung_box(series: np.ndarray, lags: list[int]) -> dict:
    s = pd.Series(np.asarray(series, float)).dropna()
    out = {}
    for L in lags:
        if len(s) <= L + 1:
            out[str(L)] = {"stat": None, "pvalue": None, "note": "n<=lag"}
            continue
        res = acorr_ljungbox(s, lags=[L], return_df=True)
        out[str(L)] = {
            "stat": float(res["lb_stat"].iloc[0]),
            "pvalue": float(res["lb_pvalue"].iloc[0]),
        }
    return out


def main() -> None:
    asri = load_asri()
    cfg = get_event_study_config("paper_v2")
    est_w = cfg.estimation_window      # (-90, -31)
    evt_w = cfg.event_window           # (-30, 10)

    results = []
    for name, dt in EVENTS:
        ev = pd.Timestamp(dt)

        # --- Reproduce the paper's naive abnormal signal / CAS / t / p ---
        abnormal_evt, cas, t_naive_paper, p_naive_paper = compute_cumulative_abnormal_signal(
            asri, dt, est_w, evt_w
        )
        n_evt = len(abnormal_evt)

        # Estimation-window residual series (paper's literal Ljung-Box claim is about THIS)
        est_start = ev + pd.Timedelta(days=est_w[0])
        est_end = ev + pd.Timedelta(days=est_w[1])
        est_data = asri[(asri.index >= est_start) & (asri.index <= est_end)]
        est_resid = (est_data - est_data.mean()).values

        # Combined window residual (-90 .. +10), de-meaned by estimation mean
        comb = asri[(asri.index >= est_start) & (asri.index <= ev + pd.Timedelta(days=evt_w[1]))]
        comb_resid = (comb - est_data.mean()).values

        # --- Ljung-Box truth ---
        lb = {
            "estimation_window_resid": ljung_box(est_resid, LB_LAGS),
            "event_window_abnormal": ljung_box(abnormal_evt.values, LB_LAGS),
            "combined_window_resid": ljung_box(comb_resid, LB_LAGS),
        }
        ar1_est = ar1(est_resid)
        ar1_evt = ar1(abnormal_evt.values)

        # --- OLS of event-window abnormal series on a constant ---
        # intercept = mean_AS; CAS = n * mean_AS.
        y = abnormal_evt.values.astype(float)
        X = np.ones((len(y), 1))
        ols = sm.OLS(y, X).fit()
        mean_as = float(ols.params[0])
        # IID (nonrobust) baseline on the SAME event-window series (apples-to-apples)
        t_iid = float(ols.tvalues[0])
        p_iid = float(ols.pvalues[0])

        hac = {}
        for L in HAC_LAGS:
            if L >= n_evt:
                hac[str(L)] = {"se_mean": None, "t": None, "p": None, "note": "lag>=n"}
                continue
            r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
            se_mean = float(r.bse[0])
            t_hac = float(r.tvalues[0])
            # CAS t == mean t (CAS = n*mean, SE(CAS)=n*SE(mean)); p two-sided normal (HAC -> z)
            p_hac = float(r.pvalues[0])
            hac[str(L)] = {
                "se_mean": se_mean,
                "se_cas": se_mean * n_evt,
                "t": t_hac,
                "p": p_hac,
                "sig_5pct": bool(p_hac < 0.05),
                "sig_1pct": bool(p_hac < 0.01),
            }

        prim = hac[str(PRIMARY_LAG)]
        results.append({
            "name": name,
            "event_date": dt.strftime("%Y-%m-%d"),
            "n_event_obs": n_evt,
            "cas": round(float(cas), 3),
            "mean_abnormal": round(mean_as, 4),
            "ar1_estimation_window": round(ar1_est, 4),
            "ar1_event_window": round(ar1_evt, 4),
            "naive_paper": {  # paper's current numbers (est-window sigma, df=n-1)
                "t": round(float(t_naive_paper), 3),
                "p": float(p_naive_paper),
                "sig_1pct": bool(p_naive_paper < 0.01),
            },
            "naive_iid_eventwindow": {  # OLS homoskedastic on event window (bridge)
                "t": round(t_iid, 3),
                "p": float(p_iid),
            },
            "ljung_box": lb,
            "hac": hac,
            "primary_hac_lag": PRIMARY_LAG,
            "hac_primary_t": None if prim["t"] is None else round(prim["t"], 3),
            "hac_primary_p": None if prim["p"] is None else round(prim["p"], 4),
            "survives_5pct_primary": None if prim.get("sig_5pct") is None else prim["sig_5pct"],
        })

    survivors = [r["name"] for r in results if r.get("survives_5pct_primary")]
    fails = [r["name"] for r in results if r.get("survives_5pct_primary") is False]

    out = {
        "method": "Newey-West HAC on event-window abnormal-signal series (OLS on constant)",
        "profile": "paper_v2",
        "estimation_window": list(est_w),
        "event_window": list(evt_w),
        "n_event_obs": results[0]["n_event_obs"],
        "primary_hac_lag": PRIMARY_LAG,
        "hac_lag_sweep": HAC_LAGS,
        "ljung_box_lags": LB_LAGS,
        "note": (
            "CAS t-stat == mean-abnormal t-stat (CAS=n*mean, SE(CAS)=n*SE(mean)). "
            "HAC p-values use the normal (z) reference per statsmodels HAC default. "
            "Paper's claim 'Ljung-Box p>0.10 for all events' is refuted below: the "
            "abnormal/residual series are strongly serially correlated (AR1>0.9)."
        ),
        "events": results,
        "survivors_5pct_primary": survivors,
        "fail_5pct_primary": fails,
    }

    out_path = PROJECT_ROOT / "results" / "event_study_hac.json"
    out_path.write_text(json.dumps(out, indent=2))

    # ---- console report ----
    print("=" * 100)
    print("ASRI EVENT STUDY -- HAC (Newey-West) RECOMPUTATION  [paper_v2, evt=(-30,10), n=%d]" % results[0]["n_event_obs"])
    print("=" * 100)
    print(f"{'Event':<14}{'CAS':>9}{'AR1est':>8}{'AR1evt':>8} | "
          f"{'naive t':>8}{'naive p':>11} | {'HAC t(L=20)':>12}{'HAC p':>10}  survive5%")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<14}{r['cas']:>9.1f}{r['ar1_estimation_window']:>8.3f}"
              f"{r['ar1_event_window']:>8.3f} | {r['naive_paper']['t']:>8.2f}"
              f"{r['naive_paper']['p']:>11.2e} | {r['hac_primary_t']:>12.2f}"
              f"{r['hac_primary_p']:>10.3f}   {r['survives_5pct_primary']}")
    print("-" * 100)
    print("HAC t across lag sweep (L = 3 / 10 / 20 / 30):")
    for r in results:
        row = []
        for L in HAC_LAGS:
            h = r["hac"][str(L)]
            row.append("n/a" if h["t"] is None else f"{h['t']:.2f}")
        print(f"  {r['name']:<14} " + "  ".join(f"L{L}={v}" for L, v in zip(HAC_LAGS, row)))
    print("-" * 100)
    print("Ljung-Box p-values (event-window abnormal series), lags 5/10/20:")
    for r in results:
        lb = r["ljung_box"]["event_window_abnormal"]
        print(f"  {r['name']:<14} " + "  ".join(
            f"L{L}: p={lb[str(L)]['pvalue']:.2e}" if lb[str(L)]["pvalue"] is not None else f"L{L}: n/a"
            for L in LB_LAGS))
    print("Ljung-Box p-values (ESTIMATION-window residuals -- paper's literal claim), lags 5/10/20:")
    for r in results:
        lb = r["ljung_box"]["estimation_window_resid"]
        print(f"  {r['name']:<14} " + "  ".join(
            f"L{L}: p={lb[str(L)]['pvalue']:.2e}" if lb[str(L)]["pvalue"] is not None else f"L{L}: n/a"
            for L in LB_LAGS))
    print("-" * 100)
    print(f"SURVIVE at 5% (HAC L=20): {survivors}")
    print(f"FAIL at 5%  (HAC L=20): {fails}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
