#!/usr/bin/env python3
"""
Reproducible PLACEBO test (HAC inference of record) + Chow/CUSUM structural
break test on the CLEAN, code-consistent ASRI series.

Replaces:
  * tab:placebo  -- the unreproducible naive placebo (mean|t|=1.24, max|t|=2.18,
    compared against the *naive* crisis benchmark 23.7/32.6). Here the placebo
    |t| is computed under the SAME Newey-West HAC (L=20) procedure as the crisis
    events, and compared against the crisis HAC (fixed-b) t-stats -- the
    inference of record -- not the invalid naive t-stats.
  * tab:robustness Chow / CUSUM rows.

Placebo methodology (matches paper design, made reproducible):
  - Candidate dates drawn from 2021-01..2024-12, excluding +/-90d around each
    crisis and the first/last 90d of the sample.
  - For each placebo date: compute CAS over the SAME paper_v2 windows
    (estimation (-90,-31), event (-30,10)); then HAC t at L=20 via OLS of the
    event-window abnormal series on a constant (intercept HAC t == CAS HAC t).
  - Report mean|t| and max|t| of the placebo HAC distribution, plus how many
    placebo dates clear the fixed-b 5% critical |t| ~ 3.52 and the naive N(0,1)
    5% critical |t| = 1.96.
  - Empirical placebo p-value for each crisis event: fraction of placebo |HAC t|
    >= the crisis |HAC t|.
Two placebo panels: (A) paper-style 10 dates (seed 42); (B) ALL eligible dates
(seed-free null) for a robust empirical p-value.
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asri.validation.event_study import compute_cumulative_abnormal_signal  # noqa: E402
from asri.validation.robustness import structural_break_test  # noqa: E402

EST_W = (-90, -31)
EVT_W = (-30, 10)
HAC_LAG = 20
# fixed-b (Kiefer-Vogelsang, Bartlett) critical |t| at b = L/n ~ 0.49 (paper: b~0.51)
FIXB_CRIT_5 = 3.52     # paper-stated 5% two-sided fixed-b critical |t|
FIXB_CRIT_BONF = 5.73  # paper-stated Bonferroni-0.0125 fixed-b critical |t|
NAIVE_CRIT_5 = 1.96

CRISES = [
    ("Terra/Luna",   datetime(2022, 5, 12)),
    ("Celsius/3AC",  datetime(2022, 6, 17)),
    ("FTX Collapse", datetime(2022, 11, 11)),
    ("SVB Crisis",   datetime(2023, 3, 11)),
]


def load_asri() -> pd.Series:
    df = pd.read_parquet(PROJECT_ROOT / "results" / "data" / "asri_history.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
    return df["asri"].sort_index()


def hac_t_for_date(asri: pd.Series, dt) -> dict | None:
    """CAS + naive t (from module) + HAC t(L=20) for one event/placebo date."""
    try:
        abnormal, cas, t_naive, p_naive = compute_cumulative_abnormal_signal(
            asri, dt, EST_W, EVT_W
        )
    except (ValueError, KeyError):
        return None
    y = abnormal.values.astype(float)
    n = len(y)
    if n <= HAC_LAG + 1:
        return None
    X = np.ones((n, 1))
    r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAG})
    return {
        "cas": float(cas),
        "n": int(n),
        "t_naive": float(t_naive),
        "t_hac": float(r.tvalues[0]),
        "p_hac_normal": float(r.pvalues[0]),
    }


def crisis_panel(asri: pd.Series) -> list[dict]:
    out = []
    for name, dt in CRISES:
        row = hac_t_for_date(asri, dt)
        row["name"] = name
        row["date"] = dt.strftime("%Y-%m-%d")
        out.append(row)
    return out


def eligible_dates(asri: pd.Series) -> list[pd.Timestamp]:
    idx = asri.index
    lo = idx.min() + pd.Timedelta(days=90)
    hi = idx.max() - pd.Timedelta(days=90)
    sel_end = pd.Timestamp("2024-12-31")
    cand = []
    crisis_ts = [pd.Timestamp(d) for _, d in CRISES]
    for d in idx:
        if d < lo or d > min(hi, sel_end):
            continue
        if d < pd.Timestamp("2021-01-01"):
            continue
        if any(abs((d - c).days) < 90 for c in crisis_ts):
            continue
        cand.append(d)
    return cand


def run_placebo(asri: pd.Series, dates: list, label: str) -> dict:
    rows = []
    for d in dates:
        r = hac_t_for_date(asri, pd.Timestamp(d))
        if r is not None:
            rows.append(r)
    t_hac = np.array([abs(r["t_hac"]) for r in rows])
    t_hac_signed = np.array([r["t_hac"] for r in rows])
    t_naive = np.array([abs(r["t_naive"]) for r in rows])
    return {
        "label": label,
        "n_dates": len(rows),
        "_t_hac_signed": t_hac_signed,
        "hac": {
            "mean_abs_t": float(np.mean(t_hac)),
            "max_abs_t": float(np.max(t_hac)),
            "median_abs_t": float(np.median(t_hac)),
            "q95_abs_t": float(np.quantile(t_hac, 0.95)),
            "n_sig_fixb_5": int(np.sum(t_hac > FIXB_CRIT_5)),
            "frac_sig_fixb_5": float(np.mean(t_hac > FIXB_CRIT_5)),
            "n_sig_naive_5": int(np.sum(t_hac > NAIVE_CRIT_5)),
            "frac_sig_naive_5": float(np.mean(t_hac > NAIVE_CRIT_5)),
        },
        "naive": {
            "mean_abs_t": float(np.mean(t_naive)),
            "max_abs_t": float(np.max(t_naive)),
        },
        "_t_hac_abs": t_hac,
    }


def main() -> None:
    asri = load_asri()
    print(f"Loaded ASRI: {len(asri)} rows, {asri.index.min().date()} -> {asri.index.max().date()}")

    # ---- crisis benchmark (HAC inference of record) ----
    crisis = crisis_panel(asri)
    print("\n=== CRISIS BENCHMARK (clean series, HAC L=20) ===")
    for r in crisis:
        print(f"  {r['name']:<12} CAS={r['cas']:>8.1f}  naive|t|={abs(r['t_naive']):>7.2f}"
              f"  HAC|t|={abs(r['t_hac']):>6.2f}  fixb5%(>3.52)={'Y' if abs(r['t_hac'])>FIXB_CRIT_5 else 'N'}")
    crisis_hac = np.array([abs(r["t_hac"]) for r in crisis])
    fixb_sig = [abs(r["t_hac"]) for r in crisis if abs(r["t_hac"]) > FIXB_CRIT_5]
    crisis_mean_all = float(np.mean(crisis_hac))
    crisis_max_all = float(np.max(crisis_hac))
    crisis_mean_sig = float(np.mean(fixb_sig)) if fixb_sig else float("nan")
    print(f"  ALL4  HAC|t| mean={crisis_mean_all:.3f} max={crisis_max_all:.3f}")
    print(f"  fixb-sig(3/4) HAC|t| mean={crisis_mean_sig:.3f} (events>3.52)")

    # ---- placebo A: paper-style 10 dates ----
    cand = eligible_dates(asri)
    print(f"\nEligible placebo dates: {len(cand)} (2021-01..2024-12, excl +/-90d crises & edges)")
    rng = np.random.default_rng(42)
    pick10 = rng.choice(np.array(cand, dtype="datetime64[ns]"), size=10, replace=False)
    pick10 = sorted(pd.Timestamp(x) for x in pick10)
    pa = run_placebo(asri, pick10, "paper_style_10_seed42")

    # ---- placebo B: all eligible dates (seed-free null) ----
    pb = run_placebo(asri, cand, "all_eligible_null")

    # ---- empirical placebo p-values per crisis (vs full null) ----
    null = pb["_t_hac_abs"]          # |HAC t| over all eligible placebo dates (two-sided)
    null_signed = pb["_t_hac_signed"]  # signed HAC t (one-sided crisis-detection)
    emp_p = {}        # two-sided: frac |placebo t| >= |crisis t|
    emp_p_1sided = {}  # one-sided: frac signed placebo t >= signed crisis t
    for r in crisis:
        emp_p[r["name"]] = float(np.mean(null >= abs(r["t_hac"])))
        emp_p_1sided[r["name"]] = float(np.mean(null_signed >= r["t_hac"]))

    print("\n=== PLACEBO A (paper-style, 10 dates, seed 42) ===")
    print(f"  HAC: mean|t|={pa['hac']['mean_abs_t']:.3f}  max|t|={pa['hac']['max_abs_t']:.3f}"
          f"  sig@fixb5%={pa['hac']['n_sig_fixb_5']}/10  sig@naive5%={pa['hac']['n_sig_naive_5']}/10")
    print(f"  (naive ref: mean|t|={pa['naive']['mean_abs_t']:.3f} max|t|={pa['naive']['max_abs_t']:.3f})")
    print("\n=== PLACEBO B (all eligible, seed-free null) ===")
    print(f"  n={pb['n_dates']}  HAC: mean|t|={pb['hac']['mean_abs_t']:.3f} median|t|={pb['hac']['median_abs_t']:.3f}"
          f"  q95={pb['hac']['q95_abs_t']:.3f}  max|t|={pb['hac']['max_abs_t']:.3f}")
    print(f"  sig@fixb5%(>3.52): {pb['hac']['n_sig_fixb_5']}/{pb['n_dates']} = {pb['hac']['frac_sig_fixb_5']:.3f}")
    print(f"  sig@naive5%(>1.96): {pb['hac']['n_sig_naive_5']}/{pb['n_dates']} = {pb['hac']['frac_sig_naive_5']:.3f}")
    print("\n  Empirical placebo p-values (two-sided |t| / one-sided signed-t):")
    for r in crisis:
        print(f"    {r['name']:<12} crisis t={r['t_hac']:>6.2f}  two-sided p={emp_p[r['name']]:.4f}"
              f"   one-sided p={emp_p_1sided[r['name']]:.4f}")

    # discriminative ratio crisis/placebo under HAC (like-for-like)
    ratio_mean = crisis_mean_sig / pb["hac"]["mean_abs_t"]
    print(f"\n  Like-for-like HAC ratio (crisis-fixb-sig mean / placebo-null mean) = "
          f"{crisis_mean_sig:.3f} / {pb['hac']['mean_abs_t']:.3f} = {ratio_mean:.2f}x")

    # ---- Chow / CUSUM on clean series ----
    print("\n=== STRUCTURAL BREAK (clean series) ===")
    chow_full = structural_break_test(asri, method="chow")
    sub = asri[(asri.index >= "2021-01-01") & (asri.index <= "2024-12-31")]
    chow_2124 = structural_break_test(sub, method="chow")
    cusum_full = structural_break_test(asri, method="cusum")
    cusum_2124 = structural_break_test(sub, method="cusum")
    print(f"  Chow  full({len(asri)}):    stat={chow_full.test_statistic:.4f} crit={chow_full.critical_value:.4f}"
          f" p={chow_full.p_value:.4f} -> {'Stable' if chow_full.is_stable else 'Break'}")
    print(f"  Chow  2021-2024({len(sub)}): stat={chow_2124.test_statistic:.4f} crit={chow_2124.critical_value:.4f}"
          f" p={chow_2124.p_value:.4f} -> {'Stable' if chow_2124.is_stable else 'Break'}")
    print(f"  CUSUM full:    stat={cusum_full.test_statistic:.4f} crit={cusum_full.critical_value:.4f}"
          f" breaks={cusum_full.n_breaks_detected} -> {'Stable' if cusum_full.is_stable else 'Breaks'}")
    print(f"  CUSUM 2021-24: stat={cusum_2124.test_statistic:.4f} crit={cusum_2124.critical_value:.4f}"
          f" breaks={cusum_2124.n_breaks_detected} -> {'Stable' if cusum_2124.is_stable else 'Breaks'}")
    print(f"  CUSUM full break dates: {[d.strftime('%Y-%m') for d in cusum_full.break_dates]}")

    # ---- save ----
    out = {
        "series": {
            "n_rows": int(len(asri)),
            "start": str(asri.index.min().date()),
            "end": str(asri.index.max().date()),
        },
        "crisis_benchmark_hac": [
            {"name": r["name"], "date": r["date"], "cas": round(r["cas"], 3),
             "t_naive": round(r["t_naive"], 3), "t_hac_L20": round(r["t_hac"], 3),
             "fixb_sig_5pct": bool(abs(r["t_hac"]) > FIXB_CRIT_5),
             "fixb_sig_bonf": bool(abs(r["t_hac"]) > FIXB_CRIT_BONF)}
            for r in crisis
        ],
        "crisis_hac_summary": {
            "all4_mean_abs_t": round(crisis_mean_all, 3),
            "all4_max_abs_t": round(crisis_max_all, 3),
            "fixb_sig_count": len(fixb_sig),
            "fixb_sig_mean_abs_t": round(crisis_mean_sig, 3),
            "fixb_crit_5pct": FIXB_CRIT_5,
            "fixb_crit_bonferroni": FIXB_CRIT_BONF,
        },
        "placebo_paper_style_10": {k: v for k, v in pa.items() if not k.startswith("_t_hac")},
        "placebo_all_eligible": {k: v for k, v in pb.items() if not k.startswith("_t_hac")},
        "placebo_empirical_pvalues_twosided": emp_p,
        "placebo_empirical_pvalues_onesided": emp_p_1sided,
        "placebo_null_signed_summary": {
            "mean": round(float(np.mean(null_signed)), 3),
            "median": round(float(np.median(null_signed)), 3),
            "min": round(float(np.min(null_signed)), 3),
            "max": round(float(np.max(null_signed)), 3),
            "frac_signed_gt_3p52": round(float(np.mean(null_signed > FIXB_CRIT_5)), 3),
            "frac_signed_gt_1p645": round(float(np.mean(null_signed > 1.645)), 3),
        },
        "like_for_like_hac_ratio": round(ratio_mean, 3),
        "structural_break": {
            "chow_full": {"stat": round(chow_full.test_statistic, 4),
                          "crit": round(chow_full.critical_value, 4),
                          "p": round(chow_full.p_value, 4), "stable": bool(chow_full.is_stable),
                          "n": int(len(asri))},
            "chow_2021_2024": {"stat": round(chow_2124.test_statistic, 4),
                               "crit": round(chow_2124.critical_value, 4),
                               "p": round(chow_2124.p_value, 4), "stable": bool(chow_2124.is_stable),
                               "n": int(len(sub))},
            "cusum_full": {"stat": round(cusum_full.test_statistic, 4),
                           "crit": round(cusum_full.critical_value, 4),
                           "n_breaks": int(cusum_full.n_breaks_detected),
                           "stable": bool(cusum_full.is_stable),
                           "break_dates": [d.strftime("%Y-%m-%d") for d in cusum_full.break_dates]},
            "cusum_2021_2024": {"stat": round(cusum_2124.test_statistic, 4),
                                "crit": round(cusum_2124.critical_value, 4),
                                "n_breaks": int(cusum_2124.n_breaks_detected),
                                "stable": bool(cusum_2124.is_stable)},
        },
        "placebo_dates_10": [d.strftime("%Y-%m-%d") for d in pick10],
        "config": {"est_window": EST_W, "event_window": EVT_W, "hac_lag": HAC_LAG},
    }
    op = PROJECT_ROOT / "results" / "placebo_chow_cusum_repro.json"
    op.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {op}")


if __name__ == "__main__":
    main()
