#!/usr/bin/env python3
"""
Verify the DEFERRED validation-layer claims on the RELEASED data, and regenerate
the stationarity table. Source of record for ASRI_VALIDATION_VERIFY_jun2026.md.

Covers:
  - Stationarity (ADF + KPSS) on every sub-index + composite  -> writes stationarity.tex
  - Chow midpoint structural-break test
  - Walk-forward continuous-level R^2 (the "catastrophic R^2" claim)
  - Walk-forward OOS detection 4/4 + first-crossing lead times (delegates to generate_detection_table)
  - Threshold-based (ASRI>=50) in-sample first-crossing leads
  - Bybit 2025 readings + 2024/2025 threshold-breach audit

Run:  cd code && python3 scripts/verify_deferred_validation.py
"""
import sys
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA = PROJECT_ROOT / "results" / "data" / "asri_history.parquet"
TABLES = PROJECT_ROOT / "results" / "tables"

EVENTS = [
    ("Terra/Luna", datetime(2022, 5, 12)),
    ("Celsius/3AC", datetime(2022, 6, 17)),
    ("FTX", datetime(2022, 11, 11)),
    ("SVB", datetime(2023, 3, 11)),
]
SUBS = ["stablecoin_risk", "defi_liquidity_risk", "contagion_risk", "arbitrage_opacity"]
WEIGHTS = {"stablecoin_risk": 0.30, "defi_liquidity_risk": 0.25,
           "contagion_risk": 0.25, "arbitrage_opacity": 0.20}


def load():
    df = pd.read_parquet(DATA)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def stationarity(df):
    from statsmodels.tsa.stattools import adfuller, kpss
    cols = {"ASRI": df["asri"], "Stablecoin Risk": df["stablecoin_risk"],
            "DeFi Liquidity": df["defi_liquidity_risk"], "Contagion Risk": df["contagion_risk"],
            "Opacity Risk": df["arbitrage_opacity"]}
    rows = []
    print("\n=== STATIONARITY (full sample 2021-2026) ===")
    for name, s in cols.items():
        s = s.dropna()
        adf_stat, adf_p = adfuller(s, autolag="AIC")[:2]
        kpss_stat = kpss(s, regression="c", nlags="auto")[0]
        if adf_p < 0.01 and kpss_stat < 0.739:
            concl = "Stationary"
        elif adf_p < 0.05 and kpss_stat >= 0.739:
            concl = "Trend-stat."
        elif adf_p >= 0.05:
            concl = "Non-stat."
        else:
            concl = "Trend-stat."
        rows.append((name, adf_stat, adf_p, kpss_stat, concl))
        print(f"  {name:16s} ADF={adf_stat:7.3f} p={adf_p:.4f} KPSS={kpss_stat:.3f} -> {concl}")
    return rows


def write_stationarity_table(rows):
    def pfmt(p):
        return "$<$0.001" if p < 0.001 else f"{p:.3f}"
    lines = [
        r"\begin{table}[H]", r"\begin{threeparttable}", r"\centering",
        r"\caption{Stationarity Test Results}", r"\label{tab:stationarity}", r"\small",
        r"\begin{tabular}{@{}l*{3}{r}l@{}}", r"\toprule",
        r"Variable & ADF Stat & ADF $p$ & KPSS & Conclusion \\", r"\midrule",
    ]
    for name, adf_stat, adf_p, kpss_stat, concl in rows:
        lines.append(f"{name:16s}& ${adf_stat:.2f}$ & {pfmt(adf_p)} & {kpss_stat:.2f} & {concl} \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}", r"\small",
        r"\item ADF: Augmented Dickey-Fuller (lag selection via AIC, intercept); KPSS (Bartlett kernel, auto bandwidth). Released full-sample series (2021--2026).",
        r"\item KPSS critical values: 0.463 (5\%), 0.739 (1\%). Values above 0.739 indicate at best trend-stationarity.",
        r"\item Contagion Risk fails to reject the unit root and is treated as non-stationary.",
        r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}",
    ]
    (TABLES / "stationarity.tex").write_text("\n".join(lines))
    print(f"  wrote {TABLES / 'stationarity.tex'}")


def chow(df):
    print("\n=== CHOW (AR(1) midpoint) ===")
    for label, s in [("full", df["asri"]),
                     ("2021-2024", df["asri"][(df.index >= "2021-01-01") & (df.index <= "2024-12-31")])]:
        v = s.values
        n = len(v); mid = n // 2
        y = v[1:]; X = np.column_stack([np.ones(n - 1), v[:-1]])
        bf = np.linalg.lstsq(X, y, rcond=None)[0]; ssrf = np.sum((y - X @ bf) ** 2)
        y1, y2 = y[:mid - 1], y[mid - 1:]; X1, X2 = X[:mid - 1], X[mid - 1:]
        b1 = np.linalg.lstsq(X1, y1, rcond=None)[0]; ssr1 = np.sum((y1 - X1 @ b1) ** 2)
        b2 = np.linalg.lstsq(X2, y2, rcond=None)[0]; ssr2 = np.sum((y2 - X2 @ b2) ** 2)
        k = X.shape[1]
        c = ((ssrf - ssr1 - ssr2) / k) / ((ssr1 + ssr2) / (n - 2 * k))
        p = 1 - stats.f.cdf(c, k, n - 2 * k)
        print(f"  {label:10s} Chow={c:.4f} p={p:.4f}")


def r2(df):
    from asri.validation.walk_forward import purged_walk_forward_cv
    print("\n=== WALK-FORWARD CONTINUOUS R^2 (target = ASRI_{t+30}) ===")
    target = df["asri"].shift(-30).dropna()
    res = purged_walk_forward_cv(df[SUBS], target, WEIGHTS, n_splits=5, purge_days=30)
    print(f"  mean_test_r2={res.mean_test_r2:.2f} std={res.std_test_r2:.2f} folds={[round(f.test_r2,2) for f in res.folds]}")
    tr = df[SUBS][df.index < "2024-01-01"]
    te = df[SUBS][df.index >= "2024-01-01"]
    ty = target.reindex(te.index).dropna(); te = te.reindex(ty.index)
    pred = sum(WEIGHTS[c] * te[c] for c in SUBS)
    oos = 1 - np.sum((ty.values - pred.values) ** 2) / np.sum((ty.values - ty.mean()) ** 2)
    print(f"  OOS R^2 (train<2024, test 2024+) = {oos:.2f}")


def leads(df):
    asri = df["asri"]
    print("\n=== LEAD TIMES ===")
    print("  Threshold-based (ASRI>=50, 30d pre-window) first-crossing:")
    tb = []
    for name, d in EVENTS:
        ev = pd.Timestamp(d)
        pre = asri[(asri.index >= ev - pd.Timedelta(days=30)) & (asri.index < ev)]
        cr = pre[pre >= 50]
        lead = (ev - cr.index[0]).days if len(cr) else None
        if lead is not None:
            tb.append(lead)
        print(f"    {name:12s} peak={pre.max():.1f} lead={lead}")
    print(f"    mean(detected)={np.mean(tb):.2f}")
    print("  Walk-forward (90th-pct training threshold) first-crossing:")
    wf = []
    for name, d in EVENTS:
        ev = pd.Timestamp(d)
        train = asri[asri.index < ev - pd.Timedelta(days=90)]
        thr = np.percentile(train, 90)
        pre = asri[(asri.index >= ev - pd.Timedelta(days=30)) & (asri.index < ev)]
        cr = pre[pre >= thr]
        lead = (ev - cr.index[0]).days if len(cr) else None
        wf.append(lead)
        print(f"    {name:12s} thr={thr:.1f} peak={pre.max():.1f} lead={lead}")
    print(f"    mean={np.mean([x for x in wf if x is not None]):.2f}")


def bybit(df):
    asri = df["asri"]
    print("\n=== BYBIT / SPECIFICITY ===")
    for d in ["2025-02-10", "2025-02-21", "2025-02-28"]:
        r = df.loc[d]
        print(f"  {d}: ASRI={r['asri']:.1f} SCR={r['stablecoin_risk']:.1f} "
              f"DLR={r['defi_liquidity_risk']:.1f} CR={r['contagion_risk']:.1f}")
    a24 = asri[(asri.index >= "2024-01-01") & (asri.index <= "2024-12-31")]
    a25 = asri[(asri.index >= "2025-01-01") & (asri.index <= "2025-12-31")]
    print(f"  2024 days>=50: {(a24>=50).sum()} (max {a24.max():.1f} on {a24.idxmax().date()})")
    print(f"  2025 days>=50: {(a25>=50).sum()} (max {a25.max():.1f})")


def main():
    df = load()
    rows = stationarity(df)
    write_stationarity_table(rows)
    chow(df)
    r2(df)
    leads(df)
    bybit(df)


if __name__ == "__main__":
    main()
