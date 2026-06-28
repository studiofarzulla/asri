#!/usr/bin/env python3
"""
Referee-demanded fair-baseline EXTENSION for the ASRI paper.

Adds three "needs-no-crypto-native-construction" baselines on the IDENTICAL
labels / window / sample / bootstrap protocol that produced ASRI=0.866:

  B1. Rolling 30-day BTC/ETH return (plain market-return proxy; the dominant
      dynamic crypto input is TVL drawdown, mechanically a price-decline proxy).
  B2. STANDALONE VIX + 10Y-Treasury macro composite = the paper's Bank_t
      construction evaluated ALONE
      (Bank_t = 0.6*norm100(DGS10,[2,6]) + 0.4*norm100(VIXCLS,[12,40])).
  B3. Detrended / first-differenced Contagion Risk (CR is non-stationary;
      does its 0.851 discrimination survive removing the linear trend?).

Reuses the EXACT machinery of scripts/baseline_comparison.py:
  - same parquet, window 2021-01-01..2024-12-31
  - same rolling D-Y warmup (real_dy_hmm_analysis.rolling_connectedness)
  - same common index, same 30d-forward crisis labels (4 events)
  - same trapezoidal AUROC/AUPRC (compute_roc_metrics)
  - same i.i.d.-day percentile bootstrap (seed=42, n_boot=1000)
PLUS a PAIRED bootstrap test of ASRI-minus-baseline AUROC on the SAME resampled
days (the direct "distinguishable from ASRI 0.866?" test).

Raw VIX/10Y/BTC/ETH are NOT in the repo (fetched live at backfill time). They are
re-fetched live here from the SAME sources the backfill uses (FRED for DGS10/VIXCLS,
DefiLlama coins for BTC/ETH). DGS10/VIXCLS are non-revised market series, so a live
pull reproduces the historical values. No fabrication: if a source is unreachable,
that baseline is reported as ABSENT, not invented.

Writes results/baseline_extension.json.
"""
import importlib.util
import io
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path("/home/purrpower/Resurrexi/projects/papers/papers-official/asri/code")
SCRIPTS = CODE / "scripts"
sys.path.insert(0, str(CODE / "src"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_rd = _load("real_dy_hmm_analysis", SCRIPTS / "real_dy_hmm_analysis.py")
_cr = _load("compute_roc_metrics", SCRIPTS / "compute_roc_metrics.py")

compute_auroc = _cr.compute_auroc
compute_auprc = _cr.compute_auprc
find_opt = _cr.find_optimal_threshold
create_crisis_labels = _rd.create_crisis_labels
rolling_connectedness = _rd.rolling_connectedness

SUB = ["stablecoin_risk", "defi_liquidity_risk", "contagion_risk", "arbitrage_opacity"]
CRISIS = [datetime(2022, 5, 12), datetime(2022, 6, 17),
          datetime(2022, 11, 11), datetime(2023, 3, 11)]
WIN_START, WIN_END = "2021-01-01", "2024-12-31"
SEED, N_BOOT = 42, 1000


# ----------------------------- bootstrap helpers ---------------------------
def auroc_oriented(y_true, raw_score):
    a_pos = compute_auroc(y_true, raw_score)
    if a_pos >= 0.5:
        return float(a_pos), +1
    return float(compute_auroc(y_true, -raw_score)), -1


def percentile_bootstrap_ci(y_true, y_score, metric_fn, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_score)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots[b] = metric_fn(y_true[idx], y_score[idx])
    lo = float(np.percentile(boots, alpha / 2 * 100))
    hi = float(np.percentile(boots, (1 - alpha / 2) * 100))
    return float(point), lo, hi


def paired_bootstrap_diff(y_true, s_asri, s_base, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Paired i.i.d.-day bootstrap of AUROC(ASRI) - AUROC(baseline) on the SAME
    resampled days. Returns (delta_point, lo, hi, p_two_sided, frac_asri_higher)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    delta_pt = compute_auroc(y_true, s_asri) - compute_auroc(y_true, s_base)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        diffs[b] = compute_auroc(yt, s_asri[idx]) - compute_auroc(yt, s_base[idx])
    lo = float(np.percentile(diffs, alpha / 2 * 100))
    hi = float(np.percentile(diffs, (1 - alpha / 2) * 100))
    frac_pos = float(np.mean(diffs > 0))
    frac_neg = float(np.mean(diffs < 0))
    p_two = float(2.0 * min(frac_pos, frac_neg))
    p_two = min(p_two, 1.0)
    return float(delta_pt), lo, hi, p_two, frac_pos


# ----------------------------- live data fetch -----------------------------
def fetch_fred_csv(series_id):
    """FRED public CSV (no API key). Non-revised market series."""
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
           f"&cosd=2020-12-01&coed=2025-01-31")
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read().decode()
    df = pd.read_csv(io.StringIO(raw))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"].sort_index()


def fetch_defillama_coin(coin):
    """Daily price for coingecko:<coin> from DefiLlama coins API (same source the
    backfill uses for BTC). Returns a daily price Series (UTC-normalized date)."""
    rows = {}
    for year in [2021, 2022, 2023, 2024]:
        start_ts = int(datetime(year, 1, 1).timestamp())
        url = (f"https://coins.llama.fi/chart/coingecko:{coin}"
               f"?start={start_ts}&span=400&period=1d")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read().decode())
            pts = data.get("coins", {}).get(f"coingecko:{coin}", {}).get("prices", [])
            for p in pts:
                d = pd.Timestamp(p["timestamp"], unit="s").normalize()
                rows[d] = float(p["price"])
        except Exception as e:
            print(f"    [warn] defillama {coin} {year}: {type(e).__name__}: {e}")
    if not rows:
        return None
    return pd.Series(rows).sort_index()


def norm100(x, lo, hi):
    return np.clip((x - lo) / (hi - lo) * 100.0, 0, 100)


# --------------------------------- main ------------------------------------
def main():
    df = pd.read_parquet(CODE / "results" / "data" / "asri_history.parquet").sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)

    win = df.loc[WIN_START:WIN_END]
    roll = rolling_connectedness(df[SUB].dropna(), window=60, lags=1, horizon=10) \
        .loc[WIN_START:WIN_END].dropna()
    asri_win = win["asri"].dropna()
    common = asri_win.index.intersection(roll.index)
    labels = create_crisis_labels(common, CRISIS, window_days=30)
    n = len(common)
    n_pos = int(labels.sum())
    asri_c = asri_win.loc[common].values  # already higher = worse (sign +1)

    # full daily ASRI index (contiguous daily) for building lagged/derived series
    full_idx = df.index
    cr_full = df["contagion_risk"]

    print(f"common sample n={n} pos={n_pos} neg={n-n_pos} "
          f"prevalence={n_pos/n:.1%} span {common.min().date()}..{common.max().date()}")

    # sanity: reproduce canon ASRI AUROC
    asri_auroc = compute_auroc(labels, asri_c)
    print(f"[sanity] ASRI AUROC reproduced = {asri_auroc:.4f} (canon 0.8657)")

    results = {}
    notes_data = {}

    def record(key, label, raw_score, family, note=""):
        au, sign = auroc_oriented(labels, raw_score)
        s = raw_score if sign == +1 else -raw_score
        ap = float(compute_auprc(labels, s))
        au_pt, au_lo, au_hi = percentile_bootstrap_ci(labels, s, compute_auroc)
        _, opt_m = find_opt(labels, s)
        d_pt, d_lo, d_hi, d_p, d_fracpos = paired_bootstrap_diff(labels, asri_c, s)
        distinguishable = not (d_lo <= 0.0 <= d_hi)
        results[key] = dict(
            label=label, family=family, auroc=au,
            auroc_ci=[au_lo, au_hi], auprc=ap, orientation_sign=int(sign),
            precision_at_youden=float(opt_m.get("precision", float("nan"))),
            recall_at_youden=float(opt_m.get("recall", float("nan"))),
            f1_at_youden=float(opt_m.get("f1", float("nan"))),
            paired_vs_asri=dict(
                delta_auroc_asri_minus_base=d_pt,
                delta_ci=[d_lo, d_hi],
                p_two_sided=d_p,
                frac_boot_asri_higher=d_fracpos,
                distinguishable_from_asri=bool(distinguishable),
            ),
            note=note,
        )
        verdict = "DISTINGUISHABLE" if distinguishable else "indistinguishable"
        print(f"  {label:46s} AUROC={au:.4f} [{au_lo:.3f},{au_hi:.3f}] "
              f"AUPRC={ap:.3f}  Δvs ASRI={d_pt:+.4f} [{d_lo:+.3f},{d_hi:+.3f}] "
              f"p={d_p:.3f} -> {verdict}")

    # reference (reproduce canon ASRI for the paired machine)
    record("asri", "ASRI composite (4-channel) [reference]", asri_c, "reference",
           "Headline; weights tuned on these events. Δ vs itself = 0 by construction.")

    # ---------- B1: rolling 30-day BTC/ETH return ----------
    print("\n[B1] rolling 30-day BTC/ETH return (live DefiLlama):")
    btc = fetch_defillama_coin("bitcoin")
    eth = fetch_defillama_coin("ethereum")
    b1_status = {}
    if btc is not None:
        btc_d = btc.reindex(pd.date_range(full_idx.min(), full_idx.max(), freq="D")).ffill()
        ret_btc = (btc_d / btc_d.shift(30) - 1.0)
        record("ret30_btc", "BTC 30-day return", ret_btc.reindex(common).values,
               "market_return", "trailing 30 calendar-day return; sign flips so decline=stress")
        b1_status["btc"] = f"ok ({btc.index.min().date()}..{btc.index.max().date()}, {len(btc)} pts)"
    else:
        b1_status["btc"] = "UNREACHABLE -> baseline absent"
    if eth is not None:
        eth_d = eth.reindex(pd.date_range(full_idx.min(), full_idx.max(), freq="D")).ffill()
        ret_eth = (eth_d / eth_d.shift(30) - 1.0)
        record("ret30_eth", "ETH 30-day return", ret_eth.reindex(common).values,
               "market_return", "trailing 30 calendar-day return; sign flips so decline=stress")
        b1_status["eth"] = f"ok ({eth.index.min().date()}..{eth.index.max().date()}, {len(eth)} pts)"
    else:
        b1_status["eth"] = "UNREACHABLE -> baseline absent"
    if btc is not None and eth is not None:
        ew = 0.5 * (ret_btc + ret_eth)
        record("ret30_btceth_ew", "BTC/ETH 30-day return (equal-weight)",
               ew.reindex(common).values, "market_return",
               "EW avg of BTC & ETH trailing 30d returns; PRIMARY market-return proxy")
    notes_data["B1_market_return"] = b1_status

    # ---------- B2: standalone VIX + 10Y macro composite (Bank_t) ----------
    print("\n[B2] standalone VIX+10Y macro composite = Bank_t alone (live FRED):")
    b2_status = {}
    try:
        dgs10 = fetch_fred_csv("DGS10")
        vix = fetch_fred_csv("VIXCLS")
        # forward-fill onto the full daily ASRI index (matches backfill's
        # "last observation on or before target date" mapping)
        cal = pd.date_range(full_idx.min(), full_idx.max(), freq="D")
        dgs10_d = dgs10.reindex(cal).ffill()
        vix_d = vix.reindex(cal).ffill()
        treasury_stress = norm100(dgs10_d, 2.0, 6.0)
        vix_stress = norm100(vix_d, 12.0, 40.0)
        bank = 0.6 * treasury_stress + 0.4 * vix_stress
        record("bank_vix10y", "VIX+10Y macro composite (Bank_t alone)",
               bank.reindex(common).values, "macro_offshelf",
               "0.6*norm100(DGS10,[2,6]) + 0.4*norm100(VIXCLS,[12,40]); the paper's "
               "Bank_t construction evaluated standalone. No crypto-native construction.")
        # robustness: each leg alone
        record("treasury10y_only", "10Y Treasury stress alone (norm100[2,6])",
               treasury_stress.reindex(common).values, "macro_offshelf",
               "single FRED series, no construction")
        record("vix_only", "VIX stress alone (norm100[12,40])",
               vix_stress.reindex(common).values, "macro_offshelf",
               "single FRED series, no construction")
        b2_status = dict(dgs10=f"ok ({dgs10.index.min().date()}..{dgs10.index.max().date()})",
                         vixcls=f"ok ({vix.index.min().date()}..{vix.index.max().date()})")
    except Exception as e:
        b2_status = dict(error=f"{type(e).__name__}: {e} -> baseline absent")
        print(f"    [warn] FRED fetch failed: {e}")
    notes_data["B2_macro"] = b2_status

    # ---------- B3: detrended / first-differenced Contagion Risk ----------
    print("\n[B3] non-stationarity check on Contagion Risk (CR=0.851 canon):")
    cr_common = cr_full.reindex(common)
    # raw CR (reproduce the 0.851 canon as the anchor)
    record("cr_raw", "Contagion Risk (raw, canon anchor)", cr_common.values,
           "cr_check", "reproduces the single-channel CR AUROC the demotion rests on")
    # (a) linear-detrended on the common window
    t = np.arange(n, dtype=float)
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, cr_common.values, rcond=None)
    cr_detrend = cr_common.values - A @ coef
    record("cr_detrended", "Contagion Risk (linear-detrended)", cr_detrend,
           "cr_check", "CR minus OLS linear time trend on the common sample")
    # (a2) quadratic-detrended (robustness on the detrend model)
    A2 = np.vstack([t ** 2, t, np.ones_like(t)]).T
    coef2, *_ = np.linalg.lstsq(A2, cr_common.values, rcond=None)
    cr_detrend2 = cr_common.values - A2 @ coef2
    record("cr_detrended_quad", "Contagion Risk (quadratic-detrended)", cr_detrend2,
           "cr_check", "CR minus OLS quadratic time trend; a parabola can absorb the "
                       "mid-sample crisis hump, so this is a lower bound on detrended skill")
    # (b) first-differenced (use full series so first common date is defined; keeps n)
    cr_fd_full = cr_full.diff()
    cr_fd = cr_fd_full.reindex(common).values
    if np.isnan(cr_fd).any():  # safety: fill any residual NaN with 0 (no-change)
        cr_fd = np.nan_to_num(cr_fd, nan=0.0)
    record("cr_first_diff", "Contagion Risk (first-differenced)", cr_fd,
           "cr_check", "CR_t - CR_{t-1}; stationary increment series")

    # stationarity diagnostics on CR (report, don't gate)
    adf_p = kpss_stat = None
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        adf_p = float(adfuller(cr_common.values, autolag="AIC")[1])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat = float(kpss(cr_common.values, regression="c", nlags="auto")[0])
    except Exception as e:
        print(f"    [warn] stationarity tests unavailable: {e}")

    # ---------- assemble ----------
    out = dict(
        meta=dict(
            generated_utc=datetime.now(timezone.utc).isoformat(),
            purpose=("Referee extension: add the two most obvious no-crypto-construction "
                     "baselines (plain BTC/ETH 30d return; standalone VIX+10Y macro "
                     "composite) plus a non-stationarity check on Contagion Risk, all on "
                     "the IDENTICAL labels/window/sample/bootstrap as baseline_comparison.json."),
            protocol=("IDENTICAL to baseline_comparison.py: parquet=asri_history.parquet, "
                      "window 2021-01-01..2024-12-31, rolling D-Y warmup=60 VAR(1) GFEVD H=10, "
                      "common index intersection, labels=30d forward pre-windows of "
                      "{Terra 2022-05-12, Celsius/3AC 2022-06-17, FTX 2022-11-11, SVB 2023-03-11}, "
                      "AUROC/AUPRC=trapezoidal (compute_roc_metrics)."),
            n_observations=n, n_crisis_imminent=n_pos, n_non_crisis=n - n_pos,
            prevalence=float(n_pos / n),
            sample_span=[str(common.min().date()), str(common.max().date())],
            asri_auroc_reproduced=float(asri_auroc),
            bootstrap=dict(
                ci_method="iid_day_percentile (matches baseline_comparison.py)",
                distinguishability_test=("PAIRED iid-day bootstrap of AUROC(ASRI)-AUROC(base) "
                                         "on the SAME resampled days; distinguishable iff 95% "
                                         "delta-CI excludes 0."),
                n_boot=N_BOOT, seed=SEED,
                caveat=("Only ~4 independent crisis episodes; i.i.d.-day bootstrap UNDERSTATES "
                        "sampling uncertainty (moving_block_bootstrap_roc.json is the honest, "
                        "wider CI). Treat 'distinguishable' verdicts as optimistic upper bounds "
                        "on separability."),
            ),
            data_sources=dict(
                fred="FRED public CSV fredgraph.csv (DGS10, VIXCLS; non-revised market series)",
                crypto="DefiLlama coins API chart/coingecko:{bitcoin,ethereum} (same source as backfill BTC)",
                contagion_risk="results/data/asri_history.parquet (engineered sub-index)",
                fetch_status=notes_data,
            ),
            contagion_risk_stationarity=dict(adf_pvalue=adf_p, kpss_level_stat=kpss_stat,
                                             note="paper reports ADF p~0.069, KPSS~2.06"),
            no_fabrication=("Raw VIX/10Y/BTC/ETH re-fetched live from the backfill's own "
                            "sources; any unreachable series is reported ABSENT, not invented."),
        ),
        baselines=results,
    )
    out_path = CODE / "results" / "baseline_extension.json"
    out_path.write_text(json.dumps(out, indent=2))

    # ---------- console summary table ----------
    print("\n" + "=" * 100)
    print(f"{'baseline':46s} {'AUROC':>7s} {'AUPRC':>7s} {'Δvs ASRI':>9s} "
          f"{'p':>6s}  distinguishable?")
    print("-" * 100)
    order = ["asri", "ret30_btc", "ret30_eth", "ret30_btceth_ew",
             "bank_vix10y", "treasury10y_only", "vix_only",
             "cr_raw", "cr_detrended", "cr_detrended_quad", "cr_first_diff"]
    for k in order:
        if k not in results:
            continue
        r = results[k]
        pv = r["paired_vs_asri"]
        print(f"{r['label']:46s} {r['auroc']:7.4f} {r['auprc']:7.3f} "
              f"{pv['delta_auroc_asri_minus_base']:+9.4f} {pv['p_two_sided']:6.3f}  "
              f"{'YES' if pv['distinguishable_from_asri'] else 'no (ties ASRI)'}")
    print("=" * 100)
    if adf_p is not None:
        print(f"CR stationarity: ADF p={adf_p:.3f}, KPSS level stat={kpss_stat:.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
