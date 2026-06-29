#!/usr/bin/env python3
"""
Fair-baseline comparison for the ASRI paper (referee-demanded).

Reuses the EXACT protocol that produced ASRI=0.866 / D-Y=0.670:
  - same parquet (results/data/asri_history.parquet)
  - same window 2021-01-01..2024-12-31
  - same rolling D-Y construction (real_dy_hmm_analysis.rolling_connectedness)
  - same common index (asri_win.index intersection roll.index)
  - same crisis labels (create_crisis_labels, 30-day forward pre-windows of the 4 events)
  - same trapezoidal AUROC (compute_roc_metrics.compute_auroc)

Computes single-channel / PC1 / best-single-feature baselines on the IDENTICAL
labels + sample, and the same i.i.d.-day percentile bootstrap CI the paper uses.

No data fabrication. Writes results/baseline_comparison.json.
"""
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent.parent
SCRIPTS = CODE / "scripts"
sys.path.insert(0, str(CODE / "src"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_rd = _load("real_dy_hmm_analysis", SCRIPTS / "real_dy_hmm_analysis.py")
_cr = _load("compute_roc_metrics", SCRIPTS / "compute_roc_metrics.py")

compute_auroc = _cr.compute_auroc            # trapezoidal AUROC (paper's)
compute_auprc = _cr.compute_auprc
find_opt = _cr.find_optimal_threshold
create_crisis_labels = _rd.create_crisis_labels
rolling_connectedness = _rd.rolling_connectedness

SUB = ["stablecoin_risk", "defi_liquidity_risk", "contagion_risk", "arbitrage_opacity"]
CRISIS = [datetime(2022, 5, 12), datetime(2022, 6, 17),
          datetime(2022, 11, 11), datetime(2023, 3, 11)]
WIN_START, WIN_END = "2021-01-01", "2024-12-31"


def percentile_bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, seed=42, alpha=0.05):
    """i.i.d.-day percentile bootstrap, matching the paper's reported CI method."""
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


def auroc_oriented(y_true, raw_score):
    """Return (auroc, sign) where sign in {+1,-1} is the orientation that makes
    the feature point in the 'higher = more crisis-imminent' direction.
    AUROC is invariant to monotone-increasing transforms but a sign flip gives
    1-AUROC; we orient honestly toward the crisis direction (the only sensible
    convention for a stress indicator) and record the sign."""
    a_pos = compute_auroc(y_true, raw_score)
    if a_pos >= 0.5:
        return float(a_pos), +1
    return float(compute_auroc(y_true, -raw_score)), -1


def main():
    # ---- rebuild the EXACT common sample / labels used for 0.866 vs 0.670 ----
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
    n_neg = n - n_pos
    runs = int(np.sum(np.diff(np.concatenate([[0], labels, [0]])) == 1))

    sub_df = win.loc[common, SUB]
    asri_c = asri_win.loc[common].values
    dy_c = roll.loc[common].values

    print(f"common sample n={n}  pos={n_pos}  neg={n_neg}  "
          f"prevalence={n_pos/n:.1%}  independent_crisis_blocks={runs}")
    print(f"span {common.min().date()} .. {common.max().date()}")

    results = {}

    def record(key, label, score, family, note=""):
        au, sign = auroc_oriented(labels, score)
        # AUPRC / opt-threshold use the oriented score (so 'higher=worse')
        s = score if sign == +1 else -score
        ap = float(compute_auprc(labels, s))
        au_pt, au_lo, au_hi = percentile_bootstrap_ci(labels, s, compute_auroc)
        _, opt_m = find_opt(labels, s)
        results[key] = dict(
            label=label, family=family, auroc=au,
            auroc_ci=[au_lo, au_hi], auprc=ap,
            orientation_sign=int(sign),
            precision_at_youden=float(opt_m.get("precision", float("nan"))),
            recall_at_youden=float(opt_m.get("recall", float("nan"))),
            f1_at_youden=float(opt_m.get("f1", float("nan"))),
            note=note,
        )
        print(f"  {label:42s} AUROC={au:.4f} [{au_lo:.3f},{au_hi:.3f}]  "
              f"AUPRC={ap:.3f}  sign={sign:+d}")

    # ---- references (reproduce canon) ----
    print("\n[reference] ASRI composite & D-Y benchmark (reproduce canon):")
    record("asri", "ASRI composite (4-channel)", asri_c, "reference",
           "Headline; weights tuned on these events.")
    record("dy", "Diebold-Yilmaz connectedness", dy_c, "reference",
           "Built from ASRI's own sub-indices (referee circularity flag).")

    # ---- 1. each sub-index alone ----
    print("\n[1] single sub-index (channel) alone:")
    pretty = {"stablecoin_risk": "Stablecoin Concentration Risk channel",
              "defi_liquidity_risk": "DeFi Liquidity Risk channel",
              "contagion_risk": "Contagion Risk channel",
              "arbitrage_opacity": "Regulatory/Arbitrage Opacity channel"}
    for c in SUB:
        record(f"channel_{c}", pretty[c], sub_df[c].values, "single_channel")

    # ---- 2. best single feature available ----
    # NOTE: raw underlying features (e.g. raw stablecoin-TVL drawdown series) are
    # NOT in the released data (only the 4 engineered sub-indices live on the
    # 2021-2024 daily index). The finest-grained single features available are the
    # sub-indices themselves; the 'best single feature' is the best of these.
    chan_aurocs = {c: results[f"channel_{c}"]["auroc"] for c in SUB}
    best_c = max(chan_aurocs, key=chan_aurocs.get)
    results["best_single_feature"] = dict(
        label=f"BEST single available feature = {pretty[best_c]}",
        family="best_single_feature",
        auroc=results[f"channel_{best_c}"]["auroc"],
        auroc_ci=results[f"channel_{best_c}"]["auroc_ci"],
        auprc=results[f"channel_{best_c}"]["auprc"],
        which=best_c,
        note=("Raw underlying features (raw TVL drawdown etc.) are NOT in the "
              "released daily data; sub-indices are the finest single features "
              "available. 'Best single feature' = best of the four channels."),
    )
    print(f"\n[2] best single available feature = {pretty[best_c]} "
          f"(AUROC={chan_aurocs[best_c]:.4f})")

    # ---- 3. PC1 of the four sub-indices ----
    print("\n[3] PC1 of the four sub-indices:")
    X = sub_df.values.astype(float)
    # standardized PC1 (conventional)
    Xz = (X - X.mean(0)) / X.std(0, ddof=0)
    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    pc1_z = Xz @ Vt[0]
    record("pc1_standardized", "PC1 of 4 sub-indices (standardized)", pc1_z,
           "pc1", f"explained_var_ratio={float(S[0]**2/np.sum(S**2)):.3f}")
    results["pc1_standardized"]["explained_var_ratio"] = float(S[0] ** 2 / np.sum(S ** 2))
    results["pc1_standardized"]["loadings"] = {c: float(v) for c, v in zip(SUB, Vt[0])}
    # unstandardized PC1 (robustness)
    Xc = X - X.mean(0)
    U2, S2, Vt2 = np.linalg.svd(Xc, full_matrices=False)
    pc1_raw = Xc @ Vt2[0]
    record("pc1_unstandardized", "PC1 of 4 sub-indices (covariance)", pc1_raw,
           "pc1", f"explained_var_ratio={float(S2[0]**2/np.sum(S2**2)):.3f}")
    results["pc1_unstandardized"]["explained_var_ratio"] = float(S2[0] ** 2 / np.sum(S2 ** 2))
    results["pc1_unstandardized"]["loadings"] = {c: float(v) for c, v in zip(SUB, Vt2[0])}

    # ---- 4. off-the-shelf index: attempt Fear&Greed (alternative.me) ----
    fng_status = "not_attempted"
    fng_block = None
    try:
        import urllib.request
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        with urllib.request.urlopen(url, timeout=15) as r:
            payload = json.loads(r.read().decode())
        recs = payload.get("data", [])
        if recs:
            fdf = pd.DataFrame(recs)
            fdf["date"] = pd.to_datetime(pd.to_numeric(fdf["timestamp"]), unit="s").dt.normalize()
            fdf["value"] = pd.to_numeric(fdf["value"])
            fser = fdf.set_index("date")["value"].sort_index()
            # higher fear = more stress -> use (100 - value) as the stress score,
            # but auroc_oriented handles sign anyway. Align to common index.
            stress = (100 - fser).reindex(common)
            n_have = int(stress.notna().sum())
            if n_have >= int(0.9 * n):
                stress = stress.ffill().bfill()
                record("fear_greed_offshelf",
                       "Crypto Fear&Greed (off-the-shelf, inverted)",
                       stress.values, "offshelf",
                       f"alternative.me F&G; coverage {n_have}/{n} days on common index")
                fng_status = f"computed (coverage {n_have}/{n})"
            else:
                fng_status = f"insufficient_coverage_{n_have}_of_{n}"
        else:
            fng_status = "empty_response"
    except Exception as e:
        fng_status = f"unavailable: {type(e).__name__}: {e}"
    print(f"\n[4] off-the-shelf F&G: {fng_status}")
    fng_block = dict(status=fng_status,
                     source="alternative.me Crypto Fear & Greed Index (live fetch)",
                     note="Off-the-shelf stress index not in repo; fetched live and "
                          "aligned to the common index. Not fabricated. If a future "
                          "run cannot reach the API, this baseline is simply absent.")

    # ---- assemble verdict ----
    asri_au = results["asri"]["auroc"]
    dy_au = results["dy"]["auroc"]
    simple_keys = [k for k in results
                   if results[k]["family"] in ("single_channel", "pc1", "offshelf")]
    best_simple_key = max(simple_keys, key=lambda k: results[k]["auroc"])
    best_simple_au = results[best_simple_key]["auroc"]
    margin = asri_au - best_simple_au
    ties_or_beats = best_simple_au >= asri_au - 1e-9

    # is best simple within ASRI's bootstrap CI (i.e. indistinguishable)?
    asri_ci = results["asri"]["auroc_ci"]
    indistinguishable = best_simple_au >= asri_ci[0]

    verdict = dict(
        asri_auroc=asri_au,
        dy_auroc=dy_au,
        best_simple_baseline=results[best_simple_key]["label"],
        best_simple_baseline_key=best_simple_key,
        best_simple_auroc=best_simple_au,
        asri_minus_best_simple=float(margin),
        any_simple_ties_or_beats_asri=bool(ties_or_beats),
        best_simple_within_asri_bootstrap_ci=bool(indistinguishable),
        asri_auroc_bootstrap_ci=asri_ci,
    )

    out = dict(
        meta=dict(
            generated_utc=datetime.now(timezone.utc).isoformat(),
            purpose=("Referee-demanded fair-baseline comparison: do simple one-line "
                     "baselines match ASRI's headline AUROC 0.866 on the same labels?"),
            protocol=("IDENTICAL to compute_roc_metrics.load_real_data / "
                      "real_dy_hmm_analysis: parquet=results/data/asri_history.parquet, "
                      "window 2021-01-01..2024-12-31, rolling D-Y warmup=60 VAR(1) GFEVD H=10, "
                      "common index = asri_win intersection roll, labels=30d forward "
                      "pre-windows of {Terra,Celsius/3AC,FTX,SVB}, AUROC=trapezoidal "
                      "(compute_roc_metrics.compute_auroc)."),
            n_observations=n, n_crisis_imminent=n_pos, n_non_crisis=n_neg,
            prevalence=float(n_pos / n), n_independent_crisis_blocks=runs,
            sample_span=[str(common.min().date()), str(common.max().date())],
            bootstrap=dict(method="iid_day_percentile (matches paper's reported CI)",
                           n_boot=1000, seed=42,
                           caveat=("Only ~4 independent crisis episodes; i.i.d.-day "
                                   "bootstrap understates sampling uncertainty. The "
                                   "repo's moving_block_bootstrap_roc.json gives the "
                                   "honest (wider) CIs. AUROC gaps among baselines are "
                                   "very unlikely to be distinguishable at 4 events.")),
            caveats=[
                "Only four crisis episodes -> wide sampling uncertainty; AUROC "
                "differences across baselines may not be statistically distinguishable.",
                "All scores (ASRI, channels, PC1) are in-sample on the tuning events; "
                "this comparison is about discrimination, not out-of-sample skill.",
                "Raw underlying features (e.g. raw stablecoin-TVL drawdown) are not in "
                "the released daily data; sub-indices are the finest single features.",
            ],
            fear_greed=fng_block,
        ),
        baselines=results,
        verdict=verdict,
    )

    out_path = CODE / "results" / "baseline_comparison.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 70)
    print(f"ASRI={asri_au:.4f}  D-Y={dy_au:.4f}")
    print(f"BEST simple baseline: {results[best_simple_key]['label']} = {best_simple_au:.4f}")
    print(f"ASRI - best_simple = {margin:+.4f}")
    print(f"any simple ties/beats ASRI: {ties_or_beats}")
    print(f"best simple within ASRI bootstrap CI [{asri_ci[0]:.3f},{asri_ci[1]:.3f}]: "
          f"{indistinguishable}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
