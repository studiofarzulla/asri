#!/usr/bin/env python3
"""
Generator for results/data/dy_rolling_connectedness_daily.csv (provenance closer).

The daily rolling Diebold--Yilmaz total-connectedness series used in the
supplement (60-day rolling VAR(1), generalized FEVD at H=10, on the four ASRI
sub-indices) previously shipped as a CSV with no generator in the repo. This
script documents and regenerates that series from the frozen canonical parquet
(results/data/asri_history.parquet) by calling the SAME rolling_connectedness()
routine the analysis uses (scripts/real_dy_hmm_analysis.py), sliced to the paper
window (WINDOW_START..WINDOW_END).

No fabrication: the only input is the frozen parquet; the VAR/FEVD computation is
deterministic. Output columns: date, connectedness (percent).

PROVENANCE NOTE (read before running). The *released* CSV (rolling mean
~28.74%) was built from the pre-repro sub-index series. Like the main ASRI series
(DATA_PROVENANCE.md sec. 2), it is a FROZEN released artefact and does NOT
bit-reproduce from the current frozen parquet: the re-frozen parquet contains more
valid sub-index days, so a fresh regen drifts (mean ~44%). This script is
therefore a provenance/recipe artefact. To protect the released figure it will
NOT overwrite an existing canonical CSV unless ASRI_DY_FORCE=1; by default it
writes a sibling ``*.regen.csv`` for inspection.

Usage:
    python scripts/build_dy_daily.py                  # writes *.regen.csv (safe)
    ASRI_DY_OUT=/tmp/check.csv python scripts/build_dy_daily.py   # write elsewhere
    ASRI_DY_FORCE=1 python scripts/build_dy_daily.py   # overwrite canonical CSV
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
DATA = PROJECT_ROOT / "results" / "data" / "asri_history.parquet"
DEFAULT_OUT = PROJECT_ROOT / "results" / "data" / "dy_rolling_connectedness_daily.csv"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    rd = _load("real_dy_hmm_analysis", SCRIPTS / "real_dy_hmm_analysis.py")

    df = pd.read_parquet(DATA)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Same spec as scripts/real_dy_hmm_analysis.py: 60-day rolling VAR(1), GFEVD H=10,
    # computed on the full sample then sliced to the paper window.
    roll_full = rd.rolling_connectedness(
        df[rd.SUB_INDICES].dropna(), window=60, lags=1, horizon=10
    )
    roll = roll_full.loc[rd.WINDOW_START:rd.WINDOW_END].dropna()

    explicit_out = os.environ.get("ASRI_DY_OUT")
    force = os.environ.get("ASRI_DY_FORCE") == "1"
    if explicit_out:
        out_path = Path(explicit_out)
    elif DEFAULT_OUT.exists() and not force:
        # Protect the frozen released figure (mean ~28.74%): write a sibling.
        out_path = DEFAULT_OUT.with_suffix(".regen.csv")
        print(
            f"[guard] frozen {DEFAULT_OUT.name} exists; writing regen to "
            f"{out_path.name} instead (set ASRI_DY_FORCE=1 to overwrite)."
        )
    else:
        out_path = DEFAULT_OUT

    out_path.parent.mkdir(parents=True, exist_ok=True)
    roll.rename("connectedness").rename_axis("date").to_csv(out_path)
    print(
        f"[ok] wrote {out_path}  (n={len(roll)}, "
        f"span {roll.index.min().date()}..{roll.index.max().date()}, "
        f"mean={roll.mean():.4f}%)"
    )


if __name__ == "__main__":
    main()
