#!/usr/bin/env python3
"""
Loader for data/peg_history.csv -- supplies REAL historical peg prices to the
ASRI backtest's Stablecoin Concentration Risk (SCR) sub-index, replacing the
hardcoded ``price=1.0 / peg_deviation=0.0 / peg_volatility=10.0`` defaults in
src/asri/backtest/backtest.py::_snapshot_to_inputs.

Exactly how SCR consumes the data (mirrors src/asri/pipeline/transform.py
``transform_stablecoin_risk``):

    peg_deviation_i = |1 - price_i|
    weighted_deviation = sum(peg_deviation_i * circulating_i) / sum(circulating_i)
    peg_volatility = normalize_to_100(weighted_deviation * 100, 0, 5)   # 0%->0, 5%+->100
    # peg_volatility enters SCR via STABLECOIN_WEIGHTS['peg_volatility'] = 0.1

Drop-in usage inside _snapshot_to_inputs (replacing the hardcoded block):

    from scripts.peg_loader import PegHistory
    PEG = PegHistory()                      # load once
    ...
    pv = PEG.peg_volatility(
        target_date,
        circulating_by_symbol={s.symbol: s.circulating for s in stablecoins},
        use_intraday_low=False,             # True = stress-sensitive (uses price_low)
    )
    stablecoin_inputs = StablecoinRiskInputs(
        tvl_ratio=tvl_risk,
        treasury_stress=treasury_stress,
        concentration_hhi=concentration_risk,
        peg_volatility=pv,                  # <-- was hardcoded 10.0
    )

The supply weights (``circulating_by_symbol``) already come from DeFiLlama in
the backtest snapshot (snapshot.stablecoin_market_caps); this loader only
supplies the real per-coin prices keyed by symbol + nearest date.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "data" / "peg_history.csv"


def normalize_to_100(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 50.0
    return max(0.0, min(100.0, (value - min_val) / (max_val - min_val) * 100.0))


class PegHistory:
    def __init__(self, csv_path: Path = CSV, max_gap_days: int = 3):
        self.df = pd.read_csv(csv_path, parse_dates=["date"])
        self.max_gap_days = max_gap_days
        self._by_symbol = {
            sym: g.sort_values("date").reset_index(drop=True)
            for sym, g in self.df.groupby("symbol")
        }

    def price(self, symbol: str, on: datetime, use_intraday_low: bool = False) -> float | None:
        """Nearest-date price for a symbol; None if no point within max_gap_days."""
        g = self._by_symbol.get(symbol)
        if g is None or g.empty:
            return None
        ts = pd.Timestamp(on).normalize()
        idx = (g["date"] - ts).abs().idxmin()
        row = g.loc[idx]
        if abs((row["date"] - ts).days) > self.max_gap_days:
            return None
        return float(row["price_low"] if use_intraday_low else row["price"])

    def peg_volatility(
        self,
        on: datetime,
        circulating_by_symbol: dict[str, float],
        use_intraday_low: bool = False,
    ) -> float:
        """Supply-weighted SCR peg_volatility (0-100) for a date.

        Falls back to per-coin par (deviation 0) for any symbol absent from the
        peg dataset, so unknown coins do not spuriously inflate the score.
        """
        num = den = 0.0
        for symbol, circ in circulating_by_symbol.items():
            if circ <= 0:
                continue
            px = self.price(symbol, on, use_intraday_low=use_intraday_low)
            dev = abs(1.0 - px) if px is not None else 0.0
            num += dev * circ
            den += circ
        weighted_deviation = (num / den) if den > 0 else 0.0
        return normalize_to_100(weighted_deviation * 100.0, 0.0, 5.0)


if __name__ == "__main__":
    PEG = PegHistory()
    demo = {
        "UST peak 2022-05-12": (datetime(2022, 5, 12),
                                 {"USDT": 83e9, "USDC": 49e9, "BUSD": 18e9, "DAI": 6e9, "UST": 11e9}),
        "SVB 2023-03-12":      (datetime(2023, 3, 12),
                                 {"USDT": 72e9, "USDC": 43e9, "BUSD": 8e9, "DAI": 5e9}),
    }
    for label, (d, w) in demo.items():
        daily = PEG.peg_volatility(d, w, use_intraday_low=False)
        intra = PEG.peg_volatility(d, w, use_intraday_low=True)
        print(f"{label}: peg_volatility daily={daily:.1f}  intraday={intra:.1f}  (hardcoded default was 10.0)")
