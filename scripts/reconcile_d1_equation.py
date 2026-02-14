#!/usr/bin/env python3
"""
Reconcile ASRI aggregate consistency in Cloudflare D1.

Ensures:
1) asri == weighted sum of stored sub-indices
2) alert_level matches deterministic threshold boundaries
3) asri_30d_avg is recomputed from reconciled asri values
4) trend reflects current vs 30d average
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

CF_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
CF_EMAIL = os.getenv("CLOUDFLARE_EMAIL", "contact@farzulla.com")
CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "917877c07fa2e1c9f223db31d3fc52d6")
D1_DATABASE_ID = "f8c08c15-a596-4d7f-8c00-0f7592da26f4"

if not CF_API_KEY:
    cf_env = Path.home() / ".env.cloudflare"
    if cf_env.exists():
        with open(cf_env, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("export CLOUDFLARE_API_KEY="):
                    CF_API_KEY = line.split("=", 1)[1].strip().strip('"')
                elif line.startswith("export CLOUDFLARE_ACCOUNT_ID="):
                    CF_ACCOUNT_ID = line.split("=", 1)[1].strip().strip('"')

WEIGHTS = {
    "stablecoin_risk": 0.30,
    "defi_liquidity_risk": 0.25,
    "contagion_risk": 0.25,
    "arbitrage_opacity": 0.20,
}


def compute_asri(row: dict) -> float:
    value = (
        WEIGHTS["stablecoin_risk"] * float(row["stablecoin_risk"])
        + WEIGHTS["defi_liquidity_risk"] * float(row["defi_liquidity_risk"])
        + WEIGHTS["contagion_risk"] * float(row["contagion_risk"])
        + WEIGHTS["arbitrage_opacity"] * float(row["arbitrage_opacity"])
    )
    return round(value, 1)


def classify_alert(asri: float) -> str:
    if asri < 30:
        return "low"
    if asri < 50:
        return "moderate"
    if asri < 70:
        return "elevated"
    return "critical"


def classify_trend(asri: float, avg_30d: float) -> str:
    delta = asri - avg_30d
    if delta > 2:
        return "rising"
    if delta < -2:
        return "falling"
    return "stable"


class D1Client:
    def __init__(self) -> None:
        if not CF_API_KEY:
            raise ValueError("CLOUDFLARE_API_KEY not found")
        self.base_url = (
            f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_DATABASE_ID}"
        )
        self.client = httpx.AsyncClient(timeout=120.0)
        self.headers = {
            "X-Auth-Email": CF_EMAIL,
            "X-Auth-Key": CF_API_KEY,
            "Content-Type": "application/json",
        }

    async def close(self) -> None:
        await self.client.aclose()

    async def query(self, sql: str, params: list | None = None) -> dict:
        payload: dict = {"sql": sql}
        if params:
            payload["params"] = params
        response = await self.client.post(
            f"{self.base_url}/query",
            headers=self.headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("success"):
            raise RuntimeError(f"D1 query failed: {data}")
        return data


async def run(dry_run: bool) -> None:
    d1 = D1Client()
    try:
        raw = await d1.query(
            """
            SELECT date, asri, asri_30d_avg, trend, alert_level,
                   stablecoin_risk, defi_liquidity_risk, contagion_risk, arbitrage_opacity
            FROM asri_daily
            ORDER BY date ASC
            """
        )
        rows = raw["result"][0]["results"]
        if not rows:
            print("No rows found in D1.")
            return

        recomputed: list[dict] = []
        rolling: list[float] = []
        for row in rows:
            asri = compute_asri(row)
            rolling.append(asri)
            window = rolling[-30:]
            avg_30d = round(sum(window) / len(window), 1)
            alert = classify_alert(asri)
            trend = classify_trend(asri, avg_30d)
            recomputed.append(
                {
                    "date": row["date"],
                    "old_asri": float(row["asri"]),
                    "new_asri": asri,
                    "old_avg": float(row["asri_30d_avg"]) if row["asri_30d_avg"] is not None else asri,
                    "new_avg": avg_30d,
                    "old_alert": row["alert_level"],
                    "new_alert": alert,
                    "old_trend": row["trend"] or "stable",
                    "new_trend": trend,
                }
            )

        changes = [
            r
            for r in recomputed
            if (
                r["old_asri"] != r["new_asri"]
                or r["old_avg"] != r["new_avg"]
                or r["old_alert"] != r["new_alert"]
                or r["old_trend"] != r["new_trend"]
            )
        ]
        print(f"Rows scanned: {len(rows)}")
        print(f"Rows needing reconciliation: {len(changes)}")

        if not changes:
            print("No reconciliation updates required.")
            return

        if dry_run:
            print("Dry run mode; no updates applied.")
            print("Sample changes:")
            for sample in changes[:10]:
                print(sample)
            return

        for row in changes:
            await d1.query(
                """
                UPDATE asri_daily
                SET asri = ?, asri_30d_avg = ?, trend = ?, alert_level = ?, updated_at = datetime('now')
                WHERE date = ?
                """,
                [
                    row["new_asri"],
                    row["new_avg"],
                    row["new_trend"],
                    row["new_alert"],
                    row["date"],
                ],
            )

        print(f"Applied updates: {len(changes)}")
    finally:
        await d1.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile D1 ASRI equation consistency.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply updates to D1 (default is dry-run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run(dry_run=not args.apply))


if __name__ == "__main__":
    main()
