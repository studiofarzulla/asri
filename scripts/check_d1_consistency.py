#!/usr/bin/env python3
"""Fail-fast integrity check for D1 ASRI equation + alert mapping."""

from __future__ import annotations

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


def _load_cloudflare_env_fallback() -> None:
    global CF_API_KEY, CF_ACCOUNT_ID
    if CF_API_KEY:
        return
    env_file = Path.home() / ".env.cloudflare"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("export CLOUDFLARE_API_KEY="):
            CF_API_KEY = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("export CLOUDFLARE_ACCOUNT_ID="):
            CF_ACCOUNT_ID = line.split("=", 1)[1].strip().strip('"')


async def main() -> None:
    _load_cloudflare_env_fallback()
    if not CF_API_KEY:
        raise SystemExit("CLOUDFLARE_API_KEY not found")

    sql = """
    SELECT
      SUM(CASE WHEN ROUND(asri,1) != ROUND((0.30*stablecoin_risk + 0.25*defi_liquidity_risk + 0.25*contagion_risk + 0.20*arbitrage_opacity),1) THEN 1 ELSE 0 END) AS mismatch_count,
      SUM(CASE
        WHEN asri < 30 AND alert_level != 'low' THEN 1
        WHEN asri >= 30 AND asri < 50 AND alert_level != 'moderate' THEN 1
        WHEN asri >= 50 AND asri < 70 AND alert_level != 'elevated' THEN 1
        WHEN asri >= 70 AND alert_level != 'critical' THEN 1
        ELSE 0 END) AS alert_mismatch_count,
      COUNT(*) AS total
    FROM asri_daily
    """
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_DATABASE_ID}/query"
    headers = {
        "X-Auth-Email": CF_EMAIL,
        "X-Auth-Key": CF_API_KEY,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, json={"sql": sql})
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(payload)
        row = payload["result"][0]["results"][0]

    mismatch_count = int(row["mismatch_count"])
    alert_mismatch_count = int(row["alert_mismatch_count"])
    total = int(row["total"])
    print(
        f"D1 integrity check: total={total}, mismatch_count={mismatch_count}, "
        f"alert_mismatch_count={alert_mismatch_count}"
    )

    if mismatch_count > 0 or alert_mismatch_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
