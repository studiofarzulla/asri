#!/usr/bin/env python3
"""Live API contract smoke-check for ASRI deployment."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ASRI-Contract-Check/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def assert_true(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check live ASRI API contract.")
    parser.add_argument(
        "--base-url",
        default="https://asri.dissensus.ai/api",
        help="Base API URL to validate (default: https://asri.dissensus.ai/api)",
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    failures: list[str] = []

    current = fetch_json(f"{base}/asri/current")
    methodology = fetch_json(f"{base}/asri/methodology")
    regime = fetch_json(f"{base}/asri/regime")
    validation = fetch_json(f"{base}/asri/validation")

    assert_true("asri" in current, "current endpoint missing `asri`", failures)
    assert_true("sub_indices" in current, "current endpoint missing `sub_indices`", failures)
    assert_true(
        current.get("methodology_profile") == "paper_v2",
        "current endpoint methodology_profile != paper_v2",
        failures,
    )
    assert_true(
        current.get("alert_level") in {"low", "moderate", "elevated", "critical"},
        "current endpoint alert_level not in canonical enum",
        failures,
    )

    transition = regime.get("transition_probs", {})
    assert_true("to_elevated" in transition, "regime endpoint missing `to_elevated`", failures)
    assert_true("to_crisis" not in transition, "regime endpoint still exposes deprecated `to_crisis`", failures)

    assert_true(
        methodology.get("methodology_profile") == "paper_v2",
        "methodology endpoint methodology_profile != paper_v2",
        failures,
    )
    assert_true(
        methodology.get("documentation_url") == "https://asri.dissensus.ai/docs",
        "methodology endpoint documentation_url mismatch",
        failures,
    )
    assert_true(
        methodology.get("validation_results", {}).get("average_lead_time_days") == 30,
        "methodology endpoint average_lead_time_days mismatch",
        failures,
    )

    es = validation.get("event_study", {})
    summary = es.get("summary", {})
    assert_true(es.get("methodology_profile") == "paper_v2", "validation event_study methodology_profile mismatch", failures)
    assert_true(summary.get("avg_lead_time") == 29.8, "validation avg_lead_time mismatch", failures)
    assert_true(es.get("terra_luna", {}).get("t_stat") == 5.47, "validation Terra t-stat mismatch", failures)

    if failures:
        print("Contract check FAILED:")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)

    print(f"Contract check PASSED for {base}")


if __name__ == "__main__":
    main()
