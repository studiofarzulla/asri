"""Guardrail tests for methodology and aggregation integrity."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asri.validation.event_study import CrisisEvent, get_event_study_config, run_event_study


def test_paper_v2_profile_defaults() -> None:
    """Canonical profile should stay pinned to adjudicated settings."""
    cfg = get_event_study_config("paper_v2")
    assert cfg.estimation_window == (-90, -31)
    assert cfg.event_window == (-30, 10)
    assert cfg.lead_method == "first_sigma_breach"
    assert cfg.max_lookback == 30


def test_profile_lead_time_behavior_is_intentional() -> None:
    """
    paper_v2 should cap lookback at 30d while legacy_v1 keeps wider search.
    """
    event_date = datetime(2022, 6, 30)
    dates = pd.date_range("2022-01-01", "2022-08-15", freq="D")
    series = pd.Series(40.0, index=dates)

    # Force a sustained rise 40 days before the event.
    rise_start = pd.Timestamp(event_date) - pd.Timedelta(days=40)
    series.loc[series.index >= rise_start] = 48.0

    event = CrisisEvent(name="Synthetic", event_date=event_date)
    paper_result = run_event_study(series, [event], profile="paper_v2")[0]
    legacy_result = run_event_study(series, [event], profile="legacy_v1")[0]

    assert paper_result.lead_days == 30
    assert legacy_result.lead_days == 40


def test_primary_reconciliation_artifact_matches_canonical_profile() -> None:
    """Reconciliation output should explicitly mark the canonical profile."""
    artifact = ROOT / "results" / "reconciliation" / "event_study_primary.json"
    assert artifact.exists()

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["profile"] == "paper_v2"
    assert payload["n_events"] == 4
    assert payload["n_significant"] == 4
