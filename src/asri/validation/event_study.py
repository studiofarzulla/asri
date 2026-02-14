"""
Event Study Methodology for ASRI

Formal event study analysis to measure ASRI behavior around crisis events.
This is the standard approach in empirical finance for measuring
abnormal returns/signals around specific events.

Key metrics:
- Cumulative Abnormal Signal (CAS): Total ASRI deviation from expected
- t-statistic: Statistical significance of the deviation
- Lead time: How many days before crisis did ASRI start rising?
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CrisisEvent:
    """Definition of a crisis event for event study."""
    name: str
    event_date: datetime  # The crisis onset date (t=0)
    description: str = ""
    severity: Literal["moderate", "severe", "extreme"] = "severe"


@dataclass
class EventStudyResult:
    """Results from event study analysis for a single event."""
    event: CrisisEvent
    methodology_profile: str
    lead_method: str
    
    # Pre-event window analysis
    pre_event_window: tuple[int, int]  # e.g., (-60, -1)
    pre_event_mean: float
    pre_event_std: float
    
    # Event window analysis
    event_window: tuple[int, int]  # e.g., (-30, 0)
    
    # Abnormal signal
    abnormal_signal: pd.Series  # Daily abnormal ASRI
    cumulative_abnormal_signal: float  # CAS
    
    # Statistical tests
    t_statistic: float
    p_value: float
    is_significant: bool
    
    # Lead time
    lead_days: int  # Days before event that ASRI started rising significantly
    peak_asri: float
    peak_date: datetime
    
    # ASRI trajectory
    asri_trajectory: pd.Series  # Full trajectory around event


@dataclass(frozen=True)
class EventStudyConfig:
    """Configuration bundle for an event-study methodology profile."""

    name: str
    estimation_window: tuple[int, int]
    event_window: tuple[int, int]
    alpha: float
    lead_method: Literal[
        "first_sigma_breach",
        "first_threshold_crossing",
        "final_sustained_threshold",
    ]
    lead_threshold_sigma: float
    lead_threshold_level: float
    max_lookback: int


METHODOLOGY_PROFILES: dict[str, EventStudyConfig] = {
    # Historical implementation used in early API payloads.
    "legacy_v1": EventStudyConfig(
        name="legacy_v1",
        estimation_window=(-90, -31),
        event_window=(-30, 10),
        alpha=0.05,
        lead_method="first_sigma_breach",
        lead_threshold_sigma=1.5,
        lead_threshold_level=50.0,
        max_lookback=60,
    ),
    # Canonical profile aligned with current paper tables and reproducible outputs.
    "paper_v2": EventStudyConfig(
        name="paper_v2",
        estimation_window=(-90, -31),
        event_window=(-30, 10),
        alpha=0.05,
        lead_method="first_sigma_breach",
        lead_threshold_sigma=1.5,
        lead_threshold_level=50.0,
        max_lookback=30,
    ),
    # Alternative specification retained for explicit reconciliation diagnostics.
    "counterfactual_60d": EventStudyConfig(
        name="counterfactual_60d",
        estimation_window=(-60, -31),
        event_window=(-30, 10),
        alpha=0.05,
        lead_method="final_sustained_threshold",
        lead_threshold_sigma=1.5,
        lead_threshold_level=50.0,
        max_lookback=60,
    ),
}


def get_event_study_config(profile: str) -> EventStudyConfig:
    """Resolve an event-study methodology profile by name."""
    if profile not in METHODOLOGY_PROFILES:
        valid = ", ".join(sorted(METHODOLOGY_PROFILES.keys()))
        raise ValueError(f"Unknown event study profile '{profile}'. Valid: {valid}")
    return METHODOLOGY_PROFILES[profile]


def compute_cumulative_abnormal_signal(
    asri: pd.Series,
    event_date: datetime,
    estimation_window: tuple[int, int] = (-90, -31),
    event_window: tuple[int, int] = (-30, 0),
) -> tuple[pd.Series, float, float, float]:
    """
    Compute Cumulative Abnormal Signal (CAS) for an event.
    
    Methodology:
    1. Estimate "normal" ASRI level from estimation window
    2. Compute abnormal signal = actual - expected in event window
    3. Cumulate abnormal signals
    4. Test significance
    
    Args:
        asri: ASRI time series with datetime index
        event_date: The event date (t=0)
        estimation_window: Days relative to event for baseline estimation
        event_window: Days relative to event for analysis
        
    Returns:
        (abnormal_signal series, CAS, t-statistic, p-value)
    """
    # Convert event_date to timestamp if needed
    if isinstance(event_date, datetime):
        event_ts = pd.Timestamp(event_date)
    else:
        event_ts = event_date
    
    # Get estimation window data
    est_start = event_ts + pd.Timedelta(days=estimation_window[0])
    est_end = event_ts + pd.Timedelta(days=estimation_window[1])
    
    estimation_data = asri[(asri.index >= est_start) & (asri.index <= est_end)]
    
    if len(estimation_data) < 20:
        raise ValueError(f"Insufficient estimation window data: {len(estimation_data)} days")
    
    # Compute expected ASRI (baseline)
    expected_mean = estimation_data.mean()
    expected_std = estimation_data.std()
    
    # Get event window data
    evt_start = event_ts + pd.Timedelta(days=event_window[0])
    evt_end = event_ts + pd.Timedelta(days=event_window[1])
    
    event_data = asri[(asri.index >= evt_start) & (asri.index <= evt_end)]
    
    if len(event_data) < 5:
        raise ValueError(f"Insufficient event window data: {len(event_data)} days")
    
    # Compute abnormal signal
    abnormal = event_data - expected_mean
    
    # Cumulative abnormal signal
    cas = abnormal.sum()
    
    # t-test for abnormal signal
    # H0: Mean abnormal signal = 0
    # Under H0, standardized CAS ~ N(0, sqrt(n) * sigma)
    n = len(abnormal)
    se = expected_std * np.sqrt(n)
    t_stat = cas / se if se > 0 else 0
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    return abnormal, cas, t_stat, p_value


def compute_lead_time(
    asri: pd.Series,
    event_date: datetime,
    estimation_window: tuple[int, int] = (-90, -31),
    threshold_sigma: float = 1.5,
    threshold_level: float = 50.0,
    method: Literal[
        "first_sigma_breach",
        "first_threshold_crossing",
        "final_sustained_threshold",
    ] = "first_sigma_breach",
    max_lookback: int = 60,
) -> int:
    """
    Compute lead time: days before event that ASRI started rising.
    
    Lead-time definitions:
    - first_sigma_breach: earliest day where ASRI > baseline + sigma*std
    - first_threshold_crossing: earliest day where ASRI >= fixed threshold_level
    - final_sustained_threshold: last crossing above threshold_level that remains
      above threshold through event onset
    
    Args:
        asri: ASRI time series
        event_date: Event date
        estimation_window: Relative window used to estimate baseline moments
        threshold_sigma: Number of std devs for sigma-based significance
        threshold_level: Fixed ASRI threshold for threshold-based methods
        method: Lead-time definition
        max_lookback: Maximum days to look back
        
    Returns:
        Number of days of lead time (0 if no early warning)
    """
    if isinstance(event_date, datetime):
        event_ts = pd.Timestamp(event_date)
    else:
        event_ts = event_date
    
    # Evaluation window where lead alerts are searched.
    eval_start = event_ts - pd.Timedelta(days=max_lookback)
    eval_window = asri[(asri.index >= eval_start) & (asri.index <= event_ts)].sort_index()
    if len(eval_window) == 0:
        return 0

    if method == "first_sigma_breach":
        est_start = event_ts + pd.Timedelta(days=estimation_window[0])
        est_end = event_ts + pd.Timedelta(days=estimation_window[1])
        baseline = asri[(asri.index >= est_start) & (asri.index <= est_end)]
        if len(baseline) < 10:
            return 0
        threshold = baseline.mean() + threshold_sigma * baseline.std()
        breaches = eval_window[eval_window > threshold]
        if len(breaches) == 0:
            return 0
        first_date = breaches.index.min()
        return (event_ts - first_date).days

    if method == "first_threshold_crossing":
        crossings = eval_window[eval_window >= threshold_level]
        if len(crossings) == 0:
            return 0
        first_date = crossings.index.min()
        return (event_ts - first_date).days

    # final_sustained_threshold
    idx = eval_window.index
    vals = eval_window.values
    candidate_dates: list[pd.Timestamp] = []
    for i in range(1, len(vals)):
        crossed_up = vals[i] >= threshold_level and vals[i - 1] < threshold_level
        if not crossed_up:
            continue
        tail = vals[i:]
        if np.all(tail >= threshold_level):
            candidate_dates.append(idx[i])

    if len(candidate_dates) == 0:
        return 0

    # The last crossing that sustains through the event is the actionable warning.
    sustained_date = max(candidate_dates)
    return (event_ts - sustained_date).days


def run_event_study(
    asri: pd.Series,
    events: list[CrisisEvent],
    estimation_window: tuple[int, int] | None = None,
    event_window: tuple[int, int] | None = None,
    alpha: float | None = None,
    profile: str = "paper_v2",
    lead_method: Literal[
        "first_sigma_breach",
        "first_threshold_crossing",
        "final_sustained_threshold",
    ] | None = None,
    lead_threshold_sigma: float | None = None,
    lead_threshold_level: float | None = None,
    max_lookback: int | None = None,
) -> list[EventStudyResult]:
    """
    Run event study for multiple crisis events.
    
    Args:
        asri: ASRI time series with datetime index
        events: List of crisis events
        estimation_window: Window for baseline estimation
        event_window: Window for event analysis
        alpha: Significance level
        
    Returns:
        List of EventStudyResult for each event
    """
    cfg = get_event_study_config(profile)
    est_window = estimation_window if estimation_window is not None else cfg.estimation_window
    evt_window = event_window if event_window is not None else cfg.event_window
    alpha_level = alpha if alpha is not None else cfg.alpha
    lead_mode = lead_method if lead_method is not None else cfg.lead_method
    lead_sigma = lead_threshold_sigma if lead_threshold_sigma is not None else cfg.lead_threshold_sigma
    lead_level = lead_threshold_level if lead_threshold_level is not None else cfg.lead_threshold_level
    lookback = max_lookback if max_lookback is not None else cfg.max_lookback

    results = []
    
    for event in events:
        try:
            # Compute abnormal signal
            abnormal, cas, t_stat, p_value = compute_cumulative_abnormal_signal(
                asri, event.event_date, est_window, evt_window
            )
            
            # Get trajectory
            evt_ts = pd.Timestamp(event.event_date)
            traj_start = evt_ts + pd.Timedelta(days=evt_window[0])
            traj_end = evt_ts + pd.Timedelta(days=evt_window[1])
            trajectory = asri[(asri.index >= traj_start) & (asri.index <= traj_end)]
            
            # Get estimation period stats
            est_start = evt_ts + pd.Timedelta(days=est_window[0])
            est_end = evt_ts + pd.Timedelta(days=est_window[1])
            est_data = asri[(asri.index >= est_start) & (asri.index <= est_end)]
            
            # Lead time
            lead_days = compute_lead_time(
                asri=asri,
                event_date=event.event_date,
                estimation_window=est_window,
                threshold_sigma=lead_sigma,
                threshold_level=lead_level,
                method=lead_mode,
                max_lookback=lookback,
            )
            
            # Peak ASRI
            if len(trajectory) > 0:
                peak_idx = trajectory.idxmax()
                peak_asri = trajectory.max()
            else:
                peak_idx = event.event_date
                peak_asri = 0
            
            results.append(EventStudyResult(
                event=event,
                methodology_profile=cfg.name,
                lead_method=lead_mode,
                pre_event_window=est_window,
                pre_event_mean=est_data.mean() if len(est_data) > 0 else 0,
                pre_event_std=est_data.std() if len(est_data) > 0 else 0,
                event_window=evt_window,
                abnormal_signal=abnormal,
                cumulative_abnormal_signal=cas,
                t_statistic=t_stat,
                p_value=p_value,
                is_significant=p_value < alpha_level,
                lead_days=lead_days,
                peak_asri=peak_asri,
                peak_date=peak_idx,
                asri_trajectory=trajectory,
            ))
            
        except (ValueError, KeyError) as e:
            # Skip events with insufficient data
            continue
    
    return results


def format_event_study_table(results: list[EventStudyResult]) -> str:
    """Format event study results as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Event Study Results: ASRI Response to Crisis Events}",
        r"\label{tab:event_study}",
        r"\small",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Event & Date & Pre-Event & Peak & CAS & $t$-stat & Lead Days & Sig. \\",
        r"\midrule",
    ]
    
    for r in results:
        sig = "Yes" if r.is_significant else "No"
        stars = "***" if r.p_value < 0.01 else ("**" if r.p_value < 0.05 else ("*" if r.p_value < 0.10 else ""))
        
        date_str = r.event.event_date.strftime("%Y-%m")
        
        lines.append(
            f"{r.event.name} & {date_str} & {r.pre_event_mean:.1f} & "
            f"{r.peak_asri:.1f} & {r.cumulative_abnormal_signal:.1f}{stars} & "
            f"{r.t_statistic:.2f} & {r.lead_days} & {sig} \\\\"
        )
    
    # Summary statistics
    significant = sum(1 for r in results if r.is_significant)
    avg_lead = np.mean([r.lead_days for r in results]) if results else 0
    avg_cas = np.mean([r.cumulative_abnormal_signal for r in results]) if results else 0
    
    lines.extend([
        r"\midrule",
        f"\\multicolumn{{8}}{{l}}{{Significant events: {significant}/{len(results)} "
        f"({100*significant/len(results):.0f}\\%)}} \\\\",
        f"\\multicolumn{{8}}{{l}}{{Average lead time: {avg_lead:.1f} days}} \\\\",
        f"\\multicolumn{{8}}{{l}}{{Average CAS: {avg_cas:.1f}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item CAS = Cumulative Abnormal Signal. Lead Days = days before event ASRI elevated.",
        r"\item *** $p<0.01$, ** $p<0.05$, * $p<0.10$",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def plot_event_study(
    results: list[EventStudyResult],
    output_path: str = "figures/event_study.pdf",
) -> str:
    """Generate matplotlib code for event study plot."""
    return f"""
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(len(results), 1, figsize=(10, 3*len(results)))
if len(results) == 1:
    axes = [axes]

for ax, result in zip(axes, results):
    # Normalize days relative to event
    trajectory = result.asri_trajectory
    event_date = pd.Timestamp(result.event.event_date)
    days = [(d - event_date).days for d in trajectory.index]
    
    # Plot trajectory
    ax.plot(days, trajectory.values, 'b-', linewidth=2, label='ASRI')
    
    # Pre-event baseline
    ax.axhline(result.pre_event_mean, color='gray', linestyle='--', 
               label=f'Baseline ({{result.pre_event_mean:.1f}})')
    ax.axhspan(result.pre_event_mean - result.pre_event_std,
               result.pre_event_mean + result.pre_event_std,
               alpha=0.2, color='gray')
    
    # Event line
    ax.axvline(0, color='red', linestyle='-', linewidth=2, label='Event')
    
    # Lead time marker
    if result.lead_days > 0:
        ax.axvline(-result.lead_days, color='orange', linestyle=':', 
                   label=f'Lead (-{{result.lead_days}}d)')
    
    ax.set_title(f'{{result.event.name}} (CAS={{result.cumulative_abnormal_signal:.1f}}, '
                 f'p={{result.p_value:.3f}})')
    ax.set_xlabel('Days Relative to Event')
    ax.set_ylabel('ASRI')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('{output_path}')
"""
