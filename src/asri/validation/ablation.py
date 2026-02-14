"""
Ablation Study (Leave-One-Out Analysis) for ASRI

Measures the contribution of each sub-index to crisis detection
by systematically removing components and measuring performance degradation.

Key questions:
1. Does any single sub-index carry most detection power?
2. Are there redundancies between sub-indices?
3. Which crises are most sensitive to component removal?
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Crisis Event Definitions (matching backtest.py)
# =============================================================================

CRISIS_EVENTS = [
    {
        "name": "Terra/Luna",
        "date": datetime(2022, 5, 12),
        "description": "UST depeg and Luna death spiral",
    },
    {
        "name": "Celsius/3AC",
        "date": datetime(2022, 6, 17),
        "description": "Three Arrows Capital insolvency",
    },
    {
        "name": "FTX Collapse",
        "date": datetime(2022, 11, 11),
        "description": "FTX/Alameda fraud",
    },
    {
        "name": "SVB Crisis",
        "date": datetime(2023, 3, 11),
        "description": "Silicon Valley Bank failure",
    },
]

# Baseline weights
BASELINE_WEIGHTS = {
    "stablecoin_risk": 0.30,
    "defi_liquidity_risk": 0.25,
    "contagion_risk": 0.25,
    "arbitrage_opacity": 0.20,
}

# Short names for display
COMPONENT_SHORT_NAMES = {
    "stablecoin_risk": "SCR",
    "defi_liquidity_risk": "DLR",
    "contagion_risk": "CR",
    "arbitrage_opacity": "OR",
}


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class CrisisDetectionResult:
    """Detection result for a single crisis."""
    crisis_name: str
    detected: bool
    peak_asri: float
    lead_time_days: int  # Days before crisis that ASRI first exceeded threshold
    first_breach_date: Optional[datetime] = None


@dataclass
class AblationResult:
    """Results from ablating (removing) one component."""
    excluded_component: str  # Component name or "None" for baseline
    excluded_short: str  # Short name (SCR, DLR, etc.)
    weights: dict[str, float]  # Renormalized weights
    weights_str: str  # Formatted string like "0/33/33/27"

    # Detection metrics
    detection_rate: str  # e.g., "4/4" or "3/4"
    detected_count: int
    total_crises: int

    # Lead time
    avg_lead_time: float  # Average days of lead time
    lead_time_delta: float  # Change from baseline

    # Per-crisis results
    crisis_results: list[CrisisDetectionResult] = field(default_factory=list)


# =============================================================================
# Core Ablation Functions
# =============================================================================

def compute_ablated_weights(
    base_weights: dict[str, float],
    exclude_component: str,
) -> dict[str, float]:
    """
    Remove one component and renormalize remaining weights.

    Args:
        base_weights: Original weight dictionary
        exclude_component: Component to remove (set to 0)

    Returns:
        New weights that sum to 1.0
    """
    new_weights = base_weights.copy()
    new_weights[exclude_component] = 0.0

    # Renormalize remaining weights
    total = sum(new_weights.values())
    if total > 0:
        new_weights = {k: v / total for k, v in new_weights.items()}

    return new_weights


def compute_asri_with_weights(
    sub_indices: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Compute ASRI from sub-indices using specified weights.

    Args:
        sub_indices: DataFrame with columns for each sub-index
        weights: Weight for each component

    Returns:
        ASRI time series
    """
    asri = pd.Series(0.0, index=sub_indices.index)

    for component, weight in weights.items():
        if component in sub_indices.columns and weight > 0:
            asri += weight * sub_indices[component]

    return asri


def detect_crisis(
    asri: pd.Series,
    crisis_date: datetime,
    threshold: float = 50.0,
    pre_window_days: int = 30,
) -> CrisisDetectionResult:
    """
    Check if ASRI detected a crisis within the pre-crisis window.

    Args:
        asri: ASRI time series
        crisis_date: Date of crisis onset
        threshold: ASRI level for detection (default 50 = Elevated)
        pre_window_days: Days before crisis to check

    Returns:
        CrisisDetectionResult with detection status and metrics
    """
    crisis_ts = pd.Timestamp(crisis_date)
    window_start = crisis_ts - pd.Timedelta(days=pre_window_days)

    # Get ASRI values in pre-crisis window
    window = asri[(asri.index >= window_start) & (asri.index < crisis_ts)]

    if len(window) == 0:
        return CrisisDetectionResult(
            crisis_name="",
            detected=False,
            peak_asri=0.0,
            lead_time_days=0,
        )

    peak_asri = window.max()
    detected = peak_asri >= threshold

    # Calculate lead time (first breach)
    breaches = window[window >= threshold]
    if len(breaches) > 0:
        first_breach = breaches.index.min()
        lead_time = (crisis_ts - first_breach).days
    else:
        first_breach = None
        lead_time = 0

    return CrisisDetectionResult(
        crisis_name="",
        detected=detected,
        peak_asri=float(peak_asri),
        lead_time_days=lead_time,
        first_breach_date=first_breach.to_pydatetime() if first_breach else None,
    )


def run_ablation_analysis(
    sub_indices: pd.DataFrame,
    threshold: float = 50.0,
    pre_window_days: int = 30,
) -> list[AblationResult]:
    """
    Run full ablation analysis: baseline + removing each component.

    Args:
        sub_indices: DataFrame with sub-index time series
        threshold: Detection threshold (default 50 = Elevated)
        pre_window_days: Pre-crisis window for detection

    Returns:
        List of AblationResult, one for baseline + one per component
    """
    results = []

    # 1. Baseline (no component removed)
    baseline_asri = compute_asri_with_weights(sub_indices, BASELINE_WEIGHTS)
    baseline_result = _evaluate_detection(
        asri=baseline_asri,
        weights=BASELINE_WEIGHTS,
        excluded_component=None,
        threshold=threshold,
        pre_window_days=pre_window_days,
        baseline_lead_time=None,
    )
    results.append(baseline_result)
    baseline_lead_time = baseline_result.avg_lead_time

    # 2. Ablate each component
    for component in BASELINE_WEIGHTS.keys():
        ablated_weights = compute_ablated_weights(BASELINE_WEIGHTS, component)
        ablated_asri = compute_asri_with_weights(sub_indices, ablated_weights)

        ablation_result = _evaluate_detection(
            asri=ablated_asri,
            weights=ablated_weights,
            excluded_component=component,
            threshold=threshold,
            pre_window_days=pre_window_days,
            baseline_lead_time=baseline_lead_time,
        )
        results.append(ablation_result)

    return results


def _evaluate_detection(
    asri: pd.Series,
    weights: dict[str, float],
    excluded_component: Optional[str],
    threshold: float,
    pre_window_days: int,
    baseline_lead_time: Optional[float],
) -> AblationResult:
    """
    Evaluate detection performance for a weight configuration.
    """
    crisis_results = []
    total_lead_time = 0
    detected_count = 0

    for crisis in CRISIS_EVENTS:
        result = detect_crisis(
            asri=asri,
            crisis_date=crisis["date"],
            threshold=threshold,
            pre_window_days=pre_window_days,
        )
        result.crisis_name = crisis["name"]
        crisis_results.append(result)

        if result.detected:
            detected_count += 1
            total_lead_time += result.lead_time_days

    # Calculate average lead time (only for detected crises)
    avg_lead_time = total_lead_time / detected_count if detected_count > 0 else 0.0

    # Calculate lead time delta from baseline
    if baseline_lead_time is not None:
        lead_time_delta = avg_lead_time - baseline_lead_time
    else:
        lead_time_delta = 0.0

    # Format weights string
    if excluded_component is None:
        excluded_short = "None"
        weights_str = "30/25/25/20"
    else:
        excluded_short = COMPONENT_SHORT_NAMES[excluded_component]
        # Format: SCR/DLR/CR/OR as percentages
        w_pct = [int(round(weights[c] * 100)) for c in [
            "stablecoin_risk", "defi_liquidity_risk",
            "contagion_risk", "arbitrage_opacity"
        ]]
        weights_str = "/".join(str(w) for w in w_pct)

    return AblationResult(
        excluded_component=excluded_component or "None (baseline)",
        excluded_short=excluded_short,
        weights=weights,
        weights_str=weights_str,
        detection_rate=f"{detected_count}/{len(CRISIS_EVENTS)}",
        detected_count=detected_count,
        total_crises=len(CRISIS_EVENTS),
        avg_lead_time=avg_lead_time,
        lead_time_delta=lead_time_delta,
        crisis_results=crisis_results,
    )


# =============================================================================
# LaTeX Output
# =============================================================================

def format_ablation_table(results: list[AblationResult]) -> str:
    """
    Generate LaTeX table for ablation results.

    Args:
        results: List of AblationResult from run_ablation_analysis

    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table}[H]",
        r"\begin{threeparttable}",
        r"\centering",
        r"\caption{Sub-Index Ablation Analysis (Leave-One-Out)}",
        r"\label{tab:ablation}",
        r"\small",
        r"\begin{tabular}{@{}lcccr@{}}",
        r"\toprule",
        r"Excluded & Weights & Detection & Lead Time & $\Delta$ Lead \\",
        r"Component & (renormalized) & Rate & (days) & (days) \\",
        r"\midrule",
    ]

    for r in results:
        # Format delta with sign
        if r.excluded_short == "None":
            delta_str = "---"
        elif r.lead_time_delta >= 0:
            delta_str = f"+{r.lead_time_delta:.0f}"
        else:
            delta_str = f"{r.lead_time_delta:.0f}"

        # Highlight baseline row
        if r.excluded_short == "None":
            row_prefix = r"\textbf{None (baseline)}"
        else:
            row_prefix = f"-- {r.excluded_short}"

        lines.append(
            f"{row_prefix} & {r.weights_str} & {r.detection_rate} & "
            f"{r.avg_lead_time:.0f} & {delta_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Detection threshold: ASRI $\geq 50$ (Elevated) within 30-day pre-crisis window.",
        r"\item Weights format: SCR/DLR/CR/OR as percentages.",
        r"\item Lead time = days between first threshold breach and crisis onset.",
        r"\item $\Delta$ Lead = change from baseline (negative indicates earlier detection).",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_detailed_ablation_table(results: list[AblationResult]) -> str:
    """
    Generate detailed LaTeX table showing per-crisis detection.

    Args:
        results: List of AblationResult

    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ablation Analysis: Per-Crisis Detection Results}",
        r"\label{tab:ablation_detailed}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Configuration & Terra/Luna & Celsius/3AC & FTX & SVB & Summary \\",
        r"\midrule",
    ]

    for r in results:
        if r.excluded_short == "None":
            config = r"\textbf{Baseline}"
        else:
            config = f"-- {r.excluded_short}"

        crisis_cells = []
        for cr in r.crisis_results:
            if cr.detected:
                crisis_cells.append(r"\checkmark")
            else:
                crisis_cells.append(r"$\times$")

        lines.append(
            f"{config} & {crisis_cells[0]} & {crisis_cells[1]} & "
            f"{crisis_cells[2]} & {crisis_cells[3]} & {r.detection_rate} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    return "\n".join(lines)


# =============================================================================
# Analysis Helpers
# =============================================================================

def compute_component_importance(results: list[AblationResult]) -> dict[str, float]:
    """
    Compute importance score for each component based on detection degradation.

    Importance = (baseline detection - ablated detection) / baseline detection

    Higher score = more important component.
    """
    baseline = results[0]
    baseline_rate = baseline.detected_count / baseline.total_crises

    importance = {}
    for r in results[1:]:
        ablated_rate = r.detected_count / r.total_crises
        degradation = baseline_rate - ablated_rate
        importance[r.excluded_short] = degradation

    return importance


def identify_redundancy(results: list[AblationResult]) -> list[str]:
    """
    Identify components that can be removed without detection loss.

    Returns list of component short names that show no detection degradation.
    """
    baseline = results[0]
    redundant = []

    for r in results[1:]:
        if r.detected_count >= baseline.detected_count:
            redundant.append(r.excluded_short)

    return redundant


def identify_critical_components(results: list[AblationResult]) -> list[str]:
    """
    Identify components whose removal causes detection loss.

    Returns list of component short names that are critical for detection.
    """
    baseline = results[0]
    critical = []

    for r in results[1:]:
        if r.detected_count < baseline.detected_count:
            critical.append(r.excluded_short)

    return critical


# =============================================================================
# Synthetic Data for Testing / Paper Results
# =============================================================================

def generate_synthetic_results() -> list[AblationResult]:
    """
    DEPRECATED: This function generates synthetic (hand-constructed) ablation
    results that were used during early development before the real ablation
    pipeline was run. The paper now uses actual ablation results computed from
    backtested data via run_ablation_analysis(). The real results (baseline 3/4
    detection, Terra/Luna consistently missed) differ substantially from these
    synthetic values (which claimed 4/4 baseline detection).

    Do NOT use this function for paper results. It is retained only for
    reference and testing purposes.

    Returns:
        List of AblationResult (synthetic, not empirical)
    """
    import warnings
    warnings.warn(
        "generate_synthetic_results() is deprecated. Use run_ablation_analysis() "
        "with real backtested data instead. Paper results already use real data.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Baseline: all 4 crises detected with ~40 day lead time
    baseline = AblationResult(
        excluded_component="None (baseline)",
        excluded_short="None",
        weights=BASELINE_WEIGHTS,
        weights_str="30/25/25/20",
        detection_rate="4/4",
        detected_count=4,
        total_crises=4,
        avg_lead_time=40.0,
        lead_time_delta=0.0,
        crisis_results=[
            CrisisDetectionResult("Terra/Luna", True, 73.0, 6),
            CrisisDetectionResult("Celsius/3AC", True, 73.0, 54),
            CrisisDetectionResult("FTX Collapse", True, 73.0, 60),
            CrisisDetectionResult("SVB Crisis", True, 74.6, 40),
        ],
    )

    # Without SCR: Miss Terra/Luna (stablecoin-driven), reduced lead time
    # SCR is critical for stablecoin crises
    no_scr = AblationResult(
        excluded_component="stablecoin_risk",
        excluded_short="SCR",
        weights=compute_ablated_weights(BASELINE_WEIGHTS, "stablecoin_risk"),
        weights_str="0/36/36/29",
        detection_rate="3/4",
        detected_count=3,
        total_crises=4,
        avg_lead_time=35.0,
        lead_time_delta=-5.0,
        crisis_results=[
            CrisisDetectionResult("Terra/Luna", False, 48.2, 0),  # Miss!
            CrisisDetectionResult("Celsius/3AC", True, 68.5, 48),
            CrisisDetectionResult("FTX Collapse", True, 65.3, 55),
            CrisisDetectionResult("SVB Crisis", True, 62.1, 35),
        ],
    )

    # Without DLR: Still detect all, but reduced lead time
    # DLR is important but has redundancy with other indices
    no_dlr = AblationResult(
        excluded_component="defi_liquidity_risk",
        excluded_short="DLR",
        weights=compute_ablated_weights(BASELINE_WEIGHTS, "defi_liquidity_risk"),
        weights_str="40/0/33/27",
        detection_rate="4/4",
        detected_count=4,
        total_crises=4,
        avg_lead_time=32.0,
        lead_time_delta=-8.0,
        crisis_results=[
            CrisisDetectionResult("Terra/Luna", True, 66.4, 4),
            CrisisDetectionResult("Celsius/3AC", True, 62.8, 42),
            CrisisDetectionResult("FTX Collapse", True, 58.7, 45),
            CrisisDetectionResult("SVB Crisis", True, 64.2, 37),
        ],
    )

    # Without CR: Miss FTX (contagion-driven), reduced detection
    # CR is critical for contagion/counterparty crises
    no_cr = AblationResult(
        excluded_component="contagion_risk",
        excluded_short="CR",
        weights=compute_ablated_weights(BASELINE_WEIGHTS, "contagion_risk"),
        weights_str="40/33/0/27",
        detection_rate="3/4",
        detected_count=3,
        total_crises=4,
        avg_lead_time=28.0,
        lead_time_delta=-12.0,
        crisis_results=[
            CrisisDetectionResult("Terra/Luna", True, 62.1, 5),
            CrisisDetectionResult("Celsius/3AC", True, 58.4, 38),
            CrisisDetectionResult("FTX Collapse", False, 47.8, 0),  # Miss!
            CrisisDetectionResult("SVB Crisis", True, 55.9, 30),
        ],
    )

    # Without OR: Detect all but with slightly reduced lead time
    # OR amplifies signals but isn't primary driver
    no_or = AblationResult(
        excluded_component="arbitrage_opacity",
        excluded_short="OR",
        weights=compute_ablated_weights(BASELINE_WEIGHTS, "arbitrage_opacity"),
        weights_str="38/31/31/0",
        detection_rate="4/4",
        detected_count=4,
        total_crises=4,
        avg_lead_time=36.0,
        lead_time_delta=-4.0,
        crisis_results=[
            CrisisDetectionResult("Terra/Luna", True, 70.2, 5),
            CrisisDetectionResult("Celsius/3AC", True, 69.8, 50),
            CrisisDetectionResult("FTX Collapse", True, 67.4, 52),
            CrisisDetectionResult("SVB Crisis", True, 71.3, 37),
        ],
    )

    return [baseline, no_scr, no_dlr, no_cr, no_or]


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run ablation analysis and print results."""
    # NOTE: Paper uses real ablation results from run_ablation_analysis().
    # Falling back to synthetic for CLI demo only.
    results = generate_synthetic_results()

    print("=" * 70)
    print("ASRI ABLATION STUDY RESULTS")
    print("=" * 70)
    print()

    # Print summary table
    print("Summary Table:")
    print("-" * 70)
    print(f"{'Excluded':<20} {'Weights':<15} {'Det.':<8} {'Lead':<10} {'Δ Lead':<10}")
    print("-" * 70)

    for r in results:
        delta_str = f"{r.lead_time_delta:+.0f}" if r.excluded_short != "None" else "---"
        print(f"{r.excluded_short:<20} {r.weights_str:<15} {r.detection_rate:<8} {r.avg_lead_time:<10.0f} {delta_str:<10}")

    print()
    print("=" * 70)
    print()

    # Importance analysis
    importance = compute_component_importance(results)
    print("Component Importance (detection degradation when removed):")
    for comp, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {comp}: {imp:.2f}")

    print()

    # Redundancy analysis
    redundant = identify_redundancy(results)
    critical = identify_critical_components(results)

    print(f"Redundant components (removable without detection loss): {redundant}")
    print(f"Critical components (removal causes detection loss): {critical}")

    print()
    print("=" * 70)
    print()

    # LaTeX output
    print("LaTeX Table:")
    print(format_ablation_table(results))


if __name__ == "__main__":
    main()
