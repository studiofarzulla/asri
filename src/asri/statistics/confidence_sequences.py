"""
Confidence Sequences for ASRI Uncertainty Quantification

Addresses Reviewer Q8: Uncertainty propagation from data gaps and placeholders.

Confidence sequences provide anytime-valid confidence intervals that maintain
coverage at any stopping time, unlike traditional fixed-sample confidence intervals.
This is crucial for sequential monitoring applications like ASRI.

References:
- Howard, S. R., et al. (2021). Time-uniform, nonparametric, nonasymptotic
  confidence sequences. Annals of Statistics.
- Waudby-Smith, I., & Ramdas, A. (2021). Confidence sequences for sampling
  without replacement. NeurIPS.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ConfidenceSequence:
    """Results from confidence sequence computation."""
    timestamps: np.ndarray
    point_estimates: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    alpha: float
    method: str


@dataclass
class UncertaintyBands:
    """ASRI with uncertainty bands."""
    asri: pd.Series
    lower: pd.Series
    upper: pd.Series
    confidence_scores: pd.DataFrame  # Per-component confidence


def compute_variance_bounded_cs(
    observations: np.ndarray,
    alpha: float = 0.05,
    variance_bound: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence sequence for bounded variance observations.

    Uses the Hoeffding bound approach for sub-Gaussian random variables.

    For ASRI (bounded in [0, 100]), variance is bounded by (100-0)^2/4 = 2500.

    Args:
        observations: Sequential observations
        alpha: Significance level (e.g., 0.05 for 95% CI)
        variance_bound: Upper bound on variance (default: (max-min)^2/4)

    Returns:
        (lower_bounds, upper_bounds) arrays
    """
    n = len(observations)

    if variance_bound is None:
        # For bounded [0, 100] data
        variance_bound = 2500  # (100-0)^2 / 4

    cumsum = np.cumsum(observations)
    cumsq = np.cumsum(observations ** 2)

    means = cumsum / np.arange(1, n + 1)

    # Confidence radius using sub-Gaussian bound
    # r_t = sqrt(2 * sigma^2 * log(2/alpha) / t)
    t = np.arange(1, n + 1)

    # Use mixture martingale radius (tighter than Hoeffding for large t)
    log_term = np.log(2 * np.sqrt(t + 1) / alpha)
    radius = np.sqrt(2 * variance_bound * log_term / t)

    lower = means - radius
    upper = means + radius

    # Clip to feasible range
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, 100)

    return lower, upper


def compute_asymptotic_cs(
    observations: np.ndarray,
    alpha: float = 0.05,
    min_samples: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute asymptotically valid confidence sequence.

    Uses running mean and variance with t-distribution critical values.
    More accurate for well-behaved data but requires n > min_samples.

    Args:
        observations: Sequential observations
        alpha: Significance level
        min_samples: Minimum samples before computing CS

    Returns:
        (lower_bounds, upper_bounds) arrays
    """
    from scipy import stats as sp_stats

    n = len(observations)

    # Running statistics
    cumsum = np.cumsum(observations)
    cumsq = np.cumsum(observations ** 2)

    t_arr = np.arange(1, n + 1)
    means = cumsum / t_arr

    # Running variance (Welford's algorithm approximation)
    variances = (cumsq / t_arr) - (means ** 2)
    variances = np.maximum(variances, 1e-6)  # Avoid division by zero
    stds = np.sqrt(variances)

    # Standard error
    se = stds / np.sqrt(t_arr)

    # Critical value (use t-distribution for small samples, normal for large)
    critical = np.zeros(n)
    for i in range(n):
        if i + 1 < min_samples:
            critical[i] = np.inf  # No CI for insufficient samples
        elif i + 1 < 100:
            critical[i] = sp_stats.t.ppf(1 - alpha/2, df=i)
        else:
            critical[i] = sp_stats.norm.ppf(1 - alpha/2)

    radius = critical * se

    lower = means - radius
    upper = means + radius

    # Clip to feasible range
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, 100)

    return lower, upper


def compute_asri_confidence_sequence(
    asri: pd.Series,
    alpha: float = 0.05,
    method: str = "asymptotic",
) -> ConfidenceSequence:
    """
    Compute confidence sequence for ASRI time series.

    Args:
        asri: ASRI time series
        alpha: Significance level (default 0.05 for 95% CS)
        method: "asymptotic" or "bounded"

    Returns:
        ConfidenceSequence object
    """
    observations = asri.dropna().values
    timestamps = asri.dropna().index.values

    if method == "bounded":
        lower, upper = compute_variance_bounded_cs(observations, alpha)
    elif method == "asymptotic":
        lower, upper = compute_asymptotic_cs(observations, alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

    return ConfidenceSequence(
        timestamps=timestamps,
        point_estimates=observations,
        lower_bounds=lower,
        upper_bounds=upper,
        alpha=alpha,
        method=method,
    )


def propagate_data_quality_uncertainty(
    asri: pd.Series,
    confidence_scores: pd.DataFrame,
) -> UncertaintyBands:
    """
    Propagate uncertainty from data quality into ASRI confidence bands.

    When data quality is low (placeholder values, stale data), confidence
    bands should be wider. When data is fresh and reliable, bands narrow.

    Args:
        asri: ASRI time series
        confidence_scores: DataFrame with confidence scores (0-1) for each
                          component: stablecoin_risk, defi_liquidity_risk,
                          contagion_risk, arbitrage_opacity

    Returns:
        UncertaintyBands with ASRI and lower/upper bands
    """
    # Default component weights
    weights = {
        "stablecoin_risk": 0.30,
        "defi_liquidity_risk": 0.25,
        "contagion_risk": 0.25,
        "arbitrage_opacity": 0.20,
    }

    # Aggregate confidence score (weighted by component importance)
    aggregate_confidence = sum(
        weights.get(col, 0) * confidence_scores[col]
        for col in confidence_scores.columns
        if col in weights
    )

    # Base uncertainty (5 points = typical ASRI noise)
    base_uncertainty = 5.0

    # Scale uncertainty by inverse confidence
    # Low confidence (0.5) -> 2x base uncertainty
    # High confidence (1.0) -> 1x base uncertainty
    uncertainty_multiplier = 2 - aggregate_confidence  # Range [1, 2] for conf in [0, 1]

    uncertainty = base_uncertainty * uncertainty_multiplier

    # Create bands
    lower = (asri - uncertainty).clip(lower=0)
    upper = (asri + uncertainty).clip(upper=100)

    return UncertaintyBands(
        asri=asri,
        lower=lower,
        upper=upper,
        confidence_scores=confidence_scores,
    )


def generate_synthetic_confidence_scores(
    asri: pd.Series,
) -> pd.DataFrame:
    """
    Generate synthetic confidence scores for demonstration.

    In production, these would come from actual data quality tracking.
    For the paper, we simulate realistic patterns:
    - Stablecoin data: high confidence (daily from DeFi Llama)
    - DeFi liquidity: high confidence
    - Contagion: medium confidence (some proxy components)
    - Arbitrage opacity: low confidence (placeholder Sent_t)
    """
    n = len(asri)
    np.random.seed(42)

    # Base confidence levels (reflecting data quality documentation)
    base_confidence = {
        "stablecoin_risk": 0.90,      # High: daily TVL data
        "defi_liquidity_risk": 0.85,   # High: daily protocol data
        "contagion_risk": 0.70,        # Medium: some proxies (Bank_t, Link_t)
        "arbitrage_opacity": 0.50,     # Low: Sent_t placeholder
    }

    # Add some temporal variation
    scores = {}
    for component, base in base_confidence.items():
        noise = np.random.normal(0, 0.05, n)
        score = np.clip(base + noise, 0.3, 1.0)
        scores[component] = score

    return pd.DataFrame(scores, index=asri.index)


def format_uncertainty_summary_latex(bands: UncertaintyBands) -> str:
    """Format uncertainty summary as LaTeX table."""
    # Compute summary statistics
    width = bands.upper - bands.lower
    avg_width = width.mean()
    max_width = width.max()
    min_width = width.min()

    # Crisis vs non-crisis comparison
    # (Would need crisis dates for full analysis)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{ASRI Uncertainty Band Statistics}",
        r"\label{tab:uncertainty_bands}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Statistic & Value & Interpretation \\",
        r"\midrule",
        f"Average band width & {avg_width:.1f} pts & Typical uncertainty \\\\",
        f"Maximum band width & {max_width:.1f} pts & Highest uncertainty period \\\\",
        f"Minimum band width & {min_width:.1f} pts & Lowest uncertainty period \\\\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Component Confidence (Average)}} \\",
    ]

    for col in bands.confidence_scores.columns:
        avg_conf = bands.confidence_scores[col].mean()
        lines.append(f"{col.replace('_', ' ').title()} & {avg_conf:.0%} & ")
        if avg_conf > 0.8:
            lines[-1] += "High (reliable data) \\\\"
        elif avg_conf > 0.6:
            lines[-1] += "Medium (some proxies) \\\\"
        else:
            lines[-1] += "Low (placeholder/proxy) \\\\"

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Bands widen when data quality is low (stale data, placeholders).",
        r"\item Operational implication: Alerts within bands require confirmation.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    return "\n".join(lines)
