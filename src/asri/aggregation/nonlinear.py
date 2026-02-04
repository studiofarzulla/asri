"""
Non-Linear Aggregation Methods for ASRI

Addresses Reviewer Q7: CES/geometric mean aggregation alternatives.

Linear aggregation (weighted sum) assumes perfect substitutability between
risk dimensions. Non-linear aggregation captures complementarity effects:
- CES: Allows varying elasticity of substitution between sub-indices
- Geometric: Penalizes extreme concentration in single risk dimension
- Min-max: Worst-case/best-case bounds

References:
- Arrow, K. J., et al. (1961). Capital-labor substitution and economic efficiency.
- Diebold, F. X., & Yilmaz, K. (2012). Better to give than to receive: Predictive
  directional measurement of volatility spillovers.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


@dataclass
class AggregationResult:
    """Results from aggregation comparison."""
    method: str
    mean: float
    std: float
    max_value: float
    min_value: float
    skewness: float
    # Crisis detection performance
    n_detected: int
    n_crises: int
    avg_lead_days: float


@dataclass
class CESParameters:
    """CES function parameters."""
    rho: float          # Substitution parameter (-∞, 1)
    weights: np.ndarray # Sub-index weights (sum to 1)
    sigma: float        # Elasticity of substitution = 1/(1-rho)


def linear_aggregate(
    sub_indices: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Standard linear weighted aggregation.

    ASRI = Σ w_i × S_i

    This is the baseline: assumes perfect substitutability.
    """
    if weights is None:
        weights = {
            "stablecoin_risk": 0.30,
            "defi_liquidity_risk": 0.25,
            "contagion_risk": 0.25,
            "arbitrage_opacity": 0.20,
        }

    result = sum(
        weights.get(col, 0) * sub_indices[col]
        for col in sub_indices.columns
        if col in weights
    )

    return pd.Series(result, index=sub_indices.index, name="asri_linear")


def ces_aggregate(
    sub_indices: pd.DataFrame,
    rho: float = 0.5,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Constant Elasticity of Substitution (CES) aggregation.

    ASRI = [Σ w_i × S_i^rho]^(1/rho)

    Parameters:
    - rho = 1: Linear (perfect substitutes)
    - rho → 0: Cobb-Douglas (geometric mean)
    - rho < 0: Complementary (amplifies when multiple risks elevated)
    - rho → -∞: Leontief (minimum determines aggregate)

    For systemic risk, rho < 0 is appropriate: risks compound when
    multiple channels are stressed simultaneously.
    """
    if weights is None:
        weights = {
            "stablecoin_risk": 0.30,
            "defi_liquidity_risk": 0.25,
            "contagion_risk": 0.25,
            "arbitrage_opacity": 0.20,
        }

    # Ensure weights sum to 1
    w = np.array([weights.get(col, 0) for col in sub_indices.columns if col in weights])
    w = w / w.sum()

    cols = [col for col in sub_indices.columns if col in weights]

    def ces_row(row):
        values = row[cols].values
        # Handle rho = 0 separately (geometric mean limit)
        if abs(rho) < 1e-6:
            return np.exp(np.sum(w * np.log(np.maximum(values, 1e-6))))

        # Standard CES formula
        # Clip values to avoid numerical issues
        values = np.maximum(values, 1e-6)

        if rho < 0:
            # For complementary goods (rho < 0), higher values dominate
            powered = np.power(values, rho)
            weighted_sum = np.sum(w * powered)
            return np.power(weighted_sum, 1/rho)
        else:
            # For substitutes (rho > 0)
            powered = np.power(values, rho)
            weighted_sum = np.sum(w * powered)
            return np.power(weighted_sum, 1/rho)

    result = sub_indices.apply(ces_row, axis=1)
    return pd.Series(result, index=sub_indices.index, name=f"asri_ces_rho{rho}")


def geometric_aggregate(
    sub_indices: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Geometric mean aggregation (Cobb-Douglas).

    ASRI = Π S_i^w_i = exp(Σ w_i × log(S_i))

    Geometric mean penalizes concentration in single risk dimension.
    If one sub-index is very low while others are high, aggregate is reduced.
    """
    if weights is None:
        weights = {
            "stablecoin_risk": 0.30,
            "defi_liquidity_risk": 0.25,
            "contagion_risk": 0.25,
            "arbitrage_opacity": 0.20,
        }

    cols = [col for col in sub_indices.columns if col in weights]
    w = np.array([weights[col] for col in cols])
    w = w / w.sum()

    def geom_row(row):
        values = row[cols].values
        # Clip to avoid log(0)
        values = np.maximum(values, 1e-6)
        return np.exp(np.sum(w * np.log(values)))

    result = sub_indices.apply(geom_row, axis=1)
    return pd.Series(result, index=sub_indices.index, name="asri_geometric")


def minmax_aggregate(
    sub_indices: pd.DataFrame,
    weights: dict[str, float] | None = None,
    mode: Literal["min", "max", "range"] = "max",
) -> pd.Series:
    """
    Min/max aggregation (extreme case).

    - mode="max": ASRI = max(S_i) - worst-case risk
    - mode="min": ASRI = min(S_i) - best-case risk
    - mode="range": ASRI = max(S_i) - min(S_i) - dispersion
    """
    if weights is None:
        weights = {
            "stablecoin_risk": 0.30,
            "defi_liquidity_risk": 0.25,
            "contagion_risk": 0.25,
            "arbitrage_opacity": 0.20,
        }

    cols = [col for col in sub_indices.columns if col in weights]

    if mode == "max":
        result = sub_indices[cols].max(axis=1)
        name = "asri_max"
    elif mode == "min":
        result = sub_indices[cols].min(axis=1)
        name = "asri_min"
    elif mode == "range":
        result = sub_indices[cols].max(axis=1) - sub_indices[cols].min(axis=1)
        name = "asri_range"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return pd.Series(result, index=sub_indices.index, name=name)


def fit_ces_rho(
    sub_indices: pd.DataFrame,
    crisis_dates: list,
    threshold: float = 50.0,
    search_range: tuple[float, float] = (-2.0, 1.0),
) -> float:
    """
    Fit optimal CES rho parameter to maximize crisis detection.

    Searches for rho that:
    1. Detects all crises (recall = 1)
    2. Maximizes lead time while maintaining detection

    Args:
        sub_indices: DataFrame with sub-index columns
        crisis_dates: List of datetime objects for crisis events
        threshold: Detection threshold
        search_range: Range of rho to search

    Returns:
        Optimal rho value
    """
    def detection_score(rho):
        """Negative score for minimization (we want to maximize detection)."""
        try:
            asri = ces_aggregate(sub_indices, rho=rho)

            # Count detections and lead time
            total_lead = 0
            detected = 0

            for crisis_date in crisis_dates:
                crisis_ts = pd.Timestamp(crisis_date)
                window_start = crisis_ts - pd.Timedelta(days=30)
                pre_crisis = asri[(asri.index >= window_start) & (asri.index < crisis_ts)]

                if len(pre_crisis) > 0 and pre_crisis.max() >= threshold:
                    detected += 1
                    # Find first detection day
                    detections = pre_crisis[pre_crisis >= threshold]
                    if len(detections) > 0:
                        first_detect = detections.index.min()
                        lead = (crisis_ts - first_detect).days
                        total_lead += lead

            # Score: prioritize detection count, then lead time
            # Return negative for minimization
            if detected == 0:
                return 1000  # Penalty for no detection

            avg_lead = total_lead / detected if detected > 0 else 0
            return -(detected * 100 + avg_lead)

        except Exception:
            return 1000  # Penalty for errors

    # Grid search (more robust than continuous optimization for this discrete problem)
    rho_values = np.linspace(search_range[0], search_range[1], 50)
    scores = [detection_score(rho) for rho in rho_values]
    best_idx = np.argmin(scores)

    return rho_values[best_idx]


def compare_aggregation_methods(
    sub_indices: pd.DataFrame,
    crisis_dates: list,
    threshold: float = 50.0,
) -> list[AggregationResult]:
    """
    Compare different aggregation methods.

    Args:
        sub_indices: DataFrame with sub-index columns
        crisis_dates: List of crisis dates
        threshold: Detection threshold

    Returns:
        List of AggregationResult for each method
    """
    from scipy.stats import skew

    methods = [
        ("Linear", linear_aggregate(sub_indices)),
        ("CES (ρ=0.5)", ces_aggregate(sub_indices, rho=0.5)),
        ("CES (ρ=0)", ces_aggregate(sub_indices, rho=0.0)),  # Cobb-Douglas
        ("CES (ρ=-0.5)", ces_aggregate(sub_indices, rho=-0.5)),
        ("CES (ρ=-1.0)", ces_aggregate(sub_indices, rho=-1.0)),
        ("Geometric", geometric_aggregate(sub_indices)),
        ("Max", minmax_aggregate(sub_indices, mode="max")),
    ]

    results = []

    for name, asri in methods:
        # Basic stats
        stats = {
            "method": name,
            "mean": asri.mean(),
            "std": asri.std(),
            "max_value": asri.max(),
            "min_value": asri.min(),
            "skewness": skew(asri.dropna()),
        }

        # Crisis detection
        detected = 0
        total_lead = 0

        for crisis_date in crisis_dates:
            crisis_ts = pd.Timestamp(crisis_date)
            window_start = crisis_ts - pd.Timedelta(days=30)
            pre_crisis = asri[(asri.index >= window_start) & (asri.index < crisis_ts)]

            if len(pre_crisis) > 0 and pre_crisis.max() >= threshold:
                detected += 1
                detections = pre_crisis[pre_crisis >= threshold]
                if len(detections) > 0:
                    first_detect = detections.index.min()
                    lead = (crisis_ts - first_detect).days
                    total_lead += lead

        stats["n_detected"] = detected
        stats["n_crises"] = len(crisis_dates)
        stats["avg_lead_days"] = total_lead / detected if detected > 0 else 0

        results.append(AggregationResult(**stats))

    return results


def format_aggregation_comparison_latex(results: list[AggregationResult]) -> str:
    """Format aggregation comparison as LaTeX table."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Aggregation Method Comparison}",
        r"\label{tab:aggregation_comparison}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Mean & Std & Max & Skew & Detection & Lead (days) \\",
        r"\midrule",
    ]

    for r in results:
        lines.append(
            f"{r.method} & {r.mean:.1f} & {r.std:.1f} & {r.max_value:.1f} & "
            f"{r.skewness:.2f} & {r.n_detected}/{r.n_crises} & "
            f"{r.avg_lead_days:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item CES($\rho$) = Constant Elasticity of Substitution with parameter $\rho$.",
        r"\item $\rho = 1$: linear; $\rho = 0$: geometric (Cobb-Douglas); $\rho < 0$: complementary.",
        r"\item Detection = crises with ASRI $\geq 50$ in 30-day pre-crisis window.",
        r"\item Lead = average days between first detection and crisis onset.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    return "\n".join(lines)
