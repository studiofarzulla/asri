"""
Bootstrap Confidence Intervals for ASRI

Uncertainty quantification is essential for credible risk indices.
This module implements block bootstrap methods appropriate for
time series data (which violates i.i.d. assumptions).

Key insight: Standard bootstrap fails for time series because it
destroys autocorrelation structure. Block bootstrap preserves it.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class ConfidenceInterval:
    """Confidence interval with metadata."""
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str
    n_bootstrap: int
    
    def __str__(self) -> str:
        return f"{self.point_estimate:.4f} [{self.lower:.4f}, {self.upper:.4f}] ({self.confidence_level*100:.0f}% CI)"
    
    @property
    def width(self) -> float:
        return self.upper - self.lower
    
    @property
    def margin_of_error(self) -> float:
        return self.width / 2


@dataclass
class ASRIDistribution:
    """Full bootstrap distribution for ASRI."""
    date: pd.Timestamp | None
    point_estimate: float
    bootstrap_samples: np.ndarray
    confidence_intervals: dict[float, ConfidenceInterval]  # level -> CI
    
    @property
    def standard_error(self) -> float:
        return np.std(self.bootstrap_samples)
    
    @property
    def bias(self) -> float:
        return np.mean(self.bootstrap_samples) - self.point_estimate


def _moving_block_bootstrap_indices(
    n: int,
    block_size: int,
    n_samples: int,
    random_state: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate indices for moving block bootstrap.
    
    The moving block bootstrap:
    1. Divides the series into overlapping blocks of length block_size
    2. Randomly samples blocks with replacement
    3. Concatenates blocks to form bootstrap samples
    
    Args:
        n: Length of original series
        block_size: Size of each block (should capture autocorrelation)
        n_samples: Number of bootstrap samples to generate
        random_state: Random generator for reproducibility
        
    Returns:
        Array of shape (n_samples, n) with bootstrap indices
    """
    if random_state is None:
        rng = np.random.default_rng()
    else:
        rng = random_state
    
    # Number of blocks needed to cover n observations
    n_blocks = int(np.ceil(n / block_size))
    
    # Possible starting positions for blocks
    max_start = n - block_size + 1
    if max_start < 1:
        max_start = 1
        block_size = n  # Use entire series as one block
    
    all_indices = np.zeros((n_samples, n), dtype=int)
    
    for i in range(n_samples):
        # Randomly select block starting positions
        starts = rng.integers(0, max_start, size=n_blocks)
        
        # Generate indices from blocks
        indices = []
        for start in starts:
            block_indices = np.arange(start, min(start + block_size, n))
            indices.extend(block_indices)
        
        # Trim to exactly n
        all_indices[i] = np.array(indices[:n])
    
    return all_indices


def _stationary_bootstrap_indices(
    n: int,
    expected_block_size: float,
    n_samples: int,
    random_state: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate indices for stationary bootstrap (Politis & Romano, 1994).
    
    Unlike moving block, block lengths are random (geometric distribution).
    This produces stationary bootstrap samples.
    
    Args:
        n: Length of original series
        expected_block_size: Expected block length (mean of geometric dist)
        n_samples: Number of bootstrap samples
        random_state: Random generator
        
    Returns:
        Array of shape (n_samples, n) with bootstrap indices
    """
    if random_state is None:
        rng = np.random.default_rng()
    else:
        rng = random_state
    
    # Probability of starting new block
    p = 1.0 / expected_block_size
    
    all_indices = np.zeros((n_samples, n), dtype=int)
    
    for i in range(n_samples):
        indices = []
        current_pos = rng.integers(0, n)
        
        while len(indices) < n:
            indices.append(current_pos)
            
            # Decide whether to continue block or start new one
            if rng.random() < p:
                # Start new block at random position
                current_pos = rng.integers(0, n)
            else:
                # Continue current block
                current_pos = (current_pos + 1) % n
        
        all_indices[i] = np.array(indices[:n])
    
    return all_indices


def block_bootstrap_ci(
    series: pd.Series | np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    block_size: int | None = None,
    confidence_level: float = 0.95,
    method: str = "percentile",
    random_state: int | None = None,
) -> ConfidenceInterval:
    """
    Compute confidence interval using block bootstrap.
    
    Args:
        series: Time series data
        statistic: Function that computes the statistic of interest
        n_bootstrap: Number of bootstrap replications
        block_size: Block size (default: n^(1/3) as per theory)
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        method: CI method ('percentile', 'basic', 'bca')
        random_state: Random seed for reproducibility
        
    Returns:
        ConfidenceInterval with point estimate and bounds
    """
    if isinstance(series, pd.Series):
        data = series.dropna().values
    else:
        data = series[~np.isnan(series)]
    
    n = len(data)
    
    # Default block size: n^(1/3) is optimal under certain conditions
    if block_size is None:
        block_size = max(1, int(n ** (1/3)))
    
    # Point estimate
    theta_hat = statistic(data)
    
    # Generate bootstrap samples
    rng = np.random.default_rng(random_state)
    indices = _moving_block_bootstrap_indices(n, block_size, n_bootstrap, rng)
    
    # Compute statistic for each bootstrap sample
    theta_boot = np.array([statistic(data[idx]) for idx in indices])
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    
    if method == "percentile":
        lower = np.percentile(theta_boot, 100 * alpha / 2)
        upper = np.percentile(theta_boot, 100 * (1 - alpha / 2))
    
    elif method == "basic":
        # Basic bootstrap: reflects around point estimate
        lower = 2 * theta_hat - np.percentile(theta_boot, 100 * (1 - alpha / 2))
        upper = 2 * theta_hat - np.percentile(theta_boot, 100 * alpha / 2)
    
    elif method == "bca":
        # Bias-corrected and accelerated (BCa)
        # More accurate but computationally intensive
        lower, upper = _bca_interval(data, statistic, theta_boot, theta_hat, alpha)
    
    else:
        raise ValueError(f"Unknown CI method: {method}")
    
    return ConfidenceInterval(
        point_estimate=theta_hat,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        method=f"block_bootstrap_{method}",
        n_bootstrap=n_bootstrap,
    )


def _bca_interval(
    data: np.ndarray,
    statistic: Callable,
    theta_boot: np.ndarray,
    theta_hat: float,
    alpha: float,
) -> tuple[float, float]:
    """Compute BCa confidence interval."""
    from scipy import stats as sp_stats
    
    n = len(data)
    
    # Bias correction factor
    z0 = sp_stats.norm.ppf(np.mean(theta_boot < theta_hat))
    
    # Acceleration factor (using jackknife)
    theta_jack = np.array([
        statistic(np.delete(data, i)) for i in range(n)
    ])
    theta_jack_mean = np.mean(theta_jack)
    
    num = np.sum((theta_jack_mean - theta_jack) ** 3)
    denom = 6 * (np.sum((theta_jack_mean - theta_jack) ** 2) ** 1.5)
    
    a = num / denom if denom != 0 else 0
    
    # Adjusted percentiles
    z_alpha_lower = sp_stats.norm.ppf(alpha / 2)
    z_alpha_upper = sp_stats.norm.ppf(1 - alpha / 2)
    
    def adjusted_percentile(z_alpha):
        num = z0 + z_alpha
        denom = 1 - a * num
        if denom == 0:
            return 0.5
        return sp_stats.norm.cdf(z0 + num / denom)
    
    lower_pct = adjusted_percentile(z_alpha_lower)
    upper_pct = adjusted_percentile(z_alpha_upper)
    
    lower = np.percentile(theta_boot, 100 * lower_pct)
    upper = np.percentile(theta_boot, 100 * upper_pct)
    
    return lower, upper


def bootstrap_asri_distribution(
    sub_indices: pd.DataFrame,
    weights: dict[str, float],
    n_bootstrap: int = 1000,
    block_size: int | None = None,
    confidence_levels: list[float] = [0.90, 0.95, 0.99],
    random_state: int | None = None,
) -> ASRIDistribution:
    """
    Compute full bootstrap distribution for ASRI.
    
    This propagates uncertainty from sub-indices through to the
    aggregate index, providing confidence intervals for ASRI.
    
    Args:
        sub_indices: DataFrame with sub-index time series
        weights: Dictionary of sub-index weights
        n_bootstrap: Number of bootstrap replications
        block_size: Block size for block bootstrap
        confidence_levels: List of confidence levels to compute
        random_state: Random seed
        
    Returns:
        ASRIDistribution with bootstrap samples and CIs
    """
    # Compute point estimate
    asri_point = sum(
        weights.get(col, 0) * sub_indices[col].iloc[-1]
        for col in sub_indices.columns
    )
    
    # Align and get data
    data = sub_indices.dropna()
    n = len(data)
    
    if block_size is None:
        block_size = max(1, int(n ** (1/3)))
    
    rng = np.random.default_rng(random_state)
    indices = _moving_block_bootstrap_indices(n, block_size, n_bootstrap, rng)
    
    # Bootstrap ASRI values
    asri_boot = np.zeros(n_bootstrap)
    
    for i, idx in enumerate(indices):
        boot_sample = data.iloc[idx]
        # Use the last value from bootstrap sample as the "current" ASRI
        asri_boot[i] = sum(
            weights.get(col, 0) * boot_sample[col].iloc[-1]
            for col in data.columns
        )
    
    # Compute confidence intervals at each level
    cis = {}
    for level in confidence_levels:
        alpha = 1 - level
        lower = np.percentile(asri_boot, 100 * alpha / 2)
        upper = np.percentile(asri_boot, 100 * (1 - alpha / 2))
        
        cis[level] = ConfidenceInterval(
            point_estimate=asri_point,
            lower=lower,
            upper=upper,
            confidence_level=level,
            method="block_bootstrap_percentile",
            n_bootstrap=n_bootstrap,
        )
    
    return ASRIDistribution(
        date=data.index[-1] if hasattr(data.index[-1], 'date') else None,
        point_estimate=asri_point,
        bootstrap_samples=asri_boot,
        confidence_intervals=cis,
    )


def compute_rolling_confidence_bands(
    sub_indices: pd.DataFrame,
    weights: dict[str, float],
    window: int = 90,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Compute rolling confidence bands for ASRI time series.
    
    This provides uncertainty bands that can be plotted alongside
    the point estimate ASRI.
    
    Args:
        sub_indices: DataFrame with sub-index time series
        weights: Sub-index weights
        window: Rolling window size for bootstrap
        n_bootstrap: Bootstrap replications per window
        confidence_level: Confidence level for bands
        
    Returns:
        DataFrame with columns: asri, lower, upper
    """
    results = []
    
    for i in range(window, len(sub_indices)):
        window_data = sub_indices.iloc[i-window:i]
        
        dist = bootstrap_asri_distribution(
            window_data,
            weights,
            n_bootstrap=n_bootstrap,
            confidence_levels=[confidence_level],
        )
        
        ci = dist.confidence_intervals[confidence_level]
        
        results.append({
            'date': sub_indices.index[i-1],
            'asri': ci.point_estimate,
            'lower': ci.lower,
            'upper': ci.upper,
            'std_error': dist.standard_error,
        })
    
    return pd.DataFrame(results).set_index('date')


def format_ci_table(
    results: dict[str, ConfidenceInterval],
    caption: str = "ASRI Confidence Intervals",
) -> str:
    """Format confidence intervals as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        r"\label{tab:confidence_intervals}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Variable & Point Estimate & 95\% CI Lower & 95\% CI Upper \\",
        r"\midrule",
    ]
    
    for name, ci in results.items():
        lines.append(
            f"{name} & {ci.point_estimate:.2f} & {ci.lower:.2f} & {ci.upper:.2f} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        f"\\item Block bootstrap with {list(results.values())[0].n_bootstrap} replications.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)
