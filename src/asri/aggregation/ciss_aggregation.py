"""
CISS-Style Portfolio Aggregation for ASRI

Implements the ECB CISS (Composite Indicator of Systemic Stress) aggregation
methodology, which uses portfolio-theoretic principles to capture systemic
amplification when risk sub-indices become correlated.

Key insight: Linear weighted sums assume independence between sub-indices.
Portfolio variance aggregation naturally amplifies when correlations spike—
precisely when systemic stress materializes. During crises, assets crash
together; this methodology captures that contagion dynamic mathematically.

The aggregation formula:
    ASRI_ciss = sqrt(s' @ C_t @ s)

Where:
    s = vector of 4 sub-index values at time t
    C_t = time-varying correlation matrix (4x4), estimated via EWMA

References:
    Hollo, D., Kremer, M., & Lo Duca, M. (2012). CISS - A Composite Indicator
    of Systemic Stress in the Financial System. ECB Working Paper No. 1426.
    https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1426.pdf

    RiskMetrics Technical Document (1996). J.P. Morgan.
    (For EWMA correlation methodology with lambda = 0.94)
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class CISSResult:
    """Results from CISS-style aggregation."""

    asri_ciss: pd.Series  # Time series of CISS-aggregated ASRI
    asri_linear: pd.Series  # Time series of linear-weighted ASRI (for comparison)
    correlation_contribution: pd.Series  # Additional stress from correlation
    mean_correlation: pd.Series  # Average pairwise correlation over time

    # Final correlation matrix (at last timestamp)
    final_correlation_matrix: pd.DataFrame


class CISSAggregator:
    """
    Portfolio-theoretic aggregation following ECB CISS methodology.

    The core mechanism: when sub-indices are uncorrelated, CISS aggregation
    approaches the sum of squared values (diversification benefit). When
    sub-indices spike together (correlation -> 1), the off-diagonal terms
    amplify the result—capturing systemic contagion.

    This is mathematically equivalent to computing portfolio volatility
    where sub-indices are "assets" and we care about total risk.
    """

    # Canonical sub-index column names (standardized across all modules)
    CANONICAL_COLUMNS = [
        "stablecoin_risk",
        "defi_liquidity_risk",
        "contagion_risk",
        "arbitrage_opacity",
    ]

    # Legacy alias for backwards compatibility
    SUBINDEX_COLUMNS = CANONICAL_COLUMNS

    # Default linear weights (from ASRI specification)
    DEFAULT_WEIGHTS = np.array([0.30, 0.25, 0.25, 0.20])

    def __init__(
        self,
        decay_factor: float = 0.94,
        use_equal_weights: bool = True,
    ):
        """
        Initialize CISS aggregator.

        Args:
            decay_factor: EWMA decay parameter (lambda). Higher values give
                more weight to recent observations. RiskMetrics standard is
                0.94 for daily data, 0.97 for monthly. ECB CISS uses 0.93.
            use_equal_weights: If True, treat all sub-indices equally in the
                portfolio (as per original CISS). If False, use DEFAULT_WEIGHTS
                to scale sub-indices before aggregation.
        """
        if not 0 < decay_factor < 1:
            raise ValueError(f"decay_factor must be in (0, 1), got {decay_factor}")

        self.decay_factor = decay_factor
        self.use_equal_weights = use_equal_weights
        self._correlation_history: list[np.ndarray] = []

    def compute_ewma_covariance(
        self,
        returns: pd.DataFrame,
        min_periods: int = 30,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Compute time-varying covariance and correlation matrices via EWMA.

        The EWMA covariance estimator:
            Sigma_t = lambda * Sigma_{t-1} + (1 - lambda) * r_t @ r_t'

        where r_t is the return (or change) vector at time t.

        This approach has several advantages over rolling windows:
        - Smoother transitions (no "cliff effect" when old data drops out)
        - More weight on recent data (appropriate for non-stationary processes)
        - Computationally efficient (recursive update)

        Args:
            returns: DataFrame of sub-index changes/returns. Should be
                stationary (e.g., first differences or log returns).
            min_periods: Minimum observations before producing valid output.
                Earlier periods use expanding window initialization.

        Returns:
            Tuple of (covariance_matrices, correlation_matrices) where each
            is a list of numpy arrays, one per timestamp.
        """
        n_obs, n_vars = returns.shape
        lambda_ = self.decay_factor

        covariance_matrices = []
        correlation_matrices = []

        # Initialize with expanding window estimate
        running_cov = np.zeros((n_vars, n_vars))

        for t in range(n_obs):
            r_t = returns.iloc[t].values

            # Handle NaN: skip if any NaN in this row
            if np.any(np.isnan(r_t)):
                # Carry forward previous estimate
                if covariance_matrices:
                    covariance_matrices.append(covariance_matrices[-1].copy())
                    correlation_matrices.append(correlation_matrices[-1].copy())
                else:
                    covariance_matrices.append(np.full((n_vars, n_vars), np.nan))
                    correlation_matrices.append(np.full((n_vars, n_vars), np.nan))
                continue

            # Outer product for rank-1 update
            outer = np.outer(r_t, r_t)

            if t < min_periods:
                # Initialization phase: use expanding window
                running_cov = (t / (t + 1)) * running_cov + (1 / (t + 1)) * outer
            else:
                # EWMA update
                running_cov = lambda_ * running_cov + (1 - lambda_) * outer

            covariance_matrices.append(running_cov.copy())

            # Convert covariance to correlation
            std = np.sqrt(np.diag(running_cov))
            std_outer = np.outer(std, std)

            # Avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = np.where(std_outer > 0, running_cov / std_outer, 0.0)

            # Ensure diagonal is exactly 1
            np.fill_diagonal(corr, 1.0)

            # Clip correlations to valid range (numerical stability)
            corr = np.clip(corr, -1.0, 1.0)

            # Apply shrinkage regularization to ensure positive semi-definiteness
            # This guarantees the matrix is always valid for portfolio variance calculation
            shrinkage = 0.01
            corr = (1 - shrinkage) * corr + shrinkage * np.eye(n_vars)

            correlation_matrices.append(corr)

        return covariance_matrices, correlation_matrices

    def aggregate_single(
        self,
        subindices: np.ndarray,
        correlation_matrix: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        """
        Single timestep CISS aggregation.

        Formula: sqrt(s' @ C @ s) where s is the (optionally weighted) sub-index
        vector and C is the correlation matrix.

        The geometric interpretation: this is the "portfolio standard deviation"
        treating sub-indices as asset returns and correlations as the dependence
        structure. When correlations spike, portfolio risk spikes.

        Args:
            subindices: Array of 4 sub-index values (already 0-100 scaled)
            correlation_matrix: 4x4 correlation matrix
            weights: Optional weights to apply to sub-indices before aggregation.
                If None and use_equal_weights=False, uses DEFAULT_WEIGHTS.

        Returns:
            CISS-aggregated ASRI value (before final scaling)
        """
        s = np.asarray(subindices).flatten()
        C = np.asarray(correlation_matrix)

        # Handle NaN
        if np.any(np.isnan(s)) or np.any(np.isnan(C)):
            return np.nan

        # Apply weights if specified
        if weights is not None:
            s = s * weights
        elif not self.use_equal_weights:
            s = s * self.DEFAULT_WEIGHTS

        # Portfolio variance: s' @ C @ s
        # Note: using correlation (not covariance) means s values act as both
        # "positions" and "volatilities" - appropriate for stress indices
        variance = s @ C @ s

        # Return standard deviation (sqrt of variance)
        # This ensures the output scales linearly with input magnitudes
        return np.sqrt(max(0, variance))

    def compute_asri_series(
        self,
        subindex_df: pd.DataFrame,
        normalize: bool = True,
        normalization_method: Literal["expanding", "fixed", "none"] = "expanding",
    ) -> pd.Series:
        """
        Compute full time series of CISS-aggregated ASRI.

        The pipeline:
        1. Compute first differences of sub-indices (for correlation estimation)
        2. Build time-varying correlation matrices via EWMA
        3. Apply portfolio aggregation formula at each timestamp
        4. Normalize to 0-100 scale (avoiding look-ahead bias)

        Args:
            subindex_df: DataFrame with sub-index columns. Expected columns:
                ['stablecoin_risk', 'defi_liquidity', 'contagion', 'opacity']
                Values should be 0-100 scaled.
            normalize: Whether to normalize output to 0-100 scale
            normalization_method:
                - "expanding": Use expanding min-max (no look-ahead bias)
                - "fixed": Use full-sample min-max (for backtesting only)
                - "none": Return raw aggregated values

        Returns:
            Series of CISS-aggregated ASRI values, indexed same as input
        """
        # Validate columns
        df = self._validate_and_prepare(subindex_df)

        # Compute first differences for correlation estimation
        # (correlations should be computed on stationary series)
        returns = df.diff()

        # Track original indices before dropna() to maintain alignment
        returns_clean = returns.dropna()
        valid_indices = returns_clean.index

        # Get time-varying correlations
        _, correlation_matrices = self.compute_ewma_covariance(
            returns_clean,
            min_periods=30,
        )

        # Store for later analysis
        self._correlation_history = correlation_matrices

        # Build index-aligned correlation lookup
        # Maps original df index to correlation matrix
        correlation_by_index = {
            idx: correlation_matrices[i]
            for i, idx in enumerate(valid_indices)
        }

        # Compute CISS aggregation at each timestamp
        raw_asri = []

        for idx in df.index:
            if idx not in correlation_by_index:
                # No correlation estimate for this index (first row or NaN gaps)
                raw_asri.append(np.nan)
            else:
                corr_matrix = correlation_by_index[idx]
                subindices = df.loc[idx].values
                asri_t = self.aggregate_single(subindices, corr_matrix)
                raw_asri.append(asri_t)

        raw_series = pd.Series(raw_asri, index=df.index, name="asri_ciss")

        # Normalize
        if normalize and normalization_method != "none":
            return self._normalize_series(raw_series, normalization_method)

        return raw_series

    def _validate_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate input DataFrame and standardize column names to CANONICAL_COLUMNS."""
        # Allow flexible column naming - maps to canonical names
        column_mapping = {
            "stablecoin_risk": ["stablecoin_risk", "scr", "stablecoin"],
            "defi_liquidity_risk": ["defi_liquidity_risk", "defi_liquidity", "dlr", "defi"],
            "contagion_risk": ["contagion_risk", "contagion", "cr"],
            "arbitrage_opacity": ["arbitrage_opacity", "opacity", "or", "regulatory_opacity"],
        }

        # Try to find matching columns
        final_columns = {}
        for standard_name, variants in column_mapping.items():
            for variant in variants:
                if variant in df.columns:
                    final_columns[variant] = standard_name
                    break
            else:
                # Check case-insensitive
                for col in df.columns:
                    if col.lower() in [v.lower() for v in variants]:
                        final_columns[col] = standard_name
                        break

        if len(final_columns) != 4:
            found = list(final_columns.keys())
            raise ValueError(
                f"Expected 4 sub-index columns, found {len(found)}: {found}. "
                f"Required: {list(column_mapping.keys())}"
            )

        # Rename and reorder
        result = df[list(final_columns.keys())].rename(columns=final_columns)
        return result[self.SUBINDEX_COLUMNS]

    def _normalize_series(
        self,
        series: pd.Series,
        method: Literal["expanding", "fixed"],
    ) -> pd.Series:
        """
        Normalize to 0-100 scale.

        Expanding normalization avoids look-ahead bias: at each time t, we only
        use information available up to t. This is crucial for proper backtesting.
        """
        if method == "fixed":
            # Full sample normalization (look-ahead bias - use for visualization only)
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                return pd.Series(50.0, index=series.index, name=series.name)
            return 100 * (series - min_val) / (max_val - min_val)

        else:  # expanding
            # Expanding min-max (no look-ahead)
            expanding_min = series.expanding().min()
            expanding_max = series.expanding().max()

            # Avoid division by zero
            range_ = expanding_max - expanding_min
            range_ = range_.replace(0, np.nan)

            normalized = 100 * (series - expanding_min) / range_

            # First valid value gets 50 (midpoint) since no range established
            first_valid = normalized.first_valid_index()
            if first_valid is not None and pd.isna(normalized.loc[first_valid]):
                normalized.loc[first_valid] = 50.0

            return normalized

    def compare_with_linear(
        self,
        subindex_df: pd.DataFrame,
        linear_weights: np.ndarray | None = None,
        normalize: bool = True,
    ) -> CISSResult:
        """
        Compare CISS aggregation with linear weighted sum.

        This is the key diagnostic: when do these methods diverge? Divergence
        indicates correlation-driven amplification—exactly what we want to
        capture as "systemic" stress.

        Args:
            subindex_df: DataFrame with sub-index columns
            linear_weights: Weights for linear aggregation. Default is
                [0.30, 0.25, 0.25, 0.20] (ASRI specification).
            normalize: Whether to normalize outputs to 0-100 scale. Set False
                for fair comparison with other methods that don't normalize.

        Returns:
            CISSResult with both series and diagnostic metrics
        """
        if linear_weights is None:
            linear_weights = self.DEFAULT_WEIGHTS
        linear_weights = np.asarray(linear_weights)

        # Validate
        df = self._validate_and_prepare(subindex_df)

        # Compute CISS series (stores correlation history)
        asri_ciss = self.compute_asri_series(df, normalize=normalize)

        # Compute linear series
        asri_linear = (df.values @ linear_weights)
        asri_linear = pd.Series(asri_linear, index=df.index, name="asri_linear")

        # Normalize linear to 0-100 (expanding, for fair comparison) if requested
        if normalize:
            asri_linear_norm = self._normalize_series(asri_linear, "expanding")
        else:
            asri_linear_norm = asri_linear

        # Correlation contribution: how much extra stress from correlations
        # When correlations = 0, CISS ≈ sqrt(sum of squares)
        # When correlations = 1, CISS ≈ sum (maximum amplification)
        correlation_contribution = asri_ciss - asri_linear_norm
        correlation_contribution.name = "correlation_contribution"

        # Build index-aligned mean correlation series
        # First, get valid indices from the returns (after diff and dropna)
        returns_clean = df.diff().dropna()
        valid_indices = returns_clean.index

        # Create correlation lookup by index
        corr_by_index = {
            idx: self._correlation_history[i]
            for i, idx in enumerate(valid_indices)
        }

        # Compute mean pairwise correlation for each timestamp
        mean_corr = []
        for idx in df.index:
            if idx not in corr_by_index:
                mean_corr.append(np.nan)
            else:
                corr_matrix = corr_by_index[idx]
                if np.any(np.isnan(corr_matrix)):
                    mean_corr.append(np.nan)
                else:
                    # Extract off-diagonal elements
                    mask = ~np.eye(4, dtype=bool)
                    off_diag = corr_matrix[mask]
                    mean_corr.append(np.mean(off_diag))

        mean_corr_series = pd.Series(
            mean_corr, index=df.index, name="mean_correlation"
        )

        # Final correlation matrix
        final_corr = pd.DataFrame(
            self._correlation_history[-1] if self._correlation_history else np.eye(4),
            index=self.SUBINDEX_COLUMNS,
            columns=self.SUBINDEX_COLUMNS,
        )

        return CISSResult(
            asri_ciss=asri_ciss,
            asri_linear=asri_linear_norm,
            correlation_contribution=correlation_contribution,
            mean_correlation=mean_corr_series,
            final_correlation_matrix=final_corr,
        )

    def get_correlation_at(self, t: int) -> np.ndarray:
        """Get correlation matrix at time index t (after first diff)."""
        if not self._correlation_history:
            raise ValueError("No correlation history. Run compute_asri_series first.")
        if t < 0 or t >= len(self._correlation_history):
            raise IndexError(f"t={t} out of range [0, {len(self._correlation_history)})")
        return self._correlation_history[t]


def format_ciss_comparison_table(result: CISSResult) -> str:
    """Format CISS vs Linear comparison as LaTeX table."""
    # Summary statistics
    ciss = result.asri_ciss.dropna()
    linear = result.asri_linear.dropna()
    contrib = result.correlation_contribution.dropna()
    mean_corr = result.mean_correlation.dropna()

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{CISS vs Linear Aggregation Comparison}",
        r"\label{tab:ciss-comparison}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Statistic & CISS Aggregation & Linear Aggregation \\",
        r"\midrule",
        f"Mean & {ciss.mean():.2f} & {linear.mean():.2f} \\\\",
        f"Std Dev & {ciss.std():.2f} & {linear.std():.2f} \\\\",
        f"Min & {ciss.min():.2f} & {linear.min():.2f} \\\\",
        f"Max & {ciss.max():.2f} & {linear.max():.2f} \\\\",
        f"Skewness & {ciss.skew():.2f} & {linear.skew():.2f} \\\\",
        r"\midrule",
        f"Mean Correlation Contribution & \\multicolumn{{2}}{{c}}{{{contrib.mean():.2f}}} \\\\",
        f"Max Correlation Contribution & \\multicolumn{{2}}{{c}}{{{contrib.max():.2f}}} \\\\",
        f"Mean Pairwise Correlation & \\multicolumn{{2}}{{c}}{{{mean_corr.mean():.3f}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item CISS uses portfolio-theoretic aggregation with EWMA correlations ($\lambda=0.94$).",
        r"\item Correlation contribution = CISS ASRI - Linear ASRI.",
        r"\item Positive contribution indicates systemic amplification from correlated stress.",
        r"\end{tablenotes}",
        r"\end{table}",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    """
    Demonstration with synthetic data.

    Creates two scenarios:
    1. Normal period: low, uncorrelated sub-indices
    2. Crisis period: high, correlated sub-indices

    Shows how CISS amplifies during crises while linear doesn't.
    """
    np.random.seed(42)

    # Generate 500 days of synthetic data
    n_days = 500
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    # Base stress levels (0-100 scale)
    base_stress = 30  # Normal regime
    crisis_stress = 70  # Crisis regime

    # Create regime indicator (crisis from day 200-300)
    regime = np.zeros(n_days)
    regime[200:300] = 1  # Crisis period

    # Generate sub-indices with regime-dependent correlation
    data = np.zeros((n_days, 4))

    for t in range(n_days):
        if regime[t] == 0:
            # Normal: low correlation, moderate stress
            mean = base_stress * np.ones(4) + np.random.randn(4) * 5
            # Low correlation via independent noise
            noise = np.random.randn(4) * 8
        else:
            # Crisis: high correlation, high stress
            mean = crisis_stress * np.ones(4) + np.random.randn(4) * 3
            # High correlation via common factor
            common_factor = np.random.randn() * 10
            idiosyncratic = np.random.randn(4) * 3
            noise = common_factor + idiosyncratic

        data[t] = np.clip(mean + noise, 0, 100)

    # Create DataFrame with canonical column names
    df = pd.DataFrame(
        data,
        index=dates,
        columns=CISSAggregator.CANONICAL_COLUMNS,
    )

    print("=" * 70)
    print("CISS Aggregation Demonstration")
    print("=" * 70)
    print(f"\nSynthetic data: {n_days} days")
    print("- Normal period: days 1-199, 301-500 (low stress, low correlation)")
    print("- Crisis period: days 200-300 (high stress, HIGH correlation)")
    print()

    # Compute CISS aggregation
    aggregator = CISSAggregator(decay_factor=0.94)
    result = aggregator.compare_with_linear(df)

    # Summary statistics
    print("Summary Statistics:")
    print("-" * 50)
    print(f"CISS ASRI:   mean={result.asri_ciss.mean():.1f}, "
          f"std={result.asri_ciss.std():.1f}, "
          f"max={result.asri_ciss.max():.1f}")
    print(f"Linear ASRI: mean={result.asri_linear.mean():.1f}, "
          f"std={result.asri_linear.std():.1f}, "
          f"max={result.asri_linear.max():.1f}")
    print()

    # Crisis period analysis
    crisis_mask = (dates >= "2020-07-19") & (dates <= "2020-10-27")  # ~days 200-300

    print("Crisis Period Analysis (days 200-300):")
    print("-" * 50)
    crisis_ciss = result.asri_ciss[crisis_mask].mean()
    crisis_linear = result.asri_linear[crisis_mask].mean()
    crisis_contrib = result.correlation_contribution[crisis_mask].mean()
    crisis_corr = result.mean_correlation[crisis_mask].mean()

    print(f"Mean CISS ASRI:            {crisis_ciss:.1f}")
    print(f"Mean Linear ASRI:          {crisis_linear:.1f}")
    print(f"Correlation Contribution:  {crisis_contrib:+.1f}")
    print(f"Mean Pairwise Correlation: {crisis_corr:.3f}")
    print()

    # Normal period analysis
    normal_mask = ~crisis_mask

    print("Normal Period Analysis (outside crisis):")
    print("-" * 50)
    normal_ciss = result.asri_ciss[normal_mask].mean()
    normal_linear = result.asri_linear[normal_mask].mean()
    normal_contrib = result.correlation_contribution[normal_mask].mean()
    normal_corr = result.mean_correlation[normal_mask].mean()

    print(f"Mean CISS ASRI:            {normal_ciss:.1f}")
    print(f"Mean Linear ASRI:          {normal_linear:.1f}")
    print(f"Correlation Contribution:  {normal_contrib:+.1f}")
    print(f"Mean Pairwise Correlation: {normal_corr:.3f}")
    print()

    # Final correlation matrix
    print("Final Correlation Matrix:")
    print("-" * 50)
    print(result.final_correlation_matrix.round(3).to_string())
    print()

    # Key insight
    print("=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    amplification = crisis_contrib - normal_contrib
    print(f"\nDuring crisis, CISS amplifies stress by an additional {amplification:.1f} points")
    print("compared to normal periods. This captures systemic contagion:")
    print("when sub-indices become correlated, total risk exceeds the sum of parts.")
    print()
    print("Linear aggregation misses this amplification because it assumes")
    print("sub-indices are independent. Portfolio theory captures what")
    print("simple averaging cannot: correlated crashes are worse than")
    print("uncorrelated ones.")
