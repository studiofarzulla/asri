"""
Copula-Based Tail Aggregation for ASRI

This module implements Clayton copula-based aggregation to capture asymmetric
tail dependence in systemic risk measurement. Traditional linear correlation
fails to capture a critical stylized fact: assets crash together more than
they boom together (asymmetric tail dependence).

Key Insight:
    Clayton copula has strong lower tail dependence (crashes) but weak upper
    tail dependence (booms). This asymmetry is exactly what we want for
    systemic risk measurement - the index should amplify during coordinated
    drawdowns.

Methodology:
    1. Transform sub-indices to uniform marginals via empirical CDF
    2. Fit Clayton (or alternative) copula to capture dependence structure
    3. Extract lower tail dependence coefficient lambda_L
    4. Compute Kendall's tau from fitted copula (rank correlation)
    5. Amplify aggregation weights when any sub-index exceeds tail threshold

Tail Dependence Formulas:
    - Clayton: lambda_L = 2^(-1/theta), lambda_U = 0
    - Gumbel:  lambda_U = 2 - 2^(1/theta), lambda_L = 0
    - Student: lambda_L = lambda_U (symmetric, depends on df and rho)

References:
    - Nelsen, R.B. (2006). An Introduction to Copulas. Springer.
    - Joe, H. (2014). Dependence Modeling with Copulas. CRC Press.
    - Embrechts, P., Lindskog, F., & McNeil, A. (2003). Modelling Dependence
      with Copulas and Applications to Risk Management. Handbook of Heavy
      Tailed Distributions in Finance.

Dependencies:
    - copulae: Copula fitting and estimation (pip install copulae)
    - scipy: Statistical functions
    - numpy, pandas, matplotlib: Standard scientific stack
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad
from scipy.stats import kendalltau

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


CopulaFamily = Literal["clayton", "gumbel", "student", "frank"]
TailDirection = Literal["upper", "lower"]

# Canonical column names for sub-indices
CANONICAL_COLUMNS = ['stablecoin_risk', 'defi_liquidity_risk', 'contagion_risk', 'arbitrage_opacity']


def _debye_1(theta: float) -> float:
    """
    First Debye function for Frank copula tau calculation.

    D_1(theta) = (1/theta) * integral_0^theta [t / (exp(t) - 1)] dt

    Args:
        theta: Frank copula parameter

    Returns:
        Value of the first Debye function
    """
    if abs(theta) < 1e-10:
        return 1.0

    def integrand(t: float) -> float:
        if t < 1e-10:
            return 1.0  # Limit as t -> 0
        return t / (np.exp(t) - 1)

    result, _ = quad(integrand, 1e-10, abs(theta))
    return result / abs(theta)


@dataclass
class TailDependenceResult:
    """Container for tail dependence estimates."""

    lower_tail: float  # lambda_L: P(Y < F_Y^{-1}(u) | X < F_X^{-1}(u)) as u -> 0
    upper_tail: float  # lambda_U: P(Y > F_Y^{-1}(u) | X > F_X^{-1}(u)) as u -> 1
    copula_parameter: float  # theta (or nu for Student)
    copula_family: str
    kendall_tau: float  # Implied Kendall's tau from copula

    def asymmetry_ratio(self) -> float:
        """
        Ratio of lower to upper tail dependence.

        Values > 1 indicate crash-clustering dominates boom-clustering.
        For Clayton, this is infinite (lambda_U = 0).
        """
        if self.upper_tail < 1e-10:
            return float("inf")
        return self.lower_tail / self.upper_tail


@dataclass
class CopulaFitResult:
    """Complete results from copula fitting."""

    family: str
    parameter: float
    log_likelihood: float
    aic: float
    bic: float
    tail_dependence: TailDependenceResult
    converged: bool
    n_observations: int


class CopulaAggregator:
    """
    Copula-based aggregation for ASRI sub-indices.

    The core innovation is using tail dependence to amplify the aggregate
    index during crisis periods when sub-indices exhibit coordinated
    extreme movements.

    Example:
        >>> aggregator = CopulaAggregator(copula_family='clayton')
        >>> aggregator.fit_copula(subindices_df)
        >>> tail_dep = aggregator.estimate_tail_dependence()
        >>> print(f"Lower tail dependence: {tail_dep.lower_tail:.3f}")
        >>> asri = aggregator.aggregate_with_tail_boost(subindices_df)
    """

    # Default sub-index weights (from theoretical justification)
    DEFAULT_WEIGHTS = np.array([0.30, 0.25, 0.25, 0.20])

    # Column order for sub-indices
    SUB_INDEX_NAMES = [
        "stablecoin_risk",
        "defi_liquidity_risk",
        "contagion_risk",
        "arbitrage_opacity",
    ]

    def __init__(
        self,
        copula_family: CopulaFamily = "clayton",
        tail_direction: TailDirection = "upper",
        random_state: int | None = 42,
    ):
        """
        Initialize copula aggregator.

        Args:
            copula_family: Copula type to fit. Options:
                - 'clayton': Lower tail dependence only (crashes cluster)
                - 'gumbel': Upper tail dependence only (booms cluster)
                - 'student': Symmetric tail dependence (both tails)
                - 'frank': No tail dependence (baseline comparison)
            tail_direction: Which tail to monitor for risk amplification.
                - 'upper': High values indicate high risk (DEFAULT for risk indices).
                    After uniform transform, high sub-index values -> high uniform values.
                    Use Gumbel for upper tail dependence, or interpret Clayton's
                    lower tail as capturing low uniform values (low risk periods).
                - 'lower': Low values indicate high risk (e.g., liquidity measures).
                    Use Clayton for lower tail dependence.

                NOTE: For ASRI sub-indices where HIGH = BAD (high risk), the uniform
                transform maps high risk to high uniform quantiles. Therefore:
                - Gumbel (upper tail) captures when multiple sub-indices are simultaneously
                  in their high-risk (high value) regime.
                - Clayton (lower tail) captures when sub-indices are simultaneously low,
                  which for risk indices means simultaneously LOW risk (not useful).

                For standard ASRI usage with high=bad risk indices, prefer 'upper'
                with Gumbel, or understand that Clayton's lambda_L captures the
                opposite of what you want for crisis detection.
            random_state: Random seed for reproducibility
        """
        self.copula_family = copula_family
        self.tail_direction = tail_direction
        self.random_state = random_state
        self.fitted_copula = None
        self._fit_result: CopulaFitResult | None = None
        self._copula_available = self._check_copula_library()

    def _check_copula_library(self) -> bool:
        """Check if copulae library is available."""
        try:
            import copulae
            return True
        except ImportError:
            warnings.warn(
                "copulae library not installed. Install with: pip install copulae. "
                "Falling back to empirical Kendall's tau for dependence estimation."
            )
            return False

    @staticmethod
    def standardize_columns(
        data: pd.DataFrame,
        column_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """
        Standardize column names to canonical ASRI sub-index names.

        This ensures consistent column ordering regardless of input naming
        conventions (e.g., 'stablecoin' vs 'stablecoin_risk').

        Args:
            data: DataFrame with sub-index columns
            column_mapping: Optional explicit mapping from input names to canonical.
                If None, attempts automatic matching based on substrings.

        Returns:
            DataFrame with standardized column names in canonical order.

        Raises:
            ValueError: If columns cannot be mapped to canonical names.

        Example:
            >>> df = pd.DataFrame({'stablecoin': [...], 'defi': [...], ...})
            >>> df_std = CopulaAggregator.standardize_columns(df, {
            ...     'stablecoin': 'stablecoin_risk',
            ...     'defi': 'defi_liquidity_risk',
            ...     'contagion': 'contagion_risk',
            ...     'opacity': 'arbitrage_opacity'
            ... })
        """
        if column_mapping is not None:
            # Use explicit mapping
            result = data.rename(columns=column_mapping)
        else:
            # Attempt automatic matching
            current_cols = set(data.columns)
            canonical = set(CANONICAL_COLUMNS)

            # If already canonical, return as-is
            if current_cols == canonical:
                return data[CANONICAL_COLUMNS].copy()

            # Try substring matching
            mapping = {}
            for col in data.columns:
                col_lower = col.lower()
                if 'stablecoin' in col_lower or 'stable' in col_lower:
                    mapping[col] = 'stablecoin_risk'
                elif 'defi' in col_lower or 'liquidity' in col_lower:
                    mapping[col] = 'defi_liquidity_risk'
                elif 'contagion' in col_lower or 'spillover' in col_lower:
                    mapping[col] = 'contagion_risk'
                elif 'arbitrage' in col_lower or 'opacity' in col_lower:
                    mapping[col] = 'arbitrage_opacity'

            if len(mapping) != 4:
                raise ValueError(
                    f"Could not automatically map columns to canonical names. "
                    f"Input columns: {list(data.columns)}. "
                    f"Expected canonical: {CANONICAL_COLUMNS}. "
                    f"Please provide explicit column_mapping."
                )

            result = data.rename(columns=mapping)

        # Ensure canonical order
        missing = set(CANONICAL_COLUMNS) - set(result.columns)
        if missing:
            raise ValueError(f"Missing canonical columns after mapping: {missing}")

        return result[CANONICAL_COLUMNS].copy()

    def transform_to_uniform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform data to uniform marginals using probability integral transform.

        This is the crucial first step for copula fitting. We use the empirical
        CDF (pseudo-observations) which is rank-based and robust to outliers.

        Args:
            data: DataFrame with sub-indices as columns

        Returns:
            Array of shape (n_obs, n_vars) with values in (0, 1)

        Note:
            We use (rank - 0.5) / n instead of rank / n to avoid boundary
            issues at 0 and 1 which cause numerical problems for some copulas.
        """
        n = len(data)
        uniform_data = np.zeros_like(data.values, dtype=float)

        for j, col in enumerate(data.columns):
            # Empirical CDF: (rank - 0.5) / n gives pseudo-observations
            ranks = stats.rankdata(data[col].values, method="average")
            uniform_data[:, j] = (ranks - 0.5) / n

        return uniform_data

    def fit_copula(self, subindices: pd.DataFrame) -> CopulaFitResult:
        """
        Fit chosen copula to the uniform-transformed sub-indices.

        For multivariate data (4 sub-indices), we fit a multivariate copula.
        If copulae library is unavailable, falls back to pairwise estimation.

        Args:
            subindices: DataFrame with 4 sub-index columns

        Returns:
            CopulaFitResult with fitted parameters and diagnostics
        """
        # Clean data
        clean_data = subindices.dropna()
        n = len(clean_data)

        if n < 50:
            raise ValueError(
                f"Insufficient observations ({n}) for reliable copula estimation. "
                "Need at least 50 observations."
            )

        # Transform to uniform marginals
        uniform = self.transform_to_uniform(clean_data)

        if self._copula_available:
            return self._fit_copula_copulae(uniform, n)
        else:
            return self._fit_copula_fallback(uniform, n)

    def _fit_copula_copulae(
        self,
        uniform: np.ndarray,
        n: int,
    ) -> CopulaFitResult:
        """Fit copula using copulae library."""
        from copulae import ClaytonCopula, FrankCopula, GumbelCopula, StudentCopula

        dim = uniform.shape[1]

        # Create copula based on family
        if self.copula_family == "clayton":
            copula = ClaytonCopula(dim=dim)
        elif self.copula_family == "gumbel":
            copula = GumbelCopula(dim=dim)
        elif self.copula_family == "student":
            copula = StudentCopula(dim=dim)
        elif self.copula_family == "frank":
            copula = FrankCopula(dim=dim)
        else:
            raise ValueError(f"Unknown copula family: {self.copula_family}")

        # Fit copula
        try:
            copula.fit(uniform, method="ml", verbose=0)  # Maximum likelihood
            converged = True

            # Extract parameter (theta for Archimedean, multiple for Student)
            if self.copula_family == "student":
                # Student has correlation matrix and degrees of freedom
                param = copula.params.df  # degrees of freedom
            else:
                param = float(copula.params)

            # Log-likelihood for model comparison
            log_lik = copula.log_lik(uniform)

            # AIC and BIC
            k = 1 if self.copula_family != "student" else dim * (dim - 1) // 2 + 1
            aic = 2 * k - 2 * log_lik
            bic = k * np.log(n) - 2 * log_lik

        except Exception as e:
            warnings.warn(f"Copula fitting failed: {e}. Using fallback.")
            return self._fit_copula_fallback(uniform, n)

        self.fitted_copula = copula

        # Compute tail dependence
        tail_dep = self._compute_tail_dependence(param)

        self._fit_result = CopulaFitResult(
            family=self.copula_family,
            parameter=param,
            log_likelihood=log_lik,
            aic=aic,
            bic=bic,
            tail_dependence=tail_dep,
            converged=converged,
            n_observations=n,
        )

        return self._fit_result

    def _fit_copula_fallback(
        self,
        uniform: np.ndarray,
        n: int,
    ) -> CopulaFitResult:
        """
        Fallback copula estimation using method of moments via Kendall's tau.

        For Archimedean copulas, there's a one-to-one mapping between
        Kendall's tau and the copula parameter theta.
        """
        # Compute average pairwise Kendall's tau
        taus = []
        for i in range(uniform.shape[1]):
            for j in range(i + 1, uniform.shape[1]):
                tau, _ = kendalltau(uniform[:, i], uniform[:, j])
                if not np.isnan(tau):
                    taus.append(tau)

        avg_tau = np.mean(taus) if taus else 0.3

        # Invert Kendall's tau to get theta
        if self.copula_family == "clayton":
            # tau = theta / (theta + 2) => theta = 2*tau / (1 - tau)
            if avg_tau >= 1:
                avg_tau = 0.99  # Clip to avoid division by zero
            if avg_tau < 0:
                avg_tau = 0.01  # Clayton requires tau > 0
            param = 2 * avg_tau / (1 - avg_tau)
            param = max(0.01, param)  # Ensure positive

        elif self.copula_family == "gumbel":
            # tau = 1 - 1/theta => theta = 1 / (1 - tau)
            if avg_tau >= 1:
                avg_tau = 0.99
            param = 1 / (1 - avg_tau)
            param = max(1.0, param)  # Gumbel requires theta >= 1

        elif self.copula_family == "frank":
            # tau = 1 - 4/theta + 4*D_1(theta)/theta where D_1 is Debye function
            # Approximate: tau ≈ (theta - 2) / (3*theta) for large theta
            param = 3 * avg_tau / (1 - avg_tau) if avg_tau < 1 else 10.0

        else:
            # Student: use correlation and assume df=5
            param = 5.0

        tail_dep = self._compute_tail_dependence(param)

        self._fit_result = CopulaFitResult(
            family=self.copula_family,
            parameter=param,
            log_likelihood=np.nan,
            aic=np.nan,
            bic=np.nan,
            tail_dependence=tail_dep,
            converged=True,  # Method of moments always "converges"
            n_observations=n,
        )

        return self._fit_result

    def _compute_tail_dependence(self, theta: float) -> TailDependenceResult:
        """
        Compute tail dependence coefficients from copula parameter.

        These are the key quantities for crisis amplification:
        - lambda_L: Lower tail dependence (crash clustering)
        - lambda_U: Upper tail dependence (boom clustering)
        """
        if self.copula_family == "clayton":
            # Clayton: strong lower tail, no upper tail
            lambda_l = 2 ** (-1 / theta) if theta > 0 else 0.0
            lambda_u = 0.0
            tau = theta / (theta + 2) if theta >= 0 else 0.0

        elif self.copula_family == "gumbel":
            # Gumbel: no lower tail, strong upper tail
            lambda_l = 0.0
            lambda_u = 2 - 2 ** (1 / theta) if theta >= 1 else 0.0
            tau = 1 - 1 / theta if theta >= 1 else 0.0

        elif self.copula_family == "frank":
            # Frank: no tail dependence (comparison baseline)
            lambda_l = 0.0
            lambda_u = 0.0
            # Correct tau formula using Debye function:
            # tau = 1 - 4/theta * (1 - D_1(theta))
            # where D_1 is the first Debye function
            if abs(theta) > 0.01:
                tau = 1 - (4 / theta) * (1 - _debye_1(theta))
            else:
                tau = 0.0

        elif self.copula_family == "student":
            # Student: symmetric tail dependence
            # lambda = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
            # tau = (2/pi) * arcsin(rho)
            nu = theta

            # Extract rho from fitted copula if available
            rho = 0.5  # Default fallback
            if self.fitted_copula is not None:
                try:
                    if hasattr(self.fitted_copula, 'params') and hasattr(self.fitted_copula.params, 'corr'):
                        corr_matrix = self.fitted_copula.params.corr
                        dim = corr_matrix.shape[0]
                        # Average off-diagonal correlation
                        rho = (corr_matrix.sum() - dim) / (dim * (dim - 1))
                        rho = np.clip(rho, -0.999, 0.999)  # Ensure valid range
                except Exception:
                    warnings.warn(
                        "Could not extract correlation from Student-t copula, using rho=0.5"
                    )
                    rho = 0.5

            t_arg = np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
            lambda_sym = 2 * stats.t.cdf(-t_arg, df=nu + 1)
            lambda_l = lambda_sym
            lambda_u = lambda_sym
            tau = 2 * np.arcsin(rho) / np.pi

        else:
            lambda_l = 0.0
            lambda_u = 0.0
            tau = 0.0

        return TailDependenceResult(
            lower_tail=lambda_l,
            upper_tail=lambda_u,
            copula_parameter=theta,
            copula_family=self.copula_family,
            kendall_tau=tau,
        )

    def estimate_tail_dependence(self) -> TailDependenceResult:
        """
        Extract lower and upper tail dependence coefficients.

        Must call fit_copula() first.

        Returns:
            TailDependenceResult with lambda_L, lambda_U, and related metrics
        """
        if self._fit_result is None:
            raise ValueError("Must call fit_copula() before estimating tail dependence")

        return self._fit_result.tail_dependence

    def compute_kendall_tau_matrix(self, subindices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 4x4 Kendall's tau matrix between sub-indices.

        Kendall's tau is preferred over Pearson correlation for copula analysis
        because it captures monotonic (not just linear) dependence and is
        invariant under monotonic transformations.

        Args:
            subindices: DataFrame with 4 sub-index columns

        Returns:
            4x4 DataFrame of pairwise Kendall's tau values
        """
        clean_data = subindices.dropna()
        cols = clean_data.columns.tolist()
        n_cols = len(cols)

        tau_matrix = np.eye(n_cols)

        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                tau, _ = kendalltau(clean_data.iloc[:, i], clean_data.iloc[:, j])
                tau_matrix[i, j] = tau
                tau_matrix[j, i] = tau

        return pd.DataFrame(tau_matrix, index=cols, columns=cols)

    def detect_tail_event(
        self,
        subindices: pd.DataFrame,
        threshold: float = 0.90,
        use_expanding: bool = True,
        min_periods: int = 30,
    ) -> pd.Series:
        """
        Detect when any sub-index exceeds its historical threshold quantile.

        A tail event occurs when at least one sub-index is in its extreme
        right tail - indicating elevated risk in that component.

        Args:
            subindices: DataFrame with sub-index values
            threshold: Quantile threshold (default 0.90 = 90th percentile)
            use_expanding: If True, use expanding window quantiles to avoid
                look-ahead bias. Each observation is compared only to its
                historical distribution. Default True.
            min_periods: Minimum observations for expanding quantile calculation.
                Earlier periods will use available data. Default 30.

        Returns:
            Boolean Series: True when any sub-index exceeds threshold

        Note:
            For backtesting and production use, `use_expanding=True` is essential
            to avoid look-ahead bias. The full-sample quantile would use future
            information not available at each historical point.
        """
        clean_data = subindices.dropna()

        if use_expanding:
            # Expanding window quantiles - no look-ahead bias
            # Each row compared only to its historical distribution
            expanding_quantiles = clean_data.expanding(min_periods=min_periods).quantile(threshold)

            # Tail event = any sub-index exceeds its expanding threshold
            tail_events = (clean_data > expanding_quantiles).any(axis=1)

            # First min_periods rows may have NaN quantiles - mark as non-tail events
            tail_events = tail_events.fillna(False)
        else:
            # Full-sample quantiles (original behavior - has look-ahead bias)
            quantiles = clean_data.quantile(threshold)
            tail_events = (clean_data > quantiles).any(axis=1)

        return tail_events

    def _compute_tail_boost_weights(
        self,
        current_values: np.ndarray,
        base_weights: np.ndarray,
        tail_dependence: float,
        boost_factor: float,
    ) -> np.ndarray:
        """
        Compute dynamically boosted weights during tail events.

        The insight: during crises, sub-indices with higher tail dependence
        should receive MORE weight because their extreme values are more
        likely to propagate to other components.

        Args:
            current_values: Current sub-index values
            base_weights: Theoretical weights [0.30, 0.25, 0.25, 0.20]
            tail_dependence: Lower tail dependence coefficient
            boost_factor: Multiplier for tail event amplification

        Returns:
            Boosted weights (normalized to sum to 1)
        """
        # Boost weights proportional to how extreme each sub-index is
        # Higher values get higher weights during tail events
        percentiles = np.zeros_like(current_values)
        for i, val in enumerate(current_values):
            if not np.isnan(val):
                # This is a simplification - in practice we'd track historical distribution
                percentiles[i] = val / 100.0  # Assume 0-100 scale

        # Weight boost proportional to extremity and tail dependence
        boost = 1 + boost_factor * tail_dependence * percentiles

        # Apply boost to base weights
        boosted_weights = base_weights * boost

        # Normalize to sum to 1
        return boosted_weights / boosted_weights.sum()

    def aggregate_with_tail_boost(
        self,
        subindices: pd.DataFrame,
        base_weights: np.ndarray | None = None,
        tail_threshold: float = 0.90,
        boost_factor: float = 1.5,
        use_expanding: bool = True,
    ) -> pd.Series:
        """
        Aggregate sub-indices with amplification during tail events.

        During normal times, uses standard weighted average.
        During tail events (any sub-index > 90th percentile), amplifies
        weights based on tail dependence structure.

        Args:
            subindices: DataFrame with 4 sub-index columns
            base_weights: Theoretical weights (default: [0.30, 0.25, 0.25, 0.20])
            tail_threshold: Quantile for tail event detection
            boost_factor: Amplification multiplier during tail events
            use_expanding: Use expanding quantiles for tail detection (avoids look-ahead bias)

        Returns:
            Series of aggregated ASRI values

        Note on tail direction:
            For ASRI where HIGH sub-index = HIGH risk, after uniform transform:
            - High risk values map to high uniform quantiles (near 1)
            - Gumbel captures UPPER tail dependence (simultaneous highs)
            - Clayton captures LOWER tail dependence (simultaneous lows)

            Therefore:
            - If tail_direction='upper' (default), we use lambda_U for boost
            - If tail_direction='lower', we use lambda_L for boost

            For standard risk indices, use Gumbel + tail_direction='upper'.
        """
        if base_weights is None:
            base_weights = self.DEFAULT_WEIGHTS.copy()

        clean_data = subindices.dropna()

        # Fit copula if not already done
        if self._fit_result is None:
            self.fit_copula(clean_data)

        # Use appropriate tail dependence based on direction
        if self.tail_direction == "upper":
            tail_dep = self._fit_result.tail_dependence.upper_tail
        else:
            tail_dep = self._fit_result.tail_dependence.lower_tail

        # Detect tail events
        tail_events = self.detect_tail_event(clean_data, tail_threshold, use_expanding=use_expanding)

        # Compute aggregated values
        aggregated = pd.Series(index=clean_data.index, dtype=float)

        for idx in clean_data.index:
            row = clean_data.loc[idx].values

            if tail_events.loc[idx]:
                # Tail event: use boosted weights
                weights = self._compute_tail_boost_weights(
                    row, base_weights, tail_dep, boost_factor
                )
            else:
                # Normal: use base weights
                weights = base_weights

            aggregated.loc[idx] = np.dot(row, weights)

        return aggregated

    def compute_rolling_tail_dependence(
        self,
        subindices: pd.DataFrame,
        window: int = 252,
        min_periods: int = 100,
    ) -> pd.DataFrame:
        """
        Compute rolling tail dependence estimates over time.

        Tail dependence is not constant - it increases during crises.
        This method tracks how lambda_L evolves, providing a dynamic
        measure of crash-clustering intensity.

        Args:
            subindices: DataFrame with sub-index time series
            window: Rolling window size (default: 252 trading days = 1 year)
            min_periods: Minimum observations for estimation

        Returns:
            DataFrame with columns: ['date', 'lambda_L', 'theta', 'kendall_tau']
        """
        clean_data = subindices.dropna()

        if len(clean_data) < window:
            raise ValueError(
                f"Insufficient data ({len(clean_data)}) for rolling window ({window})"
            )

        results = []

        for i in range(min_periods, len(clean_data)):
            end_idx = i
            start_idx = max(0, i - window)

            window_data = clean_data.iloc[start_idx:end_idx]

            if len(window_data) < min_periods:
                continue

            try:
                # Create temporary aggregator for this window
                temp_agg = CopulaAggregator(
                    copula_family=self.copula_family,
                    tail_direction=self.tail_direction,
                    random_state=self.random_state,
                )
                fit_result = temp_agg.fit_copula(window_data)

                results.append({
                    "date": clean_data.index[i],
                    "lambda_L": fit_result.tail_dependence.lower_tail,
                    "lambda_U": fit_result.tail_dependence.upper_tail,
                    "theta": fit_result.parameter,
                    "kendall_tau": fit_result.tail_dependence.kendall_tau,
                })
            except Exception:
                # Skip windows where estimation fails
                continue

        return pd.DataFrame(results)

    def plot_tail_dependence_structure(
        self,
        subindices: pd.DataFrame,
        figsize: tuple[int, int] = (12, 10),
    ) -> "plt.Figure":
        """
        Visualize the tail dependence structure.

        Creates a multi-panel figure showing:
        1. Pairwise scatter plots of uniform-transformed data
        2. Kendall's tau correlation matrix
        3. Tail dependence over time (if sufficient data)

        Args:
            subindices: DataFrame with sub-index values
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        clean_data = subindices.dropna()
        uniform = self.transform_to_uniform(clean_data)

        fig = plt.figure(figsize=figsize)

        # Top left: Uniform scatter (first two sub-indices)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.scatter(uniform[:, 0], uniform[:, 1], alpha=0.3, s=10)
        ax1.set_xlabel(clean_data.columns[0])
        ax1.set_ylabel(clean_data.columns[1])
        ax1.set_title("Copula Structure (Uniform Marginals)")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # Top right: Kendall's tau heatmap
        ax2 = fig.add_subplot(2, 2, 2)
        tau_matrix = self.compute_kendall_tau_matrix(clean_data)
        im = ax2.imshow(tau_matrix.values, cmap="RdYlBu_r", vmin=-1, vmax=1)
        ax2.set_xticks(range(len(tau_matrix.columns)))
        ax2.set_yticks(range(len(tau_matrix.index)))
        ax2.set_xticklabels(tau_matrix.columns, rotation=45, ha="right")
        ax2.set_yticklabels(tau_matrix.index)
        ax2.set_title("Kendall's Tau Matrix")
        plt.colorbar(im, ax=ax2)

        # Bottom left: Lower-tail scatter (zoomed)
        ax3 = fig.add_subplot(2, 2, 3)
        # Only show observations in lower tail (both < 0.2)
        mask = (uniform[:, 0] < 0.2) & (uniform[:, 1] < 0.2)
        ax3.scatter(uniform[mask, 0], uniform[mask, 1], alpha=0.5, s=20, c="red")
        ax3.set_xlabel(clean_data.columns[0])
        ax3.set_ylabel(clean_data.columns[1])
        ax3.set_title("Lower Tail Clustering (Both < 20th percentile)")
        ax3.set_xlim(0, 0.25)
        ax3.set_ylim(0, 0.25)

        # Fit copula if not done
        if self._fit_result is None:
            self.fit_copula(clean_data)

        # Bottom right: Tail dependence summary
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis("off")

        td = self._fit_result.tail_dependence
        summary_text = f"""
        Copula Family: {self._fit_result.family.upper()}

        Parameter (theta): {self._fit_result.parameter:.3f}

        Lower Tail Dependence (lambda_L): {td.lower_tail:.3f}
        Upper Tail Dependence (lambda_U): {td.upper_tail:.3f}

        Kendall's Tau (implied): {td.kendall_tau:.3f}

        Observations: {self._fit_result.n_observations}
        Converged: {self._fit_result.converged}

        Interpretation:
        - lambda_L = {td.lower_tail:.2%} of crash clustering
        - When one sub-index crashes, there's a
          {td.lower_tail:.0%} chance others crash too
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=10,
                 verticalalignment="center", fontfamily="monospace")
        ax4.set_title("Tail Dependence Summary")

        plt.tight_layout()
        return fig

    def compare_copula_families(
        self,
        subindices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit multiple copula families and compare using AIC/BIC.

        This helps justify the choice of Clayton copula for ASRI
        by showing it fits better than alternatives when lower
        tail dependence dominates.

        Args:
            subindices: DataFrame with sub-index values

        Returns:
            DataFrame comparing fit statistics across families
        """
        families: list[CopulaFamily] = ["clayton", "gumbel", "frank"]

        if self._copula_available:
            families.append("student")

        results = []

        for family in families:
            temp_agg = CopulaAggregator(
                copula_family=family,
                tail_direction=self.tail_direction,
                random_state=self.random_state,
            )

            try:
                fit_result = temp_agg.fit_copula(subindices)
                results.append({
                    "family": family,
                    "theta": fit_result.parameter,
                    "log_likelihood": fit_result.log_likelihood,
                    "aic": fit_result.aic,
                    "bic": fit_result.bic,
                    "lambda_L": fit_result.tail_dependence.lower_tail,
                    "lambda_U": fit_result.tail_dependence.upper_tail,
                    "kendall_tau": fit_result.tail_dependence.kendall_tau,
                    "converged": fit_result.converged,
                })
            except Exception as e:
                results.append({
                    "family": family,
                    "theta": np.nan,
                    "log_likelihood": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                    "lambda_L": np.nan,
                    "lambda_U": np.nan,
                    "kendall_tau": np.nan,
                    "converged": False,
                    "error": str(e),
                })

        return pd.DataFrame(results)

    def format_latex_table(self) -> str:
        """
        Format copula fitting results as a LaTeX table for publication.

        Returns:
            LaTeX table string
        """
        if self._fit_result is None:
            raise ValueError("Must call fit_copula() first")

        td = self._fit_result.tail_dependence

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Copula Tail Dependence Estimation}",
            r"\label{tab:copula}",
            r"\small",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Metric & Value \\",
            r"\midrule",
            f"Copula Family & {self._fit_result.family.capitalize()} \\\\",
            f"Parameter ($\\theta$) & {self._fit_result.parameter:.3f} \\\\",
            f"Lower Tail Dependence ($\\lambda_L$) & {td.lower_tail:.3f} \\\\",
            f"Upper Tail Dependence ($\\lambda_U$) & {td.upper_tail:.3f} \\\\",
            f"Kendall's $\\tau$ (implied) & {td.kendall_tau:.3f} \\\\",
        ]

        if not np.isnan(self._fit_result.log_likelihood):
            lines.extend([
                f"Log-Likelihood & {self._fit_result.log_likelihood:.2f} \\\\",
                f"AIC & {self._fit_result.aic:.2f} \\\\",
                f"BIC & {self._fit_result.bic:.2f} \\\\",
            ])

        lines.extend([
            f"Observations & {self._fit_result.n_observations} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"\item For Clayton copula: $\lambda_L = 2^{-1/\theta}$, $\lambda_U = 0$.",
            r"\item Tail dependence coefficients measure crash/boom clustering.",
            r"\end{tablenotes}",
            r"\end{table}",
        ])

        return "\n".join(lines)


def compute_copula_weighted_asri(
    subindices: pd.DataFrame,
    copula_family: CopulaFamily = "clayton",
    tail_threshold: float = 0.90,
    boost_factor: float = 1.5,
) -> tuple[pd.Series, CopulaFitResult]:
    """
    Convenience function to compute ASRI with copula-based tail amplification.

    Args:
        subindices: DataFrame with 4 sub-index columns
        copula_family: Copula type ('clayton', 'gumbel', 'student', 'frank')
        tail_threshold: Quantile for tail event detection
        boost_factor: Weight amplification during tail events

    Returns:
        Tuple of (aggregated ASRI series, copula fit results)
    """
    aggregator = CopulaAggregator(copula_family=copula_family)
    fit_result = aggregator.fit_copula(subindices)
    asri = aggregator.aggregate_with_tail_boost(
        subindices,
        tail_threshold=tail_threshold,
        boost_factor=boost_factor,
    )

    return asri, fit_result


if __name__ == "__main__":
    """
    Demonstration with synthetic data simulating sub-index behavior.

    We generate data with:
    1. Common factor (systemic risk latent variable)
    2. Idiosyncratic noise per sub-index
    3. Asymmetric tail dependence via Clayton copula simulation
    """
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n = 500  # Trading days

    # Generate correlated sub-indices with tail dependence
    # Use Clayton copula to generate dependent uniform variables

    print("=" * 60)
    print("ASRI Copula Aggregation - Synthetic Data Demonstration")
    print("=" * 60)

    # Simple approach: generate via common factor + idiosyncratic
    common_factor = np.random.normal(0, 1, n)

    # Sub-indices as factor + noise, scaled to 0-100
    stablecoin = 50 + 15 * common_factor + np.random.normal(0, 5, n)
    defi = 45 + 12 * common_factor + np.random.normal(0, 6, n)
    contagion = 55 + 18 * common_factor + np.random.normal(0, 4, n)
    arbitrage = 40 + 10 * common_factor + np.random.normal(0, 7, n)

    # Clip to valid range and add some extreme events
    stablecoin = np.clip(stablecoin, 0, 100)
    defi = np.clip(defi, 0, 100)
    contagion = np.clip(contagion, 0, 100)
    arbitrage = np.clip(arbitrage, 0, 100)

    # Inject crisis periods (coordinated spikes)
    crisis_periods = [50, 150, 300, 400]
    for t in crisis_periods:
        stablecoin[t:t+10] += 25
        defi[t:t+10] += 20
        contagion[t:t+10] += 30
        arbitrage[t:t+10] += 15

    # Clip again after crisis injection
    stablecoin = np.clip(stablecoin, 0, 100)
    defi = np.clip(defi, 0, 100)
    contagion = np.clip(contagion, 0, 100)
    arbitrage = np.clip(arbitrage, 0, 100)

    # Create DataFrame
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    subindices = pd.DataFrame({
        "stablecoin_risk": stablecoin,
        "defi_liquidity_risk": defi,
        "contagion_risk": contagion,
        "arbitrage_opacity": arbitrage,
    }, index=dates)

    print(f"\nGenerated {n} observations")
    print(f"\nSub-index summary:")
    print(subindices.describe().round(2))

    # Fit Clayton copula
    print("\n" + "-" * 40)
    print("Fitting Clayton copula...")
    print("-" * 40)

    aggregator = CopulaAggregator(copula_family="clayton")
    fit_result = aggregator.fit_copula(subindices)

    print(f"Copula parameter (theta): {fit_result.parameter:.3f}")
    print(f"Lower tail dependence (lambda_L): {fit_result.tail_dependence.lower_tail:.3f}")
    print(f"Upper tail dependence (lambda_U): {fit_result.tail_dependence.upper_tail:.3f}")
    print(f"Kendall's tau (implied): {fit_result.tail_dependence.kendall_tau:.3f}")

    # Compute Kendall's tau matrix
    print("\n" + "-" * 40)
    print("Kendall's Tau Matrix:")
    print("-" * 40)
    tau_matrix = aggregator.compute_kendall_tau_matrix(subindices)
    print(tau_matrix.round(3))

    # Compare copula families
    print("\n" + "-" * 40)
    print("Copula Family Comparison:")
    print("-" * 40)
    comparison = aggregator.compare_copula_families(subindices)
    print(comparison[["family", "theta", "lambda_L", "lambda_U", "kendall_tau"]].round(3))

    # Compute aggregated ASRI with tail boost
    print("\n" + "-" * 40)
    print("Aggregation with Tail Boost:")
    print("-" * 40)

    # Standard aggregation (no boost)
    standard_weights = np.array([0.30, 0.25, 0.25, 0.20])
    asri_standard = subindices.dot(standard_weights)

    # Copula-boosted aggregation
    asri_boosted = aggregator.aggregate_with_tail_boost(subindices)

    # Count tail events
    tail_events = aggregator.detect_tail_event(subindices, threshold=0.90)
    n_tail = tail_events.sum()

    print(f"Tail events detected: {n_tail} / {len(tail_events)} ({100*n_tail/len(tail_events):.1f}%)")
    print(f"\nStandard ASRI: mean={asri_standard.mean():.2f}, std={asri_standard.std():.2f}")
    print(f"Boosted ASRI:  mean={asri_boosted.mean():.2f}, std={asri_boosted.std():.2f}")

    # Show difference during tail events
    diff = asri_boosted - asri_standard
    print(f"\nBoost during tail events: +{diff[tail_events].mean():.2f} on average")
    print(f"Boost during normal times: +{diff[~tail_events].mean():.2f} on average")

    # Generate LaTeX table
    print("\n" + "-" * 40)
    print("LaTeX Table:")
    print("-" * 40)
    print(aggregator.format_latex_table())

    # Plot results
    print("\n" + "-" * 40)
    print("Generating visualization...")
    print("-" * 40)

    fig = plt.figure(figsize=(14, 10))

    # Subplot 1: Sub-indices over time
    ax1 = fig.add_subplot(2, 2, 1)
    for col in subindices.columns:
        ax1.plot(subindices.index, subindices[col], alpha=0.7, label=col)
    ax1.axhline(y=subindices.quantile(0.90).mean(), color="red",
                linestyle="--", alpha=0.5, label="90th percentile")
    ax1.set_title("Sub-Index Time Series")
    ax1.set_ylabel("Risk Level")
    ax1.legend(loc="upper right", fontsize=8)

    # Subplot 2: Standard vs Boosted ASRI
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(asri_standard.index, asri_standard, alpha=0.7, label="Standard ASRI")
    ax2.plot(asri_boosted.index, asri_boosted, alpha=0.7, label="Copula-Boosted ASRI")
    # Mark tail events
    ax2.scatter(asri_boosted.index[tail_events], asri_boosted[tail_events],
                color="red", s=20, alpha=0.5, label="Tail Events")
    ax2.set_title("ASRI: Standard vs Copula-Boosted")
    ax2.set_ylabel("ASRI Value")
    ax2.legend()

    # Subplot 3: Boost magnitude over time
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.fill_between(diff.index, 0, diff, alpha=0.5, label="Boost Amount")
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax3.set_title("Tail Boost Contribution")
    ax3.set_ylabel("Boost (Copula - Standard)")
    ax3.legend()

    # Subplot 4: Tail dependence structure
    ax4 = fig.add_subplot(2, 2, 4)
    uniform = aggregator.transform_to_uniform(subindices)
    ax4.scatter(uniform[:, 0], uniform[:, 1], alpha=0.3, s=10)
    ax4.set_xlabel("Stablecoin Risk (uniform)")
    ax4.set_ylabel("DeFi Risk (uniform)")
    ax4.set_title(f"Copula Structure (lambda_L = {fit_result.tail_dependence.lower_tail:.3f})")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("/tmp/copula_aggregation_demo.png", dpi=150, bbox_inches="tight")
    print("Saved: /tmp/copula_aggregation_demo.png")

    plt.show()

    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)
