"""
Regime-Conditional Aggregation for ASRI

The baseline ASRI uses fixed weights, but optimal weights likely differ by
market regime. This module implements Markov-switching aggregation where
weights vary according to the current regime state.

Methodology:
    The aggregate ASRI at time t is computed as a probability-weighted
    combination across regimes:

        ASRI_t = sum_k P(regime=k | data_t) * (w_k' @ s_t)

    where:
        - P(regime=k | data_t) is the posterior probability of regime k
        - w_k is the weight vector for regime k
        - s_t is the vector of sub-indices at time t

    This allows smooth transitions between regimes rather than hard switches,
    capturing the uncertainty in regime classification.

Regime-Specific Weight Rationale:
    - Crisis: Stablecoin and liquidity risks dominate when markets stress
    - Moderate: Balanced weights reflect normal conditions
    - Low Risk: Contagion and opacity become relatively more important
             when immediate liquidity pressures are absent

References:
    - Hamilton (1989) "A New Approach to the Economic Analysis of
      Nonstationary Time Series and the Business Cycle"
    - Kim & Nelson (1999) "State-Space Models with Regime Switching"
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
from scipy import optimize

if TYPE_CHECKING:
    from ..regime.hmm import RegimeDetector

# Sub-index column names (canonical ordering)
SUBINDEX_COLUMNS = [
    "stablecoin_risk",
    "defi_liquidity_risk",
    "contagion_risk",
    "arbitrage_opacity",
]


@dataclass
class RegimeWeights:
    """
    Weight vectors for each market regime.

    Weights are ordered as: [SCR, DLR, CR, OR]
        - SCR: Stablecoin Concentration & Treasury Exposure Risk
        - DLR: DeFi Liquidity & Composability Risk
        - CR: Cross-Market Interconnection & Contagion Risk
        - OR: Regulatory Arbitrage & Opacity Risk

    Default values reflect the economic intuition that different risk
    dimensions become more salient in different market conditions.
    """

    low_risk: np.ndarray = field(
        default_factory=lambda: np.array([0.20, 0.25, 0.30, 0.25])
    )
    moderate: np.ndarray = field(
        default_factory=lambda: np.array([0.30, 0.25, 0.25, 0.20])
    )
    crisis: np.ndarray = field(
        default_factory=lambda: np.array([0.40, 0.30, 0.20, 0.10])
    )

    def __post_init__(self) -> None:
        """Validate weights sum to 1 and are non-negative."""
        for name in ["low_risk", "moderate", "crisis"]:
            w = getattr(self, name)
            if not isinstance(w, np.ndarray):
                setattr(self, name, np.array(w))
                w = getattr(self, name)
            if len(w) != 4:
                raise ValueError(f"{name} weights must have 4 elements, got {len(w)}")
            if not np.isclose(w.sum(), 1.0, atol=1e-6):
                raise ValueError(f"{name} weights must sum to 1, got {w.sum():.4f}")
            if np.any(w < 0):
                raise ValueError(f"{name} weights must be non-negative")

    def as_matrix(self) -> np.ndarray:
        """
        Return weights as (n_regimes, n_features) matrix.

        Rows are ordered by risk level: [low_risk, moderate, crisis]
        """
        return np.vstack([self.low_risk, self.moderate, self.crisis])

    def get_weights_for_regime(self, regime: int) -> np.ndarray:
        """Get weight vector for a specific regime index."""
        if regime == 0:
            return self.low_risk
        elif regime == 1:
            return self.moderate
        elif regime == 2:
            return self.crisis
        else:
            raise ValueError(f"Unknown regime index: {regime}")

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Convert to nested dictionary format."""
        return {
            "low_risk": dict(zip(SUBINDEX_COLUMNS, self.low_risk)),
            "moderate": dict(zip(SUBINDEX_COLUMNS, self.moderate)),
            "crisis": dict(zip(SUBINDEX_COLUMNS, self.crisis)),
        }


@dataclass
class RegimeAggregationResult:
    """Results from regime-conditional aggregation."""

    # Primary output
    asri_regime_weighted: pd.Series  # The main output: regime-weighted ASRI

    # Comparison with baseline
    asri_baseline: pd.Series  # Baseline fixed-weight ASRI
    improvement: float  # Mean absolute improvement in crisis detection

    # Regime information
    regime_probabilities: pd.DataFrame  # P(regime|data) for each timestep
    most_likely_regimes: pd.Series  # Viterbi-decoded regime sequence
    regime_weights_used: RegimeWeights

    # Effective weights over time
    effective_weights: pd.DataFrame  # Time-varying effective weights

    # Diagnostics
    regime_means: dict[int, float]  # Mean ASRI in each regime
    regime_frequencies: dict[int, float]  # Proportion of time in each regime


@dataclass
class TransitionAnalysis:
    """Analysis of regime transitions around crisis events."""

    event_name: str
    event_date: pd.Timestamp

    # Regime probabilities in windows around event
    pre_crisis_probs: pd.DataFrame  # (-30, -7 days)
    crisis_probs: pd.DataFrame  # (-7, +7 days)
    post_crisis_probs: pd.DataFrame  # (+7, +30 days)

    # Summary statistics
    crisis_regime_prob_increase: float  # P(crisis) at event vs pre
    transition_lead_days: int  # Days before event that crisis prob > 0.5


class RegimeAggregator:
    """
    Aggregate ASRI sub-indices using regime-conditional weights.

    This class fits a Hidden Markov Model to learn market regimes from
    sub-index dynamics, then uses regime-specific weight vectors to
    compute a probability-weighted aggregate index.

    The key insight is that different risk dimensions become more or
    less relevant depending on market conditions. During crises,
    stablecoin depegging and liquidity freezes dominate; during calm
    periods, structural risks like opacity and contagion potential
    are relatively more important.

    Example usage:
        >>> aggregator = RegimeAggregator(n_regimes=3)
        >>> aggregator.fit_hmm(subindices)
        >>> asri_regime = aggregator.aggregate(subindices)

    Attributes:
        n_regimes: Number of hidden states (default: 3)
        regime_weights: Weight vectors for each regime
        hmm: The underlying HMM model (fitted after calling fit_hmm)
        _use_hmmlearn: Whether to use hmmlearn or custom implementation
    """

    def __init__(
        self,
        n_regimes: int = 3,
        regime_weights: RegimeWeights | None = None,
        use_hmmlearn: bool = True,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the regime aggregator.

        Args:
            n_regimes: Number of regimes to detect. Default 3 corresponds
                to low-risk, moderate, and crisis states.
            regime_weights: Custom weight vectors per regime. If None,
                uses default values based on economic intuition.
            use_hmmlearn: If True, use hmmlearn.hmm.GaussianHMM. If False,
                use the custom implementation from asri.regime.hmm.
            random_state: Random seed for HMM initialization.
        """
        if n_regimes != 3:
            warnings.warn(
                f"RegimeWeights expects 3 regimes, but n_regimes={n_regimes}. "
                "Provide custom regime_weights or expect index errors.",
                UserWarning,
            )

        self.n_regimes = n_regimes
        self.regime_weights = regime_weights or RegimeWeights()
        self._use_hmmlearn = use_hmmlearn
        self.random_state = random_state

        self.hmm = None
        self._fitted = False
        self._regime_order: np.ndarray | None = None  # Maps sorted -> original indices

    def fit_hmm(
        self,
        subindices: pd.DataFrame,
        n_iter: int = 100,
        covariance_type: str = "full",
    ) -> RegimeAggregator:
        """
        Fit a Gaussian HMM to the sub-index data.

        The HMM learns:
            1. Transition probabilities between regimes
            2. Emission distributions (mean/covariance of sub-indices per regime)
            3. Initial state probabilities

        After fitting, regimes are re-ordered by mean ASRI so that
        regime 0 = lowest risk, regime N-1 = highest risk.

        Args:
            subindices: DataFrame with sub-index columns. Must contain
                the columns in SUBINDEX_COLUMNS.
            n_iter: Maximum EM iterations.
            covariance_type: Type of covariance parameters ('full', 'diag',
                'spherical'). Only used with hmmlearn.

        Returns:
            Self (for method chaining).

        Raises:
            ValueError: If required columns are missing or data is insufficient.
        """
        # Validate columns
        missing = set(SUBINDEX_COLUMNS) - set(subindices.columns)
        if missing:
            raise ValueError(f"Missing sub-index columns: {missing}")

        # Prepare data
        X = subindices[SUBINDEX_COLUMNS].dropna()
        if len(X) < 50:
            raise ValueError(f"Insufficient data: {len(X)} rows, need at least 50")

        if self._use_hmmlearn:
            self._fit_hmmlearn(X, n_iter, covariance_type)
        else:
            self._fit_custom_hmm(X, n_iter)

        self._fitted = True
        return self

    def _fit_hmmlearn(
        self,
        X: pd.DataFrame,
        n_iter: int,
        covariance_type: str,
    ) -> None:
        """Fit using hmmlearn.hmm.GaussianHMM."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as e:
            raise ImportError(
                "hmmlearn is required for use_hmmlearn=True. "
                "Install with: pip install hmmlearn"
            ) from e

        # Initialize HMM with k-means initialization
        self.hmm = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=self.random_state,
            init_params="stmc",  # Initialize all params
        )

        # Fit to data
        X_arr = X.values
        self.hmm.fit(X_arr)

        # Check for convergence
        if hasattr(self.hmm, "monitor_") and not self.hmm.monitor_.converged:
            warnings.warn(
                f"HMM did not converge after {n_iter} iterations. "
                f"Consider increasing n_iter or checking data quality.",
                UserWarning,
            )

        # Re-order regimes by mean ASRI (using baseline weights)
        baseline_weights = np.array([0.30, 0.25, 0.25, 0.20])
        regime_mean_asri = self.hmm.means_ @ baseline_weights

        self._regime_order = np.argsort(regime_mean_asri)

        # Store reordered parameters for consistency
        self._means = self.hmm.means_[self._regime_order]
        self._covars = self.hmm.covars_[self._regime_order]
        self._transmat = self.hmm.transmat_[self._regime_order][:, self._regime_order]

    def _fit_custom_hmm(self, X: pd.DataFrame, n_iter: int) -> None:
        """Fit using custom RegimeDetector from asri.regime.hmm."""
        from ..regime.hmm import RegimeDetector

        detector = RegimeDetector(
            n_regimes=self.n_regimes,
            n_iterations=n_iter,
            random_state=self.random_state,
        )
        detector.fit(X)

        self.hmm = detector
        self._regime_order = np.arange(self.n_regimes)  # Already sorted in detector

        result = detector.result
        self._means = result.regime_means
        self._covars = result.regime_covariances
        self._transmat = result.transition_matrix

    def get_regime_probabilities(
        self,
        subindices: pd.DataFrame,
        use_filtering: bool = False,
    ) -> np.ndarray:
        """
        Get posterior regime probabilities P(regime|data) for each timestep.

        Args:
            subindices: DataFrame with sub-index columns.
            use_filtering: If True, use forward algorithm only (real-time inference,
                no future information - avoids look-ahead bias). If False (default),
                use forward-backward algorithm (retrospective smoothing, uses all data).

        Returns:
            Array of shape (n_timesteps, n_regimes) with probabilities.
            Rows sum to 1.

        Raises:
            ValueError: If HMM not fitted.

        Notes:
            - Filtering (use_filtering=True): P(regime_t | data_1:t)
              Suitable for real-time applications and backtesting.
            - Smoothing (use_filtering=False): P(regime_t | data_1:T)
              Provides more accurate regime estimates but uses future information.
        """
        if not self._fitted:
            raise ValueError("Must call fit_hmm() first")

        X = subindices[SUBINDEX_COLUMNS].dropna().values

        if self._use_hmmlearn:
            if use_filtering:
                # Forward-only algorithm for filtering (no look-ahead bias)
                posteriors = self._compute_filtering_probabilities(X)
            else:
                # Forward-backward algorithm (smoothing - uses future data)
                _, posteriors = self.hmm.score_samples(X)
            # Reorder to match sorted regime indices
            posteriors = posteriors[:, self._regime_order]
        else:
            posteriors = self.hmm.predict(subindices[SUBINDEX_COLUMNS].dropna())

        return posteriors

    def _compute_filtering_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Compute filtering probabilities using forward algorithm only.

        This avoids look-ahead bias by only using past and current observations
        to estimate P(regime_t | data_1:t).

        Args:
            X: Observation array of shape (T, n_features).

        Returns:
            Array of shape (T, n_regimes) with filtered probabilities.
        """
        from scipy.special import logsumexp

        T = len(X)
        n_components = self.hmm.n_components

        # Get HMM parameters
        log_startprob = np.log(self.hmm.startprob_ + 1e-300)
        log_transmat = np.log(self.hmm.transmat_ + 1e-300)

        # Compute log emission probabilities
        log_prob = np.zeros((T, n_components))
        for i in range(n_components):
            mean = self.hmm.means_[i]
            if self.hmm.covariance_type == "full":
                cov = self.hmm.covars_[i]
            elif self.hmm.covariance_type == "diag":
                cov = np.diag(self.hmm.covars_[i])
            elif self.hmm.covariance_type == "spherical":
                cov = np.eye(X.shape[1]) * self.hmm.covars_[i]
            else:
                cov = self.hmm.covars_[i]

            # Multivariate normal log probability
            diff = X - mean
            try:
                cov_inv = np.linalg.inv(cov)
                log_det = np.log(np.linalg.det(cov) + 1e-300)
            except np.linalg.LinAlgError:
                cov_inv = np.eye(X.shape[1])
                log_det = 0

            k = X.shape[1]
            mahal = np.sum(diff @ cov_inv * diff, axis=1)
            log_prob[:, i] = -0.5 * (k * np.log(2 * np.pi) + log_det + mahal)

        # Forward pass only (filtering)
        log_alpha = np.zeros((T, n_components))

        # Initialize
        log_alpha[0] = log_startprob + log_prob[0]

        # Forward recursion
        for t in range(1, T):
            for j in range(n_components):
                log_alpha[t, j] = logsumexp(
                    log_alpha[t - 1] + log_transmat[:, j]
                ) + log_prob[t, j]

        # Normalize to get filtering probabilities
        posteriors = np.zeros((T, n_components))
        for t in range(T):
            log_norm = logsumexp(log_alpha[t])
            posteriors[t] = np.exp(log_alpha[t] - log_norm)

        return posteriors

    def compute_model_selection_criteria(
        self,
        subindices: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Compute AIC and BIC for HMM model selection.

        These information criteria help select the optimal number of regimes
        by balancing model fit (log-likelihood) against complexity (number
        of parameters). Lower values indicate better models.

        The number of parameters for a Gaussian HMM with K states and D features:
            - Initial probabilities: K - 1 (constrained to sum to 1)
            - Transition matrix: K * (K - 1) (each row sums to 1)
            - Means: K * D
            - Covariances: depends on covariance_type
                - 'full': K * D * (D + 1) / 2 (symmetric positive definite)
                - 'diag': K * D
                - 'spherical': K

        Args:
            subindices: DataFrame with sub-index columns.

        Returns:
            Dictionary with keys:
                - 'log_likelihood': Total log-likelihood of the data
                - 'aic': Akaike Information Criterion (-2*LL + 2*k)
                - 'bic': Bayesian Information Criterion (-2*LL + k*log(n))
                - 'n_params': Number of free parameters
                - 'n_samples': Number of observations

        Raises:
            ValueError: If HMM not fitted or hmmlearn not used.

        References:
            - Akaike (1974) "A new look at the statistical model identification"
            - Schwarz (1978) "Estimating the dimension of a model"
        """
        if not self._fitted:
            raise ValueError("Must call fit_hmm() first")

        if not self._use_hmmlearn:
            raise ValueError(
                "Model selection criteria only available with hmmlearn backend. "
                "Set use_hmmlearn=True when initializing RegimeAggregator."
            )

        X = subindices[SUBINDEX_COLUMNS].dropna().values
        n_samples, n_features = X.shape

        # Log-likelihood: hmmlearn.score() returns total log probability of sequence
        ll = self.hmm.score(X)

        # Count free parameters
        k = self.n_regimes

        # Initial probabilities: k-1 free parameters (constrained to sum to 1)
        n_params = k - 1

        # Transition matrix: k rows, each with k-1 free parameters
        n_params += k * (k - 1)

        # Means: k states * d features
        n_params += k * n_features

        # Covariances: depends on covariance_type
        cov_type = getattr(self.hmm, 'covariance_type', 'full')
        if cov_type == 'full':
            # Symmetric positive definite: d*(d+1)/2 unique elements per state
            n_params += k * n_features * (n_features + 1) // 2
        elif cov_type == 'diag':
            # Diagonal: d elements per state
            n_params += k * n_features
        elif cov_type == 'spherical':
            # Single variance per state
            n_params += k
        elif cov_type == 'tied':
            # Single full covariance shared across states
            n_params += n_features * (n_features + 1) // 2
        else:
            # Default to full if unknown
            n_params += k * n_features * (n_features + 1) // 2

        # Compute information criteria
        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + n_params * np.log(n_samples)

        return {
            'log_likelihood': ll,
            'aic': aic,
            'bic': bic,
            'n_params': n_params,
            'n_samples': n_samples,
        }

    def get_most_likely_regimes(self, subindices: pd.DataFrame) -> np.ndarray:
        """
        Viterbi decoding: most likely regime sequence.

        Uses the Viterbi algorithm to find the single most probable
        sequence of hidden states given all observations.

        Args:
            subindices: DataFrame with sub-index columns.

        Returns:
            Array of regime labels (integers 0 to n_regimes-1).
        """
        if not self._fitted:
            raise ValueError("Must call fit_hmm() first")

        X = subindices[SUBINDEX_COLUMNS].dropna().values

        if self._use_hmmlearn:
            raw_states = self.hmm.predict(X)
            # Map to sorted regime indices
            states = np.array([
                np.where(self._regime_order == s)[0][0] for s in raw_states
            ])
        else:
            states = self.hmm.result.regime_labels

        return states

    def aggregate(
        self,
        subindices: pd.DataFrame,
        method: str = "probability_weighted",
    ) -> pd.Series:
        """
        Compute regime-conditional ASRI aggregation.

        The main aggregation formula:
            ASRI_t = sum_k P(regime=k|data_t) * (w_k' @ s_t)

        This produces a smooth, continuously-varying index that reflects
        regime uncertainty rather than hard switching.

        Args:
            subindices: DataFrame with sub-index columns and datetime index.
            method: Aggregation method:
                - 'probability_weighted': Weight by regime posteriors (default)
                - 'viterbi': Use most likely regime only (hard switching)

        Returns:
            Series with regime-weighted ASRI values.
        """
        if not self._fitted:
            raise ValueError("Must call fit_hmm() first")

        X = subindices[SUBINDEX_COLUMNS].dropna()
        s_t = X.values  # (T, 4) array of sub-indices

        if method == "probability_weighted":
            probs = self.get_regime_probabilities(X)  # (T, n_regimes)
            W = self.regime_weights.as_matrix()  # (n_regimes, 4)

            # For each timestep, compute weighted combination
            # ASRI_t = sum_k P(k|t) * (w_k @ s_t)
            asri_values = np.zeros(len(X))
            for t in range(len(X)):
                for k in range(self.n_regimes):
                    asri_values[t] += probs[t, k] * (W[k] @ s_t[t])

        elif method == "viterbi":
            states = self.get_most_likely_regimes(X)
            W = self.regime_weights.as_matrix()

            asri_values = np.array([
                W[states[t]] @ s_t[t] for t in range(len(X))
            ])

        else:
            raise ValueError(f"Unknown method: {method}")

        return pd.Series(asri_values, index=X.index, name="asri_regime_weighted")

    def compute_effective_weights(self, subindices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute time-varying effective weights.

        The effective weight for sub-index j at time t is:
            w_eff_j(t) = sum_k P(regime=k|t) * w_k_j

        This shows how the implicit weighting scheme varies over time.

        Args:
            subindices: DataFrame with sub-index columns.

        Returns:
            DataFrame with effective weights for each sub-index over time.
        """
        if not self._fitted:
            raise ValueError("Must call fit_hmm() first")

        X = subindices[SUBINDEX_COLUMNS].dropna()
        probs = self.get_regime_probabilities(X)  # (T, n_regimes)
        W = self.regime_weights.as_matrix()  # (n_regimes, 4)

        # w_eff(t) = P(t) @ W, where P(t) is 1 x n_regimes, W is n_regimes x 4
        effective = probs @ W  # (T, 4)

        return pd.DataFrame(effective, index=X.index, columns=SUBINDEX_COLUMNS)

    def calibrate_regime_weights(
        self,
        subindices: pd.DataFrame,
        crisis_dates: list[str],
        objective: str = "detection_rate",
        n_trials: int = 100,
        window_days: int = 30,
    ) -> RegimeWeights:
        """
        Optimize regime weights to maximize crisis detection.

        Uses constrained optimization to find weight vectors that
        maximize the chosen objective while maintaining valid
        probability constraints (non-negative, sum to 1).

        Args:
            subindices: DataFrame with sub-index columns.
            crisis_dates: List of crisis event dates (str or datetime).
            objective: Optimization criterion:
                - 'detection_rate': Maximize fraction of crises detected
                - 'lead_time': Maximize average lead time before crises
                - 'f1': Maximize F1 score for crisis prediction
            n_trials: Number of random restarts for optimization.
            window_days: Days before crisis that count as detection.

        Returns:
            Optimized RegimeWeights.
        """
        if not self._fitted:
            raise ValueError("Must call fit_hmm() first")

        X = subindices[SUBINDEX_COLUMNS].dropna()
        probs = self.get_regime_probabilities(X)

        # Convert crisis dates to timestamps
        crisis_timestamps = [pd.Timestamp(d) for d in crisis_dates]

        # Objective function
        def neg_objective(flat_weights: np.ndarray) -> float:
            # Reshape: (12,) -> (3, 4)
            W = flat_weights.reshape(3, 4)

            # Normalize rows to sum to 1
            W = W / W.sum(axis=1, keepdims=True)

            # Compute ASRI
            s_t = X.values
            asri_values = np.zeros(len(X))
            for t in range(len(X)):
                for k in range(self.n_regimes):
                    asri_values[t] += probs[t, k] * (W[k] @ s_t[t])

            asri = pd.Series(asri_values, index=X.index)

            if objective == "detection_rate":
                # Count crises detected (ASRI > 70 within window before crisis)
                detected = 0
                for crisis_date in crisis_timestamps:
                    window_start = crisis_date - pd.Timedelta(days=window_days)
                    window_asri = asri[
                        (asri.index >= window_start) & (asri.index < crisis_date)
                    ]
                    if len(window_asri) > 0 and window_asri.max() > 70:
                        detected += 1
                return -detected / len(crisis_timestamps) if crisis_timestamps else 0

            elif objective == "lead_time":
                # Average days before crisis that ASRI first exceeds threshold
                lead_times = []
                for crisis_date in crisis_timestamps:
                    window_start = crisis_date - pd.Timedelta(days=60)
                    window_asri = asri[
                        (asri.index >= window_start) & (asri.index < crisis_date)
                    ]
                    above_thresh = window_asri[window_asri > 70]
                    if len(above_thresh) > 0:
                        first_alert = above_thresh.index[0]
                        lead_days = (crisis_date - first_alert).days
                        lead_times.append(lead_days)
                return -np.mean(lead_times) if lead_times else 0

            elif objective == "f1":
                # Simplified F1 calculation
                threshold = 70
                alerts = asri > threshold

                # True positives: alerts within window before crisis
                tp = 0
                for crisis_date in crisis_timestamps:
                    window_start = crisis_date - pd.Timedelta(days=window_days)
                    window_alerts = alerts[
                        (alerts.index >= window_start) & (alerts.index < crisis_date)
                    ]
                    if window_alerts.any():
                        tp += 1

                # Precision and recall
                total_alerts = alerts.sum()
                precision = tp / total_alerts if total_alerts > 0 else 0
                recall = tp / len(crisis_timestamps) if crisis_timestamps else 0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )
                return -f1

            else:
                raise ValueError(f"Unknown objective: {objective}")

        # Constraints: each row sums to 1, all elements >= 0
        constraints = [
            {"type": "eq", "fun": lambda w: w[0:4].sum() - 1},
            {"type": "eq", "fun": lambda w: w[4:8].sum() - 1},
            {"type": "eq", "fun": lambda w: w[8:12].sum() - 1},
        ]
        bounds = [(0.05, 0.60)] * 12  # Keep weights in reasonable range

        # Multi-start optimization
        best_weights = None
        best_score = np.inf
        rng = np.random.default_rng(self.random_state)

        for _ in range(n_trials):
            # Random initialization
            x0 = rng.dirichlet(np.ones(4), size=3).flatten()

            try:
                result = optimize.minimize(
                    neg_objective,
                    x0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 100},
                )

                if result.fun < best_score:
                    best_score = result.fun
                    best_weights = result.x

            except Exception:
                continue

        if best_weights is None:
            warnings.warn("Optimization failed, returning default weights")
            return self.regime_weights

        # Reshape and normalize
        W = best_weights.reshape(3, 4)
        W = W / W.sum(axis=1, keepdims=True)

        return RegimeWeights(
            low_risk=W[0],
            moderate=W[1],
            crisis=W[2],
        )

    def analyze_regime_transitions(
        self,
        subindices: pd.DataFrame,
        crisis_dates: list[str],
        pre_window: int = 30,
        post_window: int = 30,
    ) -> list[TransitionAnalysis]:
        """
        Analyze regime behavior around crisis events.

        Examines how regime probabilities evolve before, during, and
        after each crisis, which helps validate the regime model.

        Args:
            subindices: DataFrame with sub-index columns.
            crisis_dates: List of crisis event dates.
            pre_window: Days before crisis to analyze.
            post_window: Days after crisis to analyze.

        Returns:
            List of TransitionAnalysis, one per crisis event.
        """
        if not self._fitted:
            raise ValueError("Must call fit_hmm() first")

        X = subindices[SUBINDEX_COLUMNS].dropna()
        probs_arr = self.get_regime_probabilities(X)
        probs_df = pd.DataFrame(
            probs_arr,
            index=X.index,
            columns=[f"regime_{i}" for i in range(self.n_regimes)],
        )

        results = []
        for crisis_date_str in crisis_dates:
            crisis_date = pd.Timestamp(crisis_date_str)

            # Define windows
            pre_start = crisis_date - pd.Timedelta(days=pre_window)
            pre_end = crisis_date - pd.Timedelta(days=7)
            crisis_start = crisis_date - pd.Timedelta(days=7)
            crisis_end = crisis_date + pd.Timedelta(days=7)
            post_start = crisis_date + pd.Timedelta(days=7)
            post_end = crisis_date + pd.Timedelta(days=post_window)

            # Extract windows
            pre_probs = probs_df[(probs_df.index >= pre_start) & (probs_df.index < pre_end)]
            crisis_probs = probs_df[
                (probs_df.index >= crisis_start) & (probs_df.index <= crisis_end)
            ]
            post_probs = probs_df[
                (probs_df.index > post_start) & (probs_df.index <= post_end)
            ]

            # Compute summary statistics
            pre_crisis_prob = (
                pre_probs[f"regime_{self.n_regimes - 1}"].mean()
                if len(pre_probs) > 0
                else 0
            )
            during_crisis_prob = (
                crisis_probs[f"regime_{self.n_regimes - 1}"].mean()
                if len(crisis_probs) > 0
                else 0
            )
            prob_increase = during_crisis_prob - pre_crisis_prob

            # Find when crisis probability first exceeded 0.5
            crisis_col = f"regime_{self.n_regimes - 1}"
            lookback = probs_df[
                (probs_df.index >= pre_start) & (probs_df.index <= crisis_date)
            ]
            above_threshold = lookback[lookback[crisis_col] > 0.5]
            if len(above_threshold) > 0:
                first_breach = above_threshold.index[0]
                lead_days = (crisis_date - first_breach).days
            else:
                lead_days = 0

            results.append(
                TransitionAnalysis(
                    event_name=crisis_date_str,
                    event_date=crisis_date,
                    pre_crisis_probs=pre_probs,
                    crisis_probs=crisis_probs,
                    post_crisis_probs=post_probs,
                    crisis_regime_prob_increase=prob_increase,
                    transition_lead_days=lead_days,
                )
            )

        return results

    def full_analysis(
        self,
        subindices: pd.DataFrame,
        baseline_weights: dict[str, float] | None = None,
    ) -> RegimeAggregationResult:
        """
        Run complete regime aggregation analysis.

        Computes the regime-weighted ASRI and compares it to the
        baseline fixed-weight approach.

        Args:
            subindices: DataFrame with sub-index columns.
            baseline_weights: Fixed weights for comparison. If None,
                uses default [0.30, 0.25, 0.25, 0.20].

        Returns:
            RegimeAggregationResult with all computed values.
        """
        if not self._fitted:
            raise ValueError("Must call fit_hmm() first")

        # Default baseline weights
        if baseline_weights is None:
            baseline_weights = {
                "stablecoin_risk": 0.30,
                "defi_liquidity_risk": 0.25,
                "contagion_risk": 0.25,
                "arbitrage_opacity": 0.20,
            }

        X = subindices[SUBINDEX_COLUMNS].dropna()

        # Compute regime-weighted ASRI
        asri_regime = self.aggregate(X)

        # Compute baseline ASRI
        w_baseline = np.array([baseline_weights[c] for c in SUBINDEX_COLUMNS])
        asri_baseline = pd.Series(
            X.values @ w_baseline, index=X.index, name="asri_baseline"
        )

        # Get regime information
        probs_arr = self.get_regime_probabilities(X)
        probs_df = pd.DataFrame(
            probs_arr,
            index=X.index,
            columns=[f"regime_{i}" for i in range(self.n_regimes)],
        )
        states = self.get_most_likely_regimes(X)
        states_series = pd.Series(states, index=X.index, name="regime")

        # Effective weights
        eff_weights = self.compute_effective_weights(X)

        # Regime statistics
        regime_means = {}
        regime_freqs = {}
        for k in range(self.n_regimes):
            mask = states == k
            regime_means[k] = asri_regime[mask].mean() if mask.any() else 0
            regime_freqs[k] = mask.mean()

        # Improvement metric (placeholder - would need crisis dates for proper eval)
        improvement = np.abs(asri_regime - asri_baseline).mean()

        return RegimeAggregationResult(
            asri_regime_weighted=asri_regime,
            asri_baseline=asri_baseline,
            improvement=improvement,
            regime_probabilities=probs_df,
            most_likely_regimes=states_series,
            regime_weights_used=self.regime_weights,
            effective_weights=eff_weights,
            regime_means=regime_means,
            regime_frequencies=regime_freqs,
        )

    def plot_regime_overlay(
        self,
        subindices: pd.DataFrame,
        asri_series: pd.Series | None = None,
        output_path: str | None = None,
        figsize: tuple[float, float] = (14, 8),
    ) -> None:
        """
        Plot ASRI with regime coloring overlay.

        Creates a figure with:
            1. Top panel: ASRI with regime-colored background
            2. Bottom panel: Regime probabilities over time

        Args:
            subindices: DataFrame with sub-index columns.
            asri_series: ASRI series to plot. If None, computes from subindices.
            output_path: Path to save figure. If None, displays interactively.
            figsize: Figure dimensions (width, height) in inches.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError as e:
            raise ImportError("matplotlib is required for plotting") from e

        if not self._fitted:
            raise ValueError("Must call fit_hmm() first")

        X = subindices[SUBINDEX_COLUMNS].dropna()

        # Get ASRI
        if asri_series is None:
            asri_series = self.aggregate(X)
        else:
            # Align to X index
            asri_series = asri_series.loc[X.index]

        # Get regime data
        probs = self.get_regime_probabilities(X)
        states = self.get_most_likely_regimes(X)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Regime colors: green (low risk) -> yellow (moderate) -> red (crisis)
        regime_colors = ["#2ecc71", "#f1c40f", "#e74c3c"]

        # Panel 1: ASRI with regime background
        dates = X.index

        # Plot regime backgrounds
        prev_regime = states[0]
        start_idx = 0
        for i in range(1, len(states)):
            if states[i] != prev_regime or i == len(states) - 1:
                end_idx = i if i < len(states) - 1 else i + 1
                ax1.axvspan(
                    dates[start_idx],
                    dates[min(end_idx, len(dates) - 1)],
                    alpha=0.3,
                    color=regime_colors[prev_regime],
                    label=f"Regime {prev_regime}" if start_idx == 0 else "",
                )
                start_idx = i
                prev_regime = states[i]

        # Plot ASRI
        ax1.plot(dates, asri_series.values, "k-", linewidth=1.5, label="ASRI")

        # Alert thresholds
        ax1.axhline(70, color="orange", linestyle="--", alpha=0.7, label="Elevated")
        ax1.axhline(85, color="red", linestyle="--", alpha=0.7, label="High")

        ax1.set_ylabel("ASRI")
        ax1.set_title("Regime-Conditional ASRI with Regime Overlay")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Panel 2: Regime probabilities
        regime_names = ["Low Risk", "Moderate", "Crisis"]
        for k in range(self.n_regimes):
            name = regime_names[k] if k < len(regime_names) else f"Regime {k}"
            ax2.plot(
                dates,
                probs[:, k],
                color=regime_colors[k],
                linewidth=1.5,
                label=name,
            )

        ax2.set_ylabel("Probability")
        ax2.set_xlabel("Date")
        ax2.set_title("Regime Posterior Probabilities")
        ax2.legend(loc="upper left")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def save_model(self, path: str | Path) -> None:
        """
        Save fitted HMM model to disk.

        Args:
            path: File path for saving (recommend .pkl extension).
        """
        if not self._fitted:
            raise ValueError("Must call fit_hmm() first")

        state = {
            "n_regimes": self.n_regimes,
            "regime_weights": self.regime_weights,
            "random_state": self.random_state,
            "use_hmmlearn": self._use_hmmlearn,
            "regime_order": self._regime_order,
            "means": self._means,
            "covars": self._covars,
            "transmat": self._transmat,
        }

        if self._use_hmmlearn:
            state["hmm_model"] = self.hmm

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_model(cls, path: str | Path) -> RegimeAggregator:
        """
        Load a pre-trained model from disk.

        Args:
            path: File path to load from.

        Returns:
            RegimeAggregator with restored state.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        aggregator = cls(
            n_regimes=state["n_regimes"],
            regime_weights=state["regime_weights"],
            use_hmmlearn=state["use_hmmlearn"],
            random_state=state["random_state"],
        )

        aggregator._regime_order = state["regime_order"]
        aggregator._means = state["means"]
        aggregator._covars = state["covars"]
        aggregator._transmat = state["transmat"]

        if state["use_hmmlearn"]:
            aggregator.hmm = state["hmm_model"]

        aggregator._fitted = True

        return aggregator


def format_regime_weights_table(weights: RegimeWeights) -> str:
    """
    Format regime weights as LaTeX table.

    Args:
        weights: RegimeWeights to format.

    Returns:
        LaTeX table string.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Regime-Specific ASRI Weights}",
        r"\label{tab:regime_weights}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Regime & SCR & DLR & CR & OR \\",
        r"\midrule",
    ]

    regime_data = [
        ("Low Risk", weights.low_risk),
        ("Moderate", weights.moderate),
        ("Crisis", weights.crisis),
    ]

    for name, w in regime_data:
        lines.append(f"{name} & {w[0]:.2f} & {w[1]:.2f} & {w[2]:.2f} & {w[3]:.2f} \\\\")

    lines.extend([
        r"\midrule",
        r"\textbf{Baseline} & 0.30 & 0.25 & 0.25 & 0.20 \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item SCR = Stablecoin Risk, DLR = DeFi Liquidity Risk,",
        r"\item CR = Contagion Risk, OR = Opacity Risk.",
        r"\item Weights sum to 1 within each regime.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_transition_analysis_table(analyses: list[TransitionAnalysis]) -> str:
    """
    Format transition analysis results as LaTeX table.

    Args:
        analyses: List of TransitionAnalysis results.

    Returns:
        LaTeX table string.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Regime Transition Analysis Around Crisis Events}",
        r"\label{tab:regime_transitions}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Event & Date & P(Crisis) Increase & Lead Days \\",
        r"\midrule",
    ]

    for a in analyses:
        date_str = a.event_date.strftime("%Y-%m")
        lines.append(
            f"{a.event_name[:20]} & {date_str} & "
            f"{a.crisis_regime_prob_increase:+.2f} & {a.transition_lead_days} \\\\"
        )

    # Summary
    avg_increase = np.mean([a.crisis_regime_prob_increase for a in analyses])
    avg_lead = np.mean([a.transition_lead_days for a in analyses])

    lines.extend([
        r"\midrule",
        f"\\textbf{{Average}} & -- & {avg_increase:+.2f} & {avg_lead:.1f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item P(Crisis) Increase = change in crisis regime probability from pre-event to during-event.",
        r"\item Lead Days = days before event that crisis probability exceeded 50\%.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Demonstration with synthetic data
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Regime-Conditional ASRI Aggregation - Synthetic Data Demo")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic sub-index data with regime structure
    n_days = 500
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")

    # True regime sequence: mostly moderate with periods of crisis and low-risk
    true_regimes = np.zeros(n_days, dtype=int)
    true_regimes[50:80] = 2  # Crisis period 1
    true_regimes[150:170] = 0  # Low risk period
    true_regimes[250:290] = 2  # Crisis period 2
    true_regimes[350:380] = 2  # Crisis period 3
    true_regimes[true_regimes == 0] = 1  # Default moderate
    true_regimes[:50] = 1
    true_regimes[80:150] = 1
    true_regimes[170:250] = 1
    true_regimes[290:350] = 1
    true_regimes[380:] = 1

    # Regime-specific means for sub-indices
    regime_means = {
        0: np.array([25, 30, 35, 30]),  # Low risk: all low
        1: np.array([35, 40, 40, 35]),  # Moderate: medium levels
        2: np.array([70, 65, 55, 45]),  # Crisis: high stablecoin/liquidity
    }

    # Generate data
    subindex_data = np.zeros((n_days, 4))
    for t in range(n_days):
        regime = true_regimes[t]
        mean = regime_means[regime]
        # Add autocorrelated noise
        noise = np.random.multivariate_normal(
            np.zeros(4),
            np.eye(4) * 25,  # Variance
        )
        if t > 0:
            subindex_data[t] = 0.7 * subindex_data[t - 1] + 0.3 * (mean + noise)
        else:
            subindex_data[t] = mean + noise

    # Clip to valid range
    subindex_data = np.clip(subindex_data, 0, 100)

    subindices = pd.DataFrame(
        subindex_data,
        index=dates,
        columns=SUBINDEX_COLUMNS,
    )

    print("\nSynthetic Data Summary:")
    print(subindices.describe().round(2))

    # Create and fit aggregator
    print("\n" + "-" * 70)
    print("Fitting HMM...")

    aggregator = RegimeAggregator(n_regimes=3, use_hmmlearn=True)
    aggregator.fit_hmm(subindices, n_iter=100)

    print("HMM fitted successfully.")

    # Get regime probabilities
    probs = aggregator.get_regime_probabilities(subindices)
    states = aggregator.get_most_likely_regimes(subindices)

    print("\nDetected Regime Frequencies:")
    for k in range(3):
        freq = (states == k).mean()
        names = ["Low Risk", "Moderate", "Crisis"]
        print(f"  {names[k]}: {freq:.1%}")

    # Compare true vs detected regimes
    accuracy = (states == true_regimes).mean()
    print(f"\nRegime Detection Accuracy (vs synthetic truth): {accuracy:.1%}")

    # Run full analysis
    print("\n" + "-" * 70)
    print("Running full analysis...")

    result = aggregator.full_analysis(subindices)

    print("\nRegime-Weighted vs Baseline ASRI:")
    print(f"  Mean regime-weighted: {result.asri_regime_weighted.mean():.2f}")
    print(f"  Mean baseline:        {result.asri_baseline.mean():.2f}")
    print(f"  Mean absolute diff:   {result.improvement:.2f}")

    print("\nRegime Means (ASRI by detected regime):")
    for k, mean_asri in result.regime_means.items():
        names = ["Low Risk", "Moderate", "Crisis"]
        print(f"  {names[k]}: {mean_asri:.1f}")

    # Effective weights at specific points
    print("\nEffective Weights (sample timepoints):")
    eff = result.effective_weights
    sample_dates = [dates[60], dates[200], dates[260]]  # Crisis, moderate, crisis
    for d in sample_dates:
        if d in eff.index:
            w = eff.loc[d]
            regime = states[dates.get_loc(d)]
            names = ["Low Risk", "Moderate", "Crisis"]
            print(f"  {d.date()} (Regime: {names[regime]}): " +
                  f"SCR={w.iloc[0]:.2f}, DLR={w.iloc[1]:.2f}, " +
                  f"CR={w.iloc[2]:.2f}, OR={w.iloc[3]:.2f}")

    # Simulate crisis dates for transition analysis
    crisis_dates = ["2022-02-25", "2022-09-15", "2022-12-20"]

    print("\n" + "-" * 70)
    print("Analyzing regime transitions around 'crisis' events...")

    transitions = aggregator.analyze_regime_transitions(subindices, crisis_dates)

    for t in transitions:
        print(f"\n  {t.event_name}:")
        print(f"    P(Crisis) increase: {t.crisis_regime_prob_increase:+.2f}")
        print(f"    Lead time: {t.transition_lead_days} days")

    # Print LaTeX tables
    print("\n" + "-" * 70)
    print("LaTeX Table: Regime Weights")
    print("-" * 70)
    print(format_regime_weights_table(aggregator.regime_weights))

    print("\n" + "-" * 70)
    print("LaTeX Table: Transition Analysis")
    print("-" * 70)
    print(format_transition_analysis_table(transitions))

    # Test model save/load
    print("\n" + "-" * 70)
    print("Testing model persistence...")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name

    aggregator.save_model(temp_path)
    loaded = RegimeAggregator.load_model(temp_path)

    # Verify loaded model produces same results
    loaded_asri = loaded.aggregate(subindices)
    diff = np.abs(loaded_asri - result.asri_regime_weighted).max()
    print(f"Max difference after save/load: {diff:.10f}")

    import os
    os.unlink(temp_path)

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)
