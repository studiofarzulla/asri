"""
Hidden Markov Model for Regime Detection

DeFi markets exhibit distinct regimes: bull, bear, crisis, recovery.
A single set of ASRI weights may be suboptimal across all regimes.

This module:
1. Detects regimes from sub-index dynamics
2. Estimates regime-specific weights
3. Provides regime probability forecasts
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class RegimeResult:
    """Results from regime detection."""
    n_regimes: int
    regime_labels: np.ndarray  # Most likely regime at each time
    regime_probabilities: np.ndarray  # T x n_regimes probability matrix
    
    # Regime characteristics
    regime_means: np.ndarray  # n_regimes x n_features
    regime_covariances: np.ndarray  # n_regimes x n_features x n_features
    
    # Transition matrix
    transition_matrix: np.ndarray  # n_regimes x n_regimes
    
    # Regime interpretation
    regime_names: list[str]
    
    # Model quality
    log_likelihood: float
    aic: float
    bic: float
    
    # Regime-specific weights (if computed)
    regime_weights: dict[int, dict[str, float]] = field(default_factory=dict)


class RegimeDetector:
    """
    Detect market regimes using Gaussian Hidden Markov Model.
    
    Regimes are learned from sub-index dynamics and can be used
    to switch ASRI weights dynamically.
    """
    
    REGIME_NAMES = {
        2: ["Low Risk", "High Risk"],
        3: ["Low Risk", "Moderate", "Crisis"],
        4: ["Bull", "Consolidation", "Stress", "Crisis"],
    }
    
    def __init__(
        self,
        n_regimes: int = 3,
        n_iterations: int = 100,
        convergence_threshold: float = 1e-4,
        random_state: int = 42,
    ):
        """
        Args:
            n_regimes: Number of hidden states/regimes
            n_iterations: Maximum EM iterations
            convergence_threshold: Log-likelihood change for convergence
            random_state: Random seed for initialization
        """
        self.n_regimes = n_regimes
        self.n_iterations = n_iterations
        self.convergence_threshold = convergence_threshold
        self.random_state = random_state
        self._result: RegimeResult | None = None
    
    def fit(self, data: pd.DataFrame) -> "RegimeDetector":
        """
        Fit HMM to sub-index data.
        
        Args:
            data: DataFrame with sub-indices as columns
            
        Returns:
            Self (for method chaining)
        """
        # Prepare data
        X = data.dropna().values
        T, K = X.shape
        N = self.n_regimes
        
        # Initialize parameters
        rng = np.random.default_rng(self.random_state)
        
        # Initial state probabilities
        pi = np.ones(N) / N
        
        # Transition matrix (slightly persistent)
        A = np.full((N, N), 0.1 / (N - 1))
        np.fill_diagonal(A, 0.9)
        
        # Emission parameters (means and covariances)
        # Initialize with k-means-like clustering
        percentiles = np.linspace(0, 100, N + 2)[1:-1]
        initial_means = np.percentile(X, percentiles, axis=0)
        
        means = initial_means + rng.normal(0, 0.1, (N, K))
        
        # Shared initial covariance
        global_cov = np.cov(X.T)
        covs = np.array([global_cov.copy() for _ in range(N)])
        
        # EM algorithm
        prev_ll = -np.inf
        
        for iteration in range(self.n_iterations):
            # E-step: Forward-backward algorithm
            alpha, beta, gamma, xi, ll = self._forward_backward(
                X, pi, A, means, covs
            )
            
            # Check convergence
            if abs(ll - prev_ll) < self.convergence_threshold:
                break
            prev_ll = ll
            
            # M-step: Update parameters
            pi = gamma[0] / np.sum(gamma[0])
            
            # Transition matrix
            for i in range(N):
                for j in range(N):
                    A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
            
            # Emission parameters
            for i in range(N):
                gamma_sum = np.sum(gamma[:, i])
                
                # Mean
                means[i] = np.sum(gamma[:, i:i+1] * X, axis=0) / gamma_sum
                
                # Covariance
                diff = X - means[i]
                covs[i] = (gamma[:, i:i+1] * diff).T @ diff / gamma_sum
                
                # Add regularization for stability
                covs[i] += 1e-6 * np.eye(K)
        
        # Final E-step for results
        alpha, beta, gamma, xi, ll = self._forward_backward(
            X, pi, A, means, covs
        )
        
        # Most likely sequence (Viterbi)
        regime_labels = self._viterbi(X, pi, A, means, covs)
        
        # Compute model selection criteria
        n_params = N - 1 + N * (N - 1) + N * K + N * K * (K + 1) // 2
        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + n_params * np.log(T)
        
        # Regime names
        regime_names = self.REGIME_NAMES.get(N, [f"Regime {i}" for i in range(N)])
        
        # Sort regimes by mean risk level
        mean_risk = np.mean(means, axis=1)
        sort_idx = np.argsort(mean_risk)
        
        regime_labels = np.array([np.where(sort_idx == r)[0][0] for r in regime_labels])
        gamma = gamma[:, sort_idx]
        means = means[sort_idx]
        covs = covs[sort_idx]
        A = A[sort_idx][:, sort_idx]
        
        self._result = RegimeResult(
            n_regimes=N,
            regime_labels=regime_labels,
            regime_probabilities=gamma,
            regime_means=means,
            regime_covariances=covs,
            transition_matrix=A,
            regime_names=regime_names,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
        )
        
        self._data_columns = list(data.columns)
        
        return self
    
    def _gaussian_pdf(
        self,
        x: np.ndarray,
        mean: np.ndarray,
        cov: np.ndarray,
    ) -> float:
        """Multivariate Gaussian PDF."""
        K = len(mean)
        diff = x - mean
        
        try:
            cov_inv = np.linalg.inv(cov)
            det = np.linalg.det(cov)
        except np.linalg.LinAlgError:
            return 1e-300
        
        if det <= 0:
            return 1e-300
        
        exponent = -0.5 * diff @ cov_inv @ diff
        normalization = 1 / np.sqrt((2 * np.pi) ** K * det)
        
        return max(normalization * np.exp(exponent), 1e-300)
    
    def _forward_backward(
        self,
        X: np.ndarray,
        pi: np.ndarray,
        A: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
    ) -> tuple:
        """Forward-backward algorithm."""
        T, K = X.shape
        N = len(pi)
        
        # Emission probabilities
        B = np.zeros((T, N))
        for t in range(T):
            for i in range(N):
                B[t, i] = self._gaussian_pdf(X[t], means[i], covs[i])
        
        # Forward pass
        alpha = np.zeros((T, N))
        alpha[0] = pi * B[0]
        c = np.zeros(T)  # Scaling factors
        c[0] = np.sum(alpha[0])
        alpha[0] /= c[0]
        
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = B[t, j] * np.sum(alpha[t-1] * A[:, j])
            c[t] = np.sum(alpha[t])
            if c[t] > 0:
                alpha[t] /= c[t]
        
        # Backward pass
        beta = np.zeros((T, N))
        beta[-1] = 1
        
        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[t, i] = np.sum(A[i] * B[t+1] * beta[t+1])
            if c[t+1] > 0:
                beta[t] /= c[t+1]
        
        # Posterior probabilities
        gamma = alpha * beta
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1
        gamma /= gamma_sum
        
        # Xi (transition posteriors)
        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            denom = np.sum(alpha[t][:, np.newaxis] * A * B[t+1] * beta[t+1])
            if denom > 0:
                xi[t] = (alpha[t][:, np.newaxis] * A * B[t+1] * beta[t+1]) / denom
        
        # Log-likelihood
        ll = np.sum(np.log(c[c > 0]))
        
        return alpha, beta, gamma, xi, ll
    
    def _viterbi(
        self,
        X: np.ndarray,
        pi: np.ndarray,
        A: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
    ) -> np.ndarray:
        """Viterbi algorithm for most likely state sequence."""
        T = len(X)
        N = len(pi)
        
        # Emission log-probabilities
        log_B = np.zeros((T, N))
        for t in range(T):
            for i in range(N):
                log_B[t, i] = np.log(max(self._gaussian_pdf(X[t], means[i], covs[i]), 1e-300))
        
        log_pi = np.log(pi + 1e-300)
        log_A = np.log(A + 1e-300)
        
        # Viterbi recursion
        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)
        
        delta[0] = log_pi + log_B[0]
        
        for t in range(1, T):
            for j in range(N):
                probs = delta[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(probs)
                delta[t, j] = probs[psi[t, j]] + log_B[t, j]
        
        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        
        return states
    
    def compute_regime_weights(
        self,
        sub_indices: pd.DataFrame,
        target: pd.Series,
    ) -> dict[int, dict[str, float]]:
        """
        Compute optimal weights for each regime.
        
        Uses within-regime data to estimate weights via correlation.
        """
        if self._result is None:
            raise ValueError("Must call fit() first")
        
        from ..weights.elastic_net import ElasticNetWeightDeriver
        
        common_idx = sub_indices.index.intersection(target.index)
        X = sub_indices.loc[common_idx]
        y = target.loc[common_idx]
        
        # Align with regime labels
        labels = self._result.regime_labels
        min_len = min(len(X), len(labels))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]
        labels = labels[:min_len]
        
        regime_weights = {}
        
        for regime in range(self._result.n_regimes):
            mask = labels == regime
            
            if np.sum(mask) < 50:
                # Insufficient data, use equal weights
                regime_weights[regime] = {c: 1.0 / len(X.columns) for c in X.columns}
                continue
            
            X_regime = X.iloc[mask]
            y_regime = y.iloc[mask]
            
            try:
                deriver = ElasticNetWeightDeriver()
                deriver.fit(X_regime, y_regime)
                regime_weights[regime] = deriver.weights
            except Exception:
                regime_weights[regime] = {c: 1.0 / len(X.columns) for c in X.columns}
        
        self._result.regime_weights = regime_weights
        return regime_weights
    
    @property
    def result(self) -> RegimeResult:
        if self._result is None:
            raise ValueError("Must call fit() first")
        return self._result
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities for new data."""
        if self._result is None:
            raise ValueError("Must call fit() first")
        
        X = data.dropna().values
        T = len(X)
        N = self._result.n_regimes
        
        # Use result parameters
        probs = np.zeros((T, N))
        
        for t in range(T):
            for i in range(N):
                probs[t, i] = self._gaussian_pdf(
                    X[t],
                    self._result.regime_means[i],
                    self._result.regime_covariances[i],
                )
        
        # Normalize
        probs /= np.sum(probs, axis=1, keepdims=True)
        
        return probs


def detect_regimes(
    data: pd.DataFrame,
    n_regimes: int = 3,
) -> RegimeResult:
    """
    Convenience function to detect regimes.
    
    Args:
        data: DataFrame with sub-index time series
        n_regimes: Number of regimes to detect
        
    Returns:
        RegimeResult
    """
    detector = RegimeDetector(n_regimes=n_regimes)
    detector.fit(data)
    return detector.result


def format_regime_table(result: RegimeResult) -> str:
    """Format regime detection results as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Regime Characteristics}",
        r"\label{tab:regimes}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Regime & Frequency & Mean Risk & Persistence & Interpretation \\",
        r"\midrule",
    ]
    
    T = len(result.regime_labels)
    
    for i in range(result.n_regimes):
        freq = np.sum(result.regime_labels == i) / T
        mean_risk = np.mean(result.regime_means[i])
        persistence = result.transition_matrix[i, i]
        name = result.regime_names[i] if i < len(result.regime_names) else f"Regime {i}"
        
        lines.append(
            f"{i+1} & {freq:.1%} & {mean_risk:.1f} & {persistence:.2f} & {name} \\\\"
        )
    
    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{l}{\textbf{Model Fit}} \\",
        f"Log-likelihood: {result.log_likelihood:.1f} & "
        f"AIC: {result.aic:.1f} & BIC: {result.bic:.1f} & & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Persistence = probability of staying in same regime.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def plot_regimes(
    result: RegimeResult,
    dates: pd.DatetimeIndex,
    output_path: str = "figures/regimes.pdf",
) -> str:
    """Generate matplotlib code for regime plot."""
    return f"""
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Regime labels
dates = np.array([...])  # Your dates
labels = result.regime_labels

colors = ['green', 'yellow', 'red'][:result.n_regimes]
for i in range(result.n_regimes):
    mask = labels == i
    ax1.fill_between(dates, 0, 1, where=mask, alpha=0.3, color=colors[i],
                     label=result.regime_names[i])

ax1.set_ylabel('Regime')
ax1.legend(loc='upper right')
ax1.set_title('Detected Market Regimes')

# Regime probabilities
for i in range(result.n_regimes):
    ax2.plot(dates, result.regime_probabilities[:, i], 
             label=result.regime_names[i], color=colors[i])

ax2.set_ylabel('Probability')
ax2.set_xlabel('Date')
ax2.legend(loc='upper right')
ax2.set_title('Regime Probabilities')

plt.tight_layout()
plt.savefig('{output_path}')
"""
