"""
Elastic Net Weight Derivation for ASRI

Use regularized regression to predict forward market stress from sub-indices.
The regression coefficients become the weights—variables that don't predict
stress get zero weight via L1 penalty.

Key advantages:
1. Weights are explicitly predictive (unlike PCA which is descriptive)
2. L1 penalty performs variable selection (removes irrelevant sub-indices)
3. L2 penalty handles multicollinearity (sub-indices are correlated)
4. Cross-validation prevents overfitting
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class ElasticNetResult:
    """Results from Elastic Net weight derivation."""
    weights: dict[str, float]
    coefficients: dict[str, float]  # Raw coefficients before normalization
    intercept: float
    
    # Model selection
    optimal_alpha: float  # Regularization strength
    optimal_l1_ratio: float  # Balance between L1 and L2
    
    # Performance metrics
    r_squared_train: float
    r_squared_cv: float  # Cross-validated
    mse_cv: float
    
    # Feature selection
    n_nonzero: int  # Number of non-zero coefficients
    selected_features: list[str]
    dropped_features: list[str]
    
    # Cross-validation details
    cv_scores: list[float] = field(default_factory=list)


class TimeSeriesSplit:
    """
    Time series cross-validation splitter.
    
    Unlike standard k-fold, this respects temporal ordering:
    training data always precedes test data.
    """
    
    def __init__(self, n_splits: int = 5, gap: int = 0):
        self.n_splits = n_splits
        self.gap = gap
    
    def split(self, X: np.ndarray):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end + self.gap
            test_end = min(test_start + fold_size, n)
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            if len(test_idx) > 0:
                yield train_idx, test_idx


class ElasticNetWeightDeriver:
    """
    Derive ASRI weights via Elastic Net regression.
    
    Target variable: Forward drawdown, forward volatility, or crisis indicator
    Features: Sub-index values
    Weights: Normalized absolute coefficients
    """
    
    def __init__(
        self,
        l1_ratios: list[float] = [0.1, 0.5, 0.9, 0.99],
        alphas: list[float] | None = None,
        n_alphas: int = 50,
        cv: int | TimeSeriesSplit = 5,
        target_type: Literal["drawdown", "volatility", "binary"] = "drawdown",
        forward_window: int = 30,
    ):
        """
        Args:
            l1_ratios: L1/L2 balance ratios to try (0=Ridge, 1=Lasso)
            alphas: Regularization strengths to try (None = auto)
            n_alphas: Number of alphas if auto-generating
            cv: Number of CV folds or TimeSeriesSplit object
            target_type: What to predict ('drawdown', 'volatility', 'binary')
            forward_window: Days ahead for target calculation
        """
        self.l1_ratios = l1_ratios
        self.alphas = alphas
        self.n_alphas = n_alphas
        self.cv = cv if isinstance(cv, TimeSeriesSplit) else TimeSeriesSplit(cv)
        self.target_type = target_type
        self.forward_window = forward_window
        self._result: ElasticNetResult | None = None
    
    def fit(
        self,
        sub_indices: pd.DataFrame,
        market_returns: pd.Series | None = None,
    ) -> "ElasticNetWeightDeriver":
        """
        Fit Elastic Net and derive weights.
        
        Args:
            sub_indices: DataFrame with sub-index time series
            market_returns: Market returns for target calculation (optional)
            
        Returns:
            Self (for method chaining)
        """
        # Align data
        data = sub_indices.dropna()
        n = len(data)
        
        # Create target variable
        if market_returns is not None:
            y = self._create_target(market_returns, n)
        else:
            # Use negative of aggregate index as proxy for "stress"
            # (higher sub-indices = more stress = worse forward returns)
            aggregate = data.mean(axis=1)
            y = -aggregate.shift(-self.forward_window).dropna()
            data = data.iloc[:-self.forward_window]
        
        # Align lengths
        min_len = min(len(data), len(y))
        X = data.iloc[:min_len].values
        y = y.iloc[:min_len].values if hasattr(y, 'iloc') else y[:min_len]
        
        # Standardize features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # Prevent division by zero
        X_scaled = (X - X_mean) / X_std
        
        # Generate alpha grid if not provided
        if self.alphas is None:
            # Range from very small to very large regularization
            self.alphas = np.logspace(-4, 1, self.n_alphas)
        
        # Grid search over alphas and l1_ratios
        best_score = -np.inf
        best_alpha = self.alphas[0]
        best_l1_ratio = self.l1_ratios[0]
        best_coef = None
        best_intercept = 0
        all_cv_scores = []
        
        for l1_ratio in self.l1_ratios:
            for alpha in self.alphas:
                cv_scores = []
                
                for train_idx, test_idx in self.cv.split(X_scaled):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Fit elastic net
                    coef, intercept = self._fit_elastic_net(
                        X_train, y_train, alpha, l1_ratio
                    )
                    
                    # Predict and score
                    y_pred = X_test @ coef + intercept
                    ss_res = np.sum((y_test - y_pred) ** 2)
                    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    cv_scores.append(r2)
                
                mean_score = np.mean(cv_scores)
                all_cv_scores.append(mean_score)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha
                    best_l1_ratio = l1_ratio
                    
                    # Refit on full data
                    best_coef, best_intercept = self._fit_elastic_net(
                        X_scaled, y, alpha, l1_ratio
                    )
        
        # Compute weights from coefficients
        feature_names = list(data.columns)
        
        # Normalize coefficients to weights
        abs_coef = np.abs(best_coef)
        if np.sum(abs_coef) > 0:
            weights_arr = abs_coef / np.sum(abs_coef)
        else:
            weights_arr = np.ones(len(feature_names)) / len(feature_names)
        
        weights = {name: w for name, w in zip(feature_names, weights_arr)}
        coefficients = {name: c for name, c in zip(feature_names, best_coef)}
        
        # Feature selection
        selected = [name for name, c in coefficients.items() if abs(c) > 1e-6]
        dropped = [name for name in feature_names if name not in selected]
        
        # Train R-squared
        y_pred_train = X_scaled @ best_coef + best_intercept
        ss_res = np.sum((y - y_pred_train) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_train = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        self._result = ElasticNetResult(
            weights=weights,
            coefficients=coefficients,
            intercept=best_intercept,
            optimal_alpha=best_alpha,
            optimal_l1_ratio=best_l1_ratio,
            r_squared_train=r2_train,
            r_squared_cv=best_score,
            mse_cv=np.mean((y - y_pred_train) ** 2),
            n_nonzero=len(selected),
            selected_features=selected,
            dropped_features=dropped,
            cv_scores=all_cv_scores,
        )
        
        return self
    
    def _create_target(self, returns: pd.Series, n: int) -> np.ndarray:
        """Create target variable from returns."""
        if self.target_type == "drawdown":
            # Maximum drawdown over forward window
            target = []
            for i in range(n - self.forward_window):
                window = returns.iloc[i:i + self.forward_window]
                cumulative = (1 + window).cumprod()
                peak = cumulative.cummax()
                drawdown = (cumulative - peak) / peak
                target.append(drawdown.min())
            return np.array(target)
        
        elif self.target_type == "volatility":
            # Realized volatility over forward window
            return returns.rolling(self.forward_window).std().shift(-self.forward_window).dropna().values
        
        elif self.target_type == "binary":
            # Binary crisis indicator (drawdown > 20%)
            target = []
            for i in range(n - self.forward_window):
                window = returns.iloc[i:i + self.forward_window]
                cumulative = (1 + window).cumprod()
                peak = cumulative.cummax()
                drawdown = (cumulative - peak) / peak
                target.append(1 if drawdown.min() < -0.20 else 0)
            return np.array(target)
        
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")
    
    def _fit_elastic_net(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        l1_ratio: float,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> tuple[np.ndarray, float]:
        """
        Fit Elastic Net via coordinate descent.
        
        Elastic Net objective:
        1/(2n) ||y - Xβ||² + α * [l1_ratio * ||β||₁ + (1-l1_ratio)/2 * ||β||²]
        """
        n, p = X.shape
        
        # Initialize coefficients
        coef = np.zeros(p)
        intercept = np.mean(y)
        residual = y - intercept
        
        # Precompute X^T X diagonal for efficiency
        X_sq = np.sum(X ** 2, axis=0) / n
        
        for iteration in range(max_iter):
            coef_old = coef.copy()
            
            # Coordinate descent: update each coefficient
            for j in range(p):
                # Compute partial residual
                residual_j = residual + X[:, j] * coef[j]
                
                # Soft thresholding
                rho = np.dot(X[:, j], residual_j) / n
                
                # Elastic net update
                if l1_ratio == 1:
                    # Pure Lasso
                    coef[j] = np.sign(rho) * max(abs(rho) - alpha, 0) / X_sq[j]
                else:
                    # Elastic Net
                    coef[j] = np.sign(rho) * max(abs(rho) - alpha * l1_ratio, 0) / (
                        X_sq[j] + alpha * (1 - l1_ratio)
                    )
                
                # Update residual
                residual = residual_j - X[:, j] * coef[j]
            
            # Update intercept
            intercept = np.mean(y - X @ coef)
            residual = y - X @ coef - intercept
            
            # Check convergence
            if np.max(np.abs(coef - coef_old)) < tol:
                break
        
        return coef, intercept
    
    @property
    def result(self) -> ElasticNetResult:
        if self._result is None:
            raise ValueError("Must call fit() first")
        return self._result
    
    @property
    def weights(self) -> dict[str, float]:
        return self.result.weights


def derive_elastic_net_weights(
    sub_indices: pd.DataFrame,
    market_returns: pd.Series | None = None,
    forward_window: int = 30,
) -> dict[str, float]:
    """
    Convenience function to derive Elastic Net weights.
    
    Args:
        sub_indices: DataFrame with sub-index time series
        market_returns: Optional market returns for target
        forward_window: Days ahead for prediction target
        
    Returns:
        Dictionary of variable -> weight
    """
    deriver = ElasticNetWeightDeriver(forward_window=forward_window)
    deriver.fit(sub_indices, market_returns)
    return deriver.weights


def format_elastic_net_table(result: ElasticNetResult) -> str:
    """Format Elastic Net results as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Elastic Net Regression: Derived Weights}",
        r"\label{tab:elasticnet}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Sub-Index & Coefficient & Derived Weight & Selected \\",
        r"\midrule",
    ]
    
    for name in sorted(result.coefficients.keys()):
        coef = result.coefficients[name]
        weight = result.weights[name]
        selected = r"\checkmark" if name in result.selected_features else ""
        
        lines.append(f"{name} & {coef:.4f} & {weight:.3f} & {selected} \\\\")
    
    lines.extend([
        r"\midrule",
        f"\\multicolumn{{4}}{{l}}{{Optimal $\\alpha$: {result.optimal_alpha:.4f}, "
        f"$l_1$ ratio: {result.optimal_l1_ratio:.2f}}} \\\\",
        f"\\multicolumn{{4}}{{l}}{{CV $R^2$: {result.r_squared_cv:.3f}, "
        f"Train $R^2$: {result.r_squared_train:.3f}}} \\\\",
        f"\\multicolumn{{4}}{{l}}{{Non-zero coefficients: {result.n_nonzero}/{len(result.coefficients)}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Target: 30-day forward drawdown. 5-fold time series CV.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)
