"""
Granger Causality Testing for ASRI

Tests whether ASRI sub-indices have predictive power for future market stress.
This is the core statistical validation: if sub-indices don't Granger-cause
drawdowns, the entire ASRI framework is theoretically unsound.

Methodology:
1. For each sub-index, test if lagged values improve prediction of forward drawdowns
2. Use VAR framework to test multi-variate causality
3. Report F-statistics and p-values at multiple lags (1, 3, 7, 14, 30 days)
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class GrangerResult:
    """Results from a single Granger causality test."""
    cause_variable: str
    effect_variable: str
    lag: int
    f_statistic: float
    p_value: float
    is_significant: bool
    df_num: int  # Numerator degrees of freedom
    df_denom: int  # Denominator degrees of freedom
    
    # Model fit statistics
    r_squared_restricted: float
    r_squared_unrestricted: float
    
    def __str__(self) -> str:
        sig = "***" if self.p_value < 0.01 else ("**" if self.p_value < 0.05 else ("*" if self.p_value < 0.10 else ""))
        return (f"{self.cause_variable} → {self.effect_variable} (lag={self.lag}): "
                f"F={self.f_statistic:.3f}{sig}, p={self.p_value:.4f}")


@dataclass
class GrangerCausalityMatrix:
    """Full matrix of Granger causality results."""
    variables: list[str]
    lags_tested: list[int]
    results: dict[tuple[str, str, int], GrangerResult]  # (cause, effect, lag) -> result
    
    def get_result(self, cause: str, effect: str, lag: int) -> GrangerResult | None:
        return self.results.get((cause, effect, lag))
    
    def significant_at(self, alpha: float = 0.05) -> list[GrangerResult]:
        """Return all significant causal relationships."""
        return [r for r in self.results.values() if r.p_value < alpha]
    
    def best_lag(self, cause: str, effect: str) -> GrangerResult | None:
        """Find the lag with strongest causality (lowest p-value)."""
        relevant = [r for (c, e, _), r in self.results.items() if c == cause and e == effect]
        if not relevant:
            return None
        return min(relevant, key=lambda r: r.p_value)


def _create_lagged_matrix(
    y: np.ndarray,
    X: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create lagged matrices for Granger causality regression.
    
    Returns:
        y_trimmed: Dependent variable (trimmed to match lags)
        X_restricted: Lagged y only (for restricted model)
        X_unrestricted: Lagged y + lagged X (for unrestricted model)
    """
    n = len(y)
    T = n - lag  # Effective sample size
    
    if T < lag + 10:
        raise ValueError(f"Insufficient observations ({n}) for lag={lag}")
    
    # Dependent variable (from lag onwards)
    y_trimmed = y[lag:]
    
    # Build lagged matrices
    lagged_y = np.column_stack([y[lag-i-1:n-i-1] for i in range(lag)])
    lagged_x = np.column_stack([X[lag-i-1:n-i-1] for i in range(lag)])
    
    # Add constant
    const = np.ones((T, 1))
    
    X_restricted = np.hstack([const, lagged_y])
    X_unrestricted = np.hstack([const, lagged_y, lagged_x])
    
    return y_trimmed, X_restricted, X_unrestricted


def _ols_regression(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Simple OLS regression.
    
    Returns:
        coefficients, R-squared, residual sum of squares
    """
    n, k = X.shape
    
    # beta = (X'X)^-1 X'y
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse for near-singular matrices
        XtX_inv = np.linalg.pinv(X.T @ X)
    
    beta = XtX_inv @ X.T @ y
    
    # Fitted values and residuals
    y_hat = X @ beta
    residuals = y - y_hat
    
    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return beta, r_squared, ss_res


def granger_causality_test(
    cause: pd.Series | np.ndarray,
    effect: pd.Series | np.ndarray,
    lag: int,
    cause_name: str = "X",
    effect_name: str = "Y",
) -> GrangerResult:
    """
    Test if 'cause' Granger-causes 'effect' at the specified lag.
    
    H0: Lagged values of 'cause' do not improve prediction of 'effect'
    H1: Lagged values of 'cause' DO improve prediction of 'effect'
    
    Uses F-test comparing restricted (only lagged y) vs unrestricted (lagged y + lagged x) models.
    
    Args:
        cause: Potential causal variable (X)
        effect: Effect variable (Y)
        lag: Number of lags to include
        cause_name: Name for reporting
        effect_name: Name for reporting
        
    Returns:
        GrangerResult with F-statistic and p-value
    """
    # Convert to numpy
    if isinstance(cause, pd.Series):
        x = cause.dropna().values
    else:
        x = cause[~np.isnan(cause)]
    
    if isinstance(effect, pd.Series):
        y = effect.dropna().values
    else:
        y = effect[~np.isnan(effect)]
    
    # Align lengths
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    
    # Create lagged matrices
    y_trimmed, X_restricted, X_unrestricted = _create_lagged_matrix(y, x, lag)
    
    # Fit restricted model (only lagged y)
    _, r2_restricted, ss_restricted = _ols_regression(y_trimmed, X_restricted)
    
    # Fit unrestricted model (lagged y + lagged x)
    _, r2_unrestricted, ss_unrestricted = _ols_regression(y_trimmed, X_unrestricted)
    
    # F-test for nested models
    # F = [(SSR_r - SSR_u) / q] / [SSR_u / (n - k)]
    # where q = number of restrictions (lag), k = parameters in unrestricted model
    
    n = len(y_trimmed)
    q = lag  # Number of restrictions (lagged X coefficients = 0)
    k = X_unrestricted.shape[1]  # Parameters in unrestricted model
    
    if ss_unrestricted <= 0:
        # Perfect fit (shouldn't happen with real data)
        f_stat = 0.0
        p_value = 1.0
    else:
        f_stat = ((ss_restricted - ss_unrestricted) / q) / (ss_unrestricted / (n - k))
        f_stat = max(0, f_stat)  # F-stat can't be negative
        
        # P-value from F-distribution
        p_value = 1 - stats.f.cdf(f_stat, q, n - k)
    
    return GrangerResult(
        cause_variable=cause_name,
        effect_variable=effect_name,
        lag=lag,
        f_statistic=f_stat,
        p_value=p_value,
        is_significant=p_value < 0.05,
        df_num=q,
        df_denom=n - k,
        r_squared_restricted=r2_restricted,
        r_squared_unrestricted=r2_unrestricted,
    )


def granger_causality_matrix(
    data: pd.DataFrame,
    target_column: str | None = None,
    lags: list[int] = [1, 3, 7, 14, 30],
    alpha: float = 0.05,
) -> GrangerCausalityMatrix:
    """
    Compute full Granger causality matrix for all variable pairs.
    
    If target_column is specified, only test causality TO that target.
    Otherwise, test all pairwise combinations.
    
    Args:
        data: DataFrame with time series in columns
        target_column: If specified, only test X → target for all X
        lags: List of lags to test
        alpha: Significance level for reporting
        
    Returns:
        GrangerCausalityMatrix with all results
    """
    columns = list(data.columns)
    results: dict[tuple[str, str, int], GrangerResult] = {}
    
    if target_column:
        # Only test causality TO the target
        effects = [target_column]
        causes = [c for c in columns if c != target_column]
    else:
        # Test all pairs
        effects = columns
        causes = columns
    
    for effect in effects:
        for cause in causes:
            if cause == effect:
                continue
                
            for lag in lags:
                try:
                    result = granger_causality_test(
                        cause=data[cause],
                        effect=data[effect],
                        lag=lag,
                        cause_name=cause,
                        effect_name=effect,
                    )
                    results[(cause, effect, lag)] = result
                except ValueError:
                    # Insufficient data for this lag
                    continue
    
    return GrangerCausalityMatrix(
        variables=columns,
        lags_tested=lags,
        results=results,
    )


def var_lag_selection(
    data: pd.DataFrame,
    max_lag: int = 30,
    criterion: Literal["aic", "bic", "hqic"] = "aic",
) -> int:
    """
    Select optimal VAR lag order using information criteria.
    
    Args:
        data: DataFrame with time series in columns
        max_lag: Maximum lag to consider
        criterion: Information criterion ('aic', 'bic', 'hqic')
        
    Returns:
        Optimal lag order
    """
    n, k = data.shape
    
    best_lag = 1
    best_ic = np.inf
    
    for lag in range(1, min(max_lag + 1, n // (k + 2))):
        try:
            # Estimate VAR(lag) and compute information criterion
            ic = _compute_var_ic(data.values, lag, criterion)
            if ic < best_ic:
                best_ic = ic
                best_lag = lag
        except (ValueError, np.linalg.LinAlgError):
            continue
    
    return best_lag


def _compute_var_ic(
    data: np.ndarray,
    lag: int,
    criterion: str,
) -> float:
    """Compute information criterion for VAR(lag) model."""
    n, k = data.shape
    T = n - lag  # Effective sample size
    
    if T < k * lag + 10:
        raise ValueError("Insufficient observations")
    
    # Build VAR regression matrices
    Y = data[lag:]  # T x k
    
    # Lagged regressors: [const, Y_{t-1}, Y_{t-2}, ..., Y_{t-lag}]
    X = np.ones((T, 1))
    for i in range(1, lag + 1):
        X = np.hstack([X, data[lag-i:n-i]])
    
    # OLS for each equation
    total_ssr = 0
    for j in range(k):
        y = Y[:, j]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        total_ssr += np.sum(residuals ** 2)
    
    # Log-likelihood (proportional)
    log_det_sigma = np.log(total_ssr / T)
    
    # Number of parameters
    num_params = k * (1 + k * lag)
    
    # Information criteria
    if criterion == "aic":
        ic = log_det_sigma + 2 * num_params / T
    elif criterion == "bic":
        ic = log_det_sigma + np.log(T) * num_params / T
    else:  # hqic
        ic = log_det_sigma + 2 * np.log(np.log(T)) * num_params / T
    
    return ic


def format_granger_table(
    matrix: GrangerCausalityMatrix,
    target: str,
    alpha: float = 0.05,
) -> str:
    """
    Format Granger causality results as a LaTeX table.
    
    Shows F-statistics and p-values for each sub-index → target at all lags.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Granger Causality Tests: Sub-Indices $\rightarrow$ " + target + "}",
        r"\label{tab:granger}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(matrix.lags_tested) + "}",
        r"\toprule",
        r"Variable & " + " & ".join([f"Lag {lag}" for lag in matrix.lags_tested]) + r" \\",
        r"\midrule",
    ]
    
    causes = sorted(set(r.cause_variable for r in matrix.results.values() if r.effect_variable == target))
    
    for cause in causes:
        row_values = []
        for lag in matrix.lags_tested:
            result = matrix.get_result(cause, target, lag)
            if result:
                stars = "***" if result.p_value < 0.01 else ("**" if result.p_value < 0.05 else ("*" if result.p_value < 0.10 else ""))
                row_values.append(f"{result.f_statistic:.2f}{stars}")
            else:
                row_values.append("--")
        
        lines.append(f"{cause} & " + " & ".join(row_values) + r" \\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Notes: F-statistics from Granger causality tests. H$_0$: lagged sub-index does not improve prediction.",
        r"\item *** $p<0.01$, ** $p<0.05$, * $p<0.10$",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def compute_predictive_weights(
    matrix: GrangerCausalityMatrix,
    target: str,
    lag: int | None = None,
) -> dict[str, float]:
    """
    Derive sub-index weights from Granger causality F-statistics.
    
    Logic: Variables with higher F-statistics (stronger predictive power)
    should receive higher weights in the aggregate index.
    
    Args:
        matrix: GrangerCausalityMatrix from granger_causality_matrix()
        target: The target variable being predicted
        lag: Specific lag to use (or None for best lag per variable)
        
    Returns:
        Dictionary mapping variable names to weights (sum to 1.0)
    """
    weights = {}
    
    causes = sorted(set(r.cause_variable for r in matrix.results.values() if r.effect_variable == target))
    
    for cause in causes:
        if lag is not None:
            result = matrix.get_result(cause, target, lag)
        else:
            result = matrix.best_lag(cause, target)
        
        if result and result.p_value < 0.10:  # Only include marginally significant
            # Use F-statistic as importance measure
            weights[cause] = result.f_statistic
        else:
            weights[cause] = 0.0
    
    # Normalize to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        # If nothing is significant, use equal weights
        n = len(causes)
        weights = {k: 1.0 / n for k in causes}
    
    return weights
