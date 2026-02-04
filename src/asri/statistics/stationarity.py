"""
Stationarity Testing for ASRI Time Series

Implements Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS)
tests to verify time series stationarity—a prerequisite for valid statistical inference.

Key insight: ADF and KPSS have opposite null hypotheses, so we use both for robust
inference. A series is considered stationary only if ADF rejects AND KPSS fails to reject.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


class StationarityConclusion(Enum):
    """Conclusion from joint ADF/KPSS testing."""
    STATIONARY = "stationary"
    NON_STATIONARY = "non_stationary"
    TREND_STATIONARY = "trend_stationary"
    INCONCLUSIVE = "inconclusive"


@dataclass
class StationarityResult:
    """Results from stationarity testing."""
    series_name: str
    
    # ADF results (H0: unit root exists, i.e., non-stationary)
    adf_statistic: float
    adf_pvalue: float
    adf_critical_values: dict[str, float]
    adf_rejects_null: bool  # True = evidence of stationarity
    
    # KPSS results (H0: series is stationary)
    kpss_statistic: float
    kpss_pvalue: float
    kpss_critical_values: dict[str, float]
    kpss_rejects_null: bool  # True = evidence of non-stationarity
    
    # Joint conclusion
    conclusion: StationarityConclusion
    recommended_transformation: str | None
    
    # Metadata
    n_observations: int
    n_lags_used: int


def _adf_test(series: np.ndarray, max_lag: int | None = None) -> tuple[float, float, dict, int]:
    """
    Augmented Dickey-Fuller test for unit root.
    
    H0: Series has a unit root (non-stationary)
    H1: Series is stationary
    
    We reject H0 (conclude stationarity) if p-value < alpha.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series, maxlag=max_lag, autolag='AIC')
        return (
            result[0],  # test statistic
            result[1],  # p-value
            result[4],  # critical values dict
            result[2],  # lags used
        )
    except ImportError:
        # Fallback: simplified ADF using OLS
        return _adf_fallback(series)


def _adf_fallback(series: np.ndarray) -> tuple[float, float, dict, int]:
    """Simplified ADF test when statsmodels not available."""
    n = len(series)
    
    # First difference
    diff = np.diff(series)
    lagged = series[:-1]
    
    # Regress diff on lagged level
    X = np.column_stack([np.ones(n-1), lagged])
    y = diff
    
    # OLS: beta = (X'X)^-1 X'y
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    
    # Residuals and standard error
    residuals = y - X @ beta
    sigma2 = np.sum(residuals**2) / (n - 3)
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    
    # t-statistic for rho - 1 (coefficient on lagged level)
    t_stat = beta[1] / se[1]
    
    # Critical values (approximate for n > 100)
    critical_values = {
        '1%': -3.43,
        '5%': -2.86,
        '10%': -2.57,
    }
    
    # Approximate p-value using interpolation
    if t_stat < -3.43:
        p_value = 0.005
    elif t_stat < -2.86:
        p_value = 0.025
    elif t_stat < -2.57:
        p_value = 0.075
    else:
        p_value = 0.15 + 0.05 * (t_stat + 2.57)
    
    p_value = max(0.001, min(0.999, p_value))
    
    return t_stat, p_value, critical_values, 1


def _kpss_test(series: np.ndarray, regression: Literal['c', 'ct'] = 'c') -> tuple[float, float, dict]:
    """
    KPSS test for stationarity.
    
    H0: Series is stationary (around a constant or trend)
    H1: Series has a unit root (non-stationary)
    
    We reject H0 (conclude non-stationarity) if p-value < alpha.
    
    Args:
        series: Time series data
        regression: 'c' for constant (level stationarity), 'ct' for constant+trend
    """
    try:
        from statsmodels.tsa.stattools import kpss
        # kpss returns: statistic, p-value, lags, critical_values
        result = kpss(series, regression=regression, nlags='auto')
        return result[0], result[1], result[3]
    except ImportError:
        return _kpss_fallback(series, regression)


def _kpss_fallback(series: np.ndarray, regression: str = 'c') -> tuple[float, float, dict]:
    """Simplified KPSS test when statsmodels not available."""
    n = len(series)
    
    # Demean or detrend
    if regression == 'c':
        residuals = series - np.mean(series)
    else:  # ct
        t = np.arange(n)
        X = np.column_stack([np.ones(n), t])
        beta = np.linalg.lstsq(X, series, rcond=None)[0]
        residuals = series - X @ beta
    
    # Cumulative sum of residuals
    S = np.cumsum(residuals)
    
    # Long-run variance estimate (Newey-West with automatic lag selection)
    lag = int(4 * (n / 100) ** 0.25)
    
    # Autocovariance
    gamma0 = np.sum(residuals ** 2) / n
    gamma = np.array([np.sum(residuals[j:] * residuals[:-j]) / n for j in range(1, lag + 1)])
    
    # Bartlett weights
    weights = 1 - np.arange(1, lag + 1) / (lag + 1)
    
    # Long-run variance
    s2 = gamma0 + 2 * np.sum(weights * gamma)
    
    # KPSS statistic
    kpss_stat = np.sum(S ** 2) / (n ** 2 * s2)
    
    # Critical values for level stationarity
    if regression == 'c':
        critical_values = {
            '10%': 0.347,
            '5%': 0.463,
            '2.5%': 0.574,
            '1%': 0.739,
        }
    else:  # ct
        critical_values = {
            '10%': 0.119,
            '5%': 0.146,
            '2.5%': 0.176,
            '1%': 0.216,
        }
    
    # Approximate p-value
    if kpss_stat < critical_values['10%']:
        p_value = 0.15
    elif kpss_stat < critical_values['5%']:
        p_value = 0.075
    elif kpss_stat < critical_values['1%']:
        p_value = 0.025
    else:
        p_value = 0.005
    
    return kpss_stat, p_value, critical_values


def test_stationarity(
    series: pd.Series | np.ndarray,
    name: str = "series",
    alpha: float = 0.05,
) -> StationarityResult:
    """
    Comprehensive stationarity test using both ADF and KPSS.
    
    Decision matrix:
    - ADF rejects, KPSS fails to reject → STATIONARY
    - ADF fails to reject, KPSS rejects → NON_STATIONARY  
    - ADF rejects, KPSS rejects → TREND_STATIONARY (difference-stationary)
    - ADF fails, KPSS fails → INCONCLUSIVE
    
    Args:
        series: Time series to test
        name: Name for reporting
        alpha: Significance level (default 0.05)
        
    Returns:
        StationarityResult with test statistics and conclusion
    """
    if isinstance(series, pd.Series):
        arr = series.dropna().values
    else:
        arr = series[~np.isnan(series)]
    
    n = len(arr)
    if n < 20:
        raise ValueError(f"Series '{name}' has only {n} observations; need at least 20")
    
    # Run ADF test
    adf_stat, adf_pval, adf_crit, n_lags = _adf_test(arr)
    adf_rejects = adf_pval < alpha
    
    # Run KPSS test
    kpss_stat, kpss_pval, kpss_crit = _kpss_test(arr, regression='c')
    kpss_rejects = kpss_pval < alpha
    
    # Determine conclusion
    if adf_rejects and not kpss_rejects:
        conclusion = StationarityConclusion.STATIONARY
        transformation = None
    elif not adf_rejects and kpss_rejects:
        conclusion = StationarityConclusion.NON_STATIONARY
        transformation = "first_difference"
    elif adf_rejects and kpss_rejects:
        conclusion = StationarityConclusion.TREND_STATIONARY
        transformation = "detrend"
    else:
        conclusion = StationarityConclusion.INCONCLUSIVE
        transformation = "first_difference"  # Conservative
    
    return StationarityResult(
        series_name=name,
        adf_statistic=adf_stat,
        adf_pvalue=adf_pval,
        adf_critical_values=adf_crit,
        adf_rejects_null=adf_rejects,
        kpss_statistic=kpss_stat,
        kpss_pvalue=kpss_pval,
        kpss_critical_values=kpss_crit,
        kpss_rejects_null=kpss_rejects,
        conclusion=conclusion,
        recommended_transformation=transformation,
        n_observations=n,
        n_lags_used=n_lags,
    )


def test_stationarity_suite(
    data: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, StationarityResult]:
    """
    Run stationarity tests on all columns of a DataFrame.
    
    Args:
        data: DataFrame with time series in columns
        alpha: Significance level
        
    Returns:
        Dictionary mapping column names to StationarityResult
    """
    results = {}
    for col in data.columns:
        try:
            results[col] = test_stationarity(data[col], name=col, alpha=alpha)
        except ValueError as e:
            # Skip columns with insufficient data
            results[col] = None
    return results


def recommend_transformation(result: StationarityResult) -> pd.Series:
    """
    Apply the recommended transformation to make series stationary.
    
    Args:
        result: StationarityResult from test_stationarity
        
    Returns:
        Transformed series
    """
    # This would be called with the original series
    # For now, just return the recommendation
    pass


def apply_transformation(
    series: pd.Series,
    transformation: Literal["first_difference", "log_difference", "detrend", "none"],
) -> pd.Series:
    """
    Apply a transformation to make a series stationary.
    
    Args:
        series: Original time series
        transformation: Type of transformation to apply
        
    Returns:
        Transformed series
    """
    if transformation == "none":
        return series
    elif transformation == "first_difference":
        return series.diff().dropna()
    elif transformation == "log_difference":
        return np.log(series).diff().dropna()
    elif transformation == "detrend":
        t = np.arange(len(series))
        slope, intercept = np.polyfit(t, series.values, 1)
        trend = intercept + slope * t
        return pd.Series(series.values - trend, index=series.index, name=series.name)
    else:
        raise ValueError(f"Unknown transformation: {transformation}")


def format_stationarity_table(results: dict[str, StationarityResult]) -> str:
    """
    Format stationarity results as a LaTeX table.
    
    Returns LaTeX code for a publication-ready table.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Unit Root and Stationarity Tests}",
        r"\label{tab:stationarity}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Variable & ADF Stat & ADF $p$ & KPSS Stat & KPSS $p$ & Conclusion \\",
        r"\midrule",
    ]
    
    for name, result in results.items():
        if result is None:
            continue
            
        # Significance stars
        adf_stars = "***" if result.adf_pvalue < 0.01 else ("**" if result.adf_pvalue < 0.05 else ("*" if result.adf_pvalue < 0.10 else ""))
        kpss_stars = "***" if result.kpss_pvalue < 0.01 else ("**" if result.kpss_pvalue < 0.05 else ("*" if result.kpss_pvalue < 0.10 else ""))
        
        conclusion_map = {
            StationarityConclusion.STATIONARY: "I(0)",
            StationarityConclusion.NON_STATIONARY: "I(1)",
            StationarityConclusion.TREND_STATIONARY: "TS",
            StationarityConclusion.INCONCLUSIVE: "?",
        }
        
        lines.append(
            f"{name} & {result.adf_statistic:.3f}{adf_stars} & {result.adf_pvalue:.3f} & "
            f"{result.kpss_statistic:.3f}{kpss_stars} & {result.kpss_pvalue:.3f} & "
            f"{conclusion_map[result.conclusion]} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Notes: ADF null hypothesis: unit root (non-stationary). KPSS null: stationary.",
        r"\item I(0) = stationary, I(1) = integrated order 1, TS = trend-stationary.",
        r"\item *** $p<0.01$, ** $p<0.05$, * $p<0.10$",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)
