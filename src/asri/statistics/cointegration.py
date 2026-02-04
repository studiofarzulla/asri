"""
Cointegration Testing for ASRI

If ASRI sub-indices share a common stochastic trend (are cointegrated),
we can exploit this for better forecasting via Vector Error Correction Models (VECM).

Cointegration implies a long-run equilibrium relationship that the indices
revert to after short-term deviations—potentially useful for regime detection.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CointegrationResult:
    """Results from cointegration testing."""
    test_type: Literal["johansen", "engle_granger"]
    
    # Number of cointegrating relationships found
    n_cointegrating: int
    
    # Test statistics and critical values
    trace_statistics: list[float] | None = None
    trace_critical_values: list[dict[str, float]] | None = None
    
    eigenvalue_statistics: list[float] | None = None
    eigenvalue_critical_values: list[dict[str, float]] | None = None
    
    # For Engle-Granger (2-variable case)
    eg_statistic: float | None = None
    eg_pvalue: float | None = None
    eg_critical_values: dict[str, float] | None = None
    
    # Cointegrating vectors (if found)
    cointegrating_vectors: np.ndarray | None = None
    
    # Interpretation
    conclusion: str = ""


def engle_granger_test(
    y1: pd.Series | np.ndarray,
    y2: pd.Series | np.ndarray,
    trend: Literal["c", "ct", "n"] = "c",
) -> CointegrationResult:
    """
    Engle-Granger two-step cointegration test for two variables.
    
    Step 1: Regress y1 on y2, get residuals
    Step 2: Test residuals for unit root (ADF test)
    
    H0: No cointegration (residuals have unit root)
    H1: Cointegration exists (residuals are stationary)
    
    Args:
        y1: First time series
        y2: Second time series  
        trend: Trend specification ('c' = constant, 'ct' = constant+trend, 'n' = none)
        
    Returns:
        CointegrationResult
    """
    # Convert to numpy
    if isinstance(y1, pd.Series):
        y1 = y1.dropna().values
    if isinstance(y2, pd.Series):
        y2 = y2.dropna().values
    
    # Align lengths
    n = min(len(y1), len(y2))
    y1 = y1[:n]
    y2 = y2[:n]
    
    # Step 1: Cointegrating regression
    if trend == "c":
        X = np.column_stack([np.ones(n), y2])
    elif trend == "ct":
        X = np.column_stack([np.ones(n), np.arange(n), y2])
    else:
        X = y2.reshape(-1, 1)
    
    beta = np.linalg.lstsq(X, y1, rcond=None)[0]
    residuals = y1 - X @ beta
    
    # Step 2: ADF test on residuals
    # Note: Critical values differ from standard ADF due to generated regressor
    from .stationarity import _adf_test
    adf_stat, _, _, _ = _adf_test(residuals)
    
    # Engle-Granger critical values (Phillips-Ouliaris, 2 variables)
    # These are more stringent than standard ADF
    critical_values = {
        "1%": -3.90,
        "5%": -3.34,
        "10%": -3.04,
    }
    
    # Determine p-value (approximate)
    if adf_stat < critical_values["1%"]:
        p_value = 0.005
    elif adf_stat < critical_values["5%"]:
        p_value = 0.025
    elif adf_stat < critical_values["10%"]:
        p_value = 0.075
    else:
        p_value = 0.15 + 0.05 * (adf_stat + 3.04)
    
    p_value = max(0.001, min(0.999, p_value))
    
    # Conclusion
    if p_value < 0.05:
        conclusion = "Evidence of cointegration at 5% level"
        n_coint = 1
        coint_vec = beta
    else:
        conclusion = "No evidence of cointegration"
        n_coint = 0
        coint_vec = None
    
    return CointegrationResult(
        test_type="engle_granger",
        n_cointegrating=n_coint,
        eg_statistic=adf_stat,
        eg_pvalue=p_value,
        eg_critical_values=critical_values,
        cointegrating_vectors=coint_vec,
        conclusion=conclusion,
    )


def johansen_test(
    data: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> CointegrationResult:
    """
    Johansen cointegration test for multiple time series.
    
    Tests for the number of cointegrating relationships among k variables.
    Uses both trace and maximum eigenvalue statistics.
    
    Args:
        data: DataFrame with k time series in columns
        det_order: Deterministic trend order (-1=no const, 0=const, 1=trend)
        k_ar_diff: Number of lagged differences in the VECM
        
    Returns:
        CointegrationResult with trace and eigenvalue statistics
    """
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        result = coint_johansen(data.values, det_order=det_order, k_ar_diff=k_ar_diff)
        
        # Extract results
        trace_stats = result.lr1.tolist()
        trace_cvs = [
            {"10%": result.cvt[i, 0], "5%": result.cvt[i, 1], "1%": result.cvt[i, 2]}
            for i in range(len(trace_stats))
        ]
        
        eigen_stats = result.lr2.tolist()
        eigen_cvs = [
            {"10%": result.cvm[i, 0], "5%": result.cvm[i, 1], "1%": result.cvm[i, 2]}
            for i in range(len(eigen_stats))
        ]
        
        # Count cointegrating relationships (trace test at 5%)
        n_coint = sum(1 for i, stat in enumerate(trace_stats) if stat > trace_cvs[i]["5%"])
        
        if n_coint > 0:
            conclusion = f"Found {n_coint} cointegrating relationship(s) at 5% level"
            coint_vecs = result.evec[:, :n_coint]
        else:
            conclusion = "No cointegrating relationships found"
            coint_vecs = None
        
        return CointegrationResult(
            test_type="johansen",
            n_cointegrating=n_coint,
            trace_statistics=trace_stats,
            trace_critical_values=trace_cvs,
            eigenvalue_statistics=eigen_stats,
            eigenvalue_critical_values=eigen_cvs,
            cointegrating_vectors=coint_vecs,
            conclusion=conclusion,
        )
        
    except ImportError:
        # Fallback without statsmodels
        return _johansen_fallback(data, det_order, k_ar_diff)


def _johansen_fallback(
    data: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> CointegrationResult:
    """Simplified Johansen test without statsmodels."""
    n, k = data.shape
    
    if n < 50:
        return CointegrationResult(
            test_type="johansen",
            n_cointegrating=0,
            conclusion="Insufficient observations for Johansen test",
        )
    
    # First difference
    diff_data = np.diff(data.values, axis=0)
    
    # Lagged levels
    lagged_levels = data.values[:-1]
    
    # Simple regression: ΔY_t on Y_{t-1}
    # Get eigenvalues of Π = αβ'
    
    # Reduced rank regression (simplified)
    Y = diff_data
    X = lagged_levels
    
    # OLS estimate of Π
    Pi = np.linalg.lstsq(X, Y, rcond=None)[0].T
    
    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvals(Pi @ Pi.T)
    eigenvalues = np.sort(np.real(eigenvalues))[::-1]
    
    # Trace statistic (simplified)
    trace_stats = []
    for r in range(k):
        trace_stat = -n * np.sum(np.log(1 - eigenvalues[r:]))
        trace_stats.append(trace_stat)
    
    # Approximate critical values (k=4 variables)
    trace_cvs_approx = [
        {"10%": 44.5, "5%": 47.9, "1%": 54.7},
        {"10%": 27.1, "5%": 29.8, "1%": 35.5},
        {"10%": 13.4, "5%": 15.5, "1%": 19.9},
        {"10%": 2.7, "5%": 3.8, "1%": 6.6},
    ][:k]
    
    n_coint = sum(1 for i, stat in enumerate(trace_stats) if i < len(trace_cvs_approx) and stat > trace_cvs_approx[i]["5%"])
    
    return CointegrationResult(
        test_type="johansen",
        n_cointegrating=n_coint,
        trace_statistics=trace_stats,
        trace_critical_values=trace_cvs_approx,
        conclusion=f"Found {n_coint} cointegrating relationship(s) (approximate test)",
    )


def format_cointegration_table(result: CointegrationResult) -> str:
    """Format cointegration results as a LaTeX table."""
    if result.test_type == "johansen" and result.trace_statistics:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Johansen Cointegration Test Results}",
            r"\label{tab:cointegration}",
            r"\small",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"$H_0$: $r \leq$ & Trace Stat & 10\% CV & 5\% CV & 1\% CV \\",
            r"\midrule",
        ]
        
        for i, (stat, cvs) in enumerate(zip(result.trace_statistics, result.trace_critical_values)):
            sig = "*" if stat > cvs["10%"] else ""
            sig = "**" if stat > cvs["5%"] else sig
            sig = "***" if stat > cvs["1%"] else sig
            lines.append(f"{i} & {stat:.2f}{sig} & {cvs['10%']:.2f} & {cvs['5%']:.2f} & {cvs['1%']:.2f} \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            f"\\item Conclusion: {result.conclusion}",
            r"\item *** $p<0.01$, ** $p<0.05$, * $p<0.10$",
            r"\end{tablenotes}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    elif result.test_type == "engle_granger":
        return f"""
\\textbf{{Engle-Granger Cointegration Test}}

Test statistic: {result.eg_statistic:.3f}

$p$-value: {result.eg_pvalue:.3f}

Critical values: 1\\%: {result.eg_critical_values['1%']:.2f}, 
5\\%: {result.eg_critical_values['5%']:.2f}, 
10\\%: {result.eg_critical_values['10%']:.2f}

\\textbf{{Conclusion:}} {result.conclusion}
"""
    
    return ""
