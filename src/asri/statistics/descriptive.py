"""
Descriptive Statistics for ASRI

Publication-ready summary statistics and correlation analysis.
Every empirical paper needs Table 1 with descriptive stats.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DescriptiveStats:
    """Descriptive statistics for a single variable."""
    name: str
    n: int
    mean: float
    std: float
    min: float
    q25: float
    median: float
    q75: float
    max: float
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    
    @property
    def is_normal(self) -> bool:
        """Test if distribution is approximately normal (JB test)."""
        return self.jarque_bera_pvalue > 0.05


def compute_descriptive_stats(
    series: pd.Series | np.ndarray,
    name: str = "variable",
) -> DescriptiveStats:
    """
    Compute comprehensive descriptive statistics.
    
    Args:
        series: Data to summarize
        name: Variable name for reporting
        
    Returns:
        DescriptiveStats with all summary measures
    """
    if isinstance(series, pd.Series):
        data = series.dropna().values
    else:
        data = series[~np.isnan(series)]
    
    n = len(data)
    
    if n < 3:
        raise ValueError(f"Need at least 3 observations, got {n}")
    
    # Basic stats
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    # Quantiles
    q25, median, q75 = np.percentile(data, [25, 50, 75])
    
    # Higher moments
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)  # Excess kurtosis (normal = 0)
    
    # Jarque-Bera test for normality
    jb_stat, jb_pval = stats.jarque_bera(data)
    
    return DescriptiveStats(
        name=name,
        n=n,
        mean=mean,
        std=std,
        min=np.min(data),
        q25=q25,
        median=median,
        q75=q75,
        max=np.max(data),
        skewness=skewness,
        kurtosis=kurtosis,
        jarque_bera_stat=jb_stat,
        jarque_bera_pvalue=jb_pval,
    )


def compute_descriptive_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for all columns.
    
    Args:
        data: DataFrame with variables in columns
        
    Returns:
        DataFrame with statistics as rows, variables as columns
    """
    stats_list = []
    
    for col in data.columns:
        try:
            s = compute_descriptive_stats(data[col], name=col)
            stats_list.append({
                'Variable': col,
                'N': s.n,
                'Mean': s.mean,
                'Std': s.std,
                'Min': s.min,
                'Q25': s.q25,
                'Median': s.median,
                'Q75': s.q75,
                'Max': s.max,
                'Skewness': s.skewness,
                'Kurtosis': s.kurtosis,
                'JB Stat': s.jarque_bera_stat,
                'JB p-val': s.jarque_bera_pvalue,
            })
        except ValueError:
            continue
    
    return pd.DataFrame(stats_list)


@dataclass
class CorrelationResult:
    """Correlation with significance test."""
    var1: str
    var2: str
    correlation: float
    p_value: float
    n: int
    method: Literal["pearson", "spearman", "kendall"]
    
    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05
    
    @property
    def stars(self) -> str:
        if self.p_value < 0.01:
            return "***"
        elif self.p_value < 0.05:
            return "**"
        elif self.p_value < 0.10:
            return "*"
        return ""


def correlation_with_significance(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> CorrelationResult:
    """
    Compute correlation with significance test.
    
    Args:
        x: First variable
        y: Second variable
        method: Correlation method
        
    Returns:
        CorrelationResult with correlation and p-value
    """
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if method == "pearson":
        corr, pval = stats.pearsonr(x, y)
    elif method == "spearman":
        corr, pval = stats.spearmanr(x, y)
    elif method == "kendall":
        corr, pval = stats.kendalltau(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return CorrelationResult(
        var1="x",
        var2="y",
        correlation=corr,
        p_value=pval,
        n=n,
        method=method,
    )


def correlation_matrix_with_significance(
    data: pd.DataFrame,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute correlation matrix with significance p-values.
    
    Args:
        data: DataFrame with variables in columns
        method: Correlation method
        
    Returns:
        Tuple of (correlation_matrix, p_value_matrix)
    """
    columns = data.columns
    n = len(columns)
    
    corr_matrix = np.zeros((n, n))
    pval_matrix = np.zeros((n, n))
    
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            elif i < j:
                result = correlation_with_significance(
                    data[col1], data[col2], method=method
                )
                corr_matrix[i, j] = result.correlation
                corr_matrix[j, i] = result.correlation
                pval_matrix[i, j] = result.p_value
                pval_matrix[j, i] = result.p_value
    
    return (
        pd.DataFrame(corr_matrix, index=columns, columns=columns),
        pd.DataFrame(pval_matrix, index=columns, columns=columns),
    )


def format_descriptive_table(stats_df: pd.DataFrame) -> str:
    """Format descriptive statistics as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Descriptive Statistics}",
        r"\label{tab:descriptive}",
        r"\small",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Variable & N & Mean & Std & Min & Median & Max & Skew & Kurt \\",
        r"\midrule",
    ]
    
    for _, row in stats_df.iterrows():
        lines.append(
            f"{row['Variable']} & {row['N']:.0f} & {row['Mean']:.2f} & "
            f"{row['Std']:.2f} & {row['Min']:.2f} & {row['Median']:.2f} & "
            f"{row['Max']:.2f} & {row['Skewness']:.2f} & {row['Kurtosis']:.2f} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Notes: Skew = skewness, Kurt = excess kurtosis (normal = 0).",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def format_correlation_table(
    corr_matrix: pd.DataFrame,
    pval_matrix: pd.DataFrame,
) -> str:
    """Format correlation matrix as LaTeX table with significance stars."""
    columns = corr_matrix.columns
    n = len(columns)
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Correlation Matrix}",
        r"\label{tab:correlation}",
        r"\small",
        r"\begin{tabular}{l" + "c" * n + "}",
        r"\toprule",
        r" & " + " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    
    for i, row_name in enumerate(columns):
        row_values = []
        for j, col_name in enumerate(columns):
            corr = corr_matrix.iloc[i, j]
            pval = pval_matrix.iloc[i, j]
            
            if i == j:
                row_values.append("1.00")
            else:
                stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
                row_values.append(f"{corr:.2f}{stars}")
        
        lines.append(f"{row_name} & " + " & ".join(row_values) + r" \\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item *** $p<0.01$, ** $p<0.05$, * $p<0.10$",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def compute_rolling_stats(
    series: pd.Series,
    window: int = 30,
) -> pd.DataFrame:
    """
    Compute rolling descriptive statistics.
    
    Useful for detecting regime changes and non-stationarity.
    
    Args:
        series: Time series
        window: Rolling window size
        
    Returns:
        DataFrame with rolling mean, std, skewness, kurtosis
    """
    rolling = series.rolling(window=window)
    
    return pd.DataFrame({
        'mean': rolling.mean(),
        'std': rolling.std(),
        'skewness': rolling.skew(),
        'kurtosis': rolling.kurt(),
        'min': rolling.min(),
        'max': rolling.max(),
    })
