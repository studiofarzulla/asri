"""
PCA-Based Weight Derivation for ASRI

Principal Component Analysis extracts the dominant pattern of co-movement
across sub-indices. The first principal component loadings naturally
indicate each variable's contribution to the common "risk factor."

Theoretical justification:
- If sub-indices capture correlated aspects of systemic risk,
  PC1 represents the latent "true" risk factor
- Loadings on PC1 indicate each sub-index's sensitivity to this factor
- Using PC1 loadings as weights creates an optimally weighted composite
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    """Results from PCA analysis."""
    n_components: int
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    loadings: pd.DataFrame  # Components x Variables
    scores: np.ndarray  # Observations x Components
    
    # Derived weights (normalized PC1 loadings)
    weights: dict[str, float]
    
    # Quality metrics
    kmo_statistic: float | None  # Kaiser-Meyer-Olkin measure
    bartlett_pvalue: float | None  # Bartlett's test for sphericity


class PCAWeightDeriver:
    """
    Derive ASRI weights from Principal Component Analysis.
    
    The key insight is that PC1 loadings represent each variable's
    contribution to the dominant source of variance—which, if our
    sub-indices are well-constructed, should be systemic risk.
    """
    
    def __init__(
        self,
        n_components: int | None = None,
        standardize: bool = True,
    ):
        """
        Args:
            n_components: Number of PCs to extract (None = all)
            standardize: Whether to standardize variables before PCA
        """
        self.n_components = n_components
        self.standardize = standardize
        self._result: PCAResult | None = None
    
    def fit(self, data: pd.DataFrame) -> "PCAWeightDeriver":
        """
        Fit PCA and derive weights from loadings.
        
        Args:
            data: DataFrame with sub-indices as columns
            
        Returns:
            Self (for method chaining)
        """
        # Remove NaN rows
        clean_data = data.dropna()
        n, k = clean_data.shape
        
        if n < k + 10:
            raise ValueError(f"Insufficient observations ({n}) for {k} variables")
        
        # Standardize if requested
        if self.standardize:
            means = clean_data.mean()
            stds = clean_data.std()
            X = ((clean_data - means) / stds).values
        else:
            X = clean_data.values
        
        # Compute covariance matrix
        cov_matrix = np.cov(X.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Number of components
        n_comp = self.n_components or k
        
        # Explained variance ratio
        total_var = np.sum(eigenvalues)
        explained_var = eigenvalues[:n_comp] / total_var
        cumulative_var = np.cumsum(explained_var)
        
        # Loadings (correlation between variables and components)
        # For standardized data, loadings = eigenvector * sqrt(eigenvalue)
        loadings = eigenvectors[:, :n_comp] * np.sqrt(eigenvalues[:n_comp])
        loadings_df = pd.DataFrame(
            loadings.T,
            index=[f"PC{i+1}" for i in range(n_comp)],
            columns=clean_data.columns,
        )
        
        # Scores (observations projected onto PCs)
        scores = X @ eigenvectors[:, :n_comp]
        
        # Derive weights from PC1 loadings
        pc1_loadings = loadings[:, 0]
        
        # Take absolute value (direction doesn't matter for risk)
        # and normalize to sum to 1
        abs_loadings = np.abs(pc1_loadings)
        weights_arr = abs_loadings / np.sum(abs_loadings)
        
        weights = {
            col: w for col, w in zip(clean_data.columns, weights_arr)
        }
        
        # Compute quality metrics
        kmo = self._compute_kmo(cov_matrix)
        bartlett_p = self._bartlett_test(X, n)
        
        self._result = PCAResult(
            n_components=n_comp,
            explained_variance_ratio=explained_var,
            cumulative_variance_ratio=cumulative_var,
            loadings=loadings_df,
            scores=scores,
            weights=weights,
            kmo_statistic=kmo,
            bartlett_pvalue=bartlett_p,
        )
        
        return self
    
    def _compute_kmo(self, cov_matrix: np.ndarray) -> float:
        """
        Compute Kaiser-Meyer-Olkin measure of sampling adequacy.
        
        KMO measures whether variables have enough correlation to
        justify PCA. Values > 0.6 are acceptable, > 0.8 are good.
        """
        try:
            # Correlation matrix from covariance
            d = np.sqrt(np.diag(cov_matrix))
            corr = cov_matrix / np.outer(d, d)
            
            # Partial correlations (anti-image correlations)
            inv_corr = np.linalg.inv(corr)
            d_inv = np.sqrt(np.diag(inv_corr))
            partial_corr = -inv_corr / np.outer(d_inv, d_inv)
            np.fill_diagonal(partial_corr, 0)
            
            # KMO
            r_sq = np.sum(corr ** 2) - np.trace(corr ** 2)
            pr_sq = np.sum(partial_corr ** 2)
            
            kmo = r_sq / (r_sq + pr_sq)
            return kmo
            
        except np.linalg.LinAlgError:
            return None
    
    def _bartlett_test(self, X: np.ndarray, n: int) -> float:
        """
        Bartlett's test of sphericity.
        
        H0: Correlation matrix is identity (no correlation)
        If p < 0.05, variables are correlated enough for PCA.
        """
        from scipy import stats
        
        k = X.shape[1]
        corr = np.corrcoef(X.T)
        
        # Test statistic
        det = np.linalg.det(corr)
        if det <= 0:
            return 0.0  # Singular, definitely not spherical
        
        chi2 = -((n - 1) - (2 * k + 5) / 6) * np.log(det)
        df = k * (k - 1) / 2
        
        p_value = 1 - stats.chi2.cdf(chi2, df)
        return p_value
    
    @property
    def result(self) -> PCAResult:
        if self._result is None:
            raise ValueError("Must call fit() first")
        return self._result
    
    @property
    def weights(self) -> dict[str, float]:
        return self.result.weights
    
    @property
    def explained_variance_pc1(self) -> float:
        """Proportion of variance explained by first PC."""
        return self.result.explained_variance_ratio[0]
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Project new data onto principal components."""
        if self._result is None:
            raise ValueError("Must call fit() first")
        
        clean_data = data.dropna()
        if self.standardize:
            means = clean_data.mean()
            stds = clean_data.std()
            X = ((clean_data - means) / stds).values
        else:
            X = clean_data.values
        
        # Use stored eigenvectors
        return X @ self._result.loadings.values.T


def derive_pca_weights(
    data: pd.DataFrame,
    standardize: bool = True,
) -> dict[str, float]:
    """
    Convenience function to derive PCA weights.
    
    Args:
        data: DataFrame with sub-indices
        standardize: Whether to standardize before PCA
        
    Returns:
        Dictionary of variable -> weight
    """
    deriver = PCAWeightDeriver(standardize=standardize)
    deriver.fit(data)
    return deriver.weights


def format_pca_table(result: PCAResult) -> str:
    """Format PCA results as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Principal Component Analysis: Loadings and Weights}",
        r"\label{tab:pca}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Sub-Index & PC1 Loading & PC2 Loading & Derived Weight & Theoretical Weight \\",
        r"\midrule",
    ]
    
    # Theoretical weights for comparison
    theoretical = {
        "stablecoin_risk": 0.30,
        "defi_liquidity_risk": 0.25,
        "contagion_risk": 0.25,
        "arbitrage_opacity": 0.20,
    }
    
    for col in result.loadings.columns:
        pc1 = result.loadings.loc["PC1", col]
        pc2 = result.loadings.loc["PC2", col] if "PC2" in result.loadings.index else 0
        weight = result.weights[col]
        theo = theoretical.get(col, "N/A")
        
        lines.append(
            f"{col} & {pc1:.3f} & {pc2:.3f} & {weight:.3f} & {theo} \\\\"
        )
    
    lines.extend([
        r"\midrule",
        f"Variance Explained & {result.explained_variance_ratio[0]*100:.1f}\\% & "
        f"{result.explained_variance_ratio[1]*100:.1f}\\% & -- & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        f"\\item KMO measure: {result.kmo_statistic:.3f}" if result.kmo_statistic else "",
        f"\\item Bartlett's test $p$-value: {result.bartlett_pvalue:.4f}" if result.bartlett_pvalue else "",
        r"\item Derived weights are normalized absolute PC1 loadings.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)
