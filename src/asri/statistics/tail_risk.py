"""
Tail Risk Measures: VCoVaR and Related Statistics

Addresses Reviewer Q11: VCoVaR/VCoES for tail risk in Contagion sub-index.

VCoVaR (Conditional Value-at-Risk) measures the VaR of the system conditional
on a specific asset being in distress. Unlike simple correlation, VCoVaR
captures tail dependence and asymmetric spillover effects.

References:
- Adrian, T., & Brunnermeier, M. K. (2016). CoVaR. American Economic Review.
- Girardi, G., & Ergün, A. T. (2013). Systemic risk measurement: Multivariate GARCH.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class VCoVaRResult:
    """Results from VCoVaR estimation."""
    var_unconditional: float      # Unconditional VaR of system
    covar_distress: float         # VaR of system | asset distressed
    covar_median: float           # VaR of system | asset at median
    delta_covar: float            # Systemic risk contribution
    quantile: float               # Alpha (default 0.05)
    asset_name: str               # Name of conditioning asset
    system_name: str              # Name of system


@dataclass
class TailDependenceResult:
    """Results from tail dependence analysis."""
    upper_tail: float             # Upper tail dependence coefficient
    lower_tail: float             # Lower tail dependence coefficient
    asymmetry: float              # |upper - lower|
    method: str                   # Estimation method


class VCoVaREstimator:
    """
    Estimate VCoVaR using quantile regression.

    VCoVaR measures the Value-at-Risk of a system conditional on
    a specific asset being in distress (at its VaR level).

    ΔCoVaR = CoVaR(q|Asset at VaR_q) - CoVaR(q|Asset at median)

    This difference measures the asset's contribution to systemic risk.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Quantile for VaR/CoVaR (default 5%)
        """
        self.alpha = alpha

    def estimate_var(
        self,
        returns: pd.Series,
        method: str = "historical",
    ) -> float:
        """
        Estimate Value-at-Risk.

        Args:
            returns: Return series
            method: "historical" or "parametric"

        Returns:
            VaR at alpha quantile (negative number = loss)
        """
        if method == "historical":
            return returns.quantile(self.alpha)
        elif method == "parametric":
            # Assume normal distribution
            mu = returns.mean()
            sigma = returns.std()
            return mu + sigma * stats.norm.ppf(self.alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

    def estimate_covar(
        self,
        system_returns: pd.Series,
        asset_returns: pd.Series,
        method: str = "quantile_regression",
    ) -> VCoVaRResult:
        """
        Estimate CoVaR and ΔCoVaR.

        Args:
            system_returns: Returns of the system (e.g., market index)
            asset_returns: Returns of the conditioning asset
            method: "quantile_regression" or "historical"

        Returns:
            VCoVaRResult with all metrics
        """
        # Align series
        common_idx = system_returns.index.intersection(asset_returns.index)
        system = system_returns.loc[common_idx].dropna()
        asset = asset_returns.loc[common_idx].dropna()

        if len(system) < 50:
            raise ValueError("Insufficient data for CoVaR estimation")

        # Unconditional VaR of system
        var_system = self.estimate_var(system)

        if method == "quantile_regression":
            covar_distress, covar_median = self._quantile_regression_covar(
                system, asset
            )
        elif method == "historical":
            covar_distress, covar_median = self._historical_covar(
                system, asset
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        delta_covar = covar_distress - covar_median

        return VCoVaRResult(
            var_unconditional=var_system,
            covar_distress=covar_distress,
            covar_median=covar_median,
            delta_covar=delta_covar,
            quantile=self.alpha,
            asset_name=asset.name or "Asset",
            system_name=system.name or "System",
        )

    def _quantile_regression_covar(
        self,
        system: pd.Series,
        asset: pd.Series,
    ) -> tuple[float, float]:
        """
        Estimate CoVaR using quantile regression.

        Model: system_t = α + β × asset_t + ε_t
        Estimate at quantile α to get CoVaR.
        """
        try:
            import statsmodels.api as sm
            from statsmodels.regression.quantile_regression import QuantReg
        except ImportError:
            # Fall back to historical method
            return self._historical_covar(system, asset)

        # Prepare data
        X = sm.add_constant(asset.values)
        y = system.values

        # Quantile regression at alpha
        model = QuantReg(y, X)
        result = model.fit(q=self.alpha, max_iter=1000)

        # CoVaR when asset is at VaR
        asset_var = self.estimate_var(asset)
        covar_distress = result.params[0] + result.params[1] * asset_var

        # CoVaR when asset is at median
        asset_median = asset.median()
        covar_median = result.params[0] + result.params[1] * asset_median

        return covar_distress, covar_median

    def _historical_covar(
        self,
        system: pd.Series,
        asset: pd.Series,
    ) -> tuple[float, float]:
        """
        Estimate CoVaR using historical simulation.

        Filter system returns to periods when asset is in distress.
        """
        # Asset VaR threshold
        asset_var = self.estimate_var(asset)

        # System returns when asset is at/below VaR
        distress_mask = asset <= asset_var
        system_distress = system[distress_mask]

        if len(system_distress) < 10:
            # Expand the distress window
            asset_q10 = asset.quantile(0.10)
            distress_mask = asset <= asset_q10
            system_distress = system[distress_mask]

        covar_distress = system_distress.quantile(self.alpha) if len(system_distress) > 0 else system.quantile(self.alpha)

        # System returns when asset is around median
        asset_median = asset.median()
        asset_q40 = asset.quantile(0.40)
        asset_q60 = asset.quantile(0.60)
        median_mask = (asset >= asset_q40) & (asset <= asset_q60)
        system_median = system[median_mask]

        covar_median = system_median.quantile(self.alpha) if len(system_median) > 0 else system.quantile(self.alpha)

        return covar_distress, covar_median


def estimate_tail_dependence(
    x: pd.Series,
    y: pd.Series,
    method: str = "empirical",
    threshold: float = 0.10,
) -> TailDependenceResult:
    """
    Estimate tail dependence coefficients.

    Upper tail dependence: P(Y > F_Y^{-1}(u) | X > F_X^{-1}(u)) as u → 1
    Lower tail dependence: P(Y < F_Y^{-1}(u) | X < F_X^{-1}(u)) as u → 0

    Args:
        x, y: Time series
        method: "empirical" or "copula"
        threshold: Quantile threshold (e.g., 0.10 for 10th/90th percentile)

    Returns:
        TailDependenceResult
    """
    # Align series
    common_idx = x.index.intersection(y.index)
    x = x.loc[common_idx].dropna()
    y = y.loc[common_idx].dropna()

    n = len(x)

    if method == "empirical":
        # Convert to uniform marginals (ranks)
        x_rank = x.rank() / (n + 1)
        y_rank = y.rank() / (n + 1)

        # Upper tail dependence
        upper_mask = (x_rank > 1 - threshold) & (y_rank > 1 - threshold)
        upper_joint = upper_mask.sum()
        upper_marginal = (x_rank > 1 - threshold).sum()
        upper_tail = upper_joint / upper_marginal if upper_marginal > 0 else 0

        # Lower tail dependence
        lower_mask = (x_rank < threshold) & (y_rank < threshold)
        lower_joint = lower_mask.sum()
        lower_marginal = (x_rank < threshold).sum()
        lower_tail = lower_joint / lower_marginal if lower_marginal > 0 else 0

    elif method == "copula":
        # Placeholder for copula-based estimation
        # Would require copulae library
        upper_tail = 0.0
        lower_tail = 0.0

    else:
        raise ValueError(f"Unknown method: {method}")

    return TailDependenceResult(
        upper_tail=upper_tail,
        lower_tail=lower_tail,
        asymmetry=abs(upper_tail - lower_tail),
        method=method,
    )


def compute_tail_risk_score(
    crypto_returns: pd.Series,
    equity_returns: pd.Series,
    normalize_to: tuple[float, float] = (0, 100),
) -> float:
    """
    Compute normalized tail risk score for ASRI Contagion sub-index.

    Combines VCoVaR and tail dependence into a single score.
    Higher score = more tail risk = more systemic danger.

    Args:
        crypto_returns: BTC or ETH returns
        equity_returns: S&P 500 returns
        normalize_to: Output range (default 0-100)

    Returns:
        Tail risk score in [normalize_to[0], normalize_to[1]]
    """
    # Estimate VCoVaR
    estimator = VCoVaREstimator(alpha=0.05)

    try:
        covar_result = estimator.estimate_covar(
            equity_returns, crypto_returns, method="historical"
        )
        # Delta CoVaR as percentage of unconditional VaR
        # More negative delta = crypto distress causes bigger equity loss
        delta_covar_pct = abs(covar_result.delta_covar / covar_result.var_unconditional) \
            if covar_result.var_unconditional != 0 else 0
    except Exception:
        delta_covar_pct = 0

    # Estimate tail dependence
    try:
        tail_result = estimate_tail_dependence(
            crypto_returns, equity_returns, method="empirical"
        )
        # Average tail dependence
        tail_dep = (tail_result.upper_tail + tail_result.lower_tail) / 2
    except Exception:
        tail_dep = 0

    # Combine into score (weighted average)
    # Delta CoVaR contribution: Higher = more risk
    # Tail dependence: Higher = more joint extreme moves
    raw_score = 0.6 * delta_covar_pct + 0.4 * tail_dep

    # Normalize to output range
    # Typical delta_covar_pct ranges from 0 to ~0.5
    # Typical tail_dep ranges from 0 to ~0.3
    # So raw_score typically in [0, 0.4]
    normalized = min(1, raw_score / 0.4)  # Scale to [0, 1]

    lo, hi = normalize_to
    return lo + normalized * (hi - lo)


def rolling_delta_covar(
    system_returns: pd.Series,
    asset_returns: pd.Series,
    window: int = 90,
    alpha: float = 0.05,
) -> pd.Series:
    """
    Compute rolling ΔCoVaR time series.

    Args:
        system_returns: System returns
        asset_returns: Asset returns
        window: Rolling window size
        alpha: VaR quantile

    Returns:
        Series of ΔCoVaR values
    """
    estimator = VCoVaREstimator(alpha=alpha)

    results = []
    dates = []

    common_idx = system_returns.index.intersection(asset_returns.index)
    system = system_returns.loc[common_idx]
    asset = asset_returns.loc[common_idx]

    for i in range(window, len(system)):
        window_system = system.iloc[i-window:i]
        window_asset = asset.iloc[i-window:i]

        try:
            result = estimator.estimate_covar(
                window_system, window_asset, method="historical"
            )
            results.append(result.delta_covar)
        except Exception:
            results.append(np.nan)

        dates.append(system.index[i])

    return pd.Series(results, index=dates, name="delta_covar")
