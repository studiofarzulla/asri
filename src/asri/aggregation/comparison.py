"""
Aggregation Method Comparison for ASRI

Implements and compares four aggregation methods for computing ASRI:
1. Linear: Baseline weighted sum (ASRI = w'@ s)
2. CISS: ECB-style sqrt(s' @ C_t @ s) with time-varying correlations
3. Copula: Tail-dependence amplified aggregation
4. Regime: HMM posterior-weighted regime-specific weights

Reference: Hollo et al. (2012) "CISS - A Composite Indicator of Systemic Stress"
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats

from .regime_aggregation import RegimeAggregator, RegimeWeights, SUBINDEX_COLUMNS

if TYPE_CHECKING:
    from ..regime.hmm import RegimeDetector


# =============================================================================
# Crisis Event Definitions
# =============================================================================

@dataclass
class CrisisEvent:
    """Definition of a known crisis event for validation."""
    name: str
    date: str  # Event date (YYYY-MM-DD)
    window_start: str  # Pre-event window start
    window_end: str  # Post-event window end


# Known crypto crisis events from the paper
CRISIS_EVENTS = [
    CrisisEvent("Terra/Luna", "2022-05-09", "2022-03-09", "2022-05-19"),
    CrisisEvent("FTX", "2022-11-08", "2022-09-08", "2022-11-18"),
    CrisisEvent("SVB/USDC", "2023-03-10", "2023-01-10", "2023-03-20"),
    CrisisEvent("August 2024", "2024-08-05", "2024-06-05", "2024-08-15"),
]


# =============================================================================
# Plotting Configuration
# =============================================================================

FARZULLA_BURGUNDY = "#800020"
FARZULLA_BURGUNDY_LIGHT = "#A63D5A"
FARZULLA_BURGUNDY_DARK = "#5C0017"

METHOD_COLORS = {
    "linear": FARZULLA_BURGUNDY,
    "ciss": "#4A6FA5",
    "copula": "#6B4A8E",
    "regime": "#2E8B57",
}

FIGURE_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def _setup_style() -> None:
    """Apply publication-quality matplotlib styling."""
    plt.rcParams.update(FIGURE_PARAMS)


def _ensure_output_dir(output_path: str) -> Path:
    """Ensure output directory exists and return Path object."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Aggregation Comparison Class
# =============================================================================

class AggregationComparison:
    """
    Compare different ASRI aggregation methodologies.

    Implements four aggregation methods and provides comparison metrics:
    - Linear: Simple weighted sum (baseline)
    - CISS: Portfolio variance approach with time-varying correlations
    - Copula: Tail-dependence amplified aggregation
    - Regime: HMM posterior-weighted regime-specific weights

    Example
    -------
    >>> comparator = AggregationComparison()
    >>> results = comparator.load_all_methods(
    ...     subindices=df,
    ...     ciss_aggregator=ciss,
    ...     copula_aggregator=copula,
    ...     regime_aggregator=regime,
    ... )
    >>> metrics = comparator.compute_crisis_metrics(results)
    """

    def __init__(self) -> None:
        self.methods = ["linear", "ciss", "copula", "regime"]
        self.results: dict[str, pd.Series] = {}
        self._crisis_metrics: pd.DataFrame | None = None
        self._normalized_results: dict[str, pd.Series] = {}

    def normalize_scales(
        self,
        asri_series: dict[str, pd.Series] | None = None,
        method: str = "minmax",
        reference: str | None = None,
    ) -> dict[str, pd.Series]:
        """
        Normalize all aggregation methods to the same scale for fair comparison.

        Different aggregation methods (linear, CISS, copula, regime) may produce
        outputs on different scales. This method normalizes them for comparability.

        Parameters
        ----------
        asri_series : dict[str, pd.Series], optional
            Dict of ASRI series by method. Uses stored results if None.
        method : str
            Normalization method:
            - 'minmax': Scale to [0, 100] range (default)
            - 'zscore': Standardize to mean=0, std=1
            - 'quantile': Map to [0, 100] via empirical quantiles
            - 'reference': Scale relative to a reference method
        reference : str, optional
            Reference method name for 'reference' normalization.
            Required if method='reference'.

        Returns
        -------
        dict[str, pd.Series]
            Dictionary of normalized ASRI series.

        Example
        -------
        >>> comparator = AggregationComparison()
        >>> results = comparator.load_all_methods(subindices, ...)
        >>> normalized = comparator.normalize_scales(method='minmax')
        >>> # Now all methods are on [0, 100] scale
        """
        if asri_series is None:
            asri_series = self.results

        if not asri_series:
            raise ValueError("No ASRI series provided or computed")

        normalized = {}

        if method == "minmax":
            for name, series in asri_series.items():
                s_min, s_max = series.min(), series.max()
                if s_max > s_min:
                    normalized[name] = 100 * (series - s_min) / (s_max - s_min)
                else:
                    normalized[name] = pd.Series(50.0, index=series.index, name=series.name)
                normalized[name].name = f"ASRI_{name}_normalized"

        elif method == "zscore":
            for name, series in asri_series.items():
                mean, std = series.mean(), series.std()
                if std > 0:
                    normalized[name] = (series - mean) / std
                else:
                    normalized[name] = pd.Series(0.0, index=series.index, name=series.name)
                normalized[name].name = f"ASRI_{name}_zscore"

        elif method == "quantile":
            for name, series in asri_series.items():
                # Map to percentile ranks
                ranks = series.rank(pct=True)
                normalized[name] = 100 * ranks
                normalized[name].name = f"ASRI_{name}_quantile"

        elif method == "reference":
            if reference is None:
                raise ValueError("reference method name required for 'reference' normalization")
            if reference not in asri_series:
                raise ValueError(f"Reference method '{reference}' not found in series")

            ref_series = asri_series[reference]
            ref_mean, ref_std = ref_series.mean(), ref_series.std()

            for name, series in asri_series.items():
                if name == reference:
                    normalized[name] = series.copy()
                else:
                    # Scale to match reference distribution
                    s_mean, s_std = series.mean(), series.std()
                    if s_std > 0:
                        normalized[name] = ref_mean + (series - s_mean) * (ref_std / s_std)
                    else:
                        normalized[name] = pd.Series(ref_mean, index=series.index, name=series.name)
                normalized[name].name = f"ASRI_{name}_refnorm"

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        self._normalized_results = normalized
        return normalized

    def compute_linear_asri(
        self,
        subindices: pd.DataFrame,
        weights: np.ndarray | None = None,
    ) -> pd.Series:
        """
        Baseline linear weighted sum aggregation.

        ASRI_t = w' @ s_t

        Parameters
        ----------
        subindices : pd.DataFrame
            Sub-index values with shape (T, K) where K is number of sub-indices.
        weights : np.ndarray, optional
            Weights vector of length K. Defaults to [0.30, 0.25, 0.25, 0.20].

        Returns
        -------
        pd.Series
            Linear ASRI time series.
        """
        if weights is None:
            weights = np.array([0.30, 0.25, 0.25, 0.20])

        # Ensure weights match columns
        if len(weights) != len(subindices.columns):
            raise ValueError(
                f"Weight dimension ({len(weights)}) doesn't match "
                f"sub-index count ({len(subindices.columns)})"
            )

        weights = np.array(weights) / np.sum(weights)  # Normalize

        asri = subindices.values @ weights
        return pd.Series(asri, index=subindices.index, name="ASRI_linear")

    def load_all_methods(
        self,
        subindices: pd.DataFrame,
        ciss_aggregator: Optional["object"] = None,
        copula_aggregator: Optional["object"] = None,
        regime_aggregator: Optional["RegimeDetector"] = None,
        linear_weights: np.ndarray | None = None,
    ) -> dict[str, pd.Series]:
        """
        Compute ASRI using all 4 methods and return dict of series.

        Parameters
        ----------
        subindices : pd.DataFrame
            Sub-index time series with datetime index.
        ciss_aggregator : object, optional
            CISS aggregator with .compute(subindices) method.
        copula_aggregator : object, optional
            Copula aggregator with .compute(subindices) method.
        regime_aggregator : RegimeDetector, optional
            Fitted regime detector with regime probabilities.
        linear_weights : np.ndarray, optional
            Weights for linear aggregation.

        Returns
        -------
        dict[str, pd.Series]
            Dictionary mapping method names to ASRI series.
        """
        # Always compute linear
        self.results["linear"] = self.compute_linear_asri(subindices, linear_weights)

        # CISS if aggregator provided
        if ciss_aggregator is not None:
            try:
                ciss_asri = ciss_aggregator.compute(subindices)
                self.results["ciss"] = pd.Series(
                    ciss_asri, index=subindices.index, name="ASRI_ciss"
                )
            except Exception as e:
                print(f"CISS aggregation failed: {e}")

        # Copula if aggregator provided
        if copula_aggregator is not None:
            try:
                copula_asri = copula_aggregator.compute(subindices)
                self.results["copula"] = pd.Series(
                    copula_asri, index=subindices.index, name="ASRI_copula"
                )
            except Exception as e:
                print(f"Copula aggregation failed: {e}")

        # Regime-switching if detector provided
        if regime_aggregator is not None:
            try:
                regime_asri = self._compute_regime_asri(subindices, regime_aggregator)
                self.results["regime"] = regime_asri
            except Exception as e:
                print(f"Regime aggregation failed: {e}")

        return self.results

    def _compute_regime_asri(
        self,
        subindices: pd.DataFrame,
        regime_aggregator: Union["RegimeDetector", "RegimeAggregator"],
    ) -> pd.Series:
        """
        Compute regime-weighted ASRI using HMM posterior probabilities.

        ASRI_t = sum_k P(regime_k | data_t) * (w_k' @ s_t)

        Supports both RegimeDetector (legacy) and RegimeAggregator interfaces.
        """
        # Handle RegimeAggregator (preferred path)
        if isinstance(regime_aggregator, RegimeAggregator):
            return regime_aggregator.aggregate(subindices)

        # Legacy path for RegimeDetector
        result = regime_aggregator.result
        probs = result.regime_probabilities

        # Align lengths
        min_len = min(len(subindices), len(probs))
        subindices_aligned = subindices.iloc[:min_len]
        probs = probs[:min_len]

        # Get regime weights - check for RegimeWeights object or dict
        regime_weight_arrays: dict[int, np.ndarray] = {}

        if hasattr(regime_aggregator, "regime_weights") and regime_aggregator.regime_weights is not None:
            weights_obj = regime_aggregator.regime_weights
            if isinstance(weights_obj, RegimeWeights):
                # Use the RegimeWeights dataclass
                regime_weight_arrays = {
                    0: weights_obj.low_risk,
                    1: weights_obj.moderate,
                    2: weights_obj.crisis,
                }
            elif isinstance(weights_obj, dict) and weights_obj:
                # Legacy dict format from RegimeResult.regime_weights
                for k, w_dict in weights_obj.items():
                    if isinstance(w_dict, dict):
                        # Order weights according to SUBINDEX_COLUMNS
                        w_k = np.array([w_dict.get(col, 0.25) for col in SUBINDEX_COLUMNS])
                        regime_weight_arrays[k] = w_k / w_k.sum()
                    else:
                        regime_weight_arrays[k] = np.array(w_dict)
        elif hasattr(result, "regime_weights") and result.regime_weights:
            # Check RegimeResult.regime_weights
            for k, w_dict in result.regime_weights.items():
                if isinstance(w_dict, dict):
                    w_k = np.array([w_dict.get(col, 0.25) for col in SUBINDEX_COLUMNS])
                    regime_weight_arrays[k] = w_k / w_k.sum()
                else:
                    regime_weight_arrays[k] = np.array(w_dict)

        # Fallback to equal weights if nothing found
        n_features = len(subindices_aligned.columns)
        if not regime_weight_arrays:
            warnings.warn(
                "No regime weights found in aggregator, using equal weights. "
                "Consider fitting a RegimeAggregator with explicit RegimeWeights.",
                UserWarning,
            )
            regime_weight_arrays = {
                k: np.ones(n_features) / n_features for k in range(result.n_regimes)
            }

        # Compute regime-weighted ASRI
        asri = np.zeros(min_len)

        for t in range(min_len):
            s_t = subindices_aligned.iloc[t].values
            regime_asri = 0.0

            for k in range(result.n_regimes):
                w_k = regime_weight_arrays.get(k, np.ones(n_features) / n_features)
                w_k = w_k / np.sum(w_k)  # Ensure normalized
                regime_asri += probs[t, k] * (w_k @ s_t)

            asri[t] = regime_asri

        return pd.Series(asri, index=subindices_aligned.index, name="ASRI_regime")

    def compute_crisis_metrics(
        self,
        asri_series: dict[str, pd.Series] | None = None,
        threshold: float = 70.0,
        baseline_window: int = 60,
    ) -> pd.DataFrame:
        """
        For each method and crisis, compute early warning metrics.

        Parameters
        ----------
        asri_series : dict[str, pd.Series], optional
            Dict of ASRI series by method. Uses stored results if None.
        threshold : float
            ASRI threshold for "detection" (default: 70.0).
        baseline_window : int
            Days before crisis window for computing "normal" ASRI.

        Returns
        -------
        pd.DataFrame
            MultiIndex DataFrame (crisis, method) with columns:
            - detected: Did ASRI exceed threshold before crisis?
            - lead_time: Days between threshold breach and crisis
            - peak: Maximum ASRI in pre-crisis window
            - cas: Cumulative Abnormal Signal (sum of ASRI above normal)
            - t_stat: t-statistic for abnormal signal
        """
        if asri_series is None:
            asri_series = self.results

        if not asri_series:
            raise ValueError("No ASRI series provided or computed")

        records = []

        for crisis in CRISIS_EVENTS:
            event_date = pd.Timestamp(crisis.date)
            window_start = pd.Timestamp(crisis.window_start)
            window_end = pd.Timestamp(crisis.window_end)

            # Baseline window is before the pre-crisis window
            baseline_start = window_start - pd.Timedelta(days=baseline_window)
            baseline_end = window_start - pd.Timedelta(days=1)

            for method, asri in asri_series.items():
                # Skip if data doesn't cover this crisis
                if asri.index.min() > baseline_start or asri.index.max() < event_date:
                    continue

                # Get windows
                baseline = asri[(asri.index >= baseline_start) & (asri.index <= baseline_end)]
                pre_crisis = asri[(asri.index >= window_start) & (asri.index < event_date)]

                if len(baseline) < 10 or len(pre_crisis) < 5:
                    continue

                # Metrics
                baseline_mean = baseline.mean()
                baseline_std = baseline.std()

                peak = pre_crisis.max()

                # Detection: did ASRI breach threshold before event?
                threshold_breaches = pre_crisis[pre_crisis > threshold]
                detected = len(threshold_breaches) > 0

                # Lead time: first breach before event
                if detected:
                    first_breach = threshold_breaches.index[0]
                    lead_time = (event_date - first_breach).days
                else:
                    lead_time = 0

                # Cumulative Abnormal Signal
                abnormal = pre_crisis - baseline_mean
                cas = abnormal.sum()

                # t-statistic
                n = len(abnormal)
                se = baseline_std * np.sqrt(n) if baseline_std > 0 else 1
                t_stat = cas / se

                records.append({
                    "crisis": crisis.name,
                    "method": method,
                    "detected": detected,
                    "lead_time": lead_time,
                    "peak": peak,
                    "cas": cas,
                    "t_stat": t_stat,
                })

        df = pd.DataFrame(records)
        if len(df) > 0:
            df = df.set_index(["crisis", "method"])

        self._crisis_metrics = df
        return df

    def compute_correlation_by_regime(
        self,
        asri_series: dict[str, pd.Series] | None = None,
        regime_labels: pd.Series | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Cross-method correlations split by regime.

        Useful for understanding whether aggregation methods converge
        or diverge during different market conditions.

        Parameters
        ----------
        asri_series : dict[str, pd.Series], optional
            Dict of ASRI series by method.
        regime_labels : pd.Series or np.ndarray, optional
            Regime labels (0, 1, 2, ...) for each time step.

        Returns
        -------
        pd.DataFrame
            Correlation matrix for each regime.
        """
        if asri_series is None:
            asri_series = self.results

        # Build combined DataFrame
        combined = pd.DataFrame(asri_series)

        if regime_labels is None:
            # No regime split - just compute overall correlation
            return combined.corr()

        # Align regime labels
        if isinstance(regime_labels, np.ndarray):
            regime_labels = pd.Series(regime_labels, index=combined.index[:len(regime_labels)])

        # Compute correlation by regime
        unique_regimes = sorted(regime_labels.dropna().unique())

        correlations = {}
        for regime in unique_regimes:
            mask = regime_labels == regime
            regime_data = combined.loc[mask.values[:len(combined)]]

            if len(regime_data) > 10:
                correlations[f"regime_{regime}"] = regime_data.corr()

        return pd.concat(correlations, axis=0)

    def plot_method_comparison(
        self,
        asri_series: dict[str, pd.Series] | None = None,
        output_path: str = "figures/aggregation_comparison.pdf",
    ) -> None:
        """
        4-panel figure comparing aggregation methods.

        Each panel shows one method's ASRI with crisis event shading.
        All panels use the same y-axis scale for comparability.

        Parameters
        ----------
        asri_series : dict[str, pd.Series], optional
            Dict of ASRI series by method.
        output_path : str
            Path to save figure.
        """
        _setup_style()
        output_path = _ensure_output_dir(output_path)

        if asri_series is None:
            asri_series = self.results

        # Get available methods
        available = [m for m in self.methods if m in asri_series]
        n_methods = len(available)

        if n_methods == 0:
            raise ValueError("No ASRI series available to plot")

        # Determine layout
        n_cols = 2
        n_rows = (n_methods + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows), squeeze=False)
        axes = axes.flatten()

        # Global y-limits for comparability
        y_max = max(s.max() for s in asri_series.values() if len(s) > 0)
        y_max = min(100, y_max * 1.1)

        method_labels = {
            "linear": "Linear Weighted Sum",
            "ciss": "CISS (Portfolio Variance)",
            "copula": "Copula (Tail Dependence)",
            "regime": "Regime-Switching",
        }

        for i, method in enumerate(available):
            ax = axes[i]
            asri = asri_series[method]

            # Crisis shading
            for crisis in CRISIS_EVENTS:
                window_start = pd.Timestamp(crisis.window_start)
                window_end = pd.Timestamp(crisis.window_end)
                event_date = pd.Timestamp(crisis.date)

                if asri.index.min() <= window_end and asri.index.max() >= window_start:
                    ax.axvspan(
                        max(window_start, asri.index.min()),
                        min(window_end, asri.index.max()),
                        alpha=0.15,
                        color="#DC143C",
                        linewidth=0,
                    )

                    if asri.index.min() <= event_date <= asri.index.max():
                        ax.axvline(
                            event_date,
                            color=FARZULLA_BURGUNDY_DARK,
                            linestyle="--",
                            linewidth=0.8,
                            alpha=0.7,
                        )

            # ASRI line
            ax.plot(
                asri.index,
                asri.values,
                color=METHOD_COLORS.get(method, FARZULLA_BURGUNDY),
                linewidth=1.2,
                label=method_labels.get(method, method),
            )

            # Alert threshold line
            ax.axhline(70, color="#DC143C", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.axhline(50, color="#FF8C00", linestyle=":", linewidth=0.8, alpha=0.4)

            ax.set_xlim(asri.index.min(), asri.index.max())
            ax.set_ylim(0, y_max)
            ax.set_ylabel("ASRI")
            ax.set_title(method_labels.get(method, method))

            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

        # Hide unused axes
        for j in range(n_methods, len(axes)):
            axes[j].set_visible(False)

        # Common x-label
        fig.supxlabel("Date", y=-0.02)
        fig.suptitle("ASRI Aggregation Method Comparison", fontsize=13, y=1.01)

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved: {output_path}")

    def plot_crisis_zoom(
        self,
        asri_series: dict[str, pd.Series] | None = None,
        crisis: CrisisEvent | None = None,
        output_path: str | None = None,
    ) -> None:
        """
        Single panel with all 4 methods overlaid for one crisis event.

        Parameters
        ----------
        asri_series : dict[str, pd.Series], optional
            Dict of ASRI series by method.
        crisis : CrisisEvent, optional
            Crisis to zoom in on. Defaults to first CRISIS_EVENT.
        output_path : str, optional
            Path to save figure. If None, auto-generated from crisis name.
        """
        _setup_style()

        if asri_series is None:
            asri_series = self.results

        if crisis is None:
            crisis = CRISIS_EVENTS[0]

        if output_path is None:
            safe_name = crisis.name.lower().replace("/", "_").replace(" ", "_")
            output_path = f"figures/crisis_zoom_{safe_name}.pdf"

        output_path = _ensure_output_dir(output_path)

        window_start = pd.Timestamp(crisis.window_start)
        window_end = pd.Timestamp(crisis.window_end)
        event_date = pd.Timestamp(crisis.date)

        fig, ax = plt.subplots(figsize=(8, 5))

        # Event line and shading
        ax.axvline(
            event_date,
            color=FARZULLA_BURGUNDY_DARK,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="Event Date",
        )
        ax.axvspan(
            window_start,
            event_date,
            alpha=0.1,
            color="#2E8B57",
            label="Pre-Crisis",
        )
        ax.axvspan(
            event_date,
            window_end,
            alpha=0.1,
            color="#DC143C",
            label="Post-Crisis",
        )

        # Plot each method
        method_labels = {
            "linear": "Linear",
            "ciss": "CISS",
            "copula": "Copula",
            "regime": "Regime",
        }

        for method, asri in asri_series.items():
            window_data = asri[(asri.index >= window_start) & (asri.index <= window_end)]

            if len(window_data) > 0:
                ax.plot(
                    window_data.index,
                    window_data.values,
                    color=METHOD_COLORS.get(method, "#808080"),
                    linewidth=1.5,
                    label=method_labels.get(method, method),
                )

        # Threshold lines
        ax.axhline(70, color="#DC143C", linestyle=":", linewidth=0.8, alpha=0.6, label="Critical")
        ax.axhline(50, color="#FF8C00", linestyle=":", linewidth=0.8, alpha=0.4, label="High")

        ax.set_xlim(window_start, window_end)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Date")
        ax.set_ylabel("ASRI")
        ax.set_title(f"Aggregation Methods: {crisis.name} Crisis")

        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)

        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved: {output_path}")

    def format_comparison_table(
        self,
        metrics_df: pd.DataFrame | None = None,
    ) -> str:
        """
        Generate LaTeX table comparing aggregation methods.

        Parameters
        ----------
        metrics_df : pd.DataFrame, optional
            Crisis metrics DataFrame. Uses stored metrics if None.

        Returns
        -------
        str
            LaTeX table code.
        """
        if metrics_df is None:
            metrics_df = self._crisis_metrics

        if metrics_df is None or len(metrics_df) == 0:
            return "% No metrics available"

        # Aggregate by method
        method_stats = metrics_df.groupby("method").agg({
            "detected": "mean",
            "lead_time": "mean",
            "t_stat": "mean",
            "peak": "mean",
        }).round(2)

        lines = [
            r"\begin{table}[htbp]",
            r"\begin{threeparttable}",
            r"\centering",
            r"\caption{Aggregation Method Comparison: Crisis Detection Performance}",
            r"\label{tab:aggregation_comparison}",
            r"\small",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Method & Detection Rate & Mean Lead Time & Mean $t$-stat & Mean Peak \\",
            r"\midrule",
        ]

        method_names = {
            "linear": "Linear",
            "ciss": "CISS",
            "copula": "Copula",
            "regime": "Regime-Switching",
        }

        for method in ["linear", "ciss", "copula", "regime"]:
            if method not in method_stats.index:
                continue

            row = method_stats.loc[method]
            detection_pct = row["detected"] * 100

            lines.append(
                f"{method_names.get(method, method)} & "
                f"{detection_pct:.0f}\\% & "
                f"{row['lead_time']:.1f} days & "
                f"{row['t_stat']:.2f} & "
                f"{row['peak']:.1f} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"\item Detection Rate = percentage of crises where ASRI exceeded 70 before event.",
            r"\item Lead Time = average days between first threshold breach and crisis onset.",
            r"\item $t$-statistic tests whether pre-crisis ASRI is abnormally elevated.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def normalize_for_comparison(
        self,
        asri_series: dict[str, pd.Series] | None = None,
        target_range: tuple[float, float] = (0.0, 100.0),
    ) -> dict[str, pd.Series]:
        """
        Normalize all ASRI series to a common 0-100 scale for fair comparison.

        Different aggregation methods may produce values on different scales
        (e.g., CISS uses sqrt of portfolio variance). This method applies
        min-max normalization to put all methods on the same scale.

        Parameters
        ----------
        asri_series : dict[str, pd.Series], optional
            Dict of ASRI series by method. Uses stored results if None.
        target_range : tuple[float, float]
            Target (min, max) range for normalized values. Default (0, 100).

        Returns
        -------
        dict[str, pd.Series]
            Dictionary of normalized ASRI series, all on 0-100 scale.
        """
        if asri_series is None:
            asri_series = self.results

        if not asri_series:
            raise ValueError("No ASRI series provided or computed")

        target_min, target_max = target_range
        normalized = {}

        for method, series in asri_series.items():
            s_min = series.min()
            s_max = series.max()

            if s_max - s_min < 1e-10:
                # Constant series - assign midpoint
                norm_values = np.full(len(series), (target_min + target_max) / 2)
            else:
                # Min-max normalization
                norm_values = (series.values - s_min) / (s_max - s_min)
                norm_values = norm_values * (target_max - target_min) + target_min

            normalized[method] = pd.Series(
                norm_values, index=series.index, name=f"{series.name}_normalized"
            )

        return normalized

    def format_full_comparison_table(
        self,
        in_sample_metrics: pd.DataFrame,
        oos_metrics: pd.DataFrame | None = None,
    ) -> str:
        """
        Generate LaTeX table comparing all 4 methods with in-sample and OOS metrics.

        Parameters
        ----------
        in_sample_metrics : pd.DataFrame
            In-sample crisis metrics from compute_crisis_metrics().
        oos_metrics : pd.DataFrame, optional
            Out-of-sample metrics. If None, OOS column shows "--".

        Returns
        -------
        str
            LaTeX table code comparing all methods.

        Notes
        -----
        Columns: Method | Detection Rate | Lead Time (days) | Mean t-stat | OOS Detection
        """
        if in_sample_metrics is None or len(in_sample_metrics) == 0:
            return "% No in-sample metrics available"

        # Aggregate in-sample by method
        is_stats = in_sample_metrics.groupby("method").agg({
            "detected": "mean",
            "lead_time": "mean",
            "t_stat": "mean",
        }).round(2)

        # Aggregate OOS if available
        oos_stats = None
        if oos_metrics is not None and len(oos_metrics) > 0:
            oos_stats = oos_metrics.groupby("method").agg({
                "detected": "mean",
            }).round(2)

        lines = [
            r"\begin{table}[h]",
            r"\begin{threeparttable}",
            r"\centering",
            r"\caption{Aggregation Method Comparison}",
            r"\label{tab:full_aggregation_comparison}",
            r"\small",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Method & Detection Rate & Lead Time (days) & Mean $t$-stat & OOS Detection \\",
            r"\midrule",
        ]

        method_names = {
            "linear": "Linear",
            "ciss": "CISS",
            "copula": "Copula",
            "regime": "Regime-Switching",
        }

        for method in ["linear", "ciss", "copula", "regime"]:
            if method not in is_stats.index:
                continue

            row = is_stats.loc[method]
            detection_pct = row["detected"] * 100

            # OOS detection
            if oos_stats is not None and method in oos_stats.index:
                oos_pct = oos_stats.loc[method, "detected"] * 100
                oos_str = f"{oos_pct:.0f}\\%"
            else:
                oos_str = "--"

            lines.append(
                f"{method_names.get(method, method)} & "
                f"{detection_pct:.0f}\\% & "
                f"{row['lead_time']:.1f} & "
                f"{row['t_stat']:.2f} & "
                f"{oos_str} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"\item Detection Rate = in-sample percentage of crises detected (threshold = 70).",
            r"\item Lead Time = average days between first threshold breach and crisis onset.",
            r"\item OOS Detection = out-of-sample detection rate on held-out crisis events.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def summarize_findings(
        self,
        metrics_df: pd.DataFrame | None = None,
    ) -> str:
        """
        Generate text summary of key findings for paper discussion.

        Parameters
        ----------
        metrics_df : pd.DataFrame, optional
            Crisis metrics DataFrame.

        Returns
        -------
        str
            Summary text suitable for results section.
        """
        if metrics_df is None:
            metrics_df = self._crisis_metrics

        if metrics_df is None or len(metrics_df) == 0:
            return "Insufficient data for comparison analysis."

        # Method rankings
        method_stats = metrics_df.groupby("method").agg({
            "detected": "mean",
            "lead_time": "mean",
            "t_stat": "mean",
        })

        best_detection = method_stats["detected"].idxmax()
        best_lead = method_stats.loc[method_stats["detected"] > 0, "lead_time"].idxmax()
        best_tstat = method_stats["t_stat"].idxmax()

        detection_rate = method_stats.loc[best_detection, "detected"] * 100
        lead_time = method_stats.loc[best_lead, "lead_time"]

        summary = [
            f"Comparison of four aggregation methods reveals {best_detection} achieves "
            f"the highest detection rate ({detection_rate:.0f}% of crises). ",
            f"The {best_lead} method provides the longest average lead time "
            f"({lead_time:.1f} days before crisis onset). ",
            f"Statistical significance is strongest for {best_tstat} aggregation, "
            f"suggesting more reliable separation between normal and crisis periods. ",
        ]

        # Check for convergence
        if len(method_stats) > 1:
            detection_spread = method_stats["detected"].max() - method_stats["detected"].min()
            if detection_spread < 0.2:
                summary.append(
                    "All methods show broadly similar detection rates, "
                    "indicating robustness of the underlying sub-indices. "
                )
            else:
                summary.append(
                    "Substantial variation in detection rates across methods suggests "
                    "aggregation choice materially affects early warning performance. "
                )

        return "".join(summary)


# =============================================================================
# Demo / Testing
# =============================================================================

if __name__ == "__main__":
    print("Running aggregation comparison demo with synthetic data...")

    # Generate synthetic sub-indices
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    n = len(dates)

    # Base signals with crisis spikes
    base = 35 + 10 * np.sin(np.linspace(0, 4 * np.pi, n))
    noise = np.random.normal(0, 5, (n, 4))

    subindices = pd.DataFrame(
        {
            "Market": base + noise[:, 0],
            "Liquidity": base * 0.9 + noise[:, 1] + 5,
            "Credit": base * 0.8 + noise[:, 2] + 10,
            "Contagion": base * 1.1 + noise[:, 3] - 5,
        },
        index=dates,
    )

    # Add crisis spikes
    for crisis in CRISIS_EVENTS:
        try:
            event_date = pd.Timestamp(crisis.date)
            if event_date < dates.min() or event_date > dates.max():
                continue

            crisis_mask = (
                (dates >= pd.Timestamp(crisis.window_start))
                & (dates <= pd.Timestamp(crisis.window_end))
            )
            spike = np.linspace(0, 35, crisis_mask.sum())
            spike = np.concatenate([spike[:len(spike)//2], spike[len(spike)//2:][::-1]])

            for col in subindices.columns:
                subindices.loc[crisis_mask, col] += spike + np.random.normal(0, 3, len(spike))
        except Exception:
            continue

    subindices = subindices.clip(0, 100)

    # Run comparison
    comparator = AggregationComparison()

    # Compute linear (other methods would need actual aggregators)
    linear_asri = comparator.compute_linear_asri(subindices)

    # Synthetic alternatives for demo
    asri_series = {
        "linear": linear_asri,
        "ciss": linear_asri * 1.05 + np.random.normal(0, 2, len(linear_asri)),
        "copula": linear_asri * 0.95 + np.random.normal(0, 3, len(linear_asri)),
        "regime": linear_asri + np.random.normal(0, 4, len(linear_asri)),
    }

    for method, series in asri_series.items():
        asri_series[method] = pd.Series(
            np.clip(series, 0, 100),
            index=dates,
            name=f"ASRI_{method}",
        )

    comparator.results = asri_series

    # Compute metrics
    print("\nComputing crisis metrics...")
    metrics = comparator.compute_crisis_metrics(asri_series)
    print(metrics)

    # Generate figures
    print("\nGenerating comparison figures...")
    comparator.plot_method_comparison(output_path="figures/demo/aggregation_comparison.pdf")
    comparator.plot_crisis_zoom(crisis=CRISIS_EVENTS[0], output_path="figures/demo/crisis_zoom_terra.pdf")

    # Generate table
    print("\nLaTeX table:")
    print(comparator.format_comparison_table())

    # Summary
    print("\nFindings summary:")
    print(comparator.summarize_findings())
