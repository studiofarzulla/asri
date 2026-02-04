"""
Publication-quality figure generation for ASRI paper.

Generates academic figures with consistent Farzulla Research branding.
Primary color: Farzulla Burgundy (#800020)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    import pandas as pd

# =============================================================================
# Color Palette & Style Configuration
# =============================================================================

FARZULLA_BURGUNDY = "#800020"
FARZULLA_BURGUNDY_LIGHT = "#A63D5A"
FARZULLA_BURGUNDY_DARK = "#5C0017"

# Alert level colors (traffic light system)
ALERT_COLORS = {
    "low": "#2E8B57",        # Sea green - safe
    "elevated": "#FFD700",   # Gold - caution
    "high": "#FF8C00",       # Dark orange - warning
    "critical": "#DC143C",   # Crimson - danger
}

# Regime colors
REGIME_COLORS = {
    "tranquil": "#2E8B57",
    "transitional": "#FFD700",
    "crisis": "#DC143C",
}

# Academic figure settings
FIGURE_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}


def _setup_style() -> None:
    """Apply publication-quality matplotlib styling."""
    plt.rcParams.update(FIGURE_PARAMS)
    sns.set_palette([FARZULLA_BURGUNDY, "#4A4A4A", "#7B7B7B", "#A63D5A", "#2E5B8B"])


def _ensure_output_dir(output_path: str) -> Path:
    """Ensure output directory exists and return Path object."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Main Figure Functions
# =============================================================================


def plot_asri_time_series(
    asri: "pd.Series",
    crisis_dates: list[datetime],
    output_path: str = "figures/asri_timeseries.pdf",
    crisis_labels: list[str] | None = None,
    title: str = "Aggregated Systemic Risk Index (ASRI)",
) -> None:
    """
    ASRI time series with alert level bands and crisis event markers.

    Parameters
    ----------
    asri : pd.Series
        ASRI values indexed by datetime.
    crisis_dates : list[datetime]
        Dates of crisis events to mark with vertical lines.
    output_path : str
        Path for saving the figure (PDF recommended).
    crisis_labels : list[str], optional
        Labels for each crisis event. If None, uses generic "Crisis N".
    title : str
        Figure title.
    """
    _setup_style()
    output_path = _ensure_output_dir(output_path)

    if crisis_labels is None:
        crisis_labels = [f"Crisis {i+1}" for i in range(len(crisis_dates))]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot alert level bands
    ax.axhspan(0, 30, alpha=0.15, color=ALERT_COLORS["low"], label="Low Risk")
    ax.axhspan(30, 50, alpha=0.15, color=ALERT_COLORS["elevated"], label="Elevated")
    ax.axhspan(50, 70, alpha=0.15, color=ALERT_COLORS["high"], label="High Risk")
    ax.axhspan(70, 100, alpha=0.15, color=ALERT_COLORS["critical"], label="Critical")

    # Plot ASRI line
    ax.plot(
        asri.index,
        asri.values,
        color=FARZULLA_BURGUNDY,
        linewidth=1.5,
        label="ASRI",
        zorder=5,
    )

    # Add crisis event markers
    y_offset_cycle = [85, 90, 82, 88]  # Stagger labels to avoid overlap
    for i, (date, label) in enumerate(zip(crisis_dates, crisis_labels)):
        ax.axvline(
            date,
            color=FARZULLA_BURGUNDY_DARK,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            zorder=4,
        )
        y_pos = y_offset_cycle[i % len(y_offset_cycle)]
        ax.annotate(
            label,
            xy=(date, y_pos),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color=FARZULLA_BURGUNDY_DARK,
            rotation=90,
            va="bottom",
            ha="left",
        )

    # Formatting
    ax.set_xlim(asri.index.min(), asri.index.max())
    ax.set_ylim(0, 100)
    ax.set_xlabel("Date")
    ax.set_ylabel("ASRI Value")
    ax.set_title(title)

    # Date formatting
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))

    # Legend - combine ASRI line with alert bands
    handles = [
        plt.Line2D([0], [0], color=FARZULLA_BURGUNDY, linewidth=1.5, label="ASRI"),
        mpatches.Patch(color=ALERT_COLORS["low"], alpha=0.3, label="Low (<30)"),
        mpatches.Patch(color=ALERT_COLORS["elevated"], alpha=0.3, label="Elevated (30-50)"),
        mpatches.Patch(color=ALERT_COLORS["high"], alpha=0.3, label="High (50-70)"),
        mpatches.Patch(color=ALERT_COLORS["critical"], alpha=0.3, label="Critical (>70)"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9, ncol=2)

    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_sub_index_decomposition(
    sub_indices: "pd.DataFrame",
    weights: dict[str, float],
    output_path: str = "figures/decomposition.pdf",
    title: str = "ASRI Sub-Index Decomposition",
) -> None:
    """
    Stacked area chart showing weighted contribution of each sub-index.

    Parameters
    ----------
    sub_indices : pd.DataFrame
        DataFrame with sub-index values (columns) indexed by datetime.
    weights : dict[str, float]
        Weights for each sub-index (should match column names).
    output_path : str
        Path for saving the figure.
    title : str
        Figure title.
    """
    _setup_style()
    output_path = _ensure_output_dir(output_path)

    # Compute weighted contributions
    weighted = sub_indices.copy()
    for col in weighted.columns:
        if col in weights:
            weighted[col] = weighted[col] * weights[col]

    # Color palette - burgundy gradient plus complementary colors
    n_indices = len(weighted.columns)
    colors = sns.color_palette(
        [FARZULLA_BURGUNDY, "#4A6FA5", "#6B4A8E", "#8B6914", "#2E8B57", "#708090"],
        n_colors=n_indices,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    # Stacked area
    ax.stackplot(
        weighted.index,
        weighted.T.values,
        labels=weighted.columns,
        colors=colors,
        alpha=0.8,
    )

    # Total ASRI line overlay
    total = weighted.sum(axis=1)
    ax.plot(
        total.index,
        total.values,
        color="black",
        linewidth=1.0,
        linestyle="--",
        label="Total ASRI",
        alpha=0.7,
    )

    # Formatting
    ax.set_xlim(weighted.index.min(), weighted.index.max())
    ax.set_ylim(0, max(100, total.max() * 1.05))
    ax.set_xlabel("Date")
    ax.set_ylabel("Weighted Contribution")
    ax.set_title(title)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.legend(loc="upper left", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, axis="y")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_regime_classification(
    regime_probs: np.ndarray,
    asri: "pd.Series",
    dates: "pd.DatetimeIndex",
    output_path: str = "figures/regimes.pdf",
    regime_names: list[str] | None = None,
    title: str = "ASRI Regime Classification",
) -> None:
    """
    Two-panel figure showing ASRI with regime shading and probability breakdown.

    Parameters
    ----------
    regime_probs : np.ndarray
        Array of shape (T, n_regimes) with regime probabilities.
    asri : pd.Series
        ASRI values indexed by datetime.
    dates : pd.DatetimeIndex
        Date index corresponding to regime_probs rows.
    output_path : str
        Path for saving the figure.
    regime_names : list[str], optional
        Names for each regime. Defaults to ["Tranquil", "Transitional", "Crisis"].
    title : str
        Figure title.
    """
    _setup_style()
    output_path = _ensure_output_dir(output_path)

    if regime_names is None:
        regime_names = ["Tranquil", "Transitional", "Crisis"]

    n_regimes = regime_probs.shape[1]
    regime_colors = [
        REGIME_COLORS.get(name.lower(), "#808080")
        for name in regime_names
    ]

    # Determine dominant regime at each point
    dominant = np.argmax(regime_probs, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[2, 1])

    # ----- Top Panel: ASRI with regime background -----
    ax1 = axes[0]

    # Background shading by dominant regime
    for i in range(len(dates) - 1):
        ax1.axvspan(
            dates[i],
            dates[i + 1],
            alpha=0.25,
            color=regime_colors[dominant[i]],
            linewidth=0,
        )

    # ASRI line
    ax1.plot(
        asri.index,
        asri.values,
        color=FARZULLA_BURGUNDY,
        linewidth=1.5,
        label="ASRI",
        zorder=5,
    )

    ax1.set_ylabel("ASRI Value")
    ax1.set_ylim(0, 100)
    ax1.set_title(title)

    # Regime legend
    handles = [
        mpatches.Patch(color=regime_colors[i], alpha=0.4, label=regime_names[i])
        for i in range(n_regimes)
    ]
    handles.insert(0, plt.Line2D([0], [0], color=FARZULLA_BURGUNDY, linewidth=1.5, label="ASRI"))
    ax1.legend(handles=handles, loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

    # ----- Bottom Panel: Stacked regime probabilities -----
    ax2 = axes[1]

    ax2.stackplot(
        dates,
        regime_probs.T,
        labels=regime_names,
        colors=regime_colors,
        alpha=0.7,
    )

    ax2.set_xlim(dates.min(), dates.max())
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Regime Probability")

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2.legend(loc="upper left", framealpha=0.9, ncol=3)
    ax2.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, axis="y")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_sensitivity_heatmaps(
    weight_results: list[dict],
    output_path: str = "figures/sensitivity.pdf",
    title: str = "ASRI Sensitivity to Weight Perturbations",
) -> None:
    """
    Heatmap showing ASRI stability across weight perturbations.

    Parameters
    ----------
    weight_results : list[dict]
        List of dicts with keys:
        - 'sub_index': str, name of perturbed sub-index
        - 'perturbation': float, perturbation magnitude (-0.2 to +0.2)
        - 'mean_asri': float, mean ASRI under this weight scheme
        - 'std_asri': float, std of ASRI under this weight scheme
        - 'max_asri': float, max ASRI under this weight scheme
    output_path : str
        Path for saving the figure.
    title : str
        Figure title.
    """
    _setup_style()
    output_path = _ensure_output_dir(output_path)

    # Pivot to matrix form
    import pandas as pd

    df = pd.DataFrame(weight_results)

    # Create pivot tables for mean and std
    mean_pivot = df.pivot(index="sub_index", columns="perturbation", values="mean_asri")
    std_pivot = df.pivot(index="sub_index", columns="perturbation", values="std_asri")

    # Custom colormap: burgundy gradient
    burgundy_cmap = LinearSegmentedColormap.from_list(
        "burgundy",
        ["#FFFFFF", FARZULLA_BURGUNDY_LIGHT, FARZULLA_BURGUNDY, FARZULLA_BURGUNDY_DARK],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ----- Left: Mean ASRI -----
    ax1 = axes[0]
    sns.heatmap(
        mean_pivot,
        ax=ax1,
        cmap=burgundy_cmap,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Mean ASRI"},
    )
    ax1.set_title("Mean ASRI by Weight Perturbation")
    ax1.set_xlabel("Weight Perturbation")
    ax1.set_ylabel("Sub-Index")

    # ----- Right: Std ASRI -----
    ax2 = axes[1]
    sns.heatmap(
        std_pivot,
        ax=ax2,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Std ASRI"},
    )
    ax2.set_title("ASRI Volatility by Weight Perturbation")
    ax2.set_xlabel("Weight Perturbation")
    ax2.set_ylabel("Sub-Index")

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_event_study_panels(
    event_results: list[dict],
    output_path: str = "figures/event_study.pdf",
    window: int = 30,
    title: str = "ASRI Event Study: Crisis Episodes",
) -> None:
    """
    4-panel figure showing ASRI trajectory around each crisis event.

    Parameters
    ----------
    event_results : list[dict]
        List of dicts with keys:
        - 'name': str, crisis event name
        - 'date': datetime, event date
        - 'asri_window': pd.Series, ASRI values around event
        - 'pre_mean': float, mean ASRI before event
        - 'post_mean': float, mean ASRI after event
    output_path : str
        Path for saving the figure.
    window : int
        Number of days before/after event in window.
    title : str
        Figure title.
    """
    _setup_style()
    output_path = _ensure_output_dir(output_path)

    n_events = len(event_results)
    n_cols = 2
    n_rows = (n_events + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, event in enumerate(event_results):
        ax = axes[i]

        asri_window = event["asri_window"]
        event_date = event["date"]
        name = event["name"]
        pre_mean = event.get("pre_mean", asri_window[:window].mean())
        post_mean = event.get("post_mean", asri_window[-window:].mean())

        # Convert to relative days
        if hasattr(asri_window.index, "to_pydatetime"):
            days = [(d - event_date).days for d in asri_window.index]
        else:
            days = list(range(-window, window + 1))

        # Plot ASRI
        ax.plot(days, asri_window.values, color=FARZULLA_BURGUNDY, linewidth=1.5)

        # Event line
        ax.axvline(0, color=FARZULLA_BURGUNDY_DARK, linestyle="--", linewidth=1.0, alpha=0.8)

        # Pre/post mean lines
        ax.axhline(pre_mean, color="#2E8B57", linestyle=":", linewidth=1.0, alpha=0.7, label=f"Pre: {pre_mean:.1f}")
        ax.axhline(post_mean, color="#DC143C", linestyle=":", linewidth=1.0, alpha=0.7, label=f"Post: {post_mean:.1f}")

        # Shading for pre/post periods
        ax.axvspan(-window, 0, alpha=0.08, color="#2E8B57")
        ax.axvspan(0, window, alpha=0.08, color="#DC143C")

        ax.set_xlim(-window, window)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Days from Event")
        ax.set_ylabel("ASRI")
        ax.set_title(name, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


# =============================================================================
# Utility Functions
# =============================================================================


def create_all_figures(
    asri: "pd.Series",
    sub_indices: "pd.DataFrame",
    weights: dict[str, float],
    regime_probs: np.ndarray,
    crisis_dates: list[datetime],
    crisis_labels: list[str],
    weight_results: list[dict],
    event_results: list[dict],
    output_dir: str = "figures/",
) -> None:
    """
    Generate all publication figures in one call.

    Parameters
    ----------
    asri : pd.Series
        ASRI time series.
    sub_indices : pd.DataFrame
        Sub-index values.
    weights : dict[str, float]
        Sub-index weights.
    regime_probs : np.ndarray
        Regime probabilities (T x n_regimes).
    crisis_dates : list[datetime]
        Crisis event dates.
    crisis_labels : list[str]
        Crisis event labels.
    weight_results : list[dict]
        Sensitivity analysis results.
    event_results : list[dict]
        Event study results.
    output_dir : str
        Output directory for figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating ASRI publication figures...")

    plot_asri_time_series(
        asri=asri,
        crisis_dates=crisis_dates,
        crisis_labels=crisis_labels,
        output_path=str(output_dir / "asri_timeseries.pdf"),
    )

    plot_sub_index_decomposition(
        sub_indices=sub_indices,
        weights=weights,
        output_path=str(output_dir / "decomposition.pdf"),
    )

    plot_regime_classification(
        regime_probs=regime_probs,
        asri=asri,
        dates=asri.index,
        output_path=str(output_dir / "regimes.pdf"),
    )

    plot_sensitivity_heatmaps(
        weight_results=weight_results,
        output_path=str(output_dir / "sensitivity.pdf"),
    )

    plot_event_study_panels(
        event_results=event_results,
        output_path=str(output_dir / "event_study.pdf"),
    )

    print(f"All figures saved to {output_dir}/")


if __name__ == "__main__":
    # Demo with synthetic data
    import pandas as pd

    print("Running figure generation demo with synthetic data...")

    # Synthetic ASRI
    dates = pd.date_range("2010-01-01", "2023-12-31", freq="D")
    np.random.seed(42)
    base = 35 + 15 * np.sin(np.linspace(0, 6 * np.pi, len(dates)))
    noise = np.random.normal(0, 5, len(dates))
    asri = pd.Series(np.clip(base + noise, 0, 100), index=dates, name="ASRI")

    # Add crisis spikes
    crisis_dates = [
        datetime(2011, 8, 5),   # US Downgrade
        datetime(2015, 8, 24),  # China Crash
        datetime(2020, 3, 16),  # COVID
        datetime(2022, 3, 7),   # Ukraine
    ]
    crisis_labels = ["US Downgrade", "China Crash", "COVID-19", "Ukraine Invasion"]

    for crisis in crisis_dates:
        mask = (dates >= crisis) & (dates < crisis + pd.Timedelta(days=60))
        asri[mask] += np.linspace(30, 0, mask.sum())

    asri = asri.clip(0, 100)

    # Synthetic sub-indices
    sub_indices = pd.DataFrame({
        "Credit": np.random.uniform(20, 60, len(dates)),
        "Equity": np.random.uniform(25, 55, len(dates)),
        "Funding": np.random.uniform(15, 50, len(dates)),
        "Contagion": np.random.uniform(10, 45, len(dates)),
    }, index=dates)

    weights = {"Credit": 0.30, "Equity": 0.25, "Funding": 0.25, "Contagion": 0.20}

    # Synthetic regime probs
    regime_probs = np.random.dirichlet([5, 2, 1], len(dates))

    # Synthetic sensitivity results
    weight_results = []
    for sub in weights.keys():
        for pert in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            weight_results.append({
                "sub_index": sub,
                "perturbation": pert,
                "mean_asri": 40 + pert * 10 + np.random.uniform(-2, 2),
                "std_asri": 12 + abs(pert) * 3,
                "max_asri": 75 + pert * 15,
            })

    # Synthetic event results
    event_results = []
    for name, date in zip(crisis_labels, crisis_dates):
        window_start = date - pd.Timedelta(days=30)
        window_end = date + pd.Timedelta(days=30)
        window_asri = asri[window_start:window_end]
        event_results.append({
            "name": name,
            "date": date,
            "asri_window": window_asri,
            "pre_mean": window_asri[:30].mean(),
            "post_mean": window_asri[-30:].mean(),
        })

    # Generate all figures
    create_all_figures(
        asri=asri,
        sub_indices=sub_indices,
        weights=weights,
        regime_probs=regime_probs,
        crisis_dates=crisis_dates,
        crisis_labels=crisis_labels,
        weight_results=weight_results,
        event_results=event_results,
        output_dir="figures/demo/",
    )
