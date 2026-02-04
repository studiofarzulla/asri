"""
Walk-Forward Validation for ASRI

Proper out-of-sample testing that respects temporal ordering.
Standard k-fold cross-validation is invalid for time series
because it allows "future" data to inform "past" predictions.

Key features:
1. Purged gap between train and test to prevent leakage
2. Expanding window (more data over time) or rolling window
3. Walk-forward optimization of weights
"""

from dataclasses import dataclass, field
from typing import Generator, Literal

import numpy as np
import pandas as pd


@dataclass
class WalkForwardFold:
    """Results from a single walk-forward fold."""
    fold_number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    # Performance metrics
    train_r2: float
    test_r2: float
    test_auc: float | None
    test_mse: float
    
    # Weights used (if optimized)
    weights: dict[str, float] | None = None


@dataclass
class WalkForwardResult:
    """Aggregate results from walk-forward validation."""
    n_folds: int
    folds: list[WalkForwardFold]
    
    # Aggregate metrics
    mean_test_r2: float
    std_test_r2: float
    mean_test_auc: float | None
    mean_test_mse: float
    
    # Consistency
    weight_stability: float  # Correlation of weights across folds
    
    # Combined out-of-sample predictions
    oos_predictions: pd.Series | None = None
    oos_actuals: pd.Series | None = None


class PurgedTimeSeriesSplit:
    """
    Time series cross-validation with purge gap.
    
    The purge gap prevents information leakage from test data
    bleeding into the training data via lagged features.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 30,
        embargo_days: int = 0,
        min_train_days: int = 365,
        expanding: bool = True,
    ):
        """
        Args:
            n_splits: Number of folds
            purge_days: Gap between train end and test start
            embargo_days: Additional gap after test (for sequential folds)
            min_train_days: Minimum training set size
            expanding: If True, training window expands; if False, rolls
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.min_train_days = min_train_days
        self.expanding = expanding
    
    def split(
        self,
        data: pd.DataFrame | pd.Series,
    ) -> Generator[tuple[np.ndarray, np.ndarray, dict], None, None]:
        """
        Generate train/test indices.
        
        Yields:
            (train_indices, test_indices, metadata_dict)
        """
        if isinstance(data, pd.Series):
            index = data.index
        else:
            index = data.index
        
        n = len(index)
        
        # Calculate test set size
        # Leave room for min_train_days + purge + at least some test
        available_for_test = n - self.min_train_days - self.purge_days
        if available_for_test < self.n_splits:
            raise ValueError(
                f"Insufficient data: {n} days, need at least "
                f"{self.min_train_days + self.purge_days + self.n_splits}"
            )
        
        test_size = available_for_test // self.n_splits
        
        for fold in range(self.n_splits):
            # Test window
            test_end_idx = n - fold * test_size
            test_start_idx = test_end_idx - test_size
            
            # Ensure test doesn't go below minimum
            if test_start_idx < self.min_train_days + self.purge_days:
                continue
            
            # Training window (with purge)
            train_end_idx = test_start_idx - self.purge_days
            
            if self.expanding:
                train_start_idx = 0
            else:
                # Rolling window: same size as all prior data up to this point
                window_size = min(train_end_idx, self.min_train_days * 2)
                train_start_idx = max(0, train_end_idx - window_size)
            
            if train_end_idx - train_start_idx < self.min_train_days:
                continue
            
            train_idx = np.arange(train_start_idx, train_end_idx)
            test_idx = np.arange(test_start_idx, test_end_idx)
            
            metadata = {
                'fold': fold,
                'train_start': index[train_start_idx],
                'train_end': index[train_end_idx - 1],
                'test_start': index[test_start_idx],
                'test_end': index[test_end_idx - 1],
                'purge_days': self.purge_days,
            }
            
            yield train_idx, test_idx, metadata


def purged_walk_forward_cv(
    sub_indices: pd.DataFrame,
    target: pd.Series,
    weights: dict[str, float],
    n_splits: int = 5,
    purge_days: int = 30,
) -> WalkForwardResult:
    """
    Run walk-forward validation with fixed weights.
    
    Args:
        sub_indices: DataFrame with sub-index time series
        target: Target variable (e.g., forward drawdown)
        weights: Fixed weights to use
        n_splits: Number of folds
        purge_days: Gap between train and test
        
    Returns:
        WalkForwardResult with all fold results
    """
    # Align data
    common_idx = sub_indices.index.intersection(target.index)
    X = sub_indices.loc[common_idx]
    y = target.loc[common_idx]
    
    splitter = PurgedTimeSeriesSplit(
        n_splits=n_splits,
        purge_days=purge_days,
    )
    
    folds = []
    oos_pred_list = []
    oos_actual_list = []
    
    for train_idx, test_idx, meta in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Compute ASRI with fixed weights
        asri_train = sum(weights.get(c, 0) * X_train[c] for c in X.columns)
        asri_test = sum(weights.get(c, 0) * X_test[c] for c in X.columns)
        
        # Metrics
        train_r2 = _compute_r2(asri_train.values, y_train.values)
        test_r2 = _compute_r2(asri_test.values, y_test.values)
        test_mse = np.mean((asri_test.values - y_test.values) ** 2)
        
        # AUC if target is binary-ish
        if len(np.unique(y_test.values)) <= 10:
            test_auc = _compute_spearman_auc(asri_test.values, y_test.values)
        else:
            test_auc = None
        
        folds.append(WalkForwardFold(
            fold_number=meta['fold'],
            train_start=meta['train_start'],
            train_end=meta['train_end'],
            test_start=meta['test_start'],
            test_end=meta['test_end'],
            train_r2=train_r2,
            test_r2=test_r2,
            test_auc=test_auc,
            test_mse=test_mse,
            weights=weights,
        ))
        
        # Collect OOS predictions
        oos_pred_list.extend(zip(X_test.index, asri_test.values))
        oos_actual_list.extend(zip(X_test.index, y_test.values))
    
    # Aggregate metrics
    mean_r2 = np.mean([f.test_r2 for f in folds])
    std_r2 = np.std([f.test_r2 for f in folds])
    mean_mse = np.mean([f.test_mse for f in folds])
    
    aucs = [f.test_auc for f in folds if f.test_auc is not None]
    mean_auc = np.mean(aucs) if aucs else None
    
    # Weight stability (all same weights, so perfect)
    weight_stability = 1.0
    
    # OOS predictions as series
    if oos_pred_list:
        oos_pred = pd.Series(
            dict(oos_pred_list)
        ).sort_index()
        oos_actual = pd.Series(
            dict(oos_actual_list)
        ).sort_index()
    else:
        oos_pred = None
        oos_actual = None
    
    return WalkForwardResult(
        n_folds=len(folds),
        folds=folds,
        mean_test_r2=mean_r2,
        std_test_r2=std_r2,
        mean_test_auc=mean_auc,
        mean_test_mse=mean_mse,
        weight_stability=weight_stability,
        oos_predictions=oos_pred,
        oos_actuals=oos_actual,
    )


def walk_forward_optimization(
    sub_indices: pd.DataFrame,
    target: pd.Series,
    n_splits: int = 5,
    purge_days: int = 30,
    optimization_method: Literal["elastic_net", "pca"] = "elastic_net",
) -> WalkForwardResult:
    """
    Walk-forward validation with weight re-optimization at each fold.
    
    This tests whether empirically-derived weights are stable and
    generalize out-of-sample.
    
    Args:
        sub_indices: DataFrame with sub-index time series
        target: Target variable
        n_splits: Number of folds
        purge_days: Gap between train and test
        optimization_method: How to derive weights ('elastic_net' or 'pca')
        
    Returns:
        WalkForwardResult with fold-specific weights
    """
    from ..weights.pca import PCAWeightDeriver
    from ..weights.elastic_net import ElasticNetWeightDeriver
    
    common_idx = sub_indices.index.intersection(target.index)
    X = sub_indices.loc[common_idx]
    y = target.loc[common_idx]
    
    splitter = PurgedTimeSeriesSplit(
        n_splits=n_splits,
        purge_days=purge_days,
    )
    
    folds = []
    all_weights = []
    oos_pred_list = []
    oos_actual_list = []
    
    for train_idx, test_idx, meta in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Optimize weights on training data
        if optimization_method == "elastic_net":
            deriver = ElasticNetWeightDeriver()
            deriver.fit(X_train, y_train)
            weights = deriver.weights
        else:  # pca
            deriver = PCAWeightDeriver()
            deriver.fit(X_train)
            weights = deriver.weights
        
        all_weights.append(weights)
        
        # Compute ASRI with optimized weights
        asri_train = sum(weights.get(c, 0) * X_train[c] for c in X.columns)
        asri_test = sum(weights.get(c, 0) * X_test[c] for c in X.columns)
        
        # Metrics
        train_r2 = _compute_r2(asri_train.values, y_train.values)
        test_r2 = _compute_r2(asri_test.values, y_test.values)
        test_mse = np.mean((asri_test.values - y_test.values) ** 2)
        
        if len(np.unique(y_test.values)) <= 10:
            test_auc = _compute_spearman_auc(asri_test.values, y_test.values)
        else:
            test_auc = None
        
        folds.append(WalkForwardFold(
            fold_number=meta['fold'],
            train_start=meta['train_start'],
            train_end=meta['train_end'],
            test_start=meta['test_start'],
            test_end=meta['test_end'],
            train_r2=train_r2,
            test_r2=test_r2,
            test_auc=test_auc,
            test_mse=test_mse,
            weights=weights,
        ))
        
        oos_pred_list.extend(zip(X_test.index, asri_test.values))
        oos_actual_list.extend(zip(X_test.index, y_test.values))
    
    # Compute weight stability
    weight_stability = _compute_weight_stability(all_weights)
    
    # Aggregate metrics
    mean_r2 = np.mean([f.test_r2 for f in folds])
    std_r2 = np.std([f.test_r2 for f in folds])
    mean_mse = np.mean([f.test_mse for f in folds])
    
    aucs = [f.test_auc for f in folds if f.test_auc is not None]
    mean_auc = np.mean(aucs) if aucs else None
    
    if oos_pred_list:
        oos_pred = pd.Series(dict(oos_pred_list)).sort_index()
        oos_actual = pd.Series(dict(oos_actual_list)).sort_index()
    else:
        oos_pred = None
        oos_actual = None
    
    return WalkForwardResult(
        n_folds=len(folds),
        folds=folds,
        mean_test_r2=mean_r2,
        std_test_r2=std_r2,
        mean_test_auc=mean_auc,
        mean_test_mse=mean_mse,
        weight_stability=weight_stability,
        oos_predictions=oos_pred,
        oos_actuals=oos_actual,
    )


def _compute_r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


def _compute_spearman_auc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Spearman correlation as AUC proxy."""
    from scipy import stats
    corr, _ = stats.spearmanr(y_pred, y_true)
    return abs(corr)


def _compute_weight_stability(weights_list: list[dict]) -> float:
    """Compute average pairwise correlation of weight vectors."""
    if len(weights_list) < 2:
        return 1.0
    
    # Get common keys
    common_keys = set(weights_list[0].keys())
    for w in weights_list[1:]:
        common_keys &= set(w.keys())
    common_keys = sorted(common_keys)
    
    # Convert to array
    weight_matrix = np.array([
        [w[k] for k in common_keys] for w in weights_list
    ])
    
    # Pairwise correlations
    n = len(weights_list)
    correlations = []
    for i in range(n):
        for j in range(i + 1, n):
            corr = np.corrcoef(weight_matrix[i], weight_matrix[j])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    return np.mean(correlations) if correlations else 0


def format_walk_forward_table(result: WalkForwardResult) -> str:
    """Format walk-forward results as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Walk-Forward Validation Results}",
        r"\label{tab:walk_forward}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Fold & Train Period & Test Period & Train $R^2$ & Test $R^2$ & Test AUC \\",
        r"\midrule",
    ]
    
    for fold in result.folds:
        train_period = f"{fold.train_start.strftime('%Y-%m')} -- {fold.train_end.strftime('%Y-%m')}"
        test_period = f"{fold.test_start.strftime('%Y-%m')} -- {fold.test_end.strftime('%Y-%m')}"
        auc_str = f"{fold.test_auc:.3f}" if fold.test_auc else "--"
        
        lines.append(
            f"{fold.fold_number + 1} & {train_period} & {test_period} & "
            f"{fold.train_r2:.3f} & {fold.test_r2:.3f} & {auc_str} \\\\"
        )
    
    auc_str = f"{result.mean_test_auc:.3f}" if result.mean_test_auc else "--"
    
    lines.extend([
        r"\midrule",
        f"\\textbf{{Mean}} & -- & -- & -- & "
        f"\\textbf{{{result.mean_test_r2:.3f}}} $\\pm$ {result.std_test_r2:.3f} & "
        f"\\textbf{{{auc_str}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        f"\\item Weight stability (avg pairwise correlation): {result.weight_stability:.3f}",
        f"\\item Purge gap: 30 days between train and test.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)
