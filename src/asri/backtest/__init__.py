"""ASRI Backtesting Module - Historical validation and backfill."""

from .backtest import ASRIBacktester, CrisisEvent, BacktestResult
from .historical import HistoricalDataFetcher
from .publication_lag import (
    LagAwareBacktester,
    LagAwareHistoricalFetcher,
    LagComparisonResult,
    DATA_LAGS,
    compare_lag_impact,
    format_lag_comparison_table,
)

__all__ = [
    "ASRIBacktester",
    "CrisisEvent",
    "BacktestResult",
    "HistoricalDataFetcher",
    "LagAwareBacktester",
    "LagAwareHistoricalFetcher",
    "LagComparisonResult",
    "DATA_LAGS",
    "compare_lag_impact",
    "format_lag_comparison_table",
]
