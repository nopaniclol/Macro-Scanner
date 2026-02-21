"""
Backtest Package

Backtesting engine for FX/Metals carry strategies.
"""

from .engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    TransactionCosts,
    TransactionCostModel,
    PerformanceMetrics,
    CrisisAnalysis,
    Attribution,
    CRISIS_PERIODS,
)

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'TransactionCosts',
    'TransactionCostModel',
    'PerformanceMetrics',
    'CrisisAnalysis',
    'Attribution',
    'CRISIS_PERIODS',
]
