# backtest/metrics.py
"""
Macro Metals System — Performance & Risk Metrics.

Pure-function library operating on daily return / equity series.
All functions assume a 252-trading-day year unless overridden.

Reference: macro_metals_system_memory.md §6.3

Performance targets (for reference):
    Sharpe (combined)   > 1.0
    Sharpe (per-strat)  > 0.5
    Max drawdown        < 15 %
    Annualised vol      8–12 %
    Calmar ratio        > 0.7
    Annualised turnover < 30×
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Return metrics
# ---------------------------------------------------------------------------

def annualized_return(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Annualised compounded return from a daily simple-return series.

    Args:
        returns: Daily simple returns.
        periods_per_year: Trading days per year.

    Returns:
        Annualised return as a decimal (0.12 → 12 %).
    """
    total = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def annualized_vol(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Annualised volatility (standard deviation × √periods).

    Args:
        returns: Daily simple returns.
        periods_per_year: Trading days per year.

    Returns:
        Annualised volatility as a decimal.
    """
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sharpe ratio.

    Args:
        returns: Daily simple returns.
        rf: Annualised risk-free rate (default 0).
        periods_per_year: Trading days per year.

    Returns:
        Sharpe ratio.  Returns 0.0 when volatility is zero.
    """
    vol = annualized_vol(returns, periods_per_year)
    if vol == 0:
        return 0.0
    return float((annualized_return(returns, periods_per_year) - rf) / vol)


# ---------------------------------------------------------------------------
# Drawdown metrics
# ---------------------------------------------------------------------------

def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown from an equity curve.

    Args:
        equity: Cumulative equity series (e.g. starting at 1 000 000).

    Returns:
        Max drawdown as a **positive** decimal (0.15 → 15 %).
    """
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return float(-dd.min())


def calmar_ratio(
    returns: pd.Series,
    equity: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calmar ratio: annualised return / max drawdown.

    Args:
        returns: Daily simple returns.
        equity:  Cumulative equity curve.
        periods_per_year: Trading days per year.

    Returns:
        Calmar ratio.  Returns 0.0 when drawdown is zero.
    """
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(annualized_return(returns, periods_per_year) / mdd)


# ---------------------------------------------------------------------------
# Hit rate & turnover
# ---------------------------------------------------------------------------

def hit_rate(returns: pd.Series) -> float:
    """Fraction of days with strictly positive returns.

    Args:
        returns: Daily simple returns.

    Returns:
        Hit rate as a decimal (0.53 → 53 %).
    """
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def annualized_turnover(
    turnover_series: pd.Series, periods_per_year: int = 252
) -> float:
    """Annualised turnover multiple from daily |Δposition|.

    Args:
        turnover_series: Daily absolute position change (fraction of capital).
        periods_per_year: Trading days per year.

    Returns:
        Annualised turnover multiple.
    """
    n_years = max(len(turnover_series) / periods_per_year, 1)
    return float(turnover_series.sum() / n_years)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summary(
    returns: pd.Series,
    equity: pd.Series,
    turnover: pd.Series | None = None,
) -> dict:
    """Compute all key performance metrics in one call.

    Args:
        returns:  Daily simple returns.
        equity:   Cumulative equity curve.
        turnover: Optional daily turnover series.

    Returns:
        Dict with keys matching the §6.3 performance targets.
    """
    stats: dict = {
        "annualized_return": annualized_return(returns),
        "annualized_vol": annualized_vol(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "calmar_ratio": calmar_ratio(returns, equity),
        "hit_rate": hit_rate(returns),
    }
    if turnover is not None:
        stats["annualized_turnover"] = annualized_turnover(turnover)
    return stats
