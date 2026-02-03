"""
Backtest Engine for BQNT Carry Execution.

Evaluates strategy performance with comprehensive metrics.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from ..config.parameters import BACKTEST_PARAMS, EXECUTION_PARAMS


@dataclass
class BacktestResult:
    """Container for backtest results."""
    returns: pd.Series
    cumulative_returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]
    drawdowns: pd.Series


class BacktestEngine:
    """
    Backtesting engine for carry strategy.

    Features:
    - Transaction cost modeling
    - Drawdown analysis
    - Risk-adjusted metrics
    - Benchmark comparison
    """

    def __init__(
        self,
        initial_capital: float = None,
        risk_free_rate: float = None,
        periods_per_year: int = None,
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            risk_free_rate: Risk-free rate for Sharpe calculation
            periods_per_year: Trading periods per year
        """
        self.initial_capital = initial_capital or BACKTEST_PARAMS['initial_capital']
        self.risk_free_rate = risk_free_rate or BACKTEST_PARAMS['risk_free_rate']
        self.periods_per_year = periods_per_year or BACKTEST_PARAMS['periods_per_year']

        self._fx_cost = EXECUTION_PARAMS['fx_cost_bps'] / 10000
        self._metal_cost = EXECUTION_PARAMS['metal_cost_bps'] / 10000

    def run_backtest(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Run backtest with given weights and returns.

        Args:
            weights: Portfolio weights (shifted for look-ahead bias)
            returns: Asset returns
            benchmark_returns: Benchmark returns (optional)

        Returns:
            BacktestResult with all metrics
        """
        # Align data
        common_dates = weights.index.intersection(returns.index)
        weights = weights.loc[common_dates]
        returns = returns.loc[common_dates]

        # Shift weights by 1 day (trade on signal, return next day)
        weights_shifted = weights.shift(1).dropna()
        returns_aligned = returns.loc[weights_shifted.index]

        # Calculate portfolio returns
        port_returns = (weights_shifted * returns_aligned).sum(axis=1)

        # Calculate transaction costs
        turnover = self._calculate_turnover(weights_shifted)
        costs = turnover * self._fx_cost  # Simplified: all FX cost
        port_returns_net = port_returns - costs

        # Calculate metrics
        metrics = self._calculate_metrics(port_returns_net, benchmark_returns)

        # Calculate drawdowns
        cumulative = (1 + port_returns_net).cumprod()
        drawdowns = self._calculate_drawdowns(cumulative)

        # Track trades
        trades = self._generate_trade_log(weights_shifted)

        return BacktestResult(
            returns=port_returns_net,
            cumulative_returns=cumulative,
            positions=weights_shifted,
            trades=trades,
            metrics=metrics,
            drawdowns=drawdowns,
        )

    def _calculate_turnover(self, weights: pd.DataFrame) -> pd.Series:
        """Calculate portfolio turnover."""
        weight_changes = weights.diff().abs()
        turnover = weight_changes.sum(axis=1) / 2  # Divide by 2 for round-trip
        return turnover

    def _calculate_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns (optional)

        Returns:
            Dict of metrics
        """
        # Basic stats
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (self.periods_per_year / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(self.periods_per_year)

        # Sharpe ratio
        excess_return = ann_return - self.risk_free_rate
        sharpe = excess_return / ann_vol if ann_vol > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(self.periods_per_year)
        sortino = excess_return / downside_vol if downside_vol > 0 else 0

        # Drawdown stats
        cumulative = (1 + returns).cumprod()
        drawdowns = self._calculate_drawdowns(cumulative)
        max_dd = drawdowns.min()
        avg_dd = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0

        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd < 0 else 0

        # Win rate
        win_rate = (returns > 0).sum() / len(returns)

        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        metrics = {
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_periods': len(returns),
        }

        # Benchmark comparison
        if benchmark_returns is not None:
            bench_aligned = benchmark_returns.reindex(returns.index).fillna(0)
            bench_total = (1 + bench_aligned).prod() - 1
            bench_ann = (1 + bench_total) ** (self.periods_per_year / len(bench_aligned)) - 1
            bench_vol = bench_aligned.std() * np.sqrt(self.periods_per_year)

            # Information ratio
            tracking_error = (returns - bench_aligned).std() * np.sqrt(self.periods_per_year)
            info_ratio = (ann_return - bench_ann) / tracking_error if tracking_error > 0 else 0

            # Beta and alpha
            covariance = np.cov(returns, bench_aligned)[0, 1]
            bench_variance = bench_aligned.var()
            beta = covariance / bench_variance if bench_variance > 0 else 0
            alpha = ann_return - (self.risk_free_rate + beta * (bench_ann - self.risk_free_rate))

            metrics.update({
                'benchmark_return': bench_ann,
                'excess_return': ann_return - bench_ann,
                'tracking_error': tracking_error,
                'information_ratio': info_ratio,
                'beta': beta,
                'alpha': alpha,
            })

        return metrics

    def _calculate_drawdowns(self, cumulative: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns

    def _generate_trade_log(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Generate trade log from weight changes."""
        trades = []

        for date in weights.index[1:]:
            prev_date = weights.index[weights.index.get_loc(date) - 1]

            for asset in weights.columns:
                prev_weight = weights.loc[prev_date, asset]
                curr_weight = weights.loc[date, asset]
                change = curr_weight - prev_weight

                if abs(change) > 0.001:  # Threshold for trade
                    trades.append({
                        'date': date,
                        'asset': asset,
                        'prev_weight': prev_weight,
                        'new_weight': curr_weight,
                        'change': change,
                        'direction': 'BUY' if change > 0 else 'SELL',
                    })

        return pd.DataFrame(trades)

    def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Args:
            returns: Strategy returns
            window: Rolling window size

        Returns:
            DataFrame with rolling metrics
        """
        rolling = pd.DataFrame(index=returns.index)

        # Rolling return
        rolling['return'] = returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1
        )

        # Rolling volatility
        rolling['volatility'] = returns.rolling(window).std() * np.sqrt(252)

        # Rolling Sharpe
        rolling['sharpe'] = (
            rolling['return'] - self.risk_free_rate * (window / 252)
        ) / rolling['volatility']

        # Rolling max drawdown
        def rolling_max_dd(x):
            cum = (1 + x).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            return dd.min()

        rolling['max_drawdown'] = returns.rolling(window).apply(rolling_max_dd)

        return rolling.dropna()

    def attribution_analysis(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate return attribution by asset.

        Args:
            weights: Portfolio weights
            returns: Asset returns

        Returns:
            DataFrame with attribution
        """
        # Align data
        common_dates = weights.index.intersection(returns.index)
        weights = weights.loc[common_dates]
        returns = returns.loc[common_dates]

        # Shift weights
        weights_shifted = weights.shift(1).dropna()
        returns_aligned = returns.loc[weights_shifted.index]

        # Calculate contribution
        contribution = weights_shifted * returns_aligned

        # Summarize
        summary = pd.DataFrame({
            'avg_weight': weights_shifted.mean(),
            'avg_return': returns_aligned.mean() * 252,  # Annualized
            'contribution': contribution.sum(),
            'contribution_pct': contribution.sum() / contribution.sum().sum() * 100,
        })

        return summary.sort_values('contribution', ascending=False)


def quick_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame
) -> Dict[str, float]:
    """
    Quick backtest returning key metrics only.

    Args:
        weights: Portfolio weights
        returns: Asset returns

    Returns:
        Dict with key metrics
    """
    engine = BacktestEngine()
    result = engine.run_backtest(weights, returns)

    return {
        'sharpe': result.metrics['sharpe_ratio'],
        'return': result.metrics['ann_return'],
        'vol': result.metrics['ann_volatility'],
        'max_dd': result.metrics['max_drawdown'],
    }
