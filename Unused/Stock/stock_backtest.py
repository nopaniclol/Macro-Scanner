"""
Backtest Engine for EGM/PEAD Strategy

Provides backtesting framework for:
- Historical gap + earnings signal testing
- Walk-forward validation
- Performance metrics calculation
- Risk-adjusted returns analysis
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .gap_detector import GapDetector, GapEvent
from .technical_analyzer import TechnicalAnalyzer
from .setup_analyzer import SetupAnalyzer, EXIT_RULES


# =============================================================================
# CONFIGURATION
# =============================================================================

BACKTEST_CONFIG = {
    'initial_capital': 100_000.0,
    'max_position_size_pct': 10.0,    # Max 10% per position
    'max_positions': 5,               # Max 5 concurrent positions
    'commission_per_trade': 1.0,      # $1 per trade
    'slippage_pct': 0.10,             # 0.1% slippage
    'risk_free_rate': 0.05,           # 5% annual risk-free rate
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Trade:
    """Represents a single trade."""
    ticker: str
    entry_date: datetime
    entry_price: float
    shares: int
    position_value: float
    stop_loss: float
    target_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'stop', 'target', 'time', 'signal'
    pnl: float = 0.0
    pnl_percent: float = 0.0
    holding_days: int = 0
    commission: float = 0.0


@dataclass
class BacktestMetrics:
    """Performance metrics from backtest."""
    total_return_pct: float
    annualized_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    avg_holding_days: float
    best_trade_pct: float
    worst_trade_pct: float
    final_equity: float
    start_date: datetime
    end_date: datetime


@dataclass
class BacktestResult:
    """Complete backtest results."""
    metrics: BacktestMetrics
    trades: List[Trade]
    equity_curve: pd.DataFrame
    daily_returns: pd.Series
    monthly_returns: pd.DataFrame
    config: Dict


# =============================================================================
# BACKTEST ENGINE CLASS
# =============================================================================

class BacktestEngine:
    """
    Backtests EGM/PEAD strategy on historical data.

    Simulates trading based on:
    - Gap detection signals
    - Entry triggers (breakout or fishhook)
    - Exit rules (stop, target, time, MA-based)
    """

    def __init__(self, initial_capital: float = None,
                 max_position_pct: float = None,
                 max_positions: int = None,
                 commission: float = None,
                 slippage: float = None):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            max_position_pct: Max position size as % of portfolio
            max_positions: Maximum concurrent positions
            commission: Commission per trade
            slippage: Slippage percentage
        """
        self.initial_capital = initial_capital or BACKTEST_CONFIG['initial_capital']
        self.max_position_pct = max_position_pct or BACKTEST_CONFIG['max_position_size_pct']
        self.max_positions = max_positions or BACKTEST_CONFIG['max_positions']
        self.commission = commission or BACKTEST_CONFIG['commission_per_trade']
        self.slippage = slippage or BACKTEST_CONFIG['slippage_pct']

        self.gap_detector = GapDetector()
        self.technical_analyzer = TechnicalAnalyzer()
        self.setup_analyzer = SetupAnalyzer()

    # =========================================================================
    # SIGNAL GENERATION FOR BACKTEST
    # =========================================================================

    def find_historical_gaps(self, df: pd.DataFrame,
                             ticker: str) -> List[GapEvent]:
        """
        Find all qualifying gaps in historical data.

        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol

        Returns:
            List of GapEvent objects
        """
        gaps = []
        gap_analysis = self.gap_detector.analyze(df, ticker)
        return gap_analysis.gaps

    def calculate_entry_exit(self, df: pd.DataFrame,
                             gap: GapEvent,
                             lookforward: int = 60) -> Dict:
        """
        Calculate entry and exit points after a gap.

        Args:
            df: DataFrame with OHLCV data
            gap: GapEvent object
            lookforward: Maximum days to look forward

        Returns:
            Dictionary with entry/exit details
        """
        gap_idx = df.index.get_loc(gap.gap_date) if gap.gap_date in df.index else None

        if gap_idx is None:
            return {}

        # Get post-gap data
        post_gap = df.iloc[gap_idx + 1:gap_idx + 1 + lookforward]

        if len(post_gap) == 0:
            return {}

        # Add technical indicators
        df_with_ma = self.technical_analyzer.add_moving_averages(df)
        post_gap_ma = df_with_ma.iloc[gap_idx + 1:gap_idx + 1 + lookforward]

        # Default entry: Day after gap at open
        entry_date = post_gap.index[0]
        entry_price = post_gap.iloc[0]['open']

        # Apply slippage
        entry_price = entry_price * (1 + self.slippage / 100)

        # Calculate stop and target
        gap_high = gap.high_price
        sma_20 = post_gap_ma.iloc[0].get('sma_20', entry_price * 0.9)

        stop_loss = max(sma_20, gap_high * 0.85)  # 15% below gap high or 20-SMA
        target_price = entry_price * 1.20  # 20% target

        # Find exit
        exit_date = None
        exit_price = None
        exit_reason = None

        max_hold_days = EXIT_RULES['time_stop']['max_hold_days']

        for i, (date, row) in enumerate(post_gap.iterrows()):
            # Check stop loss
            if row['low'] <= stop_loss:
                exit_date = date
                exit_price = stop_loss
                exit_reason = 'stop'
                break

            # Check target
            if row['high'] >= target_price:
                exit_date = date
                exit_price = target_price
                exit_reason = 'target'
                break

            # Check 10-SMA exit (partial would be handled separately)
            sma_10 = post_gap_ma.loc[date].get('sma_10', 0) if date in post_gap_ma.index else 0
            sma_20 = post_gap_ma.loc[date].get('sma_20', 0) if date in post_gap_ma.index else 0

            if sma_20 > 0 and row['close'] < sma_20:
                exit_date = date
                exit_price = row['close']
                exit_reason = 'below_20sma'
                break

            # Check time stop
            if i >= max_hold_days:
                exit_date = date
                exit_price = row['close']
                exit_reason = 'time'
                break

        # If no exit found, use last day
        if exit_date is None and len(post_gap) > 0:
            exit_date = post_gap.index[-1]
            exit_price = post_gap.iloc[-1]['close']
            exit_reason = 'end_of_data'

        # Apply slippage to exit
        if exit_price and exit_reason != 'target':
            exit_price = exit_price * (1 - self.slippage / 100)

        return {
            'entry_date': entry_date,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_price': target_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
        }

    # =========================================================================
    # BACKTEST EXECUTION
    # =========================================================================

    def run_backtest(self, data: Dict[str, pd.DataFrame],
                     start_date: datetime = None,
                     end_date: datetime = None,
                     verbose: bool = True) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: Dictionary mapping ticker to DataFrame
            start_date: Backtest start date
            end_date: Backtest end date
            verbose: Print progress

        Returns:
            BacktestResult with all metrics and trades
        """
        if verbose:
            print("\n" + "=" * 60)
            print("EGM/PEAD STRATEGY BACKTEST")
            print("=" * 60)

        # Initialize tracking
        equity = self.initial_capital
        equity_curve = []
        all_trades = []
        open_positions: Dict[str, Trade] = {}

        # Collect all gaps across all tickers
        all_signals = []

        for ticker, df in data.items():
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            gaps = self.find_historical_gaps(df, ticker)

            for gap in gaps:
                if gap.meets_criteria:
                    entry_exit = self.calculate_entry_exit(df, gap)
                    if entry_exit:
                        all_signals.append({
                            'ticker': ticker,
                            'gap': gap,
                            'df': df,
                            **entry_exit
                        })

        # Sort signals by entry date
        all_signals = sorted(all_signals, key=lambda x: x['entry_date'])

        if verbose:
            print(f"Found {len(all_signals)} qualifying signals")

        # Process signals chronologically
        for signal in all_signals:
            ticker = signal['ticker']

            # Check if we can open position
            if len(open_positions) >= self.max_positions:
                continue
            if ticker in open_positions:
                continue

            # Calculate position size
            position_value = equity * (self.max_position_pct / 100)
            shares = int(position_value / signal['entry_price'])

            if shares <= 0:
                continue

            # Create trade
            trade = Trade(
                ticker=ticker,
                entry_date=signal['entry_date'],
                entry_price=signal['entry_price'],
                shares=shares,
                position_value=shares * signal['entry_price'],
                stop_loss=signal['stop_loss'],
                target_price=signal['target_price'],
                exit_date=signal['exit_date'],
                exit_price=signal['exit_price'],
                exit_reason=signal['exit_reason'],
                commission=self.commission * 2,  # Entry + exit
            )

            # Calculate P&L
            if trade.exit_price:
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.shares - trade.commission
                trade.pnl_percent = ((trade.exit_price / trade.entry_price) - 1) * 100
                trade.holding_days = (trade.exit_date - trade.entry_date).days

            equity += trade.pnl
            all_trades.append(trade)

            equity_curve.append({
                'date': trade.exit_date or trade.entry_date,
                'equity': equity,
                'trade': ticker,
            })

        # Calculate metrics
        metrics = self._calculate_metrics(all_trades, equity, start_date, end_date)

        # Create equity curve DataFrame
        equity_df = pd.DataFrame(equity_curve)
        if len(equity_df) > 0:
            equity_df = equity_df.set_index('date').sort_index()

        # Calculate daily returns
        if len(equity_df) > 0:
            daily_returns = equity_df['equity'].pct_change().dropna()
        else:
            daily_returns = pd.Series()

        # Monthly returns
        monthly_returns = self._calculate_monthly_returns(equity_df) if len(equity_df) > 0 else pd.DataFrame()

        if verbose:
            self._print_summary(metrics, all_trades)

        return BacktestResult(
            metrics=metrics,
            trades=all_trades,
            equity_curve=equity_df,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns,
            config={
                'initial_capital': self.initial_capital,
                'max_position_pct': self.max_position_pct,
                'max_positions': self.max_positions,
                'commission': self.commission,
                'slippage': self.slippage,
            }
        )

    # =========================================================================
    # METRICS CALCULATION
    # =========================================================================

    def _calculate_metrics(self, trades: List[Trade],
                           final_equity: float,
                           start_date: datetime,
                           end_date: datetime) -> BacktestMetrics:
        """Calculate performance metrics from trades."""
        if not trades:
            return self._empty_metrics(start_date, end_date)

        # Basic stats
        total_trades = len(trades)
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl < 0]

        win_rate = (len(winning) / total_trades * 100) if total_trades > 0 else 0

        # Average win/loss
        avg_win = np.mean([t.pnl_percent for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl_percent for t in losing]) if losing else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Total return
        total_return = ((final_equity / self.initial_capital) - 1) * 100

        # Annualized return
        if start_date and end_date:
            days = (end_date - start_date).days
            years = days / 365.0
            annualized_return = ((final_equity / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        else:
            days = sum(t.holding_days for t in trades)
            years = days / 365.0
            annualized_return = total_return * (365.0 / days) if days > 0 else 0

        # Max drawdown
        equity_values = [self.initial_capital]
        for t in trades:
            equity_values.append(equity_values[-1] + t.pnl)
        max_dd = self._calculate_max_drawdown(equity_values)

        # Sharpe ratio (simplified)
        if trades:
            returns = [t.pnl_percent for t in trades]
            avg_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 1
            rf_per_trade = BACKTEST_CONFIG['risk_free_rate'] / 252 * 100  # Daily RF
            sharpe = (avg_return - rf_per_trade) / std_return if std_return > 0 else 0
            sharpe = sharpe * np.sqrt(252 / max(1, np.mean([t.holding_days for t in trades])))  # Annualize
        else:
            sharpe = 0

        # Sortino ratio
        downside_returns = [t.pnl_percent for t in trades if t.pnl_percent < 0]
        downside_std = np.std(downside_returns) if downside_returns else 1
        sortino = (avg_return - rf_per_trade) / downside_std if downside_std > 0 else 0

        # Best/worst
        best_trade = max([t.pnl_percent for t in trades]) if trades else 0
        worst_trade = min([t.pnl_percent for t in trades]) if trades else 0

        # Avg holding
        avg_holding = np.mean([t.holding_days for t in trades]) if trades else 0

        return BacktestMetrics(
            total_return_pct=round(total_return, 2),
            annualized_return_pct=round(annualized_return, 2),
            total_trades=total_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate_pct=round(win_rate, 1),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            avg_holding_days=round(avg_holding, 1),
            best_trade_pct=round(best_trade, 2),
            worst_trade_pct=round(worst_trade, 2),
            final_equity=round(final_equity, 2),
            start_date=start_date or datetime.now(),
            end_date=end_date or datetime.now(),
        )

    def _calculate_max_drawdown(self, equity_values: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if len(equity_values) < 2:
            return 0.0

        peak = equity_values[0]
        max_dd = 0.0

        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            max_dd = max(max_dd, dd)

        return max_dd

    def _calculate_monthly_returns(self, equity_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly returns table."""
        if len(equity_df) == 0:
            return pd.DataFrame()

        equity_df = equity_df.copy()
        equity_df['month'] = equity_df.index.month
        equity_df['year'] = equity_df.index.year

        # Group by month
        monthly = equity_df.groupby(['year', 'month'])['equity'].last()
        monthly_returns = monthly.pct_change() * 100

        # Pivot to year x month format
        monthly_df = monthly_returns.unstack(level='month')
        monthly_df.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly_df.columns)]

        return monthly_df

    def _empty_metrics(self, start_date: datetime, end_date: datetime) -> BacktestMetrics:
        """Return empty metrics object."""
        return BacktestMetrics(
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            avg_holding_days=0.0,
            best_trade_pct=0.0,
            worst_trade_pct=0.0,
            final_equity=self.initial_capital,
            start_date=start_date or datetime.now(),
            end_date=end_date or datetime.now(),
        )

    # =========================================================================
    # REPORTING
    # =========================================================================

    def _print_summary(self, metrics: BacktestMetrics, trades: List[Trade]):
        """Print backtest summary."""
        print("\n" + "-" * 60)
        print("BACKTEST RESULTS")
        print("-" * 60)
        print(f"Period: {metrics.start_date.date()} to {metrics.end_date.date()}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Equity: ${metrics.final_equity:,.2f}")
        print(f"")
        print(f"RETURNS")
        print(f"  Total Return:      {metrics.total_return_pct:+.2f}%")
        print(f"  Annualized:        {metrics.annualized_return_pct:+.2f}%")
        print(f"  Max Drawdown:      {metrics.max_drawdown_pct:.2f}%")
        print(f"")
        print(f"TRADES")
        print(f"  Total Trades:      {metrics.total_trades}")
        print(f"  Winning:           {metrics.winning_trades}")
        print(f"  Losing:            {metrics.losing_trades}")
        print(f"  Win Rate:          {metrics.win_rate_pct:.1f}%")
        print(f"")
        print(f"AVERAGES")
        print(f"  Avg Win:           {metrics.avg_win_pct:+.2f}%")
        print(f"  Avg Loss:          {metrics.avg_loss_pct:+.2f}%")
        print(f"  Profit Factor:     {metrics.profit_factor:.2f}")
        print(f"  Avg Holding Days:  {metrics.avg_holding_days:.1f}")
        print(f"")
        print(f"RISK METRICS")
        print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:     {metrics.sortino_ratio:.2f}")
        print(f"  Best Trade:        {metrics.best_trade_pct:+.2f}%")
        print(f"  Worst Trade:       {metrics.worst_trade_pct:+.2f}%")
        print("-" * 60)

        # Trade list
        if trades:
            print("\nRECENT TRADES (Last 10)")
            print("-" * 60)
            for trade in trades[-10:]:
                status = "WIN" if trade.pnl > 0 else "LOSS"
                exit_price = trade.exit_price if trade.exit_price else 0
                print(f"  {trade.ticker:6} | {trade.entry_date.strftime('%Y-%m-%d')} | "
                      f"${trade.entry_price:.2f} -> ${exit_price:.2f} | "
                      f"{trade.pnl_percent:+.1f}% | {status} | {trade.exit_reason}")

    def export_trades(self, trades: List[Trade], filename: str) -> pd.DataFrame:
        """
        Export trades to CSV.

        Args:
            trades: List of Trade objects
            filename: Output filename

        Returns:
            DataFrame of trades
        """
        data = []
        for t in trades:
            data.append({
                'ticker': t.ticker,
                'entry_date': t.entry_date,
                'entry_price': t.entry_price,
                'shares': t.shares,
                'position_value': t.position_value,
                'stop_loss': t.stop_loss,
                'target_price': t.target_price,
                'exit_date': t.exit_date,
                'exit_price': t.exit_price,
                'exit_reason': t.exit_reason,
                'pnl': t.pnl,
                'pnl_percent': t.pnl_percent,
                'holding_days': t.holding_days,
                'commission': t.commission,
            })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Trades exported to {filename}")
        return df


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

class WalkForwardValidator:
    """
    Walk-forward validation for strategy robustness testing.

    Splits data into train/test periods and validates out-of-sample.
    """

    def __init__(self, train_months: int = 6, test_months: int = 3):
        """
        Initialize walk-forward validator.

        Args:
            train_months: Months for training period
            test_months: Months for test period
        """
        self.train_months = train_months
        self.test_months = test_months
        self.backtest_engine = BacktestEngine()

    def run_validation(self, data: Dict[str, pd.DataFrame],
                       start_date: datetime,
                       end_date: datetime,
                       verbose: bool = True) -> Dict:
        """
        Run walk-forward validation.

        Args:
            data: Dictionary mapping ticker to DataFrame
            start_date: Overall start date
            end_date: Overall end date
            verbose: Print progress

        Returns:
            Dictionary with validation results
        """
        results = []
        current_start = start_date

        period = 0
        while current_start < end_date:
            period += 1

            # Define train and test periods
            train_end = current_start + timedelta(days=self.train_months * 30)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_months * 30)

            if test_end > end_date:
                test_end = end_date

            if test_start >= end_date:
                break

            if verbose:
                print(f"\nPeriod {period}: Train {current_start.date()} - {train_end.date()}, "
                      f"Test {test_start.date()} - {test_end.date()}")

            # Run backtest on test period only (in real WF, we'd optimize on train)
            result = self.backtest_engine.run_backtest(
                data,
                start_date=test_start,
                end_date=test_end,
                verbose=False
            )

            results.append({
                'period': period,
                'train_start': current_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'metrics': result.metrics,
                'trades': len(result.trades),
            })

            # Move to next period
            current_start = test_start

        # Aggregate results
        if results:
            avg_return = np.mean([r['metrics'].total_return_pct for r in results])
            avg_win_rate = np.mean([r['metrics'].win_rate_pct for r in results])
            avg_sharpe = np.mean([r['metrics'].sharpe_ratio for r in results])
            total_trades = sum(r['trades'] for r in results)

            if verbose:
                print("\n" + "=" * 60)
                print("WALK-FORWARD SUMMARY")
                print("=" * 60)
                print(f"Periods:          {len(results)}")
                print(f"Total Trades:     {total_trades}")
                print(f"Avg Return:       {avg_return:+.2f}%")
                print(f"Avg Win Rate:     {avg_win_rate:.1f}%")
                print(f"Avg Sharpe:       {avg_sharpe:.2f}")
                print("=" * 60)

        return {
            'periods': results,
            'summary': {
                'num_periods': len(results),
                'avg_return': avg_return if results else 0,
                'avg_win_rate': avg_win_rate if results else 0,
                'avg_sharpe': avg_sharpe if results else 0,
            }
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def quick_backtest(data: Dict[str, pd.DataFrame],
                   start_date: datetime = None,
                   end_date: datetime = None) -> BacktestResult:
    """
    Quick backtest with default parameters.

    Args:
        data: Dictionary mapping ticker to DataFrame
        start_date: Start date
        end_date: End date

    Returns:
        BacktestResult
    """
    engine = BacktestEngine()
    return engine.run_backtest(data, start_date, end_date)


def compare_parameters(data: Dict[str, pd.DataFrame],
                       param_sets: List[Dict],
                       verbose: bool = True) -> pd.DataFrame:
    """
    Compare different parameter sets.

    Args:
        data: Historical data
        param_sets: List of parameter dictionaries
        verbose: Print results

    Returns:
        DataFrame comparing results
    """
    results = []

    for params in param_sets:
        engine = BacktestEngine(**params)
        result = engine.run_backtest(data, verbose=False)

        results.append({
            **params,
            'return_pct': result.metrics.total_return_pct,
            'sharpe': result.metrics.sharpe_ratio,
            'win_rate': result.metrics.win_rate_pct,
            'max_dd': result.metrics.max_drawdown_pct,
            'trades': result.metrics.total_trades,
        })

    df = pd.DataFrame(results)

    if verbose:
        print("\nParameter Comparison:")
        print(df.to_string(index=False))

    return df


if __name__ == '__main__':
    print("EGM/PEAD Strategy Backtest Engine")
    print("=" * 60)

    # Example: Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')

    # Create sample stock with a gap
    base_price = 100
    prices = [base_price]

    for i in range(364):
        # Create a gap on day 50
        if i == 50:
            prices.append(prices[-1] * 1.12)  # 12% gap
        else:
            prices.append(prices[-1] * (1 + np.random.normal(0.0005, 0.015)))

    prices = np.array(prices)

    sample_df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, 365)),
        'high': prices * (1 + np.random.uniform(0.005, 0.02, 365)),
        'low': prices * (1 + np.random.uniform(-0.02, -0.005, 365)),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 365),
    }, index=dates)

    # Add volume spike on gap day
    sample_df.iloc[50, sample_df.columns.get_loc('volume')] = 15000000

    # Run backtest
    print("\nRunning backtest on sample data...")

    engine = BacktestEngine()
    result = engine.run_backtest(
        {'SAMPLE': sample_df},
        start_date=dates[0],
        end_date=dates[-1]
    )

    print("\nBacktest complete.")
