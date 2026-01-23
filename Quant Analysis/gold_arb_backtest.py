#!/usr/bin/env python3
"""
TOPIX-COMEX Gold Arbitrage Backtest
Historical analysis and strategy validation

This module provides:
1. Historical spread analysis using available data
2. Correlation analysis between COMEX gold and USDJPY
3. Simulated TOCOM prices for strategy testing
4. Performance metrics and visualization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import core arbitrage functions
from topix_comex_gold_arbitrage import (
    usd_per_oz_to_jpy_per_gram,
    jpy_per_gram_to_usd_per_oz,
    calculate_arbitrage_threshold,
    calculate_basis_spread,
    TROY_OZ_PER_KG,
    GRAMS_PER_TROY_OZ,
    GRAMS_PER_KG,
)


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_historical_data(
    start_date: str = '2024-01-01',
    end_date: str = None,
    source: str = 'yfinance'
) -> pd.DataFrame:
    """
    Fetch historical COMEX gold and USDJPY data

    Note: Direct TOCOM data is not available via yfinance.
    We simulate TOCOM prices based on COMEX + FX + typical basis.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if source == 'yfinance':
        try:
            import yfinance as yf

            # Fetch COMEX Gold and USDJPY
            gold = yf.download('GC=F', start=start_date, end=end_date, progress=False)
            usdjpy = yf.download('JPY=X', start=start_date, end=end_date, progress=False)

            # Handle multi-index columns
            if isinstance(gold.columns, pd.MultiIndex):
                gold.columns = gold.columns.get_level_values(0)
            if isinstance(usdjpy.columns, pd.MultiIndex):
                usdjpy.columns = usdjpy.columns.get_level_values(0)

            # Combine into single DataFrame
            df = pd.DataFrame({
                'comex_close': gold['Close'],
                'comex_high': gold['High'],
                'comex_low': gold['Low'],
                'comex_volume': gold['Volume'],
                'usdjpy': usdjpy['Close'],
            })

            # Forward fill any missing data
            df = df.ffill().dropna()

            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    elif source == 'bloomberg':
        # Bloomberg implementation would go here
        print("Bloomberg source not implemented in this version")
        return None

    return None


def simulate_tocom_prices(df: pd.DataFrame, basis_params: Dict = None) -> pd.DataFrame:
    """
    Simulate TOCOM gold prices based on COMEX + FX + stochastic basis

    The basis between COMEX and TOCOM typically exhibits:
    1. Mean-reverting behavior around a small premium/discount
    2. Higher volatility during Asian session opens
    3. Correlation with USDJPY momentum

    Parameters:
    - basis_params: Dict with 'mean_bps', 'std_bps', 'half_life'
    """
    if basis_params is None:
        basis_params = {
            'mean_bps': 2.0,       # Average 2 bps TOCOM premium
            'std_bps': 5.0,        # Standard deviation
            'half_life': 5,        # Mean reversion half-life (days)
            'fx_sensitivity': 0.1, # Basis sensitivity to FX moves
        }

    n = len(df)

    # Generate mean-reverting basis using OU process
    phi = np.exp(-1 / basis_params['half_life'])
    sigma = basis_params['std_bps'] * np.sqrt(1 - phi**2)
    mu = basis_params['mean_bps']

    basis = np.zeros(n)
    basis[0] = mu

    np.random.seed(42)  # For reproducibility
    for i in range(1, n):
        # OU process: basis(t) = mu + phi*(basis(t-1) - mu) + sigma*noise
        # Add FX momentum component
        if i > 1:
            fx_return = (df['usdjpy'].iloc[i] - df['usdjpy'].iloc[i-1]) / df['usdjpy'].iloc[i-1]
            fx_component = fx_return * 10000 * basis_params['fx_sensitivity']  # Convert to bps
        else:
            fx_component = 0

        basis[i] = mu + phi * (basis[i-1] - mu) + sigma * np.random.randn() + fx_component

    df = df.copy()
    df['basis_bps'] = basis

    # Calculate TOCOM prices in JPY/g
    df['comex_jpy_g'] = df.apply(
        lambda row: usd_per_oz_to_jpy_per_gram(row['comex_close'], row['usdjpy']),
        axis=1
    )

    # Apply basis and round to TOCOM tick size (JPY 1)
    df['tocom_jpy_g'] = (df['comex_jpy_g'] * (1 + df['basis_bps'] / 10000)).round(0)

    # Recalculate actual basis after rounding
    df['actual_basis_bps'] = ((df['tocom_jpy_g'] / df['comex_jpy_g']) - 1) * 10000

    return df


# ============================================================================
# SPREAD ANALYSIS
# ============================================================================

def analyze_spread_statistics(df: pd.DataFrame) -> Dict:
    """
    Comprehensive statistical analysis of the spread
    """
    basis = df['actual_basis_bps']

    stats = {
        'mean': basis.mean(),
        'median': basis.median(),
        'std': basis.std(),
        'min': basis.min(),
        'max': basis.max(),
        'skewness': basis.skew(),
        'kurtosis': basis.kurtosis(),
        'percentile_5': basis.quantile(0.05),
        'percentile_95': basis.quantile(0.95),
    }

    # Autocorrelation analysis
    stats['autocorr_1d'] = basis.autocorr(lag=1)
    stats['autocorr_5d'] = basis.autocorr(lag=5)

    # Estimate half-life
    if len(basis) > 30:
        y = basis.values[1:]
        x = basis.values[:-1]

        mean_x, mean_y = np.mean(x), np.mean(y)
        beta = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x)**2)

        if 0 < beta < 1:
            stats['half_life_days'] = -np.log(2) / np.log(beta)
        else:
            stats['half_life_days'] = None

    return stats


def identify_arbitrage_opportunities(
    df: pd.DataFrame,
    threshold_bps: float = 5.0,
    min_holding_periods: int = 1,
    max_holding_periods: int = 10
) -> pd.DataFrame:
    """
    Identify historical arbitrage entry and exit points
    """
    df = df.copy()
    basis = df['actual_basis_bps']

    # Calculate z-scores
    rolling_mean = basis.rolling(window=20).mean()
    rolling_std = basis.rolling(window=20).std()
    df['z_score'] = (basis - rolling_mean) / rolling_std

    # Generate signals
    df['signal'] = 0
    df.loc[df['z_score'] > 2.0, 'signal'] = -1   # Sell TOCOM, Buy COMEX
    df.loc[df['z_score'] < -2.0, 'signal'] = 1   # Buy TOCOM, Sell COMEX

    # Identify trades
    trades = []
    position = 0
    entry_idx = None
    entry_basis = None

    for i in range(len(df)):
        row = df.iloc[i]

        if position == 0:
            # Look for entry
            if abs(row['actual_basis_bps']) > threshold_bps:
                position = -1 if row['actual_basis_bps'] > 0 else 1
                entry_idx = i
                entry_basis = row['actual_basis_bps']

        else:
            # Look for exit
            holding_time = i - entry_idx

            # Exit conditions:
            # 1. Mean reversion (basis crosses zero)
            # 2. Max holding time reached
            # 3. Profit target hit

            exit_trade = False

            if position == -1 and row['actual_basis_bps'] < 0:  # Sold TOCOM premium, now discount
                exit_trade = True
            elif position == 1 and row['actual_basis_bps'] > 0:  # Bought TOCOM discount, now premium
                exit_trade = True
            elif holding_time >= max_holding_periods:
                exit_trade = True

            if exit_trade:
                pnl_bps = position * (entry_basis - row['actual_basis_bps'])

                trades.append({
                    'entry_date': df.index[entry_idx],
                    'exit_date': df.index[i],
                    'entry_basis_bps': entry_basis,
                    'exit_basis_bps': row['actual_basis_bps'],
                    'direction': 'SELL_TOCOM' if position == -1 else 'BUY_TOCOM',
                    'pnl_bps': pnl_bps,
                    'holding_days': holding_time,
                })

                position = 0
                entry_idx = None
                entry_basis = None

    return pd.DataFrame(trades)


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class ArbitrageBacktest:
    """
    Full backtest engine for TOPIX-COMEX arbitrage strategy
    """

    def __init__(
        self,
        initial_capital_usd: float = 1_000_000,
        contracts_per_trade: int = 10,
        transaction_cost_bps: float = 2.0,  # Round-trip
    ):
        self.initial_capital = initial_capital_usd
        self.contracts = contracts_per_trade
        self.tx_cost_bps = transaction_cost_bps

        self.equity_curve = []
        self.trades = []
        self.daily_pnl = []

    def run(
        self,
        df: pd.DataFrame,
        entry_threshold_z: float = 2.0,
        exit_threshold_z: float = 0.5,
        max_holding_days: int = 10
    ) -> Dict:
        """
        Run backtest on historical data
        """
        df = df.copy()

        # Calculate z-scores
        basis = df['actual_basis_bps']
        df['z_score'] = (basis - basis.rolling(20).mean()) / basis.rolling(20).std()
        df = df.dropna()

        capital = self.initial_capital
        position = 0
        entry_data = None

        for i, (date, row) in enumerate(df.iterrows()):
            daily_pnl = 0

            if position == 0:
                # Entry logic
                if row['z_score'] > entry_threshold_z:
                    # Sell TOCOM, Buy COMEX (basis too high)
                    position = -1
                    entry_data = {
                        'date': date,
                        'basis': row['actual_basis_bps'],
                        'z_score': row['z_score'],
                        'comex_price': row['comex_close'],
                        'usdjpy': row['usdjpy'],
                    }

                elif row['z_score'] < -entry_threshold_z:
                    # Buy TOCOM, Sell COMEX (basis too low)
                    position = 1
                    entry_data = {
                        'date': date,
                        'basis': row['actual_basis_bps'],
                        'z_score': row['z_score'],
                        'comex_price': row['comex_close'],
                        'usdjpy': row['usdjpy'],
                    }

            else:
                # Exit logic
                holding_days = (date - entry_data['date']).days if hasattr(date, 'day') else i - df.index.get_loc(entry_data['date'])

                should_exit = False
                exit_reason = None

                if position == -1 and row['z_score'] < exit_threshold_z:
                    should_exit = True
                    exit_reason = 'mean_reversion'
                elif position == 1 and row['z_score'] > -exit_threshold_z:
                    should_exit = True
                    exit_reason = 'mean_reversion'
                elif holding_days >= max_holding_days:
                    should_exit = True
                    exit_reason = 'max_holding'

                if should_exit:
                    # Calculate PnL
                    basis_change = row['actual_basis_bps'] - entry_data['basis']
                    gross_pnl_bps = position * basis_change * -1  # Profit from basis convergence

                    # Convert to USD (approximate)
                    contract_value = row['comex_close'] * 100  # COMEX contract value
                    gross_pnl_usd = (gross_pnl_bps / 10000) * contract_value * self.contracts
                    tx_cost_usd = (self.tx_cost_bps / 10000) * contract_value * self.contracts
                    net_pnl_usd = gross_pnl_usd - tx_cost_usd

                    daily_pnl = net_pnl_usd
                    capital += net_pnl_usd

                    self.trades.append({
                        'entry_date': entry_data['date'],
                        'exit_date': date,
                        'direction': 'SELL_TOCOM' if position == -1 else 'BUY_TOCOM',
                        'entry_basis': entry_data['basis'],
                        'exit_basis': row['actual_basis_bps'],
                        'entry_z': entry_data['z_score'],
                        'exit_z': row['z_score'],
                        'holding_days': holding_days,
                        'gross_pnl_bps': gross_pnl_bps,
                        'net_pnl_usd': net_pnl_usd,
                        'exit_reason': exit_reason,
                    })

                    position = 0
                    entry_data = None

            self.equity_curve.append({
                'date': date,
                'capital': capital,
                'position': position,
            })
            self.daily_pnl.append(daily_pnl)

        return self.calculate_metrics()

    def calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if not self.trades:
            return {'error': 'No trades executed'}

        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        # Basic stats
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl_usd'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl_usd'] < 0])

        # PnL stats
        total_pnl = trades_df['net_pnl_usd'].sum()
        avg_pnl = trades_df['net_pnl_usd'].mean()
        avg_win = trades_df[trades_df['net_pnl_usd'] > 0]['net_pnl_usd'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_pnl_usd'] < 0]['net_pnl_usd'].mean() if losing_trades > 0 else 0

        # Returns
        final_capital = equity_df['capital'].iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100

        # Risk metrics
        daily_returns = pd.Series(self.daily_pnl) / self.initial_capital
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

        max_drawdown = 0
        peak = self.initial_capital
        for equity in equity_df['capital']:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades * 100 if total_trades > 0 else 0,
            'total_pnl_usd': total_pnl,
            'avg_pnl_usd': avg_pnl,
            'avg_win_usd': avg_win,
            'avg_loss_usd': avg_loss,
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'avg_holding_days': trades_df['holding_days'].mean(),
            'trades_df': trades_df,
            'equity_df': equity_df,
        }


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def analyze_correlations(df: pd.DataFrame) -> Dict:
    """
    Analyze correlations between gold, FX, and basis
    """
    # Calculate returns
    df = df.copy()
    df['gold_ret'] = df['comex_close'].pct_change()
    df['fx_ret'] = df['usdjpy'].pct_change()
    df['basis_change'] = df['actual_basis_bps'].diff()

    df = df.dropna()

    correlations = {
        'gold_fx': df['gold_ret'].corr(df['fx_ret']),
        'gold_basis': df['gold_ret'].corr(df['basis_change']),
        'fx_basis': df['fx_ret'].corr(df['basis_change']),
    }

    # Rolling correlations
    window = 20
    correlations['rolling_gold_fx'] = df['gold_ret'].rolling(window).corr(df['fx_ret']).iloc[-1]

    # Simple regression test (no scipy required)
    # Does FX predict basis changes?
    df['fx_ret_lag1'] = df['fx_ret'].shift(1)
    df = df.dropna()

    # Manual OLS regression
    x = df['fx_ret_lag1'].values
    y = df['basis_change'].values
    n = len(x)

    x_mean, y_mean = np.mean(x), np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean)**2)
    ss_yy = np.sum((y - y_mean)**2)

    slope = ss_xy / ss_xx if ss_xx > 0 else 0
    r_value = ss_xy / np.sqrt(ss_xx * ss_yy) if ss_xx > 0 and ss_yy > 0 else 0

    # Approximate t-statistic and p-value
    se = np.sqrt((ss_yy - slope * ss_xy) / (n - 2)) / np.sqrt(ss_xx) if ss_xx > 0 and n > 2 else 1
    t_stat = slope / se if se > 0 else 0

    # Approximate p-value (two-tailed, assuming normal for large n)
    p_value = 2 * (1 - 0.5 * (1 + np.tanh(0.7 * abs(t_stat) / np.sqrt(2))))  # Approximation

    correlations['fx_predicts_basis'] = {
        'coefficient': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'significant': abs(t_stat) > 1.96,  # ~95% confidence
    }

    return correlations


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_backtest_report(
    start_date: str = '2024-01-01',
    end_date: str = None
):
    """
    Generate comprehensive backtest report
    """
    print("=" * 80)
    print("TOPIX-COMEX GOLD ARBITRAGE BACKTEST REPORT")
    print("=" * 80)

    # Fetch data
    print("\n1. Fetching historical data...")
    df = fetch_historical_data(start_date, end_date)

    if df is None or len(df) < 30:
        print("Insufficient data for backtest")
        return None

    print(f"   Data period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Total observations: {len(df)}")

    # Simulate TOCOM prices
    print("\n2. Simulating TOCOM prices...")
    df = simulate_tocom_prices(df)

    # Spread statistics
    print("\n3. Spread Statistics")
    print("-" * 40)
    stats = analyze_spread_statistics(df)
    print(f"   Mean basis: {stats['mean']:.2f} bps")
    print(f"   Std dev: {stats['std']:.2f} bps")
    print(f"   Range: [{stats['min']:.2f}, {stats['max']:.2f}] bps")
    print(f"   Half-life: {stats.get('half_life_days', 'N/A')} days")
    print(f"   Autocorrelation (1d): {stats['autocorr_1d']:.3f}")

    # Correlation analysis
    print("\n4. Correlation Analysis")
    print("-" * 40)
    correlations = analyze_correlations(df)
    print(f"   Gold-FX correlation: {correlations['gold_fx']:.3f}")
    print(f"   Gold-Basis correlation: {correlations['gold_basis']:.3f}")
    print(f"   FX-Basis correlation: {correlations['fx_basis']:.3f}")
    print(f"   FX predicts basis: {'Yes' if correlations['fx_predicts_basis']['significant'] else 'No'}")
    print(f"   (p-value: {correlations['fx_predicts_basis']['p_value']:.4f})")

    # Run backtest
    print("\n5. Backtest Results")
    print("-" * 40)
    backtest = ArbitrageBacktest(
        initial_capital_usd=1_000_000,
        contracts_per_trade=10,
        transaction_cost_bps=2.0
    )

    results = backtest.run(
        df,
        entry_threshold_z=2.0,
        exit_threshold_z=0.5,
        max_holding_days=10
    )

    if 'error' in results:
        print(f"   {results['error']}")
        return None

    print(f"   Total trades: {results['total_trades']}")
    print(f"   Win rate: {results['win_rate']:.1f}%")
    print(f"   Profit factor: {results['profit_factor']:.2f}")
    print(f"   Total return: {results['total_return_pct']:.2f}%")
    print(f"   Sharpe ratio: {results['sharpe_ratio']:.2f}")
    print(f"   Max drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"   Avg holding period: {results['avg_holding_days']:.1f} days")

    # Trade breakdown
    print("\n6. Trade Breakdown")
    print("-" * 40)
    trades = results['trades_df']
    print(f"   SELL_TOCOM trades: {len(trades[trades['direction'] == 'SELL_TOCOM'])}")
    print(f"   BUY_TOCOM trades: {len(trades[trades['direction'] == 'BUY_TOCOM'])}")
    print(f"   Mean reversion exits: {len(trades[trades['exit_reason'] == 'mean_reversion'])}")
    print(f"   Max holding exits: {len(trades[trades['exit_reason'] == 'max_holding'])}")

    print("\n" + "=" * 80)

    return {
        'data': df,
        'stats': stats,
        'correlations': correlations,
        'backtest_results': results,
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run backtest with 1 year of data
    report = generate_backtest_report(
        start_date='2024-01-01',
        end_date=None  # Use current date
    )

    if report:
        print("\nBacktest complete. Results stored in 'report' dictionary.")

        # Optional: Save trades to CSV
        if 'backtest_results' in report and 'trades_df' in report['backtest_results']:
            trades_df = report['backtest_results']['trades_df']
            # trades_df.to_csv('gold_arb_trades.csv', index=False)
            # print("\nTrades saved to gold_arb_trades.csv")
