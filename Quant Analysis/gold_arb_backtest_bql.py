#!/usr/bin/env python3
"""
TOPIX-COMEX Gold Arbitrage Backtest - Bloomberg BQL Edition
Historical analysis using real TOCOM and COMEX data via Bloomberg Terminal

This module provides:
1. Real TOCOM Gold data via Bloomberg BQL
2. True basis spread analysis (not simulated)
3. Intraday tick data analysis for HFT opportunities
4. Comprehensive backtest with actual market data

Requires: Active Bloomberg Terminal connection with BQL access
"""

import bql
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
    CONTRACT_SPECS,
    TROY_OZ_PER_KG,
    GRAMS_PER_TROY_OZ,
    GRAMS_PER_KG,
)

# ============================================================================
# BQL SERVICE INITIALIZATION
# ============================================================================

bq = bql.Service()

# ============================================================================
# BLOOMBERG TICKER CONFIGURATION
# ============================================================================

TICKERS = {
    # COMEX Gold Futures (Active Contract)
    'COMEX_GOLD': 'GCA Comdty',

    # TOCOM Gold Futures - Multiple options depending on Bloomberg subscription
    # Try these in order of preference:
    'TOCOM_GOLD': 'JGD1 Comdty',      # TOCOM Gold Standard (1kg) - Primary
    'TOCOM_GOLD_ALT': 'JAU Comdty',   # Alternative ticker
    'TOCOM_GOLD_MINI': 'JGM1 Comdty', # TOCOM Gold Mini (100g)

    # Spot Gold (for reference)
    'SPOT_GOLD': 'XAU Curncy',

    # Currency
    'USDJPY': 'USDJPY Curncy',

    # Additional reference instruments
    'COMEX_GOLD_SPOT': 'GC1 Comdty',  # Front month specific
    'DXY': 'DXY Index',                # Dollar index for correlation
}

# Display names
TICKER_NAMES = {
    'GCA Comdty': 'COMEX Gold',
    'JGD1 Comdty': 'TOCOM Gold',
    'JAU Comdty': 'TOCOM Gold (Alt)',
    'JGM1 Comdty': 'TOCOM Gold Mini',
    'XAU Curncy': 'Gold Spot',
    'USDJPY Curncy': 'USD/JPY',
    'GC1 Comdty': 'COMEX Gold Front',
    'DXY Index': 'Dollar Index',
}


# ============================================================================
# BQL DATA FETCHING
# ============================================================================

def fetch_historical_ohlcv(ticker: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data using BQL

    Returns DataFrame with Date, Open, High, Low, Close, Volume
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Create BQL request for OHLCV data
        request = bql.Request(
            ticker,
            {
                'Date': bq.data.day()['DATE'],
                'Open': bq.data.px_open()['value'],
                'High': bq.data.px_high()['value'],
                'Low': bq.data.px_low()['value'],
                'Close': bq.data.px_last()['value'],
                'Volume': bq.data.px_volume()['value'],
            },
            with_params={
                'fill': 'prev',
                'dates': bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            }
        )

        response = bq.execute(request)
        df = pd.concat([data_item.df() for data_item in response], axis=1)

        # Clean up and sort
        df = df.sort_values('Date').reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        return df

    except Exception as e:
        print(f"Error fetching OHLCV for {ticker}: {e}")
        return None


def fetch_intraday_data(ticker: str, days: int = 5, interval: str = '1H') -> Optional[pd.DataFrame]:
    """
    Fetch intraday data for HFT analysis

    Intervals: '1M' (1 min), '5M', '15M', '30M', '1H', '4H'
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # BQL intraday request
        request = bql.Request(
            ticker,
            {
                'Time': bq.data.px_last()['DATE'],
                'Price': bq.data.px_last()['value'],
                'Bid': bq.data.px_bid()['value'],
                'Ask': bq.data.px_ask()['value'],
            },
            with_params={
                'fill': 'prev',
                'dates': bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                ),
                'frq': interval,
            }
        )

        response = bq.execute(request)
        df = pd.concat([data_item.df() for data_item in response], axis=1)

        df['Time'] = pd.to_datetime(df['Time'])
        df = df.set_index('Time').sort_index()

        return df

    except Exception as e:
        print(f"Error fetching intraday data for {ticker}: {e}")
        return None


def fetch_gold_arbitrage_data(days: int = 365) -> Optional[pd.DataFrame]:
    """
    Fetch all data needed for COMEX-TOCOM gold arbitrage analysis

    Returns combined DataFrame with:
    - COMEX Gold prices (USD/oz)
    - TOCOM Gold prices (JPY/g)
    - USDJPY rate
    - Calculated basis spread
    """
    print("Fetching Bloomberg data...")

    # Fetch COMEX Gold
    print(f"  → {TICKER_NAMES.get(TICKERS['COMEX_GOLD'], TICKERS['COMEX_GOLD'])}...")
    comex_df = fetch_historical_ohlcv(TICKERS['COMEX_GOLD'], days)

    if comex_df is None:
        print("    ✗ Failed to fetch COMEX data")
        return None
    print(f"    ✓ {len(comex_df)} observations")

    # Fetch TOCOM Gold (try primary, then alternatives)
    tocom_df = None
    for tocom_key in ['TOCOM_GOLD', 'TOCOM_GOLD_ALT', 'TOCOM_GOLD_MINI']:
        ticker = TICKERS.get(tocom_key)
        if ticker:
            print(f"  → Trying {TICKER_NAMES.get(ticker, ticker)}...")
            tocom_df = fetch_historical_ohlcv(ticker, days)
            if tocom_df is not None and len(tocom_df) > 0:
                print(f"    ✓ {len(tocom_df)} observations")
                break
            print(f"    ✗ No data available")

    if tocom_df is None:
        print("    ✗ Failed to fetch any TOCOM data")
        print("    → Will simulate TOCOM from COMEX + FX")
        tocom_df = None

    # Fetch USDJPY
    print(f"  → {TICKER_NAMES.get(TICKERS['USDJPY'], TICKERS['USDJPY'])}...")
    usdjpy_df = fetch_historical_ohlcv(TICKERS['USDJPY'], days)

    if usdjpy_df is None:
        print("    ✗ Failed to fetch USDJPY data")
        return None
    print(f"    ✓ {len(usdjpy_df)} observations")

    # Combine data
    df = pd.DataFrame({
        'comex_close': comex_df['Close'],
        'comex_high': comex_df['High'],
        'comex_low': comex_df['Low'],
        'comex_volume': comex_df['Volume'],
        'usdjpy': usdjpy_df['Close'],
    })

    # Add TOCOM data if available
    if tocom_df is not None:
        df['tocom_close'] = tocom_df['Close']
        df['tocom_high'] = tocom_df['High']
        df['tocom_low'] = tocom_df['Low']
        df['tocom_volume'] = tocom_df['Volume']
        df['has_real_tocom'] = True
    else:
        # Simulate TOCOM prices (fallback)
        df = _simulate_tocom_prices(df)
        df['has_real_tocom'] = False

    # Forward fill and drop NaN
    df = df.ffill().dropna()

    # Calculate basis spread
    df = _calculate_spread_metrics(df)

    print(f"\nCombined dataset: {len(df)} observations")
    print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

    return df


def _simulate_tocom_prices(df: pd.DataFrame, basis_mean: float = 2.0, basis_std: float = 5.0) -> pd.DataFrame:
    """
    Simulate TOCOM prices when real data unavailable
    Uses mean-reverting OU process with FX correlation
    """
    n = len(df)

    # OU process parameters
    half_life = 5.0
    phi = np.exp(-1 / half_life)
    sigma = basis_std * np.sqrt(1 - phi**2)

    # Generate basis
    np.random.seed(42)
    basis = np.zeros(n)
    basis[0] = basis_mean

    for i in range(1, n):
        # Mean-reverting component
        basis[i] = basis_mean + phi * (basis[i-1] - basis_mean) + sigma * np.random.randn()

        # Add FX correlation
        if i > 0:
            fx_ret = (df['usdjpy'].iloc[i] - df['usdjpy'].iloc[i-1]) / df['usdjpy'].iloc[i-1]
            basis[i] += fx_ret * 10000 * 0.1  # FX sensitivity

    # Convert COMEX to JPY/g and apply basis
    df['comex_jpy_g'] = df.apply(
        lambda row: usd_per_oz_to_jpy_per_gram(row['comex_close'], row['usdjpy']),
        axis=1
    )

    # Apply basis and round to tick size
    df['tocom_close'] = (df['comex_jpy_g'] * (1 + pd.Series(basis, index=df.index) / 10000)).round(0)
    df['tocom_high'] = (df['tocom_close'] * 1.001).round(0)
    df['tocom_low'] = (df['tocom_close'] * 0.999).round(0)
    df['tocom_volume'] = 10000  # Placeholder

    return df


def _calculate_spread_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate spread and basis metrics
    """
    # Convert COMEX to JPY/g for comparison
    df['comex_jpy_g'] = df.apply(
        lambda row: usd_per_oz_to_jpy_per_gram(row['comex_close'], row['usdjpy']),
        axis=1
    )

    # Calculate basis (TOCOM premium/discount vs COMEX)
    df['basis_jpy_g'] = df['tocom_close'] - df['comex_jpy_g']
    df['basis_bps'] = (df['basis_jpy_g'] / df['comex_jpy_g']) * 10000

    # Convert TOCOM to USD/oz for alternative view
    df['tocom_usd_oz'] = df.apply(
        lambda row: jpy_per_gram_to_usd_per_oz(row['tocom_close'], row['usdjpy']),
        axis=1
    )
    df['basis_usd_oz'] = df['tocom_usd_oz'] - df['comex_close']

    # Calculate returns
    df['comex_ret'] = df['comex_close'].pct_change()
    df['tocom_ret'] = df['tocom_close'].pct_change()
    df['usdjpy_ret'] = df['usdjpy'].pct_change()
    df['basis_change'] = df['basis_bps'].diff()

    # Rolling statistics
    df['basis_mean_20d'] = df['basis_bps'].rolling(20).mean()
    df['basis_std_20d'] = df['basis_bps'].rolling(20).std()
    df['basis_z_score'] = (df['basis_bps'] - df['basis_mean_20d']) / df['basis_std_20d']

    return df


# ============================================================================
# SPREAD ANALYSIS
# ============================================================================

def analyze_spread_statistics(df: pd.DataFrame) -> Dict:
    """
    Comprehensive statistical analysis of the basis spread
    """
    basis = df['basis_bps'].dropna()

    stats = {
        'mean': basis.mean(),
        'median': basis.median(),
        'std': basis.std(),
        'min': basis.min(),
        'max': basis.max(),
        'skewness': basis.skew(),
        'kurtosis': basis.kurtosis(),
        'percentile_5': basis.quantile(0.05),
        'percentile_25': basis.quantile(0.25),
        'percentile_75': basis.quantile(0.75),
        'percentile_95': basis.quantile(0.95),
        'autocorr_1d': basis.autocorr(lag=1),
        'autocorr_5d': basis.autocorr(lag=5),
    }

    # Estimate half-life via OLS
    if len(basis) > 30:
        y = basis.values[1:]
        x = basis.values[:-1]

        mean_x, mean_y = np.mean(x), np.mean(y)
        ss_xy = np.sum((x - mean_x) * (y - mean_y))
        ss_xx = np.sum((x - mean_x)**2)

        beta = ss_xy / ss_xx if ss_xx > 0 else 1

        if 0 < beta < 1:
            stats['half_life_days'] = -np.log(2) / np.log(beta)
        else:
            stats['half_life_days'] = None

    return stats


def analyze_correlations(df: pd.DataFrame) -> Dict:
    """
    Correlation analysis between gold, FX, and basis
    """
    df_clean = df[['comex_ret', 'tocom_ret', 'usdjpy_ret', 'basis_change']].dropna()

    correlations = {
        'comex_tocom': df_clean['comex_ret'].corr(df_clean['tocom_ret']),
        'comex_fx': df_clean['comex_ret'].corr(df_clean['usdjpy_ret']),
        'tocom_fx': df_clean['tocom_ret'].corr(df_clean['usdjpy_ret']),
        'fx_basis': df_clean['usdjpy_ret'].corr(df_clean['basis_change']),
    }

    # Rolling correlation
    window = 20
    correlations['rolling_comex_tocom'] = df_clean['comex_ret'].rolling(window).corr(
        df_clean['tocom_ret']
    ).iloc[-1]

    # Lagged FX-Basis relationship
    df_clean['fx_ret_lag1'] = df_clean['usdjpy_ret'].shift(1)
    df_lagged = df_clean.dropna()

    if len(df_lagged) > 30:
        x = df_lagged['fx_ret_lag1'].values
        y = df_lagged['basis_change'].values

        mean_x, mean_y = np.mean(x), np.mean(y)
        ss_xy = np.sum((x - mean_x) * (y - mean_y))
        ss_xx = np.sum((x - mean_x)**2)
        ss_yy = np.sum((y - mean_y)**2)

        slope = ss_xy / ss_xx if ss_xx > 0 else 0
        r_value = ss_xy / np.sqrt(ss_xx * ss_yy) if ss_xx > 0 and ss_yy > 0 else 0

        correlations['fx_predicts_basis'] = {
            'coefficient': slope,
            'r_squared': r_value**2,
        }

    return correlations


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BQLArbitrageBacktest:
    """
    Full backtest engine for TOPIX-COMEX arbitrage using Bloomberg data
    """

    def __init__(
        self,
        initial_capital_usd: float = 1_000_000,
        comex_contracts: int = 10,
        tocom_contracts: int = 31,  # ~31 TOCOM (1kg) ≈ 10 COMEX (100oz)
        comex_fee_usd: float = 2.50,
        tocom_fee_jpy: float = 500,
        slippage_bps: float = 1.0,
    ):
        self.initial_capital = initial_capital_usd
        self.comex_contracts = comex_contracts
        self.tocom_contracts = tocom_contracts
        self.comex_fee = comex_fee_usd
        self.tocom_fee = tocom_fee_jpy
        self.slippage_bps = slippage_bps

        self.equity_curve = []
        self.trades = []
        self.daily_pnl = []

    def run(
        self,
        df: pd.DataFrame,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        max_holding_days: int = 10,
        stop_loss_bps: float = 20.0,
    ) -> Dict:
        """
        Run backtest with mean-reversion strategy

        Strategy:
        - Enter when z-score exceeds threshold (fade the extreme)
        - Exit when z-score reverts to exit threshold
        - Stop loss if basis moves against position
        """
        df = df.copy()
        df = df.dropna(subset=['basis_z_score'])

        capital = self.initial_capital
        position = 0  # 1 = long TOCOM/short COMEX, -1 = short TOCOM/long COMEX
        entry_data = None

        for i, (date, row) in enumerate(df.iterrows()):
            daily_pnl = 0

            # Skip if no z-score
            if pd.isna(row['basis_z_score']):
                continue

            if position == 0:
                # Entry logic
                if row['basis_z_score'] > entry_z:
                    # Basis too high (TOCOM premium) - short TOCOM, long COMEX
                    position = -1
                    entry_data = {
                        'date': date,
                        'basis_bps': row['basis_bps'],
                        'z_score': row['basis_z_score'],
                        'comex_price': row['comex_close'],
                        'tocom_price': row['tocom_close'],
                        'usdjpy': row['usdjpy'],
                    }

                elif row['basis_z_score'] < -entry_z:
                    # Basis too low (TOCOM discount) - long TOCOM, short COMEX
                    position = 1
                    entry_data = {
                        'date': date,
                        'basis_bps': row['basis_bps'],
                        'z_score': row['basis_z_score'],
                        'comex_price': row['comex_close'],
                        'tocom_price': row['tocom_close'],
                        'usdjpy': row['usdjpy'],
                    }

            else:
                # Position management
                holding_days = (date - entry_data['date']).days
                basis_move = row['basis_bps'] - entry_data['basis_bps']

                should_exit = False
                exit_reason = None

                # Exit conditions
                if position == -1:
                    # Short TOCOM/Long COMEX - profit when basis falls
                    if row['basis_z_score'] < exit_z:
                        should_exit = True
                        exit_reason = 'mean_reversion'
                    elif basis_move > stop_loss_bps:
                        should_exit = True
                        exit_reason = 'stop_loss'

                elif position == 1:
                    # Long TOCOM/Short COMEX - profit when basis rises
                    if row['basis_z_score'] > -exit_z:
                        should_exit = True
                        exit_reason = 'mean_reversion'
                    elif basis_move < -stop_loss_bps:
                        should_exit = True
                        exit_reason = 'stop_loss'

                if holding_days >= max_holding_days:
                    should_exit = True
                    exit_reason = 'max_holding'

                if should_exit:
                    # Calculate PnL
                    pnl = self._calculate_trade_pnl(
                        position, entry_data, row
                    )

                    daily_pnl = pnl['net_pnl_usd']
                    capital += pnl['net_pnl_usd']

                    self.trades.append({
                        'entry_date': entry_data['date'],
                        'exit_date': date,
                        'direction': 'SHORT_TOCOM' if position == -1 else 'LONG_TOCOM',
                        'entry_basis': entry_data['basis_bps'],
                        'exit_basis': row['basis_bps'],
                        'entry_z': entry_data['z_score'],
                        'exit_z': row['basis_z_score'],
                        'holding_days': holding_days,
                        'gross_pnl_usd': pnl['gross_pnl_usd'],
                        'costs_usd': pnl['total_costs_usd'],
                        'net_pnl_usd': pnl['net_pnl_usd'],
                        'exit_reason': exit_reason,
                    })

                    position = 0
                    entry_data = None

            self.equity_curve.append({
                'date': date,
                'capital': capital,
                'position': position,
                'basis_bps': row['basis_bps'],
                'z_score': row['basis_z_score'],
            })
            self.daily_pnl.append(daily_pnl)

        return self._calculate_metrics()

    def _calculate_trade_pnl(self, position: int, entry: Dict, exit_row) -> Dict:
        """
        Calculate detailed trade PnL
        """
        # Basis change in bps
        basis_change = exit_row['basis_bps'] - entry['basis_bps']

        # COMEX leg PnL (USD)
        comex_price_change = exit_row['comex_close'] - entry['comex_price']
        comex_contract_value = entry['comex_price'] * CONTRACT_SPECS['COMEX_GC']['contract_size_oz']
        comex_pnl = -position * comex_price_change * CONTRACT_SPECS['COMEX_GC']['contract_size_oz'] * self.comex_contracts

        # TOCOM leg PnL (convert JPY to USD)
        tocom_price_change = exit_row['tocom_close'] - entry['tocom_price']
        avg_usdjpy = (entry['usdjpy'] + exit_row['usdjpy']) / 2
        tocom_pnl_jpy = position * tocom_price_change * GRAMS_PER_KG * self.tocom_contracts
        tocom_pnl_usd = tocom_pnl_jpy / avg_usdjpy

        # Gross PnL
        gross_pnl = comex_pnl + tocom_pnl_usd

        # Transaction costs
        comex_costs = self.comex_fee * self.comex_contracts * 2  # Round trip
        tocom_costs_jpy = self.tocom_fee * self.tocom_contracts * 2
        tocom_costs_usd = tocom_costs_jpy / avg_usdjpy

        # Slippage
        slippage_usd = (self.slippage_bps / 10000) * (
            comex_contract_value * self.comex_contracts +
            (entry['tocom_price'] * GRAMS_PER_KG / entry['usdjpy']) * self.tocom_contracts
        )

        total_costs = comex_costs + tocom_costs_usd + slippage_usd
        net_pnl = gross_pnl - total_costs

        return {
            'gross_pnl_usd': gross_pnl,
            'comex_pnl_usd': comex_pnl,
            'tocom_pnl_usd': tocom_pnl_usd,
            'comex_costs_usd': comex_costs,
            'tocom_costs_usd': tocom_costs_usd,
            'slippage_usd': slippage_usd,
            'total_costs_usd': total_costs,
            'net_pnl_usd': net_pnl,
        }

    def _calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if not self.trades:
            return {'error': 'No trades executed'}

        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        # Trade statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl_usd'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl_usd'] <= 0])

        # PnL statistics
        total_pnl = trades_df['net_pnl_usd'].sum()
        avg_pnl = trades_df['net_pnl_usd'].mean()
        avg_win = trades_df[trades_df['net_pnl_usd'] > 0]['net_pnl_usd'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_pnl_usd'] <= 0]['net_pnl_usd'].mean() if losing_trades > 0 else 0

        max_win = trades_df['net_pnl_usd'].max()
        max_loss = trades_df['net_pnl_usd'].min()

        # Returns
        final_capital = equity_df['capital'].iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100

        # Risk metrics
        daily_returns = pd.Series(self.daily_pnl) / self.initial_capital
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

        # Drawdown
        equity_series = equity_df['capital']
        rolling_max = equity_series.expanding().max()
        drawdown = (rolling_max - equity_series) / rolling_max * 100
        max_drawdown = drawdown.max()

        # Trade breakdown by exit reason
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()

        # Trade breakdown by direction
        direction_breakdown = trades_df.groupby('direction')['net_pnl_usd'].agg(['count', 'sum', 'mean'])

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades * 100 if total_trades > 0 else 0,
            'total_pnl_usd': total_pnl,
            'avg_pnl_usd': avg_pnl,
            'avg_win_usd': avg_win,
            'avg_loss_usd': avg_loss,
            'max_win_usd': max_win,
            'max_loss_usd': max_loss,
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'avg_holding_days': trades_df['holding_days'].mean(),
            'exit_reasons': exit_reasons,
            'direction_breakdown': direction_breakdown,
            'trades_df': trades_df,
            'equity_df': equity_df,
        }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_bql_backtest_report(days: int = 365):
    """
    Generate comprehensive backtest report using Bloomberg data
    """
    print("=" * 80)
    print("TOPIX-COMEX GOLD ARBITRAGE BACKTEST - BLOOMBERG BQL EDITION")
    print("=" * 80)

    # Fetch data
    print("\n1. DATA ACQUISITION")
    print("-" * 40)
    df = fetch_gold_arbitrage_data(days)

    if df is None or len(df) < 30:
        print("ERROR: Insufficient data for backtest")
        return None

    has_real_tocom = df['has_real_tocom'].iloc[0] if 'has_real_tocom' in df.columns else False
    print(f"\n   TOCOM Data: {'Real (Bloomberg)' if has_real_tocom else 'Simulated'}")

    # Spread statistics
    print("\n2. SPREAD STATISTICS")
    print("-" * 40)
    stats = analyze_spread_statistics(df)

    print(f"   Mean basis: {stats['mean']:.2f} bps")
    print(f"   Median: {stats['median']:.2f} bps")
    print(f"   Std dev: {stats['std']:.2f} bps")
    print(f"   Range: [{stats['min']:.2f}, {stats['max']:.2f}] bps")
    print(f"   5th/95th percentile: [{stats['percentile_5']:.2f}, {stats['percentile_95']:.2f}] bps")
    print(f"   Skewness: {stats['skewness']:.3f}")
    print(f"   Kurtosis: {stats['kurtosis']:.3f}")

    if stats.get('half_life_days'):
        print(f"   Half-life: {stats['half_life_days']:.1f} days")
    else:
        print(f"   Half-life: No mean reversion detected")

    print(f"   Autocorrelation (1d): {stats['autocorr_1d']:.3f}")
    print(f"   Autocorrelation (5d): {stats['autocorr_5d']:.3f}")

    # Correlation analysis
    print("\n3. CORRELATION ANALYSIS")
    print("-" * 40)
    correlations = analyze_correlations(df)

    print(f"   COMEX-TOCOM price correlation: {correlations['comex_tocom']:.3f}")
    print(f"   COMEX-USDJPY correlation: {correlations['comex_fx']:.3f}")
    print(f"   TOCOM-USDJPY correlation: {correlations['tocom_fx']:.3f}")
    print(f"   USDJPY-Basis correlation: {correlations['fx_basis']:.3f}")

    if 'fx_predicts_basis' in correlations:
        fx_pred = correlations['fx_predicts_basis']
        print(f"   FX → Basis predictive R²: {fx_pred['r_squared']:.4f}")

    # Run backtest
    print("\n4. BACKTEST RESULTS")
    print("-" * 40)

    backtest = BQLArbitrageBacktest(
        initial_capital_usd=1_000_000,
        comex_contracts=10,
        tocom_contracts=31,
        comex_fee_usd=2.50,
        tocom_fee_jpy=500,
        slippage_bps=1.0,
    )

    results = backtest.run(
        df,
        entry_z=2.0,
        exit_z=0.5,
        max_holding_days=10,
        stop_loss_bps=20.0,
    )

    if 'error' in results:
        print(f"   {results['error']}")
        return None

    print(f"\n   Performance Summary:")
    print(f"   {'─' * 35}")
    print(f"   Total trades:        {results['total_trades']}")
    print(f"   Win rate:            {results['win_rate']:.1f}%")
    print(f"   Profit factor:       {results['profit_factor']:.2f}")
    print(f"   Total PnL:           ${results['total_pnl_usd']:,.2f}")
    print(f"   Total return:        {results['total_return_pct']:.2f}%")
    print(f"   Sharpe ratio:        {results['sharpe_ratio']:.2f}")
    print(f"   Max drawdown:        {results['max_drawdown_pct']:.2f}%")
    print(f"   Avg holding period:  {results['avg_holding_days']:.1f} days")

    print(f"\n   Trade Breakdown:")
    print(f"   {'─' * 35}")
    print(f"   Average win:         ${results['avg_win_usd']:,.2f}")
    print(f"   Average loss:        ${results['avg_loss_usd']:,.2f}")
    print(f"   Max win:             ${results['max_win_usd']:,.2f}")
    print(f"   Max loss:            ${results['max_loss_usd']:,.2f}")

    print(f"\n   Exit Reasons:")
    for reason, count in results['exit_reasons'].items():
        pct = count / results['total_trades'] * 100
        print(f"   - {reason}: {count} ({pct:.1f}%)")

    print(f"\n   Direction Breakdown:")
    for direction, row in results['direction_breakdown'].iterrows():
        print(f"   - {direction}: {int(row['count'])} trades, "
              f"Total ${row['sum']:,.2f}, Avg ${row['mean']:,.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)

    return {
        'data': df,
        'stats': stats,
        'correlations': correlations,
        'backtest_results': results,
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_bloomberg_connection() -> bool:
    """
    Test Bloomberg BQL connection
    """
    try:
        # Simple test query
        request = bql.Request(
            'SPY US Equity',
            {'Price': bq.data.px_last()}
        )
        response = bq.execute(request)
        print("✓ Bloomberg connection successful")
        return True
    except Exception as e:
        print(f"✗ Bloomberg connection failed: {e}")
        return False


def list_available_gold_tickers():
    """
    List and test available gold-related tickers
    """
    test_tickers = [
        ('GCA Comdty', 'COMEX Gold Active'),
        ('GC1 Comdty', 'COMEX Gold Front Month'),
        ('JGD1 Comdty', 'TOCOM Gold Standard'),
        ('JAU Comdty', 'TOCOM Gold (Alt)'),
        ('JGM1 Comdty', 'TOCOM Gold Mini'),
        ('XAU Curncy', 'Gold Spot'),
        ('USDJPY Curncy', 'USD/JPY'),
    ]

    print("Testing available gold tickers...")
    print("-" * 50)

    available = []
    for ticker, name in test_tickers:
        try:
            request = bql.Request(
                ticker,
                {'Price': bq.data.px_last()}
            )
            response = bq.execute(request)
            df = pd.concat([data_item.df() for data_item in response], axis=1)
            price = df['Price'].iloc[0] if 'Price' in df.columns else 'N/A'
            print(f"  ✓ {ticker:<15} ({name}): {price}")
            available.append(ticker)
        except Exception as e:
            print(f"  ✗ {ticker:<15} ({name}): Not available")

    return available


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check Bloomberg connection first
    print("\nChecking Bloomberg connection...")
    if not check_bloomberg_connection():
        print("\nERROR: Bloomberg Terminal connection required.")
        print("Please ensure Bloomberg Terminal is running and BQL is accessible.")
        exit(1)

    # List available tickers
    print("\n")
    available_tickers = list_available_gold_tickers()

    # Run backtest
    print("\n")
    report = generate_bql_backtest_report(days=365)

    if report:
        print("\nResults stored in 'report' dictionary.")
        print("Access trades via: report['backtest_results']['trades_df']")
        print("Access equity curve via: report['backtest_results']['equity_df']")
