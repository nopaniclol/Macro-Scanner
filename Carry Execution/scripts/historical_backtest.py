#!/usr/bin/env python3
"""
Historical Backtest Script

Full 10-year backtest using free data (yfinance/FRED):
1. Load historical data
2. Compute signals daily
3. Construct portfolio daily
4. Run backtest engine
5. Generate performance report with equity curve
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Import modules
from signals.momentum import MomentumSignalEngine
from signals.regime import compute_vix_regime_multiplier, compute_drawdown_regime_multiplier
from portfolio.construction import PortfolioConstructor, PortfolioConfig
from backtest.engine import BacktestEngine, BacktestConfig, TransactionCosts


# =============================================================================
# DATA LOADING
# =============================================================================

def load_historical_data_yfinance(years: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load historical data using yfinance.

    Args:
        years: Number of years to load

    Returns:
        Tuple of (prices_df, returns_df)
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Please run: pip install yfinance")
        return None, None

    logger.info(f"Loading {years} years of historical data from yfinance...")

    # Define tickers - yfinance format
    ticker_map = {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'AUDUSD': 'AUDUSD=X',
        'USDJPY': 'JPY=X',  # Inverted
        'USDCAD': 'CAD=X',  # Inverted
        'XAUUSD': 'GC=F',   # Gold futures
        'XAGUSD': 'SI=F',   # Silver futures
        'VIX': '^VIX'
    }

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    prices = {}

    for our_ticker, yf_ticker in ticker_map.items():
        try:
            logger.info(f"  Downloading {our_ticker} ({yf_ticker})...")
            data = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)

            if data is not None and len(data) > 0:
                # Use Adjusted Close if available, else Close
                if 'Adj Close' in data.columns:
                    price_series = data['Adj Close']
                else:
                    price_series = data['Close']

                # Handle inverted pairs (USDJPY, USDCAD)
                if our_ticker in ['USDJPY', 'USDCAD']:
                    # yfinance gives JPY per USD, CAD per USD - we want USD per foreign
                    # Actually yfinance JPY=X gives USD/JPY, CAD=X gives USD/CAD
                    pass  # No inversion needed

                prices[our_ticker] = price_series
                logger.info(f"    {our_ticker}: {len(price_series)} days loaded")
            else:
                logger.warning(f"    {our_ticker}: No data returned")

        except Exception as e:
            logger.error(f"    {our_ticker}: Error - {e}")

    if not prices:
        logger.error("No data loaded!")
        return None, None

    # Create DataFrames
    prices_df = pd.DataFrame(prices)
    prices_df = prices_df.dropna(how='all')

    # Forward fill missing values (holidays mismatch)
    prices_df = prices_df.ffill().bfill()

    # Compute returns
    returns_df = prices_df.pct_change(fill_method=None).fillna(0)

    logger.info(f"Loaded {len(prices_df)} days of data for {len(prices_df.columns)} instruments")
    logger.info(f"Date range: {prices_df.index[0].strftime('%Y-%m-%d')} to {prices_df.index[-1].strftime('%Y-%m-%d')}")

    return prices_df, returns_df


def generate_synthetic_historical_data(years: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic historical data if yfinance fails.
    Includes realistic crisis periods.
    """
    logger.info(f"Generating {years} years of synthetic historical data...")

    np.random.seed(42)

    # Generate business days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)

    logger.info(f"  Generating {n_days} business days...")

    # Instrument parameters (realistic historical behavior)
    instruments = {
        'EURUSD': {'base': 1.20, 'vol': 0.08, 'drift': -0.02},   # EUR weakened over decade
        'GBPUSD': {'base': 1.55, 'vol': 0.09, 'drift': -0.03},   # Brexit impact
        'AUDUSD': {'base': 0.90, 'vol': 0.11, 'drift': -0.02},   # Commodity weakness
        'USDJPY': {'base': 95, 'vol': 0.09, 'drift': 0.04},      # JPY weakness (Abenomics)
        'USDCAD': {'base': 1.05, 'vol': 0.07, 'drift': 0.02},    # Oil impact
        'XAUUSD': {'base': 1300, 'vol': 0.14, 'drift': 0.06},    # Gold appreciation
        'XAGUSD': {'base': 18, 'vol': 0.22, 'drift': 0.04},      # Silver appreciation
    }

    prices = {}
    returns = {}

    for ticker, params in instruments.items():
        daily_vol = params['vol'] / np.sqrt(252)
        daily_drift = params['drift'] / 252

        # Generate base returns
        daily_returns = np.random.normal(daily_drift, daily_vol, n_days)

        # Add volatility clustering
        vol_factor = np.ones(n_days)
        for i in range(1, n_days):
            vol_factor[i] = 0.92 * vol_factor[i-1] + 0.08 * (1 + 2 * abs(daily_returns[i-1]) / daily_vol)
        vol_factor = np.clip(vol_factor, 0.6, 2.0)
        daily_returns = daily_returns * vol_factor

        # Add crisis periods with realistic dates
        # 2015-2016 China/Oil crisis (approx day 500-600)
        crisis1_start = int(n_days * 0.2)
        if crisis1_start + 60 < n_days:
            daily_returns[crisis1_start:crisis1_start+60] *= 1.5

        # 2018 Vol spike (approx day 1000-1050)
        crisis2_start = int(n_days * 0.4)
        if crisis2_start + 50 < n_days:
            daily_returns[crisis2_start:crisis2_start+50] *= 1.4

        # COVID crash Feb-Mar 2020 (approx day 1800-1850)
        covid_start = int(n_days * 0.72)
        if covid_start + 50 < n_days:
            # Massive vol spike
            daily_returns[covid_start:covid_start+30] *= 3.0
            # Recovery
            daily_returns[covid_start+30:covid_start+50] *= 1.5
            # Safe haven assets (gold) rally
            if ticker == 'XAUUSD':
                daily_returns[covid_start:covid_start+30] = np.abs(daily_returns[covid_start:covid_start+30]) * 0.8

        # 2022 Fed hiking (approx day 2200-2400)
        fed_start = int(n_days * 0.88)
        if fed_start + 200 < n_days:
            if ticker == 'USDJPY':
                daily_returns[fed_start:fed_start+200] += 0.001  # JPY weakness
            if ticker in ['XAUUSD', 'XAGUSD']:
                daily_returns[fed_start:fed_start+100] -= 0.0005  # Metals weakness

        # Clip extreme returns
        daily_returns = np.clip(daily_returns, -0.06, 0.06)

        # Convert to prices
        price_series = params['base'] * np.cumprod(1 + daily_returns)

        prices[ticker] = pd.Series(price_series, index=dates)
        returns[ticker] = pd.Series(daily_returns, index=dates)

    # Generate VIX
    base_vix = 16
    vix = np.zeros(n_days)
    vix[0] = base_vix

    for i in range(1, n_days):
        mean_reversion = 0.03 * (base_vix - vix[i-1])
        shock = np.random.normal(0, 1.2)
        vix[i] = np.clip(vix[i-1] + mean_reversion + shock, 9, 80)

    # Crisis VIX spikes
    covid_start = int(n_days * 0.72)
    if covid_start + 60 < n_days:
        vix[covid_start:covid_start+20] = np.linspace(20, 82, 20)  # COVID spike to 82
        vix[covid_start+20:covid_start+60] = np.linspace(82, 30, 40)

    prices['VIX'] = pd.Series(vix, index=dates)

    prices_df = pd.DataFrame(prices)
    returns_df = pd.DataFrame(returns)

    logger.info(f"Generated {len(prices_df)} days of synthetic data")

    return prices_df, returns_df


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def compute_daily_signals(prices_df: pd.DataFrame,
                          returns_df: pd.DataFrame,
                          lookback: int = 63) -> pd.DataFrame:
    """
    Compute daily signals for all instruments.

    Args:
        prices_df: Price data
        returns_df: Returns data
        lookback: Minimum lookback for signals

    Returns:
        DataFrame of daily positions
    """
    logger.info("Computing daily signals and constructing portfolios...")

    tradeable = [c for c in prices_df.columns if c != 'VIX']
    dates = prices_df.index[lookback:]

    momentum_engine = MomentumSignalEngine()

    # Carry signals (static - would come from forward curves in production)
    carry_rates = {
        'EURUSD': -0.015, 'GBPUSD': -0.005, 'AUDUSD': 0.010,
        'USDJPY': 0.040, 'USDCAD': 0.008,
        'XAUUSD': 0.015, 'XAGUSD': 0.010
    }
    carry_signals = {t: carry_rates.get(t, 0) / 0.02 for t in tradeable}

    # Portfolio config
    config = PortfolioConfig(
        vol_target=0.12,
        max_single_position=0.25,
        max_daily_turnover=0.20,
        signal_weights={
            'momentum': 0.50,
            'carry': 0.50,
            'value': 0.0,
            'rates': 0.0,
        }
    )
    constructor = PortfolioConstructor(config)

    # Initialize positions DataFrame
    positions = pd.DataFrame(index=dates, columns=tradeable, dtype=float)

    total_days = len(dates)
    log_interval = total_days // 20  # Log every 5%

    for i, date in enumerate(dates):
        if i > 0 and i % log_interval == 0:
            pct = i / total_days * 100
            logger.info(f"  Progress: {pct:.0f}% ({i}/{total_days} days)")

        # Get data up to this date
        idx = prices_df.index.get_loc(date)

        # Compute momentum signals
        momentum_signals = {}
        for ticker in tradeable:
            try:
                price_slice = prices_df[ticker].iloc[:idx+1]
                if len(price_slice) > lookback:
                    signal = momentum_engine.compute_signal(price_slice, ticker=ticker)
                    momentum_signals[ticker] = signal.signal
                else:
                    momentum_signals[ticker] = 0.0
            except Exception:
                momentum_signals[ticker] = 0.0

        # Compute regime multiplier
        try:
            vix_slice = prices_df['VIX'].iloc[:idx+1] if 'VIX' in prices_df.columns else None
            if vix_slice is not None and len(vix_slice) > 20:
                _, vix_mult_series = compute_vix_regime_multiplier(vix_slice)
                regime_mult = float(vix_mult_series.iloc[-1])
            else:
                regime_mult = 1.0
        except Exception:
            regime_mult = 1.0

        # Construct portfolio
        try:
            returns_slice = returns_df[tradeable].iloc[:idx+1]
            portfolio = constructor.construct_portfolio(
                momentum_signals=momentum_signals,
                carry_signals=carry_signals,
                regime_multiplier=regime_mult,
                returns_df=returns_slice
            )
            positions.loc[date] = pd.Series(portfolio.final_weights)
        except Exception:
            if i > 0:
                positions.loc[date] = positions.iloc[i-1]
            else:
                positions.loc[date] = 0.0

    # Fill any remaining NaNs
    positions = positions.bfill().fillna(0)

    logger.info(f"  Completed signal generation for {len(positions)} days")

    return positions


# =============================================================================
# BACKTEST EXECUTION
# =============================================================================

def run_historical_backtest(positions_df: pd.DataFrame,
                            returns_df: pd.DataFrame,
                            prices_df: pd.DataFrame) -> Dict:
    """
    Run full backtest with transaction costs.
    """
    logger.info("Running backtest...")

    # Align data
    common_dates = positions_df.index.intersection(returns_df.index)
    positions_df = positions_df.loc[common_dates]
    returns_df = returns_df.loc[common_dates]

    # Remove VIX from returns
    tradeable = [c for c in positions_df.columns if c in returns_df.columns]
    returns_df = returns_df[tradeable]
    positions_df = positions_df[tradeable]

    # Configure backtest
    tc_config = TransactionCosts(
        fx_majors=0.5,
        fx_crosses=1.0,
        fx_em=3.0,
        gold=3.0,
        silver=10.0,
        sofr_swaps=0.3,
        impact_coefficient=0.0001
    )

    bt_config = BacktestConfig(
        initial_capital=100_000_000,
        include_tc=True,
        include_impact=True,
        target_vol=0.12,
        tc_config=tc_config
    )

    engine = BacktestEngine(bt_config)
    result = engine.run_backtest(
        positions_df=positions_df,
        returns_df=returns_df,
        prices_df=prices_df
    )

    return result


# =============================================================================
# REPORTING
# =============================================================================

def analyze_crisis_periods(equity: pd.Series,
                           returns: pd.Series) -> Dict:
    """
    Analyze performance during crisis periods.
    """
    crisis_periods = {
        'COVID_Feb_Apr_2020': ('2020-02-15', '2020-04-30'),
        'COVID_Full_2020': ('2020-02-01', '2020-06-30'),
        'Vol_Spike_2018': ('2018-01-15', '2018-03-31'),
        'Fed_Hiking_2022': ('2022-01-01', '2022-12-31'),
        'Banking_Crisis_2023': ('2023-03-01', '2023-05-31'),
    }

    results = {}

    for name, (start, end) in crisis_periods.items():
        try:
            start_date = pd.Timestamp(start)
            end_date = pd.Timestamp(end)

            # Check if period overlaps
            if start_date > equity.index[-1] or end_date < equity.index[0]:
                continue

            # Clip to available dates
            start_date = max(start_date, equity.index[0])
            end_date = min(end_date, equity.index[-1])

            period_equity = equity.loc[start_date:end_date]
            period_returns = returns.loc[start_date:end_date]

            if len(period_returns) < 5:
                continue

            period_return = (period_equity.iloc[-1] / period_equity.iloc[0]) - 1
            period_vol = period_returns.std() * np.sqrt(252)
            max_dd = (period_equity / period_equity.expanding().max() - 1).min()

            results[name] = {
                'return': period_return,
                'volatility': period_vol,
                'max_drawdown': max_dd,
                'sharpe': (period_returns.mean() * 252 - 0.05) / period_vol if period_vol > 0 else 0
            }
        except Exception:
            continue

    return results


def print_performance_report(result, crisis_analysis: Dict):
    """
    Print comprehensive performance report.
    """
    m = result.metrics

    print("\n" + "=" * 80)
    print("HISTORICAL BACKTEST PERFORMANCE REPORT")
    print("=" * 80)
    print(f"\nPeriod: {m.start_date.strftime('%Y-%m-%d')} to {m.end_date.strftime('%Y-%m-%d')} ({m.num_days} days)")
    print(f"Initial Capital: $100,000,000")

    print("\n" + "-" * 40)
    print("KEY METRICS")
    print("-" * 40)
    print(f"{'Total Return:':<25} {m.total_return:>12.2%}")
    print(f"{'Annualized Return:':<25} {m.annualized_return:>12.2%}")
    print(f"{'Annualized Volatility:':<25} {m.annualized_vol:>12.2%}")
    print(f"{'Sharpe Ratio:':<25} {m.sharpe_ratio:>12.2f}")
    print(f"{'Sortino Ratio:':<25} {m.sortino_ratio:>12.2f}")
    print(f"{'Calmar Ratio:':<25} {m.calmar_ratio:>12.2f}")

    print("\n" + "-" * 40)
    print("DRAWDOWN ANALYSIS")
    print("-" * 40)
    print(f"{'Max Drawdown:':<25} {m.max_drawdown:>12.2%}")
    print(f"{'Max DD Duration:':<25} {m.max_drawdown_duration:>12} days")
    print(f"{'Avg Drawdown:':<25} {m.avg_drawdown:>12.2%}")

    print("\n" + "-" * 40)
    print("DISTRIBUTION")
    print("-" * 40)
    print(f"{'Skewness:':<25} {m.skewness:>12.2f}")
    print(f"{'Kurtosis:':<25} {m.kurtosis:>12.2f}")
    print(f"{'VaR (95%):':<25} {m.var_95:>12.2%}")
    print(f"{'CVaR (95%):':<25} {m.cvar_95:>12.2%}")

    print("\n" + "-" * 40)
    print("TRADE STATISTICS")
    print("-" * 40)
    print(f"{'Win Rate:':<25} {m.win_rate:>12.1%}")
    print(f"{'Profit Factor:':<25} {m.profit_factor:>12.2f}")
    print(f"{'Avg Daily Turnover:':<25} {m.avg_daily_turnover:>12.2%}")
    print(f"{'Total TC Paid:':<25} ${m.total_tc_paid:>11,.0f}")
    print(f"{'Annual TC Drag:':<25} {m.tc_drag:>12.2%}")

    if crisis_analysis:
        print("\n" + "-" * 40)
        print("CRISIS PERIOD ANALYSIS")
        print("-" * 40)
        print(f"{'Period':<25} {'Return':>10} {'Max DD':>10} {'Sharpe':>10}")
        print("-" * 55)
        for name, stats in crisis_analysis.items():
            print(f"{name:<25} {stats['return']:>10.2%} {stats['max_drawdown']:>10.2%} {stats['sharpe']:>10.2f}")

    # Yearly returns
    if hasattr(result.attribution, 'yearly_returns') and len(result.attribution.yearly_returns) > 0:
        print("\n" + "-" * 40)
        print("YEARLY RETURNS")
        print("-" * 40)
        for date, ret in result.attribution.yearly_returns.items():
            year = date.year if hasattr(date, 'year') else str(date)[:4]
            print(f"  {year}: {ret:>10.2%}")

    # Attribution
    print("\n" + "-" * 40)
    print("RETURN ATTRIBUTION BY INSTRUMENT")
    print("-" * 40)
    for inst, ret in sorted(result.attribution.instrument_returns.items(),
                            key=lambda x: -abs(x[1])):
        print(f"  {inst:<15}: {ret:>10.2%}")

    print("\n" + "=" * 80)

    # Signal assessment
    if m.sharpe_ratio < 0.5:
        print("\n*** SHARPE RATIO < 0.5 - SIGNAL DEBUGGING NEEDED ***")
        print("Potential issues:")
        print("  - Momentum lookback may need tuning")
        print("  - Carry signal weights may be too high")
        print("  - Transaction costs may be too high")
        print("  - Position limits may be constraining alpha")
    else:
        print(f"\n*** Sharpe Ratio {m.sharpe_ratio:.2f} - Strategy is viable ***")


def plot_equity_curve_ascii(equity: pd.Series):
    """
    Simple ASCII equity curve visualization.
    """
    print("\n" + "-" * 40)
    print("EQUITY CURVE (ASCII)")
    print("-" * 40)

    # Resample to monthly for cleaner display
    monthly = equity.resample('ME').last()

    if len(monthly) < 3:
        print("Insufficient data for chart")
        return

    min_val = monthly.min()
    max_val = monthly.max()
    range_val = max_val - min_val if max_val != min_val else 1

    chart_width = 50
    chart_height = 15

    # Normalize to chart height
    normalized = ((monthly - min_val) / range_val * (chart_height - 1)).astype(int)

    # Create chart
    chart = [[' ' for _ in range(len(monthly))] for _ in range(chart_height)]

    for i, val in enumerate(normalized):
        row = chart_height - 1 - val
        chart[row][i] = '*'

    # Print chart
    print(f"\n${max_val/1e6:.1f}M |", end="")
    for row in chart:
        print("".join(row), end="|\n       |")
    print(f"\n${min_val/1e6:.1f}M |" + "_" * len(monthly) + "|")
    print(f"        {monthly.index[0].strftime('%Y')}{'':>{len(monthly)//2-4}}{monthly.index[-1].strftime('%Y')}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""
    print("=" * 80)
    print("HISTORICAL BACKTEST - FX/METALS CARRY STRATEGY")
    print("=" * 80)
    print()

    # Try to load real data, fall back to synthetic
    try:
        prices_df, returns_df = load_historical_data_yfinance(years=10)
        if prices_df is None or len(prices_df) < 252:
            raise ValueError("Insufficient yfinance data")
        data_source = "yfinance"
    except Exception as e:
        logger.warning(f"yfinance failed: {e}")
        logger.info("Falling back to synthetic data...")
        prices_df, returns_df = generate_synthetic_historical_data(years=10)
        data_source = "synthetic"

    print(f"\nData source: {data_source}")
    print(f"Data shape: {prices_df.shape}")
    print(f"Date range: {prices_df.index[0]} to {prices_df.index[-1]}")
    print()

    # Compute daily signals and positions
    positions_df = compute_daily_signals(prices_df, returns_df, lookback=63)

    # Run backtest
    result = run_historical_backtest(positions_df, returns_df, prices_df)

    # Analyze crisis periods
    crisis_analysis = analyze_crisis_periods(result.equity_curve, result.returns)

    # Print report
    print_performance_report(result, crisis_analysis)

    # Plot equity curve
    plot_equity_curve_ascii(result.equity_curve)

    # Return result for further analysis
    return result, prices_df, positions_df


if __name__ == '__main__':
    result, prices, positions = main()
