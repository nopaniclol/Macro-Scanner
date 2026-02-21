"""
Test Backtest Engine

Runs a 3-year backtest on synthetic data demonstrating:
- Signal generation
- Portfolio construction with regime scaling
- Transaction cost modeling
- Performance metrics computation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import modules
from backtest.engine import BacktestEngine, BacktestConfig, TransactionCosts
from portfolio.construction import PortfolioConstructor, PortfolioConfig
from signals.momentum import MomentumSignalEngine
from signals.regime import compute_vix_regime_multiplier


def generate_3year_synthetic_data():
    """
    Generate 3 years of synthetic FX/Metals data.

    Creates realistic price dynamics with:
    - Trending behavior
    - Mean reversion
    - Volatility clustering
    - Crisis periods
    """
    np.random.seed(42)

    # 3 years of business days
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)

    print(f"Generating {n_days} days of synthetic data ({start_date.year}-{end_date.year})...")

    # Instruments and base prices - realistic FX volatility (8-12% annual)
    instruments = {
        'EURUSD': {'base': 1.10, 'vol': 0.08, 'drift': -0.01},
        'GBPUSD': {'base': 1.30, 'vol': 0.09, 'drift': -0.005},
        'AUDUSD': {'base': 0.72, 'vol': 0.10, 'drift': 0.005},
        'USDJPY': {'base': 115, 'vol': 0.09, 'drift': 0.02},
        'USDCAD': {'base': 1.28, 'vol': 0.07, 'drift': 0.005},
        'XAUUSD': {'base': 1800, 'vol': 0.14, 'drift': 0.03},
        'XAGUSD': {'base': 24, 'vol': 0.20, 'drift': 0.02},
    }

    # Generate price series
    prices = {}
    returns = {}

    for ticker, params in instruments.items():
        daily_vol = params['vol'] / np.sqrt(252)
        daily_drift = params['drift'] / 252

        # Generate simple random walk with drift
        daily_returns = np.random.normal(daily_drift, daily_vol, n_days)

        # Add mild volatility clustering (GARCH-lite)
        vol_factor = np.ones(n_days)
        for i in range(1, n_days):
            # Vol increases after big moves
            vol_factor[i] = 0.9 * vol_factor[i-1] + 0.1 * (1 + abs(daily_returns[i-1]) / daily_vol)
        vol_factor = np.clip(vol_factor, 0.7, 1.5)
        daily_returns = daily_returns * vol_factor

        # Add crisis shocks (but moderate)
        # Simulated volatility spike in March 2022
        crisis_start = 252 + 40
        crisis_end = crisis_start + 20
        if crisis_end < n_days:
            daily_returns[crisis_start:crisis_end] *= 1.8
            if ticker == 'XAUUSD':
                daily_returns[crisis_start:crisis_end] = np.abs(daily_returns[crisis_start:crisis_end]) * 0.5

        # Clip extreme returns to prevent overflow (max 5% daily move)
        daily_returns = np.clip(daily_returns, -0.05, 0.05)

        # Convert to prices
        price_series = params['base'] * np.cumprod(1 + daily_returns)

        prices[ticker] = pd.Series(price_series, index=dates)
        returns[ticker] = pd.Series(daily_returns, index=dates)

    # Generate VIX (mean-reverting around 18)
    base_vix = 18
    vix = np.zeros(n_days)
    vix[0] = base_vix

    for i in range(1, n_days):
        # Mean reverting VIX
        mean_reversion = 0.05 * (base_vix - vix[i-1])
        shock = np.random.normal(0, 1.0)
        vix[i] = np.clip(vix[i-1] + mean_reversion + shock, 10, 40)

    # Crisis spike
    crisis_start = 252 + 40
    if crisis_start + 40 < n_days:
        vix[crisis_start:crisis_start+20] = np.linspace(18, 35, 20)
        vix[crisis_start+20:crisis_start+40] = np.linspace(35, 22, 20)

    prices['VIX'] = pd.Series(vix, index=dates)

    return pd.DataFrame(prices), pd.DataFrame(returns)


def generate_positions_from_signals(prices_df: pd.DataFrame,
                                     returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate daily positions based on momentum + carry signals with regime scaling.
    """
    print("Generating positions from signals...")

    momentum_engine = MomentumSignalEngine()
    portfolio_config = PortfolioConfig(
        vol_target=0.12,
        max_single_position=0.25,
        signal_weights={'momentum': 0.6, 'carry': 0.4, 'value': 0.0, 'rates': 0.0}
    )
    constructor = PortfolioConstructor(portfolio_config)

    tickers = [c for c in prices_df.columns if c != 'VIX']

    # Carry signals (static for simplicity)
    carry_rates = {
        'EURUSD': -0.015, 'GBPUSD': -0.005, 'AUDUSD': 0.010,
        'USDJPY': 0.045, 'USDCAD': 0.008, 'XAUUSD': 0.020, 'XAGUSD': 0.015
    }
    carry_signals = {t: c / 0.02 for t, c in carry_rates.items()}

    # Initialize positions DataFrame
    positions = pd.DataFrame(index=prices_df.index, columns=tickers, dtype=float)
    positions.iloc[0] = 0

    # Lookback for momentum
    lookback = 63

    for i in range(lookback, len(prices_df)):
        date = prices_df.index[i]

        # Get momentum signals
        momentum_signals = {}
        for ticker in tickers:
            prices_slice = prices_df[ticker].iloc[:i+1]
            try:
                signal = momentum_engine.compute_signal(prices_slice, ticker=ticker)
                momentum_signals[ticker] = signal.signal
            except:
                momentum_signals[ticker] = 0.0

        # Get regime multiplier
        vix_slice = prices_df['VIX'].iloc[:i+1]
        try:
            _, vix_mult_series = compute_vix_regime_multiplier(vix_slice)
            regime_mult = vix_mult_series.iloc[-1]
        except:
            regime_mult = 1.0

        # Get returns for vol calculation
        returns_slice = returns_df[tickers].iloc[:i+1]

        # Construct portfolio
        try:
            portfolio = constructor.construct_portfolio(
                momentum_signals=momentum_signals,
                carry_signals=carry_signals,
                regime_multiplier=regime_mult,
                returns_df=returns_slice
            )
            positions.loc[date] = pd.Series(portfolio.final_weights)
        except Exception as e:
            # Keep previous positions on error
            if i > lookback:
                positions.loc[date] = positions.iloc[i-1]

    # Forward fill the first few days
    positions = positions.bfill().fillna(0)

    return positions


def run_backtest_demo():
    """
    Run full 3-year backtest demonstration.
    """
    print("=" * 70)
    print("3-YEAR BACKTEST DEMONSTRATION")
    print("=" * 70)
    print()

    # Generate data
    prices_df, returns_df = generate_3year_synthetic_data()
    print(f"  Data shape: {prices_df.shape}")
    print()

    # Generate positions
    positions_df = generate_positions_from_signals(prices_df, returns_df)

    # Remove VIX from returns (not traded)
    returns_for_bt = returns_df.drop(columns=['VIX'], errors='ignore')

    # Align positions to returns columns
    positions_for_bt = positions_df[[c for c in positions_df.columns if c in returns_for_bt.columns]]

    print(f"  Positions shape: {positions_for_bt.shape}")
    print()

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

    # Run backtest
    print("Running backtest...")
    engine = BacktestEngine(bt_config)
    result = engine.run_backtest(
        positions_df=positions_for_bt,
        returns_df=returns_for_bt,
        prices_df=prices_df
    )

    # Print report
    report = engine.generate_report(result)
    print(report)

    # Additional visualizations in text
    print("\nMONTHLY RETURNS")
    print("-" * 40)
    for date, ret in result.attribution.monthly_returns.items():
        print(f"  {date.strftime('%Y-%m')}: {ret:>8.2%}")

    print("\nYEARLY RETURNS")
    print("-" * 40)
    for date, ret in result.attribution.yearly_returns.items():
        print(f"  {date.strftime('%Y')}: {ret:>8.2%}")

    # Equity curve summary
    print("\nEQUITY CURVE SUMMARY")
    print("-" * 40)
    print(f"  Starting NAV:  ${result.equity_curve.iloc[0]:>15,.0f}")
    print(f"  Ending NAV:    ${result.equity_curve.iloc[-1]:>15,.0f}")
    print(f"  Peak NAV:      ${result.equity_curve.max():>15,.0f}")
    print(f"  Trough NAV:    ${result.equity_curve.min():>15,.0f}")

    # Position statistics
    print("\nPOSITION STATISTICS")
    print("-" * 40)
    avg_positions = result.positions.abs().mean()
    print("  Average Position Size by Instrument:")
    for ticker, avg in avg_positions.sort_values(ascending=False).items():
        print(f"    {ticker:<12}: {avg:>6.2%}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    return result


if __name__ == '__main__':
    result = run_backtest_demo()
