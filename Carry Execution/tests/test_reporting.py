"""
Test Daily Report Generation

Generates a sample daily report with dummy data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from reporting.daily import DailyReportGenerator


def generate_dummy_data():
    """Generate dummy data for report testing."""

    # Portfolio state
    portfolio_state = {
        'nav': 114_289_467,
        'positions': {
            'USDJPY': {
                'weight': 0.4830,
                'notional_usd': 55_200_000,
                'entry_price': 147.50,
                'current_price': 148.25,
                'unrealized_pnl': 425_000,
                'dv01': 1_380
            },
            'EURUSD': {
                'weight': -0.1498,
                'notional_usd': 17_120_000,
                'entry_price': 1.0920,
                'current_price': 1.0850,
                'unrealized_pnl': 110_000,
                'dv01': 427
            },
            'XAUUSD': {
                'weight': 0.2306,
                'notional_usd': 26_350_000,
                'entry_price': 2580.0,
                'current_price': 2650.0,
                'unrealized_pnl': 715_000,
                'dv01': 658
            },
            'GBPUSD': {
                'weight': -0.0662,
                'notional_usd': 7_565_000,
                'entry_price': 1.2720,
                'current_price': 1.2650,
                'unrealized_pnl': 42_000,
                'dv01': 189
            },
            'XAGUSD': {
                'weight': 0.0802,
                'notional_usd': 9_165_000,
                'entry_price': 29.50,
                'current_price': 30.25,
                'unrealized_pnl': 233_000,
                'dv01': 229
            },
            'USDCAD': {
                'weight': 0.0475,
                'notional_usd': 5_430_000,
                'entry_price': 1.3580,
                'current_price': 1.3650,
                'unrealized_pnl': 28_000,
                'dv01': 136
            },
            'AUDUSD': {
                'weight': 0.0116,
                'notional_usd': 1_325_000,
                'entry_price': 0.6520,
                'current_price': 0.6550,
                'unrealized_pnl': 6_000,
                'dv01': 33
            },
        },
        'risk': {
            'var_95_1d': 892_000,
            'var_99_1d': 1_456_000,
            'cvar_95_1d': 1_125_000,
            'total_dv01': 3_052,
            'fx_dv01': 2_165,
            'rates_dv01': -887,
            'gross_leverage': 1.07,
            'net_leverage': 0.64,
            'beta_to_spx': 0.12,
            'correlation_to_dxy': -0.35
        }
    }

    # Signals
    signals = {
        'USDJPY': {'momentum': 0.226, 'carry': 1.50, 'value': -0.15},
        'EURUSD': {'momentum': 0.170, 'carry': -0.75, 'value': 0.25},
        'XAUUSD': {'momentum': 0.581, 'carry': 1.00, 'value': 0.10},
        'GBPUSD': {'momentum': -0.099, 'carry': -0.25, 'value': 0.05},
        'XAGUSD': {'momentum': 0.400, 'carry': 0.75, 'value': -0.05},
        'USDCAD': {'momentum': 0.001, 'carry': 0.40, 'value': 0.00},
        'AUDUSD': {'momentum': -0.289, 'carry': 0.50, 'value': 0.15},
    }

    # Regime
    regime = {
        'vix_level': 18.5,
        'vix_regime': 'normal',
        'vix_multiplier': 1.0,
        'fxvol_level': 9.2,
        'fxvol_regime': 'normal',
        'fxvol_multiplier': 1.0,
        'drawdown_level': -0.035,
        'drawdown_regime': 'normal',
        'drawdown_multiplier': 1.0,
        'combined_multiplier': 1.0,
        'recommendation': 'full_risk'
    }

    # P&L
    pnl = {
        'yesterday_pnl': 537_788,
        'wtd_pnl': 1_245_000,
        'mtd_pnl': 2_890_000,
        'ytd_pnl': 14_289_467,
        'inception_pnl': 14_289_467,
        'yesterday_return': 0.0047,
        'wtd_return': 0.0110,
        'mtd_return': 0.0260,
        'ytd_return': 0.1429,
    }

    # Rates trades
    rates_trades = [
        {
            'instrument': 'SOFR_2Y',
            'tenor': '2Y',
            'direction': 'pay',
            'notional_usd': 25_000_000,
            'rate_bps': 435,
            'dv01': -487,
            'mtm_pnl': 12_500
        },
        {
            'instrument': 'SOFR_5Y',
            'tenor': '5Y',
            'direction': 'pay',
            'notional_usd': 10_000_000,
            'rate_bps': 365,
            'dv01': -470,
            'mtm_pnl': -8_200
        },
    ]

    # Equity history (60 days)
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=60, freq='B')
    returns = np.random.normal(0.0004, 0.006, 60)
    equity = 100_000_000 * np.cumprod(1 + returns)
    equity_series = pd.Series(equity, index=dates)

    # Constraints
    constraints = [
        {'name': 'Max Single Position', 'limit': 0.25, 'current': 0.2306, 'status': 'PASS', 'description': '25% limit per instrument'},
        {'name': 'Max Gross Leverage', 'limit': 4.0, 'current': 1.07, 'status': 'PASS', 'description': '400% gross exposure limit'},
        {'name': 'Max Drawdown', 'limit': 0.15, 'current': 0.035, 'status': 'PASS', 'description': '15% max drawdown threshold'},
        {'name': 'Min Cash Buffer', 'limit': 0.05, 'current': 0.08, 'status': 'PASS', 'description': '5% minimum cash buffer'},
        {'name': 'Max Concentration', 'limit': 0.40, 'current': 0.48, 'status': 'WARNING', 'description': '40% max in top position'},
        {'name': 'Max Daily Turnover', 'limit': 0.15, 'current': 0.089, 'status': 'PASS', 'description': '15% daily turnover limit'},
    ]

    return portfolio_state, signals, regime, pnl, equity_series, rates_trades, constraints


def test_generate_report():
    """Generate a sample daily report."""
    print("=" * 70)
    print("DAILY REPORT GENERATION TEST")
    print("=" * 70)
    print()

    # Generate dummy data
    print("Generating dummy data...")
    portfolio_state, signals, regime, pnl, equity, rates_trades, constraints = generate_dummy_data()

    # Create report generator
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    generator = DailyReportGenerator(output_dir=output_dir)

    # Generate report
    print("Generating HTML report...")
    filepath = generator.generate_report(
        portfolio_state=portfolio_state,
        signals=signals,
        regime=regime,
        pnl=pnl,
        equity=equity,
        rates_trades=rates_trades,
        constraints=constraints
    )

    print(f"\nReport saved to: {filepath}")
    print()

    # Print summary
    print("REPORT CONTENTS SUMMARY")
    print("-" * 40)
    print(f"  Portfolio NAV:        ${portfolio_state['nav']:,.0f}")
    print(f"  Number of Positions:  {len(portfolio_state['positions'])}")
    print(f"  Number of Rates Trades: {len(rates_trades)}")
    print(f"  VIX Level:            {regime['vix_level']:.1f}")
    print(f"  Regime Multiplier:    {regime['combined_multiplier']:.0%}")
    print(f"  Yesterday P&L:        ${pnl['yesterday_pnl']:,.0f}")
    print(f"  YTD P&L:              ${pnl['ytd_pnl']:,.0f}")
    print(f"  YTD Return:           {pnl['ytd_return']:.1%}")
    print()

    # Constraint summary
    print("CONSTRAINTS STATUS")
    print("-" * 40)
    for c in constraints:
        status_icon = "+" if c['status'] == 'PASS' else ("!" if c['status'] == 'WARNING' else "X")
        print(f"  [{status_icon}] {c['name']:<25} {c['current']:.1%} / {c['limit']:.0%}")

    print()
    print("=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOpen the report in a browser: file://{os.path.abspath(filepath)}")

    return filepath


if __name__ == '__main__':
    filepath = test_generate_report()
