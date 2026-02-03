"""
Test Portfolio Construction with Regime Scaling

Demonstrates the full workflow:
1. Signal generation (momentum + carry)
2. Regime detection and scaling
3. Portfolio construction
4. Rates overlay with SOFR hedging
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import portfolio modules
from portfolio.construction import PortfolioConstructor, PortfolioConfig
from portfolio.rates_hedge import RatesOverlayEngine, FXPosition

# Import signal modules
from signals.momentum import MomentumSignalEngine, compute_tsmom_ensemble
from signals.carry import CarrySignalEngine
from signals.regime import RegimeSignalEngine, compute_vix_regime_multiplier, compute_drawdown_regime_multiplier


def generate_mock_data():
    """Generate mock price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='B')

    # Generate synthetic FX prices with trends
    tickers = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'USDCAD', 'XAUUSD', 'XAGUSD']
    base_prices = {'EURUSD': 1.08, 'GBPUSD': 1.26, 'AUDUSD': 0.65,
                   'USDJPY': 148, 'USDCAD': 1.36, 'XAUUSD': 2500, 'XAGUSD': 29}

    price_data = {}
    returns_data = {}

    for ticker in tickers:
        # Random walk with slight drift
        returns = np.random.normal(0.0001, 0.008, len(dates))
        prices = base_prices[ticker] * np.cumprod(1 + returns)
        price_data[ticker] = pd.Series(prices, index=dates)
        returns_data[ticker] = pd.Series(returns, index=dates)

    # VIX data (mean-reverting around 18)
    vix = 18 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
    vix = np.clip(vix, 10, 45)
    price_data['VIX'] = pd.Series(vix, index=dates)

    return price_data, returns_data


def generate_mock_carry_data():
    """Generate mock carry (forward points) data."""
    tickers = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'USDCAD', 'XAUUSD', 'XAGUSD']

    # Annualized carry estimates (rate differentials)
    carry_rates = {
        'EURUSD': -0.015,   # EUR rates lower than USD
        'GBPUSD': -0.005,   # GBP slightly lower
        'AUDUSD': 0.010,    # AUD higher (positive carry long AUD)
        'USDJPY': 0.045,    # JPY much lower (positive carry long USD/JPY)
        'USDCAD': 0.008,    # CAD slightly lower
        'XAUUSD': 0.020,    # Gold lease rate
        'XAGUSD': 0.015,    # Silver lease rate
    }

    return carry_rates


def test_full_portfolio_workflow():
    """
    Complete portfolio construction workflow with regime scaling.
    """
    print("=" * 70)
    print("PORTFOLIO CONSTRUCTION WITH REGIME SCALING - FULL TEST")
    print("=" * 70)
    print()

    # =========================================================================
    # STEP 1: GENERATE MOCK DATA
    # =========================================================================
    print("STEP 1: Generating mock market data...")
    price_data, returns_data = generate_mock_data()
    carry_rates = generate_mock_carry_data()
    print(f"  Generated {len(price_data)} price series, 252 days each")
    print()

    # =========================================================================
    # STEP 2: COMPUTE MOMENTUM SIGNALS
    # =========================================================================
    print("STEP 2: Computing momentum signals...")
    momentum_engine = MomentumSignalEngine()

    momentum_signals = {}
    for ticker, prices in price_data.items():
        if ticker == 'VIX':
            continue
        signal = momentum_engine.compute_signal(prices, ticker=ticker)
        momentum_signals[ticker] = signal.signal
        print(f"  {ticker}: TSMOM = {signal.signal:+.3f} ({signal.direction})")
    print()

    # =========================================================================
    # STEP 3: COMPUTE CARRY SIGNALS
    # =========================================================================
    print("STEP 3: Computing carry signals...")
    carry_engine = CarrySignalEngine()

    carry_signals = {}
    for ticker, annual_carry in carry_rates.items():
        # Normalize to z-score-like signal
        carry_z = annual_carry / 0.02  # Scale so 2% carry = 1.0 signal
        carry_z = np.clip(carry_z, -1.5, 1.5)
        carry_signals[ticker] = carry_z
        print(f"  {ticker}: Carry = {annual_carry*100:+.2f}% -> signal = {carry_z:+.3f}")
    print()

    # =========================================================================
    # STEP 4: DETECT REGIME AND COMPUTE MULTIPLIER
    # =========================================================================
    print("STEP 4: Detecting market regime...")
    regime_engine = RegimeSignalEngine()

    vix_series = price_data['VIX']
    current_vix = vix_series.iloc[-1]

    # VIX regime - use standalone function (returns tuple of series)
    vix_regime_series, vix_mult_series = compute_vix_regime_multiplier(vix_series)
    vix_mult = vix_mult_series.iloc[-1]
    vix_regime = vix_regime_series.iloc[-1]
    print(f"  Current VIX: {current_vix:.1f}")
    print(f"  VIX Regime: {vix_regime}")
    print(f"  VIX Regime Multiplier: {vix_mult:.2f}")

    # Compute drawdown regime for SPX proxy (use EURUSD as proxy)
    dd_regime_series, dd_mult_series = compute_drawdown_regime_multiplier(price_data['EURUSD'])
    dd_mult = dd_mult_series.iloc[-1]
    dd_regime = dd_regime_series.iloc[-1]
    print(f"  Drawdown Regime: {dd_regime}")
    print(f"  Drawdown Multiplier: {dd_mult:.2f}")

    # Combined regime
    combined_mult = min(vix_mult, dd_mult)
    print(f"  Combined Regime Multiplier: {combined_mult:.2f}")
    print()

    # =========================================================================
    # STEP 5: CONSTRUCT PORTFOLIO
    # =========================================================================
    print("STEP 5: Constructing portfolio...")

    # Create returns DataFrame for volatility calculation
    returns_df = pd.DataFrame(returns_data)

    # Configure portfolio constructor
    config = PortfolioConfig(
        vol_target=0.12,
        max_single_position=0.25,
        max_daily_turnover=0.15,
        signal_weights={
            'momentum': 0.6,
            'carry': 0.4,
            'value': 0.0,
            'rates': 0.0,
        }
    )

    constructor = PortfolioConstructor(config)

    # Construct portfolio with regime scaling
    portfolio = constructor.construct_portfolio(
        momentum_signals=momentum_signals,
        carry_signals=carry_signals,
        value_tilts=None,
        regime_multiplier=combined_mult,
        returns_df=returns_df
    )

    print(f"\n  Portfolio Summary:")
    print(f"  {'Ticker':<12} {'Weight':>10} {'Direction':<10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")

    for ticker, weight in sorted(portfolio.final_weights.items(), key=lambda x: -abs(x[1])):
        direction = 'LONG' if weight > 0.01 else ('SHORT' if weight < -0.01 else 'FLAT')
        print(f"  {ticker:<12} {weight:>+10.2%} {direction:<10}")

    print(f"\n  Gross Exposure: {portfolio.gross_exposure:.2%}")
    print(f"  Net Exposure:   {portfolio.net_exposure:.2%}")
    print(f"  Leverage:       {portfolio.leverage:.2f}x")
    print()

    # =========================================================================
    # STEP 6: RATES OVERLAY - SOFR HEDGING
    # =========================================================================
    print("STEP 6: Computing rates overlay (SOFR hedge)...")

    rates_engine = RatesOverlayEngine()

    # Create FX positions from portfolio weights
    # Assume $100M portfolio notional
    portfolio_notional = 100_000_000

    fx_positions = {}
    for ticker, weight in portfolio.final_weights.items():
        if abs(weight) < 0.01:
            continue
        if ticker in ['VIX']:
            continue

        notional = abs(weight * portfolio_notional)
        direction = 'long' if weight > 0 else 'short'

        fx_positions[ticker] = rates_engine.create_fx_position(
            ticker=ticker,
            notional_usd=notional,
            direction=direction,
            tenor_days=91  # 3M forwards
        )

    # Generate hedge recommendation
    hedge_result = rates_engine.analyze_and_hedge(
        fx_positions=fx_positions,
        hedge_ratio=0.6,
        max_tenors=3
    )

    # Print hedge summary
    print(rates_engine.get_hedge_summary(hedge_result))

    # =========================================================================
    # STEP 7: GENERATE TRADE ORDERS
    # =========================================================================
    print("\nSTEP 7: Generating trade orders...")

    # Get trade orders from portfolio constructor
    orders = constructor.generate_trade_orders(
        target_weights=portfolio.final_weights,
        current_weights={},  # Start from flat
        nav=portfolio_notional
    )

    print(f"\n  FX/Metals Trade Orders:")
    print(f"  {'Ticker':<12} {'Action':<6} {'Notional':>15} {'Weight':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*15} {'-'*10}")

    for order in sorted(orders, key=lambda x: -abs(x.notional)):
        print(f"  {order.ticker:<12} {order.direction:<6} ${order.notional:>14,.0f} {order.target_weight:>+10.2%}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL PORTFOLIO STATE")
    print("=" * 70)
    print(f"""
    Portfolio NAV:        ${portfolio_notional:,.0f}

    POSITIONS:
    - FX Pairs:           {sum(1 for t in portfolio.final_weights if t in ['EURUSD','GBPUSD','AUDUSD','USDJPY','USDCAD'])}
    - Metals:             {sum(1 for t in portfolio.final_weights if t in ['XAUUSD','XAGUSD'])}

    RISK METRICS:
    - Target Vol:         {config.vol_target:.0%} annualized
    - Gross Exposure:     {portfolio.gross_exposure:.2%}
    - Net Exposure:       {portfolio.net_exposure:.2%}
    - Leverage:           {portfolio.leverage:.2f}x

    REGIME STATE:
    - VIX:                {current_vix:.1f}
    - Regime Multiplier:  {combined_mult:.2f}x

    RATES HEDGE:
    - FX DV01:            ${hedge_result.fx_dv01_total:,.0f}
    - Hedge Ratio:        {hedge_result.hedge_ratio:.0%}
    - Hedge Trades:       {len(hedge_result.trades)}
    - Residual DV01:      ${hedge_result.residual_dv01:,.0f}
    - Hedge Efficiency:   {hedge_result.hedge_efficiency:.1%}
    """)

    print("=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return portfolio, hedge_result


def test_regime_scenarios():
    """
    Test portfolio behavior under different regime scenarios.
    """
    print("\n" + "=" * 70)
    print("REGIME SCENARIO ANALYSIS")
    print("=" * 70)

    # Generate base data
    price_data, returns_data = generate_mock_data()
    carry_rates = generate_mock_carry_data()

    momentum_engine = MomentumSignalEngine()
    momentum_signals = {}
    for ticker, prices in price_data.items():
        if ticker == 'VIX':
            continue
        signal = momentum_engine.compute_signal(prices, ticker=ticker)
        momentum_signals[ticker] = signal.signal

    carry_signals = {t: c / 0.02 for t, c in carry_rates.items()}

    returns_df = pd.DataFrame(returns_data)

    config = PortfolioConfig(vol_target=0.12)
    constructor = PortfolioConstructor(config)

    # Test different regime multipliers
    scenarios = [
        ("RISK-ON (Low VIX)", 1.0),
        ("NORMAL", 0.8),
        ("ELEVATED VOL", 0.5),
        ("CRISIS (High VIX)", 0.2),
    ]

    print(f"\n{'Scenario':<25} {'Regime Mult':>12} {'Gross Exp':>12} {'Net Exp':>12}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12}")

    for name, regime_mult in scenarios:
        portfolio = constructor.construct_portfolio(
            momentum_signals=momentum_signals,
            carry_signals=carry_signals,
            regime_multiplier=regime_mult,
            returns_df=returns_df
        )

        print(f"{name:<25} {regime_mult:>12.2f} {portfolio.gross_exposure:>12.2%} {portfolio.net_exposure:>+12.2%}")

    print()


if __name__ == '__main__':
    # Run main test
    portfolio, hedge = test_full_portfolio_workflow()

    # Run regime scenario analysis
    test_regime_scenarios()
