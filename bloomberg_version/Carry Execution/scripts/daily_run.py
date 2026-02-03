#!/usr/bin/env python3
"""
Daily Execution Script

Main workflow for FX/Metals Carry Strategy:
1. Load config
2. Initialize data loader (free or Bloomberg)
3. Load market data
4. Compute signals
5. Construct portfolio
6. Generate rates trades
7. Run risk checks
8. Generate daily report
9. Save outputs

Usage:
    python daily_run.py --config config/parameters.yaml --data-source free
    python daily_run.py --data-source bloomberg --lookback 100
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pandas as pd
import numpy as np

# Optional yaml import
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Import modules
from data.free_data_loader import FreeDataLoader
from signals.momentum import MomentumSignalEngine
from signals.carry import CarrySignalEngine
from signals.regime import compute_vix_regime_multiplier, compute_drawdown_regime_multiplier
from portfolio.construction import PortfolioConstructor, PortfolioConfig
from portfolio.rates_hedge import RatesOverlayEngine
from reporting.daily import DailyReportGenerator


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: str, run_date: datetime) -> logging.Logger:
    """
    Setup logging with file and console handlers.

    Args:
        log_dir: Directory for log files
        run_date: Date for log filename

    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{run_date.strftime('%Y%m%d')}_run.log")

    # Create logger
    logger = logging.getLogger('daily_run')
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers = []

    # File handler (detailed)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler (info only)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# =============================================================================
# CONFIG LOADING
# =============================================================================

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML or use defaults.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    default_config = {
        'data_source': 'free',
        'lookback_days': 252,
        'tickers': {
            'fx': ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'USDCAD'],
            'metals': ['XAUUSD', 'XAGUSD'],
            'rates': ['VIX']
        },
        'signals': {
            'momentum_weight': 0.6,
            'carry_weight': 0.4,
            'value_weight': 0.0,
            'rates_weight': 0.0
        },
        'portfolio': {
            'target_vol': 0.12,
            'max_position': 0.25,
            'max_leverage': 3.0,
            'max_turnover': 0.15
        },
        'rates_hedge': {
            'enabled': True,
            'hedge_ratio': 0.6,
            'max_tenors': 3
        },
        'risk_limits': {
            'max_drawdown': 0.15,
            'max_var_95': 0.02,
            'min_liquidity': 0.05
        },
        'reporting': {
            'output_dir': 'reports',
            'generate_html': True
        }
    }

    if config_path and os.path.exists(config_path) and HAS_YAML:
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
            # Merge with defaults
            for key, value in file_config.items():
                if isinstance(value, dict) and key in default_config:
                    default_config[key].update(value)
                else:
                    default_config[key] = value

    return default_config


# =============================================================================
# DATA LOADING
# =============================================================================

def load_market_data(config: Dict, lookback: int, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """
    Load market data based on configured data source.

    Args:
        config: Configuration dictionary
        lookback: Number of days to load
        logger: Logger instance

    Returns:
        Dictionary of DataFrames
    """
    data_source = config.get('data_source', 'free')
    logger.info(f"Loading data from source: {data_source}")

    if data_source == 'bloomberg':
        logger.info("Bloomberg data source selected - using Bloomberg loaders")
        # In production, would import and use BloombergDataLoader
        # For now, fall back to free data with a warning
        logger.warning("Bloomberg loaders require BQUANT environment. Falling back to free data.")
        data_source = 'free'

    if data_source == 'free':
        loader = FreeDataLoader()

        # Get all tickers
        fx_tickers = config['tickers'].get('fx', [])
        metals_tickers = config['tickers'].get('metals', [])

        # Load price data
        logger.info(f"Loading {lookback} days of data for {len(fx_tickers) + len(metals_tickers)} instruments...")

        prices = {}
        returns = {}

        # Load FX data
        for ticker in fx_tickers:
            try:
                df = loader.fetch_fx_spot(ticker, lookback)
                if df is not None and not df.empty:
                    price_series = df['close'] if 'close' in df.columns else df.iloc[:, 0]
                    prices[ticker] = price_series
                    returns[ticker] = price_series.pct_change().fillna(0)
                    logger.debug(f"  {ticker}: {len(price_series)} days loaded")
                else:
                    logger.warning(f"  {ticker}: No data returned")
            except Exception as e:
                logger.error(f"  {ticker}: Error loading - {e}")

        # Load metals data
        for ticker in metals_tickers:
            try:
                df = loader.fetch_metal_spot(ticker, lookback)
                if df is not None and not df.empty:
                    price_series = df['close'] if 'close' in df.columns else df.iloc[:, 0]
                    prices[ticker] = price_series
                    returns[ticker] = price_series.pct_change().fillna(0)
                    logger.debug(f"  {ticker}: {len(price_series)} days loaded")
                else:
                    logger.warning(f"  {ticker}: No data returned")
            except Exception as e:
                logger.error(f"  {ticker}: Error loading - {e}")

        # Load VIX
        try:
            vix_df = loader.fetch_risk_indicator('VIX', lookback)
            if vix_df is not None and not vix_df.empty:
                vix_col = vix_df['close'] if 'close' in vix_df.columns else vix_df.iloc[:, 0]
                # Validate VIX is numeric
                if pd.api.types.is_numeric_dtype(vix_col):
                    prices['VIX'] = vix_col
                    logger.debug(f"  VIX: {len(prices['VIX'])} days loaded")
                else:
                    raise ValueError("VIX data is not numeric")
        except Exception as e:
            logger.warning(f"  VIX: Error loading - {e}")

        # Generate synthetic VIX as fallback if not loaded
        if 'VIX' not in prices and prices:
            logger.info("  Generating synthetic VIX data")
            first_ticker = list(prices.keys())[0]
            n_days = len(prices[first_ticker])
            np.random.seed(42)
            vix = 18 + np.cumsum(np.random.normal(0, 0.5, n_days))
            vix = np.clip(vix, 10, 40)
            prices['VIX'] = pd.Series(vix, index=prices[first_ticker].index)

        logger.info(f"Loaded data for {len(prices)} instruments")

        return {
            'prices': pd.DataFrame(prices),
            'returns': pd.DataFrame(returns)
        }

    else:
        raise ValueError(f"Unknown data source: {data_source}")


# =============================================================================
# SIGNAL COMPUTATION
# =============================================================================

def compute_signals(prices_df: pd.DataFrame,
                    returns_df: pd.DataFrame,
                    config: Dict,
                    logger: logging.Logger) -> Dict[str, Dict[str, float]]:
    """
    Compute all signals for portfolio construction.

    Args:
        prices_df: Price data
        returns_df: Returns data
        config: Configuration
        logger: Logger instance

    Returns:
        Dictionary of signals by instrument
    """
    logger.info("Computing signals...")

    # Momentum signals
    momentum_engine = MomentumSignalEngine()
    momentum_signals = {}

    tradeable = [c for c in prices_df.columns if c != 'VIX']

    for ticker in tradeable:
        try:
            signal = momentum_engine.compute_signal(prices_df[ticker], ticker=ticker)
            momentum_signals[ticker] = signal.signal
            logger.debug(f"  {ticker} momentum: {signal.signal:+.3f}")
        except Exception as e:
            logger.warning(f"  {ticker} momentum error: {e}")
            momentum_signals[ticker] = 0.0

    # Carry signals (static for free data)
    carry_rates = {
        'EURUSD': -0.015, 'GBPUSD': -0.005, 'AUDUSD': 0.010,
        'USDJPY': 0.045, 'USDCAD': 0.008,
        'XAUUSD': 0.020, 'XAGUSD': 0.015
    }
    carry_signals = {t: carry_rates.get(t, 0) / 0.02 for t in tradeable}

    for ticker in tradeable:
        logger.debug(f"  {ticker} carry: {carry_signals.get(ticker, 0):+.3f}")

    # Regime signals
    regime_data = {}
    if 'VIX' in prices_df.columns:
        vix_series = prices_df['VIX']
        try:
            vix_regime, vix_mult = compute_vix_regime_multiplier(vix_series)
            regime_data['vix_level'] = vix_series.iloc[-1]
            regime_data['vix_regime'] = vix_regime.iloc[-1]
            regime_data['vix_multiplier'] = vix_mult.iloc[-1]
            logger.info(f"  VIX: {regime_data['vix_level']:.1f} ({regime_data['vix_regime']})")
        except Exception as e:
            logger.warning(f"  VIX regime error: {e}")
            regime_data['vix_level'] = 18.0
            regime_data['vix_regime'] = 'normal'
            regime_data['vix_multiplier'] = 1.0

    # Drawdown regime (use first FX pair as proxy)
    if tradeable:
        try:
            proxy = prices_df[tradeable[0]]
            dd_regime, dd_mult = compute_drawdown_regime_multiplier(proxy)
            regime_data['drawdown_level'] = dd_mult.iloc[-1]
            regime_data['drawdown_regime'] = dd_regime.iloc[-1]
            regime_data['drawdown_multiplier'] = dd_mult.iloc[-1]
        except Exception as e:
            logger.warning(f"  Drawdown regime error: {e}")
            regime_data['drawdown_regime'] = 'normal'
            regime_data['drawdown_multiplier'] = 1.0

    # Combined multiplier
    regime_data['combined_multiplier'] = min(
        regime_data.get('vix_multiplier', 1.0),
        regime_data.get('drawdown_multiplier', 1.0)
    )
    regime_data['recommendation'] = (
        'full_risk' if regime_data['combined_multiplier'] >= 0.8 else
        'reduce' if regime_data['combined_multiplier'] >= 0.5 else
        'defensive'
    )

    logger.info(f"  Combined regime multiplier: {regime_data['combined_multiplier']:.2f}")

    # Combine signals
    signals = {}
    for ticker in tradeable:
        signals[ticker] = {
            'momentum': momentum_signals.get(ticker, 0),
            'carry': carry_signals.get(ticker, 0),
            'value': 0.0  # Placeholder
        }

    return {
        'signals': signals,
        'momentum': momentum_signals,
        'carry': carry_signals,
        'regime': regime_data
    }


# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================

def construct_portfolio(signals: Dict,
                         returns_df: pd.DataFrame,
                         config: Dict,
                         logger: logging.Logger) -> Dict:
    """
    Construct portfolio from signals.

    Args:
        signals: Signal data
        returns_df: Returns data
        config: Configuration
        logger: Logger instance

    Returns:
        Portfolio data dictionary
    """
    logger.info("Constructing portfolio...")

    portfolio_config = PortfolioConfig(
        vol_target=config['portfolio'].get('target_vol', 0.12),
        max_single_position=config['portfolio'].get('max_position', 0.25),
        max_daily_turnover=config['portfolio'].get('max_turnover', 0.15),
        signal_weights={
            'momentum': config['signals'].get('momentum_weight', 0.6),
            'carry': config['signals'].get('carry_weight', 0.4),
            'value': config['signals'].get('value_weight', 0.0),
            'rates': config['signals'].get('rates_weight', 0.0),
        }
    )

    constructor = PortfolioConstructor(portfolio_config)

    # Remove VIX from returns
    tradeable_returns = returns_df.drop(columns=['VIX'], errors='ignore')

    portfolio = constructor.construct_portfolio(
        momentum_signals=signals['momentum'],
        carry_signals=signals['carry'],
        regime_multiplier=signals['regime']['combined_multiplier'],
        returns_df=tradeable_returns
    )

    logger.info(f"  Gross exposure: {portfolio.gross_exposure:.1%}")
    logger.info(f"  Net exposure: {portfolio.net_exposure:.1%}")
    logger.info(f"  Leverage: {portfolio.leverage:.2f}x")

    # Log positions
    for ticker, weight in sorted(portfolio.final_weights.items(), key=lambda x: -abs(x[1])):
        if abs(weight) > 0.01:
            direction = 'LONG' if weight > 0 else 'SHORT'
            logger.info(f"  {ticker}: {weight:+.1%} ({direction})")

    return {
        'weights': portfolio.final_weights,
        'raw_weights': portfolio.raw_weights,
        'gross_exposure': portfolio.gross_exposure,
        'net_exposure': portfolio.net_exposure,
        'leverage': portfolio.leverage,
        'vol_forecast': portfolio.vol_forecast
    }


# =============================================================================
# RATES OVERLAY
# =============================================================================

def generate_rates_trades(portfolio: Dict,
                          config: Dict,
                          logger: logging.Logger) -> Dict:
    """
    Generate SOFR rates overlay trades.

    Args:
        portfolio: Portfolio data
        config: Configuration
        logger: Logger instance

    Returns:
        Rates trades data
    """
    if not config['rates_hedge'].get('enabled', True):
        logger.info("Rates hedge disabled in config")
        return {'trades': [], 'hedge_result': None}

    logger.info("Generating rates overlay...")

    rates_engine = RatesOverlayEngine()

    # Create FX positions from portfolio weights
    portfolio_notional = 100_000_000  # $100M notional

    fx_positions = {}
    for ticker, weight in portfolio['weights'].items():
        if abs(weight) < 0.01:
            continue

        notional = abs(weight * portfolio_notional)
        direction = 'long' if weight > 0 else 'short'

        fx_positions[ticker] = rates_engine.create_fx_position(
            ticker=ticker,
            notional_usd=notional,
            direction=direction,
            tenor_days=91
        )

    # Generate hedge
    hedge_result = rates_engine.analyze_and_hedge(
        fx_positions=fx_positions,
        hedge_ratio=config['rates_hedge'].get('hedge_ratio', 0.6),
        max_tenors=config['rates_hedge'].get('max_tenors', 3)
    )

    logger.info(f"  FX DV01: ${hedge_result.fx_dv01_total:,.0f}")
    logger.info(f"  Hedge efficiency: {hedge_result.hedge_efficiency:.1%}")

    # Format trades for output
    trades = []
    for trade in hedge_result.trades:
        trades.append({
            'instrument': trade.instrument,
            'tenor': f"{trade.tenor_years:.0f}Y",
            'direction': trade.direction,
            'notional_usd': trade.notional_usd,
            'rate_bps': trade.rate_bps,
            'dv01': trade.dv01_hedge,
            'mtm_pnl': 0  # Would be computed from curve moves
        })
        logger.info(f"  {trade.instrument}: {trade.direction} ${trade.notional_usd:,.0f}")

    return {
        'trades': trades,
        'hedge_result': {
            'fx_dv01': hedge_result.fx_dv01_total,
            'hedge_ratio': hedge_result.hedge_ratio,
            'residual_dv01': hedge_result.residual_dv01,
            'hedge_efficiency': hedge_result.hedge_efficiency,
            'basis_risk': hedge_result.basis_risk_score
        }
    }


# =============================================================================
# RISK CHECKS
# =============================================================================

def run_risk_checks(portfolio: Dict,
                    signals: Dict,
                    config: Dict,
                    logger: logging.Logger) -> Dict:
    """
    Run risk limit checks.

    Args:
        portfolio: Portfolio data
        signals: Signal data
        config: Configuration
        logger: Logger instance

    Returns:
        Risk check results
    """
    logger.info("Running risk checks...")

    limits = config.get('risk_limits', {})
    checks = []

    # Handle empty portfolio
    if not portfolio['weights']:
        logger.warning("  No positions to check")
        return {
            'checks': [],
            'overall_status': 'PASS'
        }

    # Max position check
    max_position = max(abs(w) for w in portfolio['weights'].values())
    limit = config['portfolio'].get('max_position', 0.25)
    status = 'PASS' if max_position <= limit else 'BREACH'
    checks.append({
        'name': 'Max Single Position',
        'limit': limit,
        'current': max_position,
        'status': status,
        'description': f'{limit:.0%} limit per instrument'
    })
    logger.info(f"  Max position: {max_position:.1%} / {limit:.0%} [{status}]")

    # Gross leverage check
    gross = portfolio['gross_exposure']
    limit = config['portfolio'].get('max_leverage', 3.0)
    status = 'PASS' if gross <= limit else 'BREACH'
    checks.append({
        'name': 'Max Gross Leverage',
        'limit': limit,
        'current': gross,
        'status': status,
        'description': f'{limit:.0%} gross exposure limit'
    })
    logger.info(f"  Gross leverage: {gross:.1%} / {limit:.0%} [{status}]")

    # Drawdown check
    dd_level = abs(1 - signals['regime'].get('drawdown_multiplier', 1.0))
    limit = limits.get('max_drawdown', 0.15)
    status = 'PASS' if dd_level < limit * 0.5 else ('WARNING' if dd_level < limit else 'BREACH')
    checks.append({
        'name': 'Max Drawdown',
        'limit': limit,
        'current': dd_level,
        'status': status,
        'description': f'{limit:.0%} max drawdown threshold'
    })
    logger.info(f"  Drawdown: {dd_level:.1%} / {limit:.0%} [{status}]")

    # Concentration check (top position)
    top_position = max(abs(w) for w in portfolio['weights'].values())
    limit = 0.40
    status = 'PASS' if top_position < limit * 0.8 else ('WARNING' if top_position < limit else 'BREACH')
    checks.append({
        'name': 'Max Concentration',
        'limit': limit,
        'current': top_position,
        'status': status,
        'description': f'{limit:.0%} max in top position'
    })
    logger.info(f"  Concentration: {top_position:.1%} / {limit:.0%} [{status}]")

    # Overall status
    any_breach = any(c['status'] == 'BREACH' for c in checks)
    any_warning = any(c['status'] == 'WARNING' for c in checks)

    overall = 'BREACH' if any_breach else ('WARNING' if any_warning else 'PASS')
    logger.info(f"  Overall risk status: {overall}")

    return {
        'checks': checks,
        'overall_status': overall
    }


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_outputs(portfolio: Dict,
                      signals: Dict,
                      rates: Dict,
                      risk: Dict,
                      prices_df: pd.DataFrame,
                      config: Dict,
                      logger: logging.Logger) -> str:
    """
    Generate daily report and save outputs.

    Args:
        portfolio: Portfolio data
        signals: Signal data
        rates: Rates trades data
        risk: Risk check results
        prices_df: Price data for equity history
        config: Configuration
        logger: Logger instance

    Returns:
        Path to report file
    """
    logger.info("Generating outputs...")

    # Build portfolio state for report
    portfolio_notional = 100_000_000
    positions = {}

    for ticker, weight in portfolio['weights'].items():
        if abs(weight) < 0.01:
            continue

        # Get price with validation
        try:
            if ticker in prices_df.columns:
                price_series = prices_df[ticker].dropna()
                if len(price_series) > 0 and pd.api.types.is_numeric_dtype(price_series):
                    current_price = float(price_series.iloc[-1])
                    entry_price = float(price_series.iloc[-5]) if len(price_series) > 5 else current_price
                else:
                    current_price = 1.0
                    entry_price = 1.0
            else:
                current_price = 1.0
                entry_price = 1.0
        except Exception:
            current_price = 1.0
            entry_price = 1.0

        # Calculate P&L safely
        if entry_price > 0:
            pnl_pct = (current_price / entry_price - 1)
            unrealized_pnl = weight * portfolio_notional * pnl_pct
        else:
            unrealized_pnl = 0

        positions[ticker] = {
            'weight': weight,
            'notional_usd': abs(weight * portfolio_notional),
            'entry_price': entry_price,
            'current_price': current_price,
            'unrealized_pnl': unrealized_pnl,
            'dv01': abs(weight * portfolio_notional) * 91 / 365 / 10000  # Approximate DV01
        }

    portfolio_state = {
        'nav': portfolio_notional,
        'positions': positions,
        'risk': {
            'var_95_1d': portfolio_notional * portfolio['vol_forecast'] * 1.65 / np.sqrt(252),
            'var_99_1d': portfolio_notional * portfolio['vol_forecast'] * 2.33 / np.sqrt(252),
            'cvar_95_1d': portfolio_notional * portfolio['vol_forecast'] * 2.06 / np.sqrt(252),
            'total_dv01': sum(p['dv01'] for p in positions.values()),
            'fx_dv01': sum(p['dv01'] for t, p in positions.items() if 'XA' not in t),
            'rates_dv01': sum(t.get('dv01', 0) for t in rates['trades']),
            'gross_leverage': portfolio['gross_exposure'],
            'net_leverage': portfolio['net_exposure'],
            'beta_to_spx': 0.1,
            'correlation_to_dxy': -0.3
        }
    }

    # Mock P&L (would be computed from actual positions in production)
    pnl = {
        'yesterday_pnl': np.random.normal(0, portfolio_notional * 0.005),
        'wtd_pnl': np.random.normal(0, portfolio_notional * 0.01),
        'mtd_pnl': np.random.normal(0, portfolio_notional * 0.02),
        'ytd_pnl': np.random.normal(0, portfolio_notional * 0.05),
        'inception_pnl': np.random.normal(0, portfolio_notional * 0.10),
        'yesterday_return': np.random.normal(0, 0.005),
        'wtd_return': np.random.normal(0, 0.01),
        'mtd_return': np.random.normal(0, 0.02),
        'ytd_return': np.random.normal(0, 0.05),
    }

    # Build equity history (mock)
    if len(prices_df) >= 60:
        equity_base = portfolio_notional
        returns = np.random.normal(0.0003, 0.005, 60)
        equity = equity_base * np.cumprod(1 + returns)
        equity_series = pd.Series(equity, index=prices_df.index[-60:])
    else:
        equity_series = pd.Series([portfolio_notional], index=[datetime.now()])

    # Regime data
    regime = {
        'vix_level': signals['regime'].get('vix_level', 18),
        'vix_regime': signals['regime'].get('vix_regime', 'normal'),
        'vix_multiplier': signals['regime'].get('vix_multiplier', 1.0),
        'fxvol_level': 10.0,
        'fxvol_regime': 'normal',
        'fxvol_multiplier': 1.0,
        'drawdown_level': 0.0,
        'drawdown_regime': signals['regime'].get('drawdown_regime', 'normal'),
        'drawdown_multiplier': signals['regime'].get('drawdown_multiplier', 1.0),
        'combined_multiplier': signals['regime'].get('combined_multiplier', 1.0),
        'recommendation': signals['regime'].get('recommendation', 'full_risk')
    }

    # Generate HTML report
    output_dir = os.path.join(project_root, config['reporting'].get('output_dir', 'reports'))
    generator = DailyReportGenerator(output_dir=output_dir)

    report_path = generator.generate_report(
        portfolio_state=portfolio_state,
        signals=signals['signals'],
        regime=regime,
        pnl=pnl,
        equity=equity_series,
        rates_trades=rates['trades'],
        constraints=risk['checks']
    )

    logger.info(f"  Report saved: {report_path}")

    # Save JSON output
    outputs_dir = os.path.join(project_root, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    output_file = os.path.join(outputs_dir, f"daily_{datetime.now().strftime('%Y%m%d')}.json")

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'portfolio': {
            'weights': portfolio['weights'],
            'gross_exposure': portfolio['gross_exposure'],
            'net_exposure': portfolio['net_exposure'],
            'leverage': portfolio['leverage']
        },
        'signals': signals['signals'],
        'regime': signals['regime'],
        'rates_trades': rates['trades'],
        'risk_checks': risk['checks'],
        'risk_status': risk['overall_status']
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"  JSON output saved: {output_file}")

    return report_path


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """Main execution workflow."""

    # Parse arguments
    parser = argparse.ArgumentParser(description='Daily FX Carry Strategy Execution')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--data-source', type=str, default='free', choices=['free', 'bloomberg'])
    parser.add_argument('--lookback', type=int, default=100, help='Days of historical data')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without saving')
    args = parser.parse_args()

    # Setup
    run_date = datetime.now()
    log_dir = os.path.join(project_root, 'logs', 'daily')
    logger = setup_logging(log_dir, run_date)

    logger.info("=" * 70)
    logger.info("FX/METALS CARRY STRATEGY - DAILY EXECUTION")
    logger.info("=" * 70)
    logger.info(f"Run date: {run_date.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Data source: {args.data_source}")
    logger.info(f"Lookback: {args.lookback} days")

    try:
        # Step 1: Load config
        logger.info("-" * 40)
        logger.info("STEP 1: Loading configuration")
        config = load_config(args.config)
        config['data_source'] = args.data_source
        logger.info(f"  Config loaded: {len(config)} sections")

        # Step 2: Load market data
        logger.info("-" * 40)
        logger.info("STEP 2: Loading market data")
        market_data = load_market_data(config, args.lookback, logger)
        prices_df = market_data['prices']
        returns_df = market_data['returns']
        logger.info(f"  Data loaded: {prices_df.shape[0]} days, {prices_df.shape[1]} instruments")

        # Step 3: Compute signals
        logger.info("-" * 40)
        logger.info("STEP 3: Computing signals")
        signals = compute_signals(prices_df, returns_df, config, logger)

        # Step 4: Construct portfolio
        logger.info("-" * 40)
        logger.info("STEP 4: Constructing portfolio")
        portfolio = construct_portfolio(signals, returns_df, config, logger)

        # Step 5: Generate rates trades
        logger.info("-" * 40)
        logger.info("STEP 5: Generating rates overlay")
        rates = generate_rates_trades(portfolio, config, logger)

        # Step 6: Run risk checks
        logger.info("-" * 40)
        logger.info("STEP 6: Running risk checks")
        risk = run_risk_checks(portfolio, signals, config, logger)

        # Step 7: Generate outputs
        logger.info("-" * 40)
        logger.info("STEP 7: Generating outputs")
        if not args.dry_run:
            report_path = generate_outputs(portfolio, signals, rates, risk, prices_df, config, logger)
        else:
            logger.info("  Dry run - skipping output generation")
            report_path = None

        # Summary
        logger.info("=" * 70)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Data source:        {args.data_source}")
        logger.info(f"  Instruments:        {prices_df.shape[1]}")
        logger.info(f"  Days of data:       {prices_df.shape[0]}")
        logger.info(f"  VIX level:          {signals['regime'].get('vix_level', 'N/A'):.1f}")
        logger.info(f"  Regime multiplier:  {signals['regime']['combined_multiplier']:.0%}")
        logger.info(f"  Gross exposure:     {portfolio['gross_exposure']:.1%}")
        logger.info(f"  Net exposure:       {portfolio['net_exposure']:.1%}")
        logger.info(f"  Number of trades:   {sum(1 for w in portfolio['weights'].values() if abs(w) > 0.01)}")
        logger.info(f"  Rates hedges:       {len(rates['trades'])}")
        logger.info(f"  Risk status:        {risk['overall_status']}")
        if report_path:
            logger.info(f"  Report:             {report_path}")
        logger.info("=" * 70)
        logger.info("EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
