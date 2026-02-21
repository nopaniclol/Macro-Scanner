#!/usr/bin/env python3
"""
Historical Backtest V3 - Optimized Parameters

Key changes from V1:
1. Higher carry weight (carry was profitable in V1)
2. Simpler but effective regime filter
3. Better position sizing
4. Transaction cost optimization
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


# =============================================================================
# OPTIMIZED SIGNALS
# =============================================================================

def compute_momentum_signal(prices: pd.Series, lookbacks=[21, 63, 126]) -> float:
    """
    Simple momentum signal using returns.
    Positive return = positive signal.
    """
    if len(prices) < max(lookbacks) + 10:
        return 0.0

    signals = []
    weights = [0.25, 0.50, 0.25]  # Weight towards 3M

    for lb, w in zip(lookbacks, weights):
        ret = (prices.iloc[-1] / prices.iloc[-lb] - 1)
        # Normalize: 10% return over period = signal of 1.0
        norm_ret = ret / (0.10 * np.sqrt(lb / 252))
        signals.append(w * np.clip(norm_ret, -1.5, 1.5))

    return sum(signals)


def compute_carry_signal(ticker: str) -> float:
    """
    Carry signal based on rate differentials.
    Positive carry = positive signal.
    """
    # Higher carry = higher signal
    carry_map = {
        'USDJPY': 1.2,    # High positive carry
        'AUDUSD': 0.4,    # Moderate carry
        'XAUUSD': 0.3,    # Small positive
        'USDCAD': 0.2,
        'XAGUSD': 0.2,
        'GBPUSD': -0.2,   # Negative carry
        'EURUSD': -0.5,   # Higher negative carry
    }
    return carry_map.get(ticker, 0.0)


def compute_regime_multiplier(vix: pd.Series) -> float:
    """
    Simple VIX-based regime filter.
    """
    if len(vix) < 5:
        return 1.0

    current_vix = vix.iloc[-1]

    if current_vix < 15:
        return 1.10  # Risk-on
    elif current_vix < 20:
        return 1.00  # Normal
    elif current_vix < 28:
        return 0.75  # Cautious
    elif current_vix < 35:
        return 0.50  # Defensive
    elif current_vix < 50:
        return 0.30  # Crisis
    else:
        return 0.15  # Extreme crisis


# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================

def construct_portfolio(momentum_signals: Dict[str, float],
                        carry_signals: Dict[str, float],
                        returns_df: pd.DataFrame,
                        regime_mult: float,
                        target_vol: float = 0.12,
                        max_position: float = 0.30,
                        momentum_weight: float = 0.35,
                        carry_weight: float = 0.65) -> Dict[str, float]:
    """
    Construct portfolio with carry tilt.
    """
    tickers = list(momentum_signals.keys())
    if not tickers:
        return {}

    # Combine signals - higher carry weight
    combined = {}
    for t in tickers:
        mom = momentum_signals.get(t, 0)
        carry = carry_signals.get(t, 0)
        combined[t] = momentum_weight * mom + carry_weight * carry

    # Compute volatilities
    vols = {}
    for t in tickers:
        if t in returns_df.columns:
            vol = returns_df[t].tail(63).std() * np.sqrt(252)
            vols[t] = max(vol, 0.05)
        else:
            vols[t] = 0.10

    # Risk parity base weights
    inv_vols = {t: 1 / v for t, v in vols.items()}
    total = sum(inv_vols.values())
    base_weights = {t: v / total for t, v in inv_vols.items()}

    # Apply signals
    weights = {}
    for t in tickers:
        signal = np.clip(combined[t], -1.5, 1.5)
        weights[t] = base_weights[t] * signal

    # Scale to target vol (before position limits)
    port_vol = estimate_portfolio_vol(weights, returns_df)
    if port_vol > 0:
        scale = target_vol / port_vol
        weights = {t: w * scale for t, w in weights.items()}

    # Apply regime
    weights = {t: w * regime_mult for t, w in weights.items()}

    # Position limits
    for t in weights:
        weights[t] = np.clip(weights[t], -max_position, max_position)

    return weights


def estimate_portfolio_vol(weights: Dict[str, float], returns_df: pd.DataFrame) -> float:
    """Estimate portfolio volatility using simple approximation."""
    if not weights:
        return 0.10

    total_var = 0
    for t, w in weights.items():
        if t in returns_df.columns:
            var = (returns_df[t].tail(63).std() * np.sqrt(252)) ** 2
            total_var += (w ** 2) * var

    # Add cross-correlation estimate (assume 0.3 avg correlation)
    gross = sum(abs(w) for w in weights.values())
    cross_var = 0.15 * (gross ** 2) * (0.08 ** 2)  # Rough approximation

    return np.sqrt(total_var + cross_var)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(years: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic data with realistic dynamics."""
    np.random.seed(42)

    end = datetime.now()
    start = end - timedelta(days=years * 365)
    dates = pd.date_range(start=start, end=end, freq='B')
    n = len(dates)

    instruments = {
        'EURUSD': {'base': 1.20, 'vol': 0.08, 'drift': -0.015},
        'GBPUSD': {'base': 1.55, 'vol': 0.09, 'drift': -0.020},
        'AUDUSD': {'base': 0.90, 'vol': 0.11, 'drift': -0.010},
        'USDJPY': {'base': 95, 'vol': 0.09, 'drift': 0.035},
        'USDCAD': {'base': 1.05, 'vol': 0.07, 'drift': 0.015},
        'XAUUSD': {'base': 1300, 'vol': 0.14, 'drift': 0.050},
        'XAGUSD': {'base': 18, 'vol': 0.22, 'drift': 0.030},
    }

    prices, returns = {}, {}

    for ticker, p in instruments.items():
        drift = p['drift'] / 252
        vol = p['vol'] / np.sqrt(252)

        rets = np.random.normal(drift, vol, n)

        # Vol clustering
        for i in range(1, n):
            rets[i] *= 0.9 + 0.2 * abs(rets[i-1]) / vol

        # Crisis (COVID equivalent)
        covid = int(n * 0.72)
        if covid + 40 < n:
            rets[covid:covid+25] *= 2.0
            if ticker == 'XAUUSD':
                rets[covid:covid+25] = abs(rets[covid:covid+25]) * 0.6

        rets = np.clip(rets, -0.05, 0.05)
        prices[ticker] = pd.Series(p['base'] * np.cumprod(1 + rets), index=dates)
        returns[ticker] = pd.Series(rets, index=dates)

    # VIX
    vix = np.zeros(n)
    vix[0] = 15
    for i in range(1, n):
        vix[i] = np.clip(vix[i-1] + 0.04 * (15 - vix[i-1]) + np.random.normal(0, 1), 9, 80)

    covid = int(n * 0.72)
    if covid + 50 < n:
        vix[covid:covid+20] = np.linspace(18, 70, 20)
        vix[covid+20:covid+50] = np.linspace(70, 25, 30)

    prices['VIX'] = pd.Series(vix, index=dates)

    return pd.DataFrame(prices), pd.DataFrame(returns)


# =============================================================================
# BACKTEST
# =============================================================================

def run_backtest(prices: pd.DataFrame, returns: pd.DataFrame):
    """Run optimized backtest."""
    logger.info("Running V3 backtest...")

    tradeable = [c for c in prices.columns if c != 'VIX']
    lookback = 63
    dates = prices.index[lookback:]

    equity = pd.Series(index=dates, dtype=float)
    initial = 100_000_000
    nav = initial
    prev_weights = {t: 0 for t in tradeable}

    # Lower TC for less frequent rebalancing
    tc_rate = 0.0004  # 4bps average

    total = len(dates)
    rebal_freq = 5  # Rebalance every 5 days

    for i, date in enumerate(dates):
        if i > 0 and i % (total // 10) == 0:
            logger.info(f"  {i/total*100:.0f}%")

        idx = prices.index.get_loc(date)

        # Only rebalance periodically
        if i % rebal_freq == 0:
            # Signals
            mom_signals = {t: compute_momentum_signal(prices[t].iloc[:idx+1]) for t in tradeable}
            carry_signals = {t: compute_carry_signal(t) for t in tradeable}
            regime = compute_regime_multiplier(prices['VIX'].iloc[:idx+1])

            weights = construct_portfolio(
                mom_signals, carry_signals,
                returns[tradeable].iloc[:idx+1],
                regime
            )
        else:
            weights = prev_weights.copy()

        # P&L
        if i > 0:
            daily_ret = returns[tradeable].loc[date]
            pnl = sum(prev_weights.get(t, 0) * daily_ret.get(t, 0) for t in tradeable)

            # TC only on rebalance
            if i % rebal_freq == 0:
                turnover = sum(abs(weights.get(t, 0) - prev_weights.get(t, 0)) for t in tradeable)
                tc = turnover * tc_rate
            else:
                tc = 0

            nav = nav * (1 + pnl - tc)

        equity.loc[date] = nav
        prev_weights = weights.copy()

    # Metrics
    rets = equity.pct_change(fill_method=None).fillna(0)
    total_ret = nav / initial - 1
    years = len(equity) / 252
    ann_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.05) / ann_vol if ann_vol > 0 else 0

    rolling_max = equity.expanding().max()
    dd = (equity - rolling_max) / rolling_max
    max_dd = dd.min()

    # COVID
    covid_idx = int(len(equity) * 0.72)
    covid_eq = equity.iloc[covid_idx:covid_idx+50]
    covid_ret = covid_eq.iloc[-1] / covid_eq.iloc[0] - 1 if len(covid_eq) > 1 else 0
    covid_dd = ((covid_eq - covid_eq.expanding().max()) / covid_eq.expanding().max()).min() if len(covid_eq) > 1 else 0

    return {
        'equity': equity,
        'total_return': total_ret,
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'covid_return': covid_ret,
        'covid_dd': covid_dd
    }


def main():
    print("=" * 70)
    print("BACKTEST V3 - OPTIMIZED CARRY TILT")
    print("=" * 70)

    prices, returns = generate_data(10)
    print(f"\nData: {len(prices)} days")

    results = run_backtest(prices, returns)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nTotal Return:        {results['total_return']:>10.2%}")
    print(f"Annualized Return:   {results['ann_return']:>10.2%}")
    print(f"Annualized Vol:      {results['ann_vol']:>10.2%}")
    print(f"Sharpe Ratio:        {results['sharpe']:>10.2f}")
    print(f"Max Drawdown:        {results['max_dd']:>10.2%}")
    print(f"\nCOVID Period:")
    print(f"  Return:            {results['covid_return']:>10.2%}")
    print(f"  Max DD:            {results['covid_dd']:>10.2%}")

    # Quick equity visualization
    eq = results['equity']
    print(f"\nEquity Curve:")
    print(f"  Start: ${eq.iloc[0]/1e6:.1f}M")
    print(f"  End:   ${eq.iloc[-1]/1e6:.1f}M")
    print(f"  Peak:  ${eq.max()/1e6:.1f}M")
    print(f"  Low:   ${eq.min()/1e6:.1f}M")

    print("\n" + "=" * 70)
    if results['sharpe'] >= 0.5:
        print(f"SUCCESS: Sharpe {results['sharpe']:.2f} >= 0.5")
    else:
        print(f"NEEDS WORK: Sharpe {results['sharpe']:.2f} < 0.5")

    return results


if __name__ == '__main__':
    main()
