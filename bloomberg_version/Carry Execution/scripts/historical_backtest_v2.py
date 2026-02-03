#!/usr/bin/env python3
"""
Historical Backtest V2 - Signal Debugging

Improvements:
1. Better regime filtering (more aggressive risk-off)
2. Stricter position limits enforcement
3. Adjusted signal weights (more momentum, less carry)
4. Better momentum lookback tuning
5. Correlation-based diversification
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
# IMPROVED SIGNAL ENGINE
# =============================================================================

class ImprovedMomentumSignal:
    """
    Improved momentum signal with:
    - Multi-timeframe ensemble (1M, 3M, 6M, 12M)
    - Volatility-adjusted returns
    - Mean reversion filter
    """

    def __init__(self):
        self.lookbacks = [21, 63, 126, 252]  # 1M, 3M, 6M, 12M
        self.weights = [0.15, 0.30, 0.35, 0.20]  # Weight towards 3-6M

    def compute(self, prices: pd.Series) -> float:
        """Compute momentum signal."""
        if len(prices) < max(self.lookbacks) + 10:
            return 0.0

        returns = prices.pct_change(fill_method=None).fillna(0)
        vol = returns.rolling(21).std()
        current_vol = vol.iloc[-1] if not pd.isna(vol.iloc[-1]) else returns.std()

        if current_vol < 1e-8:
            current_vol = 0.01

        signals = []
        for lb in self.lookbacks:
            # Volatility-adjusted momentum
            ret = (prices.iloc[-1] / prices.iloc[-lb] - 1)
            vol_adj = ret / (current_vol * np.sqrt(lb / 252))
            signals.append(np.clip(vol_adj, -3, 3) / 3)  # Normalize to [-1, 1]

        # Weighted ensemble
        ensemble = sum(w * s for w, s in zip(self.weights, signals))

        # Mean reversion filter: reduce signal if price extended
        z_score = (prices.iloc[-1] - prices.rolling(252).mean().iloc[-1]) / prices.rolling(252).std().iloc[-1]
        if not pd.isna(z_score):
            if abs(z_score) > 2:
                ensemble *= 0.5  # Reduce momentum signal when extended

        return np.clip(ensemble, -1, 1)


class ImprovedCarrySignal:
    """
    Improved carry signal with:
    - Dynamic carry based on rate differentials
    - Carry volatility adjustment
    """

    # Carry rates (annualized) - would be dynamic in production
    CARRY_RATES = {
        'EURUSD': -0.020,   # Short EUR earns positive carry
        'GBPUSD': -0.010,
        'AUDUSD': 0.015,    # Long AUD earns positive carry
        'USDJPY': 0.045,    # Long USD/JPY earns positive carry
        'USDCAD': 0.010,
        'XAUUSD': 0.005,    # Small gold lease rate
        'XAGUSD': 0.003,
    }

    def compute(self, ticker: str, vol: float = 0.10) -> float:
        """
        Compute carry signal adjusted for volatility.

        Higher carry / vol = higher signal.
        """
        carry = self.CARRY_RATES.get(ticker, 0)

        # Carry-to-vol ratio (Sharpe of carry)
        if vol > 0:
            carry_sharpe = carry / vol
        else:
            carry_sharpe = carry / 0.10

        # Normalize to [-1, 1] assuming 0.3 carry sharpe is "strong"
        signal = np.clip(carry_sharpe / 0.30, -1, 1)

        return signal


class ImprovedRegimeFilter:
    """
    Improved regime filter with:
    - VIX level and change
    - Correlation regime
    - Trend regime
    """

    def compute_multiplier(self, vix: pd.Series, prices_df: pd.DataFrame = None) -> float:
        """
        Compute regime multiplier.

        Returns value in [0.1, 1.2]:
        - Low VIX, positive trends = 1.2 (risk-on)
        - Normal = 1.0
        - High VIX, negative trends = 0.1 (risk-off)
        """
        if len(vix) < 20:
            return 1.0

        current_vix = vix.iloc[-1]
        vix_ma = vix.rolling(20).mean().iloc[-1]
        vix_change = (current_vix - vix.iloc[-5]) / vix.iloc[-5] if vix.iloc[-5] > 0 else 0

        # VIX level component
        if current_vix < 15:
            vix_mult = 1.15
        elif current_vix < 20:
            vix_mult = 1.0
        elif current_vix < 25:
            vix_mult = 0.70
        elif current_vix < 30:
            vix_mult = 0.40
        elif current_vix < 40:
            vix_mult = 0.20
        else:
            vix_mult = 0.10

        # VIX spike component (sudden moves)
        if vix_change > 0.20:  # VIX spiked 20%+
            vix_mult *= 0.5
        elif vix_change > 0.10:
            vix_mult *= 0.7

        # Trend regime (if prices provided)
        trend_mult = 1.0
        if prices_df is not None and len(prices_df) > 50:
            # Check if majority of assets trending down
            down_count = 0
            for col in prices_df.columns:
                if col == 'VIX':
                    continue
                try:
                    ma50 = prices_df[col].rolling(50).mean().iloc[-1]
                    if prices_df[col].iloc[-1] < ma50 * 0.98:
                        down_count += 1
                except:
                    pass

            n_assets = len([c for c in prices_df.columns if c != 'VIX'])
            if n_assets > 0 and down_count / n_assets > 0.6:
                trend_mult = 0.7  # Most assets below 50MA

        return np.clip(vix_mult * trend_mult, 0.10, 1.20)


# =============================================================================
# IMPROVED PORTFOLIO CONSTRUCTION
# =============================================================================

class ImprovedPortfolioConstructor:
    """
    Improved portfolio constructor with:
    - Risk parity weighting
    - Correlation-based position limits
    - Stricter max position enforcement
    """

    def __init__(self):
        self.target_vol = 0.10  # Lower target vol for stability
        self.max_position = 0.20  # Stricter position limit
        self.max_gross = 1.5  # Max gross exposure
        self.momentum_weight = 0.60
        self.carry_weight = 0.40

    def construct(self,
                  momentum_signals: Dict[str, float],
                  carry_signals: Dict[str, float],
                  returns_df: pd.DataFrame,
                  regime_multiplier: float) -> Dict[str, float]:
        """
        Construct portfolio weights.
        """
        tickers = list(momentum_signals.keys())

        if not tickers:
            return {}

        # Combine signals
        combined_signals = {}
        for ticker in tickers:
            mom = momentum_signals.get(ticker, 0)
            carry = carry_signals.get(ticker, 0)
            combined_signals[ticker] = self.momentum_weight * mom + self.carry_weight * carry

        # Compute volatilities for risk parity
        vols = {}
        for ticker in tickers:
            if ticker in returns_df.columns:
                vol = returns_df[ticker].tail(63).std() * np.sqrt(252)
                vols[ticker] = max(vol, 0.05)  # Floor at 5%
            else:
                vols[ticker] = 0.10

        # Risk parity weights
        inv_vols = {t: 1 / v for t, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        base_weights = {t: v / total_inv_vol for t, v in inv_vols.items()}

        # Apply signals to base weights
        raw_weights = {}
        for ticker in tickers:
            signal = combined_signals[ticker]
            base = base_weights[ticker]

            # Signal determines direction and magnitude
            raw_weights[ticker] = base * signal * 2  # Scale up

        # Apply position limits
        for ticker in raw_weights:
            raw_weights[ticker] = np.clip(raw_weights[ticker], -self.max_position, self.max_position)

        # Vol targeting
        portfolio_vol = self._estimate_portfolio_vol(raw_weights, returns_df)
        if portfolio_vol > 0:
            scale = self.target_vol / portfolio_vol
        else:
            scale = 1.0

        weights = {t: w * scale for t, w in raw_weights.items()}

        # Apply regime multiplier
        weights = {t: w * regime_multiplier for t, w in weights.items()}

        # Final position limit check
        for ticker in weights:
            weights[ticker] = np.clip(weights[ticker], -self.max_position, self.max_position)

        # Gross exposure limit
        gross = sum(abs(w) for w in weights.values())
        if gross > self.max_gross:
            scale = self.max_gross / gross
            weights = {t: w * scale for t, w in weights.items()}

        return weights

    def _estimate_portfolio_vol(self, weights: Dict[str, float], returns_df: pd.DataFrame) -> float:
        """Estimate portfolio volatility."""
        tickers = [t for t in weights.keys() if t in returns_df.columns]
        if not tickers:
            return 0.10

        w = np.array([weights[t] for t in tickers])
        returns = returns_df[tickers].tail(63)

        if len(returns) < 20:
            return 0.10

        cov = returns.cov() * 252
        try:
            port_var = w @ cov.values @ w
            return np.sqrt(max(port_var, 0))
        except:
            return 0.10


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_synthetic_data(years: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic historical data."""
    logger.info(f"Generating {years} years of synthetic data...")

    np.random.seed(42)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)

    instruments = {
        'EURUSD': {'base': 1.20, 'vol': 0.08, 'drift': -0.02},
        'GBPUSD': {'base': 1.55, 'vol': 0.09, 'drift': -0.03},
        'AUDUSD': {'base': 0.90, 'vol': 0.11, 'drift': -0.02},
        'USDJPY': {'base': 95, 'vol': 0.09, 'drift': 0.04},
        'USDCAD': {'base': 1.05, 'vol': 0.07, 'drift': 0.02},
        'XAUUSD': {'base': 1300, 'vol': 0.14, 'drift': 0.06},
        'XAGUSD': {'base': 18, 'vol': 0.22, 'drift': 0.04},
    }

    prices = {}
    returns = {}

    for ticker, params in instruments.items():
        daily_vol = params['vol'] / np.sqrt(252)
        daily_drift = params['drift'] / 252

        daily_returns = np.random.normal(daily_drift, daily_vol, n_days)

        # Volatility clustering
        vol_factor = np.ones(n_days)
        for i in range(1, n_days):
            vol_factor[i] = 0.92 * vol_factor[i-1] + 0.08 * (1 + 2 * abs(daily_returns[i-1]) / daily_vol)
        vol_factor = np.clip(vol_factor, 0.6, 2.0)
        daily_returns = daily_returns * vol_factor

        # Crisis periods
        covid_start = int(n_days * 0.72)
        if covid_start + 50 < n_days:
            daily_returns[covid_start:covid_start+30] *= 2.5
            if ticker == 'XAUUSD':
                daily_returns[covid_start:covid_start+30] = np.abs(daily_returns[covid_start:covid_start+30]) * 0.6

        daily_returns = np.clip(daily_returns, -0.06, 0.06)
        price_series = params['base'] * np.cumprod(1 + daily_returns)

        prices[ticker] = pd.Series(price_series, index=dates)
        returns[ticker] = pd.Series(daily_returns, index=dates)

    # VIX
    vix = np.zeros(n_days)
    vix[0] = 16
    for i in range(1, n_days):
        vix[i] = np.clip(vix[i-1] + 0.03 * (16 - vix[i-1]) + np.random.normal(0, 1.2), 9, 80)

    covid_start = int(n_days * 0.72)
    if covid_start + 60 < n_days:
        vix[covid_start:covid_start+20] = np.linspace(20, 75, 20)
        vix[covid_start+20:covid_start+60] = np.linspace(75, 28, 40)

    prices['VIX'] = pd.Series(vix, index=dates)

    return pd.DataFrame(prices), pd.DataFrame(returns)


# =============================================================================
# BACKTEST
# =============================================================================

def run_backtest_v2(prices_df: pd.DataFrame, returns_df: pd.DataFrame):
    """Run improved backtest."""
    logger.info("Running improved backtest...")

    momentum = ImprovedMomentumSignal()
    carry = ImprovedCarrySignal()
    regime = ImprovedRegimeFilter()
    portfolio = ImprovedPortfolioConstructor()

    tradeable = [c for c in prices_df.columns if c != 'VIX']
    lookback = 63
    dates = prices_df.index[lookback:]

    # Track results
    equity = pd.Series(index=dates, dtype=float)
    positions_history = pd.DataFrame(index=dates, columns=tradeable, dtype=float)

    initial_capital = 100_000_000
    current_equity = initial_capital
    prev_weights = {t: 0 for t in tradeable}

    tc_rates = {
        'EURUSD': 0.0005, 'GBPUSD': 0.0005, 'AUDUSD': 0.0005,
        'USDJPY': 0.0005, 'USDCAD': 0.0005,
        'XAUUSD': 0.0003, 'XAGUSD': 0.0010
    }

    total_days = len(dates)
    log_interval = total_days // 10

    for i, date in enumerate(dates):
        if i > 0 and i % log_interval == 0:
            logger.info(f"  Progress: {i/total_days*100:.0f}%")

        idx = prices_df.index.get_loc(date)

        # Compute signals
        momentum_signals = {}
        carry_signals = {}
        for ticker in tradeable:
            price_slice = prices_df[ticker].iloc[:idx+1]
            momentum_signals[ticker] = momentum.compute(price_slice)

            vol = returns_df[ticker].iloc[:idx+1].tail(63).std() * np.sqrt(252)
            carry_signals[ticker] = carry.compute(ticker, vol)

        # Regime
        vix_slice = prices_df['VIX'].iloc[:idx+1]
        prices_slice = prices_df.iloc[:idx+1]
        regime_mult = regime.compute_multiplier(vix_slice, prices_slice)

        # Construct portfolio
        returns_slice = returns_df[tradeable].iloc[:idx+1]
        weights = portfolio.construct(momentum_signals, carry_signals, returns_slice, regime_mult)

        # Compute P&L
        if i > 0:
            daily_returns = returns_df[tradeable].loc[date]
            gross_pnl = sum(prev_weights.get(t, 0) * daily_returns.get(t, 0) for t in tradeable)

            # Transaction costs
            turnover = sum(abs(weights.get(t, 0) - prev_weights.get(t, 0)) for t in tradeable)
            tc = sum(
                abs(weights.get(t, 0) - prev_weights.get(t, 0)) * tc_rates.get(t, 0.0005)
                for t in tradeable
            )

            net_pnl = gross_pnl - tc
            current_equity = current_equity * (1 + net_pnl)

        equity.loc[date] = current_equity
        positions_history.loc[date] = pd.Series(weights)
        prev_weights = weights.copy()

    # Compute metrics
    returns = equity.pct_change(fill_method=None).fillna(0)

    total_return = (equity.iloc[-1] / initial_capital) - 1
    years = len(equity) / 252
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - 0.05) / ann_vol if ann_vol > 0 else 0

    # Drawdown
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Sortino
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = (ann_return - 0.05) / downside_vol if downside_vol > 0 else 0

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # COVID analysis
    covid_analysis = {}
    try:
        covid_start = pd.Timestamp('2020-02-15')
        covid_end = pd.Timestamp('2020-04-30')
        if covid_start >= equity.index[0] and covid_end <= equity.index[-1]:
            covid_equity = equity.loc[covid_start:covid_end]
            covid_return = (covid_equity.iloc[-1] / covid_equity.iloc[0]) - 1
            covid_dd = ((covid_equity - covid_equity.expanding().max()) / covid_equity.expanding().max()).min()
            covid_analysis = {'return': covid_return, 'max_dd': covid_dd}
    except:
        # Use synthetic COVID period
        covid_start_idx = int(len(equity) * 0.72)
        covid_end_idx = covid_start_idx + 50
        if covid_end_idx < len(equity):
            covid_equity = equity.iloc[covid_start_idx:covid_end_idx]
            covid_return = (covid_equity.iloc[-1] / covid_equity.iloc[0]) - 1
            covid_dd = ((covid_equity - covid_equity.expanding().max()) / covid_equity.expanding().max()).min()
            covid_analysis = {'return': covid_return, 'max_dd': covid_dd}

    return {
        'equity': equity,
        'returns': returns,
        'positions': positions_history,
        'drawdown': drawdown,
        'metrics': {
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'max_dd': max_dd,
        },
        'covid': covid_analysis
    }


def print_results(results):
    """Print backtest results."""
    m = results['metrics']

    print("\n" + "=" * 70)
    print("IMPROVED BACKTEST RESULTS (V2)")
    print("=" * 70)

    print("\nKEY METRICS:")
    print(f"  Total Return:         {m['total_return']:>10.2%}")
    print(f"  Annualized Return:    {m['ann_return']:>10.2%}")
    print(f"  Annualized Vol:       {m['ann_vol']:>10.2%}")
    print(f"  Sharpe Ratio:         {m['sharpe']:>10.2f}")
    print(f"  Sortino Ratio:        {m['sortino']:>10.2f}")
    print(f"  Calmar Ratio:         {m['calmar']:>10.2f}")
    print(f"  Max Drawdown:         {m['max_dd']:>10.2%}")

    if results['covid']:
        print("\nCOVID PERIOD (Feb-Apr 2020 equivalent):")
        print(f"  Return:               {results['covid']['return']:>10.2%}")
        print(f"  Max Drawdown:         {results['covid']['max_dd']:>10.2%}")

    # Equity curve ASCII
    equity = results['equity']
    monthly = equity.resample('ME').last()

    print("\nEQUITY CURVE:")
    min_val = monthly.min()
    max_val = monthly.max()
    range_val = max_val - min_val if max_val != min_val else 1

    for i, (date, val) in enumerate(monthly.items()):
        if i % 6 == 0:  # Every 6 months
            bar_len = int((val - min_val) / range_val * 40)
            print(f"  {date.strftime('%Y-%m')} | {'#' * bar_len} ${val/1e6:.1f}M")

    print("\n" + "=" * 70)

    if m['sharpe'] >= 0.5:
        print(f"SUCCESS: Sharpe {m['sharpe']:.2f} >= 0.5")
    else:
        print(f"NEEDS WORK: Sharpe {m['sharpe']:.2f} < 0.5")
        print("\nSuggested improvements:")
        print("  - Add more signal diversification")
        print("  - Implement better crisis detection")
        print("  - Consider adding value signals")


def main():
    """Main execution."""
    print("=" * 70)
    print("HISTORICAL BACKTEST V2 - IMPROVED SIGNALS")
    print("=" * 70)

    prices_df, returns_df = generate_synthetic_data(years=10)
    print(f"\nData: {len(prices_df)} days, {len(prices_df.columns)} instruments")

    results = run_backtest_v2(prices_df, returns_df)
    print_results(results)

    return results


if __name__ == '__main__':
    results = main()
