"""
Momentum Signal Engine

Time-series momentum (TSMOM) signals for FX, metals, and rates.
Implements multi-horizon momentum with volatility scaling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MomentumSignal:
    """Container for momentum signal output."""
    ticker: str
    signal: float              # Raw signal [-1, 1]
    signal_zscore: float       # Z-scored signal
    returns: Dict[int, float]  # Returns by lookback
    direction: str             # 'long', 'short', 'neutral'
    strength: str              # 'strong', 'moderate', 'weak'
    timestamp: pd.Timestamp


# =============================================================================
# CORE MOMENTUM FUNCTIONS
# =============================================================================

def compute_tsmom(prices: pd.Series, lookback: int) -> pd.Series:
    """
    Compute simple time-series momentum.

    TSMOM = sign of return over lookback period
    Scaled by magnitude for continuous signal.

    Args:
        prices: Price series (Close prices)
        lookback: Lookback period in days

    Returns:
        Series of momentum signals
    """
    if len(prices) < lookback + 1:
        return pd.Series(index=prices.index, dtype=float)

    # Calculate returns over lookback period
    returns = prices.pct_change(lookback)

    # Raw momentum signal (just the return)
    # Positive return = positive signal, negative return = negative signal
    momentum = returns

    return momentum


def compute_tsmom_scaled(prices: pd.Series, lookback: int,
                          vol_lookback: int = 63) -> pd.Series:
    """
    Compute volatility-scaled time-series momentum.

    Scales momentum by inverse volatility to normalize across assets.

    Args:
        prices: Price series
        lookback: Momentum lookback period
        vol_lookback: Volatility estimation lookback

    Returns:
        Series of volatility-scaled momentum signals
    """
    if len(prices) < max(lookback, vol_lookback) + 1:
        return pd.Series(index=prices.index, dtype=float)

    # Daily returns for vol calculation
    daily_returns = prices.pct_change()

    # Rolling volatility (annualized)
    rolling_vol = daily_returns.rolling(vol_lookback).std() * np.sqrt(252)

    # Momentum returns
    mom_returns = prices.pct_change(lookback)

    # Scale by inverse vol (with floor to prevent explosion)
    vol_floor = 0.05  # 5% annual vol floor
    scaled_vol = rolling_vol.clip(lower=vol_floor)

    # Volatility-scaled momentum
    scaled_momentum = mom_returns / scaled_vol

    return scaled_momentum


def compute_tsmom_ensemble(prices: pd.Series,
                            lookbacks: List[int] = [20, 60, 252],
                            weights: List[float] = [0.25, 0.50, 0.25]) -> pd.Series:
    """
    Compute weighted ensemble of multi-horizon momentum signals.

    Combines short, medium, and long-term momentum with configurable weights.

    Args:
        prices: Price series
        lookbacks: List of lookback periods [short, medium, long]
        weights: Corresponding weights (must sum to 1.0)

    Returns:
        Series of ensemble momentum signals
    """
    if len(lookbacks) != len(weights):
        raise ValueError("lookbacks and weights must have same length")

    if abs(sum(weights) - 1.0) > 0.001:
        raise ValueError("weights must sum to 1.0")

    # Compute individual momentum signals
    signals = []
    for lookback in lookbacks:
        sig = compute_tsmom_scaled(prices, lookback)
        signals.append(sig)

    # Weighted combination
    ensemble = pd.Series(0.0, index=prices.index)
    for sig, weight in zip(signals, weights):
        ensemble += sig.fillna(0) * weight

    return ensemble


def compute_ma_crossover(prices: pd.Series,
                          fast: int = 50,
                          slow: int = 200) -> pd.Series:
    """
    Compute moving average crossover signal.

    Signal = (Fast MA - Slow MA) / Slow MA
    Positive = bullish, Negative = bearish

    Args:
        prices: Price series
        fast: Fast MA period (default 50)
        slow: Slow MA period (default 200)

    Returns:
        Series of MA crossover signals
    """
    if len(prices) < slow + 1:
        return pd.Series(index=prices.index, dtype=float)

    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()

    # Percentage difference between MAs
    crossover_signal = (fast_ma - slow_ma) / slow_ma

    return crossover_signal


def compute_ma_trend_strength(prices: pd.Series,
                               periods: List[int] = [10, 20, 50]) -> pd.Series:
    """
    Compute trend strength based on MA alignment.

    +3 = All MAs in bullish order (price > 10 > 20 > 50)
    -3 = All MAs in bearish order
    Values in between indicate mixed signals

    Args:
        prices: Price series
        periods: MA periods to check

    Returns:
        Series of trend strength scores
    """
    if len(prices) < max(periods) + 1:
        return pd.Series(index=prices.index, dtype=float)

    mas = {p: prices.rolling(p).mean() for p in periods}

    # Count bullish alignments
    strength = pd.Series(0.0, index=prices.index)

    # Price above all MAs
    for p in periods:
        strength += (prices > mas[p]).astype(float)

    # MA ordering
    sorted_periods = sorted(periods)
    for i in range(len(sorted_periods) - 1):
        shorter = sorted_periods[i]
        longer = sorted_periods[i + 1]
        strength += (mas[shorter] > mas[longer]).astype(float) * 0.5

    # Normalize to [-1, 1] range
    max_score = len(periods) + (len(periods) - 1) * 0.5
    strength = (strength / max_score) * 2 - 1

    return strength


def zscore_signal(signal: pd.Series, window: int = 252,
                  winsorize_threshold: float = 3.0) -> pd.Series:
    """
    Z-score normalize a signal with optional winsorization.

    Args:
        signal: Raw signal series
        window: Rolling window for z-score calculation
        winsorize_threshold: Cap z-scores at +/- this value

    Returns:
        Z-scored signal series
    """
    rolling_mean = signal.rolling(window, min_periods=20).mean()
    rolling_std = signal.rolling(window, min_periods=20).std()

    # Prevent division by zero
    rolling_std = rolling_std.clip(lower=1e-8)

    z_signal = (signal - rolling_mean) / rolling_std

    # Winsorize
    z_signal = z_signal.clip(lower=-winsorize_threshold, upper=winsorize_threshold)

    # Normalize to [-1, 1]
    z_signal = z_signal / winsorize_threshold

    return z_signal


# =============================================================================
# MOMENTUM SIGNAL ENGINE CLASS
# =============================================================================

class MomentumSignalEngine:
    """
    Engine for generating momentum signals across multiple assets.

    Handles signal generation, normalization, and aggregation.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize momentum signal engine.

        Args:
            config: Configuration dictionary with momentum parameters
        """
        self.config = config or {}

        # Default parameters
        self.lookbacks = self.config.get('lookback_windows', {
            'short': 20,
            'medium': 60,
            'long': 252
        })
        self.weights = self.config.get('weights', {
            'short': 0.25,
            'medium': 0.50,
            'long': 0.25
        })
        self.vol_lookback = self.config.get('vol_lookback', 63)
        self.zscore_window = self.config.get('zscore_window', 252)
        self.zscore_threshold = self.config.get('zscore_threshold', 2.0)

    def compute_signal(self, prices: pd.Series,
                       ticker: str = 'unknown') -> MomentumSignal:
        """
        Compute comprehensive momentum signal for a single asset.

        Args:
            prices: Price series
            ticker: Asset ticker for identification

        Returns:
            MomentumSignal object with all components
        """
        # Get lookbacks and weights as lists
        lookback_list = [
            self.lookbacks['short'],
            self.lookbacks['medium'],
            self.lookbacks['long']
        ]
        weight_list = [
            self.weights['short'],
            self.weights['medium'],
            self.weights['long']
        ]

        # Compute ensemble momentum
        raw_signal = compute_tsmom_ensemble(prices, lookback_list, weight_list)

        # Z-score normalize
        z_signal = zscore_signal(
            raw_signal,
            window=self.zscore_window,
            winsorize_threshold=self.zscore_threshold
        )

        # Get latest values
        latest_raw = raw_signal.iloc[-1] if len(raw_signal) > 0 else 0.0
        latest_z = z_signal.iloc[-1] if len(z_signal) > 0 else 0.0

        # Calculate returns at each lookback
        returns_dict = {}
        for name, lookback in self.lookbacks.items():
            if len(prices) > lookback:
                ret = prices.pct_change(lookback).iloc[-1]
                returns_dict[lookback] = ret if not pd.isna(ret) else 0.0

        # Determine direction
        if latest_z > 0.25:
            direction = 'long'
        elif latest_z < -0.25:
            direction = 'short'
        else:
            direction = 'neutral'

        # Determine strength
        abs_z = abs(latest_z)
        if abs_z > 0.6:
            strength = 'strong'
        elif abs_z > 0.25:
            strength = 'moderate'
        else:
            strength = 'weak'

        return MomentumSignal(
            ticker=ticker,
            signal=latest_raw if not pd.isna(latest_raw) else 0.0,
            signal_zscore=latest_z if not pd.isna(latest_z) else 0.0,
            returns=returns_dict,
            direction=direction,
            strength=strength,
            timestamp=prices.index[-1] if len(prices) > 0 else pd.Timestamp.now()
        )

    def compute_signals_batch(self, price_data: Dict[str, pd.Series]) -> Dict[str, MomentumSignal]:
        """
        Compute momentum signals for multiple assets.

        Args:
            price_data: Dictionary mapping ticker to price series

        Returns:
            Dictionary mapping ticker to MomentumSignal
        """
        results = {}
        for ticker, prices in price_data.items():
            try:
                results[ticker] = self.compute_signal(prices, ticker)
            except Exception as e:
                logger.error(f"Error computing momentum for {ticker}: {e}")

        return results

    def get_signal_dataframe(self, price_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compute signals and return as DataFrame for analysis.

        Args:
            price_data: Dictionary mapping ticker to price series

        Returns:
            DataFrame with signal components
        """
        signals = self.compute_signals_batch(price_data)

        rows = []
        for ticker, sig in signals.items():
            rows.append({
                'ticker': sig.ticker,
                'signal': sig.signal,
                'signal_zscore': sig.signal_zscore,
                'direction': sig.direction,
                'strength': sig.strength,
                'return_20d': sig.returns.get(20, np.nan),
                'return_60d': sig.returns.get(60, np.nan),
                'return_252d': sig.returns.get(252, np.nan),
            })

        return pd.DataFrame(rows)

    def compute_momentum_timeseries(self, prices: pd.Series) -> pd.DataFrame:
        """
        Compute full momentum timeseries for analysis/backtesting.

        Args:
            prices: Price series

        Returns:
            DataFrame with momentum components over time
        """
        lookback_list = [
            self.lookbacks['short'],
            self.lookbacks['medium'],
            self.lookbacks['long']
        ]
        weight_list = [
            self.weights['short'],
            self.weights['medium'],
            self.weights['long']
        ]

        df = pd.DataFrame(index=prices.index)
        df['price'] = prices

        # Individual momentum components
        df['mom_short'] = compute_tsmom_scaled(prices, self.lookbacks['short'])
        df['mom_medium'] = compute_tsmom_scaled(prices, self.lookbacks['medium'])
        df['mom_long'] = compute_tsmom_scaled(prices, self.lookbacks['long'])

        # Ensemble
        df['mom_ensemble'] = compute_tsmom_ensemble(prices, lookback_list, weight_list)

        # Z-scored
        df['mom_zscore'] = zscore_signal(
            df['mom_ensemble'],
            self.zscore_window,
            self.zscore_threshold
        )

        # MA crossover
        df['ma_crossover'] = compute_ma_crossover(prices, 50, 200)

        # Trend strength
        df['trend_strength'] = compute_ma_trend_strength(prices)

        return df
