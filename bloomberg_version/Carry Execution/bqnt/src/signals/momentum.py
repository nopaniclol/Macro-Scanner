"""
Momentum Signal Engine for BQNT Carry Execution.

Calculates multi-horizon momentum signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from ..config.parameters import MOMENTUM_PARAMS


class MomentumSignalEngine:
    """
    Generates momentum signals from price data.

    Uses multiple lookback windows weighted together:
    - Short-term (5 days): 20%
    - Medium-term (21 days): 40%
    - Long-term (63 days): 40%
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize momentum signal engine.

        Args:
            params: Optional parameter overrides
        """
        self.params = {**MOMENTUM_PARAMS, **(params or {})}
        self._windows = list(self.params['window_weights'].keys())
        self._weights = self.params['window_weights']
        self._ema_span = self.params['ema_span']
        self._max_signal = self.params['max_signal']
        self._min_signal = self.params['min_signal']

    def calculate_returns(
        self,
        prices: pd.DataFrame,
        window: int
    ) -> pd.DataFrame:
        """
        Calculate returns over a specified window.

        Args:
            prices: DataFrame of prices
            window: Lookback window in days

        Returns:
            DataFrame of returns
        """
        return prices.pct_change(window)

    def calculate_momentum_single(
        self,
        prices: pd.DataFrame,
        window: int
    ) -> pd.DataFrame:
        """
        Calculate momentum signal for a single window.

        Args:
            prices: DataFrame of prices
            window: Lookback window

        Returns:
            DataFrame of z-scored momentum
        """
        # Calculate returns
        returns = self.calculate_returns(prices, window)

        # Rolling z-score
        rolling_mean = returns.rolling(window=window * 2, min_periods=window).mean()
        rolling_std = returns.rolling(window=window * 2, min_periods=window).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        zscore = (returns - rolling_mean) / rolling_std

        return zscore

    def calculate_signals(
        self,
        prices: pd.DataFrame,
        smooth: bool = True
    ) -> pd.DataFrame:
        """
        Generate combined momentum signals from price data.

        Args:
            prices: DataFrame of prices
            smooth: Whether to apply EMA smoothing

        Returns:
            DataFrame of momentum signals (-1 to +1)
        """
        if prices.empty:
            return pd.DataFrame()

        # Calculate momentum for each window
        momentum_signals = {}
        for window in self._windows:
            weight = self._weights[window]
            mom = self.calculate_momentum_single(prices, window)
            momentum_signals[window] = mom * weight

        # Combine weighted signals
        combined = pd.DataFrame(index=prices.index, columns=prices.columns)
        combined[:] = 0.0

        for window, mom in momentum_signals.items():
            combined = combined.add(mom.fillna(0), fill_value=0)

        # Apply EMA smoothing
        if smooth:
            combined = combined.ewm(span=self._ema_span).mean()

        # Cross-sectional normalization
        combined = self._cross_sectional_zscore(combined)

        # Clip to bounds
        combined = combined.clip(lower=self._min_signal, upper=self._max_signal)

        return combined

    def _cross_sectional_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cross-sectional z-score.

        Args:
            df: DataFrame of values

        Returns:
            DataFrame of z-scores
        """
        row_mean = df.mean(axis=1)
        row_std = df.std(axis=1).replace(0, 1)

        return df.sub(row_mean, axis=0).div(row_std, axis=0)

    def get_latest_signals(self, prices: pd.DataFrame) -> pd.Series:
        """
        Get most recent momentum signals.

        Args:
            prices: DataFrame of prices

        Returns:
            Series of latest signals
        """
        signals = self.calculate_signals(prices)

        if signals.empty:
            return pd.Series()

        return signals.iloc[-1]

    def calculate_trend_strength(
        self,
        prices: pd.DataFrame,
        window: int = 21
    ) -> pd.DataFrame:
        """
        Calculate trend strength indicator.

        Uses ratio of directional move to total range.

        Args:
            prices: DataFrame of prices
            window: Lookback window

        Returns:
            DataFrame of trend strength (0 to 1)
        """
        # Directional move
        direction = prices.diff(window).abs()

        # Total range (sum of absolute daily moves)
        daily_moves = prices.diff().abs()
        total_range = daily_moves.rolling(window).sum()

        # Trend strength = directional / total (0 to 1)
        # Higher = more trending, lower = more choppy
        strength = direction / total_range.replace(0, np.nan)

        return strength.clip(0, 1)

    def get_momentum_breakdown(
        self,
        prices: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Get individual momentum components.

        Args:
            prices: DataFrame of prices

        Returns:
            Dict with momentum by window
        """
        breakdown = {}
        for window in self._windows:
            mom = self.calculate_momentum_single(prices, window)
            breakdown[f'{window}d'] = mom

        return breakdown


class TrendFollowingSignal:
    """
    Simple trend-following signal based on moving average crossover.
    """

    def __init__(self, fast_window: int = 10, slow_window: int = 50):
        """
        Initialize trend signal.

        Args:
            fast_window: Fast MA period
            slow_window: Slow MA period
        """
        self.fast = fast_window
        self.slow = slow_window

    def calculate(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend signal.

        Args:
            prices: DataFrame of prices

        Returns:
            DataFrame of signals (+1 = bullish, -1 = bearish)
        """
        fast_ma = prices.rolling(self.fast).mean()
        slow_ma = prices.rolling(self.slow).mean()

        # Signal: +1 when fast > slow, -1 when fast < slow
        signal = np.sign(fast_ma - slow_ma)

        return signal
