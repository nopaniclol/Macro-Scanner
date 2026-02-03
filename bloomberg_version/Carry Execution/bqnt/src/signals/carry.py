"""
Carry Signal Engine for BQNT Carry Execution.

Calculates carry signals from FX forward points.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from ..config.parameters import CARRY_PARAMS
from ..config.tickers import ASSET_METADATA, get_carry_eligible_pairs


class CarrySignalEngine:
    """
    Generates carry signals from FX forward points.

    Carry = (Forward - Spot) / Spot * Annualization Factor

    Positive carry = Long position earns interest differential
    Negative carry = Short position earns interest differential
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize carry signal engine.

        Args:
            params: Optional parameter overrides
        """
        self.params = {**CARRY_PARAMS, **(params or {})}
        self._annualization = self.params['annualization_factor']
        self._normalize_by_vol = self.params['normalize_by_vol']
        self._vol_lookback = self.params['vol_lookback']
        self._max_signal = self.params['max_signal']
        self._min_signal = self.params['min_signal']

    def calculate_carry(
        self,
        spot_prices: pd.DataFrame,
        forward_prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate raw carry from spot and forward prices.

        Args:
            spot_prices: DataFrame of spot prices (FX pairs as columns)
            forward_prices: DataFrame of forward prices

        Returns:
            DataFrame of carry values (annualized)
        """
        # Ensure alignment
        common_dates = spot_prices.index.intersection(forward_prices.index)

        # Map forwards to spots
        carry_data = {}
        for pair in get_carry_eligible_pairs():
            if pair not in spot_prices.columns:
                continue

            # Get forward key for this pair
            meta = ASSET_METADATA.get(pair, {})
            forward_key = meta.get('forward')

            if forward_key and forward_key in forward_prices.columns:
                spot = spot_prices.loc[common_dates, pair]
                forward = forward_prices.loc[common_dates, forward_key]

                # Calculate carry: (F - S) / S * annualization
                carry = (forward - spot) / spot * self._annualization
                carry_data[pair] = carry

        return pd.DataFrame(carry_data)

    def calculate_signals(
        self,
        spot_prices: pd.DataFrame,
        forward_prices: pd.DataFrame,
        volatility: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate carry signals from price data.

        Args:
            spot_prices: DataFrame of spot prices
            forward_prices: DataFrame of forward prices
            volatility: Optional volatility for normalization

        Returns:
            DataFrame of carry signals (-1 to +1)
        """
        # Calculate raw carry
        carry = self.calculate_carry(spot_prices, forward_prices)

        if carry.empty:
            return pd.DataFrame()

        # Normalize by volatility if enabled
        if self._normalize_by_vol and volatility is not None:
            # Align volatility to carry
            vol_aligned = volatility.reindex(columns=carry.columns)
            vol_aligned = vol_aligned.reindex(carry.index)

            # Avoid division by zero
            vol_aligned = vol_aligned.replace(0, np.nan).fillna(method='ffill')

            # Normalize carry by volatility
            signals = carry / vol_aligned
        else:
            # Use raw carry as signal
            signals = carry

        # Cross-sectional z-score (rank within each day)
        signals = self._cross_sectional_zscore(signals)

        # Clip to bounds
        signals = signals.clip(lower=self._min_signal, upper=self._max_signal)

        return signals

    def _cross_sectional_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cross-sectional z-score (rank across assets each day).

        Args:
            df: DataFrame of values

        Returns:
            DataFrame of z-scores
        """
        # Calculate mean and std across columns for each row
        row_mean = df.mean(axis=1)
        row_std = df.std(axis=1)

        # Avoid division by zero
        row_std = row_std.replace(0, 1)

        # Z-score
        zscore = df.sub(row_mean, axis=0).div(row_std, axis=0)

        return zscore

    def get_latest_signals(
        self,
        spot_prices: pd.DataFrame,
        forward_prices: pd.DataFrame,
        volatility: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Get most recent carry signals.

        Args:
            spot_prices: DataFrame of spot prices
            forward_prices: DataFrame of forward prices
            volatility: Optional volatility for normalization

        Returns:
            Series of latest signals
        """
        signals = self.calculate_signals(spot_prices, forward_prices, volatility)

        if signals.empty:
            return pd.Series()

        return signals.iloc[-1]

    def get_signal_strength(self, signals: pd.Series) -> Dict[str, str]:
        """
        Categorize signal strength.

        Args:
            signals: Series of signals

        Returns:
            Dict mapping asset to strength category
        """
        strength = {}
        for asset, signal in signals.items():
            if abs(signal) > 0.7:
                strength[asset] = 'strong'
            elif abs(signal) > 0.3:
                strength[asset] = 'moderate'
            else:
                strength[asset] = 'weak'

        return strength


def calculate_implied_yield_differential(
    spot: float,
    forward: float,
    days_to_expiry: int = 90
) -> float:
    """
    Calculate implied yield differential from spot/forward.

    Args:
        spot: Spot price
        forward: Forward price
        days_to_expiry: Days until forward settlement

    Returns:
        Annualized yield differential
    """
    if spot == 0:
        return 0.0

    forward_points = forward - spot
    annualization = 360 / days_to_expiry

    return (forward_points / spot) * annualization
