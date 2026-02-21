"""
Carry Signal Engine

FX carry calculation via forward points or rate differentials.
Implements carry-to-risk and carry momentum signals.
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
class CarrySignal:
    """Container for carry signal output."""
    ticker: str
    carry_annualized: float     # Annualized carry in %
    carry_zscore: float         # Z-scored carry
    carry_momentum: float       # Change in carry (carry-of-carry)
    carry_to_risk: float        # Carry / volatility
    signal: float               # Combined carry signal [-1, 1]
    direction: str              # 'long', 'short', 'neutral'
    rank: int                   # Cross-sectional rank
    timestamp: pd.Timestamp


# =============================================================================
# CORE CARRY FUNCTIONS
# =============================================================================

def compute_carry_from_forwards(spot: pd.Series,
                                 forward: pd.Series,
                                 tenor_days: int) -> pd.Series:
    """
    Compute annualized carry from spot and forward prices.

    Carry = (Spot - Forward) / Spot * (365 / tenor_days)

    For FX: Forward > Spot means forward premium (negative carry for long position)
            Forward < Spot means forward discount (positive carry for long position)

    Note: Convention varies by pair. This assumes quote convention where
    higher forward = more expensive to buy in future.

    Args:
        spot: Spot price series
        forward: Forward price series (outright, not points)
        tenor_days: Forward tenor in days

    Returns:
        Series of annualized carry in decimal (0.05 = 5%)
    """
    if len(spot) != len(forward):
        raise ValueError("spot and forward must have same length")

    # Forward premium/discount
    forward_premium = (forward - spot) / spot

    # Annualize
    annualization = 365 / tenor_days
    carry = -forward_premium * annualization  # Negative because long spot = sell forward

    return carry


def compute_carry_from_forward_points(spot: pd.Series,
                                       forward_points: pd.Series,
                                       tenor_days: int,
                                       point_divisor: float = 10000) -> pd.Series:
    """
    Compute annualized carry from spot and forward points.

    Forward points are typically quoted in pips (1/10000 for most pairs).
    Forward outright = Spot + Forward Points / Point Divisor

    Args:
        spot: Spot price series
        forward_points: Forward points series (in pips)
        tenor_days: Forward tenor in days
        point_divisor: Divisor to convert points to price (10000 for most, 100 for JPY)

    Returns:
        Series of annualized carry in decimal
    """
    # Convert points to price adjustment
    forward_adjustment = forward_points / point_divisor

    # Forward outright
    forward_outright = spot + forward_adjustment

    return compute_carry_from_forwards(spot, forward_outright, tenor_days)


def compute_carry_from_rates(r_foreign: pd.Series,
                              r_domestic: pd.Series) -> pd.Series:
    """
    Compute carry from interest rate differential.

    For FX pair XXX/YYY:
    - Long XXX/YYY = Long XXX, Short YYY
    - Carry = r_XXX - r_YYY (you receive foreign, pay domestic)

    For USD-based pairs (EURUSD = EUR/USD):
    - Long EURUSD = Long EUR, Short USD
    - Carry = r_EUR - r_USD

    Args:
        r_foreign: Foreign (base) currency rate in decimal (0.05 = 5%)
        r_domestic: Domestic (quote) currency rate in decimal

    Returns:
        Series of carry in decimal (annualized)
    """
    # Simple rate differential
    carry = r_foreign - r_domestic

    return carry


def compute_carry_momentum(carry: pd.Series,
                            lookback: int = 20) -> pd.Series:
    """
    Compute carry momentum (change in carry over time).

    Rising carry = positive momentum (strengthening carry trade)
    Falling carry = negative momentum (weakening carry trade)

    Args:
        carry: Carry series
        lookback: Lookback for change calculation

    Returns:
        Series of carry changes
    """
    carry_change = carry.diff(lookback)
    return carry_change


def compute_carry_to_risk(carry: pd.Series,
                           prices: pd.Series,
                           vol_lookback: int = 63) -> pd.Series:
    """
    Compute carry-to-risk ratio (Sharpe-like metric).

    Carry / Volatility gives risk-adjusted carry signal.

    Args:
        carry: Annualized carry series
        prices: Price series for volatility calculation
        vol_lookback: Lookback for volatility estimation

    Returns:
        Series of carry-to-risk ratios
    """
    # Calculate rolling volatility
    returns = prices.pct_change()
    vol = returns.rolling(vol_lookback).std() * np.sqrt(252)

    # Floor volatility to prevent division issues
    vol = vol.clip(lower=0.01)

    # Carry to risk
    carry_risk = carry / vol

    return carry_risk


def zscore_carry(carry: pd.Series,
                  window: int = 252,
                  winsorize: float = 2.5) -> pd.Series:
    """
    Z-score normalize carry signal.

    Args:
        carry: Raw carry series
        window: Rolling window for stats
        winsorize: Cap at +/- this many standard deviations

    Returns:
        Z-scored carry series
    """
    rolling_mean = carry.rolling(window, min_periods=20).mean()
    rolling_std = carry.rolling(window, min_periods=20).std()

    rolling_std = rolling_std.clip(lower=1e-8)

    z_carry = (carry - rolling_mean) / rolling_std
    z_carry = z_carry.clip(lower=-winsorize, upper=winsorize)

    # Normalize to [-1, 1]
    z_carry = z_carry / winsorize

    return z_carry


def rank_carries_cross_sectional(carries: Dict[str, float]) -> Dict[str, int]:
    """
    Rank carries across assets (cross-sectional rank).

    Highest carry = rank 1.

    Args:
        carries: Dictionary of ticker -> carry value

    Returns:
        Dictionary of ticker -> rank (1 = highest carry)
    """
    sorted_tickers = sorted(carries.keys(), key=lambda x: carries[x], reverse=True)
    return {ticker: rank + 1 for rank, ticker in enumerate(sorted_tickers)}


# =============================================================================
# CARRY SIGNAL ENGINE CLASS
# =============================================================================

class CarrySignalEngine:
    """
    Engine for generating carry signals across FX pairs.

    Supports both forward-based and rate-differential carry calculation.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize carry signal engine.

        Args:
            config: Configuration dictionary with carry parameters
        """
        self.config = config or {}

        # Default parameters
        self.primary_tenor = self.config.get('tenor', '3M')
        self.tenor_days_map = {
            '1W': 7,
            '2W': 14,
            '1M': 30,
            '2M': 60,
            '3M': 91,
            '6M': 182,
            '1Y': 365,
        }
        self.point_divisors = self.config.get('point_divisor', {
            'EURUSD': 10000,
            'GBPUSD': 10000,
            'AUDUSD': 10000,
            'USDCAD': 10000,
            'USDCHF': 10000,
            'USDJPY': 100,
            'USDCNH': 10000,
        })
        self.carry_momentum_lookback = self.config.get('carry_momentum_lookback', 20)
        self.carry_momentum_weight = self.config.get('carry_momentum_blend', 0.3)
        self.zscore_window = self.config.get('zscore_window', 252)
        self.zscore_threshold = self.config.get('zscore_threshold', 2.5)
        self.vol_lookback = self.config.get('vol_lookback', 63)

    def get_tenor_days(self, tenor: str) -> int:
        """Get number of days for a tenor string."""
        return self.tenor_days_map.get(tenor, 91)

    def get_point_divisor(self, ticker: str) -> float:
        """Get point divisor for a ticker."""
        return self.point_divisors.get(ticker, 10000)

    def compute_carry_from_forward_data(self,
                                         spot: pd.Series,
                                         forward_points: pd.Series,
                                         ticker: str,
                                         tenor: str = None) -> pd.Series:
        """
        Compute carry from spot and forward points data.

        Args:
            spot: Spot price series
            forward_points: Forward points series
            ticker: Currency pair ticker
            tenor: Forward tenor (default: primary_tenor)

        Returns:
            Annualized carry series
        """
        tenor = tenor or self.primary_tenor
        tenor_days = self.get_tenor_days(tenor)
        point_divisor = self.get_point_divisor(ticker)

        return compute_carry_from_forward_points(
            spot, forward_points, tenor_days, point_divisor
        )

    def compute_carry_from_rate_data(self,
                                      r_foreign: pd.Series,
                                      r_domestic: pd.Series) -> pd.Series:
        """
        Compute carry from rate differential data.

        Args:
            r_foreign: Foreign currency rate series
            r_domestic: Domestic currency rate series

        Returns:
            Annualized carry series
        """
        return compute_carry_from_rates(r_foreign, r_domestic)

    def compute_signal(self,
                       carry: pd.Series,
                       prices: pd.Series,
                       ticker: str = 'unknown',
                       cross_sectional_carries: Optional[Dict[str, float]] = None) -> CarrySignal:
        """
        Compute comprehensive carry signal for a single asset.

        Args:
            carry: Annualized carry series
            prices: Price series for volatility calculation
            ticker: Asset ticker
            cross_sectional_carries: Dict of all asset carries for ranking

        Returns:
            CarrySignal object
        """
        # Current carry
        current_carry = carry.iloc[-1] if len(carry) > 0 else 0.0

        # Z-score
        z_carry = zscore_carry(carry, self.zscore_window, self.zscore_threshold)
        current_z = z_carry.iloc[-1] if len(z_carry) > 0 else 0.0

        # Carry momentum
        carry_mom = compute_carry_momentum(carry, self.carry_momentum_lookback)
        current_mom = carry_mom.iloc[-1] if len(carry_mom) > 0 else 0.0

        # Carry to risk
        c2r = compute_carry_to_risk(carry, prices, self.vol_lookback)
        current_c2r = c2r.iloc[-1] if len(c2r) > 0 else 0.0

        # Combined signal: blend level carry with carry momentum
        level_weight = 1.0 - self.carry_momentum_weight
        combined_signal = (current_z * level_weight +
                          np.sign(current_mom) * abs(current_z) * self.carry_momentum_weight)

        # Clip to [-1, 1]
        combined_signal = np.clip(combined_signal, -1, 1)

        # Determine direction
        if combined_signal > 0.2:
            direction = 'long'
        elif combined_signal < -0.2:
            direction = 'short'
        else:
            direction = 'neutral'

        # Cross-sectional rank
        if cross_sectional_carries:
            ranks = rank_carries_cross_sectional(cross_sectional_carries)
            rank = ranks.get(ticker, len(cross_sectional_carries))
        else:
            rank = 0

        return CarrySignal(
            ticker=ticker,
            carry_annualized=current_carry * 100 if not pd.isna(current_carry) else 0.0,  # Convert to %
            carry_zscore=current_z if not pd.isna(current_z) else 0.0,
            carry_momentum=current_mom if not pd.isna(current_mom) else 0.0,
            carry_to_risk=current_c2r if not pd.isna(current_c2r) else 0.0,
            signal=combined_signal if not pd.isna(combined_signal) else 0.0,
            direction=direction,
            rank=rank,
            timestamp=carry.index[-1] if len(carry) > 0 else pd.Timestamp.now()
        )

    def compute_signals_batch(self,
                               carry_data: Dict[str, pd.Series],
                               price_data: Dict[str, pd.Series]) -> Dict[str, CarrySignal]:
        """
        Compute carry signals for multiple assets.

        Args:
            carry_data: Dictionary mapping ticker to carry series
            price_data: Dictionary mapping ticker to price series

        Returns:
            Dictionary mapping ticker to CarrySignal
        """
        # First, get current carries for cross-sectional ranking
        current_carries = {}
        for ticker, carry in carry_data.items():
            if len(carry) > 0:
                current_carries[ticker] = carry.iloc[-1]

        # Compute individual signals
        results = {}
        for ticker in carry_data.keys():
            try:
                carry = carry_data[ticker]
                prices = price_data.get(ticker, pd.Series(dtype=float))

                if len(prices) == 0:
                    # Use carry index for prices if not provided
                    prices = pd.Series(1.0, index=carry.index)

                results[ticker] = self.compute_signal(
                    carry, prices, ticker, current_carries
                )
            except Exception as e:
                logger.error(f"Error computing carry for {ticker}: {e}")

        return results

    def get_signal_dataframe(self,
                              carry_data: Dict[str, pd.Series],
                              price_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compute signals and return as DataFrame.

        Args:
            carry_data: Dictionary mapping ticker to carry series
            price_data: Dictionary mapping ticker to price series

        Returns:
            DataFrame with carry signal components
        """
        signals = self.compute_signals_batch(carry_data, price_data)

        rows = []
        for ticker, sig in signals.items():
            rows.append({
                'ticker': sig.ticker,
                'carry_annualized': sig.carry_annualized,
                'carry_zscore': sig.carry_zscore,
                'carry_momentum': sig.carry_momentum,
                'carry_to_risk': sig.carry_to_risk,
                'signal': sig.signal,
                'direction': sig.direction,
                'rank': sig.rank,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('carry_annualized', ascending=False)

        return df

    def compute_carry_timeseries(self,
                                  spot: pd.Series,
                                  forward_points: pd.Series,
                                  ticker: str,
                                  tenor: str = None) -> pd.DataFrame:
        """
        Compute full carry timeseries for analysis.

        Args:
            spot: Spot price series
            forward_points: Forward points series
            ticker: Currency pair ticker
            tenor: Forward tenor

        Returns:
            DataFrame with carry components over time
        """
        tenor = tenor or self.primary_tenor

        df = pd.DataFrame(index=spot.index)
        df['spot'] = spot
        df['forward_points'] = forward_points

        # Compute carry
        df['carry'] = self.compute_carry_from_forward_data(
            spot, forward_points, ticker, tenor
        )

        # Z-score
        df['carry_zscore'] = zscore_carry(
            df['carry'], self.zscore_window, self.zscore_threshold
        )

        # Momentum
        df['carry_momentum'] = compute_carry_momentum(
            df['carry'], self.carry_momentum_lookback
        )

        # Carry to risk
        df['carry_to_risk'] = compute_carry_to_risk(
            df['carry'], spot, self.vol_lookback
        )

        return df
