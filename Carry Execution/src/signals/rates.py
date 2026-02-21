"""
Rates Signal Engine

Yield curve analytics for rates overlay and regime detection.
Computes level, slope, curvature, and momentum signals.
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
class RatesSignal:
    """Container for rates signal output."""
    currency: str
    level: float               # Weighted average yield
    level_zscore: float        # Z-scored level
    slope: float               # 2s10s or specified slope
    slope_zscore: float        # Z-scored slope
    curvature: float           # Butterfly (2s5s10s)
    curvature_zscore: float    # Z-scored curvature
    momentum: float            # Rate change momentum
    regime: str                # 'easing', 'hiking', 'neutral'
    curve_shape: str           # 'normal', 'flat', 'inverted'
    signal: float              # Combined rates signal
    timestamp: pd.Timestamp


# =============================================================================
# CURVE ANALYTICS FUNCTIONS
# =============================================================================

def compute_curve_level(curve_df: pd.DataFrame,
                         tenors: List[str] = None,
                         weights: Dict[str, float] = None) -> pd.Series:
    """
    Compute weighted average yield level.

    Args:
        curve_df: DataFrame with tenor columns (e.g., '2Y', '5Y', '10Y')
        tenors: List of tenors to include (default: all columns)
        weights: Tenor weights (default: equal weight)

    Returns:
        Series of weighted average yields
    """
    if tenors is None:
        tenors = [col for col in curve_df.columns if col not in ['Date', 'Currency']]

    if weights is None:
        # Equal weight by default
        weights = {t: 1.0 / len(tenors) for t in tenors}

    # Compute weighted average
    level = pd.Series(0.0, index=curve_df.index)
    total_weight = 0.0

    for tenor in tenors:
        if tenor in curve_df.columns:
            w = weights.get(tenor, 0.0)
            level += curve_df[tenor].fillna(0) * w
            total_weight += w

    if total_weight > 0:
        level = level / total_weight

    return level


def compute_curve_slope(curve_df: pd.DataFrame,
                         short_tenor: str = '2Y',
                         long_tenor: str = '10Y') -> pd.Series:
    """
    Compute yield curve slope.

    Slope = Long Rate - Short Rate (in bps)
    Positive = normal curve (steepener)
    Negative = inverted curve (flattener)

    Args:
        curve_df: DataFrame with tenor columns
        short_tenor: Short end tenor (default '2Y')
        long_tenor: Long end tenor (default '10Y')

    Returns:
        Series of slope values in basis points
    """
    if short_tenor not in curve_df.columns or long_tenor not in curve_df.columns:
        logger.warning(f"Tenors {short_tenor} or {long_tenor} not in curve data")
        return pd.Series(index=curve_df.index, dtype=float)

    # Slope in basis points (assuming input is in %)
    slope = (curve_df[long_tenor] - curve_df[short_tenor]) * 100

    return slope


def compute_curve_curvature(curve_df: pd.DataFrame,
                             short_tenor: str = '2Y',
                             mid_tenor: str = '5Y',
                             long_tenor: str = '10Y') -> pd.Series:
    """
    Compute yield curve curvature (butterfly).

    Curvature = 2 * Mid - Short - Long

    Positive = belly rich (yields lower than wings)
    Negative = belly cheap (yields higher than wings)

    Args:
        curve_df: DataFrame with tenor columns
        short_tenor: Short end tenor
        mid_tenor: Belly tenor
        long_tenor: Long end tenor

    Returns:
        Series of curvature values in basis points
    """
    required_tenors = [short_tenor, mid_tenor, long_tenor]
    for t in required_tenors:
        if t not in curve_df.columns:
            logger.warning(f"Tenor {t} not in curve data")
            return pd.Series(index=curve_df.index, dtype=float)

    # Butterfly in basis points
    curvature = (2 * curve_df[mid_tenor] - curve_df[short_tenor] - curve_df[long_tenor]) * 100

    return curvature


def compute_curve_momentum(curve_df: pd.DataFrame,
                            tenor: str = '10Y',
                            lookback: int = 20) -> pd.Series:
    """
    Compute rate momentum (change in yields).

    Args:
        curve_df: DataFrame with tenor columns
        tenor: Tenor for momentum calculation
        lookback: Lookback period for change

    Returns:
        Series of yield changes in basis points
    """
    if tenor not in curve_df.columns:
        logger.warning(f"Tenor {tenor} not in curve data")
        return pd.Series(index=curve_df.index, dtype=float)

    # Change in basis points
    momentum = curve_df[tenor].diff(lookback) * 100

    return momentum


def compute_curve_roll(curve_df: pd.DataFrame,
                        from_tenor: str = '10Y',
                        to_tenor: str = '7Y') -> pd.Series:
    """
    Compute curve roll-down (carry from duration).

    Roll = From Tenor Yield - To Tenor Yield
    Positive roll = earn yield as bond ages

    Args:
        curve_df: DataFrame with tenor columns
        from_tenor: Starting tenor
        to_tenor: Ending tenor (after roll)

    Returns:
        Series of roll values in basis points
    """
    if from_tenor not in curve_df.columns or to_tenor not in curve_df.columns:
        return pd.Series(index=curve_df.index, dtype=float)

    roll = (curve_df[from_tenor] - curve_df[to_tenor]) * 100

    return roll


# =============================================================================
# REGIME CLASSIFICATION
# =============================================================================

def classify_rate_regime(level_z: float,
                          slope_z: float,
                          level_momentum: float = 0.0) -> str:
    """
    Classify monetary policy regime from curve signals.

    Regimes:
    - 'hiking': Rising rates, flattening curve
    - 'easing': Falling rates, steepening curve
    - 'neutral': Stable rates

    Args:
        level_z: Z-scored rate level (positive = high rates)
        slope_z: Z-scored slope (positive = steep curve)
        level_momentum: Rate level momentum

    Returns:
        Regime classification string
    """
    # Hiking regime: rates rising and/or curve flattening
    if level_momentum > 0.5 or (level_z > 0.5 and slope_z < -0.3):
        return 'hiking'

    # Easing regime: rates falling and/or curve steepening
    elif level_momentum < -0.5 or (level_z < -0.5 and slope_z > 0.3):
        return 'easing'

    else:
        return 'neutral'


def classify_curve_shape(slope: float,
                          normal_threshold: float = 50,
                          inversion_threshold: float = 0) -> str:
    """
    Classify curve shape from slope.

    Args:
        slope: 2s10s slope in basis points
        normal_threshold: Slope above this = normal
        inversion_threshold: Slope below this = inverted

    Returns:
        Curve shape classification
    """
    if slope > normal_threshold:
        return 'normal'
    elif slope < inversion_threshold:
        return 'inverted'
    else:
        return 'flat'


def zscore_series(series: pd.Series,
                   window: int = 252,
                   winsorize: float = 2.0) -> pd.Series:
    """
    Z-score normalize a series.

    Args:
        series: Input series
        window: Rolling window for stats
        winsorize: Cap at +/- this value

    Returns:
        Z-scored series
    """
    rolling_mean = series.rolling(window, min_periods=20).mean()
    rolling_std = series.rolling(window, min_periods=20).std()
    rolling_std = rolling_std.clip(lower=1e-8)

    z = (series - rolling_mean) / rolling_std
    z = z.clip(lower=-winsorize, upper=winsorize)
    z = z / winsorize  # Normalize to [-1, 1]

    return z


# =============================================================================
# RATES SIGNAL ENGINE CLASS
# =============================================================================

class RatesSignalEngine:
    """
    Engine for generating rates curve signals.

    Computes level, slope, curvature, and momentum analytics.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize rates signal engine.

        Args:
            config: Configuration dictionary with rates parameters
        """
        self.config = config or {}

        # Default parameters
        self.key_tenors = self.config.get('key_tenors', {
            'short': '2Y',
            'belly': '5Y',
            'long': '10Y',
            'ultra_long': '30Y',
        })
        self.slope_tenors = self.config.get('slope_tenors', {
            'short': '2Y',
            'long': '10Y',
        })
        self.curvature_tenors = self.config.get('curvature_tenors', {
            'short': '2Y',
            'belly': '5Y',
            'long': '10Y',
        })
        self.level_weights = self.config.get('level_weights', {
            '2Y': 0.20,
            '5Y': 0.30,
            '10Y': 0.35,
            '30Y': 0.15,
        })
        self.zscore_window = self.config.get('zscore_window', 252)
        self.momentum_lookback = self.config.get('momentum_lookback', 20)
        self.normal_slope_threshold = self.config.get('normal_slope', 50)

    def compute_signal(self,
                       curve_df: pd.DataFrame,
                       currency: str = 'USD') -> RatesSignal:
        """
        Compute comprehensive rates signal.

        Args:
            curve_df: DataFrame with tenor columns
            currency: Currency code

        Returns:
            RatesSignal object
        """
        # Compute level
        level = compute_curve_level(
            curve_df,
            tenors=list(self.level_weights.keys()),
            weights=self.level_weights
        )
        level_z = zscore_series(level, self.zscore_window)

        # Compute slope
        slope = compute_curve_slope(
            curve_df,
            self.slope_tenors['short'],
            self.slope_tenors['long']
        )
        slope_z = zscore_series(slope, self.zscore_window)

        # Compute curvature
        curvature = compute_curve_curvature(
            curve_df,
            self.curvature_tenors['short'],
            self.curvature_tenors['belly'],
            self.curvature_tenors['long']
        )
        curvature_z = zscore_series(curvature, self.zscore_window)

        # Compute momentum
        momentum = compute_curve_momentum(
            curve_df,
            self.key_tenors['long'],
            self.momentum_lookback
        )

        # Get latest values
        current_level = level.iloc[-1] if len(level) > 0 else 0.0
        current_level_z = level_z.iloc[-1] if len(level_z) > 0 else 0.0
        current_slope = slope.iloc[-1] if len(slope) > 0 else 0.0
        current_slope_z = slope_z.iloc[-1] if len(slope_z) > 0 else 0.0
        current_curvature = curvature.iloc[-1] if len(curvature) > 0 else 0.0
        current_curvature_z = curvature_z.iloc[-1] if len(curvature_z) > 0 else 0.0
        current_momentum = momentum.iloc[-1] if len(momentum) > 0 else 0.0

        # Classify regime
        regime = classify_rate_regime(
            current_level_z,
            current_slope_z,
            current_momentum / 10  # Normalize momentum
        )

        # Classify curve shape
        curve_shape = classify_curve_shape(
            current_slope,
            self.normal_slope_threshold
        )

        # Combined signal: weighted combination of curve signals
        # Steeper curve and lower rates = risk-on for FX carry
        # Flatter curve and higher rates = risk-off
        combined_signal = (
            -current_level_z * 0.30 +   # Lower rates = positive
            current_slope_z * 0.40 +    # Steeper curve = positive
            current_curvature_z * 0.15 +  # Rich belly = positive
            -current_momentum / 50 * 0.15  # Falling rates = positive
        )
        combined_signal = np.clip(combined_signal, -1, 1)

        return RatesSignal(
            currency=currency,
            level=current_level if not pd.isna(current_level) else 0.0,
            level_zscore=current_level_z if not pd.isna(current_level_z) else 0.0,
            slope=current_slope if not pd.isna(current_slope) else 0.0,
            slope_zscore=current_slope_z if not pd.isna(current_slope_z) else 0.0,
            curvature=current_curvature if not pd.isna(current_curvature) else 0.0,
            curvature_zscore=current_curvature_z if not pd.isna(current_curvature_z) else 0.0,
            momentum=current_momentum if not pd.isna(current_momentum) else 0.0,
            regime=regime,
            curve_shape=curve_shape,
            signal=combined_signal if not pd.isna(combined_signal) else 0.0,
            timestamp=curve_df.index[-1] if len(curve_df) > 0 else pd.Timestamp.now()
        )

    def compute_signals_batch(self,
                               curve_data: Dict[str, pd.DataFrame]) -> Dict[str, RatesSignal]:
        """
        Compute rates signals for multiple currencies.

        Args:
            curve_data: Dictionary mapping currency to curve DataFrame

        Returns:
            Dictionary mapping currency to RatesSignal
        """
        results = {}
        for currency, curve_df in curve_data.items():
            try:
                results[currency] = self.compute_signal(curve_df, currency)
            except Exception as e:
                logger.error(f"Error computing rates for {currency}: {e}")

        return results

    def get_signal_dataframe(self,
                              curve_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute signals and return as DataFrame.

        Args:
            curve_data: Dictionary mapping currency to curve DataFrame

        Returns:
            DataFrame with rates signal components
        """
        signals = self.compute_signals_batch(curve_data)

        rows = []
        for currency, sig in signals.items():
            rows.append({
                'currency': sig.currency,
                'level': sig.level,
                'level_zscore': sig.level_zscore,
                'slope': sig.slope,
                'slope_zscore': sig.slope_zscore,
                'curvature': sig.curvature,
                'curvature_zscore': sig.curvature_zscore,
                'momentum': sig.momentum,
                'regime': sig.regime,
                'curve_shape': sig.curve_shape,
                'signal': sig.signal,
            })

        return pd.DataFrame(rows)

    def compute_rates_timeseries(self, curve_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute full rates timeseries for analysis.

        Args:
            curve_df: DataFrame with tenor columns

        Returns:
            DataFrame with rates analytics over time
        """
        df = curve_df.copy()

        # Level
        df['level'] = compute_curve_level(
            curve_df,
            tenors=list(self.level_weights.keys()),
            weights=self.level_weights
        )
        df['level_zscore'] = zscore_series(df['level'], self.zscore_window)

        # Slope
        df['slope'] = compute_curve_slope(
            curve_df,
            self.slope_tenors['short'],
            self.slope_tenors['long']
        )
        df['slope_zscore'] = zscore_series(df['slope'], self.zscore_window)

        # Curvature
        df['curvature'] = compute_curve_curvature(
            curve_df,
            self.curvature_tenors['short'],
            self.curvature_tenors['belly'],
            self.curvature_tenors['long']
        )
        df['curvature_zscore'] = zscore_series(df['curvature'], self.zscore_window)

        # Momentum
        df['momentum'] = compute_curve_momentum(
            curve_df,
            self.key_tenors['long'],
            self.momentum_lookback
        )

        return df

    def compute_curve_trade_signals(self, curve_df: pd.DataFrame) -> Dict:
        """
        Generate specific curve trade signals.

        Returns signals for:
        - Steepener/Flattener
        - Butterfly
        - Duration

        Args:
            curve_df: DataFrame with tenor columns

        Returns:
            Dictionary of trade signals
        """
        slope = compute_curve_slope(
            curve_df,
            self.slope_tenors['short'],
            self.slope_tenors['long']
        )
        slope_z = zscore_series(slope, self.zscore_window)

        curvature = compute_curve_curvature(
            curve_df,
            self.curvature_tenors['short'],
            self.curvature_tenors['belly'],
            self.curvature_tenors['long']
        )
        curvature_z = zscore_series(curvature, self.zscore_window)

        level_z = zscore_series(
            compute_curve_level(curve_df, list(self.level_weights.keys())),
            self.zscore_window
        )

        current_slope_z = slope_z.iloc[-1] if len(slope_z) > 0 else 0.0
        current_curvature_z = curvature_z.iloc[-1] if len(curvature_z) > 0 else 0.0
        current_level_z = level_z.iloc[-1] if len(level_z) > 0 else 0.0

        trades = {}

        # Steepener/Flattener
        if current_slope_z < -0.5:
            trades['curve'] = {
                'direction': 'steepener',
                'signal': -current_slope_z,
                'rationale': 'Curve too flat, expect steepening'
            }
        elif current_slope_z > 0.5:
            trades['curve'] = {
                'direction': 'flattener',
                'signal': current_slope_z,
                'rationale': 'Curve too steep, expect flattening'
            }
        else:
            trades['curve'] = {
                'direction': 'neutral',
                'signal': 0.0,
                'rationale': 'Curve shape neutral'
            }

        # Butterfly
        if current_curvature_z < -0.5:
            trades['butterfly'] = {
                'direction': 'sell_belly',
                'signal': -current_curvature_z,
                'rationale': 'Belly cheap, expect richening'
            }
        elif current_curvature_z > 0.5:
            trades['butterfly'] = {
                'direction': 'buy_belly',
                'signal': current_curvature_z,
                'rationale': 'Belly rich, expect cheapening'
            }
        else:
            trades['butterfly'] = {
                'direction': 'neutral',
                'signal': 0.0,
                'rationale': 'Curvature neutral'
            }

        # Duration (level)
        if current_level_z > 0.7:
            trades['duration'] = {
                'direction': 'receive',
                'signal': current_level_z,
                'rationale': 'Rates high, expect to fall'
            }
        elif current_level_z < -0.7:
            trades['duration'] = {
                'direction': 'pay',
                'signal': -current_level_z,
                'rationale': 'Rates low, expect to rise'
            }
        else:
            trades['duration'] = {
                'direction': 'neutral',
                'signal': 0.0,
                'rationale': 'Rate level neutral'
            }

        return trades
