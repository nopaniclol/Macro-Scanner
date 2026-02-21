"""
Value Signal Engine

PPP-based fair value calculation and valuation gap signals.
Implements mean reversion signals with multi-tier tilts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ValueSignal:
    """Container for value signal output."""
    ticker: str
    fair_value: float          # PPP-implied fair value
    spot: float                # Current spot rate
    valuation_gap: float       # % deviation from fair value
    valuation_zscore: float    # Z-scored gap
    value_tilt: float          # Position tilt [-1, 1]
    signal: float              # Combined value signal
    regime: str                # 'undervalued', 'overvalued', 'fair'
    mean_reversion_target: float  # Expected reversion
    timestamp: pd.Timestamp


# =============================================================================
# PPP FAIR VALUE CALCULATION
# =============================================================================

def compute_ppp_fair_value(ppp_base: float,
                            cpi_foreign: pd.Series,
                            cpi_domestic: pd.Series,
                            base_date: datetime = None) -> pd.Series:
    """
    Compute PPP-adjusted fair value over time.

    PPP Fair Value = PPP_base * (CPI_foreign / CPI_foreign_base) / (CPI_domestic / CPI_domestic_base)

    The PPP base is typically from IMF or OECD annual PPP estimates.
    This function adjusts the base PPP for relative inflation since the base date.

    Args:
        ppp_base: Base PPP exchange rate (from IMF/OECD)
        cpi_foreign: Foreign country CPI index series
        cpi_domestic: Domestic country CPI index series
        base_date: Date of the base PPP estimate (default: first date in series)

    Returns:
        Series of PPP fair values
    """
    # Align indices
    common_index = cpi_foreign.index.intersection(cpi_domestic.index)
    cpi_foreign = cpi_foreign.loc[common_index]
    cpi_domestic = cpi_domestic.loc[common_index]

    if len(cpi_foreign) == 0:
        return pd.Series(dtype=float)

    # Get base values
    if base_date is None:
        base_date = cpi_foreign.index[0]

    # Find closest date to base_date
    base_idx = cpi_foreign.index.get_indexer([base_date], method='nearest')[0]
    cpi_foreign_base = cpi_foreign.iloc[base_idx]
    cpi_domestic_base = cpi_domestic.iloc[base_idx]

    # Relative CPI ratios (cumulative inflation differential)
    relative_cpi = (cpi_foreign / cpi_foreign_base) / (cpi_domestic / cpi_domestic_base)

    # PPP fair value
    fair_value = ppp_base * relative_cpi

    return fair_value


def compute_ppp_fair_value_simple(ppp_level: float,
                                   inflation_diff_cumulative: pd.Series) -> pd.Series:
    """
    Simplified PPP fair value using cumulative inflation differential.

    Fair Value = PPP_base * exp(cumulative_inflation_diff)

    Args:
        ppp_level: Base PPP exchange rate
        inflation_diff_cumulative: Cumulative (foreign - domestic) inflation

    Returns:
        Series of PPP fair values
    """
    fair_value = ppp_level * np.exp(inflation_diff_cumulative)
    return fair_value


# =============================================================================
# VALUATION GAP CALCULATION
# =============================================================================

def compute_valuation_gap(spot: pd.Series,
                           fair_value: pd.Series) -> pd.Series:
    """
    Compute percentage deviation from fair value.

    Gap = (Spot - Fair Value) / Fair Value * 100

    Positive = overvalued (spot above fair value)
    Negative = undervalued (spot below fair value)

    Args:
        spot: Spot price series
        fair_value: Fair value series

    Returns:
        Series of valuation gaps in percentage
    """
    # Align indices
    common_index = spot.index.intersection(fair_value.index)

    if len(common_index) == 0:
        return pd.Series(dtype=float)

    spot_aligned = spot.loc[common_index]
    fv_aligned = fair_value.loc[common_index]

    # Percentage gap
    gap = (spot_aligned - fv_aligned) / fv_aligned * 100

    return gap


def compute_valuation_gap_log(spot: pd.Series,
                               fair_value: pd.Series) -> pd.Series:
    """
    Compute log deviation from fair value.

    More appropriate for FX as it's symmetric in both directions.

    Args:
        spot: Spot price series
        fair_value: Fair value series

    Returns:
        Series of log valuation gaps
    """
    common_index = spot.index.intersection(fair_value.index)

    if len(common_index) == 0:
        return pd.Series(dtype=float)

    spot_aligned = spot.loc[common_index]
    fv_aligned = fair_value.loc[common_index]

    # Log gap
    gap = np.log(spot_aligned / fv_aligned) * 100

    return gap


# =============================================================================
# VALUE TILT CALCULATION
# =============================================================================

def compute_value_tilt(gap: pd.Series,
                        thresholds: List[float] = None) -> pd.Series:
    """
    Compute multi-tier value tilt based on valuation gap.

    Maps valuation gaps to position tilts:
    - Extreme undervaluation: +1.4 (strong overweight)
    - Moderate undervaluation: +1.15
    - Fair value: 1.0 (neutral)
    - Moderate overvaluation: 0.85
    - Extreme overvaluation: 0.6 (strong underweight)

    Args:
        gap: Valuation gap series (in %)
        thresholds: Gap thresholds for tier transitions
                   [extreme_under, moderate_under, moderate_over, extreme_over]

    Returns:
        Series of position tilts
    """
    if thresholds is None:
        # Default thresholds (% deviation from fair value)
        thresholds = [-20, -10, 10, 20]  # Undervalue extreme, moderate, overvalue moderate, extreme

    extreme_under, moderate_under, moderate_over, extreme_over = thresholds

    # Multi-tier tilts
    tilts = [1.4, 1.15, 1.0, 0.85, 0.6]

    def get_tilt(g):
        if pd.isna(g):
            return 1.0
        if g <= extreme_under:
            return tilts[0]  # 1.4 - strong long
        elif g <= moderate_under:
            return tilts[1]  # 1.15 - moderate long
        elif g <= moderate_over:
            return tilts[2]  # 1.0 - neutral
        elif g <= extreme_over:
            return tilts[3]  # 0.85 - moderate short
        else:
            return tilts[4]  # 0.6 - strong short

    tilt = gap.apply(get_tilt)
    return tilt


def compute_value_tilt_continuous(gap: pd.Series,
                                   half_life: float = 20.0,
                                   max_tilt: float = 0.4) -> pd.Series:
    """
    Compute continuous value tilt using sigmoid function.

    Provides smoother transitions than discrete tiers.
    Tilt = tanh(gap / half_life) * max_tilt

    Args:
        gap: Valuation gap series (in %)
        half_life: Gap at which tilt is ~46% of max (tanh(1) ≈ 0.76)
        max_tilt: Maximum tilt magnitude

    Returns:
        Series of position tilts [-max_tilt, +max_tilt]
    """
    # Negative gap (undervalued) = positive tilt (buy)
    # Positive gap (overvalued) = negative tilt (sell)
    tilt = -np.tanh(gap / half_life) * max_tilt

    return tilt


# =============================================================================
# MEAN REVERSION ESTIMATION
# =============================================================================

def compute_mean_reversion_speed(gap: pd.Series,
                                  lookback: int = 252 * 5) -> float:
    """
    Estimate mean reversion speed using regression.

    Uses Ornstein-Uhlenbeck inspired approach:
    d(gap) = -kappa * gap * dt + noise

    Args:
        gap: Valuation gap series
        lookback: Period for estimation

    Returns:
        Estimated reversion speed (annual)
    """
    if len(gap) < lookback:
        lookback = len(gap)

    gap_subset = gap.iloc[-lookback:]
    gap_change = gap_subset.diff()

    # Remove NaN
    valid_mask = ~(gap_subset.isna() | gap_change.isna())
    gap_subset = gap_subset[valid_mask]
    gap_change = gap_change[valid_mask]

    if len(gap_subset) < 20:
        return 0.05  # Default assumption

    # Regression: gap_change = -kappa * gap
    # Solve using OLS
    X = gap_subset.iloc[:-1].values.reshape(-1, 1)
    y = gap_change.iloc[1:].values

    if len(X) == 0:
        return 0.05

    # Simple OLS
    kappa = -np.sum(X.flatten() * y) / np.sum(X.flatten() ** 2)

    # Annualize (assuming daily data)
    kappa_annual = kappa * 252

    # Bound to reasonable range
    kappa_annual = np.clip(kappa_annual, 0.01, 0.50)

    return kappa_annual


def compute_mean_reversion_target(current_gap: float,
                                   reversion_speed: float,
                                   horizon_months: int = 12) -> float:
    """
    Compute expected gap after mean reversion.

    Expected gap = current_gap * exp(-kappa * t)

    Args:
        current_gap: Current valuation gap (%)
        reversion_speed: Annual mean reversion speed
        horizon_months: Forecast horizon in months

    Returns:
        Expected gap after horizon
    """
    t = horizon_months / 12  # Convert to years
    expected_gap = current_gap * np.exp(-reversion_speed * t)
    return expected_gap


# =============================================================================
# VALUE SIGNAL ENGINE CLASS
# =============================================================================

class ValueSignalEngine:
    """
    Engine for generating PPP-based value signals.

    Computes fair values, valuation gaps, and mean reversion signals.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize value signal engine.

        Args:
            config: Configuration dictionary with value parameters
        """
        self.config = config or {}

        # Default parameters
        self.ppp_source = self.config.get('ppp_source', 'imf')
        self.reversion_speed = self.config.get('reversion_speed', 0.05)
        self.half_life_months = self.config.get('half_life_months', 36)
        self.lookback_years = self.config.get('lookback_years', 5)
        self.zscore_threshold = self.config.get('zscore_threshold', 2.0)

        # Gap thresholds for tiering (% deviation)
        self.gap_thresholds = self.config.get('gap_thresholds', [-20, -10, 10, 20])

        # Tilt magnitudes
        self.tilt_values = self.config.get('tilt_values', [1.4, 1.15, 1.0, 0.85, 0.6])

    def compute_fair_value_from_ppp(self,
                                     ppp_base: float,
                                     cpi_foreign: pd.Series,
                                     cpi_domestic: pd.Series,
                                     base_date: datetime = None) -> pd.Series:
        """
        Compute PPP fair value using CPI data.

        Args:
            ppp_base: Base PPP rate
            cpi_foreign: Foreign CPI series
            cpi_domestic: Domestic CPI series
            base_date: PPP base date

        Returns:
            Fair value series
        """
        return compute_ppp_fair_value(ppp_base, cpi_foreign, cpi_domestic, base_date)

    def compute_signal(self,
                       spot: pd.Series,
                       fair_value: pd.Series,
                       ticker: str = 'unknown') -> ValueSignal:
        """
        Compute comprehensive value signal for a single asset.

        Args:
            spot: Spot price series
            fair_value: Fair value series
            ticker: Asset ticker

        Returns:
            ValueSignal object
        """
        # Compute valuation gap
        gap = compute_valuation_gap(spot, fair_value)

        if len(gap) == 0:
            return ValueSignal(
                ticker=ticker,
                fair_value=np.nan,
                spot=np.nan,
                valuation_gap=0.0,
                valuation_zscore=0.0,
                value_tilt=1.0,
                signal=0.0,
                regime='fair',
                mean_reversion_target=0.0,
                timestamp=pd.Timestamp.now()
            )

        # Current values
        current_spot = spot.iloc[-1]
        current_fv = fair_value.iloc[-1] if len(fair_value) > 0 else np.nan
        current_gap = gap.iloc[-1]

        # Z-score the gap
        gap_zscore = self._zscore_gap(gap)
        current_z = gap_zscore.iloc[-1] if len(gap_zscore) > 0 else 0.0

        # Compute value tilt
        tilt = compute_value_tilt(gap, self.gap_thresholds)
        current_tilt = tilt.iloc[-1] if len(tilt) > 0 else 1.0

        # Continuous signal (normalized to [-1, 1])
        signal = compute_value_tilt_continuous(gap, half_life=15, max_tilt=1.0)
        current_signal = signal.iloc[-1] if len(signal) > 0 else 0.0

        # Estimate mean reversion
        lookback_days = self.lookback_years * 252
        reversion_speed = compute_mean_reversion_speed(gap, lookback_days)
        target_gap = compute_mean_reversion_target(
            current_gap, reversion_speed, horizon_months=12
        )

        # Determine regime
        if current_gap < -10:
            regime = 'undervalued'
        elif current_gap > 10:
            regime = 'overvalued'
        else:
            regime = 'fair'

        return ValueSignal(
            ticker=ticker,
            fair_value=current_fv if not pd.isna(current_fv) else 0.0,
            spot=current_spot if not pd.isna(current_spot) else 0.0,
            valuation_gap=current_gap if not pd.isna(current_gap) else 0.0,
            valuation_zscore=current_z if not pd.isna(current_z) else 0.0,
            value_tilt=current_tilt if not pd.isna(current_tilt) else 1.0,
            signal=current_signal if not pd.isna(current_signal) else 0.0,
            regime=regime,
            mean_reversion_target=target_gap if not pd.isna(target_gap) else 0.0,
            timestamp=spot.index[-1] if len(spot) > 0 else pd.Timestamp.now()
        )

    def _zscore_gap(self, gap: pd.Series, window: int = 252) -> pd.Series:
        """Z-score the valuation gap."""
        rolling_mean = gap.rolling(window, min_periods=20).mean()
        rolling_std = gap.rolling(window, min_periods=20).std()
        rolling_std = rolling_std.clip(lower=1e-8)

        z = (gap - rolling_mean) / rolling_std
        z = z.clip(lower=-self.zscore_threshold, upper=self.zscore_threshold)
        z = z / self.zscore_threshold

        return z

    def compute_signals_batch(self,
                               spot_data: Dict[str, pd.Series],
                               fair_value_data: Dict[str, pd.Series]) -> Dict[str, ValueSignal]:
        """
        Compute value signals for multiple assets.

        Args:
            spot_data: Dictionary mapping ticker to spot series
            fair_value_data: Dictionary mapping ticker to fair value series

        Returns:
            Dictionary mapping ticker to ValueSignal
        """
        results = {}
        for ticker in spot_data.keys():
            try:
                spot = spot_data[ticker]
                fair_value = fair_value_data.get(ticker, pd.Series(dtype=float))

                if len(fair_value) == 0:
                    logger.warning(f"No fair value data for {ticker}")
                    continue

                results[ticker] = self.compute_signal(spot, fair_value, ticker)
            except Exception as e:
                logger.error(f"Error computing value for {ticker}: {e}")

        return results

    def get_signal_dataframe(self,
                              spot_data: Dict[str, pd.Series],
                              fair_value_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compute signals and return as DataFrame.

        Args:
            spot_data: Dictionary mapping ticker to spot series
            fair_value_data: Dictionary mapping ticker to fair value series

        Returns:
            DataFrame with value signal components
        """
        signals = self.compute_signals_batch(spot_data, fair_value_data)

        rows = []
        for ticker, sig in signals.items():
            rows.append({
                'ticker': sig.ticker,
                'spot': sig.spot,
                'fair_value': sig.fair_value,
                'valuation_gap': sig.valuation_gap,
                'valuation_zscore': sig.valuation_zscore,
                'value_tilt': sig.value_tilt,
                'signal': sig.signal,
                'regime': sig.regime,
                'mean_reversion_target': sig.mean_reversion_target,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('valuation_gap')

        return df

    def compute_value_timeseries(self,
                                  spot: pd.Series,
                                  fair_value: pd.Series) -> pd.DataFrame:
        """
        Compute full value timeseries for analysis.

        Args:
            spot: Spot price series
            fair_value: Fair value series

        Returns:
            DataFrame with value components over time
        """
        df = pd.DataFrame(index=spot.index)
        df['spot'] = spot
        df['fair_value'] = fair_value

        # Valuation gap
        df['gap'] = compute_valuation_gap(spot, fair_value)
        df['gap_log'] = compute_valuation_gap_log(spot, fair_value)

        # Z-scored gap
        df['gap_zscore'] = self._zscore_gap(df['gap'])

        # Value tilt
        df['value_tilt'] = compute_value_tilt(df['gap'], self.gap_thresholds)
        df['value_tilt_continuous'] = compute_value_tilt_continuous(
            df['gap'], half_life=15, max_tilt=1.0
        )

        return df
