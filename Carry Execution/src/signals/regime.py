"""
Regime Filter Engine

Risk regime detection and position scaling multipliers.
Implements VIX, FX volatility, and drawdown-based regime filters.
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
class RegimeState:
    """Container for regime state output."""
    vix_regime: str              # 'low_vol', 'normal', 'elevated', 'high', 'extreme'
    vix_multiplier: float        # Position scaling [0.2, 1.2]
    fxvol_regime: str            # FX vol regime
    fxvol_multiplier: float      # FX vol position scaling
    drawdown_regime: str         # 'normal', 'caution', 'warning', 'stop'
    drawdown_multiplier: float   # Drawdown position scaling
    combined_multiplier: float   # Final combined multiplier
    risk_score: float            # Overall risk score [0, 10]
    recommendation: str          # 'full_risk', 'reduce', 'defensive', 'minimal'
    timestamp: pd.Timestamp


# =============================================================================
# VIX REGIME FUNCTIONS
# =============================================================================

def compute_vix_regime_multiplier(vix: pd.Series,
                                   config: Optional[Dict] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Compute VIX regime and position multiplier.

    Higher VIX = lower position sizes to manage risk.

    Args:
        vix: VIX series
        config: Configuration with thresholds

    Returns:
        Tuple of (regime_series, multiplier_series)
    """
    config = config or {}

    # Default thresholds
    thresholds = config.get('thresholds', {
        'low': 12,
        'normal': 20,
        'elevated': 25,
        'high': 30,
        'extreme': 40,
    })

    # Default scaling multipliers
    scaling = config.get('scaling', {
        'low_vol': 1.20,
        'normal': 1.00,
        'elevated': 0.70,
        'high': 0.40,
        'extreme': 0.20,
    })

    # Use percentile method if enabled
    use_percentile = config.get('percentile', {}).get('enabled', False)

    if use_percentile:
        percentile_thresholds = config.get('percentile', {})
        pct = vix.rolling(252, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )

        regime = pd.Series(index=vix.index, dtype=str)
        multiplier = pd.Series(index=vix.index, dtype=float)

        regime[pct <= percentile_thresholds.get('low', 20)] = 'low_vol'
        regime[(pct > 20) & (pct <= 60)] = 'normal'
        regime[(pct > 60) & (pct <= 80)] = 'elevated'
        regime[(pct > 80) & (pct <= 95)] = 'high'
        regime[pct > 95] = 'extreme'

    else:
        # Absolute threshold method
        regime = pd.Series('normal', index=vix.index)
        regime[vix <= thresholds['low']] = 'low_vol'
        regime[(vix > thresholds['low']) & (vix <= thresholds['normal'])] = 'normal'
        regime[(vix > thresholds['normal']) & (vix <= thresholds['elevated'])] = 'elevated'
        regime[(vix > thresholds['elevated']) & (vix <= thresholds['high'])] = 'high'
        regime[vix > thresholds['high']] = 'extreme'

    # Map regime to multiplier
    multiplier = regime.map(scaling)

    return regime, multiplier


def compute_vix_zscore(vix: pd.Series, window: int = 252) -> pd.Series:
    """
    Compute VIX z-score for regime detection.

    Args:
        vix: VIX series
        window: Rolling window for z-score

    Returns:
        VIX z-score series
    """
    rolling_mean = vix.rolling(window, min_periods=20).mean()
    rolling_std = vix.rolling(window, min_periods=20).std()
    rolling_std = rolling_std.clip(lower=1e-8)

    z = (vix - rolling_mean) / rolling_std

    return z


# =============================================================================
# FX VOLATILITY REGIME FUNCTIONS
# =============================================================================

def compute_fxvol_regime_multiplier(fx_vol: pd.Series,
                                     config: Optional[Dict] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Compute FX volatility regime and position multiplier.

    Args:
        fx_vol: FX volatility index series (e.g., CVIX)
        config: Configuration with thresholds

    Returns:
        Tuple of (regime_series, multiplier_series)
    """
    config = config or {}

    # Default thresholds for FX vol
    thresholds = config.get('thresholds', {
        'low': 7,
        'normal': 10,
        'elevated': 12,
        'high': 15,
        'extreme': 20,
    })

    # Default scaling
    scaling = config.get('scaling', {
        'low_vol': 1.10,
        'normal': 1.00,
        'elevated': 0.80,
        'high': 0.50,
        'extreme': 0.25,
    })

    # Determine regime
    regime = pd.Series('normal', index=fx_vol.index)
    regime[fx_vol <= thresholds['low']] = 'low_vol'
    regime[(fx_vol > thresholds['low']) & (fx_vol <= thresholds['normal'])] = 'normal'
    regime[(fx_vol > thresholds['normal']) & (fx_vol <= thresholds['elevated'])] = 'elevated'
    regime[(fx_vol > thresholds['elevated']) & (fx_vol <= thresholds['high'])] = 'high'
    regime[fx_vol > thresholds['high']] = 'extreme'

    # Map to multiplier
    multiplier = regime.map(scaling)

    return regime, multiplier


# =============================================================================
# DRAWDOWN REGIME FUNCTIONS
# =============================================================================

def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    """
    Compute drawdown from equity curve.

    Args:
        equity_curve: Equity/NAV series

    Returns:
        Drawdown series (negative values)
    """
    # Rolling maximum (high water mark)
    hwm = equity_curve.expanding().max()

    # Drawdown
    drawdown = (equity_curve - hwm) / hwm

    return drawdown


def compute_drawdown_regime_multiplier(equity_curve: pd.Series,
                                        config: Optional[Dict] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Compute drawdown regime and position multiplier.

    Reduces positions as drawdown increases to preserve capital.

    Args:
        equity_curve: Equity/NAV series
        config: Configuration with thresholds

    Returns:
        Tuple of (regime_series, multiplier_series)
    """
    config = config or {}

    # Default thresholds (negative values)
    thresholds = config.get('thresholds', {
        'normal': -0.03,      # -3%
        'caution': -0.05,     # -5%
        'warning': -0.08,     # -8%
        'stop': -0.10,        # -10%
    })

    # Default scaling
    scaling = config.get('scaling', {
        'normal': 1.00,
        'caution': 0.75,
        'warning': 0.50,
        'stop': 0.25,
    })

    # Compute drawdown
    drawdown = compute_drawdown(equity_curve)

    # Determine regime
    regime = pd.Series('normal', index=drawdown.index)
    regime[drawdown <= thresholds['stop']] = 'stop'
    regime[(drawdown > thresholds['stop']) & (drawdown <= thresholds['warning'])] = 'warning'
    regime[(drawdown > thresholds['warning']) & (drawdown <= thresholds['caution'])] = 'caution'
    regime[drawdown > thresholds['caution']] = 'normal'

    # Map to multiplier
    multiplier = regime.map(scaling)

    return regime, multiplier


def compute_drawdown_recovery(equity_curve: pd.Series,
                               lookback: int = 252) -> pd.Series:
    """
    Compute drawdown recovery percentage.

    Used to gradually increase exposure after drawdown recovery.

    Args:
        equity_curve: Equity/NAV series
        lookback: Lookback for max drawdown

    Returns:
        Recovery percentage (0 = at max drawdown, 1 = fully recovered)
    """
    drawdown = compute_drawdown(equity_curve)

    # Rolling max drawdown
    max_dd = drawdown.rolling(lookback, min_periods=20).min()

    # Recovery = how much of the max drawdown has been recovered
    # 0 when at max drawdown, 1 when fully recovered
    recovery = np.where(
        max_dd == 0,
        1.0,
        1.0 - (drawdown / max_dd)
    )

    return pd.Series(recovery, index=equity_curve.index).clip(0, 1)


# =============================================================================
# CORRELATION REGIME FUNCTIONS
# =============================================================================

def compute_correlation_regime(returns_df: pd.DataFrame,
                                lookback: int = 63,
                                threshold: float = 0.8) -> Tuple[float, float]:
    """
    Compute cross-asset correlation regime.

    High correlations = risk-off, reduce diversification benefit.

    Args:
        returns_df: DataFrame of asset returns
        lookback: Correlation lookback
        threshold: High correlation warning threshold

    Returns:
        Tuple of (average_correlation, multiplier)
    """
    if len(returns_df) < lookback:
        return 0.0, 1.0

    # Rolling correlation matrix
    recent_returns = returns_df.iloc[-lookback:]
    corr_matrix = recent_returns.corr()

    # Average pairwise correlation (excluding diagonal)
    n = len(corr_matrix)
    if n < 2:
        return 0.0, 1.0

    total_corr = corr_matrix.values.sum() - n  # Exclude diagonal (1s)
    avg_corr = total_corr / (n * (n - 1))

    # Multiplier: reduce when correlations spike
    if avg_corr > threshold:
        multiplier = 0.5 + 0.5 * (1 - (avg_corr - threshold) / (1 - threshold))
    else:
        multiplier = 1.0

    return avg_corr, multiplier


# =============================================================================
# COMBINED REGIME FILTER
# =============================================================================

def combine_regime_filters(vix_mult: float,
                            fxvol_mult: float,
                            dd_mult: float,
                            floor: float = 0.15) -> float:
    """
    Combine multiple regime multipliers.

    Uses the most restrictive (lowest) multiplier but with a floor.

    Args:
        vix_mult: VIX regime multiplier
        fxvol_mult: FX vol regime multiplier
        dd_mult: Drawdown regime multiplier
        floor: Minimum combined multiplier

    Returns:
        Combined position multiplier
    """
    # Take minimum of all multipliers
    combined = min(vix_mult, fxvol_mult, dd_mult)

    # Apply floor
    combined = max(combined, floor)

    return combined


def combine_regime_filters_weighted(vix_mult: float,
                                     fxvol_mult: float,
                                     dd_mult: float,
                                     weights: Dict[str, float] = None,
                                     floor: float = 0.15) -> float:
    """
    Combine regime multipliers with weights.

    Args:
        vix_mult: VIX regime multiplier
        fxvol_mult: FX vol regime multiplier
        dd_mult: Drawdown regime multiplier
        weights: Weights for each component
        floor: Minimum combined multiplier

    Returns:
        Weighted combined multiplier
    """
    weights = weights or {
        'vix': 0.40,
        'fxvol': 0.30,
        'drawdown': 0.30,
    }

    combined = (
        vix_mult * weights['vix'] +
        fxvol_mult * weights['fxvol'] +
        dd_mult * weights['drawdown']
    )

    combined = max(combined, floor)

    return combined


# =============================================================================
# REGIME SIGNAL ENGINE CLASS
# =============================================================================

class RegimeSignalEngine:
    """
    Engine for regime detection and position scaling.

    Combines VIX, FX vol, and drawdown filters.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize regime signal engine.

        Args:
            config: Configuration dictionary with regime parameters
        """
        self.config = config or {}

        # VIX config
        self.vix_config = self.config.get('vix', {
            'thresholds': {'low': 12, 'normal': 20, 'elevated': 25, 'high': 30, 'extreme': 40},
            'scaling': {'low_vol': 1.20, 'normal': 1.00, 'elevated': 0.70, 'high': 0.40, 'extreme': 0.20},
        })

        # FX vol config
        self.fxvol_config = self.config.get('fx_vol', {
            'thresholds': {'low': 7, 'normal': 10, 'elevated': 12, 'high': 15, 'extreme': 20},
            'scaling': {'low_vol': 1.10, 'normal': 1.00, 'elevated': 0.80, 'high': 0.50, 'extreme': 0.25},
        })

        # Drawdown config
        self.dd_config = self.config.get('drawdown', {
            'thresholds': {'normal': -0.03, 'caution': -0.05, 'warning': -0.08, 'stop': -0.10},
            'scaling': {'normal': 1.00, 'caution': 0.75, 'warning': 0.50, 'stop': 0.25},
        })

        # Combination config
        self.floor = self.config.get('floor', 0.15)
        self.combination_method = self.config.get('combination_method', 'minimum')

    def compute_regime_state(self,
                              vix: pd.Series = None,
                              fx_vol: pd.Series = None,
                              equity_curve: pd.Series = None) -> RegimeState:
        """
        Compute comprehensive regime state.

        Args:
            vix: VIX series
            fx_vol: FX volatility series
            equity_curve: Strategy equity curve

        Returns:
            RegimeState object
        """
        # Initialize defaults
        vix_regime = 'normal'
        vix_mult = 1.0
        fxvol_regime = 'normal'
        fxvol_mult = 1.0
        dd_regime = 'normal'
        dd_mult = 1.0

        # VIX regime
        if vix is not None and len(vix) > 0:
            regime_series, mult_series = compute_vix_regime_multiplier(vix, self.vix_config)
            vix_regime = regime_series.iloc[-1] if len(regime_series) > 0 else 'normal'
            vix_mult = mult_series.iloc[-1] if len(mult_series) > 0 else 1.0

        # FX vol regime
        if fx_vol is not None and len(fx_vol) > 0:
            regime_series, mult_series = compute_fxvol_regime_multiplier(fx_vol, self.fxvol_config)
            fxvol_regime = regime_series.iloc[-1] if len(regime_series) > 0 else 'normal'
            fxvol_mult = mult_series.iloc[-1] if len(mult_series) > 0 else 1.0

        # Drawdown regime
        if equity_curve is not None and len(equity_curve) > 0:
            regime_series, mult_series = compute_drawdown_regime_multiplier(
                equity_curve, self.dd_config
            )
            dd_regime = regime_series.iloc[-1] if len(regime_series) > 0 else 'normal'
            dd_mult = mult_series.iloc[-1] if len(mult_series) > 0 else 1.0

        # Handle NaN values
        vix_mult = 1.0 if pd.isna(vix_mult) else vix_mult
        fxvol_mult = 1.0 if pd.isna(fxvol_mult) else fxvol_mult
        dd_mult = 1.0 if pd.isna(dd_mult) else dd_mult

        # Combine multipliers
        if self.combination_method == 'minimum':
            combined = combine_regime_filters(vix_mult, fxvol_mult, dd_mult, self.floor)
        else:
            combined = combine_regime_filters_weighted(
                vix_mult, fxvol_mult, dd_mult, floor=self.floor
            )

        # Compute risk score (0 = lowest risk, 10 = highest risk)
        risk_score = (1 - combined) * 10

        # Recommendation
        if combined >= 0.9:
            recommendation = 'full_risk'
        elif combined >= 0.6:
            recommendation = 'reduce'
        elif combined >= 0.3:
            recommendation = 'defensive'
        else:
            recommendation = 'minimal'

        # Timestamp
        timestamp = pd.Timestamp.now()
        if vix is not None and len(vix) > 0:
            timestamp = vix.index[-1]

        return RegimeState(
            vix_regime=vix_regime,
            vix_multiplier=vix_mult,
            fxvol_regime=fxvol_regime,
            fxvol_multiplier=fxvol_mult,
            drawdown_regime=dd_regime,
            drawdown_multiplier=dd_mult,
            combined_multiplier=combined,
            risk_score=risk_score,
            recommendation=recommendation,
            timestamp=timestamp
        )

    def compute_regime_timeseries(self,
                                   vix: pd.Series = None,
                                   fx_vol: pd.Series = None,
                                   equity_curve: pd.Series = None) -> pd.DataFrame:
        """
        Compute regime state over time.

        Args:
            vix: VIX series
            fx_vol: FX volatility series
            equity_curve: Strategy equity curve

        Returns:
            DataFrame with regime components over time
        """
        # Use VIX index if available, otherwise fx_vol, otherwise equity
        if vix is not None:
            index = vix.index
        elif fx_vol is not None:
            index = fx_vol.index
        elif equity_curve is not None:
            index = equity_curve.index
        else:
            return pd.DataFrame()

        df = pd.DataFrame(index=index)

        # VIX
        if vix is not None and len(vix) > 0:
            df['vix'] = vix
            regime, mult = compute_vix_regime_multiplier(vix, self.vix_config)
            df['vix_regime'] = regime
            df['vix_multiplier'] = mult
            df['vix_zscore'] = compute_vix_zscore(vix)

        # FX Vol
        if fx_vol is not None and len(fx_vol) > 0:
            df['fx_vol'] = fx_vol
            regime, mult = compute_fxvol_regime_multiplier(fx_vol, self.fxvol_config)
            df['fxvol_regime'] = regime
            df['fxvol_multiplier'] = mult

        # Drawdown
        if equity_curve is not None and len(equity_curve) > 0:
            df['equity'] = equity_curve
            df['drawdown'] = compute_drawdown(equity_curve)
            regime, mult = compute_drawdown_regime_multiplier(equity_curve, self.dd_config)
            df['dd_regime'] = regime
            df['dd_multiplier'] = mult
            df['dd_recovery'] = compute_drawdown_recovery(equity_curve)

        # Combined
        if 'vix_multiplier' in df.columns:
            vix_m = df['vix_multiplier'].fillna(1.0)
        else:
            vix_m = pd.Series(1.0, index=df.index)

        if 'fxvol_multiplier' in df.columns:
            fxvol_m = df['fxvol_multiplier'].fillna(1.0)
        else:
            fxvol_m = pd.Series(1.0, index=df.index)

        if 'dd_multiplier' in df.columns:
            dd_m = df['dd_multiplier'].fillna(1.0)
        else:
            dd_m = pd.Series(1.0, index=df.index)

        df['combined_multiplier'] = pd.concat([vix_m, fxvol_m, dd_m], axis=1).min(axis=1)
        df['combined_multiplier'] = df['combined_multiplier'].clip(lower=self.floor)

        return df

    def get_current_state_summary(self, state: RegimeState) -> str:
        """
        Get human-readable summary of regime state.

        Args:
            state: RegimeState object

        Returns:
            Summary string
        """
        lines = [
            "=" * 50,
            "REGIME STATE SUMMARY",
            "=" * 50,
            f"VIX Regime:      {state.vix_regime:12} (mult: {state.vix_multiplier:.2f})",
            f"FX Vol Regime:   {state.fxvol_regime:12} (mult: {state.fxvol_multiplier:.2f})",
            f"Drawdown Regime: {state.drawdown_regime:12} (mult: {state.drawdown_multiplier:.2f})",
            "-" * 50,
            f"Combined Multiplier: {state.combined_multiplier:.2f}",
            f"Risk Score: {state.risk_score:.1f}/10",
            f"Recommendation: {state.recommendation.upper()}",
            "=" * 50,
        ]
        return "\n".join(lines)
