"""
Signal Engines Package

Provides signal generation for:
- Momentum (time-series momentum, MA crossover)
- Carry (forward-implied, rate differential)
- Value (PPP fair value, valuation gaps)
- Rates (curve level, slope, curvature)
- Regime (VIX, FX vol, drawdown filters)
"""

from .momentum import (
    MomentumSignalEngine,
    MomentumSignal,
    compute_tsmom,
    compute_tsmom_scaled,
    compute_tsmom_ensemble,
    compute_ma_crossover,
    compute_ma_trend_strength,
    zscore_signal,
)

from .carry import (
    CarrySignalEngine,
    CarrySignal,
    compute_carry_from_forwards,
    compute_carry_from_forward_points,
    compute_carry_from_rates,
    compute_carry_momentum,
    compute_carry_to_risk,
    zscore_carry,
    rank_carries_cross_sectional,
)

from .value import (
    ValueSignalEngine,
    ValueSignal,
    compute_ppp_fair_value,
    compute_ppp_fair_value_simple,
    compute_valuation_gap,
    compute_valuation_gap_log,
    compute_value_tilt,
    compute_value_tilt_continuous,
    compute_mean_reversion_speed,
    compute_mean_reversion_target,
)

from .rates import (
    RatesSignalEngine,
    RatesSignal,
    compute_curve_level,
    compute_curve_slope,
    compute_curve_curvature,
    compute_curve_momentum,
    compute_curve_roll,
    classify_rate_regime,
    classify_curve_shape,
)

from .regime import (
    RegimeSignalEngine,
    RegimeState,
    compute_vix_regime_multiplier,
    compute_fxvol_regime_multiplier,
    compute_drawdown_regime_multiplier,
    compute_drawdown,
    compute_drawdown_recovery,
    compute_correlation_regime,
    combine_regime_filters,
    combine_regime_filters_weighted,
)

__all__ = [
    # Momentum
    'MomentumSignalEngine',
    'MomentumSignal',
    'compute_tsmom',
    'compute_tsmom_scaled',
    'compute_tsmom_ensemble',
    'compute_ma_crossover',
    'compute_ma_trend_strength',
    'zscore_signal',

    # Carry
    'CarrySignalEngine',
    'CarrySignal',
    'compute_carry_from_forwards',
    'compute_carry_from_forward_points',
    'compute_carry_from_rates',
    'compute_carry_momentum',
    'compute_carry_to_risk',
    'zscore_carry',
    'rank_carries_cross_sectional',

    # Value
    'ValueSignalEngine',
    'ValueSignal',
    'compute_ppp_fair_value',
    'compute_ppp_fair_value_simple',
    'compute_valuation_gap',
    'compute_valuation_gap_log',
    'compute_value_tilt',
    'compute_value_tilt_continuous',
    'compute_mean_reversion_speed',
    'compute_mean_reversion_target',

    # Rates
    'RatesSignalEngine',
    'RatesSignal',
    'compute_curve_level',
    'compute_curve_slope',
    'compute_curve_curvature',
    'compute_curve_momentum',
    'compute_curve_roll',
    'classify_rate_regime',
    'classify_curve_shape',

    # Regime
    'RegimeSignalEngine',
    'RegimeState',
    'compute_vix_regime_multiplier',
    'compute_fxvol_regime_multiplier',
    'compute_drawdown_regime_multiplier',
    'compute_drawdown',
    'compute_drawdown_recovery',
    'compute_correlation_regime',
    'combine_regime_filters',
    'combine_regime_filters_weighted',
]
