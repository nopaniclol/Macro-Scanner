"""
Strategy parameters for BQNT Carry Execution.

All configuration stored as Python dicts (no YAML in BQNT).
"""

# =============================================================================
# SIGNAL WEIGHTS
# =============================================================================
SIGNAL_WEIGHTS = {
    'momentum': 0.35,
    'carry': 0.35,
    'value': 0.15,
    'rates': 0.15,
}

# =============================================================================
# MOMENTUM PARAMETERS
# =============================================================================
MOMENTUM_PARAMS = {
    # Lookback periods (days)
    'short_window': 5,
    'medium_window': 21,
    'long_window': 63,

    # Window weights for multi-horizon momentum
    'window_weights': {
        5: 0.20,    # 1 week
        21: 0.40,   # 1 month
        63: 0.40,   # 3 months
    },

    # Smoothing
    'ema_span': 5,

    # Signal bounds
    'max_signal': 1.0,
    'min_signal': -1.0,
}

# =============================================================================
# CARRY PARAMETERS
# =============================================================================
CARRY_PARAMS = {
    # Forward point calculation
    'annualization_factor': 4,  # 3-month forwards, annualize by 4

    # Normalization
    'normalize_by_vol': True,
    'vol_lookback': 21,

    # Signal bounds
    'max_signal': 1.0,
    'min_signal': -1.0,
}

# =============================================================================
# VALUE PARAMETERS (Mean Reversion)
# =============================================================================
VALUE_PARAMS = {
    # Z-score calculation
    'lookback': 252,  # 1 year
    'zscore_threshold': 2.0,

    # Signal scaling
    'scale_factor': 0.5,  # Reduce impact of value signal
}

# =============================================================================
# RATES PARAMETERS
# =============================================================================
RATES_PARAMS = {
    # Yield curve
    'short_tenor': 'US2Y',
    'long_tenor': 'US10Y',

    # Slope signal
    'steepening_bullish': True,  # Steepening curve = bullish risk
    'lookback': 21,
}

# =============================================================================
# RISK / REGIME PARAMETERS
# =============================================================================
REGIME_PARAMS = {
    # VIX thresholds
    'vix_thresholds': {
        'low': 12,
        'normal': 20,
        'elevated': 25,
        'high': 30,
        'extreme': 40,
    },

    # Position multipliers by VIX regime
    'vix_multipliers': {
        'low': 1.20,        # Increase exposure in low vol
        'normal': 1.00,     # Full exposure
        'elevated': 0.70,   # Reduce exposure
        'high': 0.40,       # Significant reduction
        'extreme': 0.20,    # Minimal exposure
    },

    # FX volatility thresholds
    'fxvol_thresholds': {
        'low': 6,
        'normal': 8,
        'elevated': 10,
        'high': 12,
    },
}

# =============================================================================
# PORTFOLIO CONSTRUCTION PARAMETERS
# =============================================================================
PORTFOLIO_PARAMS = {
    # Volatility targeting
    'target_vol': 0.12,  # 12% annualized
    'vol_lookback': 21,
    'vol_floor': 0.05,   # Minimum assumed vol
    'vol_cap': 0.40,     # Maximum assumed vol

    # Position limits
    'max_position': 0.25,   # 25% max per asset
    'min_position': -0.25,  # -25% min per asset
    'max_gross_exposure': 2.0,  # 200% max gross
    'max_net_exposure': 0.50,   # 50% max net

    # Risk parity
    'use_risk_parity': True,
    'risk_parity_lookback': 63,

    # Correlation adjustment
    'correlation_lookback': 63,
    'min_correlation': -0.95,
    'max_correlation': 0.95,
}

# =============================================================================
# EXECUTION PARAMETERS
# =============================================================================
EXECUTION_PARAMS = {
    # Rebalancing
    'rebalance_frequency': 'daily',
    'rebalance_threshold': 0.02,  # 2% drift triggers rebalance

    # Transaction costs (for backtest)
    'fx_cost_bps': 2,      # 2 bps for FX
    'metal_cost_bps': 5,   # 5 bps for metals
}

# =============================================================================
# DATA PARAMETERS
# =============================================================================
DATA_PARAMS = {
    # History requirements
    'min_history_days': 252,  # 1 year minimum
    'default_lookback': 504,  # 2 years default fetch

    # Cache settings
    'cache_ttl_minutes': 30,  # Cache expires after 30 min

    # Fill method
    'fill_method': 'prev',  # Forward fill missing data
}

# =============================================================================
# BACKTEST PARAMETERS
# =============================================================================
BACKTEST_PARAMS = {
    # Date range
    'default_start': '2020-01-01',
    'default_end': None,  # None = today

    # Initial capital
    'initial_capital': 1_000_000,

    # Benchmark
    'benchmark': 'DXY',

    # Performance metrics
    'risk_free_rate': 0.05,  # 5% risk-free rate
    'periods_per_year': 252,
}

# =============================================================================
# REPORTING PARAMETERS
# =============================================================================
REPORTING_PARAMS = {
    # Display
    'decimal_places': 4,
    'pct_decimal_places': 2,

    # Charts
    'chart_width': 12,
    'chart_height': 6,
    'chart_style': 'seaborn',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_vix_regime(vix_level: float) -> str:
    """Determine VIX regime from current level."""
    thresholds = REGIME_PARAMS['vix_thresholds']

    if vix_level < thresholds['low']:
        return 'low'
    elif vix_level < thresholds['normal']:
        return 'normal'
    elif vix_level < thresholds['elevated']:
        return 'elevated'
    elif vix_level < thresholds['high']:
        return 'high'
    else:
        return 'extreme'


def get_regime_multiplier(vix_level: float) -> float:
    """Get position size multiplier based on VIX regime."""
    regime = get_vix_regime(vix_level)
    return REGIME_PARAMS['vix_multipliers'][regime]


def get_signal_weight(signal_type: str) -> float:
    """Get weight for a signal type."""
    return SIGNAL_WEIGHTS.get(signal_type, 0.0)
