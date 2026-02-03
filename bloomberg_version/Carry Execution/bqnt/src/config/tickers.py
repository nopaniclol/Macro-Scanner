"""
Bloomberg ticker mappings for BQNT Carry Execution.

Maps internal asset keys to Bloomberg tickers for BQL queries.
"""

# =============================================================================
# FX SPOT TICKERS
# =============================================================================
FX_SPOT_TICKERS = {
    'EURUSD': 'EURUSD Curncy',
    'USDJPY': 'USDJPY Curncy',
    'GBPUSD': 'GBPUSD Curncy',
    'AUDUSD': 'AUDUSD Curncy',
    'USDCNH': 'USDCNH Curncy',
    'USDCAD': 'USDCAD Curncy',
    'NZDUSD': 'NZDUSD Curncy',
    'USDCHF': 'USDCHF Curncy',
}

# =============================================================================
# FX FORWARD TICKERS (3-Month)
# =============================================================================
FX_FORWARD_TICKERS = {
    'EUR3M': 'EUR3M Curncy',
    'JPY3M': 'JPY3M Curncy',
    'GBP3M': 'GBP3M Curncy',
    'AUD3M': 'AUD3M Curncy',
    'CNH3M': 'CNH3M Curncy',
    'CAD3M': 'CAD3M Curncy',
    'NZD3M': 'NZD3M Curncy',
    'CHF3M': 'CHF3M Curncy',
}

# =============================================================================
# PRECIOUS METALS TICKERS
# =============================================================================
METALS_TICKERS = {
    'XAUUSD': 'XAU Curncy',      # Gold
    'XAGUSD': 'XAG Curncy',      # Silver
    'XPTUSD': 'XPT Curncy',      # Platinum
    'XPDUSD': 'XPD Curncy',      # Palladium
}

# =============================================================================
# TREASURY YIELD TICKERS
# =============================================================================
TREASURY_TICKERS = {
    'US2Y': 'USGG2YR Index',     # 2-Year Treasury
    'US5Y': 'USGG5YR Index',     # 5-Year Treasury
    'US10Y': 'USGG10YR Index',   # 10-Year Treasury
    'US30Y': 'USGG30YR Index',   # 30-Year Treasury
}

# =============================================================================
# VOLATILITY & RISK TICKERS
# =============================================================================
VOLATILITY_TICKERS = {
    'VIX': 'VIX Index',              # CBOE VIX
    'FXVOL': 'JPMVXYG7 Index',       # JPM G7 FX Volatility
    'MOVE': 'MOVE Index',            # Bond Volatility
}

# =============================================================================
# EQUITY INDEX TICKERS
# =============================================================================
EQUITY_TICKERS = {
    'SPX': 'SPX Index',              # S&P 500
    'NDX': 'NDX Index',              # Nasdaq 100
    'DXY': 'DXY Curncy',             # Dollar Index
}

# =============================================================================
# COMBINED UNIVERSE
# =============================================================================
ALL_TICKERS = {
    **FX_SPOT_TICKERS,
    **FX_FORWARD_TICKERS,
    **METALS_TICKERS,
    **TREASURY_TICKERS,
    **VOLATILITY_TICKERS,
    **EQUITY_TICKERS,
}

# =============================================================================
# ASSET METADATA
# =============================================================================
ASSET_METADATA = {
    # FX Pairs - Carry eligible
    'EURUSD': {'type': 'fx', 'carry_eligible': True, 'forward': 'EUR3M'},
    'USDJPY': {'type': 'fx', 'carry_eligible': True, 'forward': 'JPY3M'},
    'GBPUSD': {'type': 'fx', 'carry_eligible': True, 'forward': 'GBP3M'},
    'AUDUSD': {'type': 'fx', 'carry_eligible': True, 'forward': 'AUD3M'},
    'USDCNH': {'type': 'fx', 'carry_eligible': True, 'forward': 'CNH3M'},
    'USDCAD': {'type': 'fx', 'carry_eligible': True, 'forward': 'CAD3M'},
    'NZDUSD': {'type': 'fx', 'carry_eligible': True, 'forward': 'NZD3M'},
    'USDCHF': {'type': 'fx', 'carry_eligible': True, 'forward': 'CHF3M'},

    # Precious Metals
    'XAUUSD': {'type': 'metal', 'carry_eligible': False, 'name': 'Gold'},
    'XAGUSD': {'type': 'metal', 'carry_eligible': False, 'name': 'Silver'},
    'XPTUSD': {'type': 'metal', 'carry_eligible': False, 'name': 'Platinum'},
    'XPDUSD': {'type': 'metal', 'carry_eligible': False, 'name': 'Palladium'},

    # Treasuries - For rates signal
    'US2Y': {'type': 'rate', 'tenor': 2},
    'US5Y': {'type': 'rate', 'tenor': 5},
    'US10Y': {'type': 'rate', 'tenor': 10},
    'US30Y': {'type': 'rate', 'tenor': 30},

    # Volatility - For regime detection
    'VIX': {'type': 'volatility', 'regime_indicator': True},
    'FXVOL': {'type': 'volatility', 'regime_indicator': True},
    'MOVE': {'type': 'volatility', 'regime_indicator': False},

    # Equity - For macro regime
    'SPX': {'type': 'equity', 'benchmark': True},
    'NDX': {'type': 'equity', 'benchmark': False},
    'DXY': {'type': 'fx_index', 'benchmark': True},
}

# =============================================================================
# TRADING UNIVERSE (Active positions)
# =============================================================================
TRADING_UNIVERSE = [
    'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCNH',
    'XAUUSD', 'XAGUSD',
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_bloomberg_ticker(internal_key: str) -> str:
    """Convert internal key to Bloomberg ticker."""
    return ALL_TICKERS.get(internal_key, internal_key)


def get_forward_ticker(fx_pair: str) -> str:
    """Get the forward ticker for an FX pair."""
    metadata = ASSET_METADATA.get(fx_pair, {})
    forward_key = metadata.get('forward')
    if forward_key:
        return FX_FORWARD_TICKERS.get(forward_key)
    return None


def get_tradeable_assets() -> list:
    """Get list of Bloomberg tickers for trading universe."""
    return [get_bloomberg_ticker(key) for key in TRADING_UNIVERSE]


def get_carry_eligible_pairs() -> list:
    """Get FX pairs eligible for carry calculation."""
    return [
        key for key, meta in ASSET_METADATA.items()
        if meta.get('carry_eligible', False)
    ]
