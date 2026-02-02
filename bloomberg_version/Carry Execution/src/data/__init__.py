"""
Data Layer Package

Provides data loaders for Bloomberg and free data sources.
Use get_data_loader(config) to get the appropriate loader based on configuration.
"""

from .free_data_loader import FreeDataLoader, get_data_loader

# Bloomberg loader is optional (requires Bloomberg Terminal)
try:
    from .bloomberg_loader import (
        BloombergDataLoader,
        BloombergConnection,
        FXSpotLoader,
        FXForwardLoader,
        SwapCurveLoader,
        MetalsLoader,
        ImpliedVolLoader,
        MacroDataLoader,
        RiskIndicatorLoader,
    )
    BLOOMBERG_AVAILABLE = True
except ImportError:
    BLOOMBERG_AVAILABLE = False

__all__ = [
    'get_data_loader',
    'FreeDataLoader',
    'BLOOMBERG_AVAILABLE',
]

if BLOOMBERG_AVAILABLE:
    __all__.extend([
        'BloombergDataLoader',
        'BloombergConnection',
        'FXSpotLoader',
        'FXForwardLoader',
        'SwapCurveLoader',
        'MetalsLoader',
        'ImpliedVolLoader',
        'MacroDataLoader',
        'RiskIndicatorLoader',
    ])
