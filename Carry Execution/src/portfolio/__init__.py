"""
Portfolio Package

Portfolio construction and rates overlay for FX/Metals carry strategies.
"""

from .construction import (
    PortfolioConstructor,
    PortfolioWeights,
    PortfolioConfig,
    TradeOrder,
)

from .rates_hedge import (
    RatesOverlayEngine,
    FXRiskAnalyzer,
    SOFRHedgeOptimizer,
    FXPosition,
    DV01Exposure,
    HedgeTrade,
    HedgeResult,
    SOFRCurve,
)

__all__ = [
    # Construction
    'PortfolioConstructor',
    'PortfolioWeights',
    'PortfolioConfig',
    'TradeOrder',
    # Rates Hedge
    'RatesOverlayEngine',
    'FXRiskAnalyzer',
    'SOFRHedgeOptimizer',
    'FXPosition',
    'DV01Exposure',
    'HedgeTrade',
    'HedgeResult',
    'SOFRCurve',
]
