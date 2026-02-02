"""
EGM/PEAD Stock Scanner Package

A comprehensive stock scanner based on the Episodic Gap Momentum (EGM) and
Post-Earnings Announcement Drift (PEAD) strategy.

Modules:
    - api_client: API integrations (Finnhub, Alpha Vantage, yfinance)
    - gap_detector: Gap identification and volume analysis
    - earnings_validator: Earnings surprise validation
    - technical_analyzer: ADR%, ATR%, MA cluster detection
    - industry_classifier: Industry classification and relative strength
    - industry_data: Exhaustive industry classification data
    - setup_analyzer: Consolidation pattern analysis
    - signal_generator: Signal generation with weighted scoring
    - egm_scanner: Main scanner orchestrating all components
    - stock_backtest: Backtesting framework
"""

__version__ = '1.0.0'
__author__ = 'LoneStockTrader'

# Import main classes for easy access
from .egm_scanner import EGMScanner
from .gap_detector import GapDetector
from .earnings_validator import EarningsValidator
from .technical_analyzer import TechnicalAnalyzer
from .industry_classifier import IndustryClassifier
from .setup_analyzer import SetupAnalyzer
from .signal_generator import SignalGenerator
from .api_client import APIClient
from .stock_backtest import BacktestEngine, WalkForwardValidator

__all__ = [
    'EGMScanner',
    'GapDetector',
    'EarningsValidator',
    'TechnicalAnalyzer',
    'IndustryClassifier',
    'SetupAnalyzer',
    'SignalGenerator',
    'APIClient',
    'BacktestEngine',
    'WalkForwardValidator',
]
