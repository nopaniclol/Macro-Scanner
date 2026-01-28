#!/usr/bin/env python3
"""
Correlation Analysis Engine for Commodity Trading Signals (CSV Version)
Cross-asset correlation analysis for Gold, Silver, Platinum, Palladium, and Oil

This module calculates:
- Rolling correlations between commodities and other asset classes
- Lead-lag relationships (does USD move before gold?)
- Correlation regime detection (when correlations break down)
- Cross-commodity ratios (gold/silver, gold/oil)

CSV VERSION: Uses local CSV data instead of Bloomberg BQL
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Import from CSV data loader instead of BQL
from csv_data_loader import (
    fetch_price_data,
    fetch_multi_asset_data,
    # Legacy universes
    COMMODITY_UNIVERSE,
    CURRENCY_UNIVERSE,
    BOND_UNIVERSE,
    EQUITY_UNIVERSE,
    # New expanded universes
    PRECIOUS_METALS_UNIVERSE,
    INDUSTRIAL_METALS_UNIVERSE,
    ENERGY_UNIVERSE,
    VOLATILITY_UNIVERSE,
    INFLATION_UNIVERSE,
    CREDIT_UNIVERSE,
    AGRICULTURAL_UNIVERSE,
    EM_UNIVERSE,
    FULL_MACRO_UNIVERSE,
    SPREAD_DEFINITIONS,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default correlation windows
CORRELATION_WINDOWS = [20, 60, 120]  # Short, medium, long-term

# =============================================================================
# EXPECTED CORRELATIONS - EXPANDED (40 Assets)
# =============================================================================
# These are theoretical/historical expected correlations used for:
# 1. Alignment scoring (is market behaving as expected?)
# 2. Divergence detection (correlation breakdown signals)
# 3. Regime classification (risk-on/off detection)

EXPECTED_CORRELATIONS = {
    # =========================================================================
    # GOLD - Safe haven, inverse to real yields
    # =========================================================================
    'GCA Comdty': {
        # Currencies
        'DXY Index': -0.5,        # Inverse - alternative store of value
        'USDJPY Curncy': -0.3,    # Risk-off = gold up, USDJPY down
        'EURUSD Curncy': 0.4,     # EUR strength = gold strength (both anti-USD)
        'AUDUSD Curncy': 0.3,     # Both commodity-sensitive
        # Bonds
        'TYA Comdty': -0.3,       # Inverse to nominal yields
        'TUA Comdty': -0.2,       # Weaker inverse to short end
        # Equities
        'ESA Index': -0.2,        # Mild inverse - safe haven
        'NQA Index': -0.25,       # Slightly more inverse (tech sensitive to rates)
        # Other commodities
        'SIA Comdty': 0.85,       # Highly correlated precious metals
        'PLA Comdty': 0.6,        # PGM correlation
        'PAA Comdty': 0.4,        # Weaker PGM correlation
        'CLA Comdty': 0.1,        # Weak - both dollar-denominated
        # NEW - Tier 1: Critical additions
        'VIX Index': 0.4,         # Risk-off = gold up, VIX up
        'HGA Comdty': 0.3,        # Mild positive (both commodities)
        'H15T10YIE Index': -0.7,  # CRITICAL: Inverse to real yields
        'H15T5YIE Index': -0.65,  # Inverse to real yields
        'USGGBE10 Index': 0.3,    # Positive with inflation expectations
        'USGGBE05 Index': 0.3,    # Positive with inflation expectations
        # NEW - Tier 2: Risk appetite
        'LF98OAS Index': 0.3,     # Credit stress = gold up
        'LUACOAS Index': 0.2,     # IG spread widening = gold up
        'EEM US Equity': -0.2,    # Inverse to EM risk
        'USDZAR Curncy': 0.3,     # ZAR weakness = gold strength (SA producer)
    },

    # =========================================================================
    # SILVER - Hybrid precious/industrial
    # =========================================================================
    'SIA Comdty': {
        'GCA Comdty': 0.85,       # Follows gold
        'DXY Index': -0.4,        # Inverse to dollar
        'ESA Index': 0.1,         # Slightly pro-cyclical (industrial)
        'TYA Comdty': -0.2,       # Mild inverse to yields
        # NEW
        'HGA Comdty': 0.5,        # Industrial demand linkage
        'VIX Index': 0.2,         # Mild positive (safe haven lite)
        'H15T10YIE Index': -0.5,  # Inverse to real yields (weaker than gold)
        'NGA Comdty': 0.1,        # Weak energy correlation
    },

    # =========================================================================
    # PLATINUM - Industrial precious metal
    # =========================================================================
    'PLA Comdty': {
        'GCA Comdty': 0.6,        # Follows gold but more industrial
        'SIA Comdty': 0.7,        # Precious metal correlation
        'PAA Comdty': 0.7,        # PGM correlation
        'ESA Index': 0.3,         # Pro-cyclical (auto catalysts)
        'DXY Index': -0.35,       # Inverse to dollar
        # NEW
        'HGA Comdty': 0.4,        # Industrial correlation
        'VIX Index': -0.1,        # Mild inverse (cyclical)
        'EEM US Equity': 0.3,     # EM demand (China autos)
    },

    # =========================================================================
    # PALLADIUM - Most cyclical precious metal
    # =========================================================================
    'PAA Comdty': {
        'PLA Comdty': 0.7,        # PGM correlation
        'ESA Index': 0.4,         # Most pro-cyclical precious metal
        'GCA Comdty': 0.4,        # Weaker gold correlation
        'DXY Index': -0.3,        # Inverse to dollar
        # NEW
        'HGA Comdty': 0.45,       # Industrial correlation
        'VIX Index': -0.2,        # Inverse (cyclical asset)
        'EEM US Equity': 0.35,    # EM demand
    },

    # =========================================================================
    # OIL - Growth sensitive, dollar denominated
    # =========================================================================
    'CLA Comdty': {
        'USDCAD Curncy': -0.5,    # Canada is major exporter
        'DXY Index': -0.4,        # Dollar-denominated
        'ESA Index': 0.3,         # Growth proxy
        'AUDUSD Curncy': 0.4,     # Commodity currency
        # NEW
        'HGA Comdty': 0.5,        # Both growth-sensitive
        'NGA Comdty': 0.3,        # Energy complex (weak - different drivers)
        'VIX Index': -0.3,        # Risk-off = oil down
        'EEM US Equity': 0.4,     # EM demand proxy
        'USDBRL Curncy': -0.4,    # Brazil oil exporter
        'H15T10YIE Index': 0.2,   # Mild positive with real rates (growth)
        'LF98OAS Index': -0.2,    # Credit stress = oil down
    },

    # =========================================================================
    # COPPER - THE growth/demand proxy
    # =========================================================================
    'HGA Comdty': {
        'ESA Index': 0.5,         # Strong equity correlation
        'AUDUSD Curncy': 0.6,     # Australia copper exporter
        'DXY Index': -0.4,        # Inverse to dollar
        'CLA Comdty': 0.5,        # Both growth-sensitive
        'EEM US Equity': 0.5,     # EM demand (China)
        'VIX Index': -0.35,       # Risk-off = copper down
        'NQA Index': 0.4,         # Tech demand (EVs, electronics)
        'USDBRL Curncy': -0.35,   # Brazil commodity exporter
        'USDMXN Curncy': -0.3,    # Mexico commodity exporter
        'SIA Comdty': 0.5,        # Silver industrial linkage
        'H15T10YIE Index': 0.3,   # Positive with real rates (growth)
    },

    # =========================================================================
    # NATURAL GAS - Idiosyncratic, weather-driven
    # =========================================================================
    'NGA Comdty': {
        'CLA Comdty': 0.3,        # Energy complex (weak)
        'DXY Index': -0.2,        # Weak dollar inverse
        'ESA Index': 0.1,         # Weak equity correlation
        'VIX Index': 0.1,         # Can spike during crises
        'HOA Comdty': 0.5,        # Heating oil correlation
    },

    # =========================================================================
    # VIX - Fear gauge
    # =========================================================================
    'VIX Index': {
        'ESA Index': -0.8,        # Strong inverse to equities
        'NQA Index': -0.75,       # Strong inverse to tech
        'GCA Comdty': 0.4,        # Gold up in risk-off
        'CLA Comdty': -0.3,       # Oil down in risk-off
        'HGA Comdty': -0.35,      # Copper down in risk-off
        'TYA Comdty': 0.3,        # Bonds up in risk-off
        'USDJPY Curncy': -0.5,    # JPY strengthens in risk-off
        'LF98OAS Index': 0.7,     # Credit spreads widen with VIX
        'EEM US Equity': -0.6,    # EM down in risk-off
    },

    # =========================================================================
    # CREDIT SPREADS
    # =========================================================================
    'LF98OAS Index': {  # HY OAS
        'ESA Index': -0.5,        # Spreads widen when equities fall
        'VIX Index': 0.7,         # Spreads widen with volatility
        'GCA Comdty': 0.3,        # Gold up when credit stressed
        'CLA Comdty': -0.2,       # Oil down when credit stressed
        'LUACOAS Index': 0.85,    # IG and HY correlated
    },

    # =========================================================================
    # AGRICULTURAL - Inflation hedges
    # =========================================================================
    'C A Comdty': {  # Corn
        'S A Comdty': 0.7,        # Soybean correlation
        'W A Comdty': 0.6,        # Wheat correlation
        'DXY Index': -0.3,        # Dollar denominated
        'USGGBE10 Index': 0.4,    # Inflation expectations
    },
    'S A Comdty': {  # Soybeans
        'C A Comdty': 0.7,        # Corn correlation
        'USDBRL Curncy': -0.4,    # Brazil exporter
        'USGGBE10 Index': 0.4,    # Inflation expectations
    },
}

# Key correlations for each commodity (used by visualization)
KEY_CORRELATIONS = {
    'GCA Comdty': ['DXY Index', 'TYA Comdty', 'USDJPY Curncy', 'SIA Comdty'],
    'SIA Comdty': ['GCA Comdty', 'DXY Index', 'ESA Index'],
    'PLA Comdty': ['GCA Comdty', 'PAA Comdty', 'ESA Index', 'DXY Index'],
    'PAA Comdty': ['PLA Comdty', 'ESA Index', 'GCA Comdty', 'DXY Index'],
    'CLA Comdty': ['USDCAD Curncy', 'DXY Index', 'ESA Index'],
}


# ============================================================================
# CORRELATION CALCULATIONS
# ============================================================================

class CorrelationEngine:
    """
    Handles all correlation-related calculations between commodities
    and other asset classes (currencies, bonds, equities).

    Key Methods:
    - calculate_rolling_correlations: Rolling correlations at multiple windows
    - detect_lead_lag_relationships: Cross-correlation for lead-lag
    - calculate_correlation_zscore: Current vs historical correlation
    - detect_correlation_regime: Classify correlation state
    - calculate_cross_commodity_ratios: Gold/Silver, Gold/Oil ratios
    """

    def __init__(self, lookback_days: int = 252):
        """
        Initialize correlation engine.

        Args:
            lookback_days: Days of data to fetch for analysis
        """
        self.lookback_days = lookback_days
        self.data_cache = {}
        self.correlation_cache = {}

    def _get_return_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch and cache return data for given tickers.
        """
        cache_key = tuple(sorted(tickers))

        if cache_key not in self.data_cache:
            self.data_cache[cache_key] = fetch_multi_asset_data(
                tickers,
                self.lookback_days
            )

        return self.data_cache[cache_key]

    def calculate_rolling_correlations(
        self,
        commodity_ticker: str,
        comparison_tickers: List[str],
        windows: List[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate rolling correlations between a commodity and multiple assets.
        """
        if windows is None:
            windows = CORRELATION_WINDOWS

        all_tickers = [commodity_ticker] + comparison_tickers
        returns = self._get_return_data(all_tickers)

        if returns is None or len(returns) == 0:
            return {}

        results = {}

        for window in windows:
            correlations = pd.DataFrame(index=returns.index)

            for ticker in comparison_tickers:
                if ticker in returns.columns and commodity_ticker in returns.columns:
                    correlations[ticker] = returns[commodity_ticker].rolling(
                        window=window
                    ).corr(returns[ticker])

            results[window] = correlations.dropna()

        return results

    def detect_lead_lag_relationships(
        self,
        commodity_ticker: str,
        comparison_ticker: str,
        max_lag_days: int = 5
    ) -> Dict:
        """
        Determine if one asset leads another using cross-correlation.
        """
        returns = self._get_return_data([commodity_ticker, comparison_ticker])

        if returns is None or len(returns) < max_lag_days * 2:
            return {'optimal_lag': 0, 'correlation_at_lag': 0, 'significance': 0}

        if commodity_ticker not in returns.columns or comparison_ticker not in returns.columns:
            return {'optimal_lag': 0, 'correlation_at_lag': 0, 'significance': 0}

        commodity_returns = returns[commodity_ticker].values
        comparison_returns = returns[comparison_ticker].values

        # Test different lags
        lag_correlations = {}

        for lag in range(-max_lag_days, max_lag_days + 1):
            try:
                if lag < 0:
                    corr = np.corrcoef(
                        commodity_returns[:lag],
                        comparison_returns[-lag:]
                    )[0, 1]
                elif lag > 0:
                    corr = np.corrcoef(
                        commodity_returns[lag:],
                        comparison_returns[:-lag]
                    )[0, 1]
                else:
                    corr = np.corrcoef(commodity_returns, comparison_returns)[0, 1]

                if not np.isnan(corr):
                    lag_correlations[lag] = corr
            except:
                continue

        if not lag_correlations:
            return {'optimal_lag': 0, 'correlation_at_lag': 0, 'significance': 0}

        # Find optimal lag (highest absolute correlation)
        optimal_lag = max(lag_correlations.keys(),
                         key=lambda x: abs(lag_correlations[x]))
        correlation_at_lag = lag_correlations[optimal_lag]

        # Calculate significance
        n = len(commodity_returns) - abs(optimal_lag)
        if n > 2 and abs(correlation_at_lag) < 1:
            t_stat = correlation_at_lag * np.sqrt(n - 2) / np.sqrt(1 - correlation_at_lag**2)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            significance = 1 - p_value
        else:
            significance = 0

        # Determine direction
        if optimal_lag > 0:
            direction = f"{comparison_ticker} leads by {optimal_lag} days"
        elif optimal_lag < 0:
            direction = f"{commodity_ticker} leads by {-optimal_lag} days"
        else:
            direction = "No significant lead-lag"

        return {
            'optimal_lag': optimal_lag,
            'correlation_at_lag': correlation_at_lag,
            'significance': significance,
            'direction': direction,
            'all_lags': lag_correlations,
        }

    def calculate_correlation_zscore(
        self,
        commodity_ticker: str,
        comparison_ticker: str,
        current_window: int = 20,
        lookback: int = 252
    ) -> Dict:
        """
        Calculate z-score of current correlation vs historical distribution.
        """
        all_tickers = [commodity_ticker, comparison_ticker]
        returns = self._get_return_data(all_tickers)

        if returns is None or len(returns) < lookback:
            return {'zscore': 0, 'current_corr': 0, 'percentile': 50}

        if commodity_ticker not in returns.columns or comparison_ticker not in returns.columns:
            return {'zscore': 0, 'current_corr': 0, 'percentile': 50}

        # Calculate rolling correlations
        rolling_corr = returns[commodity_ticker].rolling(
            window=current_window
        ).corr(returns[comparison_ticker])

        rolling_corr = rolling_corr.dropna()

        if len(rolling_corr) < 2:
            return {'zscore': 0, 'current_corr': 0, 'percentile': 50}

        current_corr = rolling_corr.iloc[-1]
        historical_mean = rolling_corr.iloc[:-1].mean()
        historical_std = rolling_corr.iloc[:-1].std()

        if historical_std > 0:
            zscore = (current_corr - historical_mean) / historical_std
        else:
            zscore = 0

        # Calculate percentile
        percentile = stats.percentileofscore(rolling_corr.iloc[:-1], current_corr)

        # Determine regime signal
        if abs(zscore) > 2:
            regime_signal = 'extreme'
        elif abs(zscore) > 1:
            regime_signal = 'elevated'
        else:
            regime_signal = 'normal'

        return {
            'current_corr': current_corr,
            'zscore': zscore,
            'percentile': percentile,
            'historical_mean': historical_mean,
            'historical_std': historical_std,
            'regime_signal': regime_signal,
        }

    def detect_correlation_regime(
        self,
        commodity_ticker: str,
        comparison_ticker: str,
        threshold_high: float = 0.6,
        threshold_low: float = 0.3,
        window: int = 60
    ) -> Dict:
        """
        Classify current correlation regime.
        """
        all_tickers = [commodity_ticker, comparison_ticker]
        returns = self._get_return_data(all_tickers)

        if returns is None or len(returns) == 0:
            return {'regime': 'unknown', 'confidence': 0}

        if commodity_ticker not in returns.columns or comparison_ticker not in returns.columns:
            return {'regime': 'unknown', 'confidence': 0}

        # Current correlation
        recent_data = returns.iloc[-window:] if len(returns) >= window else returns
        current_corr = recent_data[commodity_ticker].corr(recent_data[comparison_ticker])

        if pd.isna(current_corr):
            return {'regime': 'unknown', 'confidence': 0}

        # Get expected correlation
        expected_corr = EXPECTED_CORRELATIONS.get(
            commodity_ticker, {}
        ).get(comparison_ticker, 0)

        abs_corr = abs(current_corr)

        # Determine regime
        if expected_corr != 0:
            if np.sign(current_corr) != np.sign(expected_corr):
                regime = 'breakdown'
                confidence = abs_corr
            elif abs_corr > threshold_high:
                regime = 'high'
                confidence = abs_corr
            elif abs_corr > threshold_low:
                regime = 'normal'
                confidence = 0.5
            else:
                regime = 'low'
                confidence = 1 - abs_corr
        else:
            if abs_corr > threshold_high:
                regime = 'high'
            elif abs_corr > threshold_low:
                regime = 'normal'
            else:
                regime = 'low'
            confidence = 0.5

        return {
            'regime': regime,
            'current_corr': current_corr,
            'expected_corr': expected_corr,
            'confidence': confidence,
            'deviation_from_expected': current_corr - expected_corr,
        }

    def calculate_cross_commodity_ratios(
        self,
        days: int = 252
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate key cross-commodity ratios for relative value analysis.
        """
        ratio_pairs = [
            ('GCA Comdty', 'SIA Comdty', 'Gold_Silver'),
            ('GCA Comdty', 'CLA Comdty', 'Gold_Oil'),
            ('GCA Comdty', 'PLA Comdty', 'Gold_Platinum'),
            ('PLA Comdty', 'PAA Comdty', 'Platinum_Palladium'),
        ]

        results = {}

        for ticker1, ticker2, ratio_name in ratio_pairs:
            df1 = fetch_price_data(ticker1, days)
            df2 = fetch_price_data(ticker2, days)

            if df1 is None or df2 is None:
                continue

            # Merge on date
            merged = pd.merge(
                df1[['Date', 'Close']].rename(columns={'Close': 'Price1'}),
                df2[['Date', 'Close']].rename(columns={'Close': 'Price2'}),
                on='Date'
            )

            if len(merged) == 0:
                continue

            # Calculate ratio
            merged['Ratio'] = merged['Price1'] / merged['Price2']

            # Calculate z-score of ratio
            merged['Ratio_Mean'] = merged['Ratio'].expanding(min_periods=20).mean()
            merged['Ratio_Std'] = merged['Ratio'].expanding(min_periods=20).std()
            merged['Ratio_Zscore'] = (
                (merged['Ratio'] - merged['Ratio_Mean']) / merged['Ratio_Std']
            )

            # Calculate percentile
            merged['Ratio_Percentile'] = merged['Ratio'].rolling(
                window=min(252, len(merged)), min_periods=20
            ).apply(lambda x: stats.percentileofscore(x[:-1], x[-1]) if len(x) > 1 else 50, raw=True)

            results[ratio_name] = merged

        return results

    def get_correlation_summary(
        self,
        commodity_ticker: str
    ) -> Dict:
        """
        Generate comprehensive correlation summary for a commodity.
        """
        # Get all comparison assets from full macro universe
        all_comparisons = list(CURRENCY_UNIVERSE.keys()) + \
                          list(BOND_UNIVERSE.keys()) + \
                          list(EQUITY_UNIVERSE.keys()) + \
                          list(VOLATILITY_UNIVERSE.keys()) + \
                          list(INFLATION_UNIVERSE.keys()) + \
                          list(CREDIT_UNIVERSE.keys()) + \
                          list(INDUSTRIAL_METALS_UNIVERSE.keys()) + \
                          list(EM_UNIVERSE.keys())

        # Also add other commodities
        other_commodities = [t for t in COMMODITY_UNIVERSE.keys()
                           if t != commodity_ticker]
        all_comparisons.extend(other_commodities)

        # Remove duplicates while preserving order
        all_comparisons = list(dict.fromkeys(all_comparisons))

        summary = {
            'commodity': commodity_ticker,
            'timestamp': datetime.now().isoformat(),
            'correlations': {},
        }

        for comp_ticker in all_comparisons:
            try:
                # Get correlation regime
                regime_info = self.detect_correlation_regime(
                    commodity_ticker, comp_ticker
                )

                # Get z-score
                zscore_info = self.calculate_correlation_zscore(
                    commodity_ticker, comp_ticker
                )

                # Get lead-lag
                lead_lag_info = self.detect_lead_lag_relationships(
                    commodity_ticker, comp_ticker
                )

                summary['correlations'][comp_ticker] = {
                    'current_corr': regime_info.get('current_corr', 0),
                    'expected_corr': regime_info.get('expected_corr', 0),
                    'regime': regime_info.get('regime', 'unknown'),
                    'zscore': zscore_info.get('zscore', 0),
                    'percentile': zscore_info.get('percentile', 50),
                    'lead_lag': lead_lag_info.get('optimal_lag', 0),
                    'lead_lag_direction': lead_lag_info.get('direction', 'Unknown'),
                }

            except Exception as e:
                print(f"Error calculating correlation for {comp_ticker}: {e}")
                continue

        return summary

    # =========================================================================
    # NEW CORRELATION METRICS - Phase 2
    # =========================================================================

    def calculate_real_yield_impact(
        self,
        commodity_ticker: str = 'GCA Comdty',
        tips_ticker: str = 'H15T10YIE Index'
    ) -> Dict:
        """
        Calculate real yield correlation and regime.
        Gold has -0.6 to -0.8 correlation with real yields historically.

        Returns:
            real_yield_regime: 'negative', 'low', 'normal', 'high'
            correlation: current rolling correlation
            signal: bullish/bearish/neutral for the commodity
        """
        try:
            returns = self._get_return_data([commodity_ticker, tips_ticker])

            if returns is None or tips_ticker not in returns.columns:
                return {
                    'real_yield_regime': 'unknown',
                    'correlation': 0,
                    'signal': 'neutral',
                    'data_available': False
                }

            # Get TIPS price data for level analysis
            tips_data = fetch_price_data(tips_ticker, self.lookback_days)
            if tips_data is None or len(tips_data) == 0:
                return {
                    'real_yield_regime': 'unknown',
                    'correlation': 0,
                    'signal': 'neutral',
                    'data_available': False
                }

            # Calculate 60-day rolling correlation
            if commodity_ticker in returns.columns:
                rolling_corr = returns[commodity_ticker].rolling(60).corr(returns[tips_ticker])
                current_corr = rolling_corr.iloc[-1] if len(rolling_corr) > 0 else 0
            else:
                current_corr = 0

            # Determine real yield level regime
            tips_close = tips_data['Close'].iloc[-1]

            if tips_close < 0:
                real_yield_regime = 'negative'
            elif tips_close < 1.0:
                real_yield_regime = 'low'
            elif tips_close < 2.0:
                real_yield_regime = 'normal'
            else:
                real_yield_regime = 'high'

            # Calculate z-score of real yields
            tips_mean = tips_data['Close'].mean()
            tips_std = tips_data['Close'].std()
            tips_zscore = (tips_close - tips_mean) / tips_std if tips_std > 0 else 0

            # Generate signal for commodity (gold specifically)
            # Gold bullish when: real yields falling OR real yields very negative
            if commodity_ticker == 'GCA Comdty':
                if real_yield_regime == 'negative':
                    signal = 'bullish'
                elif real_yield_regime == 'high' and tips_zscore > 1.5:
                    signal = 'bearish'
                elif tips_zscore < -1:  # Real yields falling
                    signal = 'bullish'
                elif tips_zscore > 1:   # Real yields rising
                    signal = 'bearish'
                else:
                    signal = 'neutral'
            else:
                signal = 'neutral'

            return {
                'real_yield_regime': real_yield_regime,
                'real_yield_level': tips_close,
                'real_yield_zscore': tips_zscore,
                'correlation': current_corr if not pd.isna(current_corr) else 0,
                'signal': signal,
                'data_available': True
            }

        except Exception as e:
            return {
                'real_yield_regime': 'unknown',
                'correlation': 0,
                'signal': 'neutral',
                'error': str(e),
                'data_available': False
            }

    def calculate_risk_appetite_score(
        self,
        vix_ticker: str = 'VIX Index',
        hy_oas_ticker: str = 'LF98OAS Index'
    ) -> Dict:
        """
        Composite risk appetite using VIX + Credit spreads.

        Risk-Off: High VIX (>25) + Wide spreads (>500bps)
        Risk-On: Low VIX (<15) + Tight spreads (<350bps)

        Returns:
            risk_appetite: -10 (extreme risk-off) to +10 (extreme risk-on)
            regime: 'risk_on', 'neutral', 'risk_off', 'crisis'
        """
        try:
            result = {
                'risk_appetite': 0,
                'regime': 'neutral',
                'vix_contribution': 0,
                'credit_contribution': 0,
                'data_available': False
            }

            # Get VIX data
            vix_data = fetch_price_data(vix_ticker, self.lookback_days)
            if vix_data is not None and len(vix_data) > 0:
                vix_level = vix_data['Close'].iloc[-1]
                vix_mean = vix_data['Close'].mean()
                vix_std = vix_data['Close'].std()
                vix_zscore = (vix_level - vix_mean) / vix_std if vix_std > 0 else 0

                # VIX contribution: -5 to +5 based on level
                if vix_level < 12:
                    vix_contribution = 5   # Extreme complacency
                elif vix_level < 15:
                    vix_contribution = 3   # Low vol
                elif vix_level < 20:
                    vix_contribution = 1   # Normal
                elif vix_level < 25:
                    vix_contribution = -2  # Elevated
                elif vix_level < 35:
                    vix_contribution = -4  # High fear
                else:
                    vix_contribution = -5  # Panic

                result['vix_level'] = vix_level
                result['vix_zscore'] = vix_zscore
                result['vix_contribution'] = vix_contribution
                result['data_available'] = True

            # Get Credit spread data (optional)
            credit_data = fetch_price_data(hy_oas_ticker, self.lookback_days)
            if credit_data is not None and len(credit_data) > 0:
                credit_level = credit_data['Close'].iloc[-1]
                credit_mean = credit_data['Close'].mean()
                credit_std = credit_data['Close'].std()
                credit_zscore = (credit_level - credit_mean) / credit_std if credit_std > 0 else 0

                # Credit contribution: -5 to +5 based on spread level (in bps)
                if credit_level < 300:
                    credit_contribution = 5   # Very tight
                elif credit_level < 400:
                    credit_contribution = 3   # Tight
                elif credit_level < 500:
                    credit_contribution = 0   # Normal
                elif credit_level < 600:
                    credit_contribution = -2  # Wide
                elif credit_level < 800:
                    credit_contribution = -4  # Very wide
                else:
                    credit_contribution = -5  # Distressed

                result['credit_level'] = credit_level
                result['credit_zscore'] = credit_zscore
                result['credit_contribution'] = credit_contribution

            # Calculate composite score
            if result['data_available']:
                # Weight VIX more if credit data unavailable
                if 'credit_level' in result:
                    risk_appetite = result['vix_contribution'] + result['credit_contribution']
                else:
                    risk_appetite = result['vix_contribution'] * 2

                result['risk_appetite'] = risk_appetite

                # Determine regime
                if risk_appetite >= 6:
                    result['regime'] = 'risk_on'
                elif risk_appetite >= 2:
                    result['regime'] = 'mild_risk_on'
                elif risk_appetite >= -2:
                    result['regime'] = 'neutral'
                elif risk_appetite >= -6:
                    result['regime'] = 'risk_off'
                else:
                    result['regime'] = 'crisis'

            return result

        except Exception as e:
            return {
                'risk_appetite': 0,
                'regime': 'unknown',
                'error': str(e),
                'data_available': False
            }

    def calculate_yield_curve_signal(
        self,
        long_ticker: str = 'TYA Comdty',   # 10Y
        short_ticker: str = 'TUA Comdty'   # 2Y
    ) -> Dict:
        """
        Analyze yield curve shape for recession/growth signals.

        Returns:
            curve_signal: 'steepening', 'flattening', 'inverted', 'normal'
            recession_probability: estimated based on inversion
            commodity_implications: dict of signals for each commodity type
        """
        try:
            # Get bond price data
            long_data = fetch_price_data(long_ticker, self.lookback_days)
            short_data = fetch_price_data(short_ticker, self.lookback_days)

            if long_data is None or short_data is None:
                return {
                    'curve_signal': 'unknown',
                    'recession_probability': 0,
                    'data_available': False
                }

            # Merge data
            merged = pd.merge(
                long_data[['Date', 'Close']].rename(columns={'Close': 'Long'}),
                short_data[['Date', 'Close']].rename(columns={'Close': 'Short'}),
                on='Date'
            )

            if len(merged) < 20:
                return {
                    'curve_signal': 'unknown',
                    'recession_probability': 0,
                    'data_available': False
                }

            # Note: These are futures prices, not yields
            # Bond futures: higher price = lower yield
            # So: Long > Short means 10Y yield < 2Y yield (inverted)
            # We need to invert the logic

            current_spread = merged['Short'].iloc[-1] - merged['Long'].iloc[-1]
            spread_20d_ago = merged['Short'].iloc[-20] - merged['Long'].iloc[-20] if len(merged) >= 20 else current_spread

            # Calculate spread change
            spread_change = current_spread - spread_20d_ago

            # Determine curve signal
            # Positive spread = normal curve (short bond price > long bond price = short yield < long yield)
            if current_spread < 0:  # Long futures > Short futures = inverted
                curve_signal = 'inverted'
                recession_probability = min(0.8, abs(current_spread) / 5)
            elif spread_change > 0.5:
                curve_signal = 'steepening'
                recession_probability = 0.1
            elif spread_change < -0.5:
                curve_signal = 'flattening'
                recession_probability = 0.3
            else:
                curve_signal = 'normal'
                recession_probability = 0.15

            # Commodity implications
            commodity_implications = {
                'precious_metal': {
                    'inverted': 'bullish',      # Flight to safety
                    'flattening': 'neutral',
                    'steepening': 'bearish',    # Growth optimism
                    'normal': 'neutral'
                }.get(curve_signal, 'neutral'),

                'industrial_metal': {
                    'inverted': 'bearish',      # Recession fears
                    'flattening': 'cautious',
                    'steepening': 'bullish',    # Growth optimism
                    'normal': 'neutral'
                }.get(curve_signal, 'neutral'),

                'energy': {
                    'inverted': 'bearish',      # Demand concerns
                    'flattening': 'cautious',
                    'steepening': 'bullish',    # Demand optimism
                    'normal': 'neutral'
                }.get(curve_signal, 'neutral'),
            }

            return {
                'curve_signal': curve_signal,
                'spread_level': current_spread,
                'spread_change_20d': spread_change,
                'recession_probability': recession_probability,
                'commodity_implications': commodity_implications,
                'data_available': True
            }

        except Exception as e:
            return {
                'curve_signal': 'unknown',
                'recession_probability': 0,
                'error': str(e),
                'data_available': False
            }

    def calculate_china_demand_proxy(
        self,
        copper_ticker: str = 'HGA Comdty',
        em_ticker: str = 'EEM US Equity',
        aud_ticker: str = 'AUDUSD Curncy'
    ) -> Dict:
        """
        Composite China/EM demand signal using Copper + EM assets + AUD.

        Copper is the primary China demand proxy.
        Strong copper + Strong AUD + Strong EM = Positive demand outlook

        Returns:
            demand_score: -10 to +10
            trend: 'accelerating', 'decelerating', 'stable'
        """
        try:
            result = {
                'demand_score': 0,
                'trend': 'stable',
                'components': {},
                'data_available': False
            }

            score_components = []

            # Copper component (most important)
            copper_data = fetch_price_data(copper_ticker, self.lookback_days)
            if copper_data is not None and len(copper_data) >= 20:
                copper_close = copper_data['Close']
                copper_current = copper_close.iloc[-1]
                copper_ma20 = copper_close.rolling(20).mean().iloc[-1]
                copper_ma60 = copper_close.rolling(60).mean().iloc[-1] if len(copper_close) >= 60 else copper_ma20

                copper_momentum = (copper_current / copper_ma20 - 1) * 100

                if copper_current > copper_ma20 > copper_ma60:
                    copper_score = 4  # Strong uptrend
                elif copper_current > copper_ma20:
                    copper_score = 2  # Above short-term
                elif copper_current < copper_ma20 < copper_ma60:
                    copper_score = -4  # Strong downtrend
                elif copper_current < copper_ma20:
                    copper_score = -2  # Below short-term
                else:
                    copper_score = 0

                result['components']['copper'] = {
                    'price': copper_current,
                    'momentum': copper_momentum,
                    'score': copper_score
                }
                score_components.append(copper_score)
                result['data_available'] = True

            # AUD component (secondary)
            aud_data = fetch_price_data(aud_ticker, self.lookback_days)
            if aud_data is not None and len(aud_data) >= 20:
                aud_close = aud_data['Close']
                aud_current = aud_close.iloc[-1]
                aud_ma20 = aud_close.rolling(20).mean().iloc[-1]

                aud_momentum = (aud_current / aud_ma20 - 1) * 100

                if aud_current > aud_ma20:
                    aud_score = 2
                else:
                    aud_score = -2

                result['components']['aud'] = {
                    'rate': aud_current,
                    'momentum': aud_momentum,
                    'score': aud_score
                }
                score_components.append(aud_score)

            # EM ETF component (tertiary)
            em_data = fetch_price_data(em_ticker, self.lookback_days)
            if em_data is not None and len(em_data) >= 20:
                em_close = em_data['Close']
                em_current = em_close.iloc[-1]
                em_ma20 = em_close.rolling(20).mean().iloc[-1]

                em_momentum = (em_current / em_ma20 - 1) * 100

                if em_current > em_ma20:
                    em_score = 2
                else:
                    em_score = -2

                result['components']['em'] = {
                    'price': em_current,
                    'momentum': em_momentum,
                    'score': em_score
                }
                score_components.append(em_score)

            # Calculate composite score
            if score_components:
                result['demand_score'] = sum(score_components)

                # Determine trend
                if copper_data is not None and len(copper_data) >= 40:
                    copper_20d_change = copper_data['Close'].iloc[-1] / copper_data['Close'].iloc[-20] - 1
                    copper_40d_change = copper_data['Close'].iloc[-1] / copper_data['Close'].iloc[-40] - 1 if len(copper_data) >= 40 else copper_20d_change

                    if copper_20d_change > copper_40d_change / 2 and copper_20d_change > 0:
                        result['trend'] = 'accelerating'
                    elif copper_20d_change < copper_40d_change / 2 and copper_20d_change < 0:
                        result['trend'] = 'decelerating'
                    else:
                        result['trend'] = 'stable'

            return result

        except Exception as e:
            return {
                'demand_score': 0,
                'trend': 'unknown',
                'error': str(e),
                'data_available': False
            }

    def calculate_inflation_regime(
        self,
        breakeven_5y: str = 'USGGBE05 Index',
        breakeven_10y: str = 'USGGBE10 Index'
    ) -> Dict:
        """
        Inflation regime using breakevens and trend analysis.

        Returns:
            inflation_regime: 'rising', 'falling', 'stable', 'elevated', 'subdued'
            breakeven_zscore: z-score of current vs historical
            commodity_implications: dict of signals
        """
        try:
            result = {
                'inflation_regime': 'unknown',
                'breakeven_level': None,
                'breakeven_zscore': 0,
                'trend': 'stable',
                'data_available': False
            }

            # Try 10Y breakeven first
            be_data = fetch_price_data(breakeven_10y, self.lookback_days)
            if be_data is None or len(be_data) < 20:
                be_data = fetch_price_data(breakeven_5y, self.lookback_days)

            if be_data is None or len(be_data) < 20:
                return result

            be_close = be_data['Close']
            current_be = be_close.iloc[-1]
            be_mean = be_close.mean()
            be_std = be_close.std()
            be_zscore = (current_be - be_mean) / be_std if be_std > 0 else 0

            # Calculate trend
            be_20d_ago = be_close.iloc[-20] if len(be_close) >= 20 else current_be
            be_change = current_be - be_20d_ago

            # Determine regime
            if current_be > 3.0:
                base_regime = 'elevated'
            elif current_be < 1.5:
                base_regime = 'subdued'
            else:
                base_regime = 'normal'

            if be_change > 0.2:
                trend = 'rising'
                inflation_regime = f'{base_regime}_rising' if base_regime != 'normal' else 'rising'
            elif be_change < -0.2:
                trend = 'falling'
                inflation_regime = f'{base_regime}_falling' if base_regime != 'normal' else 'falling'
            else:
                trend = 'stable'
                inflation_regime = base_regime

            # Commodity implications
            commodity_implications = {
                'gold': {
                    'rising': 'bullish',
                    'elevated_rising': 'very_bullish',
                    'falling': 'neutral',
                    'subdued_falling': 'bearish',
                    'elevated': 'bullish',
                    'subdued': 'neutral',
                    'normal': 'neutral'
                }.get(inflation_regime, 'neutral'),

                'tips': {  # TIPS outperform when inflation rises
                    'rising': 'bullish',
                    'elevated_rising': 'very_bullish',
                    'falling': 'bearish',
                }.get(inflation_regime, 'neutral'),

                'agricultural': {
                    'rising': 'bullish',
                    'elevated_rising': 'very_bullish',
                    'elevated': 'bullish',
                }.get(inflation_regime, 'neutral'),
            }

            result.update({
                'inflation_regime': inflation_regime,
                'breakeven_level': current_be,
                'breakeven_zscore': be_zscore,
                'trend': trend,
                'change_20d': be_change,
                'commodity_implications': commodity_implications,
                'data_available': True
            })

            return result

        except Exception as e:
            return {
                'inflation_regime': 'unknown',
                'error': str(e),
                'data_available': False
            }

    def get_enhanced_macro_summary(self) -> Dict:
        """
        Generate comprehensive macro environment summary using all new metrics.
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'real_yield': self.calculate_real_yield_impact(),
            'risk_appetite': self.calculate_risk_appetite_score(),
            'yield_curve': self.calculate_yield_curve_signal(),
            'china_demand': self.calculate_china_demand_proxy(),
            'inflation': self.calculate_inflation_regime(),
        }

        # Calculate composite macro score
        scores = []

        if summary['risk_appetite'].get('data_available'):
            scores.append(summary['risk_appetite']['risk_appetite'])

        if summary['china_demand'].get('data_available'):
            scores.append(summary['china_demand']['demand_score'])

        if scores:
            summary['composite_macro_score'] = sum(scores) / len(scores)
        else:
            summary['composite_macro_score'] = 0

        # Determine overall regime
        risk_regime = summary['risk_appetite'].get('regime', 'neutral')
        inflation_regime = summary['inflation'].get('inflation_regime', 'unknown')
        curve_signal = summary['yield_curve'].get('curve_signal', 'unknown')

        if risk_regime == 'crisis':
            summary['overall_regime'] = 'crisis'
        elif risk_regime == 'risk_off' and curve_signal == 'inverted':
            summary['overall_regime'] = 'recession_risk'
        elif risk_regime == 'risk_on' and inflation_regime in ['rising', 'elevated_rising']:
            summary['overall_regime'] = 'reflation'
        elif risk_regime == 'risk_on':
            summary['overall_regime'] = 'goldilocks'
        elif inflation_regime in ['elevated', 'elevated_rising'] and risk_regime == 'risk_off':
            summary['overall_regime'] = 'stagflation'
        else:
            summary['overall_regime'] = 'neutral'

        return summary


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_gold_correlation_dashboard() -> Dict:
    """Get comprehensive correlation dashboard for Gold."""
    engine = CorrelationEngine(lookback_days=252)
    return engine.get_correlation_summary('GCA Comdty')


def get_oil_correlation_dashboard() -> Dict:
    """Get comprehensive correlation dashboard for Oil."""
    engine = CorrelationEngine(lookback_days=252)
    return engine.get_correlation_summary('CLA Comdty')


def get_all_commodity_ratios() -> Dict:
    """Get all cross-commodity ratios with z-scores."""
    engine = CorrelationEngine(lookback_days=504)
    return engine.calculate_cross_commodity_ratios()


def print_correlation_summary(summary: Dict):
    """Pretty print correlation summary."""
    print("\n" + "="*80)
    print(f"CORRELATION SUMMARY: {summary['commodity']}")
    print(f"Timestamp: {summary['timestamp']}")
    print("="*80)

    print(f"\n{'Asset':<20} {'Corr':>8} {'Expected':>8} {'Regime':<12} {'Z-Score':>8} {'Lead-Lag':>10}")
    print("-"*80)

    for ticker, data in summary['correlations'].items():
        name = ticker.split()[0]  # Short name
        print(f"{name:<20} {data['current_corr']:>8.3f} {data['expected_corr']:>8.3f} "
              f"{data['regime']:<12} {data['zscore']:>8.2f} {data['lead_lag']:>10}")

    print("="*80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("Correlation Analysis Engine (CSV Version) - Testing")
    print("="*50)

    # Test correlation calculations
    engine = CorrelationEngine(lookback_days=252)

    # Get Gold correlation summary
    gold_summary = engine.get_correlation_summary('GCA Comdty')
    print_correlation_summary(gold_summary)

    # Get cross-commodity ratios
    ratios = engine.calculate_cross_commodity_ratios()

    print("\nCross-Commodity Ratios:")
    for ratio_name, df in ratios.items():
        if len(df) > 0:
            latest = df.iloc[-1]
            print(f"  {ratio_name}: {latest['Ratio']:.2f} (Z-Score: {latest['Ratio_Zscore']:.2f})")
