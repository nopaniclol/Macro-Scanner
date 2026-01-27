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
    COMMODITY_UNIVERSE,
    CURRENCY_UNIVERSE,
    BOND_UNIVERSE,
    EQUITY_UNIVERSE,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default correlation windows
CORRELATION_WINDOWS = [20, 60, 120]  # Short, medium, long-term

# Expected correlation relationships (for alignment scoring)
EXPECTED_CORRELATIONS = {
    'GCA Comdty': {  # Gold
        'DXY Index': -0.5,        # Inverse - alternative store of value
        'TYA Comdty': -0.3,       # Inverse to real yields
        'USDJPY Curncy': -0.3,    # Risk-off = gold up, USDJPY down
        'SIA Comdty': 0.85,       # Highly correlated precious metals
        'ESA Index': -0.2,        # Mild inverse - safe haven
    },
    'SIA Comdty': {  # Silver
        'GCA Comdty': 0.85,       # Follows gold
        'DXY Index': -0.4,        # Inverse to dollar
        'HGA Comdty': 0.5,        # Industrial demand linkage
        'ESA Index': 0.1,         # Slightly pro-cyclical (industrial)
    },
    'PLA Comdty': {  # Platinum
        'GCA Comdty': 0.6,        # Follows gold but more industrial
        'PAA Comdty': 0.7,        # PGM correlation
        'ESA Index': 0.3,         # Pro-cyclical (auto catalysts)
    },
    'PAA Comdty': {  # Palladium
        'PLA Comdty': 0.7,        # PGM correlation
        'ESA Index': 0.4,         # Most pro-cyclical precious metal
        'GCA Comdty': 0.4,        # Weaker gold correlation
    },
    'CLA Comdty': {  # Oil
        'USDCAD Curncy': -0.5,    # Canada is major exporter
        'HGA Comdty': 0.5,        # Both growth-sensitive
        'DXY Index': -0.4,        # Dollar-denominated
        'ESA Index': 0.3,         # Growth proxy
        'NGA Comdty': 0.3,        # Energy complex correlation
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
        # Get all comparison assets
        all_comparisons = list(CURRENCY_UNIVERSE.keys()) + \
                          list(BOND_UNIVERSE.keys()) + \
                          list(EQUITY_UNIVERSE.keys())

        # Also add other commodities
        other_commodities = [t for t in COMMODITY_UNIVERSE.keys()
                           if t != commodity_ticker]
        all_comparisons.extend(other_commodities)

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
