#!/usr/bin/env python3
"""
Correlation Analysis Engine for Commodity Trading Signals
Cross-asset correlation analysis for Gold, Silver, Platinum, Palladium, and Oil

This module calculates:
- Rolling correlations between commodities and other asset classes
- Lead-lag relationships (does USD move before gold?)
- Correlation regime detection (when correlations break down)
- Cross-commodity ratios (gold/silver, gold/oil)
"""

import bql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

# Commodity universe for signal generation
COMMODITY_UNIVERSE = {
    'GCA Comdty': {'name': 'Gold', 'type': 'precious_metal'},
    'SIA Comdty': {'name': 'Silver', 'type': 'precious_metal'},
    'PLA Comdty': {'name': 'Platinum', 'type': 'precious_metal'},
    'PAA Comdty': {'name': 'Palladium', 'type': 'precious_metal'},
    'CLA Comdty': {'name': 'Oil', 'type': 'energy'},
}

# Currency universe for correlation analysis
CURRENCY_UNIVERSE = {
    'DXY Index': {'name': 'Dollar Index', 'expected_corr': 'inverse'},
    'USDJPY Curncy': {'name': 'USD/JPY', 'expected_corr': 'risk_proxy'},
    'EURUSD Curncy': {'name': 'EUR/USD', 'expected_corr': 'inverse_dxy'},
    'GBPUSD Curncy': {'name': 'GBP/USD', 'expected_corr': 'inverse_dxy'},
    'AUDUSD Curncy': {'name': 'AUD/USD', 'expected_corr': 'commodity_currency'},
    'USDCAD Curncy': {'name': 'USD/CAD', 'expected_corr': 'oil_inverse'},
}

# Bond universe for correlation analysis
BOND_UNIVERSE = {
    'TUA Comdty': {'name': '2Y Treasury', 'duration': 2},
    'FVA Comdty': {'name': '5Y Treasury', 'duration': 5},
    'TYA Comdty': {'name': '10Y Treasury', 'duration': 10},
    'USA Comdty': {'name': '30Y Treasury', 'duration': 30},
}

# Equity indices for macro context
EQUITY_UNIVERSE = {
    'ESA Index': {'name': 'S&P 500 Futures'},
    'NQA Index': {'name': 'Nasdaq 100 Futures'},
    'RTYA Index': {'name': 'Russell 2000 Futures'},
}

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

# ============================================================================
# BQL SERVICE INITIALIZATION
# ============================================================================

bq = bql.Service()

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_price_data(ticker: str, days: int = 252) -> pd.DataFrame:
    """
    Fetch historical price data for a single ticker using BQL.

    Args:
        ticker: Bloomberg ticker (e.g., 'GCA Comdty')
        days: Number of historical days to fetch

    Returns:
        DataFrame with Date and Close columns
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        request = bql.Request(
            ticker,
            {
                'Date': bq.data.px_last()['DATE'],
                'Close': bq.data.px_last()['value'],
            },
            with_params={
                'fill': 'prev',  # Forward fill missing values
                'dates': bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            }
        )

        response = bq.execute(request)
        df = pd.concat([data_item.df() for data_item in response], axis=1)
        df = df.sort_values('Date').reset_index(drop=True)
        df['Ticker'] = ticker

        return df

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def fetch_multi_asset_data(
    tickers: List[str],
    days: int = 252
) -> pd.DataFrame:
    """
    Fetch price data for multiple tickers and merge into single DataFrame.

    Args:
        tickers: List of Bloomberg tickers
        days: Number of historical days

    Returns:
        DataFrame with Date index and ticker columns containing returns
    """
    all_data = {}

    for ticker in tickers:
        df = fetch_price_data(ticker, days)
        if df is not None and len(df) > 0:
            # Calculate daily returns
            df['Return'] = df['Close'].pct_change()
            all_data[ticker] = df.set_index('Date')['Return']

    if not all_data:
        return None

    # Merge all series
    merged = pd.DataFrame(all_data)
    merged = merged.dropna()

    return merged


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
        windows: List[int] = CORRELATION_WINDOWS
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate rolling correlations between a commodity and multiple assets.

        WHY THIS WORKS:
        - Gold historically has -0.4 to -0.6 correlation with DXY
        - When this relationship strengthens/weakens, it signals regime changes
        - Rolling windows capture regime shifts in real-time
        - Multiple windows (20/60/120) show short vs long-term dynamics

        Args:
            commodity_ticker: Target commodity (e.g., 'GCA Comdty')
            comparison_tickers: List of assets to correlate against
            windows: Rolling window sizes in days

        Returns:
            Dict mapping window size to DataFrame of correlations
        """
        all_tickers = [commodity_ticker] + comparison_tickers
        returns = self._get_return_data(all_tickers)

        if returns is None:
            return {}

        results = {}

        for window in windows:
            correlations = pd.DataFrame(index=returns.index)

            for ticker in comparison_tickers:
                if ticker in returns.columns:
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

        WHY THIS WORKS:
        - Institutional flows often hit liquid markets (FX) before commodities
        - DXY movements can precede gold moves by 1-2 days
        - Bond market often leads equity and commodity markets
        - Identifying lead-lag gives entry/exit timing edge

        Args:
            commodity_ticker: Target commodity
            comparison_ticker: Asset to test for lead-lag
            max_lag_days: Maximum lag to test (positive = comparison leads)

        Returns:
            Dict with optimal_lag, correlation_at_lag, significance, direction
        """
        returns = self._get_return_data([commodity_ticker, comparison_ticker])

        if returns is None or len(returns) < max_lag_days * 2:
            return {'optimal_lag': 0, 'correlation_at_lag': 0, 'significance': 0}

        commodity_returns = returns[commodity_ticker].values
        comparison_returns = returns[comparison_ticker].values

        # Test different lags
        lag_correlations = {}

        for lag in range(-max_lag_days, max_lag_days + 1):
            if lag < 0:
                # Commodity leads (shift comparison forward)
                corr = np.corrcoef(
                    commodity_returns[:lag],
                    comparison_returns[-lag:]
                )[0, 1]
            elif lag > 0:
                # Comparison leads (shift commodity forward)
                corr = np.corrcoef(
                    commodity_returns[lag:],
                    comparison_returns[:-lag]
                )[0, 1]
            else:
                corr = np.corrcoef(commodity_returns, comparison_returns)[0, 1]

            lag_correlations[lag] = corr

        # Find optimal lag (highest absolute correlation)
        optimal_lag = max(lag_correlations.keys(),
                         key=lambda x: abs(lag_correlations[x]))
        correlation_at_lag = lag_correlations[optimal_lag]

        # Calculate significance using t-test
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

        WHY THIS WORKS:
        - Correlation regimes mean-revert over time
        - Extreme z-scores (>2 or <-2) signal regime stress
        - Breakdown in normal correlation = opportunity or elevated risk
        - Useful for detecting when traditional relationships break

        Args:
            commodity_ticker: Target commodity
            comparison_ticker: Comparison asset
            current_window: Window for current correlation
            lookback: Historical lookback for distribution

        Returns:
            Dict with current_corr, zscore, percentile, regime_signal
        """
        all_tickers = [commodity_ticker, comparison_ticker]
        returns = self._get_return_data(all_tickers)

        if returns is None or len(returns) < lookback:
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

        WHY THIS WORKS:
        - Different trading rules apply in different correlation regimes
        - 'High' regime: Traditional relationships hold, use correlation signals
        - 'Breakdown' regime: Highest uncertainty, reduce position sizes
        - 'Reversal' regime: Correlation flipped sign, major regime shift

        Regimes:
        - 'high': Strong correlation (|corr| > threshold_high)
        - 'normal': Moderate correlation (threshold_low < |corr| < threshold_high)
        - 'low': Weak correlation (|corr| < threshold_low)
        - 'breakdown': Correlation flipped sign vs expectation

        Args:
            commodity_ticker: Target commodity
            comparison_ticker: Comparison asset
            threshold_high: Threshold for 'high' regime
            threshold_low: Threshold for 'low' regime
            window: Rolling window for current correlation

        Returns:
            Dict with regime, current_corr, expected_corr, confidence
        """
        all_tickers = [commodity_ticker, comparison_ticker]
        returns = self._get_return_data(all_tickers)

        if returns is None:
            return {'regime': 'unknown', 'confidence': 0}

        # Current correlation
        current_corr = returns[commodity_ticker].iloc[-window:].corr(
            returns[comparison_ticker].iloc[-window:]
        )

        # Get expected correlation
        expected_corr = EXPECTED_CORRELATIONS.get(
            commodity_ticker, {}
        ).get(comparison_ticker, 0)

        abs_corr = abs(current_corr)

        # Determine regime
        if expected_corr != 0:
            # Check if correlation has flipped sign
            if np.sign(current_corr) != np.sign(expected_corr):
                regime = 'breakdown'
                confidence = abs_corr  # Higher correlation = more confident breakdown
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
            # No expected correlation defined
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

        WHY THIS WORKS:
        - Ratios mean-revert over medium-term (months to years)
        - Gold/Silver ratio historical range ~50-90 (expanded due to silver industrial demand)
        - Extreme ratios signal relative value opportunities
        - Cross-commodity signals reduce single-asset bias

        Ratios calculated:
        - Gold/Silver: Monetary vs industrial precious metal
        - Gold/Oil: Safe haven vs growth (historical mean ~15-20)
        - Gold/Platinum: Investment vs industrial precious metal
        - Platinum/Palladium: Auto catalyst substitution

        Returns:
            Dict with ratio DataFrames including zscore and percentile
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

            # Calculate ratio
            merged['Ratio'] = merged['Price1'] / merged['Price2']

            # Calculate z-score of ratio
            merged['Ratio_Mean'] = merged['Ratio'].expanding(min_periods=60).mean()
            merged['Ratio_Std'] = merged['Ratio'].expanding(min_periods=60).std()
            merged['Ratio_Zscore'] = (
                (merged['Ratio'] - merged['Ratio_Mean']) / merged['Ratio_Std']
            )

            # Calculate percentile
            merged['Ratio_Percentile'] = merged['Ratio'].rolling(
                window=252, min_periods=60
            ).apply(lambda x: stats.percentileofscore(x[:-1], x[-1]) if len(x) > 1 else 50, raw=True)

            results[ratio_name] = merged

        return results

    def get_correlation_summary(
        self,
        commodity_ticker: str
    ) -> Dict:
        """
        Generate comprehensive correlation summary for a commodity.

        Returns current state of all correlation relationships including:
        - Rolling correlations at all windows
        - Lead-lag relationships
        - Z-scores vs historical
        - Regime classifications
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
                    'current_corr': regime_info['current_corr'],
                    'expected_corr': regime_info['expected_corr'],
                    'regime': regime_info['regime'],
                    'zscore': zscore_info['zscore'],
                    'percentile': zscore_info['percentile'],
                    'lead_lag': lead_lag_info['optimal_lag'],
                    'lead_lag_direction': lead_lag_info['direction'],
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
    engine = CorrelationEngine(lookback_days=504)  # 2 years for ratio analysis
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
# VISUALIZATION / CHARTING FUNCTIONS
# ============================================================================

def plot_correlation_heatmap(
    engine: CorrelationEngine,
    commodities: List[str] = None,
    comparison_assets: List[str] = None,
    window: int = 60,
    figsize: Tuple[int, int] = (14, 10),
    save_path: str = None
) -> plt.Figure:
    """
    Plot correlation heatmap between commodities and comparison assets.

    Args:
        engine: CorrelationEngine instance with loaded data
        commodities: List of commodity tickers (default: all)
        comparison_assets: List of comparison asset tickers (default: currencies + bonds)
        window: Rolling window for correlation calculation
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    if commodities is None:
        commodities = list(COMMODITY_UNIVERSE.keys())

    if comparison_assets is None:
        comparison_assets = list(CURRENCY_UNIVERSE.keys()) + list(BOND_UNIVERSE.keys())

    # Calculate correlation matrix
    corr_matrix = []
    commodity_names = []
    asset_names = []

    for commodity in commodities:
        row = []
        commodity_names.append(COMMODITY_UNIVERSE.get(commodity, {}).get('name', commodity))
        for asset in comparison_assets:
            corr_data = engine.calculate_rolling_correlations(
                commodity, [asset], windows=[window]
            )
            if f'{window}d' in corr_data and asset in corr_data[f'{window}d']:
                current_corr = corr_data[f'{window}d'][asset].iloc[-1]
                row.append(current_corr)
            else:
                row.append(0)
        corr_matrix.append(row)

    # Get asset names
    for asset in comparison_assets:
        if asset in CURRENCY_UNIVERSE:
            asset_names.append(CURRENCY_UNIVERSE[asset]['name'])
        elif asset in BOND_UNIVERSE:
            asset_names.append(BOND_UNIVERSE[asset]['name'])
        else:
            asset_names.append(asset)

    # Create DataFrame for heatmap
    corr_df = pd.DataFrame(corr_matrix, index=commodity_names, columns=asset_names)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_df,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation', 'shrink': 0.8},
        ax=ax
    )

    ax.set_title(f'Commodity Correlation Heatmap ({window}-Day Rolling)\n{datetime.now().strftime("%Y-%m-%d")}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Comparison Assets', fontsize=11)
    ax.set_ylabel('Commodities', fontsize=11)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")

    return fig


def plot_rolling_correlation_timeseries(
    engine: CorrelationEngine,
    commodity_ticker: str,
    comparison_assets: List[str] = None,
    windows: List[int] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: str = None
) -> plt.Figure:
    """
    Plot rolling correlation time series for a commodity against multiple assets.

    Args:
        engine: CorrelationEngine instance
        commodity_ticker: Target commodity (e.g., 'GCA Comdty')
        comparison_assets: Assets to compare against
        windows: Correlation windows (default: [20, 60])
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    if comparison_assets is None:
        comparison_assets = KEY_CORRELATIONS.get(commodity_ticker, ['DXY Index', 'TYA Comdty'])[:4]

    if windows is None:
        windows = [20, 60]

    commodity_name = COMMODITY_UNIVERSE.get(commodity_ticker, {}).get('name', commodity_ticker)

    # Calculate correlations
    corr_data = engine.calculate_rolling_correlations(commodity_ticker, comparison_assets, windows)

    # Create figure with subplots
    n_assets = len(comparison_assets)
    fig, axes = plt.subplots(n_assets, 1, figsize=figsize, sharex=True)

    if n_assets == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(windows)))

    for idx, asset in enumerate(comparison_assets):
        ax = axes[idx]

        # Get asset name
        if asset in CURRENCY_UNIVERSE:
            asset_name = CURRENCY_UNIVERSE[asset]['name']
        elif asset in BOND_UNIVERSE:
            asset_name = BOND_UNIVERSE[asset]['name']
        elif asset in COMMODITY_UNIVERSE:
            asset_name = COMMODITY_UNIVERSE[asset]['name']
        else:
            asset_name = asset

        # Plot each window
        for w_idx, window in enumerate(windows):
            window_key = f'{window}d'
            if window_key in corr_data and asset in corr_data[window_key]:
                series = corr_data[window_key][asset]
                ax.plot(series.index, series.values, label=f'{window}d',
                       color=colors[w_idx], linewidth=1.5)

        # Add expected correlation line
        expected = EXPECTED_CORRELATIONS.get(commodity_ticker, {}).get(asset, None)
        if expected is not None:
            ax.axhline(y=expected, color='green', linestyle='--', alpha=0.7,
                      label=f'Expected ({expected:.2f})')

        # Styling
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.fill_between(ax.get_xlim(), -0.3, 0.3, alpha=0.1, color='gray')
        ax.set_ylabel(f'{asset_name}', fontsize=10)
        ax.set_ylim(-1, 1)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date', fontsize=11)
    axes[0].set_title(f'{commodity_name} Rolling Correlations\n{datetime.now().strftime("%Y-%m-%d")}',
                      fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")

    return fig


def plot_commodity_ratios(
    engine: CorrelationEngine,
    days: int = 504,
    figsize: Tuple[int, int] = (14, 12),
    save_path: str = None
) -> plt.Figure:
    """
    Plot cross-commodity ratios with z-score bands.

    Shows:
    - Gold/Silver ratio (50-90 historical range)
    - Gold/Oil ratio
    - Gold/Platinum ratio
    - Platinum/Palladium ratio

    Args:
        engine: CorrelationEngine instance
        days: Days of historical data
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    ratios = engine.calculate_cross_commodity_ratios(days=days)

    if not ratios:
        print("No ratio data available")
        return None

    # Historical ranges for reference
    ratio_ranges = {
        'Gold_Silver': (50, 90, 70),      # (low, high, typical)
        'Gold_Oil': (12, 30, 18),
        'Gold_Platinum': (0.8, 2.5, 1.5),
        'Platinum_Palladium': (0.3, 2.0, 1.0),
    }

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    ratio_names = list(ratios.keys())

    for idx, ratio_name in enumerate(ratio_names[:4]):
        ax = axes[idx]
        df = ratios[ratio_name]

        if len(df) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            continue

        # Convert Date column if needed
        if 'Date' in df.columns:
            dates = pd.to_datetime(df['Date'])
        else:
            dates = df.index

        # Plot ratio
        ax.plot(dates, df['Ratio'], color='navy', linewidth=1.5, label='Ratio')

        # Add moving average
        if 'Ratio_MA' in df.columns:
            ax.plot(dates, df['Ratio_MA'], color='orange', linewidth=1,
                   linestyle='--', label='50-day MA', alpha=0.8)

        # Add historical range bands
        if ratio_name in ratio_ranges:
            low, high, typical = ratio_ranges[ratio_name]
            ax.axhline(y=high, color='red', linestyle=':', alpha=0.7, label=f'High ({high})')
            ax.axhline(y=low, color='green', linestyle=':', alpha=0.7, label=f'Low ({low})')
            ax.fill_between(dates, low, high, alpha=0.1, color='gray')

        # Add z-score on secondary axis
        ax2 = ax.twinx()
        if 'Ratio_Zscore' in df.columns:
            ax2.plot(dates, df['Ratio_Zscore'], color='purple', linewidth=1,
                    alpha=0.5, label='Z-Score')
            ax2.axhline(y=2, color='red', linestyle='--', alpha=0.3)
            ax2.axhline(y=-2, color='green', linestyle='--', alpha=0.3)
            ax2.set_ylabel('Z-Score', fontsize=9, color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
            ax2.set_ylim(-3, 3)

        # Styling
        display_name = ratio_name.replace('_', '/')
        current_ratio = df['Ratio'].iloc[-1]
        current_zscore = df['Ratio_Zscore'].iloc[-1] if 'Ratio_Zscore' in df.columns else 0

        ax.set_title(f'{display_name} Ratio\nCurrent: {current_ratio:.2f} (Z: {current_zscore:+.2f})',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Ratio', fontsize=9)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle(f'Cross-Commodity Ratios Analysis\n{datetime.now().strftime("%Y-%m-%d")}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")

    return fig


def plot_correlation_regime_dashboard(
    engine: CorrelationEngine,
    commodity_ticker: str,
    figsize: Tuple[int, int] = (16, 10),
    save_path: str = None
) -> plt.Figure:
    """
    Plot comprehensive correlation regime dashboard for a commodity.

    Includes:
    - Current correlation status vs expected
    - Correlation regime indicator
    - Lead-lag relationships
    - Z-score deviation chart

    Args:
        engine: CorrelationEngine instance
        commodity_ticker: Target commodity
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    commodity_name = COMMODITY_UNIVERSE.get(commodity_ticker, {}).get('name', commodity_ticker)
    summary = engine.get_correlation_summary(commodity_ticker)

    if not summary or 'correlations' not in summary:
        print(f"No correlation data for {commodity_ticker}")
        return None

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Extract data
    corr_data = summary['correlations']
    assets = list(corr_data.keys())
    current_corrs = [corr_data[a]['current_corr'] for a in assets]
    expected_corrs = [corr_data[a]['expected_corr'] for a in assets]
    zscores = [corr_data[a]['zscore'] for a in assets]
    regimes = [corr_data[a]['regime'] for a in assets]
    lead_lags = [corr_data[a]['lead_lag'] for a in assets]

    # Get short asset names
    asset_names = []
    for a in assets:
        if a in CURRENCY_UNIVERSE:
            asset_names.append(CURRENCY_UNIVERSE[a]['name'])
        elif a in BOND_UNIVERSE:
            asset_names.append(BOND_UNIVERSE[a]['name'])
        elif a in COMMODITY_UNIVERSE:
            asset_names.append(COMMODITY_UNIVERSE[a]['name'])
        else:
            asset_names.append(a.split()[0])

    # 1. Current vs Expected Correlation (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0:2])
    x = np.arange(len(asset_names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, current_corrs, width, label='Current', color='steelblue')
    bars2 = ax1.bar(x + width/2, expected_corrs, width, label='Expected', color='lightcoral', alpha=0.7)

    ax1.set_ylabel('Correlation', fontsize=10)
    ax1.set_title(f'{commodity_name} - Current vs Expected Correlations', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(asset_names, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='upper right')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylim(-1, 1)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Correlation Regime Status (Pie/Donut)
    ax2 = fig.add_subplot(gs[0, 2])
    regime_counts = {'normal': 0, 'high': 0, 'low': 0, 'breakdown': 0}
    for r in regimes:
        if r in regime_counts:
            regime_counts[r] += 1

    colors_regime = {'normal': 'green', 'high': 'blue', 'low': 'orange', 'breakdown': 'red'}
    labels = [k for k, v in regime_counts.items() if v > 0]
    sizes = [regime_counts[k] for k in labels]
    colors_pie = [colors_regime[k] for k in labels]

    if sizes:
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie,
                                           autopct='%1.0f%%', startangle=90,
                                           wedgeprops=dict(width=0.5))
        ax2.set_title('Correlation Regimes', fontsize=12, fontweight='bold')

    # 3. Z-Score Deviation (Horizontal Bar)
    ax3 = fig.add_subplot(gs[1, 0])
    colors_zscore = ['red' if abs(z) > 2 else 'orange' if abs(z) > 1 else 'green' for z in zscores]

    y_pos = np.arange(len(asset_names))
    ax3.barh(y_pos, zscores, color=colors_zscore, alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(asset_names, fontsize=9)
    ax3.axvline(x=0, color='black', linewidth=0.5)
    ax3.axvline(x=2, color='red', linestyle='--', alpha=0.5)
    ax3.axvline(x=-2, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Z-Score', fontsize=10)
    ax3.set_title('Correlation Z-Scores\n(vs Historical)', fontsize=11, fontweight='bold')
    ax3.set_xlim(-3, 3)
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Lead-Lag Relationships (Horizontal Bar)
    ax4 = fig.add_subplot(gs[1, 1])
    colors_lead = ['blue' if l > 0 else 'purple' if l < 0 else 'gray' for l in lead_lags]

    ax4.barh(y_pos, lead_lags, color=colors_lead, alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(asset_names, fontsize=9)
    ax4.axvline(x=0, color='black', linewidth=0.5)
    ax4.set_xlabel('Lead-Lag (Days)', fontsize=10)
    ax4.set_title(f'Lead-Lag vs {commodity_name}\n(+ve = asset leads)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # 5. Summary Table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    # Create summary text
    breakdown_count = sum(1 for r in regimes if r == 'breakdown')
    avg_zscore = np.mean([abs(z) for z in zscores])

    summary_text = f"""
    CORRELATION SUMMARY
    {'='*30}

    Commodity: {commodity_name}
    Date: {summary['timestamp'][:10]}

    Assets Analyzed: {len(assets)}
    Breakdowns: {breakdown_count}
    Avg |Z-Score|: {avg_zscore:.2f}

    Status: {'⚠️ CAUTION' if breakdown_count > 0 else '✓ NORMAL'}

    {'='*30}
    """

    ax5.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Correlation Regime Dashboard: {commodity_name}\n{datetime.now().strftime("%Y-%m-%d %H:%M")}',
                 fontsize=14, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")

    return fig


def plot_all_correlation_charts(
    lookback_days: int = 252,
    save_dir: str = None
) -> Dict[str, plt.Figure]:
    """
    Generate all correlation analysis charts.

    Convenience function that generates:
    - Correlation heatmap
    - Rolling correlations for each commodity
    - Cross-commodity ratios
    - Regime dashboards

    Args:
        lookback_days: Days of historical data
        save_dir: Optional directory to save all charts

    Returns:
        Dict of figure names to Figure objects
    """
    engine = CorrelationEngine(lookback_days=lookback_days)
    figures = {}

    print("Generating correlation analysis charts...")

    # 1. Correlation Heatmap
    print("  - Generating correlation heatmap...")
    save_path = f"{save_dir}/correlation_heatmap.png" if save_dir else None
    figures['heatmap'] = plot_correlation_heatmap(engine, save_path=save_path)

    # 2. Commodity Ratios
    print("  - Generating commodity ratios chart...")
    save_path = f"{save_dir}/commodity_ratios.png" if save_dir else None
    figures['ratios'] = plot_commodity_ratios(engine, save_path=save_path)

    # 3. Rolling correlations for each commodity
    for ticker, info in COMMODITY_UNIVERSE.items():
        name = info['name']
        print(f"  - Generating {name} rolling correlations...")
        save_path = f"{save_dir}/{name.lower()}_correlations.png" if save_dir else None
        figures[f'{name}_rolling'] = plot_rolling_correlation_timeseries(
            engine, ticker, save_path=save_path
        )

    # 4. Regime dashboards for key commodities
    for ticker in ['GCA Comdty', 'CLA Comdty']:  # Gold and Oil
        name = COMMODITY_UNIVERSE[ticker]['name']
        print(f"  - Generating {name} regime dashboard...")
        save_path = f"{save_dir}/{name.lower()}_regime_dashboard.png" if save_dir else None
        figures[f'{name}_regime'] = plot_correlation_regime_dashboard(
            engine, ticker, save_path=save_path
        )

    print(f"Generated {len(figures)} charts")

    return figures


# Key correlations for each commodity (used by visualization)
KEY_CORRELATIONS = {
    'GCA Comdty': ['DXY Index', 'TYA Comdty', 'USDJPY Curncy', 'SIA Comdty'],
    'SIA Comdty': ['GCA Comdty', 'DXY Index', 'HGA Comdty', 'ESA Index'],
    'PLA Comdty': ['GCA Comdty', 'PAA Comdty', 'ESA Index', 'DXY Index'],
    'PAA Comdty': ['PLA Comdty', 'ESA Index', 'GCA Comdty', 'DXY Index'],
    'CLA Comdty': ['USDCAD Curncy', 'DXY Index', 'ESA Index', 'HGA Comdty'],
}


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("Correlation Analysis Engine - Testing")
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
