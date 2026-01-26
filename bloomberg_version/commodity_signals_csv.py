#!/usr/bin/env python3
"""
Commodity Trading Signal Generator (CSV Version)
Generates buy/sell signals using data exported from Bloomberg.

This version reads from CSV files instead of Bloomberg, allowing testing
and optimization on computers without Bloomberg access.

Usage:
    from commodity_signals_csv import CommoditySignalGenerator
    generator = CommoditySignalGenerator(data_dir='./exported_data')
    signals = generator.generate_all_signals()
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Ticker to filename mapping
TICKER_TO_FILE = {
    'GCA Comdty': 'Gold',
    'SIA Comdty': 'Silver',
    'PLA Comdty': 'Platinum',
    'PAA Comdty': 'Palladium',
    'CLA Comdty': 'Oil',
    'DXY Index': 'Dollar_Index',
    'USDJPY Curncy': 'USDJPY',
    'EURUSD Curncy': 'EURUSD',
    'GBPUSD Curncy': 'GBPUSD',
    'AUDUSD Curncy': 'AUDUSD',
    'USDCAD Curncy': 'USDCAD',
    'TUA Comdty': 'Treasury_2Y',
    'FVA Comdty': 'Treasury_5Y',
    'TYA Comdty': 'Treasury_10Y',
    'USA Comdty': 'Treasury_30Y',
    'ESA Index': 'SP500_Futures',
    'NQA Index': 'Nasdaq_Futures',
    'RTYA Index': 'Russell2000_Futures',
}

COMMODITY_UNIVERSE = {
    'GCA Comdty': {'name': 'Gold', 'type': 'precious_metal'},
    'SIA Comdty': {'name': 'Silver', 'type': 'precious_metal'},
    'PLA Comdty': {'name': 'Platinum', 'type': 'precious_metal'},
    'PAA Comdty': {'name': 'Palladium', 'type': 'precious_metal'},
    'CLA Comdty': {'name': 'Oil', 'type': 'energy'},
}

CURRENCY_UNIVERSE = {
    'DXY Index': {'name': 'Dollar Index', 'expected_corr': 'inverse'},
    'USDJPY Curncy': {'name': 'USD/JPY', 'expected_corr': 'risk_proxy'},
    'EURUSD Curncy': {'name': 'EUR/USD', 'expected_corr': 'inverse_dxy'},
    'GBPUSD Curncy': {'name': 'GBP/USD', 'expected_corr': 'inverse_dxy'},
    'AUDUSD Curncy': {'name': 'AUD/USD', 'expected_corr': 'commodity_currency'},
    'USDCAD Curncy': {'name': 'USD/CAD', 'expected_corr': 'oil_inverse'},
}

BOND_UNIVERSE = {
    'TUA Comdty': {'name': '2Y Treasury', 'duration': 2},
    'FVA Comdty': {'name': '5Y Treasury', 'duration': 5},
    'TYA Comdty': {'name': '10Y Treasury', 'duration': 10},
    'USA Comdty': {'name': '30Y Treasury', 'duration': 30},
}

EQUITY_UNIVERSE = {
    'ESA Index': {'name': 'S&P 500 Futures'},
    'NQA Index': {'name': 'Nasdaq 100 Futures'},
    'RTYA Index': {'name': 'Russell 2000 Futures'},
}

# Key correlations for each commodity
KEY_CORRELATIONS = {
    'GCA Comdty': ['DXY Index', 'TYA Comdty', 'USDJPY Curncy', 'SIA Comdty'],
    'SIA Comdty': ['GCA Comdty', 'DXY Index', 'ESA Index'],
    'PLA Comdty': ['GCA Comdty', 'PAA Comdty', 'ESA Index'],
    'PAA Comdty': ['PLA Comdty', 'ESA Index', 'GCA Comdty'],
    'CLA Comdty': ['USDCAD Curncy', 'DXY Index', 'ESA Index'],
}

# Expected correlation relationships
EXPECTED_CORRELATIONS = {
    'GCA Comdty': {
        'DXY Index': -0.5,
        'TYA Comdty': -0.3,
        'USDJPY Curncy': -0.3,
        'SIA Comdty': 0.85,
        'ESA Index': -0.2,
    },
    'SIA Comdty': {
        'GCA Comdty': 0.85,
        'DXY Index': -0.4,
        'ESA Index': 0.1,
    },
    'PLA Comdty': {
        'GCA Comdty': 0.6,
        'PAA Comdty': 0.7,
        'ESA Index': 0.3,
    },
    'PAA Comdty': {
        'PLA Comdty': 0.7,
        'ESA Index': 0.4,
        'GCA Comdty': 0.4,
    },
    'CLA Comdty': {
        'USDCAD Curncy': -0.5,
        'DXY Index': -0.4,
        'ESA Index': 0.3,
    },
}

# Signal weights
SIGNAL_WEIGHTS = {
    'sentiment': 0.40,
    'correlation': 0.25,
    'divergence': 0.20,
    'regime': 0.15,
}

# Signal thresholds
SIGNAL_THRESHOLDS = {
    'very_bullish': 7.0,
    'bullish': 4.0,
    'neutral_high': 0.0,
    'bearish': -4.0,
    'very_bearish': -7.0,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_to_range(values, min_val=-10, max_val=10):
    """Normalize values to range using tanh."""
    if pd.isna(values) if np.isscalar(values) else np.any(pd.isna(values)):
        if np.isscalar(values):
            values = 0
        else:
            values = np.nan_to_num(values, nan=0.0)
    values = np.clip(values, -100, 100)
    return np.tanh(values / 3) * max_val


def safe_get(row: pd.Series, key: str, default: float = 0.0) -> float:
    """Safely get a value from a Series, returning default if missing or NaN."""
    val = row.get(key, default)
    return default if pd.isna(val) else val


def classify_signal(score: float) -> Tuple[str, str]:
    """Classify signal score into category and recommended action."""
    if pd.isna(score):
        return 'N/A', 'No Data'
    elif score >= SIGNAL_THRESHOLDS['very_bullish']:
        return 'Very Bullish', 'Strong Buy / Add to Long'
    elif score >= SIGNAL_THRESHOLDS['bullish']:
        return 'Bullish', 'Buy / Hold Long'
    elif score >= SIGNAL_THRESHOLDS['bearish']:
        return 'Neutral', 'No Position / Reduce'
    elif score >= SIGNAL_THRESHOLDS['very_bearish']:
        return 'Bearish', 'Sell / Hold Short'
    else:
        return 'Very Bearish', 'Strong Sell / Add to Short'


# ============================================================================
# DATA LOADING
# ============================================================================

class CSVDataLoader:
    """Loads data from exported CSV files."""

    def __init__(self, data_dir: str = './exported_data'):
        self.data_dir = data_dir
        self.ohlcv_dir = os.path.join(data_dir, 'ohlcv')
        self.prices_dir = os.path.join(data_dir, 'prices')
        self.cache = {}

    def load_ohlcv(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load OHLCV data for a commodity."""
        if ticker in self.cache:
            return self.cache[ticker]

        name = TICKER_TO_FILE.get(ticker)
        if not name:
            print(f"Unknown ticker: {ticker}")
            return None

        # Try OHLCV directory first (for commodities)
        filepath = os.path.join(self.ohlcv_dir, f"{name}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            self.cache[ticker] = df
            return df

        # Fall back to prices directory
        filepath = os.path.join(self.prices_dir, f"{name}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            self.cache[ticker] = df
            return df

        print(f"File not found for {ticker}: {filepath}")
        return None

    def load_price(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load price data for any asset."""
        return self.load_ohlcv(ticker)

    def load_all_prices(self) -> Dict[str, pd.DataFrame]:
        """Load all available price data."""
        all_data = {}
        for ticker in TICKER_TO_FILE.keys():
            df = self.load_price(ticker)
            if df is not None:
                all_data[ticker] = df
        return all_data


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLCV data."""
    df = df.copy()

    # ROC (Rate of Change)
    for period in [1, 2, 5, 10, 20]:
        df[f'ROC_{period}'] = df['Close'].pct_change(period) * 100

    # Moving Averages
    for period in [5, 10, 20, 50]:
        df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()

    # MA Slopes
    for period in [5, 10, 20]:
        ma_col = f'MA_{period}'
        df[f'MA_{period}_slope'] = df[ma_col].pct_change(5) * 100

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # Volume indicators
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['Vol_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA_20']
        df['Vol_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Vol_Trend'] = df['Vol_MA_5'] / df['Vol_MA_20']
        df['Vol_Price_Corr'] = np.sign(df['ROC_1']) * (df['Vol_Ratio'] - 1)
    else:
        df['Vol_Ratio'] = 1.0
        df['Vol_Trend'] = 1.0
        df['Vol_Price_Corr'] = 0.0

    # TD Sequential
    df = calculate_td_sequential(df)

    return df


def calculate_td_sequential(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Tom DeMark TD Sequential."""
    n = len(df)
    df['TD_Setup_Buy'] = 0
    df['TD_Setup_Sell'] = 0

    for i in range(4, n):
        if df['Close'].iloc[i] < df['Close'].iloc[i-4]:
            prev_count = df['TD_Setup_Buy'].iloc[i-1]
            if prev_count < 9:
                df.loc[df.index[i], 'TD_Setup_Buy'] = prev_count + 1
            else:
                df.loc[df.index[i], 'TD_Setup_Buy'] = 9
        else:
            df.loc[df.index[i], 'TD_Setup_Buy'] = 0

        if df['Close'].iloc[i] > df['Close'].iloc[i-4]:
            prev_count = df['TD_Setup_Sell'].iloc[i-1]
            if prev_count < 9:
                df.loc[df.index[i], 'TD_Setup_Sell'] = prev_count + 1
            else:
                df.loc[df.index[i], 'TD_Setup_Sell'] = 9
        else:
            df.loc[df.index[i], 'TD_Setup_Sell'] = 0

    return df


def get_td_signal_score(df_row: pd.Series) -> float:
    """Convert TD Sequential state to signal contribution."""
    buy_count = df_row.get('TD_Setup_Buy', 0)
    sell_count = df_row.get('TD_Setup_Sell', 0)

    score = 0
    if buy_count == 9:
        score += 7.5
    elif buy_count >= 7:
        score += 3.0

    if sell_count == 9:
        score -= 7.5
    elif sell_count >= 7:
        score -= 3.0

    return score


# ============================================================================
# SENTIMENT CALCULATION
# ============================================================================

def calculate_sentiment_score(df_row: pd.Series) -> float:
    """Calculate sentiment score based on technical indicators."""
    # ROC Signals (35%)
    roc_score = (
        safe_get(df_row, 'ROC_1') * 0.15 +
        safe_get(df_row, 'ROC_2') * 0.15 +
        safe_get(df_row, 'ROC_5') * 0.25 +
        safe_get(df_row, 'ROC_10') * 0.25 +
        safe_get(df_row, 'ROC_20') * 0.20
    )
    roc_normalized = normalize_to_range(roc_score) * 0.35

    # MA Trend Signals (25%)
    ma_slope_score = (
        safe_get(df_row, 'MA_5_slope') * 0.3 +
        safe_get(df_row, 'MA_10_slope') * 0.3 +
        safe_get(df_row, 'MA_20_slope') * 0.4
    )

    ma_cross_score = 0
    ma_5 = safe_get(df_row, 'MA_5')
    ma_10 = safe_get(df_row, 'MA_10')
    ma_20 = safe_get(df_row, 'MA_20')
    if ma_5 > ma_10 > ma_20:
        ma_cross_score = 5
    elif ma_5 < ma_10 < ma_20:
        ma_cross_score = -5

    ma_normalized = (
        normalize_to_range(ma_slope_score) * 0.7 +
        normalize_to_range(ma_cross_score) * 0.3
    ) * 0.25

    # Volume & Conviction (15%)
    vol_score = (
        normalize_to_range((safe_get(df_row, 'Vol_Ratio', 1.0) - 1) * 100) * 0.40 +
        normalize_to_range((safe_get(df_row, 'Vol_Trend', 1.0) - 1) * 100) * 0.30 +
        normalize_to_range(safe_get(df_row, 'Vol_Price_Corr') * 100) * 0.30
    ) * 0.15

    # RSI (10%)
    rsi_val = safe_get(df_row, 'RSI', 50.0)
    rsi_normalized = (rsi_val - 50) / 5
    rsi_score = normalize_to_range(rsi_normalized) * 0.10

    # TD Sequential (15%)
    td_raw_score = get_td_signal_score(df_row)
    td_score = normalize_to_range(td_raw_score) * 0.15

    total_score = roc_normalized + ma_normalized + vol_score + rsi_score + td_score
    return np.clip(total_score, -10, 10)


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def calculate_correlation_alignment_score(
    commodity_ticker: str,
    price_data: Dict[str, pd.DataFrame]
) -> float:
    """Calculate score based on whether correlated assets confirm direction."""
    key_correlations = KEY_CORRELATIONS.get(commodity_ticker, [])

    if not key_correlations:
        return 0.0

    if commodity_ticker not in price_data or price_data[commodity_ticker] is None:
        return 0.0

    commodity_df = price_data[commodity_ticker]
    if len(commodity_df) < 10:
        return 0.0

    commodity_trend = commodity_df['Close'].pct_change(10).iloc[-1] * 100
    commodity_bullish = commodity_trend > 0

    alignment_scores = []

    for comp_ticker in key_correlations:
        if comp_ticker not in price_data or price_data[comp_ticker] is None:
            continue

        comp_df = price_data[comp_ticker]
        if len(comp_df) < 10:
            continue

        comp_trend = comp_df['Close'].pct_change(10).iloc[-1] * 100
        comp_bullish = comp_trend > 0

        expected_corr = EXPECTED_CORRELATIONS.get(commodity_ticker, {}).get(comp_ticker, 0)

        if expected_corr < 0:
            if commodity_bullish != comp_bullish:
                alignment = 1.0
            else:
                alignment = -0.5
        elif expected_corr > 0:
            if commodity_bullish == comp_bullish:
                alignment = 1.0
            else:
                alignment = -0.5
        else:
            alignment = 0.0

        weight = abs(expected_corr)
        alignment_scores.append(alignment * weight)

    if not alignment_scores:
        return 0.0

    avg_alignment = np.mean(alignment_scores)
    return normalize_to_range(avg_alignment * 10)


# ============================================================================
# DIVERGENCE DETECTION
# ============================================================================

def detect_divergence_signal(
    commodity_ticker: str,
    price_data: Dict[str, pd.DataFrame],
    lookback: int = 10
) -> Dict:
    """Detect when commodity diverges from normally correlated assets."""
    key_correlations = KEY_CORRELATIONS.get(commodity_ticker, [])

    if not key_correlations:
        return {'divergence_type': 'none', 'score': 0, 'magnitude': 0}

    if commodity_ticker not in price_data or price_data[commodity_ticker] is None:
        return {'divergence_type': 'none', 'score': 0, 'magnitude': 0}

    commodity_df = price_data[commodity_ticker]
    if len(commodity_df) < lookback:
        return {'divergence_type': 'none', 'score': 0, 'magnitude': 0}

    commodity_return = commodity_df['Close'].pct_change(lookback).iloc[-1] * 100

    divergences = []

    for comp_ticker in key_correlations:
        if comp_ticker not in price_data or price_data[comp_ticker] is None:
            continue

        comp_df = price_data[comp_ticker]
        if len(comp_df) < lookback:
            continue

        comp_return = comp_df['Close'].pct_change(lookback).iloc[-1] * 100
        expected_corr = EXPECTED_CORRELATIONS.get(commodity_ticker, {}).get(comp_ticker, 0)

        if expected_corr == 0:
            continue

        if expected_corr < 0:
            if np.sign(commodity_return) == np.sign(comp_return):
                implied_commodity_return = -comp_return * abs(expected_corr)
                divergence = implied_commodity_return - commodity_return
                divergences.append({'ticker': comp_ticker, 'divergence': divergence, 'implied_return': implied_commodity_return})
        else:
            if np.sign(commodity_return) != np.sign(comp_return):
                implied_commodity_return = comp_return * abs(expected_corr)
                divergence = implied_commodity_return - commodity_return
                divergences.append({'ticker': comp_ticker, 'divergence': divergence, 'implied_return': implied_commodity_return})

    if not divergences:
        return {'divergence_type': 'none', 'score': 0, 'magnitude': 0}

    avg_divergence = np.mean([d['divergence'] for d in divergences])

    if avg_divergence > 2:
        divergence_type = 'bullish'
        score = normalize_to_range(avg_divergence)
    elif avg_divergence < -2:
        divergence_type = 'bearish'
        score = normalize_to_range(avg_divergence)
    else:
        divergence_type = 'none'
        score = 0

    return {'divergence_type': divergence_type, 'score': score, 'magnitude': abs(avg_divergence), 'details': divergences}


# ============================================================================
# MACRO REGIME
# ============================================================================

def detect_macro_quadrant(price_data: Dict[str, pd.DataFrame]) -> Dict:
    """Determine current macro quadrant using Fidenza framework."""
    # Use S&P 500 as growth proxy
    growth_signal = 0
    if 'ESA Index' in price_data and price_data['ESA Index'] is not None:
        spx = price_data['ESA Index']
        if len(spx) >= 60:
            growth_signal = spx['Close'].pct_change(60).iloc[-1] * 100

    # Use commodities as inflation proxy
    inflation_signal = 0
    inflation_tickers = ['CLA Comdty', 'GCA Comdty']
    inflation_signals = []
    for ticker in inflation_tickers:
        if ticker in price_data and price_data[ticker] is not None:
            df = price_data[ticker]
            if len(df) >= 60:
                inflation_signals.append(df['Close'].pct_change(60).iloc[-1] * 100)

    if inflation_signals:
        inflation_signal = np.mean(inflation_signals)

    growth_rising = growth_signal > 0
    inflation_rising = inflation_signal > 0

    if growth_rising and not inflation_rising:
        quadrant = 1
        name = "Goldilocks"
        description = "Growth up, Inflation down - Risk-on, commodities neutral"
    elif growth_rising and inflation_rising:
        quadrant = 2
        name = "Reflation"
        description = "Growth up, Inflation up - Commodities bullish"
    elif not growth_rising and inflation_rising:
        quadrant = 3
        name = "Stagflation"
        description = "Growth down, Inflation up - Gold bullish, Oil mixed"
    else:
        quadrant = 4
        name = "Deflation"
        description = "Growth down, Inflation down - Risk-off, USD bullish"

    return {
        'quadrant': quadrant,
        'name': name,
        'description': description,
        'growth_signal': growth_signal,
        'growth_rising': growth_rising,
        'inflation_signal': inflation_signal,
        'inflation_rising': inflation_rising,
    }


def apply_regime_adjustment(base_signal: float, quadrant: int, commodity_type: str) -> float:
    """Apply regime-based adjustment to signal."""
    adjustments = {
        'precious_metal': {1: 0.9, 2: 1.1, 3: 1.2, 4: 1.0},
        'energy': {1: 1.0, 2: 1.2, 3: 0.8, 4: 0.7},
    }

    multiplier = adjustments.get(commodity_type, {}).get(quadrant, 1.0)
    return base_signal * multiplier


# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

class CommoditySignalGenerator:
    """Generate trading signals from CSV data."""

    def __init__(self, data_dir: str = './exported_data'):
        self.data_loader = CSVDataLoader(data_dir)
        self.price_data_cache = {}

    def _load_all_data(self):
        """Load and prepare all data."""
        if self.price_data_cache:
            return

        raw_data = self.data_loader.load_all_prices()

        for ticker, df in raw_data.items():
            if ticker in COMMODITY_UNIVERSE:
                # Add technical indicators for commodities
                self.price_data_cache[ticker] = add_technical_indicators(df)
            else:
                self.price_data_cache[ticker] = df

    def generate_signal(self, commodity_ticker: str) -> Dict:
        """Generate composite signal for a commodity."""
        self._load_all_data()

        if commodity_ticker not in self.price_data_cache:
            return {
                'signal': 0,
                'classification': 'N/A',
                'action': 'No Data',
                'error': f'No data for {commodity_ticker}'
            }

        commodity_df = self.price_data_cache[commodity_ticker]
        commodity_info = COMMODITY_UNIVERSE.get(commodity_ticker, {})
        commodity_type = commodity_info.get('type', 'precious_metal')

        # Component 1: Sentiment (40%)
        sentiment_score = calculate_sentiment_score(commodity_df.iloc[-1])
        sentiment_weighted = sentiment_score * SIGNAL_WEIGHTS['sentiment']

        # Component 2: Correlation Alignment (25%)
        correlation_score = calculate_correlation_alignment_score(
            commodity_ticker, self.price_data_cache
        )
        correlation_weighted = correlation_score * SIGNAL_WEIGHTS['correlation']

        # Component 3: Divergence Detection (20%)
        divergence_result = detect_divergence_signal(
            commodity_ticker, self.price_data_cache
        )
        divergence_score = divergence_result['score']
        divergence_weighted = divergence_score * SIGNAL_WEIGHTS['divergence']

        # Component 4: Regime Adjustment (15%)
        macro_quadrant = detect_macro_quadrant(self.price_data_cache)
        base_signal = sentiment_weighted + correlation_weighted + divergence_weighted
        adjusted_signal = apply_regime_adjustment(
            base_signal, macro_quadrant['quadrant'], commodity_type
        )
        regime_contribution = adjusted_signal - base_signal

        final_signal = np.clip(adjusted_signal, -10, 10)
        classification, action = classify_signal(final_signal)

        return {
            'signal': round(final_signal, 2),
            'classification': classification,
            'action': action,
            'components': {
                'sentiment': round(sentiment_score, 2),
                'sentiment_weighted': round(sentiment_weighted, 2),
                'correlation': round(correlation_score, 2),
                'correlation_weighted': round(correlation_weighted, 2),
                'divergence': round(divergence_score, 2),
                'divergence_weighted': round(divergence_weighted, 2),
                'regime_adjustment': round(regime_contribution, 2),
            },
            'macro_regime': macro_quadrant,
            'divergence_details': divergence_result,
            'commodity_type': commodity_type,
            'timestamp': datetime.now().isoformat(),
        }

    def generate_all_signals(self) -> Dict[str, Dict]:
        """Generate signals for all commodities."""
        signals = {}
        for ticker in COMMODITY_UNIVERSE.keys():
            signals[ticker] = self.generate_signal(ticker)
        return signals

    def get_price_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get cached price data for a ticker."""
        self._load_all_data()
        return self.price_data_cache.get(ticker)


# ============================================================================
# REPORTING
# ============================================================================

def print_signal_report(signals: Dict[str, Dict]):
    """Print formatted signal report."""
    print("\n" + "="*100)
    print(f"{'COMMODITY TRADING SIGNALS':^100}")
    print(f"{'Sentiment (40%) + Correlation (25%) + Divergence (20%) + Regime (15%)':^100}")
    print("="*100)

    # Get macro environment from first signal
    first_signal = list(signals.values())[0]
    macro = first_signal.get('macro_regime', {})
    print(f"\nMacro Environment: {macro.get('name', 'Unknown')} (Quad {macro.get('quadrant', 0)})")
    print(f"  Growth: {macro.get('growth_signal', 0):+.2f} ({'Rising' if macro.get('growth_rising') else 'Falling'})")
    print(f"  Inflation: {macro.get('inflation_signal', 0):+.2f} ({'Rising' if macro.get('inflation_rising') else 'Falling'})")

    print(f"\n{'Commodity':<15} {'Signal':>8} {'Classification':<15} {'Action':<28} {'Sent':>6} {'Corr':>6} {'Div':>6} {'Reg':>6}")
    print("-"*100)

    sorted_signals = sorted(signals.items(), key=lambda x: x[1].get('signal', 0), reverse=True)

    for ticker, data in sorted_signals:
        name = COMMODITY_UNIVERSE.get(ticker, {}).get('name', ticker)
        signal = data.get('signal', 0)
        classification = data.get('classification', 'N/A')
        action = data.get('action', 'N/A')
        components = data.get('components', {})

        print(f"{name:<15} {signal:>8.2f} {classification:<15} {action:<28} "
              f"{components.get('sentiment', 0):>6.2f} "
              f"{components.get('correlation', 0):>6.2f} "
              f"{components.get('divergence', 0):>6.2f} "
              f"{components.get('regime_adjustment', 0):>6.2f}")

    print("="*100)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Commodity Signal Generator (CSV Version) - Testing")
    print("="*50)

    # Check if data exists
    if not os.path.exists('./exported_data'):
        print("\nERROR: No exported data found!")
        print("Please run export_data_to_csv.py on Bloomberg first.")
        print("Then copy the 'exported_data' folder to this location.")
    else:
        generator = CommoditySignalGenerator(data_dir='./exported_data')
        signals = generator.generate_all_signals()
        print_signal_report(signals)
