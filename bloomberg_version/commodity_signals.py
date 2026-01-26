#!/usr/bin/env python3
"""
Commodity Trading Signal Generator
Generates buy/sell signals for Gold, Silver, Platinum, Palladium, and Oil

Signal Components:
- Sentiment Score (CCS): 40% - Existing momentum/trend analysis
- Correlation Alignment: 25% - Cross-asset confirmation
- Divergence Detection: 20% - Mean-reversion opportunities
- Regime Adjustment: 15% - Macro context

Signal Range: -10 (Very Bearish) to +10 (Very Bullish)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Import from existing modules
from correlation_analysis import (
    CorrelationEngine,
    COMMODITY_UNIVERSE,
    CURRENCY_UNIVERSE,
    BOND_UNIVERSE,
    EQUITY_UNIVERSE,
    EXPECTED_CORRELATIONS,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Signal component weights
SIGNAL_WEIGHTS = {
    'sentiment': 0.40,       # CCS momentum/trend score
    'correlation': 0.25,     # Cross-asset alignment
    'divergence': 0.20,      # Divergence detection
    'regime': 0.15,          # Macro regime adjustment
}

# Signal thresholds
SIGNAL_THRESHOLDS = {
    'very_bullish': 7.0,     # Strong buy
    'bullish': 4.0,          # Buy
    'neutral_high': 0.0,     # No signal
    'bearish': -4.0,         # Sell
    'very_bearish': -7.0,    # Strong sell
}

# Regime adjustments by commodity type and macro quadrant
# Quadrant 1: Goldilocks (growth up, inflation down)
# Quadrant 2: Reflation (growth up, inflation up)
# Quadrant 3: Stagflation (growth down, inflation up)
# Quadrant 4: Risk-Off (growth down, inflation down)
REGIME_ADJUSTMENTS = {
    'precious_metal': {  # Gold, Silver, Platinum, Palladium
        1: -0.20,   # Goldilocks: Risk-on, gold underperforms
        2: 0.00,    # Reflation: Mixed - inflation helps, risk-on hurts
        3: 0.30,    # Stagflation: Gold shines (inflation + uncertainty)
        4: 0.20,    # Risk-Off: Safe-haven demand
    },
    'energy': {  # Oil, Natural Gas
        1: 0.15,    # Goldilocks: Growth supports demand
        2: 0.25,    # Reflation: Strong growth + inflation
        3: 0.00,    # Stagflation: Mixed - inflation up, demand down
        4: -0.30,   # Risk-Off: Demand collapse
    },
}

# Key correlations to monitor for each commodity
KEY_CORRELATIONS = {
    'GCA Comdty': ['DXY Index', 'TYA Comdty', 'USDJPY Curncy', 'SIA Comdty'],
    'SIA Comdty': ['GCA Comdty', 'DXY Index', 'ESA Index'],
    'PLA Comdty': ['GCA Comdty', 'PAA Comdty', 'ESA Index'],
    'PAA Comdty': ['PLA Comdty', 'ESA Index', 'GCA Comdty'],
    'CLA Comdty': ['USDCAD Curncy', 'DXY Index', 'ESA Index', 'NGA Comdty'],
}

# ============================================================================
# BQL SERVICE INITIALIZATION
# ============================================================================

bq = bql.Service()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_to_range(values, min_val=-10, max_val=10):
    """
    Normalize values to range using tanh.
    More aggressive scaling (divide by 3) for stronger signals.
    """
    values = np.clip(values, -100, 100)
    return np.tanh(values / 3) * max_val


def classify_signal(score: float) -> Tuple[str, str]:
    """
    Classify signal score into category and recommended action.

    Returns:
        Tuple of (classification, action)
    """
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
# SENTIMENT SCORE CALCULATION (from existing CCS)
# ============================================================================

def fetch_historical_data(ticker: str, days: int = 90) -> pd.DataFrame:
    """
    Fetch historical OHLCV data using BQL.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        request = bql.Request(
            ticker,
            {
                'Date': bq.data.px_open()['DATE'],
                'Open': bq.data.px_open()['value'],
                'High': bq.data.px_high()['value'],
                'Low': bq.data.px_low()['value'],
                'Close': bq.data.px_last()['value'],
                'Volume': bq.data.px_volume()['value'],
            },
            with_params={
                'fill': 'na',
                'dates': bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            }
        )

        response = bq.execute(request)
        df = pd.concat([data_item.df() for data_item in response], axis=1)
        df = df.sort_values('Date').reset_index(drop=True)

        return df

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for sentiment scoring.
    """
    # ROC - Rate of Change (multiple timeframes)
    df['ROC_1'] = df['Close'].pct_change(1) * 100
    df['ROC_2'] = df['Close'].pct_change(2) * 100
    df['ROC_5'] = df['Close'].pct_change(5) * 100
    df['ROC_10'] = df['Close'].pct_change(10) * 100
    df['ROC_20'] = df['Close'].pct_change(20) * 100

    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # MA Slopes (trend direction)
    df['MA_5_slope'] = df['MA_5'].pct_change(5) * 100
    df['MA_10_slope'] = df['MA_10'].pct_change(5) * 100
    df['MA_20_slope'] = df['MA_20'].pct_change(10) * 100

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR for volatility
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

    # TD Sequential (Tom DeMark)
    df = calculate_td_sequential(df)

    return df


def calculate_td_sequential(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Tom DeMark TD Sequential (Setup + Countdown).

    TD Sequential is a counter-trend indicator that identifies exhaustion points:

    TD Setup (9-count):
    - Buy Setup: 9 consecutive closes < close[4] (bearish exhaustion)
    - Sell Setup: 9 consecutive closes > close[4] (bullish exhaustion)

    Signal Interpretation:
    - Setup 9 complete = potential exhaustion warning
    - Indicates trend may be near reversal point

    WHY THIS WORKS FOR COMMODITIES:
    - Commodities tend to trend then mean-revert
    - TD Sequential catches exhaustion before momentum indicators
    - Complements momentum-based CCS scoring
    """
    n = len(df)
    df['TD_Setup_Buy'] = 0
    df['TD_Setup_Sell'] = 0

    # Calculate TD Setup
    for i in range(4, n):
        # Buy Setup: Close < Close[4] (indicates downtrend exhaustion)
        if df['Close'].iloc[i] < df['Close'].iloc[i-4]:
            prev_count = df['TD_Setup_Buy'].iloc[i-1]
            if prev_count < 9:
                df.loc[df.index[i], 'TD_Setup_Buy'] = prev_count + 1
            else:
                df.loc[df.index[i], 'TD_Setup_Buy'] = 9
        else:
            df.loc[df.index[i], 'TD_Setup_Buy'] = 0

        # Sell Setup: Close > Close[4] (indicates uptrend exhaustion)
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
    """
    Convert TD Sequential state to signal contribution.

    Returns score in range [-10, 10]:
    - Buy Setup 9 complete: +7.5 (potential bottom, bullish signal)
    - Sell Setup 9 complete: -7.5 (potential top, bearish signal)
    - Partial setups contribute proportionally

    Note: TD Sequential is COUNTER-TREND, so:
    - Buy setup (downtrend exhaustion) = bullish reversal signal
    - Sell setup (uptrend exhaustion) = bearish reversal signal
    """
    buy_count = df_row.get('TD_Setup_Buy', 0)
    sell_count = df_row.get('TD_Setup_Sell', 0)

    score = 0

    # Buy setup complete (9) = strong bullish reversal signal
    if buy_count == 9:
        score += 7.5  # Potential bottom
    elif buy_count >= 7:
        score += 3.0  # Approaching potential bottom

    # Sell setup complete (9) = strong bearish reversal signal
    if sell_count == 9:
        score -= 7.5  # Potential top
    elif sell_count >= 7:
        score -= 3.0  # Approaching potential top

    return score


def calculate_sentiment_score(df_row: pd.Series) -> float:
    """
    Calculate sentiment score based on technical indicators.
    This implements the CCS (Carnival Core Score) algorithm with TD Sequential.

    Components (Updated with TD Sequential):
    - ROC (Price Momentum): 35% (was 40%)
    - MA Trend: 25% (was 30%)
    - Volume & Conviction: 15% (was 18%)
    - RSI: 10% (was 12%)
    - TD Sequential: 15% (NEW - counter-trend exhaustion)
    """
    # === ROC SIGNALS (35%) ===
    roc_score = (
        df_row['ROC_1'] * 0.15 +
        df_row['ROC_2'] * 0.15 +
        df_row['ROC_5'] * 0.25 +
        df_row['ROC_10'] * 0.25 +
        df_row['ROC_20'] * 0.20
    )
    roc_normalized = normalize_to_range(roc_score) * 0.35

    # === MA TREND SIGNALS (25%) ===
    ma_slope_score = (
        df_row['MA_5_slope'] * 0.3 +
        df_row['MA_10_slope'] * 0.3 +
        df_row['MA_20_slope'] * 0.4
    )

    # MA crossover signals
    ma_cross_score = 0
    if df_row['MA_5'] > df_row['MA_10'] > df_row['MA_20']:
        ma_cross_score = 5  # Bullish alignment
    elif df_row['MA_5'] < df_row['MA_10'] < df_row['MA_20']:
        ma_cross_score = -5  # Bearish alignment

    ma_normalized = (
        normalize_to_range(ma_slope_score) * 0.7 +
        normalize_to_range(ma_cross_score) * 0.3
    ) * 0.25

    # === VOLUME & CONVICTION (15%) ===
    vol_score = (
        normalize_to_range((df_row['Vol_Ratio'] - 1) * 100) * 0.40 +
        normalize_to_range((df_row['Vol_Trend'] - 1) * 100) * 0.30 +
        normalize_to_range(df_row['Vol_Price_Corr'] * 100) * 0.30
    ) * 0.15

    # === RSI (10%) ===
    rsi_normalized = (df_row['RSI'] - 50) / 5
    rsi_score = normalize_to_range(rsi_normalized) * 0.10

    # === TD SEQUENTIAL (15%) ===
    td_raw_score = get_td_signal_score(df_row)
    td_score = normalize_to_range(td_raw_score) * 0.15

    # === TOTAL SENTIMENT SCORE ===
    total_score = roc_normalized + ma_normalized + vol_score + rsi_score + td_score

    return np.clip(total_score, -10, 10)


# ============================================================================
# CORRELATION ALIGNMENT SCORE
# ============================================================================

def calculate_correlation_alignment_score(
    commodity_ticker: str,
    correlation_engine: CorrelationEngine,
    price_data: Dict[str, pd.DataFrame]
) -> float:
    """
    Calculate score based on whether correlated assets confirm direction.

    WHY THIS WORKS:
    - Multi-asset agreement = higher conviction
    - If Gold is bullish AND DXY is bearish: Confirming (+1.0)
    - If Gold is bullish AND DXY is bullish: Warning (-0.5)
    - Weights by historical correlation strength

    Args:
        commodity_ticker: Target commodity
        correlation_engine: CorrelationEngine instance
        price_data: Dict of price DataFrames for all assets

    Returns:
        Alignment score in range [-10, 10]
    """
    key_correlations = KEY_CORRELATIONS.get(commodity_ticker, [])

    if not key_correlations:
        return 0.0

    # Get commodity's recent trend (10-day ROC)
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

        # Get expected correlation
        expected_corr = EXPECTED_CORRELATIONS.get(
            commodity_ticker, {}
        ).get(comp_ticker, 0)

        # Calculate alignment
        if expected_corr < 0:
            # Inverse correlation expected
            # Confirming: commodity up, comparison down (or vice versa)
            if commodity_bullish != comp_bullish:
                alignment = 1.0  # Confirming
            else:
                alignment = -0.5  # Warning - moving same direction
        elif expected_corr > 0:
            # Positive correlation expected
            # Confirming: both moving same direction
            if commodity_bullish == comp_bullish:
                alignment = 1.0  # Confirming
            else:
                alignment = -0.5  # Warning - diverging
        else:
            alignment = 0.0

        # Weight by expected correlation strength
        weight = abs(expected_corr)
        alignment_scores.append(alignment * weight)

    if not alignment_scores:
        return 0.0

    # Average alignment, normalized to [-10, 10]
    avg_alignment = np.mean(alignment_scores)
    return normalize_to_range(avg_alignment * 10)


# ============================================================================
# DIVERGENCE DETECTION
# ============================================================================

def detect_divergence_signal(
    commodity_ticker: str,
    correlation_engine: CorrelationEngine,
    price_data: Dict[str, pd.DataFrame],
    lookback: int = 10
) -> Dict:
    """
    Detect when commodity diverges from normally correlated assets.

    WHY THIS WORKS:
    - Assets in correlation eventually converge
    - Divergence = mean-reversion opportunity
    - BULLISH DIVERGENCE: Gold flat/down while DXY falling (gold should rise)
    - BEARISH DIVERGENCE: Gold flat/up while DXY rising (gold should fall)

    Args:
        commodity_ticker: Target commodity
        correlation_engine: CorrelationEngine instance
        price_data: Dict of price DataFrames
        lookback: Days to measure divergence

    Returns:
        Dict with divergence_type, score, magnitude
    """
    key_correlations = KEY_CORRELATIONS.get(commodity_ticker, [])

    if not key_correlations:
        return {'divergence_type': 'none', 'score': 0, 'magnitude': 0}

    # Get commodity's recent performance
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

        # Get expected correlation
        expected_corr = EXPECTED_CORRELATIONS.get(
            commodity_ticker, {}
        ).get(comp_ticker, 0)

        if expected_corr == 0:
            continue

        # Calculate expected vs actual relationship
        if expected_corr < 0:
            # Should move opposite
            # Divergence = both moving same direction
            if np.sign(commodity_return) == np.sign(comp_return):
                # Calculate implied move
                implied_commodity_return = -comp_return * abs(expected_corr)
                divergence = implied_commodity_return - commodity_return
                divergences.append({
                    'ticker': comp_ticker,
                    'divergence': divergence,
                    'implied_return': implied_commodity_return,
                })
        else:
            # Should move together
            # Divergence = moving opposite directions
            if np.sign(commodity_return) != np.sign(comp_return):
                implied_commodity_return = comp_return * abs(expected_corr)
                divergence = implied_commodity_return - commodity_return
                divergences.append({
                    'ticker': comp_ticker,
                    'divergence': divergence,
                    'implied_return': implied_commodity_return,
                })

    if not divergences:
        return {'divergence_type': 'none', 'score': 0, 'magnitude': 0}

    # Average divergence
    avg_divergence = np.mean([d['divergence'] for d in divergences])

    # Classify divergence
    if avg_divergence > 2:
        divergence_type = 'bullish'  # Commodity should catch up (go up)
        score = normalize_to_range(avg_divergence)
    elif avg_divergence < -2:
        divergence_type = 'bearish'  # Commodity should catch down
        score = normalize_to_range(avg_divergence)
    else:
        divergence_type = 'none'
        score = 0

    return {
        'divergence_type': divergence_type,
        'score': score,
        'magnitude': abs(avg_divergence),
        'details': divergences,
    }


# ============================================================================
# MACRO REGIME DETECTION
# ============================================================================

def detect_macro_quadrant(price_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Determine current macro quadrant using Fidenza framework.

    Quadrants:
    - Quad 1 (Goldilocks): Rising growth + Falling inflation
    - Quad 2 (Reflation): Rising growth + Rising inflation
    - Quad 3 (Stagflation): Falling growth + Rising inflation
    - Quad 4 (Risk-Off): Falling growth + Falling inflation

    Growth Signal: Equity indices (70%) + Cyclical commodities (30%)
    Inflation Signal: Treasury prices inverted (50%) + Inflation hedges (50%)
    """
    # === GROWTH SIGNAL ===
    growth_tickers = ['ESA Index', 'NQA Index', 'RTYA Index']  # Equities
    cyclical_tickers = ['HGA Comdty', 'CLA Comdty']  # Copper, Oil

    equity_returns = []
    for ticker in growth_tickers:
        if ticker in price_data and price_data[ticker] is not None:
            df = price_data[ticker]
            if len(df) >= 20:
                ret = df['Close'].pct_change(20).iloc[-1] * 100
                equity_returns.append(ret)

    cyclical_returns = []
    for ticker in cyclical_tickers:
        if ticker in price_data and price_data[ticker] is not None:
            df = price_data[ticker]
            if len(df) >= 20:
                ret = df['Close'].pct_change(20).iloc[-1] * 100
                cyclical_returns.append(ret)

    equity_signal = np.mean(equity_returns) if equity_returns else 0
    cyclical_signal = np.mean(cyclical_returns) if cyclical_returns else 0
    growth_signal = equity_signal * 0.7 + cyclical_signal * 0.3
    growth_rising = growth_signal > 0

    # === INFLATION SIGNAL ===
    # Bond prices: falling bonds = rising yields = rising inflation
    bond_tickers = ['TUA Comdty', 'FVA Comdty', 'TYA Comdty', 'USA Comdty']
    inflation_hedge_tickers = ['GCA Comdty', 'SIA Comdty', 'CLA Comdty']

    bond_returns = []
    for ticker in bond_tickers:
        if ticker in price_data and price_data[ticker] is not None:
            df = price_data[ticker]
            if len(df) >= 20:
                ret = df['Close'].pct_change(20).iloc[-1] * 100
                bond_returns.append(ret)

    inflation_returns = []
    for ticker in inflation_hedge_tickers:
        if ticker in price_data and price_data[ticker] is not None:
            df = price_data[ticker]
            if len(df) >= 20:
                ret = df['Close'].pct_change(20).iloc[-1] * 100
                inflation_returns.append(ret)

    # Invert bond signal (falling bonds = rising yields = rising inflation)
    bond_signal = -np.mean(bond_returns) if bond_returns else 0
    inflation_hedge_signal = np.mean(inflation_returns) if inflation_returns else 0
    inflation_signal = bond_signal * 0.5 + inflation_hedge_signal * 0.5
    inflation_rising = inflation_signal > 0

    # === DETERMINE QUADRANT ===
    if growth_rising and not inflation_rising:
        quadrant = 1  # Goldilocks
        name = "Goldilocks"
        description = "Rising growth + Falling inflation"
    elif growth_rising and inflation_rising:
        quadrant = 2  # Reflation
        name = "Reflation"
        description = "Rising growth + Rising inflation"
    elif not growth_rising and inflation_rising:
        quadrant = 3  # Stagflation
        name = "Stagflation"
        description = "Falling growth + Rising inflation"
    else:
        quadrant = 4  # Risk-Off
        name = "Risk-Off"
        description = "Falling growth + Falling inflation"

    return {
        'quadrant': quadrant,
        'name': name,
        'description': description,
        'growth_signal': growth_signal,
        'growth_rising': growth_rising,
        'inflation_signal': inflation_signal,
        'inflation_rising': inflation_rising,
    }


def apply_regime_adjustment(
    base_signal: float,
    quadrant: int,
    commodity_type: str
) -> float:
    """
    Adjust signal based on macro quadrant.

    WHY THIS WORKS:
    - Different macro environments favor different commodities
    - Stagflation (Quad 3): Gold outperforms (inflation hedge + safe haven)
    - Reflation (Quad 2): Oil outperforms (growth proxy)
    - Risk-Off (Quad 4): Gold outperforms, Oil underperforms
    """
    adjustment = REGIME_ADJUSTMENTS.get(commodity_type, {}).get(quadrant, 0)
    adjusted_signal = base_signal * (1 + adjustment)

    return np.clip(adjusted_signal, -10, 10)


# ============================================================================
# COMPOSITE SIGNAL GENERATOR
# ============================================================================

class CommoditySignalGenerator:
    """
    Generates composite buy/sell signals for commodities.

    Signal = Sentiment (40%) + Correlation (25%) + Divergence (20%) + Regime (15%)
    """

    def __init__(self, lookback_days: int = 252):
        """
        Initialize signal generator.

        Args:
            lookback_days: Days of historical data for analysis
        """
        self.lookback_days = lookback_days
        self.correlation_engine = CorrelationEngine(lookback_days)
        self.price_data_cache = {}

    def _fetch_all_price_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch price data for all relevant assets.
        """
        all_tickers = (
            list(COMMODITY_UNIVERSE.keys()) +
            list(CURRENCY_UNIVERSE.keys()) +
            list(BOND_UNIVERSE.keys()) +
            list(EQUITY_UNIVERSE.keys())
        )

        # Add additional commodities for macro analysis
        all_tickers.extend(['HGA Comdty', 'NGA Comdty'])
        all_tickers = list(set(all_tickers))  # Remove duplicates

        price_data = {}

        for ticker in all_tickers:
            df = fetch_historical_data(ticker, days=self.lookback_days)
            if df is not None and len(df) > 20:
                df = calculate_technical_indicators(df)
                price_data[ticker] = df

        self.price_data_cache = price_data
        return price_data

    def generate_signal(
        self,
        commodity_ticker: str,
        as_of_date: datetime = None
    ) -> Dict:
        """
        Generate composite signal for a commodity.

        Args:
            commodity_ticker: Target commodity (e.g., 'GCA Comdty')
            as_of_date: Date for signal (default: latest)

        Returns:
            Dict with signal, components breakdown, classification, action
        """
        # Fetch all price data if not cached
        if not self.price_data_cache:
            self._fetch_all_price_data()

        price_data = self.price_data_cache

        if commodity_ticker not in price_data:
            return {
                'signal': 0,
                'classification': 'N/A',
                'action': 'No Data',
                'error': f'No data for {commodity_ticker}'
            }

        commodity_df = price_data[commodity_ticker]

        # Get commodity type for regime adjustment
        commodity_info = COMMODITY_UNIVERSE.get(commodity_ticker, {})
        commodity_type = commodity_info.get('type', 'precious_metal')

        # === COMPONENT 1: SENTIMENT SCORE (40%) ===
        sentiment_score = calculate_sentiment_score(commodity_df.iloc[-1])
        sentiment_weighted = sentiment_score * SIGNAL_WEIGHTS['sentiment']

        # === COMPONENT 2: CORRELATION ALIGNMENT (25%) ===
        correlation_score = calculate_correlation_alignment_score(
            commodity_ticker,
            self.correlation_engine,
            price_data
        )
        correlation_weighted = correlation_score * SIGNAL_WEIGHTS['correlation']

        # === COMPONENT 3: DIVERGENCE DETECTION (20%) ===
        divergence_result = detect_divergence_signal(
            commodity_ticker,
            self.correlation_engine,
            price_data
        )
        divergence_score = divergence_result['score']
        divergence_weighted = divergence_score * SIGNAL_WEIGHTS['divergence']

        # === COMPONENT 4: REGIME ADJUSTMENT (15%) ===
        macro_quadrant = detect_macro_quadrant(price_data)

        # Base signal before regime adjustment
        base_signal = sentiment_weighted + correlation_weighted + divergence_weighted

        # Apply regime adjustment
        adjusted_signal = apply_regime_adjustment(
            base_signal,
            macro_quadrant['quadrant'],
            commodity_type
        )

        # Final signal (regime adjustment is multiplicative, not additive)
        # So we calculate the regime contribution as the difference
        regime_contribution = adjusted_signal - base_signal

        # === FINAL SIGNAL ===
        final_signal = np.clip(adjusted_signal, -10, 10)

        # Classify signal
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
        """
        Generate signals for all commodities in universe.

        Returns:
            Dict mapping commodity ticker to signal result
        """
        # Refresh price data
        self._fetch_all_price_data()

        signals = {}

        for commodity_ticker in COMMODITY_UNIVERSE.keys():
            signals[commodity_ticker] = self.generate_signal(commodity_ticker)

        return signals

    def generate_entry_signal(
        self,
        commodity_ticker: str,
        current_position: Optional[str] = None  # 'long', 'short', or None
    ) -> Dict:
        """
        Generate entry signal with position management logic.

        Entry Criteria:
        1. Signal above threshold (>4 for long, <-4 for short)
        2. Correlation regime not 'breakdown'
        3. No existing position in same direction

        Args:
            commodity_ticker: Target commodity
            current_position: Existing position direction

        Returns:
            Dict with entry_signal, entry_direction, confidence
        """
        signal_result = self.generate_signal(commodity_ticker)

        signal = signal_result['signal']
        classification = signal_result['classification']

        # Get correlation regime
        key_correlations = KEY_CORRELATIONS.get(commodity_ticker, [])
        regime_check = 'normal'

        for comp_ticker in key_correlations[:2]:  # Check top 2 correlations
            regime = self.correlation_engine.detect_correlation_regime(
                commodity_ticker, comp_ticker
            )
            if regime['regime'] == 'breakdown':
                regime_check = 'breakdown'
                break

        # Determine entry signal
        entry_signal = False
        entry_direction = None
        confidence = 0

        if regime_check == 'breakdown':
            # Don't enter new positions during correlation breakdown
            entry_signal = False
            reason = "Correlation breakdown - avoid new entries"
        elif signal >= SIGNAL_THRESHOLDS['bullish']:
            if current_position != 'long':
                entry_signal = True
                entry_direction = 'long'
                confidence = min(signal / 10, 1.0)
                reason = f"Bullish signal ({signal:.1f})"
            else:
                reason = "Already long"
        elif signal <= SIGNAL_THRESHOLDS['bearish']:
            if current_position != 'short':
                entry_signal = True
                entry_direction = 'short'
                confidence = min(abs(signal) / 10, 1.0)
                reason = f"Bearish signal ({signal:.1f})"
            else:
                reason = "Already short"
        else:
            reason = "Signal in neutral zone"

        return {
            'entry_signal': entry_signal,
            'entry_direction': entry_direction,
            'confidence': round(confidence, 2),
            'reason': reason,
            'correlation_regime': regime_check,
            'full_signal': signal_result,
        }

    def generate_exit_signal(
        self,
        commodity_ticker: str,
        entry_price: float,
        current_price: float,
        entry_signal: float,
        holding_days: int,
        position_direction: str  # 'long' or 'short'
    ) -> Dict:
        """
        Generate exit/profit-taking signal.

        Exit Triggers:
        1. SIGNAL DETERIORATION: Signal drops below entry threshold
        2. CORRELATION BREAKDOWN: Confirmation assets diverge
        3. TIME-BASED: Holding > 20 days without improvement
        4. STOP-LOSS: Price against position by 2x ATR
        5. PROFIT TARGET: 3:1 reward/risk achieved

        Args:
            commodity_ticker: Target commodity
            entry_price: Position entry price
            current_price: Current price
            entry_signal: Signal at entry
            holding_days: Days position has been held
            position_direction: 'long' or 'short'

        Returns:
            Dict with exit_signal, exit_reason, urgency
        """
        current_signal = self.generate_signal(commodity_ticker)
        signal = current_signal['signal']

        # Get ATR for stop calculation
        if commodity_ticker in self.price_data_cache:
            df = self.price_data_cache[commodity_ticker]
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.02
        else:
            atr = current_price * 0.02  # Default 2% ATR

        # Calculate P&L
        if position_direction == 'long':
            pnl_pct = (current_price - entry_price) / entry_price * 100
            stop_price = entry_price - (2 * atr)
            stop_hit = current_price <= stop_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
            stop_price = entry_price + (2 * atr)
            stop_hit = current_price >= stop_price

        exit_signal = False
        exit_reason = None
        urgency = 'low'

        # Check exit conditions in order of urgency
        if stop_hit:
            exit_signal = True
            exit_reason = f"Stop-loss hit (2x ATR at {stop_price:.2f})"
            urgency = 'high'

        elif position_direction == 'long' and signal < 0:
            exit_signal = True
            exit_reason = f"Signal turned bearish ({signal:.1f})"
            urgency = 'medium'

        elif position_direction == 'short' and signal > 0:
            exit_signal = True
            exit_reason = f"Signal turned bullish ({signal:.1f})"
            urgency = 'medium'

        elif holding_days > 20 and abs(signal) < abs(entry_signal):
            exit_signal = True
            exit_reason = f"Signal deteriorating after {holding_days} days"
            urgency = 'low'

        elif pnl_pct > 6:  # ~3:1 reward assuming 2% risk
            exit_signal = True
            exit_reason = f"Profit target reached ({pnl_pct:.1f}%)"
            urgency = 'low'

        return {
            'exit_signal': exit_signal,
            'exit_reason': exit_reason,
            'urgency': urgency,
            'current_signal': signal,
            'pnl_pct': round(pnl_pct, 2),
            'stop_price': round(stop_price, 2),
            'holding_days': holding_days,
        }


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def print_signal_report(signals: Dict[str, Dict]):
    """Print formatted signal report for all commodities."""
    print("\n" + "="*100)
    print(f"{'COMMODITY TRADING SIGNALS':^100}")
    print(f"{'Sentiment (40%) + Correlation (25%) + Divergence (20%) + Regime (15%)':^100}")
    print("="*100 + "\n")

    # Get macro regime (same for all)
    first_signal = list(signals.values())[0]
    macro = first_signal.get('macro_regime', {})

    print(f"Macro Environment: {macro.get('name', 'Unknown')} (Quad {macro.get('quadrant', 0)})")
    print(f"  Growth: {macro.get('growth_signal', 0):+.2f} ({'Rising' if macro.get('growth_rising') else 'Falling'})")
    print(f"  Inflation: {macro.get('inflation_signal', 0):+.2f} ({'Rising' if macro.get('inflation_rising') else 'Falling'})")
    print()

    # Header
    print(f"{'Commodity':<15} {'Signal':>8} {'Classification':<15} {'Action':<25} {'Sent':>6} {'Corr':>6} {'Div':>6} {'Reg':>6}")
    print("-"*100)

    # Sort by signal (descending)
    sorted_signals = sorted(
        signals.items(),
        key=lambda x: x[1].get('signal', 0),
        reverse=True
    )

    for ticker, data in sorted_signals:
        name = COMMODITY_UNIVERSE.get(ticker, {}).get('name', ticker)
        signal = data.get('signal', 0)
        classification = data.get('classification', 'N/A')
        action = data.get('action', 'N/A')

        components = data.get('components', {})
        sentiment = components.get('sentiment', 0)
        correlation = components.get('correlation', 0)
        divergence = components.get('divergence', 0)
        regime_adj = components.get('regime_adjustment', 0)

        print(f"{name:<15} {signal:>8.2f} {classification:<15} {action:<25} "
              f"{sentiment:>6.2f} {correlation:>6.2f} {divergence:>6.2f} {regime_adj:>6.2f}")

    print("="*100)
    print(f"\nSignal Interpretation:")
    print(f"  >= 7.0: Very Bullish (Strong Buy)")
    print(f"  >= 4.0: Bullish (Buy)")
    print(f"  -4.0 to 4.0: Neutral (No Position)")
    print(f"  <= -4.0: Bearish (Sell)")
    print(f"  <= -7.0: Very Bearish (Strong Sell)")
    print("="*100 + "\n")


# ============================================================================
# DATA VALIDATION FUNCTIONS
# ============================================================================

def validate_price_data(ticker: str, days: int = 30) -> Dict:
    """
    Validate BQL data against Bloomberg Terminal reference.

    Steps for manual verification:
    1. Run this function to print recent data points
    2. Open Bloomberg Terminal and run 'HP {ticker}' command
    3. Compare the last 5 data points

    Args:
        ticker: Bloomberg ticker to validate
        days: Days of data to fetch

    Returns:
        Dict with validation summary
    """
    df = fetch_historical_data(ticker, days=days)

    if df is None:
        return {'error': f'No data returned for {ticker}'}

    print(f"\n{'='*60}")
    print(f"DATA VALIDATION: {ticker}")
    print(f"{'='*60}")
    print(f"\nTo verify, run 'HP {ticker}' in Bloomberg Terminal")
    print(f"\nLast 5 data points from BQL:")
    print(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(5).to_string(index=False))

    return {
        'ticker': ticker,
        'latest_date': str(df['Date'].iloc[-1]),
        'latest_close': df['Close'].iloc[-1],
        'data_points': len(df),
        'date_range': f"{df['Date'].iloc[0]} to {df['Date'].iloc[-1]}",
    }


def run_data_sanity_checks(ticker: str, days: int = 90) -> Dict:
    """
    Automated sanity checks for data quality.

    Checks performed:
    1. No excessive date gaps (>4 days, accounting for holidays)
    2. Prices are positive
    3. No extreme daily moves (>20%)
    4. Volume data exists (for volume-traded instruments)
    5. OHLC relationships are valid (High >= Low, etc.)

    Args:
        ticker: Bloomberg ticker to check
        days: Days of data to analyze

    Returns:
        Dict with check results and pass/fail status
    """
    df = fetch_historical_data(ticker, days=days)

    if df is None:
        return {'error': f'No data returned for {ticker}', 'all_passed': False}

    checks = {}

    # Check 1: No missing dates (gaps > 4 days excluding weekends/holidays)
    date_series = pd.to_datetime(df['Date'])
    date_diffs = date_series.diff().dt.days
    checks['max_gap_days'] = int(date_diffs.max()) if not pd.isna(date_diffs.max()) else 0
    checks['gap_check'] = 'PASS' if checks['max_gap_days'] <= 4 else 'FAIL'

    # Check 2: Prices in reasonable range (positive)
    checks['min_price'] = float(df['Close'].min())
    checks['max_price'] = float(df['Close'].max())
    checks['price_check'] = 'PASS' if checks['min_price'] > 0 else 'FAIL'

    # Check 3: No extreme daily moves (>20% unusual for commodities)
    daily_returns = df['Close'].pct_change().abs()
    checks['max_daily_move_pct'] = float(daily_returns.max() * 100)
    checks['move_check'] = 'PASS' if daily_returns.max() < 0.20 else 'WARNING'

    # Check 4: Volume consistency
    checks['total_volume'] = float(df['Volume'].sum())
    checks['volume_check'] = 'PASS' if df['Volume'].sum() > 0 else 'N/A (no volume data)'

    # Check 5: OHLC relationship valid
    ohlc_valid = (
        (df['High'] >= df['Low']).all() and
        (df['High'] >= df['Open']).all() and
        (df['High'] >= df['Close']).all() and
        (df['Low'] <= df['Open']).all() and
        (df['Low'] <= df['Close']).all()
    )
    checks['ohlc_check'] = 'PASS' if ohlc_valid else 'FAIL'

    # Overall status
    critical_checks = [checks['gap_check'], checks['price_check'], checks['ohlc_check']]
    checks['all_passed'] = all(c == 'PASS' for c in critical_checks)

    print(f"\n{'='*60}")
    print(f"SANITY CHECKS: {ticker}")
    print(f"{'='*60}")
    for key, value in checks.items():
        print(f"  {key}: {value}")

    return checks


def verify_signal_calculation(ticker: str, days: int = 90) -> Dict:
    """
    Step-by-step verification of signal calculation.

    Prints each component of the signal calculation for manual review.
    Useful for debugging and understanding signal drivers.

    Args:
        ticker: Bloomberg ticker to verify
        days: Days of data to use

    Returns:
        Dict with calculation breakdown
    """
    df = fetch_historical_data(ticker, days=days)

    if df is None:
        return {'error': f'No data returned for {ticker}'}

    df = calculate_technical_indicators(df)
    latest = df.iloc[-1]

    print(f"\n{'='*70}")
    print(f"SIGNAL CALCULATION VERIFICATION: {ticker}")
    print(f"{'='*70}")

    print(f"\n1. RAW INDICATOR VALUES:")
    print(f"   Close Price: {latest['Close']:.2f}")
    print(f"   ROC_1 (1-day): {latest['ROC_1']:.4f}%")
    print(f"   ROC_5 (5-day): {latest['ROC_5']:.4f}%")
    print(f"   ROC_10 (10-day): {latest['ROC_10']:.4f}%")
    print(f"   ROC_20 (20-day): {latest['ROC_20']:.4f}%")
    print(f"   RSI (14): {latest['RSI']:.2f}")
    print(f"   MA_5: {latest['MA_5']:.2f}")
    print(f"   MA_10: {latest['MA_10']:.2f}")
    print(f"   MA_20: {latest['MA_20']:.2f}")

    # TD Sequential
    print(f"\n2. TD SEQUENTIAL STATUS:")
    print(f"   TD_Setup_Buy: {latest.get('TD_Setup_Buy', 0)}")
    print(f"   TD_Setup_Sell: {latest.get('TD_Setup_Sell', 0)}")

    # Calculate components
    roc_raw = (
        latest['ROC_1'] * 0.15 +
        latest['ROC_2'] * 0.15 +
        latest['ROC_5'] * 0.25 +
        latest['ROC_10'] * 0.25 +
        latest['ROC_20'] * 0.20
    )

    td_raw = get_td_signal_score(latest)

    print(f"\n3. COMPONENT CALCULATIONS:")
    print(f"   ROC raw weighted: {roc_raw:.4f}")
    print(f"   ROC normalized (35%): {normalize_to_range(roc_raw) * 0.35:.4f}")
    print(f"   TD Sequential raw: {td_raw:.4f}")
    print(f"   TD normalized (15%): {normalize_to_range(td_raw) * 0.15:.4f}")

    # Full signal
    signal = calculate_sentiment_score(latest)

    print(f"\n4. FINAL SENTIMENT SCORE: {signal:.2f}")
    classification, action = classify_signal(signal)
    print(f"   Classification: {classification}")
    print(f"   Recommended Action: {action}")
    print(f"{'='*70}")

    return {
        'ticker': ticker,
        'close': latest['Close'],
        'signal': signal,
        'classification': classification,
        'roc_raw': roc_raw,
        'td_raw': td_raw,
        'td_setup_buy': latest.get('TD_Setup_Buy', 0),
        'td_setup_sell': latest.get('TD_Setup_Sell', 0),
    }


def verify_correlation_calculations() -> Dict:
    """
    Verify correlation calculations are mathematically correct.

    Compares manual correlation calculation with engine output.
    Gold-DXY correlation should typically be -0.3 to -0.6.

    Returns:
        Dict with correlation verification results
    """
    from correlation_analysis import CorrelationEngine

    print(f"\n{'='*60}")
    print("CORRELATION CALCULATION VERIFICATION")
    print(f"{'='*60}")

    # Fetch Gold and DXY
    gold_df = fetch_historical_data('GCA Comdty', days=60)
    dxy_df = fetch_historical_data('DXY Index', days=60)

    if gold_df is None or dxy_df is None:
        return {'error': 'Could not fetch data for verification'}

    # Merge on date
    gold_df['Date'] = pd.to_datetime(gold_df['Date'])
    dxy_df['Date'] = pd.to_datetime(dxy_df['Date'])

    merged = pd.merge(
        gold_df[['Date', 'Close']],
        dxy_df[['Date', 'Close']],
        on='Date',
        suffixes=('_gold', '_dxy')
    )

    # Calculate correlation manually
    gold_returns = merged['Close_gold'].pct_change().dropna()
    dxy_returns = merged['Close_dxy'].pct_change().dropna()

    manual_corr = gold_returns.corr(dxy_returns)

    # Expected range
    expected_low = -0.7
    expected_high = -0.2
    status = 'OK' if expected_low < manual_corr < expected_high else 'UNUSUAL - investigate'

    print(f"\n  Gold-DXY Correlation (manual calc): {manual_corr:.4f}")
    print(f"  Expected range: {expected_low} to {expected_high}")
    print(f"  Status: {status}")

    # Also check with correlation engine
    engine = CorrelationEngine(lookback_days=60)
    engine_result = engine.calculate_correlation_zscore('GCA Comdty', 'DXY Index')

    print(f"\n  Correlation Engine result: {engine_result.get('current_corr', 'N/A'):.4f}")
    print(f"  Z-Score: {engine_result.get('zscore', 'N/A'):.2f}")
    print(f"{'='*60}")

    return {
        'manual_correlation': manual_corr,
        'engine_correlation': engine_result.get('current_corr'),
        'expected_range': f"{expected_low} to {expected_high}",
        'status': status,
    }


def run_full_validation(tickers: List[str] = None) -> None:
    """
    Run full validation suite for all specified tickers.

    Args:
        tickers: List of tickers to validate (default: all commodities)
    """
    if tickers is None:
        tickers = list(COMMODITY_UNIVERSE.keys())

    print("\n" + "="*70)
    print("FULL DATA VALIDATION SUITE")
    print("="*70)

    all_results = {}

    for ticker in tickers:
        print(f"\n>>> Validating {ticker}...")
        name = COMMODITY_UNIVERSE.get(ticker, {}).get('name', ticker)

        # Run sanity checks
        sanity = run_data_sanity_checks(ticker)
        all_results[ticker] = {
            'name': name,
            'sanity_checks': sanity,
        }

        # Run signal verification
        signal_verify = verify_signal_calculation(ticker)
        all_results[ticker]['signal_verification'] = signal_verify

    # Run correlation verification
    print("\n>>> Verifying correlation calculations...")
    corr_verify = verify_correlation_calculations()
    all_results['correlation_verification'] = corr_verify

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    for ticker, results in all_results.items():
        if ticker == 'correlation_verification':
            continue
        sanity = results.get('sanity_checks', {})
        status = 'PASS' if sanity.get('all_passed', False) else 'REVIEW NEEDED'
        print(f"  {results.get('name', ticker)}: {status}")

    print(f"\n  Correlation Check: {corr_verify.get('status', 'Unknown')}")
    print("="*70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("Commodity Signal Generator - Testing")
    print("="*50)

    # Initialize generator
    generator = CommoditySignalGenerator(lookback_days=120)

    # Generate all signals
    signals = generator.generate_all_signals()

    # Print report
    print_signal_report(signals)

    # Test entry/exit signals for Gold
    print("\n" + "="*50)
    print("Entry Signal Test - Gold")
    print("="*50)

    entry = generator.generate_entry_signal('GCA Comdty')
    print(f"Entry Signal: {entry['entry_signal']}")
    print(f"Direction: {entry['entry_direction']}")
    print(f"Confidence: {entry['confidence']}")
    print(f"Reason: {entry['reason']}")
