#!/usr/bin/env python3
"""
Commodity Trading Signal Generator (CSV Version)
Generates buy/sell signals for Gold, Silver, Platinum, Palladium, and Oil

Signal Components:
- Sentiment Score (CCS): 40% - Existing momentum/trend analysis
- Correlation Alignment: 25% - Cross-asset confirmation
- Divergence Detection: 20% - Mean-reversion opportunities
- Regime Adjustment: 15% - Macro context

Signal Range: -10 (Very Bearish) to +10 (Very Bullish)

CSV VERSION: Uses local CSV data instead of Bloomberg BQL
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Import from CSV data loader instead of BQL
from csv_data_loader import (
    fetch_historical_data,
    fetch_price_data,
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
)

# Import from CSV correlation analysis module
from correlation_analysis_csv import (
    CorrelationEngine,
    EXPECTED_CORRELATIONS,
    KEY_CORRELATIONS,
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

# =============================================================================
# ENHANCED REGIME ADJUSTMENTS - Multi-Layer
# =============================================================================
# Layer 1: Base quadrant adjustments (existing Fidenza framework)
# Layer 2: VIX regime overlay (volatility environment)
# Layer 3: Real yield regime overlay (rate environment)
# Layer 4: Credit regime overlay (risk appetite)

# Base regime adjustments by commodity type and macro quadrant
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
    'industrial_metal': {  # Copper, Aluminum, Nickel
        1: 0.20,    # Goldilocks: Growth supports demand
        2: 0.25,    # Reflation: Strong growth
        3: -0.20,   # Stagflation: Demand concerns
        4: -0.30,   # Risk-Off: Demand collapse
    },
    'agricultural': {  # Corn, Soybeans, Wheat
        1: 0.05,    # Goldilocks: Modest support
        2: 0.15,    # Reflation: Inflation hedge
        3: 0.20,    # Stagflation: Inflation hedge
        4: -0.10,   # Risk-Off: Mild negative
    },
}

# VIX regime adjustments (overlay on base quadrant)
VIX_REGIME_ADJUSTMENTS = {
    'precious_metal': {
        'low_vol': -0.15,      # Complacency = less safe haven demand
        'normal': 0.00,
        'elevated': 0.20,       # Fear building = gold bid
        'crisis': 0.40,         # Panic = strong safe haven
    },
    'energy': {
        'low_vol': 0.10,        # Stable growth = demand
        'normal': 0.00,
        'elevated': -0.10,      # Uncertainty = demand concerns
        'crisis': -0.25,        # Panic = demand collapse
    },
    'industrial_metal': {
        'low_vol': 0.10,        # Stable growth = demand
        'normal': 0.00,
        'elevated': -0.15,      # Uncertainty = demand concerns
        'crisis': -0.30,        # Panic = demand collapse
    },
    'agricultural': {
        'low_vol': 0.00,
        'normal': 0.00,
        'elevated': 0.05,       # Mild inflation hedge
        'crisis': 0.10,         # Inflation hedge
    },
}

# Real yield regime adjustments (primarily affects gold)
REAL_YIELD_REGIME_ADJUSTMENTS = {
    'precious_metal': {
        'negative': 0.25,       # Negative real yields = very gold bullish
        'low': 0.10,            # Low but positive = supportive
        'normal': 0.00,
        'high': -0.20,          # High real yields = gold headwind
    },
    'energy': {
        'negative': 0.05,
        'low': 0.05,
        'normal': 0.00,
        'high': -0.05,
    },
    'industrial_metal': {
        'negative': 0.05,
        'low': 0.05,
        'normal': 0.00,
        'high': 0.05,           # Higher rates often = growth
    },
    'agricultural': {
        'negative': 0.10,       # Inflation environment
        'low': 0.05,
        'normal': 0.00,
        'high': -0.05,
    },
}

# Credit regime adjustments
CREDIT_REGIME_ADJUSTMENTS = {
    'precious_metal': {
        'tight': -0.10,         # Risk-on = less gold demand
        'normal': 0.00,
        'wide': 0.15,           # Stress = gold bid
        'stress': 0.25,         # High stress = strong gold bid
    },
    'energy': {
        'tight': 0.10,          # Risk-on = energy demand
        'normal': 0.00,
        'wide': -0.15,          # Stress = demand concerns
        'stress': -0.25,        # High stress = demand collapse
    },
    'industrial_metal': {
        'tight': 0.15,          # Risk-on = industrial demand
        'normal': 0.00,
        'wide': -0.15,          # Stress = demand concerns
        'stress': -0.30,        # High stress = demand collapse
    },
    'agricultural': {
        'tight': 0.00,
        'normal': 0.00,
        'wide': 0.05,
        'stress': 0.10,         # Mild inflation hedge
    },
}


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
# TECHNICAL INDICATORS
# ============================================================================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for sentiment scoring.
    """
    df = df.copy()

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
    if 'High' in df.columns and 'Low' in df.columns:
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
    else:
        df['ATR'] = df['Close'] * 0.02  # Default 2% ATR

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
                df.iloc[i, df.columns.get_loc('TD_Setup_Buy')] = prev_count + 1
            else:
                df.iloc[i, df.columns.get_loc('TD_Setup_Buy')] = 9
        else:
            df.iloc[i, df.columns.get_loc('TD_Setup_Buy')] = 0

        # Sell Setup: Close > Close[4] (indicates uptrend exhaustion)
        if df['Close'].iloc[i] > df['Close'].iloc[i-4]:
            prev_count = df['TD_Setup_Sell'].iloc[i-1]
            if prev_count < 9:
                df.iloc[i, df.columns.get_loc('TD_Setup_Sell')] = prev_count + 1
            else:
                df.iloc[i, df.columns.get_loc('TD_Setup_Sell')] = 9
        else:
            df.iloc[i, df.columns.get_loc('TD_Setup_Sell')] = 0

    return df


def get_td_signal_score(df_row: pd.Series) -> float:
    """
    Convert TD Sequential state to signal contribution.
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
    """
    # === ROC SIGNALS (35%) ===
    roc_score = (
        df_row.get('ROC_1', 0) * 0.15 +
        df_row.get('ROC_2', 0) * 0.15 +
        df_row.get('ROC_5', 0) * 0.25 +
        df_row.get('ROC_10', 0) * 0.25 +
        df_row.get('ROC_20', 0) * 0.20
    )
    roc_normalized = normalize_to_range(roc_score) * 0.35

    # === MA TREND SIGNALS (25%) ===
    ma_slope_score = (
        df_row.get('MA_5_slope', 0) * 0.3 +
        df_row.get('MA_10_slope', 0) * 0.3 +
        df_row.get('MA_20_slope', 0) * 0.4
    )

    # MA crossover signals
    ma_cross_score = 0
    ma_5 = df_row.get('MA_5', 0)
    ma_10 = df_row.get('MA_10', 0)
    ma_20 = df_row.get('MA_20', 0)
    if ma_5 > ma_10 > ma_20:
        ma_cross_score = 5  # Bullish alignment
    elif ma_5 < ma_10 < ma_20:
        ma_cross_score = -5  # Bearish alignment

    ma_normalized = (
        normalize_to_range(ma_slope_score) * 0.7 +
        normalize_to_range(ma_cross_score) * 0.3
    ) * 0.25

    # === VOLUME & CONVICTION (15%) ===
    vol_ratio = df_row.get('Vol_Ratio', 1)
    vol_trend = df_row.get('Vol_Trend', 1)
    vol_price_corr = df_row.get('Vol_Price_Corr', 0)

    vol_score = (
        normalize_to_range((vol_ratio - 1) * 100) * 0.40 +
        normalize_to_range((vol_trend - 1) * 100) * 0.30 +
        normalize_to_range(vol_price_corr * 100) * 0.30
    ) * 0.15

    # === RSI (10%) ===
    rsi = df_row.get('RSI', 50)
    if pd.isna(rsi):
        rsi = 50
    rsi_normalized = (rsi - 50) / 5
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
            if commodity_bullish != comp_bullish:
                alignment = 1.0  # Confirming
            else:
                alignment = -0.5  # Warning
        elif expected_corr > 0:
            if commodity_bullish == comp_bullish:
                alignment = 1.0  # Confirming
            else:
                alignment = -0.5  # Warning
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
    correlation_engine: CorrelationEngine,
    price_data: Dict[str, pd.DataFrame],
    lookback: int = 10
) -> Dict:
    """
    Detect when commodity diverges from normally correlated assets.
    """
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

        expected_corr = EXPECTED_CORRELATIONS.get(
            commodity_ticker, {}
        ).get(comp_ticker, 0)

        if expected_corr == 0:
            continue

        if expected_corr < 0:
            if np.sign(commodity_return) == np.sign(comp_return):
                implied_commodity_return = -comp_return * abs(expected_corr)
                divergence = implied_commodity_return - commodity_return
                divergences.append({
                    'ticker': comp_ticker,
                    'divergence': divergence,
                    'implied_return': implied_commodity_return,
                })
        else:
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
    """
    # === GROWTH SIGNAL ===
    growth_tickers = ['ESA Index', 'NQA Index', 'RTYA Index']
    cyclical_tickers = ['CLA Comdty']  # Oil (removed HGA - not in CSV)

    equity_returns = []
    for ticker in growth_tickers:
        if ticker in price_data and price_data[ticker] is not None:
            df = price_data[ticker]
            if len(df) >= 20:
                ret = df['Close'].pct_change(20).iloc[-1] * 100
                if not pd.isna(ret):
                    equity_returns.append(ret)

    cyclical_returns = []
    for ticker in cyclical_tickers:
        if ticker in price_data and price_data[ticker] is not None:
            df = price_data[ticker]
            if len(df) >= 20:
                ret = df['Close'].pct_change(20).iloc[-1] * 100
                if not pd.isna(ret):
                    cyclical_returns.append(ret)

    equity_signal = np.mean(equity_returns) if equity_returns else 0
    cyclical_signal = np.mean(cyclical_returns) if cyclical_returns else 0
    growth_signal = equity_signal * 0.7 + cyclical_signal * 0.3
    growth_rising = growth_signal > 0

    # === INFLATION SIGNAL ===
    bond_tickers = ['TUA Comdty', 'FVA Comdty', 'TYA Comdty', 'USA Comdty']
    inflation_hedge_tickers = ['GCA Comdty', 'SIA Comdty', 'CLA Comdty']

    bond_returns = []
    for ticker in bond_tickers:
        if ticker in price_data and price_data[ticker] is not None:
            df = price_data[ticker]
            if len(df) >= 20:
                ret = df['Close'].pct_change(20).iloc[-1] * 100
                if not pd.isna(ret):
                    bond_returns.append(ret)

    inflation_returns = []
    for ticker in inflation_hedge_tickers:
        if ticker in price_data and price_data[ticker] is not None:
            df = price_data[ticker]
            if len(df) >= 20:
                ret = df['Close'].pct_change(20).iloc[-1] * 100
                if not pd.isna(ret):
                    inflation_returns.append(ret)

    bond_signal = -np.mean(bond_returns) if bond_returns else 0
    inflation_hedge_signal = np.mean(inflation_returns) if inflation_returns else 0
    inflation_signal = bond_signal * 0.5 + inflation_hedge_signal * 0.5
    inflation_rising = inflation_signal > 0

    # === DETERMINE QUADRANT ===
    if growth_rising and not inflation_rising:
        quadrant = 1
        name = "Goldilocks"
        description = "Rising growth + Falling inflation"
    elif growth_rising and inflation_rising:
        quadrant = 2
        name = "Reflation"
        description = "Rising growth + Rising inflation"
    elif not growth_rising and inflation_rising:
        quadrant = 3
        name = "Stagflation"
        description = "Falling growth + Rising inflation"
    else:
        quadrant = 4
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


def detect_vix_regime(price_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Determine VIX regime for volatility overlay.

    Returns:
        regime: 'low_vol', 'normal', 'elevated', 'crisis'
        level: current VIX level
        zscore: z-score vs history
    """
    vix_ticker = 'VIX Index'

    if vix_ticker not in price_data or price_data[vix_ticker] is None:
        return {'regime': 'normal', 'level': 20, 'zscore': 0, 'data_available': False}

    df = price_data[vix_ticker]
    if len(df) < 20:
        return {'regime': 'normal', 'level': 20, 'zscore': 0, 'data_available': False}

    current_vix = df['Close'].iloc[-1]
    vix_mean = df['Close'].mean()
    vix_std = df['Close'].std()
    zscore = (current_vix - vix_mean) / vix_std if vix_std > 0 else 0

    # Determine regime based on absolute level
    if current_vix < 15:
        regime = 'low_vol'
    elif current_vix < 22:
        regime = 'normal'
    elif current_vix < 35:
        regime = 'elevated'
    else:
        regime = 'crisis'

    return {
        'regime': regime,
        'level': current_vix,
        'zscore': zscore,
        'data_available': True
    }


def detect_real_yield_regime(price_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Determine real yield regime for rate environment overlay.

    Returns:
        regime: 'negative', 'low', 'normal', 'high'
        level: current TIPS yield
    """
    tips_ticker = 'H15T10YIE Index'

    if tips_ticker not in price_data or price_data[tips_ticker] is None:
        return {'regime': 'normal', 'level': 1.5, 'data_available': False}

    df = price_data[tips_ticker]
    if len(df) < 20:
        return {'regime': 'normal', 'level': 1.5, 'data_available': False}

    current_yield = df['Close'].iloc[-1]

    # Determine regime
    if current_yield < 0:
        regime = 'negative'
    elif current_yield < 1.0:
        regime = 'low'
    elif current_yield < 2.0:
        regime = 'normal'
    else:
        regime = 'high'

    return {
        'regime': regime,
        'level': current_yield,
        'data_available': True
    }


def detect_credit_regime(price_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Determine credit regime for risk appetite overlay.

    Returns:
        regime: 'tight', 'normal', 'wide', 'stress'
        level: current HY OAS spread (bps)
    """
    hy_ticker = 'LF98OAS Index'

    if hy_ticker not in price_data or price_data[hy_ticker] is None:
        return {'regime': 'normal', 'level': 400, 'data_available': False}

    df = price_data[hy_ticker]
    if len(df) < 20:
        return {'regime': 'normal', 'level': 400, 'data_available': False}

    current_spread = df['Close'].iloc[-1]

    # Determine regime (spreads in bps)
    if current_spread < 350:
        regime = 'tight'
    elif current_spread < 500:
        regime = 'normal'
    elif current_spread < 700:
        regime = 'wide'
    else:
        regime = 'stress'

    return {
        'regime': regime,
        'level': current_spread,
        'data_available': True
    }


def detect_macro_quadrant_enhanced(price_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Enhanced macro quadrant with volatility, credit, and real yield overlays.

    Returns:
        quadrant: 1-4 (Goldilocks, Reflation, Stagflation, Risk-Off)
        vix_regime: 'low_vol', 'normal', 'elevated', 'crisis'
        credit_regime: 'tight', 'normal', 'wide', 'stress'
        real_yield_regime: 'negative', 'low', 'normal', 'high'
        confidence: 0-1 (how clear the regime is)
    """
    # Get base quadrant
    base_quadrant = detect_macro_quadrant(price_data)

    # Get overlay regimes
    vix_regime = detect_vix_regime(price_data)
    real_yield_regime = detect_real_yield_regime(price_data)
    credit_regime = detect_credit_regime(price_data)

    # Calculate confidence based on signal strength
    growth_strength = abs(base_quadrant['growth_signal'])
    inflation_strength = abs(base_quadrant['inflation_signal'])
    base_confidence = min((growth_strength + inflation_strength) / 10, 1.0)

    # Adjust confidence based on overlay consistency
    overlay_signals = []
    if vix_regime['data_available']:
        # VIX should be elevated in risk-off (quad 4), low in goldilocks (quad 1)
        if base_quadrant['quadrant'] == 4 and vix_regime['regime'] in ['elevated', 'crisis']:
            overlay_signals.append(1)
        elif base_quadrant['quadrant'] == 1 and vix_regime['regime'] == 'low_vol':
            overlay_signals.append(1)
        elif base_quadrant['quadrant'] == 4 and vix_regime['regime'] == 'low_vol':
            overlay_signals.append(-1)  # Inconsistent
        else:
            overlay_signals.append(0)

    confidence = base_confidence
    if overlay_signals:
        overlay_adjustment = np.mean(overlay_signals) * 0.2
        confidence = np.clip(confidence + overlay_adjustment, 0.2, 1.0)

    return {
        **base_quadrant,
        'vix_regime': vix_regime['regime'],
        'vix_level': vix_regime.get('level', 20),
        'real_yield_regime': real_yield_regime['regime'],
        'real_yield_level': real_yield_regime.get('level', 1.5),
        'credit_regime': credit_regime['regime'],
        'credit_level': credit_regime.get('level', 400),
        'confidence': round(confidence, 2),
        'overlays_available': {
            'vix': vix_regime['data_available'],
            'real_yield': real_yield_regime['data_available'],
            'credit': credit_regime['data_available'],
        }
    }


def apply_regime_adjustment(
    base_signal: float,
    quadrant: int,
    commodity_type: str
) -> float:
    """
    Adjust signal based on macro quadrant (basic version).
    """
    adjustment = REGIME_ADJUSTMENTS.get(commodity_type, {}).get(quadrant, 0)
    adjusted_signal = base_signal * (1 + adjustment)

    return np.clip(adjusted_signal, -10, 10)


def apply_enhanced_regime_adjustment(
    base_signal: float,
    macro_regime: Dict,
    commodity_type: str
) -> Tuple[float, Dict]:
    """
    Apply multi-layer regime adjustment including VIX, credit, and real yields.

    Returns:
        adjusted_signal: Final adjusted signal
        adjustment_breakdown: Dict showing each adjustment layer
    """
    adjustment_breakdown = {
        'quadrant': 0,
        'vix': 0,
        'real_yield': 0,
        'credit': 0,
        'total': 0
    }

    # Layer 1: Base quadrant adjustment
    quadrant = macro_regime.get('quadrant', 2)
    quadrant_adj = REGIME_ADJUSTMENTS.get(commodity_type, {}).get(quadrant, 0)
    adjustment_breakdown['quadrant'] = quadrant_adj

    # Layer 2: VIX regime adjustment
    vix_regime = macro_regime.get('vix_regime', 'normal')
    vix_adj = VIX_REGIME_ADJUSTMENTS.get(commodity_type, {}).get(vix_regime, 0)
    adjustment_breakdown['vix'] = vix_adj

    # Layer 3: Real yield regime adjustment
    real_yield_regime = macro_regime.get('real_yield_regime', 'normal')
    real_yield_adj = REAL_YIELD_REGIME_ADJUSTMENTS.get(commodity_type, {}).get(real_yield_regime, 0)
    adjustment_breakdown['real_yield'] = real_yield_adj

    # Layer 4: Credit regime adjustment
    credit_regime = macro_regime.get('credit_regime', 'normal')
    credit_adj = CREDIT_REGIME_ADJUSTMENTS.get(commodity_type, {}).get(credit_regime, 0)
    adjustment_breakdown['credit'] = credit_adj

    # Calculate total adjustment
    total_adjustment = quadrant_adj + vix_adj + real_yield_adj + credit_adj
    adjustment_breakdown['total'] = total_adjustment

    # Apply adjustment to signal
    adjusted_signal = base_signal * (1 + total_adjustment)

    return np.clip(adjusted_signal, -10, 10), adjustment_breakdown


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
        as_of_date: datetime = None,
        use_enhanced_regime: bool = True
    ) -> Dict:
        """
        Generate composite signal for a commodity.

        Args:
            commodity_ticker: Bloomberg-style ticker
            as_of_date: Date for signal (default: latest)
            use_enhanced_regime: If True, use multi-layer regime adjustment
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
        # Check in expanded universes first
        commodity_info = COMMODITY_UNIVERSE.get(commodity_ticker, {})
        if not commodity_info:
            commodity_info = INDUSTRIAL_METALS_UNIVERSE.get(commodity_ticker, {})
        if not commodity_info:
            commodity_info = ENERGY_UNIVERSE.get(commodity_ticker, {})
        if not commodity_info:
            commodity_info = AGRICULTURAL_UNIVERSE.get(commodity_ticker, {})

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
        # Use enhanced or basic regime detection
        if use_enhanced_regime:
            macro_regime = detect_macro_quadrant_enhanced(price_data)
        else:
            macro_regime = detect_macro_quadrant(price_data)

        # Base signal before regime adjustment
        base_signal = sentiment_weighted + correlation_weighted + divergence_weighted

        # Apply regime adjustment (enhanced or basic)
        if use_enhanced_regime:
            adjusted_signal, regime_breakdown = apply_enhanced_regime_adjustment(
                base_signal,
                macro_regime,
                commodity_type
            )
            regime_contribution = adjusted_signal - base_signal
        else:
            adjusted_signal = apply_regime_adjustment(
                base_signal,
                macro_regime['quadrant'],
                commodity_type
            )
            regime_contribution = adjusted_signal - base_signal
            regime_breakdown = {
                'quadrant': regime_contribution,
                'vix': 0,
                'real_yield': 0,
                'credit': 0,
                'total': regime_contribution
            }

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
                'regime_breakdown': {
                    k: round(v, 3) for k, v in regime_breakdown.items()
                },
            },
            'macro_regime': macro_regime,
            'divergence_details': divergence_result,
            'commodity_type': commodity_type,
            'timestamp': datetime.now().isoformat(),
        }

    def generate_all_signals(self) -> Dict[str, Dict]:
        """
        Generate signals for all commodities in universe.
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
        current_position: Optional[str] = None
    ) -> Dict:
        """
        Generate entry signal with position management logic.
        """
        signal_result = self.generate_signal(commodity_ticker)

        signal = signal_result['signal']
        classification = signal_result['classification']

        # Get correlation regime
        key_correlations = KEY_CORRELATIONS.get(commodity_ticker, [])
        regime_check = 'normal'

        for comp_ticker in key_correlations[:2]:
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


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def print_signal_report(signals: Dict[str, Dict]):
    """Print formatted signal report for all commodities."""
    print("\n" + "="*110)
    print(f"{'COMMODITY TRADING SIGNALS (CSV Version - Enhanced)':^110}")
    print(f"{'Sentiment (40%) + Correlation (25%) + Divergence (20%) + Regime (15%)':^110}")
    print("="*110 + "\n")

    # Get macro regime (same for all)
    first_signal = list(signals.values())[0]
    macro = first_signal.get('macro_regime', {})

    print(f"MACRO ENVIRONMENT: {macro.get('name', 'Unknown')} (Quad {macro.get('quadrant', 0)})")
    print(f"  Growth: {macro.get('growth_signal', 0):+.2f} ({'Rising' if macro.get('growth_rising') else 'Falling'})")
    print(f"  Inflation: {macro.get('inflation_signal', 0):+.2f} ({'Rising' if macro.get('inflation_rising') else 'Falling'})")

    # Enhanced regime overlays if available
    if 'vix_regime' in macro:
        print(f"\nREGIME OVERLAYS:")
        vix_level = macro.get('vix_level', 'N/A')
        vix_level_str = f"{vix_level:.1f}" if isinstance(vix_level, (int, float)) else vix_level
        print(f"  VIX: {macro.get('vix_regime', 'N/A')} (Level: {vix_level_str})")

        if macro.get('overlays_available', {}).get('real_yield'):
            ry_level = macro.get('real_yield_level', 'N/A')
            ry_level_str = f"{ry_level:.2f}%" if isinstance(ry_level, (int, float)) else ry_level
            print(f"  Real Yield: {macro.get('real_yield_regime', 'N/A')} (10Y TIPS: {ry_level_str})")

        if macro.get('overlays_available', {}).get('credit'):
            credit_level = macro.get('credit_level', 'N/A')
            credit_level_str = f"{credit_level:.0f}bps" if isinstance(credit_level, (int, float)) else credit_level
            print(f"  Credit: {macro.get('credit_regime', 'N/A')} (HY OAS: {credit_level_str})")

        print(f"  Confidence: {macro.get('confidence', 0):.0%}")
    print()

    # Header
    print(f"{'Commodity':<12} {'Signal':>7} {'Class':<12} {'Action':<20} {'Sent':>5} {'Corr':>5} {'Div':>5} {'Quad':>5} {'VIX':>5} {'RY':>5} {'Cr':>5}")
    print("-"*110)

    # Sort by signal (descending)
    sorted_signals = sorted(
        signals.items(),
        key=lambda x: x[1].get('signal', 0),
        reverse=True
    )

    for ticker, data in sorted_signals:
        # Look up name in all universes
        name = COMMODITY_UNIVERSE.get(ticker, {}).get('name')
        if not name:
            name = INDUSTRIAL_METALS_UNIVERSE.get(ticker, {}).get('name')
        if not name:
            name = ENERGY_UNIVERSE.get(ticker, {}).get('name')
        if not name:
            name = AGRICULTURAL_UNIVERSE.get(ticker, {}).get('name', ticker)

        signal = data.get('signal', 0)
        classification = data.get('classification', 'N/A')[:12]
        action = data.get('action', 'N/A')[:20]

        components = data.get('components', {})
        sentiment = components.get('sentiment', 0)
        correlation = components.get('correlation', 0)
        divergence = components.get('divergence', 0)

        # Regime breakdown
        regime_breakdown = components.get('regime_breakdown', {})
        quad_adj = regime_breakdown.get('quadrant', 0)
        vix_adj = regime_breakdown.get('vix', 0)
        ry_adj = regime_breakdown.get('real_yield', 0)
        cr_adj = regime_breakdown.get('credit', 0)

        print(f"{name:<12} {signal:>7.2f} {classification:<12} {action:<20} "
              f"{sentiment:>5.1f} {correlation:>5.1f} {divergence:>5.1f} "
              f"{quad_adj:>5.2f} {vix_adj:>5.2f} {ry_adj:>5.2f} {cr_adj:>5.2f}")

    print("="*110)
    print(f"\nSignal Interpretation:")
    print(f"  >= 7.0: Very Bullish (Strong Buy)    |  Regime columns: Quad=Quadrant, VIX=Volatility,")
    print(f"  >= 4.0: Bullish (Buy)                |  RY=Real Yield, Cr=Credit adjustments")
    print(f"  -4.0 to 4.0: Neutral (No Position)   |")
    print(f"  <= -4.0: Bearish (Sell)              |")
    print(f"  <= -7.0: Very Bearish (Strong Sell)  |")
    print("="*110 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("Commodity Signal Generator (CSV Version) - Testing")
    print("="*50)

    # Initialize generator
    generator = CommoditySignalGenerator(lookback_days=120)

    # Generate all signals
    print("\nGenerating signals...")
    signals = generator.generate_all_signals()

    # Print report
    print_signal_report(signals)

    # Test entry signal for Gold
    print("\n" + "="*50)
    print("Entry Signal Test - Gold")
    print("="*50)

    entry = generator.generate_entry_signal('GCA Comdty')
    print(f"Entry Signal: {entry['entry_signal']}")
    print(f"Direction: {entry['entry_direction']}")
    print(f"Confidence: {entry['confidence']}")
    print(f"Reason: {entry['reason']}")
