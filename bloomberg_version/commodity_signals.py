#!/usr/bin/env python3
"""
Commodity Trading Signal Generator
Generates buy/sell signals for Gold, Silver, Platinum, Palladium, and Oil

Signal Components:
- Sentiment Score (CCS): 35% - Momentum/trend analysis
- Correlation Alignment: 22% - Cross-asset confirmation
- Divergence Detection: 18% - Mean-reversion opportunities
- Options Flow: 12% - Call/Put ratio, IV analysis
- Regime Adjustment: 13% - Macro context

Signal Range: -10 (Very Bearish) to +10 (Very Bullish)
"""

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

# ============================================================================
# CONFIGURATION
# ============================================================================

# Signal component weights (updated to include options flow)
SIGNAL_WEIGHTS = {
    'sentiment': 0.35,       # CCS momentum/trend score (was 40%)
    'correlation': 0.22,     # Cross-asset alignment (was 25%)
    'divergence': 0.18,      # Divergence detection (was 20%)
    'regime': 0.13,          # Macro regime adjustment (was 15%)
    'options': 0.12,         # Options flow signal (NEW)
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
        'low_vol': -0.15, 'normal': 0.00, 'elevated': 0.20, 'crisis': 0.40,
    },
    'energy': {
        'low_vol': 0.10, 'normal': 0.00, 'elevated': -0.10, 'crisis': -0.25,
    },
    'industrial_metal': {
        'low_vol': 0.10, 'normal': 0.00, 'elevated': -0.15, 'crisis': -0.30,
    },
    'agricultural': {
        'low_vol': 0.00, 'normal': 0.00, 'elevated': 0.05, 'crisis': 0.10,
    },
}

# Real yield regime adjustments
REAL_YIELD_REGIME_ADJUSTMENTS = {
    'precious_metal': {
        'negative': 0.25, 'low': 0.10, 'normal': 0.00, 'high': -0.20,
    },
    'energy': {
        'negative': 0.05, 'low': 0.05, 'normal': 0.00, 'high': -0.05,
    },
    'industrial_metal': {
        'negative': 0.05, 'low': 0.05, 'normal': 0.00, 'high': 0.05,
    },
    'agricultural': {
        'negative': 0.10, 'low': 0.05, 'normal': 0.00, 'high': -0.05,
    },
}

# Credit regime adjustments
CREDIT_REGIME_ADJUSTMENTS = {
    'precious_metal': {
        'tight': -0.10, 'normal': 0.00, 'wide': 0.15, 'stress': 0.25,
    },
    'energy': {
        'tight': 0.10, 'normal': 0.00, 'wide': -0.15, 'stress': -0.25,
    },
    'industrial_metal': {
        'tight': 0.15, 'normal': 0.00, 'wide': -0.15, 'stress': -0.30,
    },
    'agricultural': {
        'tight': 0.00, 'normal': 0.00, 'wide': 0.05, 'stress': 0.10,
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
# OPTIONS DATA FETCHING & SCORING
# ============================================================================

# Initialize BQL service
bq = bql.Service()


def is_futures_instrument(ticker: str) -> bool:
    """Check if instrument is futures/commodities"""
    return 'Comdty' in ticker or ('Index' in ticker and 'VIX' not in ticker)


def is_equity_instrument(ticker: str) -> bool:
    """Check if instrument is equity (stock/ETF)"""
    return 'Equity' in ticker


def fetch_options_data(ticker: str) -> Optional[Dict]:
    """
    Fetch options flow data using BQL.

    Returns dict with:
    - implied_volatility: Current IV
    - call_put_ratio: Call IV / Put IV (>1 = bullish, <1 = bearish)
    - iv_percentile: IV rank vs historical (for mean-reversion)
    - put_call_skew: Put IV - Call IV (positive = fear premium)

    For commodities, uses futures options data.
    For equities/ETFs, uses 60-day IV data.
    """
    # Only fetch for instruments that have options
    if not (is_equity_instrument(ticker) or is_futures_instrument(ticker)):
        return None

    try:
        is_futures = is_futures_instrument(ticker)

        if is_futures:
            # Futures use different IV fields
            request = bql.Request(
                ticker,
                {
                    'Implied_Volatility': bq.data.IMPLIED_VOLATILITY(),
                    'Fut_Call_IV': bq.data.FUT_CALL_IMPLIED_VOLATILITY(),
                    'Fut_Put_IV': bq.data.FUT_PUT_IMPLIED_VOLATILITY(),
                    'IV_30D_High': bq.data.IMPLIED_VOLATILITY_30D_HIGH(),
                    'IV_30D_Low': bq.data.IMPLIED_VOLATILITY_30D_LOW(),
                }
            )
        else:
            # Equities use 60-day call/put IV
            request = bql.Request(
                ticker,
                {
                    'Implied_Volatility': bq.data.IMPLIED_VOLATILITY(),
                    'Call_IV_60D': bq.data.CALL_IMP_VOL_60D(),
                    'Put_IV_60D': bq.data.PUT_IMP_VOL_60D(),
                    'IV_30D_High': bq.data.IMPLIED_VOLATILITY_30D_HIGH(),
                    'IV_30D_Low': bq.data.IMPLIED_VOLATILITY_30D_LOW(),
                }
            )

        response = bq.execute(request)
        data = pd.concat([data_item.df() for data_item in response], axis=1)

        # Extract implied volatility
        implied_vol = data['Implied_Volatility'].iloc[-1] if 'Implied_Volatility' in data.columns else None

        # Calculate call/put ratio
        if is_futures:
            call_iv = data['Fut_Call_IV'].iloc[-1] if 'Fut_Call_IV' in data.columns else None
            put_iv = data['Fut_Put_IV'].iloc[-1] if 'Fut_Put_IV' in data.columns else None
        else:
            call_iv = data['Call_IV_60D'].iloc[-1] if 'Call_IV_60D' in data.columns else None
            put_iv = data['Put_IV_60D'].iloc[-1] if 'Put_IV_60D' in data.columns else None

        # Call/Put ratio
        if call_iv and put_iv and put_iv > 0:
            call_put_ratio = call_iv / put_iv
            put_call_skew = put_iv - call_iv  # Positive = fear premium
        else:
            call_put_ratio = 1.0
            put_call_skew = 0.0

        # IV percentile (where current IV sits in 30-day range)
        iv_high = data['IV_30D_High'].iloc[-1] if 'IV_30D_High' in data.columns else None
        iv_low = data['IV_30D_Low'].iloc[-1] if 'IV_30D_Low' in data.columns else None

        if iv_high and iv_low and implied_vol and (iv_high - iv_low) > 0:
            iv_percentile = (implied_vol - iv_low) / (iv_high - iv_low) * 100
        else:
            iv_percentile = 50.0  # Default to middle

        return {
            'implied_volatility': implied_vol if implied_vol else 25.0,
            'call_put_ratio': call_put_ratio,
            'put_call_skew': put_call_skew,
            'iv_percentile': iv_percentile,
            'call_iv': call_iv,
            'put_iv': put_iv,
            'data_available': True,
        }

    except Exception as e:
        print(f"Warning: Could not fetch options data for {ticker}: {e}")
        return None


def calculate_options_flow_score(
    options_data: Optional[Dict],
    price_trend: float,
    commodity_type: str = 'precious_metal'
) -> Dict:
    """
    Calculate options flow score for signal generation.

    Components:
    1. Call/Put Ratio (40%): Directional bias from options market
    2. IV Level (25%): High IV in uptrend = momentum, in downtrend = capitulation
    3. Put/Call Skew (20%): Fear premium in puts
    4. IV Percentile (15%): Mean-reversion signal

    Args:
        options_data: Dict from fetch_options_data()
        price_trend: Recent price trend (e.g., 10-day ROC)
        commodity_type: Type of commodity for context-specific scoring

    Returns:
        Dict with score and component breakdown
    """
    if options_data is None or not options_data.get('data_available', False):
        return {
            'score': 0.0,
            'components': {},
            'data_available': False,
        }

    # === CALL/PUT RATIO (40%) ===
    # Higher ratio = more call buying = bullish
    cp_ratio = options_data.get('call_put_ratio', 1.0)

    if cp_ratio > 1.5:
        cp_score = 10  # Very bullish (extreme call buying)
    elif cp_ratio > 1.2:
        cp_score = 5   # Bullish
    elif cp_ratio > 0.8:
        cp_score = 0   # Neutral
    elif cp_ratio > 0.6:
        cp_score = -5  # Bearish
    else:
        cp_score = -10  # Very bearish (extreme put buying)

    cp_weighted = normalize_to_range(cp_score) * 0.40

    # === IV LEVEL (25%) ===
    # Context-dependent: High IV in uptrend = conviction, in downtrend = panic
    implied_vol = options_data.get('implied_volatility', 25.0)

    # Commodity-specific IV thresholds
    iv_thresholds = {
        'precious_metal': {'low': 12, 'normal': 18, 'high': 28, 'extreme': 40},
        'energy': {'low': 20, 'normal': 35, 'high': 50, 'extreme': 70},
        'industrial_metal': {'low': 15, 'normal': 25, 'high': 40, 'extreme': 55},
        'agricultural': {'low': 15, 'normal': 25, 'high': 38, 'extreme': 50},
    }

    thresholds = iv_thresholds.get(commodity_type, iv_thresholds['precious_metal'])

    # Determine IV regime
    if implied_vol <= thresholds['low']:
        iv_regime = 'low'
        iv_base = -3  # Low IV = complacency, potential for vol expansion
    elif implied_vol <= thresholds['normal']:
        iv_regime = 'normal'
        iv_base = 0
    elif implied_vol <= thresholds['high']:
        iv_regime = 'high'
        iv_base = 2 if price_trend > 0 else -2
    else:
        iv_regime = 'extreme'
        # Extreme IV: bullish in uptrend (conviction), potential reversal in downtrend
        iv_base = 5 if price_trend > 0 else 3  # Capitulation can signal bottom

    iv_weighted = normalize_to_range(iv_base) * 0.25

    # === PUT/CALL SKEW (20%) ===
    # Positive skew = fear premium in puts
    skew = options_data.get('put_call_skew', 0.0)

    if skew > 5:
        skew_score = -5  # High fear premium = bearish near-term
    elif skew > 2:
        skew_score = -2
    elif skew > -2:
        skew_score = 0  # Balanced
    elif skew > -5:
        skew_score = 2  # Call premium = bullish sentiment
    else:
        skew_score = 5  # Extreme call premium

    skew_weighted = normalize_to_range(skew_score) * 0.20

    # === IV PERCENTILE (15%) ===
    # Mean-reversion signal: extreme low = vol expansion likely, extreme high = contraction
    iv_pct = options_data.get('iv_percentile', 50.0)

    if iv_pct <= 10:
        pct_score = 3 if price_trend > 0 else -2  # Low IV, uptrend = breakout potential
    elif iv_pct <= 30:
        pct_score = 1
    elif iv_pct >= 90:
        pct_score = -3 if price_trend < 0 else 2  # High IV exhaustion
    elif iv_pct >= 70:
        pct_score = -1
    else:
        pct_score = 0

    pct_weighted = normalize_to_range(pct_score) * 0.15

    # === TOTAL OPTIONS SCORE ===
    total_score = cp_weighted + iv_weighted + skew_weighted + pct_weighted

    return {
        'score': float(np.clip(total_score, -10, 10)),
        'components': {
            'call_put_ratio': {'value': round(cp_ratio, 3), 'score': round(cp_weighted, 2)},
            'iv_level': {'value': round(implied_vol, 1), 'regime': iv_regime, 'score': round(iv_weighted, 2)},
            'put_call_skew': {'value': round(skew, 2), 'score': round(skew_weighted, 2)},
            'iv_percentile': {'value': round(iv_pct, 1), 'score': round(pct_weighted, 2)},
        },
        'data_available': True,
    }


# ============================================================================
# SENTIMENT SCORE CALCULATION (from existing CCS)
# ============================================================================

def fetch_historical_data(ticker: str, days: int = 180) -> pd.DataFrame:
    """
    Fetch historical OHLCV data using BQL.
    Note: Default increased to 180 days to support ROC_120 calculation.
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
    # Longer periods better suited for macro-driven commodity analysis
    df['ROC_5'] = df['Close'].pct_change(5) * 100      # Weekly momentum
    df['ROC_10'] = df['Close'].pct_change(10) * 100    # Bi-weekly trend
    df['ROC_20'] = df['Close'].pct_change(20) * 100    # Monthly trend
    df['ROC_60'] = df['Close'].pct_change(60) * 100    # Quarterly - captures macro shifts
    df['ROC_120'] = df['Close'].pct_change(120) * 100  # 6-month trend confirmation

    # Moving Averages (longer periods for macro-driven commodities)
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_100'] = df['Close'].rolling(window=100).mean()

    # MA Slopes (trend direction) - adjusted for new MAs
    df['MA_10_slope'] = df['MA_10'].pct_change(5) * 100
    df['MA_20_slope'] = df['MA_20'].pct_change(10) * 100
    df['MA_50_slope'] = df['MA_50'].pct_change(20) * 100

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
        df['Vol_Price_Corr'] = np.sign(df['ROC_5']) * (df['Vol_Ratio'] - 1)
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
    # Weights adjusted for macro-driven commodity analysis
    roc_score = (
        df_row['ROC_5'] * 0.15 +    # Weekly momentum confirmation
        df_row['ROC_10'] * 0.20 +   # Bi-weekly trend
        df_row['ROC_20'] * 0.30 +   # Monthly trend (primary)
        df_row['ROC_60'] * 0.25 +   # Quarterly macro shifts
        df_row['ROC_120'] * 0.10    # 6-month trend confirmation
    )
    roc_normalized = normalize_to_range(roc_score) * 0.35

    # === MA TREND SIGNALS (25%) ===
    # Adjusted for longer MAs suited to commodity macro analysis
    ma_slope_score = (
        df_row['MA_10_slope'] * 0.25 +
        df_row['MA_20_slope'] * 0.35 +
        df_row['MA_50_slope'] * 0.40
    )

    # MA crossover signals (using 10/20/50 alignment)
    ma_cross_score = 0
    if df_row['MA_10'] > df_row['MA_20'] > df_row['MA_50']:
        ma_cross_score = 5  # Bullish alignment
    elif df_row['MA_10'] < df_row['MA_20'] < df_row['MA_50']:
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
    # Using 60-day lookback for more stable regime classification
    growth_tickers = ['ESA Index', 'NQA Index', 'RTYA Index']  # Equities
    cyclical_tickers = ['HGA Comdty', 'CLA Comdty']  # Copper, Oil

    equity_returns = []
    for ticker in growth_tickers:
        if ticker in price_data and price_data[ticker] is not None:
            df = price_data[ticker]
            if len(df) >= 60:
                ret = df['Close'].pct_change(60).iloc[-1] * 100
                equity_returns.append(ret)

    cyclical_returns = []
    for ticker in cyclical_tickers:
        if ticker in price_data and price_data[ticker] is not None:
            df = price_data[ticker]
            if len(df) >= 60:
                ret = df['Close'].pct_change(60).iloc[-1] * 100
                cyclical_returns.append(ret)

    equity_signal = np.mean(equity_returns) if equity_returns else 0
    cyclical_signal = np.mean(cyclical_returns) if cyclical_returns else 0
    growth_signal = equity_signal * 0.7 + cyclical_signal * 0.3
    growth_rising = growth_signal > 0

    # === INFLATION SIGNAL ===
    # Bond prices: falling bonds = rising yields = rising inflation
    # NOTE: Removed gold/silver from inflation hedges to avoid circular logic
    # (gold signal was being influenced by gold's own price movement)
    bond_tickers = ['TUA Comdty', 'FVA Comdty', 'TYA Comdty', 'USA Comdty']
    inflation_hedge_tickers = ['CLA Comdty', 'HGA Comdty']  # Oil and Copper only

    bond_returns = []
    for ticker in bond_tickers:
        if ticker in price_data and price_data[ticker] is not None:
            df = price_data[ticker]
            if len(df) >= 60:
                ret = df['Close'].pct_change(60).iloc[-1] * 100
                bond_returns.append(ret)

    inflation_returns = []
    for ticker in inflation_hedge_tickers:
        if ticker in price_data and price_data[ticker] is not None:
            df = price_data[ticker]
            if len(df) >= 60:
                ret = df['Close'].pct_change(60).iloc[-1] * 100
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


def detect_vix_regime(price_data: Dict[str, pd.DataFrame]) -> Dict:
    """Determine VIX regime for volatility overlay."""
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

    if current_vix < 15:
        regime = 'low_vol'
    elif current_vix < 22:
        regime = 'normal'
    elif current_vix < 35:
        regime = 'elevated'
    else:
        regime = 'crisis'

    return {'regime': regime, 'level': current_vix, 'zscore': zscore, 'data_available': True}


def detect_real_yield_regime(price_data: Dict[str, pd.DataFrame]) -> Dict:
    """Determine real yield regime for rate environment overlay."""
    tips_ticker = 'H15T10YIE Index'
    if tips_ticker not in price_data or price_data[tips_ticker] is None:
        return {'regime': 'normal', 'level': 1.5, 'data_available': False}

    df = price_data[tips_ticker]
    if len(df) < 20:
        return {'regime': 'normal', 'level': 1.5, 'data_available': False}

    current_yield = df['Close'].iloc[-1]
    if current_yield < 0:
        regime = 'negative'
    elif current_yield < 1.0:
        regime = 'low'
    elif current_yield < 2.0:
        regime = 'normal'
    else:
        regime = 'high'

    return {'regime': regime, 'level': current_yield, 'data_available': True}


def detect_credit_regime(price_data: Dict[str, pd.DataFrame]) -> Dict:
    """Determine credit regime for risk appetite overlay."""
    hy_ticker = 'LF98OAS Index'
    if hy_ticker not in price_data or price_data[hy_ticker] is None:
        return {'regime': 'normal', 'level': 400, 'data_available': False}

    df = price_data[hy_ticker]
    if len(df) < 20:
        return {'regime': 'normal', 'level': 400, 'data_available': False}

    current_spread = df['Close'].iloc[-1]
    if current_spread < 350:
        regime = 'tight'
    elif current_spread < 500:
        regime = 'normal'
    elif current_spread < 700:
        regime = 'wide'
    else:
        regime = 'stress'

    return {'regime': regime, 'level': current_spread, 'data_available': True}


def detect_macro_quadrant_enhanced(price_data: Dict[str, pd.DataFrame]) -> Dict:
    """Enhanced macro quadrant with volatility, credit, and real yield overlays."""
    base_quadrant = detect_macro_quadrant(price_data)
    vix_regime = detect_vix_regime(price_data)
    real_yield_regime = detect_real_yield_regime(price_data)
    credit_regime = detect_credit_regime(price_data)

    growth_strength = abs(base_quadrant['growth_signal'])
    inflation_strength = abs(base_quadrant['inflation_signal'])
    confidence = min((growth_strength + inflation_strength) / 10, 1.0)

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
    """Apply multi-layer regime adjustment including VIX, credit, and real yields."""
    adjustment_breakdown = {'quadrant': 0, 'vix': 0, 'real_yield': 0, 'credit': 0, 'total': 0}

    quadrant = macro_regime.get('quadrant', 2)
    quadrant_adj = REGIME_ADJUSTMENTS.get(commodity_type, {}).get(quadrant, 0)
    adjustment_breakdown['quadrant'] = quadrant_adj

    vix_regime = macro_regime.get('vix_regime', 'normal')
    vix_adj = VIX_REGIME_ADJUSTMENTS.get(commodity_type, {}).get(vix_regime, 0)
    adjustment_breakdown['vix'] = vix_adj

    real_yield_regime = macro_regime.get('real_yield_regime', 'normal')
    real_yield_adj = REAL_YIELD_REGIME_ADJUSTMENTS.get(commodity_type, {}).get(real_yield_regime, 0)
    adjustment_breakdown['real_yield'] = real_yield_adj

    credit_regime = macro_regime.get('credit_regime', 'normal')
    credit_adj = CREDIT_REGIME_ADJUSTMENTS.get(commodity_type, {}).get(credit_regime, 0)
    adjustment_breakdown['credit'] = credit_adj

    total_adjustment = quadrant_adj + vix_adj + real_yield_adj + credit_adj
    adjustment_breakdown['total'] = total_adjustment

    adjusted_signal = base_signal * (1 + total_adjustment)
    return np.clip(adjusted_signal, -10, 10), adjustment_breakdown


# ============================================================================
# GOLD-SPECIFIC REGIME DETECTOR
# ============================================================================
# Addresses gold's non-traditional drivers: central bank buying, de-dollarization,
# ETF flows, and correlation breakdowns that standard macro frameworks miss.

def get_asset_return(price_data: Dict, ticker: str, lookback: int) -> float:
    """
    Helper to get percentage return for a ticker over lookback period.

    Args:
        price_data: Dict of price DataFrames
        ticker: Bloomberg ticker
        lookback: Number of days for return calculation

    Returns:
        Percentage return, or 0 if data unavailable
    """
    if ticker not in price_data or price_data[ticker] is None:
        return 0.0
    df = price_data[ticker]
    if len(df) < lookback:
        return 0.0
    return (df['Close'].iloc[-1] / df['Close'].iloc[-lookback] - 1) * 100


def calculate_etf_flow_signal(price_data: Dict, lookback_short: int = 20, lookback_long: int = 60) -> Dict:
    """
    Calculate gold ETF flow signal based on price/AUM momentum.

    In production, would use EQY_SH_OUT field for shares outstanding.
    Here we use price as a proxy since ETF prices track NAV closely.

    Logic:
    - Rising ETF prices with volume = inflows = bullish
    - Strong 20d move confirmed by 60d trend = high conviction

    Scoring:
    - Strong inflow (>2% 20d): +5
    - Moderate inflow (0.5-2%): +2
    - Neutral (-0.5% to 0.5%): 0
    - Moderate outflow (-2% to -0.5%): -2
    - Strong outflow (<-2%): -5

    Args:
        price_data: Dict of price DataFrames
        lookback_short: Short-term lookback (default 20 days)
        lookback_long: Long-term lookback (default 60 days)

    Returns:
        Dict with signal, flow metrics, and data availability
    """
    etf_tickers = ['GLD US Equity', 'IAU US Equity']

    total_flow_short = 0.0
    total_flow_long = 0.0
    count = 0

    for ticker in etf_tickers:
        if ticker in price_data and price_data[ticker] is not None:
            df = price_data[ticker]
            if len(df) >= lookback_long:
                flow_short = (df['Close'].iloc[-1] / df['Close'].iloc[-lookback_short] - 1) * 100
                flow_long = (df['Close'].iloc[-1] / df['Close'].iloc[-lookback_long] - 1) * 100
                total_flow_short += flow_short
                total_flow_long += flow_long
                count += 1

    if count == 0:
        return {'signal': 0, 'flow_20d': 0, 'flow_60d': 0, 'data_available': False}

    avg_flow_short = total_flow_short / count
    avg_flow_long = total_flow_long / count

    # Score based on short-term flow
    if avg_flow_short > 2:
        base_score = 5
    elif avg_flow_short > 0.5:
        base_score = 2
    elif avg_flow_short > -0.5:
        base_score = 0
    elif avg_flow_short > -2:
        base_score = -2
    else:
        base_score = -5

    # Trend confirmation adjustment
    trend_confirms = (avg_flow_short > 0 and avg_flow_long > 0) or \
                     (avg_flow_short < 0 and avg_flow_long < 0)

    if trend_confirms:
        final_score = base_score * 1.5
    else:
        final_score = base_score * 0.5

    return {
        'signal': float(np.clip(final_score, -10, 10)),
        'flow_20d': round(avg_flow_short, 2),
        'flow_60d': round(avg_flow_long, 2),
        'trend_confirms': trend_confirms,
        'data_available': True
    }


def calculate_correlation_breakdown_score(
    price_data: Dict,
    correlation_engine,
    commodity_ticker: str = 'GCA Comdty'
) -> Dict:
    """
    Detect when gold's traditional correlations have broken down.

    Key Relationships:
    - Gold/DXY: Expected -0.5, breakdown if > -0.2
    - Gold/Real Yields: Expected -0.7, breakdown if > -0.3
    - Gold/10Y Treasury: Expected -0.3, breakdown if > 0.0

    Interpretation:
    - Breakdown + gold rising = structural demand (very bullish)
    - Breakdown + gold falling = concerning weakness
    - No breakdown = traditional framework applies

    Args:
        price_data: Dict of price DataFrames
        correlation_engine: CorrelationEngine instance
        commodity_ticker: Gold ticker (default GCA Comdty)

    Returns:
        Dict with signal, regime classification, and breakdown details
    """
    key_correlations = {
        'DXY Index': {'expected': -0.5, 'breakdown_threshold': -0.2},
        'H15T10YIE Index': {'expected': -0.7, 'breakdown_threshold': -0.3},
        'TYA Comdty': {'expected': -0.3, 'breakdown_threshold': 0.0},
    }

    breakdown_count = 0
    breakdown_details = {}

    for ticker, params in key_correlations.items():
        try:
            result = correlation_engine.detect_correlation_regime(
                commodity_ticker, ticker
            )

            current_corr = result.get('current_corr', 0)
            expected_corr = params['expected']
            threshold = params['breakdown_threshold']

            # For inverse correlations, breakdown = correlation moved toward positive
            if expected_corr < 0:
                is_breakdown = current_corr > threshold
            else:
                is_breakdown = abs(current_corr - expected_corr) > 0.3

            if is_breakdown:
                breakdown_count += 1
                breakdown_details[ticker] = {
                    'current': round(current_corr, 3),
                    'expected': expected_corr,
                    'deviation': round(current_corr - expected_corr, 3)
                }
        except Exception:
            continue

    # Get gold's recent trend
    gold_trend = get_asset_return(price_data, commodity_ticker, 20)

    # Scoring logic
    if breakdown_count == 0:
        regime = 'traditional'
        score = 0
    elif breakdown_count >= 2:
        regime = 'structural'
        if gold_trend > 0:
            # Gold rising despite correlation breakdown = very bullish
            score = 5 + (breakdown_count * 1.5)
        else:
            # Gold falling during breakdown = concerning
            score = -3
    else:
        regime = 'transitional'
        score = 2 if gold_trend > 0 else -1

    return {
        'signal': float(np.clip(score, -10, 10)),
        'regime': regime,
        'breakdown_count': breakdown_count,
        'breakdown_details': breakdown_details,
        'gold_trend': round(gold_trend, 2),
        'data_available': True
    }


def calculate_safe_haven_index(price_data: Dict, lookback: int = 20) -> Dict:
    """
    Calculate safe haven demand index for gold.

    Distinguishes fear-driven demand from structural demand by comparing
    gold's behavior to VIX, JPY, and Treasuries.

    Interpretation:
    - Gold up + VIX flat = STRUCTURAL (non-fear driven, most bullish)
    - Gold up + VIX up = FEAR-DRIVEN (bullish but may reverse)
    - Gold flat + VIX up = Gold lagging (neutral)
    - Gold down + VIX down = Risk-on (bearish for gold)

    Args:
        price_data: Dict of price DataFrames
        lookback: Days for return calculation (default 20)

    Returns:
        Dict with signal, regime classification, and components
    """
    gold_return = get_asset_return(price_data, 'GCA Comdty', lookback)
    vix_change = get_asset_return(price_data, 'VIX Index', lookback)
    usdjpy_return = get_asset_return(price_data, 'USDJPY Curncy', lookback)
    treasury_return = get_asset_return(price_data, 'TYA Comdty', lookback)

    # Gold/VIX divergence score
    if gold_return > 2:  # Gold meaningfully up
        if vix_change < 5:  # VIX not spiking
            gold_vix_score = 4  # Structural demand
        else:
            gold_vix_score = 2  # Fear-driven
    elif gold_return < -2:  # Gold meaningfully down
        if vix_change > 10:
            gold_vix_score = -2  # Unusual - gold should rise with VIX
        else:
            gold_vix_score = -3  # Normal risk-on behavior
    else:
        gold_vix_score = 0

    # JPY score (negative USDJPY return = JPY strength = safe haven flow)
    jpy_score = 2 if usdjpy_return < -1 else (-1 if usdjpy_return > 1 else 0)

    # Treasury score (positive return = flight to quality)
    treasury_score = 2 if treasury_return > 1 else (-1 if treasury_return < -1 else 0)

    # Gold participation in safe haven flow
    other_havens_bid = usdjpy_return < 0 or treasury_return > 0
    if gold_return > 0 and other_havens_bid:
        participation_score = 2  # Gold participating in safe haven bid
    elif gold_return < 0 and other_havens_bid:
        participation_score = -2  # Gold NOT participating (structural weakness)
    else:
        participation_score = 0

    total_score = gold_vix_score + jpy_score + treasury_score + participation_score

    # Classify regime
    if total_score > 5:
        regime = 'structural_demand'
    elif total_score > 2:
        regime = 'fear_driven'
    elif total_score > -2:
        regime = 'neutral'
    else:
        regime = 'risk_on_headwind'

    return {
        'signal': float(np.clip(total_score, -10, 10)),
        'regime': regime,
        'components': {
            'gold_vix_score': gold_vix_score,
            'jpy_score': jpy_score,
            'treasury_score': treasury_score,
            'participation_score': participation_score,
        },
        'gold_return': round(gold_return, 2),
        'vix_change': round(vix_change, 2),
        'data_available': True
    }


def calculate_dedollarization_proxy(price_data: Dict, lookback: int = 60) -> Dict:
    """
    Calculate de-dollarization / central bank demand proxy.

    Key Signal:
    - Gold rising DESPITE DXY strength = structural demand from price-insensitive
      actors (central banks, sovereign wealth funds)
    - EM currency stress often accompanies central bank gold buying as reserves
      are diversified away from USD

    Args:
        price_data: Dict of price DataFrames
        lookback: Days for return calculation (default 60 for macro view)

    Returns:
        Dict with signal, regime classification, and metrics
    """
    gold_return = get_asset_return(price_data, 'GCA Comdty', lookback)
    dxy_return = get_asset_return(price_data, 'DXY Index', lookback)

    # EM currency stress (positive return = USD strength = EM weakness)
    em_tickers = ['USDZAR Curncy', 'USDBRL Curncy', 'USDMXN Curncy']
    em_stress = 0.0
    em_count = 0
    for ticker in em_tickers:
        ret = get_asset_return(price_data, ticker, lookback)
        if ret != 0:
            em_stress += ret
            em_count += 1

    avg_em_stress = em_stress / em_count if em_count > 0 else 0

    # Gold/DXY divergence score
    if gold_return > 5:
        if dxy_return > 0:
            divergence_score = 8  # Extreme structural demand
        elif dxy_return > -2:
            divergence_score = 5  # Strong structural demand
        else:
            divergence_score = 2  # Normal inverse relationship
    elif gold_return > 2:
        if dxy_return > 0:
            divergence_score = 4  # Moderate structural demand
        else:
            divergence_score = 1  # Normal
    elif gold_return < -2:
        if dxy_return < -2:
            divergence_score = -4  # Gold weak despite dollar weakness (very bearish)
        else:
            divergence_score = -2  # Normal inverse
    else:
        divergence_score = 0

    # EM stress contribution
    if avg_em_stress > 5 and gold_return > 2:
        em_contribution = 3  # Central banks likely buying
    elif avg_em_stress > 5 and gold_return < 0:
        em_contribution = -2  # EM selling gold? Concerning
    else:
        em_contribution = 0

    total_score = divergence_score + em_contribution

    # Classify regime
    if total_score > 6:
        regime = 'strong_structural_demand'
        confidence = 'high'
    elif total_score > 3:
        regime = 'moderate_structural_demand'
        confidence = 'medium'
    elif total_score > -2:
        regime = 'traditional_dynamics'
        confidence = 'medium'
    else:
        regime = 'structural_weakness'
        confidence = 'high'

    return {
        'signal': float(np.clip(total_score, -10, 10)),
        'regime': regime,
        'confidence': confidence,
        'gold_return': round(gold_return, 2),
        'dxy_return': round(dxy_return, 2),
        'em_stress': round(avg_em_stress, 2),
        'divergence_score': divergence_score,
        'em_contribution': em_contribution,
        'data_available': True
    }


def detect_gold_specific_regime(
    price_data: Dict,
    correlation_engine
) -> Dict:
    """
    Master gold regime detection combining all four components.

    Components (25% each):
    1. ETF Flow Signal: Institutional demand via GLD/IAU
    2. Correlation Breakdown: Traditional relationships failing
    3. Safe Haven Index: Fear vs structural demand
    4. De-Dollarization Proxy: Central bank/non-dollar buying

    Output:
    - gold_regime_score: -10 to +10
    - gold_regime: 'traditional', 'transitional', 'structural'
    - framework_confidence: How much to trust standard macro framework
    - recommendation: Signal adjustment recommendation

    Args:
        price_data: Dict of price DataFrames
        correlation_engine: CorrelationEngine instance

    Returns:
        Comprehensive gold regime analysis dict
    """
    # Calculate all components
    etf_flow = calculate_etf_flow_signal(price_data)
    correlation_breakdown = calculate_correlation_breakdown_score(
        price_data, correlation_engine
    )
    safe_haven = calculate_safe_haven_index(price_data)
    dedollarization = calculate_dedollarization_proxy(price_data)

    # Weighted combination (equal weights: 25% each)
    weights = {
        'etf_flow': 0.25,
        'correlation_breakdown': 0.25,
        'safe_haven': 0.25,
        'dedollarization': 0.25,
    }

    total_score = (
        etf_flow['signal'] * weights['etf_flow'] +
        correlation_breakdown['signal'] * weights['correlation_breakdown'] +
        safe_haven['signal'] * weights['safe_haven'] +
        dedollarization['signal'] * weights['dedollarization']
    )

    # Determine overall regime
    breakdown_regime = correlation_breakdown.get('regime', 'traditional')
    dedollar_regime = dedollarization.get('regime', 'traditional_dynamics')

    if breakdown_regime == 'structural' or 'structural' in dedollar_regime:
        gold_regime = 'structural'
        framework_confidence = 0.3  # Low confidence in standard macro
    elif breakdown_regime == 'transitional':
        gold_regime = 'transitional'
        framework_confidence = 0.6  # Moderate confidence
    else:
        gold_regime = 'traditional'
        framework_confidence = 1.0  # Full confidence in standard macro

    # Generate recommendation
    if gold_regime == 'structural' and total_score > 3:
        recommendation = {
            'action': 'override_macro_bearish',
            'description': 'Structural demand strong - override bearish macro signals',
            'signal_adjustment': min(total_score * 0.5, 3),  # Add up to +3 to signal
        }
    elif gold_regime == 'structural' and total_score < -3:
        recommendation = {
            'action': 'confirm_weakness',
            'description': 'Structural indicators confirm weakness',
            'signal_adjustment': max(total_score * 0.3, -2),  # Subtract up to -2
        }
    else:
        recommendation = {
            'action': 'standard_framework',
            'description': 'Use standard macro quadrant framework',
            'signal_adjustment': 0,
        }

    return {
        'gold_regime_score': round(total_score, 2),
        'gold_regime': gold_regime,
        'framework_confidence': framework_confidence,
        'recommendation': recommendation,
        'component_scores': {
            'etf_flow': etf_flow,
            'correlation_breakdown': correlation_breakdown,
            'safe_haven': safe_haven,
            'dedollarization': dedollarization,
        },
        'timestamp': datetime.now().isoformat(),
    }


# ============================================================================
# COMPOSITE SIGNAL GENERATOR
# ============================================================================

class CommoditySignalGenerator:
    """
    Generates composite buy/sell signals for commodities.

    Signal Components:
    - Sentiment (35%): CCS momentum/trend score
    - Correlation (22%): Cross-asset alignment
    - Divergence (18%): Mean-reversion opportunities
    - Regime (13%): Macro regime adjustment
    - Options (12%): Options flow signals (call/put ratio, IV, skew)
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
        self.options_data_cache = {}

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

        # Add gold ETFs for flow analysis
        all_tickers.extend(['GLD US Equity', 'IAU US Equity'])

        # Add EM currencies for de-dollarization proxy
        all_tickers.extend(['USDZAR Curncy', 'USDBRL Curncy', 'USDMXN Curncy'])

        # Add real yield indicators
        all_tickers.extend(['H15T10YIE Index', 'H15T5YIE Index'])

        all_tickers = list(set(all_tickers))  # Remove duplicates

        price_data = {}
        options_data = {}

        for ticker in all_tickers:
            df = fetch_historical_data(ticker, days=self.lookback_days)
            if df is not None and len(df) > 20:
                df = calculate_technical_indicators(df)
                price_data[ticker] = df

                # Fetch options data for commodities and key equities
                if ticker in COMMODITY_UNIVERSE or is_futures_instrument(ticker):
                    opts = fetch_options_data(ticker)
                    if opts is not None:
                        options_data[ticker] = opts

        self.price_data_cache = price_data
        self.options_data_cache = options_data
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

        # === COMPONENT 1: SENTIMENT SCORE (35%) ===
        sentiment_score = calculate_sentiment_score(commodity_df.iloc[-1])
        sentiment_weighted = sentiment_score * SIGNAL_WEIGHTS['sentiment']

        # === COMPONENT 2: CORRELATION ALIGNMENT (22%) ===
        correlation_score = calculate_correlation_alignment_score(
            commodity_ticker,
            self.correlation_engine,
            price_data
        )
        correlation_weighted = correlation_score * SIGNAL_WEIGHTS['correlation']

        # === COMPONENT 3: DIVERGENCE DETECTION (18%) ===
        divergence_result = detect_divergence_signal(
            commodity_ticker,
            self.correlation_engine,
            price_data
        )
        divergence_score = divergence_result['score']
        divergence_weighted = divergence_score * SIGNAL_WEIGHTS['divergence']

        # === COMPONENT 4: OPTIONS FLOW (12%) ===
        # Get recent price trend for context
        price_trend = commodity_df['Close'].pct_change(10).iloc[-1] * 100

        # Fetch options data (use cache if available)
        options_data = self.options_data_cache.get(commodity_ticker)
        if options_data is None:
            options_data = fetch_options_data(commodity_ticker)
            if options_data:
                self.options_data_cache[commodity_ticker] = options_data

        options_result = calculate_options_flow_score(
            options_data,
            price_trend,
            commodity_type
        )
        options_score = options_result['score']
        options_weighted = options_score * SIGNAL_WEIGHTS['options']

        # === COMPONENT 5: REGIME ADJUSTMENT (13%) ===
        macro_quadrant = detect_macro_quadrant(price_data)

        # Base signal before regime adjustment (now includes options)
        base_signal = sentiment_weighted + correlation_weighted + divergence_weighted + options_weighted

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

        # === GOLD-SPECIFIC REGIME ADJUSTMENT ===
        # For gold, apply additional adjustment based on structural demand indicators
        gold_regime_result = None
        gold_regime_adjustment = 0.0

        if commodity_ticker == 'GCA Comdty':
            gold_regime_result = detect_gold_specific_regime(
                price_data,
                self.correlation_engine
            )

            # Apply gold-specific adjustment if recommended
            adjustment = gold_regime_result['recommendation'].get('signal_adjustment', 0)
            if adjustment != 0:
                gold_regime_adjustment = adjustment
                final_signal = np.clip(final_signal + adjustment, -10, 10)

        # Classify signal
        classification, action = classify_signal(final_signal)

        result = {
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
                'options': round(options_score, 2),
                'options_weighted': round(options_weighted, 2),
                'regime_adjustment': round(regime_contribution, 2),
            },
            'macro_regime': macro_quadrant,
            'divergence_details': divergence_result,
            'options_details': options_result,
            'commodity_type': commodity_type,
            'timestamp': datetime.now().isoformat(),
        }

        # Add gold-specific regime info if applicable
        if gold_regime_result is not None:
            result['gold_regime'] = gold_regime_result
            result['components']['gold_regime_adjustment'] = round(gold_regime_adjustment, 2)

        return result

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
    print("\n" + "="*120)
    print(f"{'COMMODITY TRADING SIGNALS':^120}")
    print(f"{'Sentiment (35%) + Correlation (22%) + Divergence (18%) + Options (12%) + Regime (13%)':^120}")
    print("="*120 + "\n")

    # Get macro regime (same for all)
    first_signal = list(signals.values())[0]
    macro = first_signal.get('macro_regime', {})

    print(f"Macro Environment: {macro.get('name', 'Unknown')} (Quad {macro.get('quadrant', 0)})")
    print(f"  Growth: {macro.get('growth_signal', 0):+.2f} ({'Rising' if macro.get('growth_rising') else 'Falling'})")
    print(f"  Inflation: {macro.get('inflation_signal', 0):+.2f} ({'Rising' if macro.get('inflation_rising') else 'Falling'})")
    print()

    # Header
    print(f"{'Commodity':<15} {'Signal':>8} {'Classification':<15} {'Action':<25} {'Sent':>6} {'Corr':>6} {'Div':>6} {'Opts':>6} {'Reg':>6}")
    print("-"*120)

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
        options = components.get('options', 0)
        regime_adj = components.get('regime_adjustment', 0)

        print(f"{name:<15} {signal:>8.2f} {classification:<15} {action:<25} "
              f"{sentiment:>6.2f} {correlation:>6.2f} {divergence:>6.2f} {options:>6.2f} {regime_adj:>6.2f}")

    print("="*120)

    # Print Options Flow Details for each commodity
    print(f"\n{'OPTIONS FLOW DETAILS':^120}")
    print("-"*120)
    print(f"{'Commodity':<12} {'C/P Ratio':>10} {'IV':>8} {'IV %ile':>8} {'Skew':>8} {'IV Regime':<12} {'Opts Score':>10}")
    print("-"*120)

    for ticker, data in sorted_signals:
        name = COMMODITY_UNIVERSE.get(ticker, {}).get('name', ticker)[:11]
        options_details = data.get('options_details', {})

        if options_details.get('data_available', False):
            components = options_details.get('components', {})
            cp_ratio = components.get('call_put_ratio', {}).get('value', 1.0)
            iv_level = components.get('iv_level', {}).get('value', 0)
            iv_regime = components.get('iv_level', {}).get('regime', 'N/A')
            iv_pct = components.get('iv_percentile', {}).get('value', 50)
            skew = components.get('put_call_skew', {}).get('value', 0)
            opts_score = options_details.get('score', 0)

            print(f"{name:<12} {cp_ratio:>10.3f} {iv_level:>8.1f} {iv_pct:>8.1f} {skew:>8.2f} {iv_regime:<12} {opts_score:>10.2f}")
        else:
            print(f"{name:<12} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'No Data':<12} {'0.00':>10}")

    print("-"*120)

    # Print Gold-Specific Regime Info if available
    gold_signal = signals.get('GCA Comdty', {})
    if 'gold_regime' in gold_signal:
        gold_regime = gold_signal['gold_regime']
        print(f"\n{'GOLD-SPECIFIC REGIME ANALYSIS':^100}")
        print("-"*100)
        print(f"  Regime: {gold_regime.get('gold_regime', 'N/A').upper()}")
        print(f"  Regime Score: {gold_regime.get('gold_regime_score', 0):+.2f}")
        print(f"  Framework Confidence: {gold_regime.get('framework_confidence', 1.0):.0%}")

        recommendation = gold_regime.get('recommendation', {})
        print(f"  Action: {recommendation.get('action', 'N/A')}")
        print(f"  Description: {recommendation.get('description', 'N/A')}")

        # Component breakdown
        components = gold_regime.get('component_scores', {})
        print(f"\n  Component Scores:")
        print(f"    ETF Flow:              {components.get('etf_flow', {}).get('signal', 0):+.2f}")
        print(f"    Correlation Breakdown: {components.get('correlation_breakdown', {}).get('signal', 0):+.2f}")
        print(f"    Safe Haven Index:      {components.get('safe_haven', {}).get('signal', 0):+.2f}")
        print(f"    De-Dollarization:      {components.get('dedollarization', {}).get('signal', 0):+.2f}")

        # Show breakdown details if any
        breakdown = components.get('correlation_breakdown', {})
        if breakdown.get('breakdown_count', 0) > 0:
            print(f"\n  Correlation Breakdowns Detected: {breakdown.get('breakdown_count', 0)}")
            for ticker, detail in breakdown.get('breakdown_details', {}).items():
                print(f"    {ticker}: Current={detail['current']:.2f}, Expected={detail['expected']:.2f}")

        print("-"*120)

    print(f"\nSignal Interpretation:")
    print(f"  >= 7.0: Very Bullish (Strong Buy)")
    print(f"  >= 4.0: Bullish (Buy)")
    print(f"  -4.0 to 4.0: Neutral (No Position)")
    print(f"  <= -4.0: Bearish (Sell)")
    print(f"  <= -7.0: Very Bearish (Strong Sell)")

    print(f"\nOptions Flow Guide:")
    print(f"  C/P Ratio > 1.2: Bullish options positioning")
    print(f"  C/P Ratio < 0.8: Bearish options positioning")
    print(f"  IV %ile > 80: High volatility (potential reversal)")
    print(f"  IV %ile < 20: Low volatility (breakout potential)")
    print("="*120 + "\n")


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
    print(f"   ROC_5 (5-day): {latest['ROC_5']:.4f}%")
    print(f"   ROC_10 (10-day): {latest['ROC_10']:.4f}%")
    print(f"   ROC_20 (20-day): {latest['ROC_20']:.4f}%")
    print(f"   ROC_60 (60-day): {latest['ROC_60']:.4f}%")
    print(f"   ROC_120 (120-day): {latest['ROC_120']:.4f}%")
    print(f"   RSI (14): {latest['RSI']:.2f}")
    print(f"   MA_10: {latest['MA_10']:.2f}")
    print(f"   MA_20: {latest['MA_20']:.2f}")
    print(f"   MA_50: {latest['MA_50']:.2f}")
    print(f"   MA_100: {latest['MA_100']:.2f}")

    # TD Sequential
    print(f"\n2. TD SEQUENTIAL STATUS:")
    print(f"   TD_Setup_Buy: {latest.get('TD_Setup_Buy', 0)}")
    print(f"   TD_Setup_Sell: {latest.get('TD_Setup_Sell', 0)}")

    # Calculate components
    roc_raw = (
        latest['ROC_5'] * 0.15 +
        latest['ROC_10'] * 0.20 +
        latest['ROC_20'] * 0.30 +
        latest['ROC_60'] * 0.25 +
        latest['ROC_120'] * 0.10
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
