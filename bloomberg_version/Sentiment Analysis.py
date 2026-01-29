#!/usr/bin/env python3
"""
Carnival Core Score (CCS) - Bloomberg Terminal Edition
Momentum/trend scanner using BQL (Bloomberg Query Language) for professional-grade data
Includes Phase 1 (Volume) and Phase 2 (Options Flow) analysis
"""

import bql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Bloomberg tickers (different format than yfinance)
TICKERS = {
    'MACRO': [
        'ESA Index',    # S&P 500 E-mini Futures
        'NQA Index',    # Nasdaq 100 E-mini Futures
        'RTYA Index',   # Russell 2000 E-mini Futures
        'TUA Comdty',   # 2-Year Treasury Futures
        'FVA Comdty',   # 5-Year Treasury Futures
        'TYA Comdty',   # 10-Year Treasury Futures
        'USA Comdty',   # 30-Year Treasury Futures
        'DXY Index',    # US Dollar Index
        'USDJPY Curncy',
        'EURUSD Curncy',
        'GBPUSD Curncy',
        'AUDUSD Curncy',
        'USDCAD Curncy',
        'CLA Comdty',   # WTI Crude Oil
        'NGA Comdty',   # Natural Gas
        'HGA Comdty',   # Copper
        'PAA Comdty',   # Palladium
        'PLA Comdty',   # Platinum
        'GCA Comdty',   # Gold
        'SIA Comdty',   # Silver
        'BTC Curncy',   # Bitcoin
    ],
    'SECTORS': [
        'XLK US Equity', 'XLV US Equity', 'XLF US Equity', 'XLY US Equity',
        'XLC US Equity', 'XLI US Equity', 'XLP US Equity', 'XLE US Equity',
        'XLU US Equity', 'XLB US Equity', 'XLRE US Equity', 'XHB US Equity',
        'XBI US Equity', 'SMH US Equity', 'SPHB US Equity', 'SPLV US Equity',
    ],
    'TOP_STOCKS': [
        'AAPL US Equity', 'NVDA US Equity', 'MSFT US Equity', 'AMZN US Equity',
        'META US Equity', 'TSLA US Equity', 'GOOGL US Equity', 'AVGO US Equity',
        'GOOG US Equity', 'BRK/B US Equity', 'JPM US Equity', 'LLY US Equity',
        'V US Equity', 'XOM US Equity', 'UNH US Equity', 'MA US Equity',
        'COST US Equity', 'HD US Equity', 'WMT US Equity', 'PG US Equity',
        'NFLX US Equity', 'JNJ US Equity', 'BAC US Equity', 'ABBV US Equity',
        'CRM US Equity', 'AMD US Equity',
    ],
    'WORLD_ETFS': [
        'VEA US Equity', 'IEMG US Equity', 'EEM US Equity',
        'ACWI US Equity', 'BNDX US Equity', 'VWOB US Equity',
    ],
}

# Display names mapping (short ticker symbols)
TICKER_NAMES = {
    'ESA Index': 'ES', 'NQA Index': 'NQ', 'RTYA Index': 'RTY',
    'TUA Comdty': 'ZT', 'FVA Comdty': 'ZF', 'TYA Comdty': 'ZN', 'USA Comdty': 'ZB',
    'DXY Index': 'DXY', 'USDJPY Curncy': 'USD/JPY', 'EURUSD Curncy': 'EUR/USD',
    'GBPUSD Curncy': 'GBP/USD', 'AUDUSD Curncy': 'AUD/USD', 'USDCAD Curncy': 'USD/CAD',
    'CLA Comdty': 'Oil', 'NGA Comdty': 'Nat Gas', 'HGA Comdty': 'Copper',
    'PAA Comdty': 'Palladium', 'PLA Comdty': 'Platinum',
    'GCA Comdty': 'Gold', 'SIA Comdty': 'Silver', 'BTC Curncy': 'Bitcoin',
}

# Full names mapping for all tickers
TICKER_FULL_NAMES = {
    # Macro - Index Futures
    'ESA Index': 'S&P 500 E-mini',
    'NQA Index': 'Nasdaq 100 E-mini',
    'RTYA Index': 'Russell 2000 E-mini',
    # Macro - Treasury Futures
    'TUA Comdty': '2-Year Treasury',
    'FVA Comdty': '5-Year Treasury',
    'TYA Comdty': '10-Year Treasury',
    'USA Comdty': '30-Year Treasury',
    # Macro - Currencies
    'DXY Index': 'US Dollar Index',
    'USDJPY Curncy': 'US Dollar/Japanese Yen',
    'EURUSD Curncy': 'Euro/US Dollar',
    'GBPUSD Curncy': 'British Pound/US Dollar',
    'AUDUSD Curncy': 'Australian Dollar/US Dollar',
    'USDCAD Curncy': 'US Dollar/Canadian Dollar',
    # Macro - Commodities
    'CLA Comdty': 'WTI Crude Oil',
    'NGA Comdty': 'Natural Gas',
    'HGA Comdty': 'Copper',
    'PAA Comdty': 'Palladium',
    'PLA Comdty': 'Platinum',
    'GCA Comdty': 'Gold',
    'SIA Comdty': 'Silver',
    'BTC Curncy': 'Bitcoin',
    # Sectors
    'XLK US Equity': 'Technology Select Sector',
    'XLV US Equity': 'Health Care Select Sector',
    'XLF US Equity': 'Financial Select Sector',
    'XLY US Equity': 'Consumer Discretionary',
    'XLC US Equity': 'Communication Services',
    'XLI US Equity': 'Industrial Select Sector',
    'XLP US Equity': 'Consumer Staples',
    'XLE US Equity': 'Energy Select Sector',
    'XLU US Equity': 'Utilities Select Sector',
    'XLB US Equity': 'Materials Select Sector',
    'XLRE US Equity': 'Real Estate Select Sector',
    'XHB US Equity': 'Homebuilders ETF',
    'XBI US Equity': 'Biotech ETF',
    'SMH US Equity': 'Semiconductor ETF',
    'SPHB US Equity': 'S&P 500 High Beta ETF',
    'SPLV US Equity': 'S&P 500 Low Volatility ETF',
    # Top Stocks
    'AAPL US Equity': 'Apple Inc',
    'NVDA US Equity': 'NVIDIA Corp',
    'MSFT US Equity': 'Microsoft Corp',
    'AMZN US Equity': 'Amazon.com Inc',
    'META US Equity': 'Meta Platforms Inc',
    'TSLA US Equity': 'Tesla Inc',
    'GOOGL US Equity': 'Alphabet Inc Class A',
    'AVGO US Equity': 'Broadcom Inc',
    'GOOG US Equity': 'Alphabet Inc Class C',
    'BRK/B US Equity': 'Berkshire Hathaway Inc',
    'JPM US Equity': 'JPMorgan Chase & Co',
    'LLY US Equity': 'Eli Lilly & Co',
    'V US Equity': 'Visa Inc',
    'XOM US Equity': 'Exxon Mobil Corp',
    'UNH US Equity': 'UnitedHealth Group Inc',
    'MA US Equity': 'Mastercard Inc',
    'COST US Equity': 'Costco Wholesale Corp',
    'HD US Equity': 'Home Depot Inc',
    'WMT US Equity': 'Walmart Inc',
    'PG US Equity': 'Procter & Gamble Co',
    'NFLX US Equity': 'Netflix Inc',
    'JNJ US Equity': 'Johnson & Johnson',
    'BAC US Equity': 'Bank of America Corp',
    'ABBV US Equity': 'AbbVie Inc',
    'CRM US Equity': 'Salesforce Inc',
    'AMD US Equity': 'Advanced Micro Devices',
    # World ETFs
    'VEA US Equity': 'Developed Markets ETF',
    'IEMG US Equity': 'Emerging Markets ETF',
    'EEM US Equity': 'MSCI Emerging Markets ETF',
    'ACWI US Equity': 'All Country World Index ETF',
    'BNDX US Equity': 'Intl Bond ETF',
    'VWOB US Equity': 'EM Government Bond ETF',
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
    More aggressive scaling (divide by 3 instead of 10) for stronger signals.
    """
    values = np.clip(values, -100, 100)
    return np.tanh(values / 3) * max_val


def get_display_name(ticker):
    """Get display name for ticker"""
    if ticker in TICKER_NAMES:
        return TICKER_NAMES[ticker]
    # Extract symbol from Bloomberg format (e.g., "AAPL US Equity" -> "AAPL")
    return ticker.split()[0]


def get_full_name(ticker):
    """Get full name for ticker"""
    if ticker in TICKER_FULL_NAMES:
        return TICKER_FULL_NAMES[ticker]
    # Fallback to ticker itself if no full name defined
    return ticker


def is_equity_instrument(ticker):
    """Check if instrument is equity (stock/ETF) for options analysis"""
    return 'Equity' in ticker


def is_futures_instrument(ticker):
    """Check if instrument is futures/commodities"""
    return 'Comdty' in ticker or 'Index' in ticker


def is_volume_instrument(ticker):
    """Check if instrument has volume data (not forex/index)"""
    return 'Curncy' not in ticker or 'BTC' in ticker


# ============================================================================
# BQL DATA FETCHING
# ============================================================================

def fetch_historical_data(ticker, days=90):
    """
    Fetch historical OHLCV data using BQL
    Returns DataFrame with Date, Open, High, Low, Close, Volume
    """
    try:
        # Define date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Create BQL request for OHLCV data
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

        # Execute request
        response = bq.execute(request)

        # Convert to DataFrame using correct method
        df = pd.concat([data_item.df() for data_item in response], axis=1)

        # Sort by date and reset index
        df = df.sort_values('Date').reset_index(drop=True)

        return df

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def fetch_options_data(ticker):
    """
    Fetch options flow data using BQL (Phase 2)
    Returns dict with implied_volatility and call_put_ratio
    Different calculation for Futures vs Equities

    Call/Put Ratio = Call IV / Put IV
    - Higher ratio (>1.2) = Bullish (more call buying)
    - Lower ratio (<0.7) = Bearish (more put buying)
    """
    # Only fetch for instruments that have options (equities and futures)
    if not (is_equity_instrument(ticker) or is_futures_instrument(ticker)):
        return None

    try:
        # Determine if this is a futures instrument
        is_futures = is_futures_instrument(ticker)

        # Create BQL request for options metrics
        if is_futures:
            # Futures use different IV fields
            request = bql.Request(
                ticker,
                {
                    'Implied_Volatility': bq.data.IMPLIED_VOLATILITY(),
                    'Fut_Call_IV': bq.data.FUT_CALL_IMPLIED_VOLATILITY(),
                    'Fut_Put_IV': bq.data.FUT_PUT_IMPLIED_VOLATILITY(),
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
                }
            )

        # Execute request
        response = bq.execute(request)
        data = pd.concat([data_item.df() for data_item in response], axis=1)

        # Extract latest values and calculate call/put ratio
        if is_futures:
            fut_call_iv = data['Fut_Call_IV'].iloc[-1] if 'Fut_Call_IV' in data.columns else None
            fut_put_iv = data['Fut_Put_IV'].iloc[-1] if 'Fut_Put_IV' in data.columns else None

            # Calculate call/put ratio: Call IV / Put IV
            if fut_call_iv and fut_put_iv and fut_put_iv > 0:
                call_put_ratio = fut_call_iv / fut_put_iv
            else:
                call_put_ratio = 1.0
        else:
            call_iv_60d = data['Call_IV_60D'].iloc[-1] if 'Call_IV_60D' in data.columns else None
            put_iv_60d = data['Put_IV_60D'].iloc[-1] if 'Put_IV_60D' in data.columns else None

            # Calculate call/put ratio: Call IV / Put IV
            if call_iv_60d and put_iv_60d and put_iv_60d > 0:
                call_put_ratio = call_iv_60d / put_iv_60d
            else:
                call_put_ratio = 1.0

        # Extract implied volatility
        implied_vol = data['Implied_Volatility'].iloc[-1] if 'Implied_Volatility' in data.columns else 50

        result = {
            'call_put_ratio': call_put_ratio,
            'implied_volatility': implied_vol,
        }

        return result

    except Exception as e:
        print(f"Warning: Could not fetch options data for {ticker}: {e}")
        return None


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_indicators(df):
    """Calculate all technical indicators"""

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

    # Volume indicators (Phase 1)
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        # Volume ratio vs 20-day average
        df['Vol_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA_20']

        # Volume trend (5-day vs 20-day MA)
        df['Vol_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Vol_Trend'] = df['Vol_MA_5'] / df['Vol_MA_20']

        # Volume-Price correlation
        df['Vol_Price_Corr'] = np.sign(df['ROC_1']) * (df['Vol_Ratio'] - 1)
    else:
        # Neutral values for instruments without volume
        df['Vol_Ratio'] = 1.0
        df['Vol_Trend'] = 1.0
        df['Vol_Price_Corr'] = 0.0

    return df


# ============================================================================
# SCORING ALGORITHM
# ============================================================================

def calculate_daily_score(df_row, options_data=None):
    """
    Calculate composite score with volume and options flow.

    Base Components (92%):
    - ROC (Price Momentum): 35%
    - MA Trend: 25%
    - Volume & Conviction: 20%
    - RSI: 12%

    Options Flow (8%) - Stocks/ETFs/Futures with options:
    - Call/Put Ratio: 4%
    - Implied Volatility: 4%
    """

    # === ROC SIGNALS (35%) ===
    # Multi-timeframe momentum with recent bias reduced
    roc_score = (
        df_row['ROC_1'] * 0.15 +   # Daily (reduced from 35%)
        df_row['ROC_2'] * 0.15 +   # 2-day (reduced from 25%)
        df_row['ROC_5'] * 0.25 +   # Weekly (increased)
        df_row['ROC_10'] * 0.25 +  # 2-week (increased)
        df_row['ROC_20'] * 0.20    # Monthly
    )
    roc_score = normalize_to_range(roc_score) * 0.35

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

    ma_score = (normalize_to_range(ma_slope_score) * 0.7 +
                normalize_to_range(ma_cross_score) * 0.3) * 0.25

    # === VOLUME & CONVICTION (20%) - Phase 1 ===
    vol_ratio_score = normalize_to_range((df_row['Vol_Ratio'] - 1) * 100) * 0.40
    vol_trend_score = normalize_to_range((df_row['Vol_Trend'] - 1) * 100) * 0.30
    vol_price_corr_score = normalize_to_range(df_row['Vol_Price_Corr'] * 100) * 0.30

    volume_score = (vol_ratio_score + vol_trend_score + vol_price_corr_score) * 0.20

    # === RSI (12%) ===
    rsi_normalized = (df_row['RSI'] - 50) / 5  # Center at 50, scale
    rsi_score = normalize_to_range(rsi_normalized) * 0.12

    # === PRICE VS MA (8%) ===
    # Distance from key moving averages
    price_ma_score = 0
    if not pd.isna(df_row['MA_20']):
        pct_from_ma20 = ((df_row['Close'] - df_row['MA_20']) / df_row['MA_20']) * 100
        price_ma_score = normalize_to_range(pct_from_ma20) * 0.08

    # === BASE SCORE (92%) ===
    base_score = roc_score + ma_score + volume_score + rsi_score + price_ma_score

    # === OPTIONS FLOW (8%) - Phase 2 ===
    options_score = 0
    if options_data is not None:
        # Call/Put Ratio (4%) - Call IV / Put IV
        # Higher ratio = more calls being bought (bullish)
        # Lower ratio = more puts being bought (bearish)
        cp_ratio = options_data['call_put_ratio']
        if cp_ratio > 1.5:
            cp_score = 10  # Very bullish (extreme call buying)
        elif cp_ratio > 1.2:
            cp_score = 5   # Bullish
        elif cp_ratio > 0.7:
            cp_score = 0   # Neutral
        elif cp_ratio > 0.5:
            cp_score = -5  # Bearish
        else:
            cp_score = -10 # Very bearish (extreme put buying)

        cp_normalized = normalize_to_range(cp_score) * 0.04

        # Implied Volatility (4%) - contextual to price trend
        # High IV can signal conviction (uptrend) or panic (downtrend)
        implied_vol = options_data['implied_volatility']
        price_trend = df_row['ROC_10']  # Use 10-day trend for context

        # Normalize IV to typical range (20-60 for equities, can be higher for commodities)
        # IV > 40 = high, IV < 20 = low (rough guideline)
        iv_deviation = (implied_vol - 30) / 10  # Center at 30, normalize

        if price_trend > 0:
            # Uptrend: High IV = strong conviction/momentum
            iv_score = iv_deviation
        else:
            # Downtrend: High IV = panic/capitulation (potential reversal)
            iv_score = -iv_deviation

        iv_normalized = normalize_to_range(iv_score) * 0.04

        options_score = cp_normalized + iv_normalized

    # === TOTAL SCORE ===
    total_score = base_score + options_score

    return np.clip(total_score, -10, 10)


def calculate_scores_for_ticker(ticker):
    """Calculate historical scores for a ticker"""

    # Fetch OHLCV data
    df = fetch_historical_data(ticker, days=90)
    if df is None or len(df) < 20:
        return None

    # Calculate indicators
    df = calculate_indicators(df)

    # Fetch options data (if applicable)
    # Function already checks for equity/futures inside, so no need for external check
    options_data = fetch_options_data(ticker)

    # Calculate daily scores
    scores = []
    for idx, row in df.iterrows():
        if idx < 20:  # Need 20 days for all indicators
            scores.append(np.nan)
        else:
            score = calculate_daily_score(row, options_data)
            scores.append(score)

    df['Score'] = scores

    # Apply EMA smoothing (5-day)
    df['Score_EMA'] = df['Score'].ewm(span=5, adjust=False).mean()

    return df


# ============================================================================
# SENTIMENT CLASSIFICATION
# ============================================================================

def classify_sentiment(score):
    """Classify score into sentiment category"""
    if pd.isna(score):
        return 'N/A', ''
    elif score >= 7:
        return 'Very Bullish (L/S)', '\033[1;32m'  # Bold green
    elif score >= 4:
        return 'Bullish (L)', '\033[32m'  # Green
    elif score >= -4:
        return 'Neutral (L/S chop)', '\033[37m'  # White
    elif score >= -7:
        return 'Bearish (S)', '\033[31m'  # Red
    else:
        return 'Very Bearish (L/S)', '\033[1;31m'  # Bold red


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_score(score):
    """Format score with color"""
    if pd.isna(score):
        return 'N/A'
    sentiment, color = classify_sentiment(score)
    reset = '\033[0m'
    return f"{color}{score:>5.1f}{reset}"


def print_results(results_by_category):
    """Print formatted results without color coding for proper alignment"""

    print("\n" + "="*140)
    print(f"{'CARNIVAL CORE SCORE (CCS) - BLOOMBERG EDITION':^140}")
    print(f"{'Phase 1 (Volume) + Phase 2 (Options Flow) Enabled':^140}")
    print("="*140 + "\n")

    for category, results in results_by_category.items():
        print(f"\n{'='*20} {category} {'='*20}\n")

        # Header - fixed width alignment with Name column
        print(f"{'Ticker':<12} {'Name':<28} {'Score':>6} {'Sentiment':<20} {'5D Avg':>7} {'D-1':>7} {'D-2':>7} {'D-3':>7} {'D-4':>7} {'10D Avg':>8} {'20D Avg':>8}")
        print("-" * 140)

        # Sort by current score (descending)
        results_sorted = sorted(results, key=lambda x: x['current_score'] if not pd.isna(x['current_score']) else -999, reverse=True)

        for result in results_sorted:
            display_name = get_display_name(result['ticker'])
            full_name = get_full_name(result['ticker'])
            # Truncate full name if too long
            if len(full_name) > 27:
                full_name = full_name[:24] + "..."
            score = result['current_score']
            sentiment, _ = classify_sentiment(score)

            # Format score without color
            if pd.isna(score):
                score_str = f"{'N/A':>6}"
            else:
                score_str = f"{score:>6.1f}"

            # Sentiment without color, fixed 20 character width
            sentiment_str = f"{sentiment:<20}"

            # Format all numeric columns
            avg_5d = f"{result['avg_5d']:>7.1f}" if not pd.isna(result['avg_5d']) else f"{'N/A':>7}"
            d_1 = f"{result['d_minus_1']:>7.1f}" if not pd.isna(result['d_minus_1']) else f"{'N/A':>7}"
            d_2 = f"{result['d_minus_2']:>7.1f}" if not pd.isna(result['d_minus_2']) else f"{'N/A':>7}"
            d_3 = f"{result['d_minus_3']:>7.1f}" if not pd.isna(result['d_minus_3']) else f"{'N/A':>7}"
            d_4 = f"{result['d_minus_4']:>7.1f}" if not pd.isna(result['d_minus_4']) else f"{'N/A':>7}"
            avg_10d = f"{result['avg_10d']:>8.1f}" if not pd.isna(result['avg_10d']) else f"{'N/A':>8}"
            avg_20d = f"{result['avg_20d']:>8.1f}" if not pd.isna(result['avg_20d']) else f"{'N/A':>8}"

            # Build row with consistent spacing including full name
            row = f"{display_name:<12} {full_name:<28} {score_str} {sentiment_str} {avg_5d} {d_1} {d_2} {d_3} {d_4} {avg_10d} {avg_20d}"
            print(row)

        print()


# ============================================================================
# MACRO QUADRANT ANALYSIS (FIDENZA FRAMEWORK)
# ============================================================================

def analyze_macro_quadrant(results_by_category):
    """
    Analyzes current macro environment using Fidenza four-quadrant framework.

    Quadrants:
    - Quad 1 (Goldilocks): Rising growth + Falling inflation
    - Quad 2 (Reflation): Rising growth + Rising inflation
    - Quad 3 (Stagflation): Falling growth + Rising inflation
    - Quad 4 (Risk-Off): Falling growth + Falling inflation

    Returns: Dictionary with quadrant, growth signal, inflation signal, and analysis text
    """

    # Extract key indicators from results
    all_results = {}
    for category, results in results_by_category.items():
        for result in results:
            all_results[result['ticker']] = result

    # Helper function to safely get score
    def get_score(ticker):
        return all_results.get(ticker, {}).get('current_score', np.nan)

    def get_avg_5d(ticker):
        return all_results.get(ticker, {}).get('avg_5d', np.nan)

    # === GROWTH SIGNALS ===
    # Equity indices momentum (S&P, Nasdaq, Russell)
    equity_scores = []
    for ticker in ['ESA Index', 'NQA Index', 'RTYA Index']:
        score = get_score(ticker)
        if not pd.isna(score):
            equity_scores.append(score)

    equity_signal = np.mean(equity_scores) if equity_scores else 0

    # Cyclical commodities (Copper, Oil as economic demand proxies)
    commodity_scores = []
    for ticker in ['HGA Comdty', 'CLA Comdty']:  # Copper, Oil
        score = get_score(ticker)
        if not pd.isna(score):
            commodity_scores.append(score)

    commodity_signal = np.mean(commodity_scores) if commodity_scores else 0

    # Combined growth signal
    growth_signal = (equity_signal * 0.7 + commodity_signal * 0.3)
    growth_rising = growth_signal > 0

    # === INFLATION SIGNALS ===
    # Treasury yields (inverted - falling bonds = rising yields = rising inflation)
    bond_scores = []
    for ticker in ['TUA Comdty', 'FVA Comdty', 'TYA Comdty', 'USA Comdty']:
        score = get_score(ticker)
        if not pd.isna(score):
            bond_scores.append(score)

    bond_signal = -np.mean(bond_scores) if bond_scores else 0  # Inverted

    # Commodities (Gold, Silver, Oil as inflation hedges)
    inflation_commodity_scores = []
    for ticker in ['GCA Comdty', 'SIA Comdty', 'CLA Comdty']:  # Gold, Silver, Oil
        score = get_score(ticker)
        if not pd.isna(score):
            inflation_commodity_scores.append(score)

    commodity_inflation_signal = np.mean(inflation_commodity_scores) if inflation_commodity_scores else 0

    # Combined inflation signal
    inflation_signal = (bond_signal * 0.5 + commodity_inflation_signal * 0.5)
    inflation_rising = inflation_signal > 0

    # === DETERMINE QUADRANT ===
    if growth_rising and not inflation_rising:
        quadrant = "Quad 1: Goldilocks (Disinflation)"
        quadrant_num = 1
        description = "Rising growth with falling inflation"
        outperformers = "Equities, credit, crypto, EM assets"
        underperformers = "Cyclical commodities"
        strategy = "Risk-on positioning. Favor growth equities and credit."
    elif growth_rising and inflation_rising:
        quadrant = "Quad 2: Reflation"
        quadrant_num = 2
        description = "Rising growth and rising inflation"
        outperformers = "Equities, commodities, risky currencies"
        underperformers = "Precious metals"
        strategy = "Strong risk appetite. Watch for central bank tightening signals."
    elif not growth_rising and inflation_rising:
        quadrant = "Quad 3: Stagflation"
        quadrant_num = 3
        description = "Falling growth with rising inflation"
        outperformers = "USD, commodities (non-precious metals)"
        underperformers = "Equities, bonds"
        strategy = "Defensive positioning. Volatility likely to spike. Reduce equity exposure."
    else:  # not growth_rising and not inflation_rising
        quadrant = "Quad 4: Risk-Off (Deflation)"
        quadrant_num = 4
        description = "Falling growth and falling inflation"
        outperformers = "Safe-haven currencies, quality bonds"
        underperformers = "Commodities, equities"
        strategy = "Flight to safety. Favor treasuries and defensive assets."

    # === BUILD DETAILED ANALYSIS ===
    analysis_lines = []
    analysis_lines.append("\n" + "="*140)
    analysis_lines.append(f"{'MACRO ENVIRONMENT ANALYSIS - FIDENZA FOUR-QUADRANT FRAMEWORK':^140}")
    analysis_lines.append("="*140 + "\n")

    analysis_lines.append(f"Current Quadrant: {quadrant}")
    analysis_lines.append(f"Description: {description}\n")

    analysis_lines.append("Signal Breakdown:")
    analysis_lines.append(f"  Growth Signal: {growth_signal:+.2f} ({'Rising' if growth_rising else 'Falling'})")
    analysis_lines.append(f"    - Equity Indices (S&P/Nasdaq/Russell): {equity_signal:+.2f}")
    analysis_lines.append(f"    - Cyclical Commodities (Copper/Oil): {commodity_signal:+.2f}")
    analysis_lines.append(f"  Inflation Signal: {inflation_signal:+.2f} ({'Rising' if inflation_rising else 'Falling'})")
    analysis_lines.append(f"    - Treasury Yields (inverted): {bond_signal:+.2f}")
    analysis_lines.append(f"    - Inflation Hedges (Gold/Silver/Oil): {commodity_inflation_signal:+.2f}\n")

    # Key instrument scores
    analysis_lines.append("Key Instrument Scores:")
    key_instruments = {
        'ESA Index': 'S&P 500 Futures',
        'NQA Index': 'Nasdaq 100 Futures',
        'TYA Comdty': '10Y Treasury',
        'GCA Comdty': 'Gold',
        'CLA Comdty': 'Crude Oil',
        'DXY Index': 'US Dollar Index',
    }

    for ticker, name in key_instruments.items():
        score = get_score(ticker)
        avg_5d = get_avg_5d(ticker)
        if not pd.isna(score):
            sentiment, _ = classify_sentiment(score)
            analysis_lines.append(f"  {name:<20} Score: {score:>6.1f}  (5D Avg: {avg_5d:>6.1f})  {sentiment}")
        else:
            analysis_lines.append(f"  {name:<20} Score: {'N/A':>6}  (5D Avg: {'N/A':>6})  N/A")

    analysis_lines.append(f"\nAsset Class Outlook:")
    analysis_lines.append(f"  Outperformers: {outperformers}")
    analysis_lines.append(f"  Underperformers: {underperformers}")
    analysis_lines.append(f"\nRecommended Strategy:")
    analysis_lines.append(f"  {strategy}")

    # Transition signals
    analysis_lines.append(f"\nQuadrant Transition Watch:")
    if quadrant_num == 1:
        analysis_lines.append(f"  → Quad 2 (Reflation): Watch for stimulative policy and ISM acceleration")
    elif quadrant_num == 2:
        analysis_lines.append(f"  → Quad 3 (Stagflation): Monitor Fed tightening signals and yield spikes")
    elif quadrant_num == 3:
        analysis_lines.append(f"  → Quad 4 (Risk-Off): Look for economic weakness with persistent inflation")
    else:
        analysis_lines.append(f"  → Quad 1 (Goldilocks): Watch for policy accommodation and falling yields")

    analysis_lines.append("\n" + "="*140)
    analysis_lines.append("Framework Reference: https://www.fidenzamacro.com/p/the-four-quadrant-global-macro-framework")
    analysis_lines.append("="*140 + "\n")

    # Print analysis
    for line in analysis_lines:
        print(line)

    return {
        'quadrant': quadrant,
        'quadrant_num': quadrant_num,
        'growth_signal': growth_signal,
        'growth_rising': growth_rising,
        'inflation_signal': inflation_signal,
        'inflation_rising': inflation_rising,
        'description': description,
        'outperformers': outperformers,
        'underperformers': underperformers,
        'strategy': strategy,
    }


# ============================================================================
# GOLD-SPECIFIC REGIME ANALYSIS
# ============================================================================

def analyze_gold_regime(results_by_category):
    """
    Analyze gold-specific regime to detect structural demand that may override
    standard macro framework signals.

    Components:
    1. ETF Flow Proxy: Gold price momentum as flow indicator
    2. Correlation Breakdown: Gold vs DXY divergence detection
    3. Safe Haven Index: Gold vs VIX behavior
    4. De-Dollarization Proxy: Gold strength vs USD

    Returns: Dictionary with gold regime analysis
    """
    # Extract all results
    all_results = {}
    for category, results in results_by_category.items():
        for result in results:
            all_results[result['ticker']] = result

    def get_score(ticker):
        return all_results.get(ticker, {}).get('current_score', np.nan)

    def get_avg_5d(ticker):
        return all_results.get(ticker, {}).get('avg_5d', np.nan)

    def get_avg_20d(ticker):
        return all_results.get(ticker, {}).get('avg_20d', np.nan)

    # === COMPONENT 1: ETF FLOW PROXY (25%) ===
    # Use gold score momentum as proxy for ETF flows
    gold_score = get_score('GCA Comdty')
    gold_avg_5d = get_avg_5d('GCA Comdty')
    gold_avg_20d = get_avg_20d('GCA Comdty')

    if not pd.isna(gold_score) and not pd.isna(gold_avg_5d):
        # Strong momentum = strong flows
        if gold_score > 4 and gold_avg_5d > 2:
            etf_flow_score = 7.5
            etf_flow_regime = 'strong_inflow'
        elif gold_score > 2:
            etf_flow_score = 4.0
            etf_flow_regime = 'moderate_inflow'
        elif gold_score > -2:
            etf_flow_score = 0.0
            etf_flow_regime = 'neutral'
        elif gold_score > -4:
            etf_flow_score = -4.0
            etf_flow_regime = 'moderate_outflow'
        else:
            etf_flow_score = -7.5
            etf_flow_regime = 'strong_outflow'

        # Trend confirmation
        if not pd.isna(gold_avg_20d):
            if (gold_score > 0 and gold_avg_20d > 0) or (gold_score < 0 and gold_avg_20d < 0):
                etf_flow_score *= 1.3  # Trend confirms
    else:
        etf_flow_score = 0
        etf_flow_regime = 'no_data'

    # === COMPONENT 2: CORRELATION BREAKDOWN (25%) ===
    # Check if gold is moving independently of DXY
    dxy_score = get_score('DXY Index')

    if not pd.isna(gold_score) and not pd.isna(dxy_score):
        # Normal: Gold inverse to DXY
        # Breakdown: Both moving same direction or gold strong despite DXY strength
        if gold_score > 2 and dxy_score > 0:
            # Gold up despite dollar strength = structural demand
            breakdown_score = 8.0
            breakdown_regime = 'structural_demand'
        elif gold_score > 2 and dxy_score < -2:
            # Gold up with dollar weak = normal inverse
            breakdown_score = 2.0
            breakdown_regime = 'traditional'
        elif gold_score < -2 and dxy_score < -2:
            # Gold down despite dollar weakness = structural weakness
            breakdown_score = -6.0
            breakdown_regime = 'structural_weakness'
        else:
            breakdown_score = 0.0
            breakdown_regime = 'traditional'
    else:
        breakdown_score = 0
        breakdown_regime = 'no_data'

    # === COMPONENT 3: SAFE HAVEN INDEX (25%) ===
    # Compare gold behavior to other safe havens
    treasury_score = get_score('TYA Comdty')  # 10Y Treasury
    usdjpy_score = get_score('USDJPY Curncy')  # JPY (inverse)

    safe_haven_components = []

    if not pd.isna(gold_score):
        # Gold/Treasury alignment
        if not pd.isna(treasury_score):
            if gold_score > 0 and treasury_score > 0:
                safe_haven_components.append(3)  # Both bid = flight to safety
            elif gold_score > 0 and treasury_score < 0:
                safe_haven_components.append(2)  # Gold unique bid = structural
            elif gold_score < 0 and treasury_score > 0:
                safe_haven_components.append(-2)  # Gold lagging = weakness
            else:
                safe_haven_components.append(0)

        # Gold/JPY alignment (USDJPY down = JPY strength)
        if not pd.isna(usdjpy_score):
            if gold_score > 0 and usdjpy_score < 0:
                safe_haven_components.append(2)  # Both safe havens bid
            elif gold_score > 0 and usdjpy_score > 0:
                safe_haven_components.append(3)  # Gold up despite risk-on = structural
            else:
                safe_haven_components.append(0)

    if safe_haven_components:
        safe_haven_score = np.mean(safe_haven_components) * 2
        if safe_haven_score > 4:
            safe_haven_regime = 'structural_demand'
        elif safe_haven_score > 1:
            safe_haven_regime = 'fear_driven'
        elif safe_haven_score > -2:
            safe_haven_regime = 'neutral'
        else:
            safe_haven_regime = 'risk_on_headwind'
    else:
        safe_haven_score = 0
        safe_haven_regime = 'no_data'

    # === COMPONENT 4: DE-DOLLARIZATION PROXY (25%) ===
    # Gold strength relative to broad USD strength
    if not pd.isna(gold_score) and not pd.isna(dxy_score):
        # Strong gold + Strong DXY = de-dollarization / central bank buying
        gold_dxy_diff = gold_score - (-dxy_score)  # Expecting inverse relationship

        if gold_dxy_diff > 6:
            dedollar_score = 8.0
            dedollar_regime = 'strong_structural_demand'
        elif gold_dxy_diff > 3:
            dedollar_score = 4.0
            dedollar_regime = 'moderate_structural_demand'
        elif gold_dxy_diff > -3:
            dedollar_score = 0.0
            dedollar_regime = 'traditional_dynamics'
        else:
            dedollar_score = -4.0
            dedollar_regime = 'structural_weakness'
    else:
        dedollar_score = 0
        dedollar_regime = 'no_data'

    # === COMPOSITE GOLD REGIME SCORE ===
    component_scores = [etf_flow_score, breakdown_score, safe_haven_score, dedollar_score]
    valid_scores = [s for s in component_scores if s != 0 or True]  # Include all
    gold_regime_score = np.mean(valid_scores) if valid_scores else 0

    # Determine overall regime
    if breakdown_regime == 'structural_demand' or dedollar_regime == 'strong_structural_demand':
        gold_regime = 'structural'
        framework_confidence = 0.3
    elif breakdown_regime == 'structural_weakness':
        gold_regime = 'structural_weakness'
        framework_confidence = 0.5
    elif etf_flow_regime in ['strong_inflow', 'moderate_inflow'] and gold_regime_score > 3:
        gold_regime = 'transitional'
        framework_confidence = 0.6
    else:
        gold_regime = 'traditional'
        framework_confidence = 1.0

    # Generate recommendation
    if gold_regime == 'structural' and gold_regime_score > 3:
        recommendation = {
            'action': 'override_macro_bearish',
            'description': 'Structural demand strong - override bearish macro signals',
            'signal_adjustment': min(gold_regime_score * 0.5, 3),
        }
    elif gold_regime == 'structural_weakness' and gold_regime_score < -3:
        recommendation = {
            'action': 'confirm_weakness',
            'description': 'Structural indicators confirm weakness',
            'signal_adjustment': max(gold_regime_score * 0.3, -2),
        }
    else:
        recommendation = {
            'action': 'standard_framework',
            'description': 'Use standard macro quadrant framework',
            'signal_adjustment': 0,
        }

    result = {
        'gold_regime_score': round(gold_regime_score, 2),
        'gold_regime': gold_regime,
        'framework_confidence': framework_confidence,
        'recommendation': recommendation,
        'component_scores': {
            'etf_flow': {'score': round(etf_flow_score, 2), 'regime': etf_flow_regime},
            'correlation_breakdown': {'score': round(breakdown_score, 2), 'regime': breakdown_regime},
            'safe_haven': {'score': round(safe_haven_score, 2), 'regime': safe_haven_regime},
            'dedollarization': {'score': round(dedollar_score, 2), 'regime': dedollar_regime},
        },
        'inputs': {
            'gold_score': round(gold_score, 2) if not pd.isna(gold_score) else None,
            'dxy_score': round(dxy_score, 2) if not pd.isna(dxy_score) else None,
            'treasury_score': round(treasury_score, 2) if not pd.isna(treasury_score) else None,
        }
    }

    # Print analysis
    print("\n" + "="*140)
    print(f"{'GOLD-SPECIFIC REGIME ANALYSIS':^140}")
    print("="*140 + "\n")

    print(f"Gold Regime: {gold_regime.upper()}")
    print(f"Regime Score: {gold_regime_score:+.2f}")
    print(f"Framework Confidence: {framework_confidence:.0%}")
    print(f"\nRecommendation: {recommendation['action']}")
    print(f"Description: {recommendation['description']}")

    print(f"\nComponent Breakdown:")
    print(f"  ETF Flow Proxy:        {etf_flow_score:+.2f}  ({etf_flow_regime})")
    print(f"  Correlation Breakdown: {breakdown_score:+.2f}  ({breakdown_regime})")
    print(f"  Safe Haven Index:      {safe_haven_score:+.2f}  ({safe_haven_regime})")
    print(f"  De-Dollarization:      {dedollar_score:+.2f}  ({dedollar_regime})")

    print(f"\nInput Scores:")
    print(f"  Gold (GCA):    {gold_score:+.2f}" if not pd.isna(gold_score) else "  Gold: N/A")
    print(f"  DXY Index:     {dxy_score:+.2f}" if not pd.isna(dxy_score) else "  DXY: N/A")
    print(f"  10Y Treasury:  {treasury_score:+.2f}" if not pd.isna(treasury_score) else "  10Y Treasury: N/A")

    print("\n" + "="*140)

    return result


# ============================================================================
# REPORT GENERATION (PDF / IMAGE)
# ============================================================================

def generate_report(results_by_category, macro_analysis, gold_regime=None, output_path=None, format='pdf'):
    """
    Generate a professional PDF or image report for circulation.

    Args:
        results_by_category: Dict of category -> results list
        macro_analysis: Dict from analyze_macro_quadrant()
        gold_regime: Dict from analyze_gold_regime() (optional)
        output_path: Output file path (default: auto-generated with timestamp)
        format: 'pdf' or 'png'

    Returns:
        Path to generated file
    """
    # Set up the figure
    fig = plt.figure(figsize=(16, 22))
    fig.patch.set_facecolor('white')

    # Title
    fig.suptitle('Carnival Core Score (CCS) - Market Analysis Report',
                 fontsize=20, fontweight='bold', y=0.98)

    # Timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(0.5, 0.955, f'Generated: {timestamp}', ha='center', fontsize=10, style='italic')

    # Create grid for subplots
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3,
                          left=0.05, right=0.95, top=0.93, bottom=0.02)

    # === SECTION 1: MACRO QUADRANT (Top Left) ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')

    quadrant_colors = {1: '#2ECC71', 2: '#F39C12', 3: '#E74C3C', 4: '#3498DB'}
    quad_num = macro_analysis.get('quadrant_num', 1)
    quad_color = quadrant_colors.get(quad_num, '#95A5A6')

    ax1.add_patch(mpatches.FancyBboxPatch((0.05, 0.3), 0.9, 0.6,
                                           boxstyle="round,pad=0.02",
                                           facecolor=quad_color, alpha=0.3,
                                           edgecolor=quad_color, linewidth=2))

    ax1.text(0.5, 0.75, 'MACRO ENVIRONMENT', ha='center', va='center',
             fontsize=14, fontweight='bold')
    ax1.text(0.5, 0.55, macro_analysis.get('quadrant', 'Unknown'),
             ha='center', va='center', fontsize=12, fontweight='bold', color=quad_color)
    ax1.text(0.5, 0.4, macro_analysis.get('description', ''),
             ha='center', va='center', fontsize=10)

    growth_txt = f"Growth: {macro_analysis.get('growth_signal', 0):+.2f} ({'↑' if macro_analysis.get('growth_rising') else '↓'})"
    inflation_txt = f"Inflation: {macro_analysis.get('inflation_signal', 0):+.2f} ({'↑' if macro_analysis.get('inflation_rising') else '↓'})"
    ax1.text(0.25, 0.15, growth_txt, ha='center', va='center', fontsize=10)
    ax1.text(0.75, 0.15, inflation_txt, ha='center', va='center', fontsize=10)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('Macro Quadrant Analysis', fontsize=12, fontweight='bold', pad=10)

    # === SECTION 2: GOLD REGIME (Top Right) ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    if gold_regime:
        regime_colors = {
            'structural': '#FFD700',
            'transitional': '#FFA500',
            'traditional': '#C0C0C0',
            'structural_weakness': '#CD5C5C'
        }
        g_regime = gold_regime.get('gold_regime', 'traditional')
        g_color = regime_colors.get(g_regime, '#C0C0C0')

        ax2.add_patch(mpatches.FancyBboxPatch((0.05, 0.3), 0.9, 0.6,
                                               boxstyle="round,pad=0.02",
                                               facecolor=g_color, alpha=0.3,
                                               edgecolor=g_color, linewidth=2))

        ax2.text(0.5, 0.75, 'GOLD REGIME', ha='center', va='center',
                 fontsize=14, fontweight='bold')
        ax2.text(0.5, 0.55, g_regime.upper().replace('_', ' '),
                 ha='center', va='center', fontsize=12, fontweight='bold', color='#8B4513')

        score_txt = f"Regime Score: {gold_regime.get('gold_regime_score', 0):+.2f}"
        conf_txt = f"Framework Confidence: {gold_regime.get('framework_confidence', 1.0):.0%}"
        ax2.text(0.5, 0.4, score_txt, ha='center', va='center', fontsize=10)
        ax2.text(0.5, 0.25, conf_txt, ha='center', va='center', fontsize=10)

        rec = gold_regime.get('recommendation', {})
        ax2.text(0.5, 0.1, rec.get('action', '').replace('_', ' ').title(),
                 ha='center', va='center', fontsize=9, style='italic')
    else:
        ax2.text(0.5, 0.5, 'Gold Regime Analysis\nNot Available',
                 ha='center', va='center', fontsize=12)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Gold-Specific Regime', fontsize=12, fontweight='bold', pad=10)

    # === SECTION 3: STRATEGY RECOMMENDATIONS ===
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    strategy_text = f"""
STRATEGY RECOMMENDATIONS

Outperformers: {macro_analysis.get('outperformers', 'N/A')}
Underperformers: {macro_analysis.get('underperformers', 'N/A')}

Strategy: {macro_analysis.get('strategy', 'N/A')}
"""
    ax3.text(0.02, 0.9, strategy_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    # === SECTION 4-7: SCORE TABLES BY CATEGORY ===
    categories = list(results_by_category.keys())
    for idx, category in enumerate(categories[:4]):  # Max 4 categories
        row = 2 + idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])

        results = results_by_category[category]
        if not results:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(category, fontsize=11, fontweight='bold')
            ax.axis('off')
            continue

        # Sort by score
        results_sorted = sorted(results,
                               key=lambda x: x['current_score'] if not pd.isna(x['current_score']) else -999,
                               reverse=True)[:10]  # Top 10

        # Create table data
        table_data = []
        colors = []
        for r in results_sorted:
            ticker = get_display_name(r['ticker'])
            score = r['current_score']
            avg_5d = r['avg_5d']

            if pd.isna(score):
                score_str = 'N/A'
                row_color = 'white'
            else:
                score_str = f'{score:+.1f}'
                if score >= 4:
                    row_color = '#90EE90'  # Light green
                elif score <= -4:
                    row_color = '#FFB6C1'  # Light red
                else:
                    row_color = 'white'

            avg_str = f'{avg_5d:+.1f}' if not pd.isna(avg_5d) else 'N/A'
            sentiment, _ = classify_sentiment(score)

            table_data.append([ticker, score_str, avg_str, sentiment[:15]])
            colors.append([row_color] * 4)

        if table_data:
            table = ax.table(cellText=table_data,
                            colLabels=['Ticker', 'Score', '5D Avg', 'Sentiment'],
                            cellColours=colors,
                            loc='center',
                            cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)

            # Style header
            for j in range(4):
                table[(0, j)].set_facecolor('#4A90D9')
                table[(0, j)].set_text_props(color='white', fontweight='bold')

        ax.set_title(category, fontsize=11, fontweight='bold')
        ax.axis('off')

    # === FOOTER ===
    fig.text(0.5, 0.01, 'Carnival Core Score (CCS) - Bloomberg Edition | Confidential',
             ha='center', fontsize=8, style='italic', color='gray')

    # Generate output path if not provided
    if output_path is None:
        timestamp_file = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'CCS_Report_{timestamp_file}.{format}'

    # Save figure
    if format.lower() == 'pdf':
        with PdfPages(output_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
    else:
        plt.savefig(output_path, format='png', bbox_inches='tight', dpi=150)

    plt.close(fig)

    print(f"\n{'='*80}")
    print(f"Report generated: {output_path}")
    print(f"{'='*80}\n")

    return output_path


def generate_gold_regime_chart(gold_regime, output_path=None):
    """
    Generate a focused gold regime analysis chart.

    Args:
        gold_regime: Dict from analyze_gold_regime()
        output_path: Output file path

    Returns:
        Path to generated file
    """
    if gold_regime is None:
        print("No gold regime data available")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Gold-Specific Regime Analysis', fontsize=16, fontweight='bold')

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(0.5, 0.95, f'Generated: {timestamp}', ha='center', fontsize=10, style='italic')

    components = gold_regime.get('component_scores', {})

    # Component scores bar chart
    ax1 = axes[0, 0]
    comp_names = ['ETF Flow', 'Corr. Breakdown', 'Safe Haven', 'De-Dollar']
    comp_scores = [
        components.get('etf_flow', {}).get('score', 0),
        components.get('correlation_breakdown', {}).get('score', 0),
        components.get('safe_haven', {}).get('score', 0),
        components.get('dedollarization', {}).get('score', 0),
    ]
    colors = ['#2ECC71' if s > 0 else '#E74C3C' for s in comp_scores]

    bars = ax1.barh(comp_names, comp_scores, color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.set_xlim(-10, 10)
    ax1.set_xlabel('Score')
    ax1.set_title('Component Scores', fontweight='bold')

    for bar, score in zip(bars, comp_scores):
        ax1.text(score + 0.3 if score >= 0 else score - 0.3,
                 bar.get_y() + bar.get_height()/2,
                 f'{score:+.1f}', va='center', fontsize=9)

    # Regime indicator
    ax2 = axes[0, 1]
    ax2.axis('off')

    regime = gold_regime.get('gold_regime', 'unknown')
    regime_score = gold_regime.get('gold_regime_score', 0)
    confidence = gold_regime.get('framework_confidence', 1.0)

    regime_colors = {
        'structural': '#FFD700',
        'transitional': '#FFA500',
        'traditional': '#C0C0C0',
        'structural_weakness': '#CD5C5C'
    }
    color = regime_colors.get(regime, '#808080')

    circle = plt.Circle((0.5, 0.6), 0.3, color=color, alpha=0.5)
    ax2.add_patch(circle)
    ax2.text(0.5, 0.6, regime.upper().replace('_', '\n'),
             ha='center', va='center', fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.2, f'Score: {regime_score:+.2f}\nConfidence: {confidence:.0%}',
             ha='center', va='center', fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Current Regime', fontweight='bold')

    # Input values
    ax3 = axes[1, 0]
    ax3.axis('off')

    inputs = gold_regime.get('inputs', {})
    input_text = f"""
INPUT SCORES

Gold (GCA):      {inputs.get('gold_score', 'N/A')}
DXY Index:       {inputs.get('dxy_score', 'N/A')}
10Y Treasury:    {inputs.get('treasury_score', 'N/A')}
"""
    ax3.text(0.1, 0.8, input_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax3.set_title('Market Inputs', fontweight='bold')

    # Recommendation
    ax4 = axes[1, 1]
    ax4.axis('off')

    rec = gold_regime.get('recommendation', {})
    rec_text = f"""
RECOMMENDATION

Action: {rec.get('action', 'N/A').replace('_', ' ').title()}

{rec.get('description', '')}

Signal Adjustment: {rec.get('signal_adjustment', 0):+.2f}
"""
    rec_color = '#90EE90' if rec.get('signal_adjustment', 0) > 0 else '#FFB6C1' if rec.get('signal_adjustment', 0) < 0 else 'lightgray'
    ax4.text(0.1, 0.8, rec_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=rec_color, alpha=0.3))
    ax4.set_title('Action Recommendation', fontweight='bold')

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])

    # Save
    if output_path is None:
        timestamp_file = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'Gold_Regime_{timestamp_file}.png'

    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)

    print(f"Gold regime chart saved: {output_path}")
    return output_path


# ============================================================================
# MAIN SCANNER
# ============================================================================

def run_ccs_scan(generate_pdf=False, generate_png=False, output_dir=None):
    """
    Execute full CCS scan across all instruments.

    Args:
        generate_pdf: If True, generate a PDF report
        generate_png: If True, generate a PNG image report
        output_dir: Directory for output files (default: current directory)

    Returns:
        Dictionary with results, macro analysis, gold regime, and report paths
    """

    print("\n" + "="*80)
    print("Carnival Core Score (CCS) Scanner - Bloomberg Edition")
    print("Initializing BQL connection and fetching data...")
    print("="*80)

    results_by_category = {}

    for category, tickers in TICKERS.items():
        print(f"\nProcessing {category}... ({len(tickers)} instruments)")
        results = []

        for ticker in tickers:
            try:
                df = calculate_scores_for_ticker(ticker)
                if df is None:
                    continue

                # Extract scores
                current_score = df['Score_EMA'].iloc[-1] if len(df) > 0 else np.nan
                d_minus_1 = df['Score_EMA'].iloc[-2] if len(df) > 1 else np.nan
                d_minus_2 = df['Score_EMA'].iloc[-3] if len(df) > 2 else np.nan
                d_minus_3 = df['Score_EMA'].iloc[-4] if len(df) > 3 else np.nan
                d_minus_4 = df['Score_EMA'].iloc[-5] if len(df) > 4 else np.nan

                # Calculate averages
                avg_5d = df['Score_EMA'].iloc[-5:].mean() if len(df) >= 5 else np.nan
                avg_10d = df['Score_EMA'].iloc[-10:].mean() if len(df) >= 10 else np.nan
                avg_20d = df['Score_EMA'].iloc[-20:].mean() if len(df) >= 20 else np.nan

                results.append({
                    'ticker': ticker,
                    'current_score': current_score,
                    'd_minus_1': d_minus_1,
                    'd_minus_2': d_minus_2,
                    'd_minus_3': d_minus_3,
                    'd_minus_4': d_minus_4,
                    'avg_5d': avg_5d,
                    'avg_10d': avg_10d,
                    'avg_20d': avg_20d,
                })

                print(f"  ✓ {get_display_name(ticker)}")

            except Exception as e:
                print(f"  ✗ {get_display_name(ticker)}: {e}")

        results_by_category[category] = results

    # Print results
    print_results(results_by_category)

    # Macro quadrant analysis
    macro_analysis = analyze_macro_quadrant(results_by_category)

    # Gold-specific regime analysis
    gold_regime = analyze_gold_regime(results_by_category)

    # Generate reports if requested
    report_paths = {}

    if generate_pdf or generate_png:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Determine output directory
        if output_dir is None:
            output_dir = os.getcwd()
        else:
            os.makedirs(output_dir, exist_ok=True)

        if generate_pdf:
            pdf_path = os.path.join(output_dir, f'CCS_Report_{timestamp}.pdf')
            report_paths['pdf'] = generate_report(
                results_by_category, macro_analysis, gold_regime,
                output_path=pdf_path, format='pdf'
            )

        if generate_png:
            png_path = os.path.join(output_dir, f'CCS_Report_{timestamp}.png')
            report_paths['png'] = generate_report(
                results_by_category, macro_analysis, gold_regime,
                output_path=png_path, format='png'
            )

            # Also generate focused gold regime chart
            gold_chart_path = os.path.join(output_dir, f'Gold_Regime_{timestamp}.png')
            report_paths['gold_chart'] = generate_gold_regime_chart(
                gold_regime, output_path=gold_chart_path
            )

    print("\n" + "="*80)
    print(f"Scan complete! Processed {sum(len(r) for r in results_by_category.values())} instruments")
    if report_paths:
        print(f"Reports generated: {', '.join(report_paths.keys())}")
    print("="*80 + "\n")

    return {
        'results_by_category': results_by_category,
        'macro_analysis': macro_analysis,
        'gold_regime': gold_regime,
        'report_paths': report_paths,
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys

    # Parse command line arguments
    generate_pdf = '--pdf' in sys.argv or '-p' in sys.argv
    generate_png = '--png' in sys.argv or '-i' in sys.argv
    generate_both = '--report' in sys.argv or '-r' in sys.argv

    if generate_both:
        generate_pdf = True
        generate_png = True

    # Check for output directory argument
    output_dir = None
    for i, arg in enumerate(sys.argv):
        if arg in ['--output', '-o'] and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]

    # Run scanner
    results = run_ccs_scan(
        generate_pdf=generate_pdf,
        generate_png=generate_png,
        output_dir=output_dir
    )

    # Print usage hint if no reports requested
    if not (generate_pdf or generate_png):
        print("Tip: Run with --pdf, --png, or --report (-r) to generate reports")
        print("     Use --output (-o) <dir> to specify output directory")
