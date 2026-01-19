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

# Display names mapping
TICKER_NAMES = {
    'ESA Index': 'ES', 'NQA Index': 'NQ', 'RTYA Index': 'RTY',
    'TUA Comdty': 'ZT', 'FVA Comdty': 'ZF', 'TYA Comdty': 'ZN', 'USA Comdty': 'ZB',
    'DXY Index': 'DXY', 'USDJPY Curncy': 'USD/JPY', 'EURUSD Curncy': 'EUR/USD',
    'GBPUSD Curncy': 'GBP/USD', 'AUDUSD Curncy': 'AUD/USD', 'USDCAD Curncy': 'USD/CAD',
    'CLA Comdty': 'Oil', 'NGA Comdty': 'Nat Gas', 'HGA Comdty': 'Copper',
    'PAA Comdty': 'Palladium', 'PLA Comdty': 'Platinum',
    'GCA Comdty': 'Gold', 'SIA Comdty': 'Silver', 'BTC Curncy': 'Bitcoin',
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


def is_equity_instrument(ticker):
    """Check if instrument is equity (stock/ETF) for options analysis"""
    return 'Equity' in ticker


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
                'Open': bq.data.px_open(),
                'High': bq.data.px_high(),
                'Low': bq.data.px_low(),
                'Close': bq.data.px_last(),
                'Volume': bq.data.volume(),
            },
            with_params={
                'dates': bql.func.range(start_date.strftime('%Y-%m-%d'),
                                       end_date.strftime('%Y-%m-%d'))
            }
        )

        # Execute request
        response = bq.execute(request)

        # Convert to DataFrame
        df = response.combine().reset_index()
        df = df.rename(columns={'DATE': 'Date'})
        df = df.sort_values('Date').reset_index(drop=True)

        return df

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def fetch_options_data(ticker):
    """
    Fetch options flow data using BQL (Phase 2)
    Returns dict with put_call_ratio, iv_percentile, options_volume
    """
    if not is_equity_instrument(ticker):
        return None

    try:
        # Create BQL request for options metrics
        request = bql.Request(
            ticker,
            {
                'put_call_ratio': bq.data.put_call_ratio(),
                'iv_rank': bq.data.historical_volatility_rank(),  # IV percentile
                'options_volume': bq.data.options_volume(),
                'stock_volume': bq.data.volume(),
            }
        )

        # Execute request
        response = bq.execute(request)
        data = response.combine()

        # Extract latest values
        result = {
            'put_call_ratio': data['put_call_ratio'].iloc[-1] if 'put_call_ratio' in data else 1.0,
            'iv_rank': data['iv_rank'].iloc[-1] if 'iv_rank' in data else 50,
            'options_volume': data['options_volume'].iloc[-1] if 'options_volume' in data else 0,
            'stock_volume': data['stock_volume'].iloc[-1] if 'stock_volume' in data else 1,
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

    Options Flow (8%) - Stocks/ETFs only:
    - Put/Call Ratio: 4%
    - IV Rank: 4%
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
        # Put/Call Ratio (4%)
        pc_ratio = options_data['put_call_ratio']
        if pc_ratio < 0.5:
            pc_score = 10  # Very bullish (extreme call buying)
        elif pc_ratio < 0.7:
            pc_score = 5   # Bullish
        elif pc_ratio < 1.0:
            pc_score = 0   # Neutral
        elif pc_ratio < 1.3:
            pc_score = -5  # Bearish
        else:
            pc_score = -10 # Very bearish (extreme put buying)

        pc_normalized = normalize_to_range(pc_score) * 0.04

        # IV Rank (4%) - contextual to price trend
        iv_rank = options_data['iv_rank']
        price_trend = df_row['ROC_10']  # Use 10-day trend for context

        if price_trend > 0:
            # Uptrend: High IV = conviction
            iv_score = (iv_rank - 50) / 5
        else:
            # Downtrend: High IV = panic
            iv_score = -(iv_rank - 50) / 5

        iv_normalized = normalize_to_range(iv_score) * 0.04

        options_score = pc_normalized + iv_normalized

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
    options_data = fetch_options_data(ticker) if is_equity_instrument(ticker) else None

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
    """Print formatted results matching reference style"""

    print("\n" + "="*150)
    print(f"{'CARNIVAL CORE SCORE (CCS) - BLOOMBERG EDITION':^150}")
    print(f"{'Phase 1 (Volume) + Phase 2 (Options Flow) Enabled':^150}")
    print("="*150 + "\n")

    for category, results in results_by_category.items():
        print(f"\n{'='*20} {category} {'='*20}\n")

        # Header
        header = (
            f"{'Ticker':<20} {'Score':>6} {'Sentiment':<25} "
            f"{'5D Avg':>10} {'D-1':>8} {'D-2':>8} {'D-3':>8} {'D-4':>8} "
            f"{'10D Avg':>10} {'20D Avg':>10}"
        )
        print(header)
        print("-" * 150)

        # Sort by current score (descending)
        results_sorted = sorted(results, key=lambda x: x['current_score'] if not pd.isna(x['current_score']) else -999, reverse=True)

        for result in results_sorted:
            display_name = get_display_name(result['ticker'])
            score = result['current_score']
            sentiment, color = classify_sentiment(score)
            reset = '\033[0m'

            # Format scores with colors
            score_str = format_score(score)
            sentiment_str = f"{color}{sentiment}{reset}"

            row = (
                f"{display_name:<20} {score_str} {sentiment_str:<25} "
                f"{result['avg_5d']:>10.1f} {result['d_minus_1']:>8.1f} "
                f"{result['d_minus_2']:>8.1f} {result['d_minus_3']:>8.1f} "
                f"{result['d_minus_4']:>8.1f} {result['avg_10d']:>10.1f} "
                f"{result['avg_20d']:>10.1f}"
            )
            print(row)

        print()


# ============================================================================
# MAIN SCANNER
# ============================================================================

def run_ccs_scan():
    """Execute full CCS scan across all instruments"""

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

    print("\n" + "="*80)
    print(f"Scan complete! Processed {sum(len(r) for r in results_by_category.values())} instruments")
    print("="*80 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_ccs_scan()
