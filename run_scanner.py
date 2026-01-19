#!/usr/bin/env python3
"""
Carnival Core Score (CCS) - Macro Scanner
Standalone execution script
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

TICKERS = {
    'MACRO': [
        'ES=F', 'NQ=F', 'RTY=F',  # Equity Index Futures
        'ZT=F', 'ZF=F', 'ZN=F', 'ZB=F',  # Treasury Futures
        'DX-Y.NYB', 'USDJPY=X', 'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X',  # Currencies
        'CL=F', 'NG=F', 'HG=F', 'PA=F', 'PL=F', 'GC=F', 'SI=F', 'BTC-USD',  # Commodities
    ],
    'SECTORS': [
        'XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLU', 'XLB', 'XLRE',
        'XHB', 'XBI', 'SMH', 'SPHB', 'SPLV',
    ],
    'TOP_STOCKS': [
        'AAPL', 'NVDA', 'MSFT', 'AMZN', 'META', 'TSLA', 'GOOGL',
        'AVGO', 'GOOG', 'BRK-B', 'JPM', 'LLY', 'V', 'XOM',
        'UNH', 'MA', 'COST', 'HD', 'WMT', 'PG', 'NFLX',
        'JNJ', 'BAC', 'ABBV', 'CRM', 'AMD',
    ],
    'WORLD_ETFS': [
        'VEA', 'IEMG', 'EEM', 'ACWI', 'BNDX', 'VWOB',
    ],
}

TICKER_NAMES = {
    'ES=F': 'ES', 'NQ=F': 'NQ', 'RTY=F': 'RTY',
    'ZT=F': 'ZT (2y)', 'ZF=F': 'ZF (5y)', 'ZN=F': 'ZN (10y)', 'ZB=F': 'ZB (30y)',
    'DX-Y.NYB': 'DXY', 'USDJPY=X': 'USDJPY', 'EURUSD=X': 'EURUSD',
    'GBPUSD=X': 'GBPUSD', 'AUDUSD=X': 'AUDUSD', 'USDCAD=X': 'USDCAD',
    'CL=F': 'Oil', 'NG=F': 'Nat Gas', 'HG=F': 'Copper',
    'PA=F': 'Palladium(PA)', 'PL=F': 'Platinum', 'GC=F': 'Gold',
    'SI=F': 'Silver', 'BTC-USD': 'BTC',
    'XLK': 'XLK (Tech)', 'XLV': 'XLV (Health)', 'XLF': 'XLF (Financials)',
    'XLY': 'XLY (Disc)', 'XLC': 'XLC (Comm)', 'XLI': 'XLI (Ind)',
    'XLP': 'XLP (Stap)', 'XLE': 'XLE (Energy)', 'XLU': 'XLU (Util)',
    'XLB': 'XLB (Mat)', 'XLRE': 'XLRE (RE)', 'XHB': 'XHB (Home)',
    'XBI': 'XBI (Bio)', 'SMH': 'SMH (Semi)', 'SPHB': 'SPHB (HiB)',
    'SPLV': 'SPLV (LoVol)',
    'VEA': 'VEA (Dev ex-US Eq)', 'IEMG': 'IEMG (EM Eq)',
    'EEM': 'EEM (EM Eq - Legacy)', 'ACWI': 'ACWI (Global Eq)',
    'BNDX': 'BNDX (Int\'l IG Bonds Hdg)', 'VWOB': 'VWOB (EM USD Sov Bonds)',
}

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_market_data(ticker, period='90d'):
    """Fetch historical price data"""
    try:
        data = yf.download(ticker, period=period, progress=False)

        if data.empty:
            return None

        # Handle multi-index columns (yfinance sometimes returns this)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Check if we have enough data
        if len(data) < 20:
            return None

        # Verify we have Close column
        if 'Close' not in data.columns:
            return None

        return data
    except Exception as e:
        return None

def fetch_all_data(ticker_dict):
    """Fetch data for all tickers"""
    all_data = {}
    print(f"\nFetching data...")

    for category, tickers in ticker_dict.items():
        print(f"\n{category}:")
        for ticker in tickers:
            display_name = TICKER_NAMES.get(ticker, ticker)
            print(f"  {display_name}...", end=' ')
            data = fetch_market_data(ticker)
            if data is not None:
                all_data[ticker] = data
                print(f"✓ ({len(data)} days)")
            else:
                print("✗")

    print(f"\nLoaded: {len(all_data)} instruments")
    return all_data

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_roc(prices, period):
    return ((prices - prices.shift(period)) / prices.shift(period)) * 100

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ma_slope(prices, period):
    ma = prices.rolling(window=period).mean()
    return ((ma - ma.shift(1)) / ma.shift(1)) * 100

def calculate_price_vs_ma(prices, ma_period):
    ma = prices.rolling(window=ma_period).mean()
    return ((prices - ma) / ma) * 100

def calculate_technical_indicators(df):
    """Calculate all technical indicators including volume analysis"""
    close = df['Close']

    # Price-based indicators
    df['ROC_1'] = calculate_roc(close, 1)
    df['ROC_2'] = calculate_roc(close, 2)
    df['ROC_5'] = calculate_roc(close, 5)
    df['ROC_10'] = calculate_roc(close, 10)
    df['ROC_20'] = calculate_roc(close, 20)

    df['SMA_5'] = close.rolling(window=5).mean()
    df['SMA_10'] = close.rolling(window=10).mean()
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()

    df['MA5_Slope'] = calculate_ma_slope(close, 5)
    df['MA10_Slope'] = calculate_ma_slope(close, 10)
    df['MA20_Slope'] = calculate_ma_slope(close, 20)

    df['Price_vs_MA5'] = calculate_price_vs_ma(close, 5)
    df['Price_vs_MA10'] = calculate_price_vs_ma(close, 10)
    df['Price_vs_MA20'] = calculate_price_vs_ma(close, 20)

    df['RSI'] = calculate_rsi(close, 14)

    df['MA_Cross_5_10'] = np.where(df['SMA_5'] > df['SMA_10'], 1, -1)
    df['MA_Cross_10_20'] = np.where(df['SMA_10'] > df['SMA_20'], 1, -1)

    # Volume-based indicators (check if volume exists - forex doesn't have volume)
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        volume = df['Volume']
        df['Vol_SMA_5'] = volume.rolling(window=5).mean()
        df['Vol_SMA_20'] = volume.rolling(window=20).mean()

        # Volume ratio (current vs 20-day average)
        df['Vol_Ratio'] = volume / df['Vol_SMA_20']

        # Volume trend (5-day vs 20-day MA)
        df['Vol_Trend'] = df['Vol_SMA_5'] / df['Vol_SMA_20']

        # Volume-Price correlation (price direction * volume strength)
        df['Vol_Price_Corr'] = np.sign(df['ROC_1']) * (df['Vol_Ratio'] - 1)
    else:
        # For forex or instruments without volume, set neutral values
        df['Vol_Ratio'] = 1.0
        df['Vol_Trend'] = 1.0
        df['Vol_Price_Corr'] = 0.0

    return df

# ============================================================================
# SCORING ALGORITHM
# ============================================================================

def normalize_to_range(values, min_val=-10, max_val=10):
    """Normalize values with more aggressive scaling to match reference output"""
    values = np.clip(values, -100, 100)
    # Changed from /10 to /3 for much stronger signals
    return np.tanh(values / 3) * max_val

def calculate_daily_score(df_row):
    """
    Calculate composite score for a single day with volume conviction.

    New allocation (total 100%):
    - ROC (Price Momentum): 35%
    - MA Trend: 25%
    - Volume & Conviction: 20%
    - RSI: 12%
    - Price vs MA: 8%
    """

    # Component 1: ROC signals (35%) - medium-term trends over daily noise
    roc_score = (
        df_row['ROC_1'] * 0.15 +
        df_row['ROC_2'] * 0.15 +
        df_row['ROC_5'] * 0.25 +
        df_row['ROC_10'] * 0.25 +
        df_row['ROC_20'] * 0.20
    )
    roc_score = normalize_to_range(roc_score) * 0.35

    # Component 2: MA trend signals (25%)
    ma_slope_score = (
        df_row['MA5_Slope'] * 0.40 +
        df_row['MA10_Slope'] * 0.35 +
        df_row['MA20_Slope'] * 0.25
    )
    ma_cross_score = (df_row['MA_Cross_5_10'] + df_row['MA_Cross_10_20']) * 2.5
    ma_score = (normalize_to_range(ma_slope_score) * 0.7 + ma_cross_score * 0.3) * 0.25

    # Component 3: Volume & Conviction (20%)
    # Sub-component 3a: Volume ratio vs average (40% of 20% = 8%)
    vol_ratio_score = normalize_to_range((df_row['Vol_Ratio'] - 1) * 100) * 0.40

    # Sub-component 3b: Volume trend (30% of 20% = 6%)
    vol_trend_score = normalize_to_range((df_row['Vol_Trend'] - 1) * 100) * 0.30

    # Sub-component 3c: Volume-Price correlation (30% of 20% = 6%)
    vol_price_corr_score = normalize_to_range(df_row['Vol_Price_Corr'] * 100) * 0.30

    volume_score = (vol_ratio_score + vol_trend_score + vol_price_corr_score) * 0.20

    # Component 4: RSI momentum (12%)
    rsi_score = ((df_row['RSI'] - 50) / 5)
    rsi_score = np.clip(rsi_score, -10, 10) * 0.12

    # Component 5: Price vs MA (8%)
    price_ma_score = (
        df_row['Price_vs_MA5'] * 0.40 +
        df_row['Price_vs_MA10'] * 0.35 +
        df_row['Price_vs_MA20'] * 0.25
    )
    price_ma_score = normalize_to_range(price_ma_score) * 0.08

    # Combine all components
    total_score = roc_score + ma_score + volume_score + rsi_score + price_ma_score
    return np.clip(total_score, -10, 10)

def calculate_scores(df):
    """Calculate daily scores for entire dataframe"""
    df = df.copy()
    df['Score'] = df.apply(calculate_daily_score, axis=1)
    df['Score'] = df['Score'].ewm(span=3, adjust=False).mean()
    df['Score'] = df['Score'].round(1)
    return df

# ============================================================================
# SENTIMENT & METRICS
# ============================================================================

def classify_sentiment(score):
    """Classify score into sentiment category"""
    if score >= 7.0:
        return ("Very Bullish (L/S)", "green_bold")
    elif score >= 4.0:
        return ("Bullish (L)", "green")
    elif score >= -4.0:
        return ("Neutral (L/S chop)", "white")
    elif score >= -7.0:
        return ("Bearish (S)", "red")
    else:
        return ("Very Bearish (L/S)", "red_bold")

def calculate_rolling_metrics(df):
    """Calculate rolling averages and historical scores"""
    if 'Score' not in df.columns or len(df) < 5:
        return None

    recent_scores = df['Score'].tail(25)

    return {
        'score': recent_scores.iloc[-1] if len(recent_scores) >= 1 else np.nan,
        'score_d1': recent_scores.iloc[-2] if len(recent_scores) >= 2 else np.nan,
        'score_d2': recent_scores.iloc[-3] if len(recent_scores) >= 3 else np.nan,
        'score_d3': recent_scores.iloc[-4] if len(recent_scores) >= 4 else np.nan,
        'score_d4': recent_scores.iloc[-5] if len(recent_scores) >= 5 else np.nan,
        'avg_5d': recent_scores.tail(5).mean() if len(recent_scores) >= 5 else np.nan,
        'avg_10d': recent_scores.tail(10).mean() if len(recent_scores) >= 10 else np.nan,
        'avg_20d': recent_scores.tail(20).mean() if len(recent_scores) >= 20 else np.nan,
    }

def process_all_tickers(market_data):
    """Process all tickers"""
    results = {}
    print("\nProcessing indicators and scores...")

    for ticker, df in market_data.items():
        try:
            df_with_indicators = calculate_technical_indicators(df.copy())
            df_with_scores = calculate_scores(df_with_indicators)
            metrics = calculate_rolling_metrics(df_with_scores)

            if metrics is not None:
                sentiment, color = classify_sentiment(metrics['score'])
                metrics['sentiment'] = sentiment
                metrics['color'] = color
                results[ticker] = metrics
        except:
            continue

    print(f"Processed: {len(results)} instruments")
    return results

# ============================================================================
# OUTPUT FORMATTER
# ============================================================================

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def colorize(text, color_code):
    color_map = {
        'red_bold': f"{Colors.BOLD}{Colors.RED}",
        'red': Colors.RED,
        'green_bold': f"{Colors.BOLD}{Colors.GREEN}",
        'green': Colors.GREEN,
        'white': Colors.WHITE,
    }
    color = color_map.get(color_code, Colors.WHITE)
    return f"{color}{text}{Colors.RESET}"

def format_score(value):
    """Format score value to match reference output"""
    if pd.isna(value):
        return "  "
    # Convert to int if it's a whole number, otherwise 1 decimal
    if abs(value - round(value)) < 0.05:
        return f"{int(round(value))}"
    return f"{value:.1f}"

def print_category_table(category_name, results, ticker_list):
    """Print formatted table for a category"""
    print(f"\n=== Carnival Core Score (CCS) - {category_name} ===")

    # Header line
    header = f"{'Ticker':<20} {'Score':>5} {'Sentiment':<25} {'Avg Score 5D':>12} {'Score D-1':>10} {'Score D-2':>10} {'Score D-3':>10} {'Score D-4':>10} {'Avg Score 10D':>13} {'Avg Score 20D':>13}"
    print(header)
    print("-" * 150)

    for ticker in ticker_list:
        if ticker not in results:
            continue

        metrics = results[ticker]
        display_name = TICKER_NAMES.get(ticker, ticker)

        # Format score with proper alignment
        score_val = metrics['score']
        if pd.isna(score_val):
            score_str = "    "
        elif abs(score_val - round(score_val)) < 0.05:
            score_str = f"{int(round(score_val)):4d}"
        else:
            score_str = f"{score_val:4.1f}"

        sentiment_colored = colorize(metrics['sentiment'], metrics['color'])

        # Format all numeric values
        avg_5d_str = format_score(metrics['avg_5d'])
        d1_str = format_score(metrics['score_d1'])
        d2_str = format_score(metrics['score_d2'])
        d3_str = format_score(metrics['score_d3'])
        d4_str = format_score(metrics['score_d4'])
        avg_10d_str = format_score(metrics['avg_10d'])
        avg_20d_str = format_score(metrics['avg_20d'])

        row = (
            f"{display_name:<20} "
            f"{score_str:>5} "
            f"{sentiment_colored:<35} "  # Extra space for ANSI codes
            f"{avg_5d_str:>12} "
            f"{d1_str:>10} "
            f"{d2_str:>10} "
            f"{d3_str:>10} "
            f"{d4_str:>10} "
            f"{avg_10d_str:>13} "
            f"{avg_20d_str:>13}"
        )

        print(row)

def generate_report(results, ticker_dict):
    """Generate complete CCS report"""
    print("\n" + "=" * 150)
    print("CARNIVAL CORE SCORE (CCS) - DAILY REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 150)

    # Print each category
    category_names = {
        'MACRO': 'MACRO',
        'SECTORS': 'SECTORS',
        'TOP_STOCKS': 'TOP STOCKS',
        'WORLD_ETFS': 'WORLD ETFS (Charts Only)'
    }

    for category in ['MACRO', 'SECTORS', 'TOP_STOCKS', 'WORLD_ETFS']:
        if category in ticker_dict:
            print_category_table(category_names[category], results, ticker_dict[category])

    print("\n" + "=" * 150)
    print("END OF REPORT")
    print("=" * 150)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_ccs_scan():
    """Main function to run complete CCS scan"""
    print("\n" + "=" * 80)
    print("STARTING CARNIVAL CORE SCORE SCAN")
    print("=" * 80)

    market_data = fetch_all_data(TICKERS)
    if len(market_data) == 0:
        print("ERROR: No market data fetched.")
        return None

    results = process_all_tickers(market_data)
    if len(results) == 0:
        print("ERROR: No results generated.")
        return None

    generate_report(results, TICKERS)
    return results

if __name__ == '__main__':
    results = run_ccs_scan()
