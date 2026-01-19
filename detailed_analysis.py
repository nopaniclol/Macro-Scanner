#!/usr/bin/env python3
"""
Detailed Score Breakdown Analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import from run_scanner
import sys
sys.path.insert(0, '/Users/lisa/Documents/Macro Scanner/Macro-Scanner')
from run_scanner import (
    TICKERS, TICKER_NAMES, fetch_all_data, calculate_technical_indicators,
    calculate_scores, calculate_rolling_metrics, classify_sentiment,
    normalize_to_range, calculate_daily_score
)

def analyze_score_components(ticker, df_with_scores):
    """
    Break down the score components for a ticker
    """
    if len(df_with_scores) == 0:
        return None

    # Get the most recent row
    latest = df_with_scores.iloc[-1]

    # Recalculate components for analysis
    # ROC component (40%)
    roc_score = (
        latest['ROC_1'] * 0.35 +
        latest['ROC_2'] * 0.25 +
        latest['ROC_5'] * 0.20 +
        latest['ROC_10'] * 0.12 +
        latest['ROC_20'] * 0.08
    )
    roc_normalized = normalize_to_range(roc_score) * 0.40

    # MA trend component (30%)
    ma_slope_score = (
        latest['MA5_Slope'] * 0.40 +
        latest['MA10_Slope'] * 0.35 +
        latest['MA20_Slope'] * 0.25
    )
    ma_cross_score = (latest['MA_Cross_5_10'] + latest['MA_Cross_10_20']) * 2.5
    ma_normalized = (normalize_to_range(ma_slope_score) * 0.7 + ma_cross_score * 0.3) * 0.30

    # RSI component (15%)
    rsi_score = ((latest['RSI'] - 50) / 5) * 0.15
    rsi_score = np.clip(rsi_score, -1.5, 1.5)

    # Price vs MA component (15%)
    price_ma_score = (
        latest['Price_vs_MA5'] * 0.40 +
        latest['Price_vs_MA10'] * 0.35 +
        latest['Price_vs_MA20'] * 0.25
    )
    price_ma_normalized = normalize_to_range(price_ma_score) * 0.15

    return {
        'ticker': ticker,
        'display_name': TICKER_NAMES.get(ticker, ticker),
        'final_score': latest['Score'],
        'date': df_with_scores.index[-1].strftime('%Y-%m-%d'),
        'close_price': float(latest['Close']),

        # Component scores
        'roc_component': roc_normalized,
        'ma_component': ma_normalized,
        'rsi_component': rsi_score,
        'price_ma_component': price_ma_normalized,

        # Raw indicators
        'roc_1d': latest['ROC_1'],
        'roc_5d': latest['ROC_5'],
        'roc_20d': latest['ROC_20'],
        'rsi': latest['RSI'],
        'ma5_slope': latest['MA5_Slope'],
        'ma20_slope': latest['MA20_Slope'],
        'price_vs_ma20': latest['Price_vs_MA20'],
    }

def main():
    print("=" * 120)
    print("CARNIVAL CORE SCORE - DETAILED BREAKDOWN ANALYSIS")
    print("=" * 120)

    # Fetch data
    print("\nFetching market data (this may take a minute)...")
    market_data = fetch_all_data(TICKERS)

    if len(market_data) == 0:
        print("ERROR: No data fetched")
        return

    print(f"Processing {len(market_data)} instruments...")

    # Process all tickers
    all_breakdowns = []

    for ticker, df in market_data.items():
        try:
            df_with_indicators = calculate_technical_indicators(df.copy())
            df_with_scores = calculate_scores(df_with_indicators)

            breakdown = analyze_score_components(ticker, df_with_scores)
            if breakdown:
                all_breakdowns.append(breakdown)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_breakdowns)

    # Display summary
    print(f"\n{'=' * 120}")
    print(f"DATA DATE: {results_df['date'].iloc[0] if len(results_df) > 0 else 'N/A'}")
    print(f"Total instruments analyzed: {len(results_df)}")
    print(f"{'=' * 120}")

    # Top 10 Strongest
    print("\n" + "=" * 120)
    print("TOP 10 STRONGEST (Highest Scores)")
    print("=" * 120)
    top_10 = results_df.nlargest(10, 'final_score')
    print(f"\n{'Rank':<6} {'Ticker':<25} {'Score':>7} {'ROC':>7} {'MA':>7} {'RSI':>7} {'PvMA':>7} {'RSI Val':>8} {'1D ROC%':>9}")
    print("-" * 120)
    for i, row in enumerate(top_10.itertuples(), 1):
        print(f"{i:<6} {row.display_name:<25} {row.final_score:>7.1f} "
              f"{row.roc_component:>7.2f} {row.ma_component:>7.2f} "
              f"{row.rsi_component:>7.2f} {row.price_ma_component:>7.2f} "
              f"{row.rsi:>8.1f} {row.roc_1d:>9.2f}")

    # Bottom 10 Weakest
    print("\n" + "=" * 120)
    print("BOTTOM 10 WEAKEST (Lowest Scores)")
    print("=" * 120)
    bottom_10 = results_df.nsmallest(10, 'final_score')
    print(f"\n{'Rank':<6} {'Ticker':<25} {'Score':>7} {'ROC':>7} {'MA':>7} {'RSI':>7} {'PvMA':>7} {'RSI Val':>8} {'1D ROC%':>9}")
    print("-" * 120)
    for i, row in enumerate(bottom_10.itertuples(), 1):
        print(f"{i:<6} {row.display_name:<25} {row.final_score:>7.1f} "
              f"{row.roc_component:>7.2f} {row.ma_component:>7.2f} "
              f"{row.rsi_component:>7.2f} {row.price_ma_component:>7.2f} "
              f"{row.rsi:>8.1f} {row.roc_1d:>9.2f}")

    # Component contribution analysis
    print("\n" + "=" * 120)
    print("SCORE COMPONENT BREAKDOWN - ALL INSTRUMENTS")
    print("=" * 120)
    print(f"\nColumns: ROC (40%), MA Trend (30%), RSI (15%), Price vs MA (15%)")
    print(f"{'Ticker':<25} {'Final':>7} {'ROC':>7} {'MA':>7} {'RSI':>7} {'PvMA':>7} {'Close':>10} {'Date':<12}")
    print("-" * 120)

    # Sort by final score descending
    sorted_results = results_df.sort_values('final_score', ascending=False)

    for row in sorted_results.itertuples():
        print(f"{row.display_name:<25} {row.final_score:>7.1f} "
              f"{row.roc_component:>7.2f} {row.ma_component:>7.2f} "
              f"{row.rsi_component:>7.2f} {row.price_ma_component:>7.2f} "
              f"{row.close_price:>10.2f} {row.date:<12}")

    print("\n" + "=" * 120)
    print("END OF DETAILED ANALYSIS")
    print("=" * 120)

    # Save to CSV
    csv_file = '/Users/lisa/Documents/Macro Scanner/Macro-Scanner/score_breakdown.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"\nDetailed results saved to: {csv_file}")

if __name__ == '__main__':
    main()
