#!/usr/bin/env python3
"""
Bloomberg Data Export Script
Exports all necessary data for commodity signal analysis to CSV files.

Run this script in your Bloomberg Terminal / BQUANT environment to export data,
then use the CSV files for testing on another computer.

Usage:
    %run export_data_to_csv.py

    # Or call the function directly:
    export_all_data(days=252, output_dir='./exported_data')
"""

import bql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Initialize BQL
bq = bql.Service()

# ============================================================================
# UNIVERSE DEFINITIONS
# ============================================================================

COMMODITY_UNIVERSE = {
    'GCA Comdty': 'Gold',
    'SIA Comdty': 'Silver',
    'PLA Comdty': 'Platinum',
    'PAA Comdty': 'Palladium',
    'CLA Comdty': 'Oil',
}

CURRENCY_UNIVERSE = {
    'DXY Index': 'Dollar_Index',
    'USDJPY Curncy': 'USDJPY',
    'EURUSD Curncy': 'EURUSD',
    'GBPUSD Curncy': 'GBPUSD',
    'AUDUSD Curncy': 'AUDUSD',
    'USDCAD Curncy': 'USDCAD',
}

BOND_UNIVERSE = {
    'TUA Comdty': 'Treasury_2Y',
    'FVA Comdty': 'Treasury_5Y',
    'TYA Comdty': 'Treasury_10Y',
    'USA Comdty': 'Treasury_30Y',
}

EQUITY_UNIVERSE = {
    'ESA Index': 'SP500_Futures',
    'NQA Index': 'Nasdaq_Futures',
    'RTYA Index': 'Russell2000_Futures',
}

# Combine all universes
ALL_TICKERS = {
    **COMMODITY_UNIVERSE,
    **CURRENCY_UNIVERSE,
    **BOND_UNIVERSE,
    **EQUITY_UNIVERSE,
}

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def fetch_ohlcv_data(ticker, days=252):
    """
    Fetch OHLCV data for a single ticker.

    Returns DataFrame with Date, Open, High, Low, Close, Volume columns.
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
                'fill': 'prev',
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
        print(f"Error fetching {ticker}: {e}")
        return None


def fetch_price_only(ticker, days=252):
    """
    Fetch just Date and Close price for correlation/ratio analysis.
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
                'fill': 'prev',
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
        print(f"Error fetching {ticker}: {e}")
        return None


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_all_data(days=504, output_dir='./exported_data'):
    """
    Export all data needed for commodity signal analysis.

    Args:
        days: Number of historical days to fetch (default 504 = 2 years)
        output_dir: Directory to save CSV files

    Creates:
        - {output_dir}/ohlcv/{ticker}.csv - Full OHLCV data for commodities
        - {output_dir}/prices/{ticker}.csv - Price data for all assets
        - {output_dir}/combined_prices.csv - All prices in one file
        - {output_dir}/metadata.csv - Export metadata
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/ohlcv", exist_ok=True)
    os.makedirs(f"{output_dir}/prices", exist_ok=True)

    print("="*60)
    print("BLOOMBERG DATA EXPORT")
    print(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Days of History: {days}")
    print(f"Output Directory: {output_dir}")
    print("="*60)

    export_summary = []
    all_prices = {}

    # 1. Export OHLCV data for commodities (needed for technical indicators)
    print("\n1. Exporting OHLCV data for commodities...")
    for ticker, name in COMMODITY_UNIVERSE.items():
        print(f"   Fetching {name} ({ticker})...")
        df = fetch_ohlcv_data(ticker, days)

        if df is not None and len(df) > 0:
            filename = f"{output_dir}/ohlcv/{name}.csv"
            df.to_csv(filename, index=False)
            print(f"   -> Saved {len(df)} rows to {filename}")
            export_summary.append({
                'ticker': ticker,
                'name': name,
                'type': 'commodity',
                'data_type': 'ohlcv',
                'rows': len(df),
                'start_date': str(df['Date'].iloc[0]),
                'end_date': str(df['Date'].iloc[-1]),
                'file': filename,
            })
            all_prices[ticker] = df[['Date', 'Close', 'Ticker']]
        else:
            print(f"   -> FAILED to fetch {ticker}")

    # 2. Export price data for currencies
    print("\n2. Exporting price data for currencies...")
    for ticker, name in CURRENCY_UNIVERSE.items():
        print(f"   Fetching {name} ({ticker})...")
        df = fetch_price_only(ticker, days)

        if df is not None and len(df) > 0:
            filename = f"{output_dir}/prices/{name}.csv"
            df.to_csv(filename, index=False)
            print(f"   -> Saved {len(df)} rows to {filename}")
            export_summary.append({
                'ticker': ticker,
                'name': name,
                'type': 'currency',
                'data_type': 'price',
                'rows': len(df),
                'start_date': str(df['Date'].iloc[0]),
                'end_date': str(df['Date'].iloc[-1]),
                'file': filename,
            })
            all_prices[ticker] = df
        else:
            print(f"   -> FAILED to fetch {ticker}")

    # 3. Export price data for bonds
    print("\n3. Exporting price data for bonds...")
    for ticker, name in BOND_UNIVERSE.items():
        print(f"   Fetching {name} ({ticker})...")
        df = fetch_price_only(ticker, days)

        if df is not None and len(df) > 0:
            filename = f"{output_dir}/prices/{name}.csv"
            df.to_csv(filename, index=False)
            print(f"   -> Saved {len(df)} rows to {filename}")
            export_summary.append({
                'ticker': ticker,
                'name': name,
                'type': 'bond',
                'data_type': 'price',
                'rows': len(df),
                'start_date': str(df['Date'].iloc[0]),
                'end_date': str(df['Date'].iloc[-1]),
                'file': filename,
            })
            all_prices[ticker] = df
        else:
            print(f"   -> FAILED to fetch {ticker}")

    # 4. Export price data for equities
    print("\n4. Exporting price data for equities...")
    for ticker, name in EQUITY_UNIVERSE.items():
        print(f"   Fetching {name} ({ticker})...")
        df = fetch_price_only(ticker, days)

        if df is not None and len(df) > 0:
            filename = f"{output_dir}/prices/{name}.csv"
            df.to_csv(filename, index=False)
            print(f"   -> Saved {len(df)} rows to {filename}")
            export_summary.append({
                'ticker': ticker,
                'name': name,
                'type': 'equity',
                'data_type': 'price',
                'rows': len(df),
                'start_date': str(df['Date'].iloc[0]),
                'end_date': str(df['Date'].iloc[-1]),
                'file': filename,
            })
            all_prices[ticker] = df
        else:
            print(f"   -> FAILED to fetch {ticker}")

    # 5. Create combined prices file (wide format)
    print("\n5. Creating combined prices file...")
    if all_prices:
        # Merge all price series on Date
        combined = None
        for ticker, df in all_prices.items():
            df_price = df[['Date', 'Close']].copy()
            df_price = df_price.rename(columns={'Close': ticker})

            if combined is None:
                combined = df_price
            else:
                combined = pd.merge(combined, df_price, on='Date', how='outer')

        combined = combined.sort_values('Date').reset_index(drop=True)
        combined_file = f"{output_dir}/combined_prices.csv"
        combined.to_csv(combined_file, index=False)
        print(f"   -> Saved combined prices ({len(combined)} rows, {len(all_prices)} columns)")

    # 6. Save export metadata
    print("\n6. Saving export metadata...")
    metadata_df = pd.DataFrame(export_summary)
    metadata_df['export_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    metadata_df['days_requested'] = days
    metadata_file = f"{output_dir}/metadata.csv"
    metadata_df.to_csv(metadata_file, index=False)

    # Print summary
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"Total tickers exported: {len(export_summary)}")
    print(f"Output directory: {output_dir}")
    print("\nFiles created:")
    print(f"  - OHLCV files: {output_dir}/ohlcv/")
    print(f"  - Price files: {output_dir}/prices/")
    print(f"  - Combined prices: {combined_file}")
    print(f"  - Metadata: {metadata_file}")
    print("\nTo use on another computer:")
    print("  1. Copy the entire 'exported_data' folder")
    print("  2. Use commodity_signals_csv.py instead of commodity_signals.py")
    print("="*60)

    return metadata_df


def export_sample_data(output_dir='./exported_data'):
    """
    Export a smaller sample (90 days) for quick testing.
    """
    return export_all_data(days=90, output_dir=output_dir)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Export 2 years of data by default
    export_all_data(days=504, output_dir='./exported_data')
