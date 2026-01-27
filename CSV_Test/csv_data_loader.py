"""
CSV Data Loader Module

Provides functions to load CSV data as a replacement for Bloomberg BQL calls.
This allows testing the commodity signal scanner without Bloomberg terminal access.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to CSV files (relative to this module)
CSV_DIR = os.path.dirname(os.path.abspath(__file__))

# Mapping from Bloomberg tickers to CSV data sources
TICKER_TO_OHLV = {
    'GCA Comdty': 'OHLV_gold.csv',
    'SIA Comdty': 'OHLV_silver.csv',
    'PLA Comdty': 'OHLV_platinum.csv',
    'PAA Comdty': 'OHLV_palladium.csv',
    'CLA Comdty': 'OHLV_oil.csv',
}

# Columns available in combined_prices.csv
COMBINED_PRICE_COLUMNS = [
    'GCA Comdty', 'SIA Comdty', 'PLA Comdty', 'PAA Comdty', 'CLA Comdty',
    'DXY Index', 'USDJPY Curncy', 'EURUSD Curncy', 'GBPUSD Curncy',
    'AUDUSD Curncy', 'USDCAD Curncy', 'TUA Comdty', 'FVA Comdty',
    'TYA Comdty', 'USA Comdty', 'ESA Index', 'NQA Index', 'RTYA Index'
]

# Tickers that have corrupted data in early rows (need filtering)
CORRUPTED_TICKERS = ['PAA Comdty', 'CLA Comdty', 'USDJPY Curncy']

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_combined_prices(filter_valid: bool = True) -> pd.DataFrame:
    """
    Load the combined_prices.csv file.

    Parameters:
        filter_valid: If True, filter out rows with corrupted data

    Returns:
        DataFrame with Date index and price columns
    """
    csv_path = os.path.join(CSV_DIR, 'combined_prices.csv')
    df = pd.read_csv(csv_path)

    # Parse Date column
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

    if filter_valid:
        # Filter rows where corrupted columns contain valid numeric data
        # PAA, CLA, USDJPY become valid around row 20 (27/09/2024)
        for col in CORRUPTED_TICKERS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Keep only rows where all key columns are numeric
        df = df.dropna(subset=['GCA Comdty', 'SIA Comdty', 'PLA Comdty'])

    df = df.set_index('Date')
    df = df.sort_index()

    return df


def load_ohlv_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load OHLV data for a specific commodity ticker.

    Parameters:
        ticker: Bloomberg-style ticker (e.g., 'GCA Comdty')

    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume columns
    """
    if ticker not in TICKER_TO_OHLV:
        print(f"Warning: No OHLV file for ticker {ticker}")
        return None

    csv_file = TICKER_TO_OHLV[ticker]
    csv_path = os.path.join(CSV_DIR, csv_file)

    if not os.path.exists(csv_path):
        print(f"Warning: File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # The CSV has columns: ,Date,Open,High,Low,Close,Volume,Ticker
    # First column is unnamed index, Date is the actual date

    # Parse Date column (format: DD/MM/YYYY)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

    # Convert price columns to numeric, coercing errors to NaN
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter out rows where Close is NaN (corrupted rows)
    df = df.dropna(subset=['Close'])

    # Keep only needed columns
    cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[[c for c in cols_to_keep if c in df.columns]]

    df = df.sort_values('Date').reset_index(drop=True)

    return df


def fetch_historical_data(ticker: str, days: int = 90) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data - CSV replacement for BQL function.

    This function mimics the Bloomberg BQL fetch_historical_data() function
    used in commodity_signals.py.

    Parameters:
        ticker: Bloomberg-style ticker (e.g., 'GCA Comdty')
        days: Number of days of history to return

    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume columns
    """
    # First try to load from OHLV file
    if ticker in TICKER_TO_OHLV:
        df = load_ohlv_data(ticker)
        if df is not None and len(df) > 0:
            # Filter to requested date range
            end_date = df['Date'].max()
            start_date = end_date - timedelta(days=days)
            df = df[df['Date'] >= start_date]
            return df

    # Fallback: Create from combined_prices (Close only, no OHLV)
    combined = load_combined_prices()
    if ticker in combined.columns:
        df = pd.DataFrame({
            'Date': combined.index,
            'Close': combined[ticker].values
        })
        df = df.dropna(subset=['Close'])

        # Add placeholder OHLV (use Close for all)
        df['Open'] = df['Close']
        df['High'] = df['Close']
        df['Low'] = df['Close']
        df['Volume'] = 1.0  # Default volume

        # Filter to requested date range
        end_date = df['Date'].max()
        start_date = end_date - timedelta(days=days)
        df = df[df['Date'] >= start_date]

        df = df.reset_index(drop=True)
        return df

    print(f"Warning: No data available for ticker {ticker}")
    return None


def fetch_price_data(ticker: str, days: int = 252) -> Optional[pd.DataFrame]:
    """
    Fetch close price data - CSV replacement for BQL function.

    This function mimics the Bloomberg BQL fetch_price_data() function
    used in correlation_analysis.py.

    Parameters:
        ticker: Bloomberg-style ticker
        days: Number of days of history

    Returns:
        DataFrame with Date, Close, Ticker columns
    """
    combined = load_combined_prices()

    if ticker not in combined.columns:
        print(f"Warning: Ticker {ticker} not found in combined_prices.csv")
        return None

    df = pd.DataFrame({
        'Date': combined.index,
        'Close': combined[ticker].values
    })

    # Forward fill missing values (mimics BQL 'fill': 'prev')
    df['Close'] = df['Close'].fillna(method='ffill')
    df = df.dropna(subset=['Close'])

    # Filter to date range
    if len(df) > 0:
        end_date = df['Date'].max()
        start_date = end_date - timedelta(days=days)
        df = df[df['Date'] >= start_date]

    df['Ticker'] = ticker
    df = df.reset_index(drop=True)

    return df


def fetch_multi_asset_data(tickers: List[str], days: int = 252) -> pd.DataFrame:
    """
    Fetch price data for multiple tickers and merge into returns DataFrame.

    This function mimics the Bloomberg BQL fetch_multi_asset_data() function
    used in correlation_analysis.py.

    Parameters:
        tickers: List of Bloomberg-style tickers
        days: Number of days of history

    Returns:
        DataFrame with Date index and return columns for each ticker
    """
    combined = load_combined_prices()

    # Filter to date range
    if len(combined) > 0:
        end_date = combined.index.max()
        start_date = end_date - timedelta(days=days)
        combined = combined[combined.index >= start_date]

    # Calculate returns for each ticker
    returns_data = {}
    for ticker in tickers:
        if ticker in combined.columns:
            prices = combined[ticker].fillna(method='ffill')
            returns = prices.pct_change()
            returns_data[ticker] = returns

    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()

    return returns_df


def get_available_tickers() -> Dict[str, List[str]]:
    """
    Get list of available tickers from the CSV files.

    Returns:
        Dictionary with 'ohlv' and 'close' ticker lists
    """
    combined = load_combined_prices()

    return {
        'ohlv': list(TICKER_TO_OHLV.keys()),
        'close': [col for col in combined.columns if combined[col].notna().sum() > 100]
    }


def validate_data() -> Dict[str, any]:
    """
    Run data validation checks and return summary.

    Returns:
        Dictionary with validation results
    """
    results = {
        'combined_prices': {},
        'ohlv_files': {},
        'issues': []
    }

    # Check combined_prices.csv
    try:
        combined = load_combined_prices(filter_valid=False)
        results['combined_prices']['total_rows'] = len(combined)
        results['combined_prices']['date_range'] = f"{combined.index.min()} to {combined.index.max()}"
        results['combined_prices']['columns'] = list(combined.columns)

        # Check for corrupted columns
        for col in combined.columns:
            valid_count = combined[col].notna().sum()
            results['combined_prices'][f'{col}_valid_rows'] = valid_count
            if valid_count < len(combined) * 0.5:
                results['issues'].append(f"Column {col} has >50% missing data")
    except Exception as e:
        results['issues'].append(f"Error loading combined_prices.csv: {e}")

    # Check OHLV files
    for ticker, filename in TICKER_TO_OHLV.items():
        try:
            df = load_ohlv_data(ticker)
            if df is not None:
                results['ohlv_files'][ticker] = {
                    'rows': len(df),
                    'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
                    'has_volume': df['Volume'].notna().sum() > 0
                }
            else:
                results['issues'].append(f"Failed to load OHLV for {ticker}")
        except Exception as e:
            results['issues'].append(f"Error loading OHLV for {ticker}: {e}")

    return results


# =============================================================================
# UNIVERSE DEFINITIONS (matching bloomberg_version)
# =============================================================================

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


if __name__ == "__main__":
    # Test the data loader
    print("Testing CSV Data Loader...")
    print("=" * 60)

    # Validate data
    validation = validate_data()

    print("\nCombined Prices:")
    for key, value in validation['combined_prices'].items():
        print(f"  {key}: {value}")

    print("\nOHLV Files:")
    for ticker, info in validation['ohlv_files'].items():
        print(f"  {ticker}: {info['rows']} rows, {info['date_range']}")

    print("\nIssues:")
    for issue in validation['issues']:
        print(f"  - {issue}")

    print("\n" + "=" * 60)
    print("Available tickers:", get_available_tickers())
