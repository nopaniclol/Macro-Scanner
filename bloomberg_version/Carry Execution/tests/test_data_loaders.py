"""
Test Data Loaders

Tests for FreeDataLoader and BloombergDataLoader (if available).
Run with: pytest tests/test_data_loaders.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.free_data_loader import FreeDataLoader, get_data_loader


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def free_loader():
    """Create a FreeDataLoader instance."""
    return FreeDataLoader()


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        'project': {
            'data_mode': 'free',
        },
        'api_keys': {
            'fred': None,  # Will work without API key for basic data
        }
    }


# =============================================================================
# FX SPOT TESTS
# =============================================================================

class TestFXSpot:
    """Test FX spot data fetching."""

    def test_fetch_eurusd_100_days(self, free_loader):
        """Test fetching 100 days of EURUSD data."""
        df = free_loader.fetch_fx_spot('EURUSD', days=100)

        # Basic checks
        assert not df.empty, "DataFrame should not be empty"
        assert len(df) >= 50, f"Expected at least 50 rows, got {len(df)}"

        # Column checks
        assert 'Date' in df.columns, "Date column should exist"
        assert 'Close' in df.columns, "Close column should exist"
        assert 'Ticker' in df.columns, "Ticker column should exist"

        # Data quality checks
        assert df['Close'].notna().all(), "Close should not have NaN values"
        assert (df['Close'] > 0).all(), "Close prices should be positive"
        assert df['Close'].dtype in [np.float64, np.float32, float], "Close should be float"

        # Value sanity check (EURUSD typically between 0.80 and 1.50)
        assert df['Close'].min() > 0.50, f"EURUSD min {df['Close'].min()} seems too low"
        assert df['Close'].max() < 2.00, f"EURUSD max {df['Close'].max()} seems too high"

        print(f"\n✓ EURUSD: {len(df)} rows, range [{df['Close'].min():.4f}, {df['Close'].max():.4f}]")

    def test_fetch_usdjpy_100_days(self, free_loader):
        """Test fetching 100 days of USDJPY data."""
        df = free_loader.fetch_fx_spot('USDJPY', days=100)

        assert not df.empty, "DataFrame should not be empty"
        assert len(df) >= 50, f"Expected at least 50 rows, got {len(df)}"

        # USDJPY sanity check (typically between 100 and 160)
        assert df['Close'].min() > 80, f"USDJPY min {df['Close'].min()} seems too low"
        assert df['Close'].max() < 200, f"USDJPY max {df['Close'].max()} seems too high"

        print(f"\n✓ USDJPY: {len(df)} rows, range [{df['Close'].min():.2f}, {df['Close'].max():.2f}]")

    def test_fetch_multiple_fx_pairs(self, free_loader):
        """Test fetching multiple FX pairs."""
        pairs = ['EURUSD', 'USDJPY', 'GBPUSD']
        results = free_loader.fetch_multiple(pairs, asset_type='fx', days=100)

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        for pair in pairs:
            assert pair in results, f"{pair} should be in results"
            df = results[pair]
            assert not df.empty, f"{pair} DataFrame should not be empty"

        print(f"\n✓ Multiple FX: fetched {len(results)} pairs successfully")


# =============================================================================
# METALS TESTS
# =============================================================================

class TestMetals:
    """Test metals data fetching."""

    def test_fetch_xauusd_100_days(self, free_loader):
        """Test fetching 100 days of Gold data."""
        df = free_loader.fetch_metal_spot('XAUUSD', days=100)

        assert not df.empty, "DataFrame should not be empty"
        assert len(df) >= 50, f"Expected at least 50 rows, got {len(df)}"

        # Column checks
        assert 'Date' in df.columns
        assert 'Close' in df.columns

        # Data quality
        assert df['Close'].notna().all(), "Close should not have NaN values"

        # Gold sanity check (typically between $1500 and $3000)
        assert df['Close'].min() > 1000, f"Gold min ${df['Close'].min():.2f} seems too low"
        assert df['Close'].max() < 5000, f"Gold max ${df['Close'].max():.2f} seems too high"

        print(f"\n✓ XAUUSD: {len(df)} rows, range [${df['Close'].min():.2f}, ${df['Close'].max():.2f}]")


# =============================================================================
# OHLCV TESTS
# =============================================================================

class TestOHLCV:
    """Test OHLCV data fetching."""

    def test_fetch_eurusd_ohlcv(self, free_loader):
        """Test fetching EURUSD OHLCV data."""
        df = free_loader.fetch_fx_ohlcv('EURUSD', days=100)

        assert not df.empty, "DataFrame should not be empty"

        # Check all OHLCV columns exist
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            assert col in df.columns, f"{col} column should exist"

        # OHLC relationship checks
        assert (df['High'] >= df['Low']).all(), "High should be >= Low"
        assert (df['High'] >= df['Open']).all(), "High should be >= Open"
        assert (df['High'] >= df['Close']).all(), "High should be >= Close"
        assert (df['Low'] <= df['Open']).all(), "Low should be <= Open"
        assert (df['Low'] <= df['Close']).all(), "Low should be <= Close"

        print(f"\n✓ EURUSD OHLCV: {len(df)} rows, OHLC relationships valid")


# =============================================================================
# UNIFIED INTERFACE TESTS
# =============================================================================

class TestUnifiedInterface:
    """Test the unified fetch_spot interface."""

    def test_auto_detect_fx(self, free_loader):
        """Test auto-detection of FX assets."""
        df = free_loader.fetch_spot('EURUSD', asset_type='auto', days=50)
        assert not df.empty
        assert 'Close' in df.columns

    def test_auto_detect_metal(self, free_loader):
        """Test auto-detection of metal assets."""
        df = free_loader.fetch_spot('XAUUSD', asset_type='auto', days=50)
        assert not df.empty
        assert 'Close' in df.columns

    def test_explicit_asset_type(self, free_loader):
        """Test explicit asset type specification."""
        df = free_loader.fetch_spot('EURUSD', asset_type='fx', days=50)
        assert not df.empty


# =============================================================================
# DATA LOADER FACTORY TESTS
# =============================================================================

class TestDataLoaderFactory:
    """Test the get_data_loader factory function."""

    def test_free_loader_from_config(self, test_config):
        """Test getting free loader from config."""
        loader = get_data_loader(test_config)
        assert isinstance(loader, FreeDataLoader)

    def test_loader_is_available(self, test_config):
        """Test that loader reports availability correctly."""
        loader = get_data_loader(test_config)
        assert loader.is_available() == True


# =============================================================================
# DATA QUALITY TESTS
# =============================================================================

class TestDataQuality:
    """Test data quality and consistency."""

    def test_date_ordering(self, free_loader):
        """Test that dates are in ascending order."""
        df = free_loader.fetch_fx_spot('EURUSD', days=100)
        dates = df['Date'].tolist()
        assert dates == sorted(dates), "Dates should be in ascending order"

    def test_no_duplicate_dates(self, free_loader):
        """Test that there are no duplicate dates."""
        df = free_loader.fetch_fx_spot('EURUSD', days=100)
        assert df['Date'].duplicated().sum() == 0, "Should have no duplicate dates"

    def test_recent_data(self, free_loader):
        """Test that data is reasonably recent."""
        df = free_loader.fetch_fx_spot('EURUSD', days=100)
        latest_date = pd.to_datetime(df['Date'].iloc[-1])
        days_ago = (datetime.now() - latest_date).days

        # Should be within last 5 days (accounting for weekends/holidays)
        assert days_ago <= 5, f"Latest data is {days_ago} days old"


# =============================================================================
# QUICK DEMO FUNCTION (not a test)
# =============================================================================

def demo_100_days_eurusd():
    """
    Demo function to fetch and display 100 days of EURUSD data.
    Run directly: python tests/test_data_loaders.py
    """
    print("\n" + "=" * 60)
    print("DEMO: Fetching 100 days of EURUSD data")
    print("=" * 60)

    loader = FreeDataLoader()
    df = loader.fetch_fx_spot('EURUSD', days=100)

    print(f"\nData Shape: {df.shape}")
    print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Price Range: {df['Close'].min():.4f} to {df['Close'].max():.4f}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print(f"\nLast 5 rows:")
    print(df.tail().to_string(index=False))

    # Check for NaN
    nan_count = df['Close'].isna().sum()
    print(f"\nNaN values in Close column: {nan_count}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)

    return df


if __name__ == '__main__':
    # Run demo
    demo_100_days_eurusd()
