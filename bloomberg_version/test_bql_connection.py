#!/usr/bin/env python3
"""
BQL Connection Test Script
Quick verification that Bloomberg Terminal connection is working properly
"""

import sys
from datetime import datetime, timedelta

print("="*80)
print("Bloomberg BQL Connection Test")
print("="*80 + "\n")

# Test 1: Import BQL
print("Test 1: Importing BQL library...")
try:
    import bql
    print("✓ BQL imported successfully\n")
except ImportError as e:
    print(f"✗ Failed to import BQL: {e}")
    print("\nInstall BQL with: pip install bql")
    sys.exit(1)

# Test 2: Initialize Service
print("Test 2: Initializing BQL service...")
try:
    bq = bql.Service()
    print("✓ BQL service initialized\n")
except Exception as e:
    print(f"✗ Failed to initialize BQL service: {e}")
    print("\nMake sure Bloomberg Terminal is running and logged in")
    sys.exit(1)

# Test 3: Simple Price Query
print("Test 3: Fetching AAPL last price...")
try:
    request = bql.Request(
        'AAPL US Equity',
        {'price': bq.data.px_last()}
    )
    response = bq.execute(request)
    data = response.combine()
    price = data['price'].iloc[-1]
    print(f"✓ AAPL Last Price: ${price:.2f}\n")
except Exception as e:
    print(f"✗ Failed to fetch price: {e}\n")
    sys.exit(1)

# Test 4: Historical Data
print("Test 4: Fetching 10 days of OHLCV data for AAPL...")
try:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)

    request = bql.Request(
        'AAPL US Equity',
        {
            'Open': bq.data.px_open(),
            'High': bq.data.px_high(),
            'Low': bq.data.px_low(),
            'Close': bq.data.px_last(),
            'Volume': bq.data.volume(),
        },
        with_params={
            'dates': bql.func.range(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        }
    )

    response = bq.execute(request)
    df = response.combine().reset_index()

    print(f"✓ Fetched {len(df)} days of data")
    print(f"  Date range: {df['DATE'].min()} to {df['DATE'].max()}")
    print(f"  Latest close: ${df['Close'].iloc[-1]:.2f}\n")
except Exception as e:
    print(f"✗ Failed to fetch historical data: {e}\n")
    sys.exit(1)

# Test 5: Options Data
print("Test 5: Fetching options data for AAPL...")
try:
    request = bql.Request(
        'AAPL US Equity',
        {
            'put_call_ratio': bq.data.put_call_ratio(),
            'iv_rank': bq.data.historical_volatility_rank(),
            'options_volume': bq.data.options_volume(),
        }
    )

    response = bq.execute(request)
    data = response.combine()

    pc_ratio = data['put_call_ratio'].iloc[-1] if 'put_call_ratio' in data else None
    iv_rank = data['iv_rank'].iloc[-1] if 'iv_rank' in data else None

    if pc_ratio is not None:
        print(f"✓ Put/Call Ratio: {pc_ratio:.2f}")
    else:
        print("⚠ Put/Call Ratio: Not available")

    if iv_rank is not None:
        print(f"✓ IV Rank: {iv_rank:.1f}")
    else:
        print("⚠ IV Rank: Not available")

    print()
except Exception as e:
    print(f"⚠ Options data error (this may be normal): {e}")
    print("  Note: Options data requires specific Bloomberg license\n")

# Test 6: Futures Data
print("Test 6: Fetching futures data (ES)...")
try:
    request = bql.Request(
        'ESA Index',  # S&P 500 E-mini
        {
            'Close': bq.data.px_last(),
            'Volume': bq.data.volume(),
        },
        with_params={
            'dates': bql.func.range('-5d', '0d')
        }
    )

    response = bq.execute(request)
    df = response.combine().reset_index()

    if len(df) > 0:
        print(f"✓ ES Latest Price: {df['Close'].iloc[-1]:.2f}")
        print(f"  Volume: {df['Volume'].iloc[-1]:,.0f}\n")
    else:
        print("⚠ No data returned for ES futures\n")
except Exception as e:
    print(f"⚠ Futures data error: {e}")
    print("  This may indicate ticker format issue or lack of futures license\n")

# Test 7: Currency Data
print("Test 7: Fetching forex data (EUR/USD)...")
try:
    request = bql.Request(
        'EURUSD Curncy',
        {'rate': bq.data.px_last()},
        with_params={
            'dates': bql.func.range('-3d', '0d')
        }
    )

    response = bq.execute(request)
    df = response.combine().reset_index()

    if len(df) > 0:
        print(f"✓ EUR/USD Rate: {df['rate'].iloc[-1]:.4f}\n")
    else:
        print("⚠ No data returned for EUR/USD\n")
except Exception as e:
    print(f"⚠ Forex data error: {e}\n")

# Summary
print("="*80)
print("TEST SUMMARY")
print("="*80)
print("""
If all tests passed (✓):
  → Your Bloomberg connection is fully operational
  → You can run: python3 bql_scanner.py

If some tests failed (✗):
  → Check that Bloomberg Terminal is running
  → Verify you're logged in to Bloomberg Terminal
  → Check Bloomberg API license (BLPAPI <GO>)
  → Contact Bloomberg support if issues persist

If options/futures tests showed warnings (⚠):
  → This may be normal depending on your Bloomberg license
  → Core scanner will work, but some features may be limited
  → Contact your Bloomberg sales rep to add data packages

Next Steps:
  1. If all tests pass, run: python3 bql_scanner.py
  2. Review documentation: README_BLOOMBERG.md
  3. Setup guide: SETUP_GUIDE.md
""")
print("="*80)
