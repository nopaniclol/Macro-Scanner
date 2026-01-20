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
    import pandas as pd
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)

    request = bql.Request(
        'AAPL US Equity',
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

    print(f"✓ Fetched {len(df)} days of data")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
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
            'Implied_Volatility': bq.data.IMPLIED_VOLATILITY(),
            'Call_IV_60D': bq.data.CALL_IMP_VOL_60D(),
            'Put_IV_60D': bq.data.PUT_IMP_VOL_60D(),
        }
    )

    response = bq.execute(request)
    data = pd.concat([data_item.df() for data_item in response], axis=1)

    implied_vol = data['Implied_Volatility'].iloc[-1] if 'Implied_Volatility' in data.columns else None
    call_iv = data['Call_IV_60D'].iloc[-1] if 'Call_IV_60D' in data.columns else None
    put_iv = data['Put_IV_60D'].iloc[-1] if 'Put_IV_60D' in data.columns else None

    if implied_vol is not None:
        print(f"✓ Implied Volatility: {implied_vol:.2f}")
    else:
        print("⚠ Implied Volatility: Not available")

    if call_iv is not None and put_iv is not None and call_iv > 0:
        pc_ratio = put_iv / call_iv
        print(f"✓ Put/Call Ratio (IV): {pc_ratio:.2f}")
    else:
        print("⚠ Put/Call Ratio: Not available")

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
