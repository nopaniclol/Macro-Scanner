#!/usr/bin/env python3
"""
Test script to run CCS Scanner
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Test basic imports and functionality
print("=" * 80)
print("Testing CCS Scanner Components")
print("=" * 80)

# Test 1: Import check
print("\n[1/3] Testing imports...")
try:
    print(f"  ✓ yfinance: {yf.__version__}")
    print(f"  ✓ pandas: {pd.__version__}")
    print(f"  ✓ numpy: {np.__version__}")
except Exception as e:
    print(f"  ✗ Import error: {e}")
    exit(1)

# Test 2: Data fetch
print("\n[2/3] Testing data fetch...")
try:
    test_data = yf.download('AAPL', period='90d', progress=False)
    if len(test_data) > 0:
        print(f"  ✓ Successfully fetched {len(test_data)} days of AAPL data")
        print(f"  ✓ Date range: {test_data.index[0].date()} to {test_data.index[-1].date()}")
    else:
        print("  ✗ No data received")
        exit(1)
except Exception as e:
    print(f"  ✗ Data fetch error: {e}")
    exit(1)

# Test 3: Sample indicator calculation
print("\n[3/3] Testing indicator calculation...")
try:
    close = test_data['Close']
    roc_1 = ((close - close.shift(1)) / close.shift(1)) * 100
    sma_20 = close.rolling(window=20).mean()

    roc_val = float(roc_1.iloc[-1])
    sma_val = float(sma_20.iloc[-1])
    price_val = float(close.iloc[-1])

    print(f"  ✓ ROC (1-day): Latest = {roc_val:.2f}%")
    print(f"  ✓ SMA (20-day): Latest = ${sma_val:.2f}")
    print(f"  ✓ Current Price: ${price_val:.2f}")
except Exception as e:
    print(f"  ✗ Calculation error: {e}")
    exit(1)

print("\n" + "=" * 80)
print("All tests passed! Ready to run full scanner.")
print("=" * 80)
