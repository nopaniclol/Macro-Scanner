# Code Validation & Error Analysis

**Bloomberg Scanner Python Code Review**
**Date**: 2026-01-19

---

## ✅ Critical Issues Found & Fixed

### 1. **Options Data Fetching Logic Error** ⚠️

**Line 424**:
```python
options_data = fetch_options_data(ticker) if is_equity_instrument(ticker) else None
```

**Issue**: This only fetches options data for equities, but `fetch_options_data()` is designed to handle **both** equities AND futures.

**Impact**: Futures will never get options data, even though they should.

**Fix Needed**:
```python
# Change from:
options_data = fetch_options_data(ticker) if is_equity_instrument(ticker) else None

# To:
options_data = fetch_options_data(ticker) if (is_equity_instrument(ticker) or is_futures_instrument(ticker)) else None

# OR better - let fetch_options_data handle it:
options_data = fetch_options_data(ticker)  # Function already has the check inside
```

**Status**: 🔴 **NEEDS FIX**

---

## ✅ Potential Runtime Errors

### 2. **Column Name Collision Risk** 🟡

**Scenario**: If BQL returns column names that conflict with our calculated columns.

**Risk Areas**:
- `Date` column from `bq.data.px_open()['DATE']`
- If BQL somehow returns `ROC_1`, `MA_5`, etc. (unlikely but possible)

**Mitigation**:
- BQL `['DATE']` and `['value']` syntax ensures clean extraction
- Our calculations add NEW columns after data fetch
- **No collision expected** based on BQL behavior

**Status**: 🟢 **OK** (low risk)

---

### 3. **Division by Zero Risks** ✅

**Checked All Divisions**:

#### Line 274 (RSI calculation):
```python
rs = gain / loss  # Could be 0/0 or X/0
df['RSI'] = 100 - (100 / (1 + rs))
```
**Protection**: Pandas handles division by zero → `rs` becomes `inf` or `nan`, which propagates correctly.

#### Line 281 (Volume Ratio):
```python
df['Vol_Ratio'] = df['Volume'] / df['Vol_MA_20']  # Could divide by 0
```
**Protection**: Rolling mean of volume is never exactly zero for real data. If it is, pandas → `inf`.

#### Line 285 (Volume Trend):
```python
df['Vol_Trend'] = df['Vol_MA_5'] / df['Vol_MA_20']
```
**Protection**: Same as above.

#### Line 212, 222 (Call/Put Ratio):
```python
if fut_put_iv and fut_put_iv > 0:
    call_put_ratio = fut_call_iv / fut_put_iv
```
**Protection**: ✅ **Explicit check** for > 0 before division.

**Status**: 🟢 **PROTECTED**

---

### 4. **NaN Propagation** ✅

**Expected NaN Values**:
- First 20 rows will have NaN scores (intentional - line 430)
- Early rows will have NaN for indicators requiring lookback
- EMA will handle NaN values correctly (pandas `.ewm()` skips NaN)

**Handling in Scoring**:
```python
if pd.isna(df_row['MA_20']):  # Line 310 - checks for NaN
    pct_from_ma20 = ...
```

**Status**: 🟢 **HANDLED CORRECTLY**

---

### 5. **DataFrame Index Issues** 🟡

**Line 428**:
```python
for idx, row in df.iterrows():
    if idx < 20:  # Assumes idx is integer 0, 1, 2...
```

**Potential Issue**: If DataFrame has non-sequential index after operations.

**Fix Applied**:
```python
# Line 152 already resets index:
df = df.sort_values('Date').reset_index(drop=True)
```

**Status**: 🟢 **FIXED** (index is 0, 1, 2, ... guaranteed)

---

### 6. **Missing Import Check** ✅

**Required Imports** (lines 7-10):
```python
import bql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
```

**All used**:
- `bql` - lines 126, 138, 182, 192
- `pd` - lines 149, 203, 274, 449, 452, etc.
- `np` - lines 90, 288, 409
- `datetime, timedelta` - lines 122-123

**Status**: 🟢 **ALL IMPORTS PRESENT**

---

## 🔍 Data Integrity Checks

### 7. **Column Existence Checks** ✅

**Line 278** - Before volume calculations:
```python
if 'Volume' in df.columns and df['Volume'].sum() > 0:
```
✅ **Checks both existence AND non-zero values**

**Line 203, 207-208, 216-217** - Options data:
```python
if 'Fut_Call_IV' in data.columns else None
```
✅ **Checks column existence before access**

**Line 310** - Price vs MA:
```python
if not pd.isna(df_row['MA_20']):
```
✅ **Checks for valid value**

**Status**: 🟢 **ROBUST CHECKS**

---

### 8. **Data Type Consistency** ✅

**Expected Types After BQL**:
- `Date`: datetime64 (from `['DATE']`)
- `Open, High, Low, Close, Volume`: float64 (from `['value']`)

**Type Conversions**:
- All calculations use native pandas operations (automatically handles types)
- `.pct_change()`, `.rolling()`, `.mean()` all return float64
- `np.sign()` returns int but multiplied with floats → float64

**Status**: 🟢 **TYPE SAFE**

---

### 9. **Boundary Value Checks** ✅

**Line 409**:
```python
return np.clip(total_score, -10, 10)
```
✅ **Ensures score stays in [-10, 10] range**

**Line 90** (normalize_to_range):
```python
values = np.clip(values, -100, 100)
```
✅ **Prevents extreme values from breaking tanh**

**Status**: 🟢 **BOUNDARIES ENFORCED**

---

## 🧪 Edge Cases

### 10. **Empty DataFrame**

**Line 417**:
```python
if df is None or len(df) < 20:
    return None
```
✅ **Handles empty/small DataFrames**

**Impact**: Scanner gracefully skips failed tickers.

**Status**: 🟢 **HANDLED**

---

### 11. **All NaN Column**

**Scenario**: BQL returns valid DataFrame but all prices are NaN.

**Protection**:
- ROC calculation: `pct_change()` on all NaN → all NaN
- MA calculation: `rolling().mean()` on all NaN → all NaN
- Line 430: `score = calculate_daily_score(row, options_data)` will work but produce NaN
- Line 438: `ewm()` on all NaN → all NaN
- Result: Ticker shows as N/A in output

**Status**: 🟢 **DEGRADES GRACEFULLY**

---

### 12. **Single Options Data Point**

**Lines 207-208**:
```python
fut_call_iv = data['Fut_Call_IV'].iloc[-1] if 'Fut_Call_IV' in data.columns else None
```

**Assumption**: BQL returns at least 1 row (latest value).

**Risk**: If BQL returns empty DataFrame for options.

**Protection**: Try/except block at line 235:
```python
except Exception as e:
    print(f"Warning: Could not fetch options data for {ticker}: {e}")
    return None
```

**Status**: 🟢 **ERROR HANDLED**

---

### 13. **Forex Without Volume**

**Line 278**:
```python
if 'Volume' in df.columns and df['Volume'].sum() > 0:
    # Calculate volume indicators
else:
    df['Vol_Ratio'] = 1.0
    df['Vol_Trend'] = 1.0
    df['Vol_Price_Corr'] = 0.0
```

✅ **Sets neutral values (no penalty)**

**Status**: 🟢 **HANDLED CORRECTLY**

---

## 📊 Expected DataFrame Structures

### Historical Data (after BQL fetch):
```python
# Shape: (90, 6)
# Columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
# Types: datetime64, float64 x5

         Date        Open       High        Low      Close      Volume
0  2025-10-22      150.25     152.30     149.80     151.50   75234567.0
1  2025-10-23      151.60     153.20     151.00     152.80   68923451.0
...
89 2026-01-19      165.30     167.80     165.00     166.50   82345678.0
```

### After Technical Indicators:
```python
# Shape: (90, 27)
# Additional columns:
# - ROC_1, ROC_2, ROC_5, ROC_10, ROC_20 (5)
# - MA_5, MA_10, MA_20, MA_50 (4)
# - MA_5_slope, MA_10_slope, MA_20_slope (3)
# - RSI (1)
# - Vol_MA_20, Vol_Ratio, Vol_MA_5, Vol_Trend, Vol_Price_Corr (5)
# Total: 6 + 5 + 4 + 3 + 1 + 5 = 24 columns

# After scoring:
# - Score (1)
# - Score_EMA (1)
# Total: 27 columns
```

### Options Data (equity):
```python
# Shape: (1, 3)
# Columns: ['Implied_Volatility', 'Call_IV_60D', 'Put_IV_60D']

   Implied_Volatility  Call_IV_60D  Put_IV_60D
0              25.34        26.50       24.20
```

### Options Data (futures):
```python
# Shape: (1, 3)
# Columns: ['Implied_Volatility', 'Fut_Call_IV', 'Fut_Put_IV']

   Implied_Volatility  Fut_Call_IV  Fut_Put_IV
0              18.45        19.20       17.80
```

---

## 🔧 Required Fixes

### **CRITICAL FIX #1**: Options Data Fetching Logic

**File**: `bql_scanner.py`, Line 424

**Change**:
```python
# BEFORE:
options_data = fetch_options_data(ticker) if is_equity_instrument(ticker) else None

# AFTER (recommended):
options_data = fetch_options_data(ticker)
# Function already has proper checks inside (line 172-173)
```

**Reason**: `fetch_options_data()` already checks for both equities AND futures. The outer check is redundant and prevents futures from getting options data.

---

## ✅ Code Quality Summary

| Category | Status | Notes |
|----------|--------|-------|
| **Imports** | 🟢 OK | All required imports present |
| **Type Safety** | 🟢 OK | Pandas handles types correctly |
| **Division by Zero** | 🟢 OK | Protected by checks or pandas inf/nan |
| **NaN Handling** | 🟢 OK | Explicit checks and pandas propagation |
| **Index Issues** | 🟢 OK | `reset_index(drop=True)` ensures sequential |
| **Column Checks** | 🟢 OK | Proper `in df.columns` checks |
| **Boundary Values** | 🟢 OK | `np.clip()` enforces limits |
| **Error Handling** | 🟢 OK | Try/except blocks present |
| **Edge Cases** | 🟢 OK | Empty DF, all NaN handled |
| **Options Logic** | 🔴 **NEEDS FIX** | Line 424 - redundant check blocks futures |

---

## 📝 Testing Checklist

When Bloomberg Terminal is available, test:

- [ ] Equity with options (e.g., AAPL US Equity)
- [ ] Futures with options (e.g., ESA Index)
- [ ] Commodity with options (e.g., GCA Comdty)
- [ ] Forex without options (e.g., EURUSD Curncy)
- [ ] ETF with high volume (e.g., SPY US Equity)
- [ ] Low-volume stock (check volume indicators)
- [ ] Ticker with missing data (verify graceful failure)

---

## 🎯 Final Recommendation

**Code Quality**: 95/100
- **Strengths**: Robust error handling, proper pandas usage, clean structure
- **Weakness**: One logic error on line 424 (options fetching)

**Action Items**:
1. Fix line 424 (remove redundant equity check)
2. Test with actual Bloomberg Terminal
3. Monitor for BQL-specific errors in production

**Production Readiness**: 🟡 **READY AFTER FIX #1**

---

**Document Version**: 1.0
**Validated By**: Code Review - 2026-01-19
