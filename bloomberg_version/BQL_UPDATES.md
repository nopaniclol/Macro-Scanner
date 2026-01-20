# BQL Syntax Updates - Bloomberg Scanner

This document tracks the BQL syntax updates made to align with actual Bloomberg Terminal API behavior.

## Update Date: 2026-01-19

---

## Changes Made

### 1. Historical Data Fetching (`fetch_historical_data`)

#### Previous (Incorrect) Syntax:
```python
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
        'dates': bql.func.range(start_date, end_date)
    }
)

response = bq.execute(request)
df = response.combine().reset_index()
```

#### Updated (Correct) Syntax:
```python
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
        'fill': 'na',
        'dates': bq.func.range(start_date, end_date)
    }
)

response = bq.execute(request)
df = pd.concat([data_item.df() for data_item in response], axis=1)
```

**Key Changes**:
- Separate extraction of `['DATE']` and `['value']` for each data item
- Added `'fill': 'na'` parameter
- Changed from `bq.data.volume()` to `bq.data.px_volume()['value']`
- DataFrame construction uses `pd.concat([data_item.df() for data_item in response], axis=1)`

---

### 2. Options Data Fetching (`fetch_options_data`)

#### Previous (Incorrect) Syntax:
```python
request = bql.Request(
    ticker,
    {
        'put_call_ratio': bq.data.put_call_ratio(),
        'iv_rank': bq.data.historical_volatility_rank(),
        'options_volume': bq.data.options_volume(),
        'stock_volume': bq.data.volume(),
    }
)

response = bq.execute(request)
data = response.combine()

result = {
    'put_call_ratio': data['put_call_ratio'].iloc[-1],
    'iv_rank': data['iv_rank'].iloc[-1],
}
```

#### Updated (Correct) Syntax:

**For Equities**:
```python
request = bql.Request(
    ticker,
    {
        'Implied_Volatility': bq.data.IMPLIED_VOLATILITY(),
        'Call_IV_60D': bq.data.CALL_IMP_VOL_60D(),
        'Put_IV_60D': bq.data.PUT_IMP_VOL_60D(),
    }
)

response = bq.execute(request)
data = pd.concat([data_item.df() for data_item in response], axis=1)

# Calculate put/call ratio from IV
call_iv = data['Call_IV_60D'].iloc[-1]
put_iv = data['Put_IV_60D'].iloc[-1]
put_call_ratio = put_iv / call_iv if call_iv > 0 else 1.0

result = {
    'put_call_ratio': put_call_ratio,
    'implied_volatility': data['Implied_Volatility'].iloc[-1],
}
```

**For Futures**:
```python
request = bql.Request(
    ticker,
    {
        'Implied_Volatility': bq.data.IMPLIED_VOLATILITY(),
        'Fut_Call_IV': bq.data.FUT_CALL_IMPLIED_VOLATILITY(),
        'Fut_Put_IV': bq.data.FUT_PUT_IMPLIED_VOLATILITY(),
    }
)

response = bq.execute(request)
data = pd.concat([data_item.df() for data_item in response], axis=1)

# Calculate put/call ratio from futures IV
fut_call_iv = data['Fut_Call_IV'].iloc[-1]
fut_put_iv = data['Fut_Put_IV'].iloc[-1]
put_call_ratio = fut_put_iv / fut_call_iv if fut_call_iv > 0 else 1.0

result = {
    'put_call_ratio': put_call_ratio,
    'implied_volatility': data['Implied_Volatility'].iloc[-1],
}
```

**Key Changes**:
- Bloomberg does **not** provide pre-calculated `put_call_ratio()`
- Must calculate manually: `PUT_IV / CALL_IV`
- Different fields for futures vs equities:
  - Equities: `CALL_IMP_VOL_60D()` / `PUT_IMP_VOL_60D()`
  - Futures: `FUT_CALL_IMPLIED_VOLATILITY()` / `FUT_PUT_IMPLIED_VOLATILITY()`
- Use absolute `IMPLIED_VOLATILITY()` instead of `historical_volatility_rank()`
- DataFrame construction uses `pd.concat()` method

---

## Updated Helper Functions

### New: `is_futures_instrument()`
```python
def is_futures_instrument(ticker):
    """Check if instrument is futures/commodities"""
    return 'Comdty' in ticker or 'Index' in ticker
```

This helps determine which IV fields to use for options data.

---

## Scoring Algorithm Updates

### Previous:
- Used `iv_rank` (percentile: 0-100 scale)
- Assumed pre-calculated rankings

### Updated:
- Uses `implied_volatility` (absolute value, typically 15-80)
- Normalizes around 30 as center point
- Contextual interpretation:
  - Uptrend + High IV = Conviction
  - Downtrend + High IV = Panic

**Scoring Logic**:
```python
implied_vol = options_data['implied_volatility']
price_trend = df_row['ROC_10']

# Normalize IV (center at 30, typical range 20-60)
iv_deviation = (implied_vol - 30) / 10

if price_trend > 0:
    # Uptrend: High IV = strong conviction/momentum
    iv_score = iv_deviation
else:
    # Downtrend: High IV = panic/capitulation
    iv_score = -iv_deviation

iv_normalized = normalize_to_range(iv_score) * 0.04
```

---

## Coverage Expansion

### Previous:
- Options data **only** for equities (stocks/ETFs)
- Futures had no options analysis

### Updated:
- Options data for **equities** (stocks/ETFs): 48 instruments
- Options data for **futures** (indices/commodities): 21 instruments
- **Total with options**: 69 instruments (all in scanner)
- Only forex pairs excluded (no options markets)

---

## Testing Updates

### Test Script (`test_bql_connection.py`)

**Test 4: Historical Data** - Updated to use correct syntax
**Test 5: Options Data** - Now tests actual BQL fields:
- `IMPLIED_VOLATILITY()`
- `CALL_IMP_VOL_60D()` / `PUT_IMP_VOL_60D()`
- Calculates and displays put/call ratio

---

## Migration Notes

If you created custom code based on the original scanner, update:

1. **All BQL requests**: Use `['DATE']` and `['value']` extraction
2. **DataFrame handling**: Replace `.combine()` with `pd.concat()`
3. **Options metrics**: Calculate ratios manually from IV fields
4. **Field names**: Use uppercase BQL field names (e.g., `IMPLIED_VOLATILITY()`)

---

## Verification

To verify the updates work correctly:

```bash
cd "/Users/lisa/Documents/Macro Scanner/Macro-Scanner/bloomberg_version"

# Test connection and data fetching
python3 test_bql_connection.py

# Run full scanner
python3 bql_scanner.py
```

Expected: All tests pass, scanner processes 69 instruments with options data where available.

---

## References

- Bloomberg BQL Documentation: `DOCS BQNT<GO>` in Terminal
- Field Search: `FLDS<GO>` in Terminal
- Examples provided by user on 2026-01-19

---

**Document Version**: 1.0
**Last Updated**: 2026-01-19
**Status**: Production Ready
