# Bloomberg Terminal Setup Guide

Step-by-step guide to configure and run the CCS Bloomberg scanner.

---

## Prerequisites

### 1. Bloomberg Terminal Access
- Active Bloomberg subscription
- Bloomberg Terminal installed on your machine
- Valid Bloomberg credentials

### 2. Bloomberg Python API

#### Installation
```bash
# Install Bloomberg Python API (bql)
pip install bql

# Verify installation
python -c "import bql; print('BQL installed successfully')"
```

#### Alternative: Install via Bloomberg Terminal
1. Open Bloomberg Terminal
2. Type `BLPAPI <GO>`
3. Download Python API package
4. Follow Bloomberg's installation instructions

---

## Initial Setup

### Step 1: Verify Bloomberg Connection

#### Start Bloomberg Terminal
- Launch Bloomberg Terminal application
- Log in with your credentials
- Wait for full terminal load (blue screen ready)

#### Test Python Connection
```python
import bql

# Initialize BQL service
bq = bql.Service()

# Test simple query
request = bql.Request('AAPL US Equity', {'price': bq.data.px_last()})
response = bq.execute(request)
print(response.combine())
```

**Expected Output**: DataFrame with AAPL's last price

---

### Step 2: Configure Tickers

#### Edit Bloomberg Tickers
File: [bql_scanner.py](bql_scanner.py), lines 25-65

**Important**: Bloomberg uses different ticker formats:
- yfinance: `AAPL` → Bloomberg: `AAPL US Equity`
- yfinance: `ES=F` → Bloomberg: `ESA Index`
- yfinance: `GC=F` → Bloomberg: `GCA Comdty`

#### Ticker Format Guide

| Asset Class | yfinance Format | Bloomberg Format | Example |
|-------------|-----------------|------------------|---------|
| US Stocks | `AAPL` | `AAPL US Equity` | Apple Inc |
| US ETFs | `SPY` | `SPY US Equity` | S&P 500 ETF |
| Equity Futures | `ES=F` | `ESA Index` | S&P 500 E-mini |
| Commodity Futures | `GC=F` | `GCA Comdty` | Gold Futures |
| Currencies | `EURUSD=X` | `EURUSD Curncy` | EUR/USD |
| Crypto | `BTC-USD` | `BTC Curncy` | Bitcoin |
| Indices | Custom | `DXY Index` | Dollar Index |

#### Find Bloomberg Tickers
In Bloomberg Terminal:
```
{SYMBOL} DES <GO>
```
Example: `AAPL DES <GO>` shows "AAPL US Equity"

---

### Step 3: Test Individual Ticker

Create test script `test_bql.py`:
```python
import bql
from datetime import datetime, timedelta

bq = bql.Service()

# Test single ticker
ticker = 'AAPL US Equity'
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

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
        'dates': bql.func.range(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    }
)

response = bq.execute(request)
df = response.combine()
print(df.tail())
```

**Run Test**:
```bash
python test_bql.py
```

**Expected**: 30 days of AAPL OHLCV data

---

### Step 4: Test Options Data

Create `test_options.py`:
```python
import bql

bq = bql.Service()

ticker = 'AAPL US Equity'

request = bql.Request(
    ticker,
    {
        'put_call_ratio': bq.data.put_call_ratio(),
        'iv_rank': bq.data.historical_volatility_rank(),
        'options_volume': bq.data.options_volume(),
    }
)

try:
    response = bq.execute(request)
    data = response.combine()
    print(data)
    print("\n✓ Options data available for", ticker)
except Exception as e:
    print(f"✗ Options data error: {e}")
```

**Run Test**:
```bash
python test_options.py
```

---

## Running the Scanner

### Full Scan
```bash
cd "/Users/lisa/Documents/Macro Scanner/Macro-Scanner/bloomberg_version"
python3 bql_scanner.py
```

### Expected Output Flow

#### 1. Initialization
```
==============================================================================
Carnival Core Score (CCS) Scanner - Bloomberg Edition
Initializing BQL connection and fetching data...
==============================================================================
```

#### 2. Processing Messages
```
Processing MACRO... (21 instruments)
  ✓ ES
  ✓ NQ
  ✓ Silver
  ✓ Gold
  ...

Processing SECTORS... (16 instruments)
  ✓ XLK
  ✓ XLV
  ...

Processing TOP_STOCKS... (26 instruments)
  ✓ AAPL
  ✓ NVDA
  ...
```

#### 3. Results Display
```
==================== MACRO ====================
Ticker               Score Sentiment                 5D Avg      D-1  ...
------------------------------------------------------------------------------
Silver                 7.8 Very Bullish (L/S)           7.5      7.6  ...
```

---

## Troubleshooting

### Error: "Cannot connect to BQL service"

**Solution**:
1. Ensure Bloomberg Terminal is running (minimized is OK)
2. Check Bloomberg API license: `BLPAPI <GO>`
3. Restart Bloomberg Terminal
4. Reinstall BQL package

---

### Error: "Ticker not found: AAPL US Equity"

**Solution**:
1. Verify ticker format in Bloomberg Terminal:
   ```
   AAPL US DES <GO>
   ```
2. Check ticker is valid and active
3. Some tickers may be delisted or renamed

---

### Error: "Options data unavailable"

**Possible Causes**:
1. Stock doesn't have liquid options (small cap)
2. Bloomberg options license required (check with admin)
3. Options chain not populated yet (early morning runs)

**Workaround**:
Scanner automatically falls back to Phase 1 scoring (92%) for instruments without options data. No manual intervention needed.

---

### Slow Performance

**BQL queries take 60-90 seconds for 69 instruments**

**Optimization Options**:
1. Reduce historical data period:
   ```python
   # In bql_scanner.py, line ~152
   df = fetch_historical_data(ticker, days=60)  # Was 90
   ```

2. Cache data locally:
   ```python
   import pickle
   from pathlib import Path

   cache_file = Path('data_cache.pkl')
   if cache_file.exists():
       with open(cache_file, 'rb') as f:
           data = pickle.load(f)
   else:
       data = fetch_all_data()
       with open(cache_file, 'wb') as f:
           pickle.dump(data, f)
   ```

3. Run during off-peak hours (before 9:30am ET)

---

### Error: "Rate limit exceeded"

Bloomberg API has rate limits per user:
- Wait 60 seconds and retry
- Reduce number of tickers scanned
- Contact Bloomberg support for enterprise limits

---

## Advanced Configuration

### Custom Date Ranges

Edit `fetch_historical_data()` in [bql_scanner.py](bql_scanner.py):

```python
# Use relative dates
with_params={
    'dates': bql.func.range('-60d', '0d')  # Last 60 days
}

# Use absolute dates
with_params={
    'dates': bql.func.range('2025-10-01', '2026-01-19')
}
```

---

### Additional BQL Data Items

Add to `fetch_historical_data()`:

```python
request = bql.Request(
    ticker,
    {
        'Open': bq.data.px_open(),
        'High': bq.data.px_high(),
        'Low': bq.data.px_low(),
        'Close': bq.data.px_last(),
        'Volume': bq.data.volume(),

        # NEW: Additional fields
        'vwap': bq.data.vwap(),                    # VWAP
        'open_interest': bq.data.open_interest(),  # Futures OI
        'short_interest': bq.data.short_interest(), # Stock SI
    },
    ...
)
```

---

### Currency Conversion

BQL supports automatic currency conversion:

```python
request = bql.Request(
    'AAPL US Equity',
    {'price': bq.data.px_last()},
    with_params={
        'currency': 'EUR'  # Convert to EUR
    }
)
```

---

### Pricing Source Selection

For instruments with multiple pricing sources:

```python
request = bql.Request(
    ticker,
    {'price': bq.data.px_last()},
    with_params={
        'pricing_source': 'CBOE'  # Use CBOE prices
    }
)
```

---

## Automation

### Schedule Daily Runs (macOS/Linux)

#### Create Wrapper Script
File: `run_daily_scan.sh`
```bash
#!/bin/bash
cd "/Users/lisa/Documents/Macro Scanner/Macro-Scanner/bloomberg_version"
source /path/to/venv/bin/activate  # If using virtual environment
python3 bql_scanner.py > "output_$(date +%Y%m%d).txt" 2>&1
```

#### Make Executable
```bash
chmod +x run_daily_scan.sh
```

#### Add to Crontab
```bash
crontab -e

# Add line (run daily at 5:30 PM ET after market close)
30 17 * * 1-5 /Users/lisa/Documents/Macro\ Scanner/Macro-Scanner/bloomberg_version/run_daily_scan.sh
```

---

### Schedule Daily Runs (Windows)

#### Create Batch File
File: `run_daily_scan.bat`
```batch
@echo off
cd /d "C:\Users\lisa\Documents\Macro Scanner\Macro-Scanner\bloomberg_version"
python bql_scanner.py > output_%date:~-4,4%%date:~-10,2%%date:~-7,2%.txt 2>&1
```

#### Task Scheduler
1. Open Task Scheduler
2. Create Basic Task
3. Trigger: Daily at 5:30 PM
4. Action: Start Program → `run_daily_scan.bat`
5. Conditions: Run only if Bloomberg Terminal is running

---

## Performance Benchmarks

### Expected Runtime (69 instruments)
- **BQL Data Fetch**: 60-90 seconds
- **Indicator Calculation**: 3-5 seconds
- **Options Data Fetch**: 10-15 seconds (stocks/ETFs only)
- **Scoring & Formatting**: 2-3 seconds
- **Total**: ~90-120 seconds

### Memory Usage
- **Typical**: 200-300 MB
- **Peak**: 500 MB (during BQL bulk fetch)

### Network Bandwidth
- **Per Run**: ~5-10 MB (compressed BQL responses)

---

## Support & Resources

### Bloomberg Documentation
- **BQL Reference Guide**: `DOCS BQNT<GO>` in Terminal
- **Python API Docs**: `DOCS BLPAPI<GO>`
- **Help Desk**: `HELP HELP<GO>`

### Code Documentation
- [README_BLOOMBERG.md](README_BLOOMBERG.md) - Full feature documentation
- [bql_scanner.py](bql_scanner.py) - Inline code comments
- Parent directory: [enhanced_scoring_proposal.md](../enhanced_scoring_proposal.md) - Methodology

### Common BQL Functions
```python
# Date functions
bql.func.range('2025-01-01', '2025-12-31')
bql.func.range('-30d', '0d')

# Aggregation
bq.func.avg()
bq.func.sum()
bq.func.max()

# Field universe
bq.univ.members('SPX Index')  # S&P 500 constituents
```

---

**Setup Guide Version**: 1.0
**Last Updated**: 2026-01-19
**Maintained by**: Carnival Core Score System
