# Carnival Core Score - Bloomberg Terminal Edition

Professional-grade momentum/trend scanner using Bloomberg Query Language (BQL) for institutional data access.

## Key Differences from yfinance Version

### Data Source
- **Bloomberg Terminal**: Institutional-grade data with superior quality and coverage
- **BQL API**: `bql.Service()` for programmatic data retrieval
- **Professional Options Data**: Full options chain, put/call ratios, IV metrics

### Enhanced Features
✅ **Phase 1 (Volume Analysis)**: 20% weight
✅ **Phase 2 (Options Flow)**: 8% weight - **ENABLED** for stocks/ETFs

### Score Allocation (Total 100%)
- **ROC (Price Momentum)**: 35%
- **MA Trend**: 25%
- **Volume & Conviction**: 20% (Phase 1)
- **RSI**: 12%
- **Options Flow**: 8% (Phase 2 - stocks/ETFs only)
  - Put/Call Ratio: 4%
  - IV Rank: 4%

---

## Requirements

### Bloomberg Terminal Access
- Active Bloomberg Terminal subscription
- Bloomberg Python API (`bql`) installed
- Authentication via Bloomberg credentials

### Python Dependencies
```bash
pip install bql pandas numpy
```

---

## Bloomberg Ticker Format

### Equities
- **Stocks**: `AAPL US Equity`, `NVDA US Equity`
- **ETFs**: `XLK US Equity`, `SPY US Equity`

### Futures
- **Equity Indices**: `ESA Index` (S&P 500), `NQA Index` (Nasdaq 100)
- **Commodities**: `GCA Comdty` (Gold), `SIA Comdty` (Silver), `CLA Comdty` (Oil)

### Currencies
- **Forex**: `EURUSD Curncy`, `USDJPY Curncy`
- **Crypto**: `BTC Curncy`

### Indices
- **Cash Indices**: `DXY Index` (US Dollar Index)

---

## Usage

### Daily Execution
```bash
cd "/Users/lisa/Documents/Macro Scanner/Macro-Scanner/bloomberg_version"
python3 bql_scanner.py
```

### Expected Runtime
- **Data Fetch**: 60-90 seconds (69 instruments via BQL)
- **Processing**: 5-10 seconds
- **Total**: ~2 minutes

---

## BQL Data Items Used

### Price Data
- `bq.data.px_open()` - Open price
- `bq.data.px_high()` - High price
- `bq.data.px_low()` - Low price
- `bq.data.px_last()` - Close price
- `bq.data.volume()` - Trading volume

### Options Data (Phase 2)
- `bq.data.put_call_ratio()` - Put/Call volume ratio
- `bq.data.historical_volatility_rank()` - IV percentile
- `bq.data.options_volume()` - Total options volume

### Date Ranges
```python
with_params={
    'dates': bql.func.range('2025-10-01', '2026-01-19')
}
```

---

## Phase 2: Options Flow Analysis (8% Weight)

### Stocks & ETFs Only
Options analysis applies to instruments with "Equity" designation:
- All TOP_STOCKS (26 instruments)
- All SECTORS (16 ETFs)
- All WORLD_ETFS (6 ETFs)

### Components

#### 1. Put/Call Ratio (4%)
**Interpretation**:
- **< 0.5**: Very bullish (extreme call buying) → +10 score
- **0.5-0.7**: Bullish → +5 score
- **0.7-1.0**: Neutral → 0 score
- **1.0-1.3**: Bearish → -5 score
- **> 1.3**: Very bearish (extreme put buying) → -10 score

**Contrarian Signal**: Extreme ratios can indicate potential reversals.

#### 2. IV Rank (4%)
**Contextual Scoring**:
- **Uptrend + High IV**: Bullish conviction (institutions loading calls)
- **Uptrend + Low IV**: Weak conviction
- **Downtrend + High IV**: Panic/capitulation (potential bottom)
- **Downtrend + Low IV**: Complacency

**Calculation**:
```python
if price_trend > 0:
    iv_score = (iv_rank - 50) / 5  # High IV bullish
else:
    iv_score = -(iv_rank - 50) / 5  # High IV bearish
```

### Example Impact

**Scenario 1: Strong Bullish with Options Conviction**
- Base Score: +6.5 (from price, volume, MA, RSI)
- Put/Call: 0.4 → +0.4 points
- IV Rank: 75 (high) in uptrend → +0.4 points
- **Final Score**: +7.3 (Very Bullish) ✅

**Scenario 2: Price Up but Weak Options Signal**
- Base Score: +6.5
- Put/Call: 1.5 → -0.4 points
- IV Rank: 30 (low) in uptrend → -0.2 points
- **Final Score**: +5.9 (Bullish, but cautious) ⚠️

---

## Instruments Without Options Data

### Futures, Commodities, Indices, Forex
These instruments skip Phase 2 (8% weight redistributed to base score):
- MACRO futures (ES, NQ, RTY, etc.)
- Commodities (Gold, Silver, Oil, etc.)
- Forex pairs (EUR/USD, USD/JPY, etc.)
- Indices (DXY)

**Scoring**: Uses Phase 1 (92% base) only - no penalty applied.

---

## Output Format

```
==============================================================================
                    CARNIVAL CORE SCORE (CCS) - BLOOMBERG EDITION
                    Phase 1 (Volume) + Phase 2 (Options Flow) Enabled
==============================================================================

==================== MACRO ====================

Ticker               Score Sentiment                 5D Avg      D-1     D-2     D-3     D-4    10D Avg    20D Avg
----------------------------------------------------------------------------------------------------------------------------------------------
Silver                 7.8 Very Bullish (L/S)           7.5      7.6     7.4     7.2     7.0        7.3        7.1
Gold                   6.9 Bullish (L)                 6.5      6.7     6.3     6.1     5.9        6.4        6.2
ES                     1.2 Neutral (L/S chop)          1.4      1.3     1.5     1.7     1.2        1.3        1.1
Nat Gas               -6.5 Bearish (S)                -6.2     -6.3    -6.0    -5.8    -6.1       -6.0       -5.7

==================== TOP_STOCKS ====================

Ticker               Score Sentiment                 5D Avg      D-1     D-2     D-3     D-4    10D Avg    20D Avg
----------------------------------------------------------------------------------------------------------------------------------------------
NVDA                   8.2 Very Bullish (L/S)           7.9      8.0     7.8     7.6     7.5        7.8        7.6
WMT                    7.1 Very Bullish (L/S)           6.8      7.0     6.7     6.5     6.3        6.7        6.5
CRM                   -5.8 Bearish (S)                -5.5     -5.6    -5.4    -5.2    -5.0       -5.3       -5.1
```

**Color Coding**:
- Green (Bullish/Very Bullish)
- White (Neutral)
- Red (Bearish/Very Bearish)

---

## Editing Tickers

### Location
Edit the `TICKERS` dictionary in [bql_scanner.py](bql_scanner.py) (lines 25-65)

### Adding New Instruments
```python
TICKERS = {
    'MACRO': [
        'ESA Index',      # Existing
        'VXA Index',      # Add VIX futures
    ],
    'TOP_STOCKS': [
        'AAPL US Equity', # Existing
        'SHOP US Equity', # Add Shopify
    ],
}
```

### Display Names
Update `TICKER_NAMES` dictionary for custom display:
```python
TICKER_NAMES = {
    'ESA Index': 'ES',
    'VXA Index': 'VIX',
    # Stocks auto-extract symbol from "AAPL US Equity" → "AAPL"
}
```

---

## Troubleshooting

### BQL Authentication Errors
```bash
# Ensure Bloomberg Terminal is running
# Check BLPAPI connection
python -c "import bql; bq = bql.Service(); print('Connected')"
```

### Missing Options Data
- Some stocks may not have liquid options markets
- Scanner gracefully falls back to Phase 1 (92%) scoring
- Warning message printed: "Could not fetch options data for {ticker}"

### Slow Performance
- BQL fetches take longer than yfinance
- Consider reducing `days=90` to `days=60` for faster runs
- Run during off-peak Bloomberg API hours

---

## Comparison: Bloomberg vs yfinance

| Feature | yfinance Version | Bloomberg Version |
|---------|------------------|-------------------|
| **Data Quality** | Free, delayed | Professional, real-time capable |
| **Options Data** | Limited, manual aggregation | Full options chain, IV, Greeks |
| **Coverage** | Public markets only | All asset classes |
| **Phase 2** | Not implemented | ✅ Fully implemented |
| **Cost** | Free | Bloomberg subscription required |
| **Speed** | ~45 seconds | ~2 minutes |
| **Reliability** | API rate limits | Enterprise SLA |

---

## Advanced BQL Queries

### Custom Date Ranges
```python
# Last 30 days only
with_params={
    'dates': bql.func.range('-30d', '0d')
}

# Specific period
with_params={
    'dates': bql.func.range('2025-12-01', '2026-01-19')
}
```

### Additional Data Items
```python
# Add Open Interest for futures
'open_interest': bq.data.open_interest()

# Add Bid/Ask spread
'bid': bq.data.px_bid()
'ask': bq.data.px_ask()

# Add Greeks for options
'delta': bq.data.delta()
'gamma': bq.data.gamma()
```

---

## Future Enhancements

### Potential Phase 3 Ideas
1. **Dark Pool Flow** (BQL has institutional flow data)
2. **Short Interest** (days to cover, borrow rates)
3. **Analyst Estimates** (earnings revisions, target prices)
4. **Credit Spreads** (CDS, bond spreads for risk-off signals)
5. **Seasonal Patterns** (historical performance by month/quarter)

### Automation
- Schedule via cron/Task Scheduler
- Email delivery with formatted HTML tables
- Database persistence (PostgreSQL, MongoDB)
- Real-time alerts for score changes

---

## Support

### Bloomberg API Documentation
- BQL Reference: `DOCS BQNT<GO>` in Bloomberg Terminal
- Python API: https://bql.bloomberg.com/

### Code Issues
- Check inline comments in [bql_scanner.py](bql_scanner.py)
- Refer to enhanced_scoring_proposal.md for methodology

---

**Version**: 2.0 Bloomberg Edition
**Last Updated**: 2026-01-19
**Status**: Production Ready (requires Bloomberg Terminal access)
**Phase 1**: ✅ Volume Analysis (20%)
**Phase 2**: ✅ Options Flow (8%)
