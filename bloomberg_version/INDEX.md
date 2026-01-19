# Bloomberg Terminal Edition - Documentation Index

**Carnival Core Score (CCS) Scanner - Professional Version**

This folder contains the Bloomberg Terminal implementation with full Phase 1 (Volume) and Phase 2 (Options Flow) analysis.

---

## Quick Start

### 1. Prerequisites Check
- Bloomberg Terminal subscription active
- Bloomberg Terminal running and logged in
- Python 3.7+ installed

### 2. Test Connection
```bash
cd "/Users/lisa/Documents/Macro Scanner/Macro-Scanner/bloomberg_version"
python3 test_bql_connection.py
```

### 3. Run Scanner
```bash
python3 bql_scanner.py
```

---

## Documentation Files

### 📘 [README_BLOOMBERG.md](README_BLOOMBERG.md)
**Main documentation** covering:
- Feature overview
- Score allocation (with Phase 2 options flow)
- BQL data items used
- Output format and interpretation
- Ticker format guide
- Performance benchmarks

**Read this first** to understand what the Bloomberg version does.

---

### 🔧 [SETUP_GUIDE.md](SETUP_GUIDE.md)
**Step-by-step setup instructions**:
- Bloomberg API installation
- Connection verification
- Ticker configuration
- Troubleshooting common errors
- Automation setup (cron/Task Scheduler)
- Advanced BQL queries

**Use this** to get the scanner running on your system.

---

### 📊 [VERSION_COMPARISON.md](VERSION_COMPARISON.md)
**Detailed comparison** between yfinance and Bloomberg versions:
- Feature matrix
- Data quality differences
- Performance benchmarks
- Score accuracy comparison
- Cost-benefit analysis
- Migration guide

**Read this** to understand why you'd use Bloomberg vs free version.

---

### 💻 [bql_scanner.py](bql_scanner.py)
**Main scanner script** - production-ready Python code:
- BQL data fetching for OHLCV
- Options flow data retrieval (Phase 2)
- Technical indicator calculation
- Composite scoring algorithm
- Terminal output formatting

**Run this** to perform daily scans.

---

### 🧪 [test_bql_connection.py](test_bql_connection.py)
**Connection test script**:
- Tests BQL library import
- Verifies Bloomberg Terminal connection
- Validates data access for stocks, futures, forex
- Checks options data availability
- Provides diagnostic output

**Run this first** before attempting full scan.

---

## Key Features

### ✅ Phase 1: Volume Analysis (20% weight)
- Volume ratio vs 20-day average
- Volume trend (5-day vs 20-day MA)
- Volume-price correlation
- Works for all instruments with volume data

### ✅ Phase 2: Options Flow (8% weight)
- **Put/Call Ratio** (4%): Institutional sentiment indicator
- **IV Rank** (4%): Implied volatility percentile for conviction
- **Stocks/ETFs only**: 48 instruments in current configuration
- **Automatically skipped** for futures, forex, commodities

---

## File Structure

```
bloomberg_version/
│
├── INDEX.md                    ← You are here
├── README_BLOOMBERG.md         ← Feature documentation
├── SETUP_GUIDE.md              ← Installation & setup
├── VERSION_COMPARISON.md       ← yfinance vs Bloomberg
│
├── bql_scanner.py              ← Main scanner (run this)
└── test_bql_connection.py      ← Connection test (run first)
```

---

## Workflow

### First-Time Setup
1. ✅ Verify Bloomberg Terminal access
2. ✅ Run `test_bql_connection.py` to validate setup
3. ✅ Read [SETUP_GUIDE.md](SETUP_GUIDE.md) for configuration
4. ✅ Edit ticker list in `bql_scanner.py` if needed
5. ✅ Run `bql_scanner.py` for first scan

### Daily Usage
1. Launch Bloomberg Terminal (if not running)
2. Run: `python3 bql_scanner.py`
3. Review color-coded output
4. Interpret scores with [README_BLOOMBERG.md](README_BLOOMBERG.md) reference

### Troubleshooting
1. Check [SETUP_GUIDE.md](SETUP_GUIDE.md) troubleshooting section
2. Run `test_bql_connection.py` to diagnose issues
3. Verify Bloomberg Terminal is active
4. Check Bloomberg API license (BLPAPI <GO>)

---

## Comparison to Parent Directory

### Parent Directory (yfinance version)
**Location**: `/Users/lisa/Documents/Macro Scanner/Macro-Scanner/`

**Files**:
- `run_scanner.py` - yfinance-based scanner
- `Macro Scanner.ipynb` - Jupyter notebook version
- `enhanced_scoring_proposal.md` - Original Phase 1+2 proposal
- `IMPLEMENTATION_SUMMARY.md` - yfinance implementation summary

**Features**:
- ✅ Phase 1 (Volume): Implemented
- ❌ Phase 2 (Options): Not implemented
- Data source: Yahoo Finance (free)
- Speed: ~45 seconds
- Cost: $0

---

### This Directory (Bloomberg version)
**Location**: `/Users/lisa/Documents/Macro Scanner/Macro-Scanner/bloomberg_version/`

**Features**:
- ✅ Phase 1 (Volume): Implemented
- ✅ Phase 2 (Options): Fully implemented
- Data source: Bloomberg Terminal (professional)
- Speed: ~90-120 seconds
- Cost: ~$24,000/year (Bloomberg subscription)

**Advantage**: Superior data quality + options flow insights

---

## Score Interpretation Guide

### Score Range: -10 to +10

| Score Range | Sentiment | Color | Meaning |
|-------------|-----------|-------|---------|
| **7.0 to 10.0** | Very Bullish (L/S) | Bold Green | Strong uptrend with conviction |
| **4.0 to 6.9** | Bullish (L) | Green | Uptrend |
| **-4.0 to 3.9** | Neutral (L/S chop) | White | Range-bound, no clear trend |
| **-7.0 to -4.1** | Bearish (S) | Red | Downtrend |
| **-10.0 to -7.0** | Very Bearish (L/S) | Bold Red | Strong downtrend |

### Options Flow Impact (Phase 2)

**For stocks/ETFs with options data:**

- **High score + Bullish options flow**: Maximum conviction ✅
  - Example: Score 7.5, P/C ratio 0.4, IV rank 80 → Strong buy signal

- **High score + Bearish options flow**: Caution ⚠️
  - Example: Score 7.0, P/C ratio 1.6, IV rank 25 → Weak breakout, be careful

- **Low score + Bullish options flow**: Potential reversal 🔄
  - Example: Score -6.0, P/C ratio 0.3, IV rank 15 → Institutions accumulating

---

## Common Questions

### Q: Do I need Bloomberg Terminal to run this?
**A**: Yes. The Bloomberg version requires an active Bloomberg Terminal subscription and the BQL Python API. If you don't have Bloomberg access, use the yfinance version in the parent directory.

---

### Q: Can I run both versions simultaneously?
**A**: Yes! They're completely independent. You can compare results:
```bash
# Terminal 1: yfinance version
cd "/Users/lisa/Documents/Macro Scanner/Macro-Scanner"
python3 run_scanner.py

# Terminal 2: Bloomberg version
cd "/Users/lisa/Documents/Macro Scanner/Macro-Scanner/bloomberg_version"
python3 bql_scanner.py
```

---

### Q: Why is the Bloomberg version slower?
**A**: Bloomberg queries are more comprehensive:
- Fetches higher-quality data with validation
- Retrieves additional options data for 48 equities
- Enterprise API overhead for security/compliance
- Trade-off for institutional-grade accuracy

---

### Q: What if options data fails for some stocks?
**A**: The scanner gracefully handles missing options data:
- Automatically falls back to Phase 1 scoring (92%)
- No error or crash
- Warning message printed to console
- Other stocks continue to use Phase 2 normally

---

### Q: How do I customize the ticker list?
**A**: Edit `bql_scanner.py` lines 25-65:
1. Use Bloomberg ticker format (e.g., `AAPL US Equity`)
2. Add to appropriate category (MACRO, SECTORS, TOP_STOCKS, WORLD_ETFS)
3. Optionally add display name to `TICKER_NAMES` dictionary

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for ticker format conversion guide.

---

### Q: Can I export results to CSV/Excel?
**A**: Not currently implemented, but easy to add. Modify `run_ccs_scan()` function:
```python
# After calculating results
import csv
with open('ccs_results.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Ticker', 'Score', 'Sentiment', ...])
    for result in results:
        writer.writerow([result['ticker'], result['current_score'], ...])
```

---

## Next Steps

### For New Users
1. ✅ Read [README_BLOOMBERG.md](README_BLOOMBERG.md)
2. ✅ Follow [SETUP_GUIDE.md](SETUP_GUIDE.md)
3. ✅ Run `test_bql_connection.py`
4. ✅ Run `bql_scanner.py`

### For Advanced Users
1. ✅ Review [VERSION_COMPARISON.md](VERSION_COMPARISON.md)
2. ✅ Customize ticker list for your strategy
3. ✅ Set up automation (cron/Task Scheduler)
4. ✅ Integrate with your trading systems

### For Developers
1. ✅ Review `bql_scanner.py` source code
2. ✅ Explore BQL documentation: `DOCS BQNT<GO>` in Terminal
3. ✅ Add custom indicators or data items
4. ✅ Contribute improvements back to the project

---

## Support Resources

### Documentation (This Folder)
- [README_BLOOMBERG.md](README_BLOOMBERG.md) - Features
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Setup & troubleshooting
- [VERSION_COMPARISON.md](VERSION_COMPARISON.md) - yfinance vs Bloomberg

### Parent Directory Documentation
- `enhanced_scoring_proposal.md` - Original methodology proposal
- `IMPLEMENTATION_SUMMARY.md` - yfinance implementation details

### Bloomberg Resources
- **BQL Reference**: `DOCS BQNT<GO>` in Bloomberg Terminal
- **Python API**: `DOCS BLPAPI<GO>`
- **Help Desk**: `HELP HELP<GO>`

---

## Changelog

### Version 2.0 - Bloomberg Edition (2026-01-19)
- ✅ Initial Bloomberg Terminal implementation
- ✅ BQL data fetching for all asset classes
- ✅ Phase 1 (Volume Analysis): 20% weight
- ✅ Phase 2 (Options Flow): 8% weight
  - Put/Call ratio analysis
  - IV rank percentile
- ✅ Comprehensive documentation suite
- ✅ Connection test script
- ✅ Production-ready code

---

**Index Version**: 1.0
**Last Updated**: 2026-01-19
**Status**: Production Ready
**Requirements**: Bloomberg Terminal subscription

---

## Quick Reference Card

```bash
# Test connection (run first)
python3 test_bql_connection.py

# Run daily scan
python3 bql_scanner.py

# View documentation
cat README_BLOOMBERG.md

# Edit tickers
nano bql_scanner.py  # Lines 25-65
```

**Data**: 69 instruments (21 macro + 16 sectors + 26 stocks + 6 world ETFs)

**Phases**:
- ✅ Phase 1 (Volume): 20%
- ✅ Phase 2 (Options): 8%

**Runtime**: ~90-120 seconds

**Output**: Color-coded terminal display with current scores + 5D/10D/20D averages
