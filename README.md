# Carnival Core Score (CCS) - Macro Scanner

A quantitative momentum/trend analysis system that scans 69 financial instruments across macro markets, sectors, stocks, and world ETFs.

## 📌 Two Versions Available

This project has **two implementations**:

1. **Free Version** (this folder): Uses Yahoo Finance - Phase 1 only
2. **Bloomberg Version** ([bloomberg_version/](bloomberg_version/)): Uses Bloomberg Terminal - Phase 1 + Phase 2

See [VERSIONS_OVERVIEW.md](VERSIONS_OVERVIEW.md) for detailed comparison.

## Quick Start

### Command Line (Daily Use)
```bash
cd "/Users/lisa/Documents/Macro Scanner/Macro-Scanner"
python3 run_scanner.py
```

### Jupyter Notebook (Development)
```bash
jupyter notebook "Macro Scanner.ipynb"
# Then run all cells or execute: run_ccs_scan()
```

## Features

### Instrument Coverage (69 Total)
- **MACRO (21)**: Futures (ES, NQ, RTY), Treasuries, Currencies, Commodities, Bitcoin
- **SECTORS (16)**: XLK, XLV, XLF, XLY, XLC, XLI, XLP, XLE, XLU, XLB, XLRE, XHB, XBI, SMH, SPHB, SPLV
- **TOP STOCKS (26)**: AAPL, NVDA, MSFT, AMZN, META, TSLA, GOOGL, and more
- **WORLD ETFS (6)**: VEA, IEMG, EEM, ACWI, BNDX, VWOB

### Scoring Algorithm

**Score Range**: -10 (Very Bearish) to +10 (Very Bullish)

**Components** (Free Version - Phase 1):
- ROC (Rate of Change) - 35% weight
  - Multi-timeframe momentum (1d, 2d, 5d, 10d, 20d)
  - Recent periods weighted more heavily
- MA Trend Signals - 25% weight
  - Moving average slopes and crossovers
  - Trend direction and alignment
- Volume & Conviction - 20% weight ← **Phase 1**
  - Volume ratio vs average
  - Volume trend analysis
  - Volume-price correlation
- RSI Momentum - 12% weight
  - Overbought/oversold conditions
  - Normalized from 14-period RSI
- Price vs MA - 8% weight
  - Position relative to key moving averages
  - Support/resistance analysis

**Note**: Bloomberg version adds **Phase 2 (Options Flow)** - 8% additional weight for stocks/ETFs.
See [bloomberg_version/README_BLOOMBERG.md](bloomberg_version/README_BLOOMBERG.md)

**Sentiment Classification**:
- **Very Bullish (L/S)**: Score ≥ 7.0 (Green, Bold)
- **Bullish (L)**: Score 4.0 - 6.9 (Green)
- **Neutral (L/S chop)**: Score -4.0 to 3.9 (White)
- **Bearish (S)**: Score -7.0 to -4.1 (Red)
- **Very Bearish (L/S)**: Score ≤ -7.0 (Red, Bold)

## Output Format

The scanner generates a formatted report with:
- Current score for each instrument
- Sentiment classification (color-coded)
- Historical scores (D-1, D-2, D-3, D-4)
- Rolling averages (5D, 10D, 20D)

### Sample Output
```
=== Carnival Core Score (CCS) - MACRO ===
Ticker               Score Sentiment                 Avg Score 5D  Score D-1  Score D-2  Score D-3  Score D-4 Avg Score 10D Avg Score 20D
------------------------------------------------------------------------------------------------------------------------------------------------------
ES                     0.7 Neutral (L/S chop)                    1        0.7        0.8        1.2        1.5           0.9           0.7
Silver                 5.2 Bullish (L)                         5.4        6.1        5.7        5.1        4.8           4.6           4.5
Nat Gas               -4.3 Bearish (S)                        -4.1       -4.7       -4.5       -3.1       -3.9          -4.5          -3.2
```

## Data

- **Source**: Yahoo Finance (yfinance)
- **History**: 90 days maintained
- **Update Frequency**: Manual daily (automation coming soon)
- **Data Date**: Typically T-1 (previous market close)

## Files

### Main Files (Free Version)
- **[run_scanner.py](run_scanner.py)**: Standalone Python script for command-line execution
- **[Macro Scanner.ipynb](Macro Scanner.ipynb)**: Jupyter notebook with modular components
- **[detailed_analysis.py](detailed_analysis.py)**: Script for detailed score component breakdown
- **score_breakdown.csv**: Detailed component analysis (generated on analysis runs)

### Documentation
- **[README.md](README.md)**: This file (free version guide)
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**: Phase 1 implementation details
- **[enhanced_scoring_proposal.md](enhanced_scoring_proposal.md)**: Original Phase 1+2 proposal
- **[VERSIONS_OVERVIEW.md](VERSIONS_OVERVIEW.md)**: Comparison of free vs Bloomberg versions

### Bloomberg Version (Separate Folder)
- **[bloomberg_version/](bloomberg_version/)**: Complete Bloomberg Terminal implementation
  - Includes Phase 1 (Volume) + Phase 2 (Options Flow)
  - See [bloomberg_version/INDEX.md](bloomberg_version/INDEX.md) for details

## Editing Tickers

To add or remove instruments, edit the `TICKERS` dictionary in either:
- `run_scanner.py` (lines 25-72)
- `Macro Scanner.ipynb` (cell 2)

```python
TICKERS = {
    'MACRO': ['ES=F', 'NQ=F', ...],
    'SECTORS': ['XLK', 'XLV', ...],
    'TOP_STOCKS': ['AAPL', 'NVDA', ...],
    'WORLD_ETFS': ['VEA', 'IEMG', ...],
}
```

Also update `TICKER_NAMES` for custom display names.

## Technical Details

### Requirements
- Python 3.7+
- yfinance
- pandas
- numpy

### Performance
- Fetches ~69 instruments in ~30-45 seconds
- Processes indicators and scores in ~2-3 seconds
- Total runtime: ~1 minute

### Data Quality
- Automatically handles yfinance multi-index columns
- Validates minimum 20 days of data per instrument
- Graceful error handling for failed downloads

## Next Steps

1. **Calibration**: Fine-tune scoring weights based on backtesting
2. **Persistence**: Save historical scores for trend analysis
3. **Automation**: Schedule daily runs and email delivery
4. **Visualization**: Heatmaps, sector rotation charts
5. **Filtering**: Sort by strongest/weakest, filter by sentiment

## Support

For issues or questions, refer to the code comments or the detailed inline documentation in the notebook.

## Related Versions

### Bloomberg Terminal Edition
For institutional-grade data and Phase 2 (Options Flow) analysis:
- **Location**: [bloomberg_version/](bloomberg_version/)
- **Features**: Phase 1 + Phase 2 (Options Flow with Put/Call ratio, IV rank)
- **Requires**: Bloomberg Terminal subscription
- **Documentation**: [bloomberg_version/INDEX.md](bloomberg_version/INDEX.md)

### Version Comparison
See [VERSIONS_OVERVIEW.md](VERSIONS_OVERVIEW.md) for:
- Feature comparison matrix
- Cost-benefit analysis
- Migration guide
- Use case recommendations

---

**Generated**: 2026-01-19
**Version**: 2.0 (Free Edition with Phase 1)
**Author**: Carnival Core Score System
**Bloomberg Edition**: Available in [bloomberg_version/](bloomberg_version/) folder
