# Carnival Core Score (CCS) - Macro Scanner

A quantitative momentum/trend analysis system that scans 69 financial instruments across macro markets, sectors, stocks, and world ETFs.

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

**Components**:
- ROC (Rate of Change) - 40% weight
  - Multi-timeframe momentum (1d, 2d, 5d, 10d, 20d)
  - Recent periods weighted more heavily
- MA Trend Signals - 30% weight
  - Moving average slopes and crossovers
  - Trend direction and alignment
- RSI Momentum - 15% weight
  - Overbought/oversold conditions
  - Normalized from 14-period RSI
- Price vs MA - 15% weight
  - Position relative to key moving averages
  - Support/resistance analysis

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

- **run_scanner.py**: Standalone Python script for command-line execution
- **Macro Scanner.ipynb**: Jupyter notebook with modular components
- **score_breakdown.csv**: Detailed component analysis (generated on analysis runs)
- **detailed_analysis.py**: Script for detailed score component breakdown

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

---

**Generated**: 2026-01-18
**Version**: 1.0
**Author**: Carnival Core Score System
