# Commodity Trading Signal System

A comprehensive trading signal system for Gold, Silver, Platinum, Palladium, and Oil built on Bloomberg BQL. Combines sentiment analysis, cross-asset correlation, and macro regime detection with backtesting and walk-forward validation.

---

## Table of Contents

1. [Overview](#overview)
2. [Signal Components & Weights](#signal-components--weights)
3. [Methodology](#methodology)
4. [TD Sequential Indicator](#td-sequential-indicator)
5. [Correlation Analysis](#correlation-analysis)
6. [Macro Regime Framework](#macro-regime-framework)
7. [Backtesting & Validation](#backtesting--validation)
8. [Data Validation](#data-validation)
9. [Quick Start Guide](#quick-start-guide)
10. [File Structure](#file-structure)

---

## Overview

This system generates buy/sell signals for commodities by combining multiple analytical approaches:

- **Sentiment Scoring**: Technical momentum indicators (ROC, MA, RSI, TD Sequential)
- **Correlation Analysis**: Cross-asset confirmation and divergence detection
- **Regime Detection**: Fidenza four-quadrant macro framework
- **Risk Management**: ATR-based stops, position sizing, correlation limits

**Signal Range**: -10 (Very Bearish) to +10 (Very Bullish)

| Signal | Classification | Recommended Action |
|--------|----------------|-------------------|
| >= 7.0 | Very Bullish | Strong Buy / Add to Long |
| >= 4.0 | Bullish | Buy / Hold Long |
| -4.0 to 4.0 | Neutral | No Position / Reduce |
| <= -4.0 | Bearish | Sell / Hold Short |
| <= -7.0 | Very Bearish | Strong Sell / Add to Short |

---

## Signal Components & Weights

The composite signal is built from four main components:

### Overall Signal Formula
```
Composite Signal = Sentiment (40%) + Correlation (25%) + Divergence (20%) + Regime (15%)
```

### Sentiment Score Breakdown (40% of total)

The sentiment component uses technical indicators with the following internal weights:

| Indicator | Weight | Description |
|-----------|--------|-------------|
| **ROC (Rate of Change)** | 35% | Multi-timeframe momentum |
| **MA Trend** | 25% | Moving average slopes & crossovers |
| **Volume & Conviction** | 15% | Volume ratio and price-volume correlation |
| **RSI** | 10% | Relative Strength Index (14-period) |
| **TD Sequential** | 15% | Tom DeMark exhaustion indicator |

#### ROC (Rate of Change) - 35%
Multi-timeframe momentum measurement:
- ROC_1 (1-day): 15%
- ROC_2 (2-day): 15%
- ROC_5 (5-day): 25%
- ROC_10 (10-day): 25%
- ROC_20 (20-day): 20%

#### MA Trend - 25%
- MA slope analysis (5/10/20-day): 70%
- MA crossover alignment: 30%
  - Bullish: MA5 > MA10 > MA20 (+5 points)
  - Bearish: MA5 < MA10 < MA20 (-5 points)

#### Volume & Conviction - 15%
- Volume ratio (current vs 20-day MA): 40%
- Volume trend (5-day vs 20-day MA): 30%
- Volume-price correlation: 30%

#### RSI - 10%
- 14-period RSI, normalized around 50
- Formula: (RSI - 50) / 5, then normalized to [-10, 10]

#### TD Sequential - 15%
- Counter-trend exhaustion indicator
- Buy Setup 9 complete: +7.5 points
- Sell Setup 9 complete: -7.5 points
- Partial setups (7-8): ±3.0 points

---

## Methodology

### Why This Approach Works

#### 1. Momentum Persistence (ROC/MA - 60% of sentiment)
Commodities exhibit momentum persistence because:
- Institutional flows take time to fully deploy
- Supply/demand imbalances resolve slowly
- Technical traders create self-fulfilling momentum

**Academic Support**: Asness, Moskowitz, Pedersen (2013) documented momentum premium across asset classes including commodities.

#### 2. Counter-Trend Exhaustion (TD Sequential - 15%)
Tom DeMark's indicator identifies potential reversal points:
- Complements momentum by catching exhaustion
- 9 consecutive closes vs close[4] signals potential reversal
- Well-suited for commodities that trend then mean-revert

#### 3. Cross-Asset Confirmation (Correlation - 25%)
Multi-asset agreement increases conviction:
- Gold + DXY inverse correlation confirms moves
- Divergence from correlated assets signals opportunity
- Correlation regime changes flag increased uncertainty

#### 4. Macro Context (Regime - 15%)
Different macro environments favor different commodities:
- Stagflation: Gold outperforms (+30% signal boost)
- Reflation: Oil outperforms (+25% boost)
- Risk-Off: Gold up, Oil down

---

## TD Sequential Indicator

### What is TD Sequential?

Tom DeMark's TD Sequential is a counter-trend indicator that identifies potential exhaustion points in trending markets.

### TD Setup (9-Count)

**Buy Setup (Bearish Exhaustion)**:
- Counts consecutive closes BELOW the close 4 bars ago
- 9 consecutive lower closes = downtrend exhaustion
- Signal: Potential bottom, bullish reversal

**Sell Setup (Bullish Exhaustion)**:
- Counts consecutive closes ABOVE the close 4 bars ago
- 9 consecutive higher closes = uptrend exhaustion
- Signal: Potential top, bearish reversal

### Signal Contribution

| TD State | Score Contribution | Interpretation |
|----------|-------------------|----------------|
| Buy Setup = 9 | +7.5 | Strong bullish reversal signal |
| Buy Setup 7-8 | +3.0 | Approaching potential bottom |
| Sell Setup = 9 | -7.5 | Strong bearish reversal signal |
| Sell Setup 7-8 | -3.0 | Approaching potential top |
| No setup | 0 | No counter-trend signal |

### Why TD Sequential for Commodities

1. **Mean Reversion**: Commodities tend to trend then revert
2. **Timing**: Catches exhaustion before momentum indicators reverse
3. **Complementary**: Balances momentum-following bias of ROC/MA

---

## Correlation Analysis

### Key Correlation Relationships

#### Gold (GCA Comdty)
| Asset | Expected Correlation | Rationale |
|-------|---------------------|-----------|
| DXY Index | -0.4 to -0.6 | Alternative stores of value |
| 10Y Treasury (TYA) | -0.3 | Inverse to real yields |
| USD/JPY | -0.3 | Risk-on/off proxy |
| Silver (SIA) | +0.85 | Precious metals correlation |

#### Oil (CLA Comdty)
| Asset | Expected Correlation | Rationale |
|-------|---------------------|-----------|
| USD/CAD | -0.5 | Canada is major exporter |
| Copper (HGA) | +0.5 | Both growth-sensitive |
| DXY | -0.4 | Dollar-denominated |
| S&P 500 (ESA) | +0.3 | Growth proxy |

### Cross-Commodity Ratios

| Ratio | Historical Range | Interpretation |
|-------|-----------------|----------------|
| Gold/Silver | 50-90 | >90: Silver undervalued, <50: Gold undervalued |
| Gold/Oil | 15-25 | Safe haven vs growth proxy |
| Gold/Platinum | 1.5-2.5 | Monetary vs industrial precious metal |

### Correlation Alignment Score (25% of signal)

Measures whether correlated assets confirm commodity direction:

- **Confirming** (+1.0): Gold bullish AND DXY bearish
- **Warning** (-0.5): Gold bullish AND DXY bullish
- **Neutral** (0): No clear relationship

### Divergence Detection (20% of signal)

Identifies mean-reversion opportunities when assets diverge:

- **Bullish Divergence**: Commodity flat while inverse-correlated asset falls
- **Bearish Divergence**: Commodity flat while inverse-correlated asset rises

---

## Macro Regime Framework

### Fidenza Four-Quadrant Model

The system classifies the macro environment using growth and inflation signals:

```
                    INFLATION
                    Rising  |  Falling
                 -----------+-----------
        Rising   |  Quad 2  |  Quad 1  |
GROWTH           | Reflation| Goldilocks|
                 -----------+-----------
        Falling  |  Quad 3  |  Quad 4  |
                 |Stagflation| Risk-Off |
                 -----------+-----------
```

### Quadrant Definitions

| Quadrant | Name | Conditions | Commodity Impact |
|----------|------|------------|------------------|
| 1 | Goldilocks | Growth up, Inflation down | Gold -20%, Oil +15% |
| 2 | Reflation | Growth up, Inflation up | Gold neutral, Oil +25% |
| 3 | Stagflation | Growth down, Inflation up | Gold +30%, Oil neutral |
| 4 | Risk-Off | Growth down, Inflation down | Gold +20%, Oil -30% |

### Signal Calculation

**Growth Signal** (70% equities, 30% cyclicals):
- Equity indices: S&P 500, Nasdaq 100, Russell 2000
- Cyclical commodities: Copper, Oil

**Inflation Signal** (50% bonds, 50% hedges):
- Treasury yields (inverted): 2Y, 5Y, 10Y, 30Y
- Inflation hedges: Gold, Silver, Oil

---

## Backtesting & Validation

### Backtest Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Initial Capital | $10,000,000 | Starting portfolio value |
| Transaction Costs | 2 bps | Round-trip commission |
| Slippage | 1 bp | Market impact |
| Max Position | 20% | Maximum single position size |
| Stop Loss | 2x ATR | Volatility-adjusted stops |
| Profit Target | 3:1 R/R | Reward/risk ratio |
| Max Holding | 20 days | Maximum holding period |

### Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Sharpe Ratio | > 1.0 | Risk-adjusted return |
| Sortino Ratio | > 1.2 | Downside-adjusted return |
| Max Drawdown | < 20% | Worst peak-to-trough |
| Win Rate | > 45% | Percentage of winning trades |
| Profit Factor | > 1.5 | Gross profit / Gross loss |

### Walk-Forward Optimization

Prevents overfitting by testing out-of-sample:

| Parameter | Value |
|-----------|-------|
| In-Sample Period | 12 months |
| Out-of-Sample Period | 3 months |
| Window Type | Rolling |

**Degradation Ratio Interpretation**:
- > 0.8: Excellent - Strategy is robust
- 0.6-0.8: Good - Some overfitting
- 0.4-0.6: Concerning - Significant overfitting
- < 0.4: Poor - Strategy is overfit

---

## Data Validation

### Automated Sanity Checks

The system performs these automated checks:

| Check | Criteria | Action if Failed |
|-------|----------|------------------|
| Date Gaps | Max 4 days | Review for data issues |
| Price Range | All prices > 0 | Critical - investigate |
| Daily Moves | Max 20% | Warning - verify extreme |
| Volume | Sum > 0 | N/A for forex |
| OHLC Valid | High >= Low, etc. | Critical - data corrupt |

### Manual Verification Checklist

1. **Price Data**: Run `HP {ticker}` in Terminal, compare last 5 closes
2. **Indicators**: Calculate ROC manually, compare with output
3. **Correlations**: Run `CORR GCA Comdty, DXY Index`, compare 60-day
4. **Backtest Trades**: Verify 3-5 trades on historical charts

### Validation Functions

```python
# Run all validations
from commodity_signals import run_full_validation
run_full_validation(['GCA Comdty', 'SIA Comdty', 'CLA Comdty'])

# Individual validations
from commodity_signals import (
    validate_price_data,        # Print data for Terminal comparison
    run_data_sanity_checks,     # Automated quality checks
    verify_signal_calculation,  # Step-by-step signal breakdown
    verify_correlation_calculations  # Gold-DXY correlation check
)
```

---

## Quick Start Guide

### 1. Generate Current Signals

```python
from commodity_signals import CommoditySignalGenerator, print_signal_report

# Initialize generator
generator = CommoditySignalGenerator(lookback_days=120)

# Generate signals for all commodities
signals = generator.generate_all_signals()

# Print formatted report
print_signal_report(signals)
```

### 2. Get Signal for Specific Commodity

```python
# Generate signal for Gold
gold_signal = generator.generate_signal('GCA Comdty')

print(f"Signal: {gold_signal['signal']}")
print(f"Classification: {gold_signal['classification']}")
print(f"Action: {gold_signal['action']}")

# View component breakdown
components = gold_signal['components']
print(f"Sentiment: {components['sentiment']}")
print(f"Correlation: {components['correlation']}")
print(f"Divergence: {components['divergence']}")
print(f"Regime Adjustment: {components['regime_adjustment']}")
```

### 3. Generate Entry/Exit Signals

```python
# Check if should enter position
entry = generator.generate_entry_signal('GCA Comdty')
print(f"Entry Signal: {entry['entry_signal']}")
print(f"Direction: {entry['entry_direction']}")
print(f"Confidence: {entry['confidence']}")

# Check if should exit existing position
exit_signal = generator.generate_exit_signal(
    commodity_ticker='GCA Comdty',
    entry_price=2000.00,
    current_price=2050.00,
    entry_signal=5.5,
    holding_days=10,
    position_direction='long'
)
print(f"Exit Signal: {exit_signal['exit_signal']}")
print(f"Reason: {exit_signal['exit_reason']}")
```

### 4. Run Correlation Analysis

```python
from correlation_analysis import CorrelationEngine, print_correlation_summary

# Initialize engine
engine = CorrelationEngine(lookback_days=252)

# Get comprehensive correlation summary
gold_summary = engine.get_correlation_summary('GCA Comdty')
print_correlation_summary(gold_summary)

# Get cross-commodity ratios
ratios = engine.calculate_cross_commodity_ratios()
for name, df in ratios.items():
    latest = df.iloc[-1]
    print(f"{name}: {latest['Ratio']:.2f} (Z-Score: {latest['Ratio_Zscore']:.2f})")
```

### 5. Run Backtest

```python
from backtest_engine import CommodityBacktest, print_backtest_report

# Initialize backtest
backtest = CommodityBacktest()

# Run 2-year backtest
result = backtest.run_backtest(
    start_date='2023-01-01',
    end_date='2025-01-24',
    commodities=['GCA Comdty', 'SIA Comdty', 'CLA Comdty']
)

# Print results
print_backtest_report(result)
```

### 6. Run Walk-Forward Validation

```python
from backtest_engine import WalkForwardOptimizer

# Initialize optimizer
wf = WalkForwardOptimizer(
    in_sample_months=12,
    out_of_sample_months=3
)

# Run walk-forward
wf_result = wf.run_walk_forward(
    start_date='2021-01-01',
    end_date='2025-01-24',
    commodities=['GCA Comdty', 'CLA Comdty']
)

# Check degradation ratio
print(f"Degradation Ratio: {wf_result['summary']['degradation_ratio']}")
print(f"Interpretation: {wf_result['summary']['interpretation']}")
```

### 7. Validate Data

```python
from commodity_signals import run_full_validation, verify_signal_calculation

# Run complete validation suite
run_full_validation()

# Or verify specific commodity signal calculation
verify_signal_calculation('GCA Comdty')
```

---

## File Structure

```
bloomberg_version/
├── correlation_analysis.py      # Cross-asset correlation engine
│   ├── CorrelationEngine class
│   ├── Rolling correlations
│   ├── Lead-lag detection
│   ├── Correlation regime classification
│   └── Cross-commodity ratios
│
├── commodity_signals.py         # Main signal generation
│   ├── Technical indicators (ROC, MA, RSI, ATR)
│   ├── TD Sequential calculation
│   ├── Sentiment scoring
│   ├── Correlation alignment
│   ├── Divergence detection
│   ├── Macro regime detection
│   ├── CommoditySignalGenerator class
│   └── Data validation functions
│
├── backtest_engine.py           # Backtesting framework
│   ├── CommodityBacktest class
│   ├── Trade execution simulation
│   ├── Performance metrics
│   ├── WalkForwardOptimizer class
│   └── Parameter sensitivity
│
├── Commodity_Signal_Analysis.ipynb  # Interactive analysis
│   ├── Current signals dashboard
│   ├── Correlation heatmaps
│   ├── Cross-commodity ratios
│   ├── Lead-lag analysis
│   ├── Backtest visualization
│   ├── Walk-forward results
│   └── Data validation section
│
└── COMMODITY_SIGNALS_README.md  # This documentation
```

---

## Bloomberg Tickers Reference

### Commodities
| Ticker | Name |
|--------|------|
| GCA Comdty | Gold (COMEX) |
| SIA Comdty | Silver |
| PLA Comdty | Platinum |
| PAA Comdty | Palladium |
| CLA Comdty | WTI Crude Oil |

### Currencies
| Ticker | Name |
|--------|------|
| DXY Index | US Dollar Index |
| USDJPY Curncy | USD/JPY |
| EURUSD Curncy | EUR/USD |
| GBPUSD Curncy | GBP/USD |
| AUDUSD Curncy | AUD/USD |
| USDCAD Curncy | USD/CAD |

### Bonds
| Ticker | Name |
|--------|------|
| TUA Comdty | 2-Year Treasury Futures |
| FVA Comdty | 5-Year Treasury Futures |
| TYA Comdty | 10-Year Treasury Futures |
| USA Comdty | 30-Year Treasury Futures |

### Equities
| Ticker | Name |
|--------|------|
| ESA Index | S&P 500 E-mini Futures |
| NQA Index | Nasdaq 100 E-mini Futures |
| RTYA Index | Russell 2000 E-mini Futures |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-24 | Initial release |
| 1.1 | 2025-01-24 | Added TD Sequential (15% weight) |
| 1.1 | 2025-01-24 | Updated Gold/Silver ratio range to 50-90 |
| 1.1 | 2025-01-24 | Added data validation functions |

---

## Contact & Support

For questions or issues, refer to the main project documentation or contact the development team.
