# Macro Quadrant Analysis - Fidenza Framework Integration

**Feature Added**: 2026-01-21
**Framework Reference**: https://www.fidenzamacro.com/p/the-four-quadrant-global-macro-framework

---

## Overview

The Bloomberg scanner now includes automatic macro environment analysis using the **Fidenza Four-Quadrant Framework**. After processing all instruments, the scanner analyzes key indicators to determine which macro regime we're currently in and provides actionable strategic guidance.

---

## The Four Quadrants

The framework uses two variables to identify market regimes:
- **Y-axis**: Growth expectations (Rising vs Falling)
- **X-axis**: Inflation expectations (Rising vs Falling)

### Quad 1: Goldilocks (Disinflation)
**Conditions**: Rising growth + Falling inflation

**Characteristics**:
- Middling but steady growth
- Stable or falling yields
- Minimal central bank tightening concerns

**Outperformers**: Equities, credit, crypto, EM assets
**Underperformers**: Cyclical commodities
**Strategy**: Risk-on positioning. Favor growth equities and credit.

---

### Quad 2: Reflation
**Conditions**: Rising growth + Rising inflation

**Characteristics**:
- Strong risk appetite
- Stimulative policy begins to work
- ISM accelerates
- Central bank response not yet a headwind

**Outperformers**: Equities, commodities, risky currencies
**Underperformers**: Precious metals
**Strategy**: Strong risk appetite. Watch for central bank tightening signals.

---

### Quad 3: Stagflation
**Conditions**: Falling growth + Rising inflation

**Characteristics**:
- Market tops form
- Selloffs turn violent
- Volatility explodes
- Fed signals tightening while growth weakens

**Outperformers**: USD, commodities (non-precious metals)
**Underperformers**: Equities, bonds
**Strategy**: Defensive positioning. Volatility likely to spike. Reduce equity exposure.

---

### Quad 4: Risk-Off (Deflation)
**Conditions**: Falling growth + Falling inflation

**Characteristics**:
- Existential threats to growth
- Flight to safety
- Simultaneous equity and bond concerns
- Policy accommodation typically follows

**Outperformers**: Safe-haven currencies, quality bonds
**Underperformers**: Commodities, equities
**Strategy**: Flight to safety. Favor treasuries and defensive assets.

---

## How the Scanner Determines the Quadrant

### Growth Signal Calculation

**Components** (weighted average):
- **Equity Indices (70% weight)**: S&P 500, Nasdaq 100, Russell 2000 futures scores
- **Cyclical Commodities (30% weight)**: Copper, Oil (economic demand proxies)

**Formula**:
```python
growth_signal = (equity_signal * 0.7) + (commodity_signal * 0.3)
growth_rising = growth_signal > 0
```

**Interpretation**:
- Positive score = Rising growth expectations
- Negative score = Falling growth expectations

---

### Inflation Signal Calculation

**Components** (weighted average):
- **Treasury Yields (50% weight)**: Inverted bond scores (falling bonds = rising yields = rising inflation)
  - 2Y, 5Y, 10Y, 30Y Treasury futures
- **Inflation Hedges (50% weight)**: Gold, Silver, Oil scores

**Formula**:
```python
bond_signal = -(avg of treasury scores)  # Inverted
commodity_inflation_signal = avg(Gold, Silver, Oil)
inflation_signal = (bond_signal * 0.5) + (commodity_inflation_signal * 0.5)
inflation_rising = inflation_signal > 0
```

**Interpretation**:
- Positive score = Rising inflation expectations
- Negative score = Falling inflation expectations

---

## Output Format

The scanner produces a detailed analysis report at the end of each scan:

```
================================================================================
            MACRO ENVIRONMENT ANALYSIS - FIDENZA FOUR-QUADRANT FRAMEWORK
================================================================================

Current Quadrant: Quad 1: Goldilocks (Disinflation)
Description: Rising growth with falling inflation

Signal Breakdown:
  Growth Signal: +3.45 (Rising)
    - Equity Indices (S&P/Nasdaq/Russell): +4.20
    - Cyclical Commodities (Copper/Oil): +1.50
  Inflation Signal: -2.10 (Falling)
    - Treasury Yields (inverted): -1.80
    - Inflation Hedges (Gold/Silver/Oil): -2.40

Key Instrument Scores:
  S&P 500 Futures      Score:    5.2  (5D Avg:    4.8)  Bullish (L)
  Nasdaq 100 Futures   Score:    6.1  (5D Avg:    5.5)  Bullish (L)
  10Y Treasury         Score:    2.3  (5D Avg:    2.1)  Neutral (L/S chop)
  Gold                 Score:   -3.1  (5D Avg:   -2.8)  Neutral (L/S chop)
  Crude Oil            Score:    1.8  (5D Avg:    2.2)  Neutral (L/S chop)
  US Dollar Index      Score:   -1.5  (5D Avg:   -1.2)  Neutral (L/S chop)

Asset Class Outlook:
  Outperformers: Equities, credit, crypto, EM assets
  Underperformers: Cyclical commodities

Recommended Strategy:
  Risk-on positioning. Favor growth equities and credit.

Quadrant Transition Watch:
  → Quad 2 (Reflation): Watch for stimulative policy and ISM acceleration

================================================================================
Framework Reference: https://www.fidenzamacro.com/p/the-four-quadrant-global-macro-framework
================================================================================
```

---

## Key Instruments Monitored

### Growth Indicators
| Ticker | Description | Role |
|--------|-------------|------|
| ESA Index | S&P 500 E-mini Futures | Primary equity market trend |
| NQA Index | Nasdaq 100 E-mini Futures | Growth/tech sector trend |
| RTYA Index | Russell 2000 E-mini Futures | Small-cap risk appetite |
| HGA Comdty | Copper Futures | Industrial demand proxy |
| CLA Comdty | Crude Oil Futures | Economic activity proxy |

### Inflation Indicators
| Ticker | Description | Role |
|--------|-------------|------|
| TUA Comdty | 2-Year Treasury Futures | Short-term rate expectations |
| FVA Comdty | 5-Year Treasury Futures | Medium-term rate expectations |
| TYA Comdty | 10-Year Treasury Futures | Long-term rate expectations |
| USA Comdty | 30-Year Treasury Futures | Ultra long-term rate expectations |
| GCA Comdty | Gold Futures | Inflation hedge / safe haven |
| SIA Comdty | Silver Futures | Inflation hedge |

### Risk Sentiment
| Ticker | Description | Role |
|--------|-------------|------|
| DXY Index | US Dollar Index | Safe-haven demand |

---

## Quadrant Transition Signals

The analysis includes warnings for potential quadrant transitions:

| Current Quadrant | Watch For | Transition To |
|------------------|-----------|---------------|
| **Quad 1** | Stimulative policy, ISM acceleration | Quad 2 (Reflation) |
| **Quad 2** | Fed tightening signals, yield spikes | Quad 3 (Stagflation) |
| **Quad 3** | Economic weakness with persistent inflation | Quad 4 (Risk-Off) |
| **Quad 4** | Policy accommodation, falling yields | Quad 1 (Goldilocks) |

---

## Integration with Carnival Core Score

The macro quadrant analysis complements the individual instrument scores:

### Score Interpretation by Quadrant

**Quad 1 (Goldilocks)**:
- High equity scores = Confirmation of bullish regime
- Rising commodity scores = Watch for transition to Quad 2
- Falling bond scores (rising yields) = Healthy growth signal

**Quad 2 (Reflation)**:
- High commodity scores = Confirmation of reflationary regime
- Very high equity scores = Peak euphoria risk
- Bond scores turning negative = Tightening pressure building

**Quad 3 (Stagflation)**:
- Negative equity scores = Confirmation of risk-off
- High commodity scores = Inflation persistence
- Negative bond scores = Central bank in tough position

**Quad 4 (Risk-Off)**:
- Negative equity scores = Risk aversion
- Positive bond scores (falling yields) = Flight to safety
- Negative commodity scores = Demand destruction

---

## Usage Notes

### Automatic Execution
The macro quadrant analysis runs automatically at the end of every scan. No additional configuration needed.

### Manual Analysis
You can also call the function directly in Python:

```python
from bql_scanner import analyze_macro_quadrant

# After running scan and getting results_by_category
quadrant_data = analyze_macro_quadrant(results_by_category)

# Access individual components
print(f"Current quadrant: {quadrant_data['quadrant']}")
print(f"Growth signal: {quadrant_data['growth_signal']:.2f}")
print(f"Inflation signal: {quadrant_data['inflation_signal']:.2f}")
print(f"Strategy: {quadrant_data['strategy']}")
```

### Interpretation Guidelines

1. **Signal Strength**: Stronger signals (further from 0) indicate more conviction
   - Signal > +5: Very strong
   - Signal +2 to +5: Moderate
   - Signal -2 to +2: Weak/transitioning
   - Signal < -5: Very strong (opposite direction)

2. **Quadrant Transitions**: Watch for signals moving from one side of zero to the other
   - Both signals near zero = High uncertainty, choppy markets
   - Signals diverging rapidly = Potential regime shift

3. **Confirmation**: Compare quadrant determination with individual scores
   - High agreement = High confidence in regime
   - Low agreement = Mixed signals, proceed with caution

---

## Limitations

1. **Backward-Looking**: Uses recent price momentum, not forward-looking economic data
2. **No ISM Data**: Original framework uses ISM Composite; scanner uses price-based proxies
3. **Simplified Inflation Gauge**: Uses commodity prices and bond yields, not actual CPI/PCE data
4. **No Central Bank Input**: Doesn't directly incorporate Fed statements or policy expectations

**Recommendation**: Use this analysis as a **complement** to fundamental macro research, not a replacement.

---

## Examples

### Example 1: Goldilocks Environment
```
Current Quadrant: Quad 1: Goldilocks (Disinflation)
Growth Signal: +4.5 (Rising)
Inflation Signal: -3.2 (Falling)

Key Scores:
- S&P 500: +6.5 (Very Bullish)
- Nasdaq: +7.2 (Very Bullish)
- 10Y Treasury: +3.1 (Bullish - falling yields)
- Gold: -2.5 (Bearish)
- Oil: +1.2 (Neutral)

Interpretation: Strong equity rally with falling yields = Classic Goldilocks.
Continue risk-on positioning.
```

### Example 2: Stagflation Warning
```
Current Quadrant: Quad 3: Stagflation
Growth Signal: -2.8 (Falling)
Inflation Signal: +5.1 (Rising)

Key Scores:
- S&P 500: -4.2 (Bearish)
- Nasdaq: -5.5 (Bearish)
- 10Y Treasury: -6.8 (Very Bearish - yields spiking)
- Gold: +3.5 (Bullish)
- Oil: +6.2 (Bullish)

Interpretation: Equities falling while commodities and yields rise = Stagflation.
Reduce equity exposure, increase defensive positioning.
```

### Example 3: Transition Period
```
Current Quadrant: Quad 2: Reflation
Growth Signal: +2.1 (Rising - weakening)
Inflation Signal: +3.8 (Rising)

Quadrant Transition Watch:
→ Quad 3 (Stagflation): Monitor Fed tightening signals and yield spikes

Key Scores:
- S&P 500: +3.2 (Bullish but fading from +5.5)
- 10Y Treasury: -5.1 (Very Bearish - yields rising fast)
- Oil: +6.8 (Very Bullish)

Interpretation: Growth signal weakening while inflation persists =
Watch for transition to Quad 3. Consider taking profits on equity longs.
```

---

## Technical Details

### Function Signature
```python
def analyze_macro_quadrant(results_by_category):
    """
    Analyzes current macro environment using Fidenza four-quadrant framework.

    Args:
        results_by_category (dict): Scanner results organized by category

    Returns:
        dict: {
            'quadrant': str,           # Full quadrant name
            'quadrant_num': int,       # 1-4
            'growth_signal': float,    # Combined growth signal
            'growth_rising': bool,     # True if growth rising
            'inflation_signal': float, # Combined inflation signal
            'inflation_rising': bool,  # True if inflation rising
            'description': str,        # Quadrant description
            'outperformers': str,      # Expected outperformers
            'underperformers': str,    # Expected underperformers
            'strategy': str,           # Recommended strategy
        }
    """
```

### Dependencies
- Requires MACRO category tickers to be present in scan results
- Falls back gracefully if some instruments are missing (uses available data)
- Returns neutral signals (0) if category is completely missing

---

## Changelog

### Version 1.0 - 2026-01-21
- ✅ Initial implementation
- ✅ Four-quadrant framework integration
- ✅ Growth signal calculation (equities + commodities)
- ✅ Inflation signal calculation (bonds + inflation hedges)
- ✅ Automatic quadrant determination
- ✅ Detailed analysis output
- ✅ Transition warning system
- ✅ Key instrument score display

---

**Document Version**: 1.0
**Last Updated**: 2026-01-21
**Status**: Production Ready
**Framework Credit**: Fidenza Macro (https://www.fidenzamacro.com)
