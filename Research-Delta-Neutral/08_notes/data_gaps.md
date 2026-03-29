# Data Gaps — What Blocks Production Backtest Quality

## Critical (must fix before institutional pitch)

### 1. Roll methodology
- **Problem**: `gc_fut_front` is a rolling continuation series. Actual rolls create
  price gaps that contaminate spread calculations.
- **Impact**: Spread may jump discontinuously at each roll — creates false signals.
- **Fix**: Map specific contract months (GCZ24, GCG25, etc.) and build explicit
  roll-adjusted spreads.
- **Status of data_contract_dates.csv**: Contains FDD/FND/LTD data but alignment
  with the price series is unverified — flagged as unreliable in research phase.

### 2. Exact contract-specific prices
- **Problem**: We have front/second/third but don't know WHICH contracts these are
  at each date.
- **Impact**: Prevents understanding of whether the spread represents 1-month,
  2-month, or 3-month tenor — affects fair value calculation.
- **Fix**: Pull contract-specific GC prices from Bloomberg (GCZ24, GCH25, etc.)

### 3. Transaction cost calibration
- **Problem**: Used $0.50 flat placeholder. Real costs depend on lot size, timing,
  counterparty, and spread.
- **Fix**: Use actual bid-ask from tick data. For GC, typical bid-ask is 0.10/oz
  = $10/contract one-way.

## Important (affects signal quality)

### 4. SOFR-implied fair value
- **Problem**: Rolling z-score mean-reverts to historical average, not fundamental value.
  When rates shift structurally (2022), the strategy bets against a moving target.
- **Fix**: Model `fair_cal_12 = -spot × SOFR × (days_to_expiry / 365)` and
  trade deviations from this model-based fair value.
- **Data available**: SOFR OIS curve is in data-2.csv ✓

### 5. Gold EFP data
- **Problem**: gold_efp only available from mid-2025 (92% missing).
  The EFP (Exchange for Physical) is the key arbitrage mechanism that keeps
  futures/spot aligned — crucial for understanding calendar spread dynamics.
- **Fix**: Obtain historical EFP data from Bloomberg or OTC sources.

### 6. Lease rate data
- **Problem**: Gold lease rates are not directly available. They are DERIVED
  from XAU swaps minus SOFR — but the XAU swaps in the dataset don't cover
  all tenors or all dates cleanly.
- **Fix**: Calculate implied lease rates from swap data already in dataset.
  Use as carry-adjustment factor for fair value model.

## Nice to have

### 7. Volume and open interest data
- Needed for: position sizing, liquidity checks, roll timing optimisation
- Not in current dataset

### 8. Intraday price data
- Needed for: realistic execution simulation
- Daily close prices overstate achievable PnL

### 9. Contract-specific open interest
- Needed for: identifying which contracts are liquid and tradeable
- Would help verify roll dates

## Summary table

| Gap | Severity | Fix Available | Est. Effort |
|-----|----------|---------------|-------------|
| Roll methodology | Critical | Bloomberg GC contracts | Medium |
| Contract-specific prices | Critical | Bloomberg pull | Low |
| Transaction cost calibration | Critical | Tick data | Medium |
| SOFR fair value model | Important | Already have SOFR data | Medium |
| EFP data | Important | Bloomberg | Low (data exists from 2025) |
| Lease rates | Important | Derive from swap data | Low |
| Volume/OI | Nice-to-have | Bloomberg | Low |
| Intraday data | Nice-to-have | Expensive to obtain | High |
