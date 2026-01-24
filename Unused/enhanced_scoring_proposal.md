# Enhanced Carnival Core Score - Proposal
## Adding Volume & Options Flow Analysis

### Current Score Breakdown (100%)
- ROC (Rate of Change): 40%
- MA Trend: 30%
- RSI: 15%
- Price vs MA: 15%

**Problem**: Pure price-based scoring can miss important signals about conviction and sustainability of moves.

---

## Proposed Enhanced Model

### New Component: Volume & Conviction Analysis (20% weight)

We'll add a new 20% component and rebalance existing components to 80%:

**New Allocation:**
- **ROC (Price Momentum)**: 35% (reduced from 40%)
- **MA Trend**: 25% (reduced from 30%)
- **Volume & Conviction**: 20% (NEW)
- **RSI**: 12% (reduced from 15%)
- **Price vs MA**: 8% (reduced from 15%)

---

## Volume & Conviction Component (20%)

### 1. Volume Analysis (60% of this component = 12% total)

**Metrics:**
- **Relative Volume Ratio**: Current volume vs 20-day average
  - Score: `(current_vol / avg_vol_20d - 1) * 100`
  - Normalize to -10 to +10 scale

- **Volume Trend**: Volume moving average slope
  - Compare 5-day volume MA vs 20-day volume MA
  - Increasing volume = bullish confirmation

- **Volume-Price Correlation**:
  - When price rises on high volume = bullish (add points)
  - When price rises on low volume = weak signal (reduce points)
  - When price falls on high volume = bearish (subtract points)
  - When price falls on low volume = not confirmed (neutral)

**Calculation:**
```python
vol_ratio_score = normalize((current_vol / avg_vol_20) - 1) * 0.40
vol_trend_score = normalize(vol_ma5 / vol_ma20 - 1) * 0.30
vol_price_corr = sign(roc_1) * normalize(vol_ratio) * 0.30

volume_score = (vol_ratio_score + vol_trend_score + vol_price_corr) * 0.60
```

### 2. Options Flow Analysis (40% of this component = 8% total)

**For Stocks/ETFs Only** (futures/forex/commodities skip this):

**Metrics:**
- **Implied Volatility (IV) Rank/Percentile**
  - High IV = uncertainty/fear (can be bullish or bearish context)
  - Rising IV on uptrend = strong bullish conviction
  - Rising IV on downtrend = panic/capitulation

- **Put/Call Ratio**
  - Ratio < 0.7 = bullish (more calls than puts)
  - Ratio 0.7-1.3 = neutral
  - Ratio > 1.3 = bearish (more puts than calls)
  - Can be contrarian indicator at extremes

- **Options Volume vs Stock Volume**
  - High options volume relative to stock volume = institutional interest

**Calculation:**
```python
# Put/Call Ratio scoring (contrarian at extremes)
if put_call_ratio < 0.5:
    pc_score = 5  # Very bullish (extreme optimism - could be contrarian bearish)
elif put_call_ratio < 0.7:
    pc_score = 3  # Bullish
elif put_call_ratio < 1.0:
    pc_score = 0  # Neutral
elif put_call_ratio < 1.3:
    pc_score = -3  # Bearish
else:
    pc_score = -5  # Very bearish (extreme pessimism - could be contrarian bullish)

# IV Rank scoring (contextual)
if price_trend > 0:  # Uptrend
    iv_score = iv_rank / 10  # Higher IV in uptrend = conviction
else:  # Downtrend
    iv_score = -(iv_rank / 10)  # Higher IV in downtrend = panic

options_score = (pc_score * 0.5 + iv_score * 0.5) * 0.40
```

---

## Data Sources

### Volume Data (Already Available)
- ✅ Volume is included in yfinance OHLCV data
- No additional API needed

### Options Data Sources

**Option 1: Free APIs (Limited)**
- `yfinance` has options data: `ticker.option_chain(date)`
  - Provides: calls, puts, strike prices, volume, open interest, IV
  - Limitation: Need to aggregate across strikes

**Option 2: Paid APIs (Professional)**
- TDAmeritrade API (free with account)
- Interactive Brokers API
- CBOE DataShop (official options data)
- Tradier API ($0.35/month for delayed data)

**Option 3: Derived Metrics (Free via yfinance)**
- Calculate our own Put/Call ratio from options chain
- IV from individual options (need to aggregate)

---

## Implementation Strategy

### Phase 1: Volume Only (Immediate)
Add volume analysis (12% weight) to all instruments:
- Volume ratio
- Volume trend
- Volume-price correlation

**Benefits:**
- Works for all asset classes
- No new data dependencies
- Immediate implementation

### Phase 2: Options Flow (Future Enhancement)
Add options analysis (8% weight) for stocks/ETFs only:
- Put/Call ratio
- IV percentile
- Options volume

**Benefits:**
- Better stock/ETF signals
- Institutional activity insights
- Early reversal warnings

---

## Code Structure

```python
def calculate_volume_score(df):
    """Calculate volume conviction score"""
    # Volume ratio vs average
    avg_vol_20 = df['Volume'].rolling(20).mean()
    vol_ratio = df['Volume'] / avg_vol_20

    # Volume trend
    vol_ma5 = df['Volume'].rolling(5).mean()
    vol_ma20 = df['Volume'].rolling(20).mean()
    vol_trend = vol_ma5 / vol_ma20

    # Volume-Price correlation
    roc_sign = np.sign(df['ROC_1'])
    vol_price_corr = roc_sign * (vol_ratio - 1)

    # Combine
    vol_score = (
        normalize_to_range((vol_ratio - 1) * 100) * 0.40 +
        normalize_to_range((vol_trend - 1) * 100) * 0.30 +
        normalize_to_range(vol_price_corr * 100) * 0.30
    )

    return vol_score * 0.12  # 12% total weight

def calculate_options_score(ticker, price_trend):
    """Calculate options flow score (stocks/ETFs only)"""
    try:
        stock = yf.Ticker(ticker)

        # Get options chain for nearest expiry
        exp_dates = stock.options
        if not exp_dates:
            return 0

        chain = stock.option_chain(exp_dates[0])

        # Calculate Put/Call ratio
        total_put_vol = chain.puts['volume'].sum()
        total_call_vol = chain.calls['volume'].sum()
        pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 1.0

        # Get IV rank (simplified - use ATM options)
        atm_iv = chain.calls.iloc[len(chain.calls)//2]['impliedVolatility']

        # Score Put/Call ratio
        if pc_ratio < 0.7:
            pc_score = (1 - pc_ratio/0.7) * 5
        elif pc_ratio > 1.3:
            pc_score = -(pc_ratio - 1.3) / 0.7 * 5
        else:
            pc_score = 0

        # Score IV (contextual to trend)
        iv_score = atm_iv * 100 if price_trend > 0 else -atm_iv * 100
        iv_score = np.clip(iv_score, -10, 10)

        options_score = (pc_score * 0.5 + iv_score * 0.5) * 0.08
        return options_score

    except:
        return 0  # Skip if options data unavailable
```

---

## Expected Improvements

### Better Signal Quality
- **Avoid False Breakouts**: Low volume breakouts get lower scores
- **Confirm Strong Moves**: High volume rallies get boosted scores
- **Institutional Activity**: Options flow shows smart money positioning

### Example Scenarios

**Scenario 1: Strong Bullish with Conviction**
- Price: +5% (ROC score: +3.5)
- Volume: 2x average (Volume score: +2.0)
- Put/Call: 0.5 (Options score: +0.8)
- **Total boost: +2.8 points**

**Scenario 2: Weak Breakout**
- Price: +5% (ROC score: +3.5)
- Volume: 0.5x average (Volume score: -1.0)
- **Total reduction: -1.0 points**
- Scanner shows this is a weak move, avoid

**Scenario 3: Capitulation Bottom**
- Price: -3% (ROC score: -2.5)
- Volume: 3x average (Volume score: -3.0)
- Put/Call: 2.0 (extreme fear)
- **Total: -5.5 but signals potential reversal**

---

## Recommendation

**Start with Phase 1 (Volume Only)** immediately:
1. Adds significant value with zero dependencies
2. Works across all asset classes
3. Quick to implement and test

**Add Phase 2 (Options)** once volume analysis is validated:
1. Requires API integration and testing
2. Only benefits stocks/ETFs
3. More complex aggregation logic

Would you like me to proceed with implementing Phase 1 (Volume Analysis)?
