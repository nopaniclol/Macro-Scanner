# Macro Metals System — Master Memory & Specification

> **Owner:** Bank precious metals trader (spot / forwards / physical)
> **Purpose:** Hybrid discretionary + systematic framework for daily/weekly trading, designed to improve PnL and serve as an institutional-grade track record suitable for buyside presentation (Jump, Millennium, etc.)
> **Stack:** Python 3 · Bloomberg BQNT / BQL · Claude Code

---

## 1. Project Overview

- Build a multi-strategy systematic trading system centred on precious metals, augmented by FX and rates signals.
- The system combines four alpha modules (momentum, carry, relative-value, cross-jurisdiction arb) under a unified risk/portfolio layer.
- Initial implementation targets daily signal generation and position sizing. Weekly rebalance for slower strategies.
- The system must produce a clean, auditable paper track record from day one, with full backtest support.
- All code lives in this repository (`Macro-Scanner/Macro_Metals/`) and is maintained via Claude Code sessions that reference this memory file.
- End-state deliverables: backtested strategies with Sharpe/drawdown stats, an analytics dashboard, and a live/paper execution layer.

---

## 2. Trading Universe

### 2.1 Metals

| Metal | Primary Exchange | Contracts | Notes |
|-------|-----------------|-----------|-------|
| Gold | COMEX (CME) | GC futures (G/J/M/Q/V/Z), spot XAU | Core instrument. Also LBMA OTC spot/forward. |
| Silver | COMEX (CME) | SI futures (H/K/N/U/Z), spot XAG | Core instrument. Higher vol than gold. |
| Platinum | NYMEX (CME) | PL futures, spot XPT | Relative-value vs gold. Thinner liquidity. |
| Palladium | NYMEX (CME) | PA futures, spot XPD | Relative-value. Very illiquid — use for signals, size small. |
| Copper | COMEX (CME) | HG futures | Base metal reference for macro regime. |
| Gold (Shanghai) | SHFE | AU futures (CNY-denominated) | Cross-jurisdiction arb leg. Subject to price limits. |
| Gold (Tokyo) | TOCOM/OSE | Gold futures (JPY-denominated) | Cross-jurisdiction arb leg. |
| Gold/Silver (London) | LBMA | OTC spot/forward, EFP | Physical delivery hub. Benchmark fixes (AM/PM). |
| LME Metals | LME | Copper, Zinc, Nickel (3M) | Reference for base metals regime only. Not actively traded in system v1. |

### 2.2 FX

| Pair | Role |
|------|------|
| DXY (Dollar Index) | Broad USD strength indicator. |
| XAUUSD, XAGUSD | Metal spot quotes (technically FX). |
| USDCNH / USDCNY | Required for SHFE arb fair-value. Offshore (CNH) is tradeable; onshore (CNY) for reference. |
| USDJPY | Required for TOCOM arb fair-value. Also carry signal (JPY funding). |
| EURUSD | G10 macro signal; EUR gold demand proxy. |
| GBPUSD | LBMA settlement currency relevance. |
| AUDUSD | Gold-beta commodity currency. Carry signal. |
| USDCHF | Safe-haven co-movement with gold. |

### 2.3 Rates

| Instrument | Ticker Pattern | Role |
|------------|---------------|------|
| SOFR IRS 1M–2Y | USOSFRM, USOSFRA, USOSFRB, USOSFR1, USOSFR2 Curncy | Core USD discount curve for carry and fair-value. |
| SOFR futures (SR3) | SR3 + month codes | Short-end rate expectations, momentum signals. |
| Fed Funds target | FDTR Index | Policy rate reference. |
| US Treasuries 2Y/5Y/10Y/30Y | USGG series | Curve shape signals (2s10s, 5s30s). |
| US TIPS / Breakevens | USGGT, USGGBE series | Real yield and inflation expectations. |

---

## 3. Alpha / Strategy Blocks

### 3.1 Time-Series Momentum

**Intent:** Capture trending behaviour in metals, FX, and rates using price-based signals.

- **Signals:** Exponential moving average crossovers (fast/slow), rate-of-change over multiple lookbacks (10d, 21d, 63d, 126d, 252d), and breakout/channel rules.
- **Instruments:** Gold, silver, platinum futures; EURUSD, USDJPY, AUDUSD; SOFR futures front 4 contracts.
- **Sizing:** Volatility-targeted per instrument (see Risk layer). Signal strength scales position between 0 and max allocation.
- **Rebalance:** Daily at settlement (COMEX 1:30pm ET) or Asian open for TOCOM/SHFE.
- **Key design choice:** Blend multiple lookback windows into a composite score to avoid whipsaw from any single timeframe.

### 3.2 Carry / Term-Structure

**Intent:** Harvest roll yield from futures curves and interest rate differentials, filtered by trend to avoid carry traps.

- **FX carry:** Rank G10 pairs by 3M forward-implied carry. Go long high-carry, short low-carry. Apply trend filter (only take carry positions aligned with 3M momentum).
- **Metals carry:** Compute calendar spread carry (implied vs theoretical using SOFR + storage) as built in the COMEX_Calendar_Spreads notebook. Target rich/cheap spreads identified by z-score.
- **SOFR curve roll-down:** Identify steepness in the SOFR curve. When curve is steep and policy expectations are stable, go long front contracts to capture roll-down.
- **Signals:** Carry z-score, carry momentum (change in carry over 20d), carry-to-risk ratio.
- **Rebalance:** Weekly for FX carry; daily for metals spreads and SOFR roll.

### 3.3 Metals Relative-Value

**Intent:** Trade mean-reverting ratios and spreads between precious metals, exploiting supply/demand dislocations and substitution dynamics.

- **Gold/Silver ratio (GSR):** Core signal. Mean-reverts over 1–6 month horizons. Trade via futures spread or spot hedge. Z-score with 1Y rolling window, enter at ±1.5σ.
- **Gold/Platinum ratio:** Platinum discount to gold has structural drivers (auto demand, mining supply). Trade dislocations vs 2Y mean.
- **Gold/Palladium:** Mostly a signal generator due to palladium illiquidity. Use as confirmation for broader precious regime.
- **Copper/Gold ratio:** Macro risk-on/risk-off barometer. Rising ratio = growth optimism. Use as a regime filter for other strategies, not a direct trade.
- **Implementation:** Ratios computed on spot prices. Trades executed via futures legs with appropriate hedge ratios (beta-adjusted for different volatilities).

### 3.4 Cross-Jurisdiction Arbitrage

**Intent:** Exploit price dislocations between the same metal traded on different exchanges/venues, accounting for FX, funding, taxes, and logistics.

- **COMEX vs SHFE Gold:**
  - Fair value: SHFE_AU_CNY / USDCNY ≈ COMEX_GC + (funding differential × time + VAT adjustment).
  - China VAT on gold imports = 0% (exempt), but silver = 13%. This creates structural premium/discount.
  - Key risks: SHFE daily price limits (±7–13%), CNY convertibility, legging risk between time zones.
  - Signal: premium/discount to fair value in USD terms, z-scored over 60d rolling.

- **COMEX vs TOCOM Gold:**
  - Fair value: TOCOM_AU_JPY / USDJPY ≈ COMEX_GC + (SOFR–TONAR differential × time).
  - Historically tighter arb due to open capital account. Main edge is in term-structure differences.
  - Signal: basis vs fair value, with JPY funding cost adjustment.

- **COMEX vs LBMA (EFP):**
  - EFP = COMEX futures – LBMA spot. Driven by physical delivery flows, warehouse stocks, and lease rates.
  - Already have EFP analysis code in this repo. Integrate signals here.
  - Signal: EFP level vs 30d/90d z-score, physical flow indicators (COT, warehouse stocks).

- **Execution considerations:**
  - Always model round-trip transaction costs (commissions + bid-ask + funding).
  - Never assume simultaneous execution across venues — build in legging buffer (e.g. ±0.5σ of recent EFP vol).
  - Size constrained by the less liquid leg.

---

## 4. Risk and Portfolio Layer

### 4.1 Volatility Targeting

- Each instrument position is scaled to a target annualised volatility (default: 10% per leg).
- Use 30-day exponentially weighted realised volatility, updated daily.
- Position size = (target vol × capital allocation) / (instrument vol × contract notional).
- Apply a hard cap of 2× target size to prevent over-leverage in low-vol regimes.

### 4.2 Risk Budget Allocation

| Strategy Block | Base Risk Budget | Notes |
|---------------|-----------------|-------|
| Time-Series Momentum | 30% | Diversified across metals, FX, rates. |
| Carry / Term-Structure | 25% | Lower turnover, steady PnL contribution. |
| Metals Relative-Value | 25% | Mean-reversion; uncorrelated to momentum. |
| Cross-Jurisdiction Arb | 20% | Capacity-constrained; higher Sharpe when available. |

- Budgets are notional risk allocations, not capital allocations. Actual capital usage depends on margin requirements.
- Rebalance risk budgets monthly, or immediately if a block exceeds 1.5× its allocation.

### 4.3 Correlation and Regime Adjustment

- Compute rolling 60-day correlations between strategy blocks.
- When cross-strategy correlation rises above 0.5 (i.e., everything trending together), reduce gross exposure by 20%.
- Simple regime indicator: VIX level + 2s10s slope + DXY momentum → classify as Risk-On, Risk-Off, or Transition.
  - **Risk-Off:** Increase metals momentum allocation, reduce carry, tighten stops.
  - **Risk-On:** Increase carry and relative-value, reduce safe-haven momentum.
  - **Transition:** Reduce gross to 70% of target until regime clarifies.
- Maximum portfolio gross exposure: 400% of capital (i.e., 4× leverage across all legs).
- Maximum portfolio net delta to gold: ±150% of capital.

---

## 5. Data Conventions

### 5.1 Futures Rolling

- **Roll method:** Volume-based roll. Switch to next contract when back-month open interest exceeds front-month, or at first notice date minus 5 business days, whichever comes first.
- **Continuous series:** Construct using backward-adjusted (Panama canal) method to preserve returns and avoid level jumps.
- **Roll calendar:** Maintain a static table of roll dates per contract, updated quarterly.

### 5.2 Holidays and Time Zones

- **Base timezone:** US Eastern (ET). All timestamps stored as UTC internally, displayed in ET.
- **Holiday calendar:** Use CME, LME, SHFE, and TOCOM exchange holiday calendars. On partial-overlap days, only trade instruments on open exchanges.
- **Daily cut-off:** COMEX settlement (1:30 PM ET) is the primary mark. Asian signals computed at SGT 4:00 PM / JST 3:00 PM.

### 5.3 Currency Conventions

- **Base currency:** USD. All PnL, risk metrics, and position values reported in USD.
- **CNH vs CNY:** Use CNH (offshore) for all tradeable FX calculations. Use CNY (onshore) fixing only as a reference for SHFE fair-value.
- **JPY:** Standard USDJPY quoting. Convert TOCOM yen-denominated prices to USD using spot USDJPY at TOCOM settlement time.
- **Cross rates:** Derive from USD pairs (e.g., EURJPY = EURUSD × USDJPY). Do not use independent cross-rate feeds to avoid inconsistency.

### 5.4 Transaction Cost Approximation

- Where exact fee schedules are unavailable, use these defaults:

| Instrument | Half-spread (one-way) | Commission (per contract) |
|------------|----------------------|--------------------------|
| COMEX Gold (GC) | $0.10/oz | $2.50 |
| COMEX Silver (SI) | $0.005/oz | $2.50 |
| G10 FX spot | 0.5 pips | — |
| SOFR futures (SR3) | 0.25 bp | $2.00 |
| SHFE Gold (AU) | ¥0.02/g | ¥10 |
| TOCOM Gold | ¥1/g | ¥300 |

- Funding cost: assume SOFR flat for USD, apply OIS + 25bp spread for non-USD.
- Slippage buffer: add 20% to half-spread estimates for backtest conservatism.

---

## 6. Backtest Framework

### 6.1 Sample Periods

| Period | Dates | Purpose |
|--------|-------|---------|
| In-sample (IS) | 2015-01-01 to 2022-12-31 | Strategy development and calibration. |
| Out-of-sample (OOS) | 2023-01-01 to present | Validation. No parameter changes based on OOS results. |
| Stress periods | 2020-03 (COVID), 2022-03 (Russia/Ukraine), 2023-03 (SVB) | Must survive these without catastrophic drawdown. |

### 6.2 Robustness Tests

- **Parameter sweeps:** For each strategy, vary key parameters ±30% from chosen values. Strategy must remain profitable (Sharpe > 0.5) across >70% of parameter space.
- **Sub-sample stability:** Split IS period into 3 equal sub-periods. Strategy must be profitable in at least 2 of 3.
- **Regime-based analysis:** Report performance separately for Risk-On, Risk-Off, and Transition regimes.
- **Turnover analysis:** Track daily turnover. Flag if annualised turnover exceeds 50× for any strategy (likely over-fit to noise).
- **Transaction cost sensitivity:** Run backtests at 1×, 2×, and 3× estimated costs. Strategy must survive at 2× costs.

### 6.3 Performance and Risk Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Annualised Sharpe ratio | > 1.0 (combined) | Per-strategy target > 0.5. |
| Max drawdown | < 15% | From peak to trough on daily marks. |
| Annualised volatility | 8–12% | Controlled by vol-targeting layer. |
| Hit rate (daily) | > 52% | Not a hard requirement; depends on payoff skew. |
| Calmar ratio | > 0.7 | Return / max drawdown. |
| Annualised turnover | < 30× | Combined across all strategies. |
| Factor exposures | Report | Regress returns on: gold beta, USD beta, rates beta, equity beta, momentum factor, carry factor. |

---

## 7. Implementation Notes for Claude Code

### 7.1 Coding Standards

- **Python 3.10+** with type hints on all function signatures.
- **Vectorised operations:** Use pandas/numpy for all data transformations. Avoid Python-level loops over dates/rows except where unavoidable (e.g., event-driven logic).
- **Modular design:** Each strategy block is its own module. Shared utilities (data loading, risk, plotting) in `common/` or `utils/`.
- **Docstrings:** Every public function and class must have a docstring explaining purpose, args, returns, and any non-obvious logic.
- **Inline comments:** Minimal — only where the formula or logic is not self-evident from the code.
- **No magic numbers:** All parameters must be defined in config dicts or the memory file, never hardcoded in function bodies.

<!-- **UPDATED** — BQuant notebook standards added below -->

- **Notebooks (BQuant):**
  - Self-contained execution — no external `%run .py` files; all logic lives inside the notebook.
  - Target 12–18 cells per notebook: imports → config → data (BQL) → strategy → backtest → charts → export.
  - Use markdown headers between major sections (`## Data` | `## Signals` | `## Results`).
  - Use Plotly interactive charts (`px.line`, `px.imshow`, `px.scatter_matrix`) as the default visualisation library.
  - Export signals and results to the `outputs/` directory: `signals.to_csv("outputs/name.csv")`, `summary.to_html()`.
  - Print key DataFrames inline for quick verification: `signals.head(10)`, `metrics.round(3)`.

<!-- end **UPDATED** -->

<!-- **UPDATED** — Project structure replaced for BQuant notebook-only architecture -->

### 7.2 Project Structure (Target)

```
Macro_Metals/
├── macro_metals_system_memory.md        # This file — master spec
├── notebooks/                           # Self-contained research notebooks
│   ├── 00_data_test.ipynb               # BQL pipeline validation
│   ├── 01_momentum_research.ipynb
│   ├── 02_carry_research.ipynb
│   ├── 03_relative_value_research.ipynb
│   ├── 04_efp_basis_research.ipynb
│   ├── 05_cross_exchange_research.ipynb
│   └── 06_full_portfolio_research.ipynb  # Risk layer + combine all
├── config/
│   ├── parameters.yaml
│   └── tickers.yaml
└── outputs/                             # Generated CSVs/charts from notebooks
    └── signals_YYYYMMDD.csv
```

<!-- end **UPDATED** -->

### 7.3 Consistency Rule

> **All code generated by Claude Code must be consistent with this memory file.** If a new decision contradicts something here, update the memory file first, then write the code. Never silently deviate.

---

## 8. Open Questions / Parking Lot

- [ ] **Macro data factors:** Should we incorporate PMI, CPI surprises, central bank meeting dates as regime signals or alpha inputs?
- [ ] **Options overlay:** Could we use gold/silver options (vol surface, skew, put/call ratios) as additional signals or for tail-risk hedging?
- [ ] **Commodity expansion:** Extend to energy (crude, nat gas) or agriculture? These have different dynamics but could diversify the portfolio.
- [ ] **Execution modelling:** Build a more sophisticated execution simulator (TWAP/VWAP, market impact, queue position) beyond simple cost assumptions.
- [ ] **Machine learning:** Explore gradient-boosted trees or neural nets for signal combination, but only after linear models are established as baseline.
- [ ] **Lease rates / GOFO:** Gold forward offered rate is discontinued but can be backed out from forwards. Could be a useful carry signal.
- [ ] **COT positioning:** CFTC Commitments of Traders data — use as a contrarian or confirmation signal for metals.
- [ ] **Central bank gold demand:** Track official sector purchases (WGC data) as a slow-moving structural signal.
- [ ] **Warehouse stock data:** COMEX registered/eligible inventory, LME warehouse stocks as physical tightness indicators.
- [ ] **Multi-account / fund structure:** If presenting to buyside, will need to model management fees, performance fees, and high-water marks in backtest.
- [ ] **Real-time alerting:** Build a notification layer (email/Slack) for signal triggers and risk limit breaches.
- [ ] **Dashboard:** Streamlit or Dash web app for daily monitoring — positions, PnL, signals, risk usage.

---

*Last updated: 2026-02-15 — Updated §7.1 and §7.2 for BQuant notebook-only architecture.*
