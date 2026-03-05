# Macro Metals System — Master Memory & Specification

> **Owner:** Bank precious metals trader (spot / forwards / physical)
> **Purpose:** Hybrid discretionary + systematic framework for daily/weekly trading, designed to improve PnL and serve as an institutional-grade track record suitable for buyside presentation (Jump, Millennium, etc.)
> **Stack:** Python 3 · Bloomberg BQNT / BQL · Claude Code

---

## 1. Project Overview

- Build a multi-strategy systematic trading system centred on precious metals, augmented by FX and rates signals.
- The system combines four alpha modules (momentum, carry, lease rate curve, COMEX EFP) under a unified risk/portfolio layer.
- Initial implementation targets daily signal generation and position sizing. Weekly rebalance for slower strategies.
- The system must produce a clean, auditable paper track record from day one, with full backtest support.
- All code lives in this repository (`Macro-Scanner/Macro_Metals/`) and is maintained via Claude Code sessions that reference this memory file.
- End-state deliverables: backtested strategies with Sharpe/drawdown stats, an analytics dashboard, and a live/paper execution layer.

---

## 2. Trading Universe

<!-- UPDATED 2026-03-05 -->
### 2.1 Metals

| Metal | Primary Exchange | Contracts in data.csv | Notes |
|-------|-----------------|----------------------|-------|
| Gold | COMEX (CME) | gc_fut_front, gc_fut_second, gc_fut_third; xauusd_spot | Core instrument. Also LBMA OTC spot/forward. |
| Silver | COMEX (CME) | si_fut_front, si_fut_second, si_fut_third; xagusd_spot | Core instrument. Higher vol than gold. |
| Platinum | NYMEX (CME) | pl_fut_front, pl_fut_second; xptusd_spot | Relative-value vs gold. Thinner liquidity. |
| Palladium | NYMEX (CME) | pa_fut_front, pa_fut_second; xpdusd_spot | Relative-value. Very illiquid — use for signals, size small. |
| Copper | COMEX (CME) | hg_fut_front, hg_fut_second | Base metal reference for macro regime. |
| Gold (Shanghai) | SHFE | AU futures (CNY-denominated) | Cross-jurisdiction arb leg (deferred — see §8). |
| Gold (Tokyo) | TOCOM/OSE | Gold futures (JPY-denominated) | Cross-jurisdiction arb leg (deferred — see §8). |
| LBMA Fixes | LBMA | lbma_gold_am, lbma_gold_pm, lbma_silver_fix, lbma_platinum_am | OTC benchmark fixes. No LBMA palladium fix in dataset. |
| LME Base Metals | LME | lme_copper_3m, lme_zinc_3m, lme_nickel_3m, lme_aluminium_3m, lme_lead_3m, lme_tin_3m | Six 3M rolling contracts. Reference for base metals regime; not actively traded in v1. |

<!-- UPDATED 2026-03-05 -->
### 2.2 FX

| Pair / Ticker | Role |
|---------------|------|
| DXY (dxy_index) | Broad USD strength indicator. |
| XAUUSD, XAGUSD (xauusd_spot, xagusd_spot) | Metal spot quotes (technically FX). |
| EURUSD (eurusd_spot, eurusd_3m_fwd) | G10 macro signal; EUR gold demand proxy. |
| USDJPY (usdjpy_spot, usdjpy_3m_fwd) | Required for TOCOM arb fair-value. Carry signal (JPY funding). |
| GBPUSD (gbpusd_spot, gbpusd_3m_fwd) | LBMA settlement currency relevance. |
| AUDUSD (audusd_spot, audusd_3m_fwd) | Gold-beta commodity currency. Carry signal. |
| USDCHF (usdchf_spot) | Safe-haven co-movement with gold. |
| USDCNH / USDCNY (usdcnh_spot, usdcny_spot, usdcnh_3m_fwd) | Required for SHFE arb fair-value. CNH tradeable; CNY for reference. |
| NZDUSD (nzdusd_spot) | Commodity currency exposure. |
| USDCAD (usdcad_spot, usdcad_3m_fwd) | Energy-linked commodity currency. |
| USDNOK (usdnok_spot) | Oil-correlated G10 carry. |
| USDSEK (usdsek_spot) | G10 completion. |
| USDSGD (usdsgd_spot) | EM Asia proxy. |
| USDBRL (usdbrl_spot) | EM commodity currency; Brazil iron ore / agriculture. |
| USDMXN (usdmxn_spot) | EM energy-linked; near-shoring macro. |
| USDKRW (usdkrw_spot) | EM Asia industrial cycle proxy. |
| USDINR (usdinr_spot) | EM Asia; gold physical demand (India). |
| USDTRY (usdtry_spot) | EM high-carry; inflation-hedging demand proxy. |
| USDZAR (usdzar_spot) | EM gold mining currency (South Africa). |
| USDPLN (usdpln_spot) | EM Europe; CEE proxy. |
| USDCLP (usdclp_spot) | EM copper-linked (Chile). |
| USDIDR (usdidr_spot) | EM Asia commodity importer. |

3M FX forwards in dataset: EURUSD, USDJPY, GBPUSD, AUDUSD, USDCNH, USDCAD.

<!-- UPDATED 2026-03-05 -->
### 2.3 Rates

| Instrument | Logical Name Pattern | Role |
|------------|---------------------|------|
| **SOFR OIS — 15 pillars** | sofr_ois_1w, sofr_ois_2w, sofr_ois_1m … sofr_ois_11m, sofr_ois_1y, sofr_ois_2y | Core USD discount curve for carry and fair-value. Full pillar set: 1w, 2w, and every calendar month 1m–11m, then 1y and 2y. Bloomberg tickers: USOSFR1Z, USOSFR2Z, USOSFRA–USOSFRK, USOSFR1, USOSFR2 Curncy. |
| **SOFR futures** | sofr_fut_front … sofr_fut_fourth | Short-end rate expectations; momentum signals. |
| **XAU swap rates — 8 tenors** | xau_swap_1m, xau_swap_2m, xau_swap_3m, xau_swap_4m, xau_swap_5m, xau_swap_6m, xau_swap_9m, xau_swap_2y | Gold forward rates by tenor. Used to derive XAU lease rates. |
| **XAG swap rates — 6 tenors** | xag_swap_1m, xag_swap_2m, xag_swap_3m, xag_swap_6m, xag_swap_9m, xag_swap_2y | Silver forward rates. Note: xag_swap_9m and xag_swap_2y flagged as low-quality and excluded in lease rate model. |
| **XAU / XAG lease rates** | DERIVED — not a raw Bloomberg field | Computed as: `lease_rate = xau_swap_{tenor} − sofr_ois_{tenor}` for each matched tenor. NOT pulled directly from Bloomberg. |
| **Fed Funds target** | fed_funds_target | Policy rate reference. |
| **US Treasury yields** | us_2y_yield, us_5y_yield, us_10y_yield, us_30y_yield | Curve shape signals (2s10s, 5s30s). |
| **TIPS real yields** | tips_5y_real_yield, tips_10y_real_yield | Real yield and inflation expectations. |
| **UST futures** | ust_2y_fut, ust_5y_fut, ust_10y_fut, ust_30y_fut | Rates momentum signals (TU, FV, TY, US). |
| **Global OIS** | estr_ois_1y, sonia_ois_1y | EUR (ESTR) and GBP (SONIA) 1Y OIS rates. |
| **Euribor futures** | euribor_3m_fut_front | EUR short-end expectations. |
| **STIBOR** | stibor_3m | SEK 3M reference rate. |
| **Government bond futures** | bund_fut_front, bobl_fut_front, schatz_fut_front, oat_fut_front, btp_fut_front, gilt_fut_front, jgb_fut_front, aus_bond_fut_front, cad_bond_fut_front | Global duration exposure for TSMOM rates bucket. EUR (Bund/Bobl/Schatz/OAT/BTP), GBP (Gilt), JPY (JGB), AUD (ASX bond), CAD. |
| **Index-linked bonds** | uk_rii_10y, fr_oati_10y | UK and French inflation-linked bond yields. |

<!-- NEW 2026-03-05 -->
### 2.4 Equities

All equity instruments are index futures. Present in data.csv and included in the TSMOM universe.

| Index | Logical Name | Region |
|-------|-------------|--------|
| S&P 500 | es_fut_front | US large cap |
| Nasdaq 100 | nq_fut_front | US tech |
| Russell 2000 | russell2000_fut | US small cap |
| Euro STOXX 50 | stoxx50_fut_front | EUR large cap |
| DAX | dax_fut_front | Germany |
| FTSE 100 | ftse_fut_front | UK |
| CAC 40 | cac_fut_front | France |
| SMI | smi_fut_front | Switzerland |
| IBEX 35 | ibex_fut_front | Spain |
| Nikkei 225 | nikkei_fut_front | Japan |
| Hang Seng | hsi_fut_front | Hong Kong / China |
| ASX 200 | asx_fut_front | Australia |
| KOSPI 200 | kospi_fut_front | South Korea |
| S&P/TSX | tsx_fut_front | Canada |
| OMX 30 | omx_fut_front | Sweden |

<!-- NEW 2026-03-05 -->
### 2.5 Commodities Expansion

Beyond the core metals, the following commodities are in data.csv and the TSMOM universe.

**Energy**

| Instrument | Logical Name | Notes |
|-----------|-------------|-------|
| WTI Crude (front) | cl_fut_front | Core energy |
| WTI Crude (2nd) | cl_fut_second | Calendar spread reference |
| Brent Crude (front) | brent_fut_front | International benchmark |
| Brent Crude (2nd) | brent_fut_second | Calendar spread reference |
| Heating Oil | ho_fut_front | Distillate crack spread |
| RBOB Gasoline | rb_fut_front | Gasoline crack spread |
| Gasoil | gasoil_fut_front | European distillate |
| Natural Gas (front) | ng_fut_front | US Henry Hub |
| Natural Gas (2nd) | ng_fut_second | Calendar spread reference |
| Carbon EUA | carbon_eua_front | EU Emissions Allowance |

**Agriculture**

| Instrument | Logical Name |
|-----------|-------------|
| CBOT Wheat | w_fut_front |
| KC Wheat | kw_fut_front |
| CBOT Corn | c_fut_front |
| CBOT Soybean | s_fut_front |
| Soybean Oil | bo_fut_front |
| Soybean Meal | sm_fut_front |
| Cotton | ct_fut_front |
| Cocoa | cc_fut_front |
| KC Coffee | kc_fut_front |
| Sugar #11 | sb_fut_front |
| Live Cattle | lc_fut_front |
| Feeder Cattle | fc_fut_front |
| Lean Hogs | lh_fut_front |
| Oats | oz_fut_front |
| Rough Rice | rr_fut_front |
| Lumber | lumber_fut_front |

<!-- NEW 2026-03-05 -->
### 2.6 Risk Indicators

| Indicator | Logical Name | Notes |
|-----------|-------------|-------|
| VIX | vix | S&P 500 implied vol (30d) |
| MOVE | move_index | US Treasury implied vol index |
| Gold vol (GVZ) | gold_vol_index | Gold ETF implied vol |
| Oil vol (OVX) | oil_vol_index | Crude oil implied vol |
| VSTOXX (V2X) | vstoxx_index | Euro STOXX 50 implied vol |
| VXN | vxn_index | Nasdaq 100 implied vol |
| RVX | rvx_index | Russell 2000 implied vol |
| SKEW | skew_index | S&P 500 tail risk index |
| VXEEM | vxeem_index | EM equities implied vol |

---

## 3. Alpha / Strategy Blocks

<!-- UPDATED 2026-03-05 -->
### 3.1 Time-Series Momentum

**Intent:** Capture trending behaviour across a broad multi-asset universe using price-based signals (Moskowitz, Ooi & Pedersen 2012).

- **Signal construction:** `sign(r_252d)` — sign of the 12-month trailing return. Monthly rebalance: signal frozen at each month-end and forward-filled daily until next rebalance.
- **Lookback:** 252 trading days (12-month return). Sensitivity grid tested at 126d and 504d; 252d selected.
- **Vol scaling:** EWMA realised volatility with λ = 0.94, updated daily. Position size = (TARGET_VOL × capital) / (instrument_vol × contract_notional). TARGET_VOL = 10% p.a.
- **Leverage cap:** 2× target size per instrument (LEV_CAP = 2.0).
- **Universe:** 71 instruments across 4 buckets (drawn from the 163-series master dataset in data.csv):
  - Commodities: 25 (precious metals, energy, agriculture, base metals)
  - Rates: 18 (UST futures, global bond futures, SOFR futures)
  - Equities: 15 (global index futures)
  - FX: 14 (G10 + major EM pairs)
- **Transaction cost assumption:** 2 bp per side (TC_BP = 2.0).
- **Rebalance:** Monthly at month-end settlement.
- **IS/OOS split:** IS 2015–2022; OOS 2023–present.
- **Note:** Strategy is implemented entirely within `01_momentum_research.ipynb`. BQL/BQuant constraint means all data-pull logic must live inside the notebook.

<!-- UPDATED 2026-03-05 -->
### 3.2 Cash & Carry (COMEX Calendar Spread)

**Intent:** Harvest the positive roll yield when COMEX metal futures curves are in contango, net of funding and storage costs.

- **Instruments:** GC (Gold, delivery months: Feb/Apr/Jun/Aug/Oct/Dec) and SI (Silver, delivery months: Jan/Mar/May/Jul/Sep/Dec). Front vs second contract only.
- **Net carry formula:**
  ```
  net_carry_bp_ann = switch_bp_ann − sofr_ois_interpolated_bp_ann − storage_bp_ann
  where:
    switch_bp_ann  = (F2 − F1) / F1 × (360 / days_F1_to_F2) × 10000
    storage_bp_ann = 20 bp p.a. (GC) | 32 bp p.a. (SI)
    SOFR OIS       = interpolated from 15-pillar curve to match DTE
  ```
- **Entry:** net_carry_bp_ann ≥ CARRY_MIN_BP (= 80 bp) AND dte_fnd1 ≥ MIN_DTE_FND (= 5 days before First Notice Date).
- **Position:** long front contract, short second contract (calendar spread, no outright delta).
- **Exit / roll:** At FDD of the front contract. If carry still ≥ 80 bp, roll forward; otherwise close.
- **Execution cost:** EXEC_COST_BP = 0.50 bp per leg.
- **P&L per leg:** `locked_carry_bp_ann × hold_days / 360 − exec_cost_bp × 2`.
- **Output files:** `cc_trades_gc.csv`, `cc_trades_si.csv`, `cc_sweep_results.csv`, plus PNG charts (prefix `cc_`).
- **Note:** Implemented as a self-contained Jupyter notebook (`comex_cash_and_carry.ipynb`). No `.py` module imports.

<!-- UPDATED 2026-03-05 -->
### 3.3 Lease Rate Curve Model

**Intent:** Trade mean-reversion in XAU and XAG implied lease rates across the forward curve, using z-score signals derived from metal swap rates relative to SOFR OIS.

- **Lease rate derivation:** `lease_rate_{tenor} = xau_swap_{tenor} − sofr_ois_{tenor}` (same for XAG). Units: annualised %, ACT/360. NOT a direct Bloomberg field — derived from swap and OIS pillars.
- **XAU tenors traded:** 1m, 2m, 3m, 4m, 5m, 6m, 9m, 2y (8 tenors in dataset).
- **XAG tenors traded:** 1m, 2m, 3m, 6m (xag_swap_9m and xag_swap_2y excluded as anomalous).
- **Focus tenors for trading signals:** 1m, 3m, 6m, 1y.
- **Curve shape features:**
  - Slope: `lease_3m − lease_1m` (and other tenor pairs)
  - Forward lease: derived from swap discount factors across adjacent tenors
  - Butterfly: `t1 − 2×t2 + t3`
  - Cross-metal spread: `xag_lease_{tenor} − xau_lease_{tenor}`
- **Z-score lookback:** 252-day rolling (basis/EFP features); 504-day rolling (calendar spread features). Min periods: 126 / 252 respectively.
- **Signal thresholds:** Entry |z| ≥ 1.5σ; exit |z| ≤ 0.5σ; stop |z| ≥ 3.5σ.
- **Backwardation vs contango:** Negative lease rates = backwardation (gold scarce to borrow, typically in stress); positive = contango (normal).
- **Portfolio:** 10 strategies selected post-correlation filter (threshold = 0.6) to ensure orthogonality.
- **Output files:** `02_lease_features.csv`, `02_lease_history_xau.png`, `03_zscore_heatmap_lease.png`.
- **Note:** Implemented in `silver_gold_lease_curve_v2.ipynb`.

<!-- UPDATED 2026-03-05 -->
### 3.4 COMEX EFP Model

**Intent:** Trade mean-reversion in the EFP (Exchange for Physical) basis between COMEX gold/silver futures and OTC spot, z-scored to identify statistically extreme dislocations.

- **EFP definition:** `efp_ann_pct = (F_front − spot) / spot × (360 / dte_front) × 100`. Front contract only (EFP_DAYS ≈ 30).
- **Fair value:** EFP should approximately equal SOFR OIS + lease rate for the same tenor. Deviations represent mispricings driven by physical delivery flows, warehouse stocks, and funding dislocations.
- **Signal construction:** Z-score of EFP basis vs 252-day rolling mean/std (min 126 periods). Calendar spread features use 504-day window (min 252 periods).
- **Signal thresholds (sweep grid):** Entry z ∈ {1.5, 2.0, 2.5}; exit z ∈ {0.3, 0.5, 0.75}; stop z = 3.5.
- **Directional logic:** z > entry_z → sell futures / buy spot (basis too wide); z < −entry_z → buy futures / sell spot (basis too narrow).
- **DTE guard:** EFP_MIN_DTE = 3 days (exclude entries within 3 days of FND to avoid roll contamination).
- **OI roll period flag:** `gc_roll_period = (gc_oi_pct_front < 0.50)` — suppress calendar spread signals when front OI share falls below 50% (active roll).
- **EFP basis clip:** Artefacts beyond ±800 bp annualised are clipped (EFP_CLIP_BP = 800).
- **Output files:** `01_comex_features.csv`, `02_comex_signals.csv`, `03_zscore_validation.png`, `04_sweep_results.csv`, `05_comex_portfolio.csv`, `06_combined_portfolio.csv`, `07_attribution.csv`, `08_daily_blotter.csv`, `3c_efp_sweep.csv`, `3c_summary.csv`.
- **Note:** Implemented in `comex_efp_expansion.ipynb`. Files prefixed `01_`–`08_` and `3c_`. No naming conflict with C&C outputs (which use `cc_` prefix).

---

## 4. Risk and Portfolio Layer

<!-- UPDATED 2026-03-05 -->
### 4.1 Volatility Targeting

- Each instrument position is scaled to TARGET_VOL = 10% annualised volatility per leg.
- Use EWMA realised volatility with decay factor λ = 0.94, updated daily.
- Position size = (target_vol × capital_allocation) / (instrument_vol × contract_notional).
- Apply a hard cap of 2× target size per instrument to prevent over-leverage in low-vol regimes (LEV_CAP = 2.0).

<!-- UPDATED 2026-03-05 -->
### 4.2 Risk Budget Allocation

| Strategy | Risk Budget | Capital Weight | Notes |
|----------|------------|---------------|-------|
| Time-Series Momentum | 30% | 60% | Diversified across 71 instruments. |
| Cash & Carry | 10% | 10% | Low turnover; capacity-limited by physical delivery. |
| Lease Rate Curve | 20% | 20% | Mean-reversion; uncorrelated to momentum. |
| COMEX EFP | 20% | 10% | Higher Sharpe when dislocations occur; capacity-constrained. |

- Capital weights reflect a $100M NAV reference book. Risk budgets are notional; actual capital usage depends on margin.
- COMEX GC initial margin currently ~9%; SI ~18% (as of Jan/Feb 2026 post-volatility spike).
- Rebalance risk budgets monthly, or immediately if a block exceeds 1.5× its allocation.

<!-- UPDATED 2026-03-05 -->
### 4.3 Correlation and Regime Adjustment

- Compute rolling 60-day correlations between strategy blocks.
- When cross-strategy correlation rises above 0.5 (i.e., everything trending together), reduce gross exposure by 20%.
- Simple regime indicator: VIX level + 2s10s slope + DXY momentum → classify as Risk-On, Risk-Off, or Transition.
  - **Risk-Off:** Increase metals momentum allocation, reduce carry, tighten stops.
  - **Risk-On:** Increase carry and relative-value, reduce safe-haven momentum.
  - **Transition:** Reduce gross to 70% of target until regime clarifies.
- **HIGH_VOL regime definition:** When 20-day realised vol of GC front contract exceeds 30% annualised — halve vol targets for TSMOM and Cash & Carry; maintain Lease Rate and EFP targets (basis strategies tend to widen during vol spikes, creating opportunity).
- Maximum portfolio gross exposure: 800% of capital (8× hard cap across all legs).
- Maximum portfolio net delta to gold: ±150% of capital.

<!-- NEW 2026-03-05 -->
### 4.4 Per-Strategy Leverage Limits

| Strategy | Max Gross Leverage | Notes |
|----------|--------------------|-------|
| TSMOM | 8× | Diversified across 71 instruments; gross scales with breadth. |
| Lease Rate Curve | 4× | Basis strategy; leverage applied per tenor. |
| COMEX EFP | 5× | Basis strategy; leverage applied per metal. |
| Cash & Carry | 1.5× | Physical delivery constraint limits leverage. |
| Portfolio hard cap | 8× | Aggregate across all strategies. |

---

## 5. Data Conventions

<!-- UPDATED 2026-03-05 -->
### 5.1 Futures Rolling

- **Roll trigger:** Enter/exit positions at least 5 business days before First Notice Date (MIN_DTE_FND = 5). FND = last business day before First Delivery Date (FDD).
- **Continuous series:** Construct using backward-adjusted (Panama canal) method to preserve returns and avoid level jumps.
- **Contract date data:** BQL fields used — `fut_dlv_dt_first` (FDD), `fut_notice_first` (FND), `last_tradeable_dt` (LTD), `open_int` (OI time series). Output stored in `data_contract_dates.csv`.
- **Note on BQL FDD data:** `fut_dlv_dt_first()` on generic rolling contracts (GC1) is point-in-time only — all historical rows return the current active contract's FDD. Use synthetic FDD calendar (`first_biz_day(year, month)` via `np.busday_offset`) for historical backtesting. OI columns (`open_int`) are valid historical time series.
- **Roll calendar:** Maintain via synthetic business-day calendar. GC delivery months: Feb/Apr/Jun/Aug/Oct/Dec. SI delivery months: Jan/Mar/May/Jul/Sep/Dec.

### 5.2 Holidays and Time Zones

- **Base timezone:** US Eastern (ET). All timestamps stored as UTC internally, displayed in ET.
- **Holiday calendar:** Use CME, LME, SHFE, and TOCOM exchange holiday calendars. On partial-overlap days, only trade instruments on open exchanges.
- **Daily cut-off:** COMEX settlement (1:30 PM ET) is the primary mark. Asian signals computed at SGT 4:00 PM / JST 3:00 PM.

### 5.3 Currency Conventions

- **Base currency:** USD. All PnL, risk metrics, and position values reported in USD.
- **CNH vs CNY:** Use CNH (offshore) for all tradeable FX calculations. Use CNY (onshore) fixing only as a reference for SHFE fair-value.
- **JPY:** Standard USDJPY quoting. Convert TOCOM yen-denominated prices to USD using spot USDJPY at TOCOM settlement time.
- **Cross rates:** Derive from USD pairs (e.g., EURJPY = EURUSD × USDJPY). Do not use independent cross-rate feeds to avoid inconsistency.

<!-- UPDATED 2026-03-05 -->
### 5.4 Transaction Cost Approximation

Per-strategy cost assumptions:

| Strategy | Slippage | Financing | Roll Cost |
|----------|----------|-----------|-----------|
| TSMOM | 3–5 bps | SOFR + 25 bps | 1–2 bps |
| Lease Rate / EFP | 1–2 bps | SOFR + 15 bps | ~0 |
| Cash & Carry | 0.5–1 bps | SOFR + 40 bps | ~0 |

Instrument-level defaults (where exact fee schedules are unavailable):

| Instrument | Half-spread (one-way) | Commission (per contract) |
|------------|----------------------|--------------------------|
| COMEX Gold (GC) | $0.10/oz | $2.50 |
| COMEX Silver (SI) | $0.005/oz | $2.50 |
| G10 FX spot | 0.5 pips | — |
| SOFR futures (SR3) | 0.25 bp | $2.00 |
| SHFE Gold (AU) | ¥0.02/g | ¥10 |
| TOCOM Gold | ¥1/g | ¥300 |

Slippage buffer: add 20% to half-spread estimates for backtest conservatism.

---

## 6. Backtest Framework

<!-- UPDATED 2026-03-05 -->
### 6.1 Sample Periods

| Period | Dates | Purpose |
|--------|-------|---------|
| In-sample (IS) | 2015-01-01 to 2022-12-31 | Strategy development and calibration. |
| Out-of-sample (OOS) | 2023-01-01 to present (2026) | Validation. No parameter changes based on OOS results. |
| Stress: COVID | 2020-03-01 to 2020-06-30 | Liquidity crisis; extreme vol. |
| Stress: Russia/Ukraine | 2022-02-24 to 2022-06-30 | Commodity supply shock; sanctions. |
| Stress: SVB / Banking | 2023-03-01 to 2023-05-31 | Credit stress; rates vol spike. |
| Stress: 2025 Tariff Shock | 2025-03-01 to 2025-04-30 | Trade war escalation; metals-FX dislocations. |

### 6.2 Robustness Tests

- **Parameter sweeps:** For each strategy, vary key parameters ±30% from chosen values. Strategy must remain profitable (Sharpe > 0.5) across >70% of parameter space.
- **Sub-sample stability:** Split IS period into 3 equal sub-periods. Strategy must be profitable in at least 2 of 3.
- **Regime-based analysis:** Report performance separately for Risk-On, Risk-Off, and Transition regimes.
- **Turnover analysis:** Track daily turnover. Flag if annualised turnover exceeds 50× for any strategy (likely over-fit to noise).
- **Transaction cost sensitivity:** Run backtests at 1×, 2×, and 3× estimated costs. Strategy must survive at 2× costs.

<!-- UPDATED 2026-03-05 -->
### 6.3 Performance and Risk Metrics

Target metrics for the combined $100M NAV reference book:

| Metric | Target | Notes |
|--------|--------|-------|
| Sharpe (combined) | > 0.9 | Target range 0.9–1.3. Per-strategy target > 0.5. |
| Max Drawdown | < 18% | Hard pause at 10% drawdown for strategy review. |
| Ann. Volatility | 8–12% | Vol-targeting controls this. |
| Calmar ratio | > 0.7 | Ann. return / max drawdown. |
| Gross Leverage | 3–6× avg | Hard cap 8×. |
| Ann. Return (net) | 10–18% | $100M reference book, net of costs. |
| Hit rate (daily) | > 52% | Not a hard requirement; depends on payoff skew. |
| Factor exposures | Report | Regress on: gold beta, USD beta, rates beta, equity beta, momentum factor, carry factor. |

---

## 7. Implementation Notes for Claude Code

<!-- UPDATED 2026-03-05 -->
### 7.1 Coding Standards — BQuant Notebook Standards

- **Python 3.10+** with type hints on all function signatures.
- **Vectorised operations:** Use pandas/numpy for all data transformations. Avoid Python-level loops over dates/rows except where unavoidable (e.g., event-driven backtest logic).
- **No magic numbers:** All parameters must be defined in CONFIG dicts or this memory file, never hardcoded in function bodies.
- **Docstrings:** Every public function must have a docstring explaining purpose, args, returns, and any non-obvious logic.
- **Inline comments:** Minimal — only where the formula or logic is not self-evident from the code.

- **Notebooks (BQuant):**
  - Self-contained execution — no external `%run .py` files or module imports; all logic lives inside the notebook.
  - Target 12–18 cells per notebook: imports → config → data (BQL) → strategy → backtest → charts → export.
  - Use markdown headers between major sections (`## Data` | `## Signals` | `## Results`).
  - Use Plotly interactive charts (`px.line`, `px.imshow`, `px.scatter_matrix`) as the default visualisation library.
  - Export signals and results to the `outputs_comex/` directory.
  - Print key DataFrames inline for quick verification: `signals.head(10)`, `metrics.round(3)`.

- **BQL constraint:** Bloomberg BQL API only works inside Bloomberg BQuant Jupyter notebooks. All data-pull logic must therefore live in notebook cells, never in separate `.py` files. This applies to `DragData.ipynb` and all live signal notebooks. The LOCAL CSV variants (`*_LOCAL.ipynb`) are the exception — they use `LocalCSVLoader` to replicate BQL outputs from saved CSVs for offline testing.

<!-- UPDATED 2026-03-05 -->
### 7.2 Project Structure

```
Macro-Scanner/
└── Macro_Metals/
    ├── macro_metals_system_memory.md        ← This file — master spec
    ├── notebooks/
    │   ├── DragData.ipynb                   ← BQL data pipeline (pulls data.csv + data_contract_dates.csv)
    │   ├── 01_momentum_research.ipynb       ← TSMOM: 71-instrument universe, BQL live version
    │   ├── 01_momentum_research_LOCAL.ipynb ← TSMOM: CSV-based version for offline testing
    │   ├── silver_gold_lease_curve_v2.ipynb ← Lease rate curve model
    │   ├── comex_efp_expansion.ipynb        ← EFP basis model
    │   ├── comex_cash_and_carry.ipynb       ← Calendar spread cash & carry model
    │   ├── MasterBacktest.ipynb             ← (TO BUILD) Unified backtest, all 4 strategies
    │   ├── LiveSignalEngine.ipynb           ← (TO BUILD) Daily live signal engine
    │   ├── data/
    │   │   ├── data.csv                     ← Master dataset: 163 series, 2018–present
    │   │   └── data_contract_dates.csv      ← FDD/FND/LTD + OI per contract (GC, SI, HG, CL, etc.)
    │   └── outputs_comex/                   ← Generated CSVs and charts from all notebooks
    └── config/
        ├── tickers.yaml                     ← Single source of truth for all Bloomberg tickers
        └── parameters.yaml
```

Next two notebooks to build:
- **`MasterBacktest.ipynb`** — Unified backtest combining all 4 strategies. $100M NAV reference. Daily P&L engine with per-strategy attribution. Cells 1–8 target structure.
- **`LiveSignalEngine.ipynb`** — Daily live signal engine. Runs at 17:00 ET post-COMEX close. BQL data pull → signal recompute → position diff → trade blotter + risk report. Output to `outputs_comex/signals/YYYY-MM-DD/`.

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
- [ ] **SHFE/TOCOM cross-jurisdiction arb:** Originally planned as a core strategy block. SHFE AU futures (CNY) vs COMEX GC, and TOCOM gold (JPY) vs COMEX. Fair value requires USDCNY/USDJPY adjustment + funding differential. Key risks: SHFE daily price limits (±7–13%), CNY convertibility, legging risk across time zones. Deferred — add as Strategy 5 once core four strategies are live.
- [ ] **MasterBacktest.ipynb:** Build unified backtester combining all 4 strategies. $100M NAV reference. Daily P&L engine with per-strategy attribution. Target metrics: Sharpe 0.9–1.3, Max DD < 18%, Gross leverage 3–6×.
- [ ] **LiveSignalEngine.ipynb:** Daily live signal notebook. Runs at 17:00 ET post-COMEX close. BQL data pull → signal recompute → position diff → trade blotter + risk report. Output to `signals/YYYY-MM-DD/`.
- [ ] **Margin environment monitoring:** COMEX raised GC margin to ~9% and SI to ~18% following Jan/Feb 2026 volatility. Build a margin tracker that reads current SPAN requirements from CME and adjusts effective leverage ceilings dynamically.
- [ ] **Gold/Silver ratio (GSR) strategy:** Mean-reverts over 1–6 month horizons. Z-score with 1Y rolling window, enter at ±1.5σ. Trade via futures spread. Not yet implemented — natural extension of lease curve model.

---

*Last updated: 2026-03-05 — Major update: expanded trading universe (§2.1–2.6), replaced §3.2–3.4 with actual implemented strategies, updated risk/portfolio layer (§4), revised data conventions (§5.1, §5.4), updated backtest targets (§6), updated project structure (§7.2).*
