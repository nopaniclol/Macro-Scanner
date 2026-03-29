# PROJECT MANDATE: Gold Delta-Neutral Research

## PHASE
Research layer only — signal discovery + validation.
No production backtesting, execution simulation, or portfolio construction.

## SLEEVES
1. **Gold Curve RV** — calendar spreads (GC1-GC2, GC2-GC3) and flies (GC1 - 2×GC2 + GC3)
2. **Gold-Silver Ratio RV** — cross-asset ratio mean reversion (GC_front / SI_front)

## NON-OBJECTIVES
- Production backtesting or execution simulation
- Contract roll perfectionism
- Portfolio-level construction
- Overfitting to specific sub-periods

## DATA
- **Primary**: `01_data_raw/data-2.csv` — gold/silver futures + spot/LBMA/swap context
- **Secondary (cautious)**: `01_data_raw/data_contract_dates.csv` — may have alignment issues; do not use for roll logic in research phase

### Key columns
- Gold futures: `gc_fut_front`, `gc_fut_second`, `gc_fut_third`
- Silver futures: `si_fut_front`, `si_fut_second`, `si_fut_third`
- Spot: `xauusd_spot`, `xagusd_spot`
- LBMA: `lbma_gold_am`, `lbma_gold_pm`, `lbma_silver_fix`
- Gold swaps: `xau_swap_1m`, `xau_swap_3m`, `xau_swap_6m`, `xau_swap_1y`
- Vol: `gold_vol_index`, `vix`

## SUCCESS CRITERIA
Identify 1-2 interview-ready gold RV ideas with:
- Positive Sharpe (>0.5 net of costs) across multiple sub-periods
- Plausible economic story (carry / mean reversion / structural)
- Robust to lookback and threshold parameter changes
- Clean year-by-year breakdown showing the regime sensitivity

## WORKING PRINCIPLES
- Delta-neutral: directional AND notional neutrality
- Simplicity and explainability > headline Sharpe
- Robustness > optimization
- Research first, implementation second
- Every output answers: "Which gold RV structure is the best interview candidate?"

## GUARDRAILS
- No future leakage — rolling stats use only past data
- Remove weekends / non-trading days
- Test threshold sensitivity and holding period sensitivity
- Flag data quality issues clearly
- Report full-sample AND yearly breakdown

## FOLDER MAP
```
gold_rv_research/
├── 00_memory/          ← this file + guardrails
├── 01_data_raw/        ← raw CSVs
├── 02_data_clean/      ← trading-day cleaned data
├── 03_features_curve/  ← calendar spreads, flies, z-scores
├── 04_features_ratio/  ← gold-silver ratio features
├── 05_research_curve/  ← calendar + fly backtest outputs
├── 06_research_ratio/  ← gold-silver ratio backtest outputs
├── 07_results/         ← comparisons, rankings
└── 08_notes/           ← interview framing
```

## MODULE MAP
- `02_clean_data.py` — load, clean, export to 02_data_clean/
- `03_build_curve_features.py` — spreads, flies, z-scores → 03_features_curve/
- `04_build_ratio_features.py` — GC/SI ratio, hedge ratios, z-scores → 04_features_ratio/
- `05_backtest_curve.py` — calendar + fly backtest engine → 05_research_curve/
- `06_backtest_ratio.py` — ratio backtest engine → 06_research_ratio/
- `07_compare_results.py` — strategy ranking + sensitivity → 07_results/

## LAST UPDATED
2026-03-29
