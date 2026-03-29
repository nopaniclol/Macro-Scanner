# Data Audit Summary — data-2.csv

**Generated**: 2026-03-29

## Date Range
- Raw rows: 2,976
- Date range: 2018-01-01 → 2026-02-27
- Trading-day rows (post-clean): 2,129

## Weekend Filtering
- Weekend rows removed: 1 (GC-missing weekedays)
- Weekend rows: 846

## Gold Futures Continuity
- **gc_fut_front**: 2,975/2,976 valid (0.0% missing) | range 2018-01-02 → 2026-02-27 | price [1179.6, 5318.4] ✓
- **gc_fut_second**: 2,975/2,976 valid (0.0% missing) | range 2018-01-02 → 2026-02-27 | price [1184.0, 5354.8] ✓
- **gc_fut_third**: 2,975/2,976 valid (0.0% missing) | range 2018-01-02 → 2026-02-27 | price [1189.5, 5395.8] ✓

## Silver Futures Continuity
- **si_fut_front**: 2,975/2,976 valid (0.0% missing) | range 2018-01-02 → 2026-02-27 | price [11.772, 115.504] ⚠️ POSSIBLE_OUTLIERS(25)
- **si_fut_second**: 2,975/2,976 valid (0.0% missing) | range 2018-01-02 → 2026-02-27 | price [11.805, 116.388] ⚠️ POSSIBLE_OUTLIERS(25)
- **si_fut_third**: 2,975/2,976 valid (0.0% missing) | range 2018-01-02 → 2026-02-27 | price [11.83, 117.261] ⚠️ POSSIBLE_OUTLIERS(25)

## Spread Plausibility Checks
- **cal_12_gc**: min=-39.90 max=1.70 mean=-12.996 outside[-20,20]=764 ⚠️
- **cal_23_gc**: min=-41.00 max=1.60 mean=-12.616 outside[-20,20]=704 ⚠️
- **cal_12_si**: min=-0.89 max=0.05 mean=-0.194 outside[-5,5]=0 ✓
- **gs_ratio**: min=44.0 max=125.5 mean=82.8 outside[30,200]=0 ✓

## Year-by-Year Coverage (GC front)
| Year | Rows | Missing |
|------|------|---------|
| 2018 | 260 | 0 |
| 2019 | 261 | 0 |
| 2020 | 262 | 0 |
| 2021 | 261 | 0 |
| 2022 | 260 | 0 |
| 2023 | 260 | 0 |
| 2024 | 262 | 0 |
| 2025 | 261 | 0 |
| 2026 | 42 | 0 |

## Recommendations
- ✅ Use `gc_fut_front`, `gc_fut_second`, `gc_fut_third` for Track 1 (curve RV)
- ✅ Use `si_fut_front` for Track 2 (gold-silver ratio)
- ⚠️ Check silver third-leg continuity before using for 3-leg silver fly
- ✅ Gold-silver ratio range looks plausible (50–100 typical post-2018)
- ⚠️ Jan 2018 has some missing values on first row (holiday) — already filtered by GC completeness check

## First Research Pass Recommendation
**Start with Track 1 (Gold Calendar Spreads)** — all three GC legs are clean,
data goes back to early 2018, and the calendar spread story is the simplest to explain.
Track 2 (gold-silver ratio) can run in parallel once Track 1 is validated.