# BQuant install — what to paste, and in what order

Three changes. The first two are one-liners you must not skip; the third is a new cell.

---

## 1. Log the EFP spread  (Cell 8, ~line 1317)

In `fetch_all_data`, inside the `history_rows.append({...})` block, add:

```python
    'Spot Bid': df.loc[contract, 'Spot Bid'],
    'Spot Ask': df.loc[contract, 'Spot Ask'],
    'Fut Bid':  df.loc[contract, 'Fut Bid'],
    'Fut Ask':  df.loc[contract, 'Fut Ask'],
    'EFP Bid':  df.loc[contract, 'EFP Bid'],
    'EFP Ask':  df.loc[contract, 'EFP Ask'],
    'EFP Spread': df.loc[contract, 'EFP Spread'],
```

You already compute all seven and throw them away on every refresh. The spread is
the dealing cost every backtest figure is struck gross of — **50bp costs 14% of
P&L** — and it is what settles the capacity question. Nothing else in this file
matters as much per line of code.

*(Already applied in `efp_dashboard.py`. If you are running your original
notebook, add it by hand.)*

## 2. Fix the forward-curve defaults  (Cell 3)

`DEFAULT_METAL_FORWARD_CURVES` has gold 3M at **1.10%** when the market implies
**~4.38%**. On a 180-day contract that manufactures **$69.62/oz of phantom
richness** against a sub-$1 bound — a permanent SELL EFP on every contract.

Harmless while `pm_forward_curves_latest.csv` exists. A loaded gun on a fresh
environment. Replace with SOFR minus a realistic lease:

```python
DEFAULT_METAL_FORWARD_CURVES = {
    'XAU': {'1M': 4.30, '2M': 4.28, '3M': 4.25, '6M': 4.18, '9M': 4.10, '12M': 4.05},
    'XAG': {'1M': 3.90, '2M': 3.88, '3M': 3.85, '6M': 3.75, '9M': 3.68, '12M': 3.60},
    'XPT': {'1M': 2.80, '2M': 2.85, '3M': 2.90, '6M': 3.00, '9M': 3.10, '12M': 3.20},
    'XPD': {'1M': 3.20, '2M': 3.25, '3M': 3.30, '6M': 3.40, '9M': 3.48, '12M': 3.55},
}
```

These are placeholders in the right postcode, not quotes. **Overwrite them with
your own curve on first run** — the point is that a wrong default now fails
loudly rather than silently.

## 3. Paste `CELL_12_golive.py` as a new cell after Cell 11

Then, once:

```python
self_test()          # 9 offline checks, no Bloomberg call
```

And every day after `fetch_all_data()`:

```python
daily_checks()
```

---

## What Cell 12 does

| function | purpose |
|---|---|
| `capture_intraday_snap()` | 13:30 NY spot → `pm_intraday_snap.csv`. **Cannot be backfilled.** |
| `snap_vs_close()` | measures the clock offset once you have data |
| `log_reconciliation()` | live 2-way residual vs settle-based → `pm_reconciliation.csv` |
| `check_forward_defaults()` | flags entered-vs-EFP-implied forward gaps > 1.5pp |
| `regime_high(metal)` | causal lease-vol classifier |
| `size_multiplier(bx, hi)` | 0.5 / 1.0 / 1.5 / 2.0 by bound multiple, high regime only |
| `daily_checks()` | runs all of it and prints one report |
| `self_test()` | 9 offline assertions |

## What to expect on day one

**Every metal will read `burn-in: N/250 sessions` and size at 1.0×.** That is
correct. The classifier needs 250 sessions of lease history before it fires, and
until then the book behaves exactly as it does today. It will not guess.

The lease history rebuilds itself from `efp_history.csv` as `SOFR − forward`,
verified to round-trip to 7e-15 against the true lease series, so it accumulates
automatically once you are refreshing daily.

**Reconciliation needs 20 sessions** before it means anything. Target is
agreement within $0.50/oz on 95%+ of observations. Until that passes, the
backtest calibration does not transfer to your live residual.

## Validation behind this

- `size_multiplier` matches the backtest engine on every input tested
- lease reconstruction round-trips to **7.1e-15**
- regime flags match the engine metal by metal
- shocking today's lease **cannot** move an earlier flag
- in the engine, the same classifier passes a **10,644-flag** look-ahead audit
  and the ported bound logic reproduces **4,364/4,364** signals

Walking the classifier forward over history, it reads normal through 2022–23,
flags palladium from mid-2024, everything through 2025, and returns to normal by
August 2026 as lease vol collapsed — which is why the book is on equal weight
today.
