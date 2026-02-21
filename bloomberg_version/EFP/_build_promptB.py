#!/usr/bin/env python3
"""Append Sections 8-12 (Prompt B) to EFP_Beta_Analysis.ipynb."""
import json

def src(text):
    lines = text.strip("\n").split("\n")
    return [l + "\n" if i < len(lines)-1 else l for i, l in enumerate(lines)]

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": src(text)}

def code(text):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src(text)}

# Load existing notebook
nb_path = "/Users/lisa/Documents/Macro Scanner/Macro-Scanner/bloomberg_version/EFP/EFP_Beta_Analysis.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

existing_count = len(nb["cells"])
new_cells = []

# ═══════════════════════════════════════════════════════════════════
# Reload guard
# ═══════════════════════════════════════════════════════════════════
new_cells.append(md("""---
# Prompt B — Forward Rate Interpolation, OTC Forward, EFP Construction

> Sections 8–12: Interpolate the forward rate curve to each contract's
> exact delta\_T, construct the OTC forward price anchored to FND,
> compute EFP series (raw and adjusted), and build differenced series
> for regression in Prompt C."""))

new_cells.append(code("""# ── Reload guard: load from CSV if master_df not in memory ────────
try:
    _ = master_df.shape
    print(f"master_df already in memory: {master_df.shape}")
except NameError:
    import pandas as pd
    import numpy as np
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    from datetime import date, datetime
    import warnings
    warnings.filterwarnings("ignore")

    master_df = pd.read_csv('efp_master_data.csv', parse_dates=['date', 'fnd'])
    contracts_meta = pd.read_csv('efp_contracts_meta.csv',
                                  parse_dates=['fnd', 'fdd', 'ltd'])
    print(f"Loaded master_df from CSV: {master_df.shape}")
    print(f"Loaded contracts_meta from CSV: {contracts_meta.shape}")

    # Rebuild config if not present
    config = {
        'metals': ['gold', 'silver'],
        'start_date': '2023-01-01',
        'end_date': date.today().strftime('%Y-%m-%d'),
        'roll_days_before_fnd': 5,
        'regression_window_days': 60,
        'k_sigma': 3,
        'generic_depth': 4,
    }"""))

# ═══════════════════════════════════════════════════════════════════
# SECTION 8: Forward Rate Interpolation
# ═══════════════════════════════════════════════════════════════════
new_cells.append(md("""## Section 8 — Forward Rate Interpolation to Delta\_T

The forward rate curve has fixed tenors (1W, 1M, 2M, 3M, 6M, 12M).
Each futures contract has its own delta\_T (time to FND in years) that
generally falls **between** these tenors.

We interpolate the curve to each row's exact delta\_T using `scipy.interp1d`
with flat extrapolation at the edges.

Edge cases handled:
- delta\_T <= 0 → rate = 0 (at or past FND)
- < 2 non-NaN rate points → NaN (insufficient data)
- delta\_T < 1W → use 1W rate directly (no extrapolation below shortest tenor)"""))

new_cells.append(code("""def interpolate_fwd_rate(row, method='linear'):
    \"\"\"Interpolate the forward rate curve to this row's exact delta_T.

    Parameters
    ----------
    row : pd.Series
        A row from master_df with fwd_1W...fwd_12M and delta_T.
    method : str
        Interpolation kind passed to scipy.interpolate.interp1d.

    Returns
    -------
    float
        Interpolated forward rate as a decimal.
    \"\"\"
    delta_T = row.get('delta_T', np.nan)

    # Edge case: at or past FND
    if pd.isna(delta_T) or delta_T <= 0:
        return 0.0

    # Build tenor/rate arrays from the row's forward columns
    tenors_all = np.array([7/365, 1/12, 2/12, 3/12, 6/12, 12/12])
    rates_all  = np.array([
        row.get('fwd_1W', np.nan),
        row.get('fwd_1M', np.nan),
        row.get('fwd_2M', np.nan),
        row.get('fwd_3M', np.nan),
        row.get('fwd_6M', np.nan),
        row.get('fwd_12M', np.nan),
    ])

    # Drop NaN pairs
    valid = ~np.isnan(rates_all)
    tenors = tenors_all[valid]
    rates  = rates_all[valid]

    if len(rates) < 2:
        return np.nan

    # Edge case: delta_T shorter than 1W — use shortest available rate
    if delta_T < tenors[0]:
        return float(rates[0])

    # Interpolate
    interp_fn = interp1d(
        tenors, rates,
        kind=method,
        bounds_error=False,
        fill_value=(rates[0], rates[-1]),  # flat extrapolation at edges
    )

    return float(interp_fn(delta_T))


# ── Apply to master_df ───────────────────────────────────────────
master_df['fwd_rate_interp'] = master_df.apply(interpolate_fwd_rate, axis=1)

n_nan = master_df['fwd_rate_interp'].isna().sum()
n_total = len(master_df)
print(f"Forward rate interpolation complete:")
print(f"  {n_total - n_nan:,} rows interpolated, {n_nan} NaN ({n_nan/n_total*100:.1f}%)")

# ── Per-metal summary ────────────────────────────────────────────
print("\\n" + "=" * 70)
print("  INTERPOLATED FORWARD RATE SUMMARY")
print("=" * 70)
for metal in config['metals']:
    sub = master_df[master_df['metal'] == metal]['fwd_rate_interp'].dropna()
    if len(sub) == 0:
        continue
    print(f"\\n  {metal.upper()}:")
    print(f"    Mean  : {sub.mean()*100:+.4f}%")
    print(f"    Median: {sub.median()*100:+.4f}%")
    print(f"    Std   : {sub.std()*100:.4f}%")
    print(f"    Range : [{sub.min()*100:+.4f}%, {sub.max()*100:+.4f}%]")
    print(f"    NaN   : {master_df[master_df['metal']==metal]['fwd_rate_interp'].isna().sum()}")"""))

new_cells.append(md("""### Interpolation Sanity Check

Plot the interpolated rate overlaid with raw 1M and 3M rates.
The interpolated rate should track between them when delta\_T
is in the 1–3 month range."""))

new_cells.append(code("""fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for i, metal in enumerate(config['metals']):
    ax = axes[i]
    sub = master_df[master_df['metal'] == metal].copy()
    sub = sub.sort_values('date')

    ax.plot(sub['date'], sub['fwd_rate_interp'] * 100,
            linewidth=1.2, color='#2c3e50', label='Interpolated (at delta_T)')

    # Overlay raw tenors
    if 'fwd_1M' in sub.columns:
        ax.plot(sub['date'], sub['fwd_1M'] * 100,
                linewidth=0.8, color='#3498db', alpha=0.6, label='1M raw')
    if 'fwd_3M' in sub.columns:
        ax.plot(sub['date'], sub['fwd_3M'] * 100,
                linewidth=0.8, color='#e67e22', alpha=0.6, label='3M raw')

    ax.set_title(f'{metal.upper()} — Interpolated Forward Rate vs Raw Tenors',
                 fontsize=12)
    ax.set_ylabel('Rate (%)')
    ax.legend(loc='best', fontsize=9)
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Flag NaN rows
nan_rows = master_df[master_df['fwd_rate_interp'].isna()]
if len(nan_rows) > 0:
    print(f"\\nWARNING: {len(nan_rows)} rows with NaN interpolated rate:")
    print(nan_rows[['date', 'metal', 'delta_T', 'fwd_1M', 'fwd_3M']].head(10))
else:
    print("\\nAll rows have valid interpolated forward rates.")"""))

# ═══════════════════════════════════════════════════════════════════
# SECTION 9: OTC Forward Price
# ═══════════════════════════════════════════════════════════════════
new_cells.append(md("""## Section 9 — OTC Forward Price Anchored to FND

The OTC forward is the theoretical price at which a dealer would sell
gold/silver for delivery on FND, given today's spot and the interpolated
forward rate:

$$F_{OTC} = S \\times (1 + r_{fwd} \\times \\Delta T)$$

This uses **simple interest** (not continuous compounding), consistent
with precious metals market convention.

F\_OTC should be very close to the COMEX futures price in normal markets.
Any gap is the EFP."""))

new_cells.append(code("""def compute_otc_forward(spot, fwd_rate_dec, delta_T):
    \"\"\"Compute OTC forward price using simple interest.

    F_OTC = spot * (1 + fwd_rate * delta_T)

    Parameters
    ----------
    spot : float
        Current spot price.
    fwd_rate_dec : float
        Forward rate as a decimal (e.g. 0.045 for 4.5%).
    delta_T : float
        Time to FND in years.

    Returns
    -------
    float
        OTC forward price.
    \"\"\"
    if pd.isna(spot) or pd.isna(fwd_rate_dec) or pd.isna(delta_T):
        return np.nan
    if delta_T <= 0:
        return spot  # at FND, forward = spot
    return spot * (1.0 + fwd_rate_dec * delta_T)


# ── Apply ─────────────────────────────────────────────────────────
master_df['F_OTC'] = master_df.apply(
    lambda row: compute_otc_forward(
        row['spot'], row['fwd_rate_interp'], row['delta_T']
    ), axis=1
)

# ── Sanity check: F_OTC vs futures_price ─────────────────────────
print("=" * 70)
print("  OTC FORWARD vs COMEX FUTURES — SANITY CHECK")
print("=" * 70)

for metal in config['metals']:
    sub = master_df[master_df['metal'] == metal].dropna(
        subset=['futures_price', 'F_OTC'])
    if len(sub) == 0:
        continue

    diff = sub['futures_price'] - sub['F_OTC']
    print(f"\\n  {metal.upper()} (futures_price - F_OTC):")
    print(f"    Mean   : ${diff.mean():+.4f}")
    print(f"    Std    : ${diff.std():.4f}")
    print(f"    Median : ${diff.median():+.4f}")
    print(f"    Range  : [${diff.min():+.4f}, ${diff.max():+.4f}]")
    print(f"    Latest : ${diff.iloc[-1]:+.4f}  "
          f"(futures=${sub['futures_price'].iloc[-1]:,.2f}, "
          f"F_OTC=${sub['F_OTC'].iloc[-1]:,.2f})")"""))

new_cells.append(md("""### F\_OTC vs Futures Price — Visual Check"""))

new_cells.append(code("""fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for i, metal in enumerate(config['metals']):
    ax = axes[i]
    sub = master_df[master_df['metal'] == metal].sort_values('date')

    ax.plot(sub['date'], sub['futures_price'],
            linewidth=1.2, color='#2c3e50', label='COMEX Futures')
    ax.plot(sub['date'], sub['F_OTC'],
            linewidth=1.2, color='#e74c3c', linestyle='--', label='OTC Forward')

    ax.set_title(f'{metal.upper()} — COMEX Futures vs OTC Forward (anchored to FND)',
                 fontsize=12)
    ax.set_ylabel('Price ($/oz)')
    ax.legend(loc='best', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════
# SECTION 10: EFP Series Construction
# ═══════════════════════════════════════════════════════════════════
new_cells.append(md("""## Section 10 — EFP Series Construction

Six EFP measures computed from master\_df:

| Column | Definition | Interpretation |
|--------|-----------|----------------|
| `EFP_raw` | futures − spot | Classic basis; includes carry |
| `EFP_adj` | futures − F\_OTC | Adjusted EFP; strips carry, isolates COMEX vs OTC basis |
| `EFP_raw_pct` | (EFP\_raw / spot) × 100 | Basis as % of spot |
| `EFP_adj_pct` | (EFP\_adj / spot) × 100 | Adjusted basis as % |
| `EFP_raw_ann` | EFP\_raw / (spot × delta\_T) | Annualised raw basis |
| `EFP_adj_ann` | EFP\_adj / (spot × delta\_T) | Annualised adjusted basis |

`EFP_adj` close to zero = normal market. Spikes indicate physical
scarcity, tariff risk, or cross-market dislocation."""))

new_cells.append(code("""# ── EFP computations ──────────────────────────────────────────────
# 1. Raw EFP (classic basis)
master_df['EFP_raw'] = master_df['futures_price'] - master_df['spot']

# 2. Adjusted EFP (carry-stripped)
master_df['EFP_adj'] = master_df['futures_price'] - master_df['F_OTC']

# 3. Percentage versions
master_df['EFP_raw_pct'] = (master_df['EFP_raw'] / master_df['spot']) * 100
master_df['EFP_adj_pct'] = (master_df['EFP_adj'] / master_df['spot']) * 100

# 4. Annualised versions (guard against tiny delta_T)
min_delta_T_for_ann = 0.01  # ~3.65 days

master_df['EFP_adj_ann'] = np.where(
    master_df['delta_T'] >= min_delta_T_for_ann,
    master_df['EFP_adj'] / (master_df['spot'] * master_df['delta_T']),
    np.nan
)

master_df['EFP_raw_ann'] = np.where(
    master_df['delta_T'] >= min_delta_T_for_ann,
    master_df['EFP_raw'] / (master_df['spot'] * master_df['delta_T']),
    np.nan
)

# ── Descriptive statistics ───────────────────────────────────────
print("=" * 75)
print("  EFP DESCRIPTIVE STATISTICS")
print("=" * 75)

for metal in config['metals']:
    sub = master_df[master_df['metal'] == metal]
    print(f"\\n  {metal.upper()}:")

    for col, unit in [('EFP_raw', '$/oz'), ('EFP_adj', '$/oz'),
                      ('EFP_raw_pct', '%'), ('EFP_adj_pct', '%'),
                      ('EFP_adj_ann', 'ann dec')]:
        s = sub[col].dropna()
        if len(s) == 0:
            continue
        print(f"\\n    {col} ({unit}):")
        print(f"      Mean   : {s.mean():+.4f}")
        print(f"      Std    : {s.std():.4f}")
        print(f"      Median : {s.median():+.4f}")
        print(f"      Min    : {s.min():+.4f}")
        print(f"      Max    : {s.max():+.4f}")
        if 'adj' in col.lower() and 'ann' not in col.lower():
            pct_pos = (s > 0).mean() * 100
            print(f"      %% > 0  : {pct_pos:.1f}%")

# ── Data quality flag: exclude_from_regression ───────────────────
# Rolling stats for outlier detection
for metal in config['metals']:
    mask = master_df['metal'] == metal
    roll_mean = master_df.loc[mask, 'EFP_adj'].rolling(60, min_periods=20).mean()
    roll_std  = master_df.loc[mask, 'EFP_adj'].rolling(60, min_periods=20).std()
    master_df.loc[mask, '_efp_adj_zscore'] = (
        (master_df.loc[mask, 'EFP_adj'] - roll_mean) / roll_std.replace(0, np.nan)
    ).abs()

master_df['exclude_from_regression'] = (
    (master_df['is_roll_date'] == True) |
    (master_df['delta_T'] < 5/365) |
    (master_df['fwd_rate_interp'].isna()) |
    (master_df['_efp_adj_zscore'] > 10)
)

# Drop temp column
master_df = master_df.drop(columns=['_efp_adj_zscore'], errors='ignore')

n_excluded = master_df['exclude_from_regression'].sum()
print(f"\\n  Exclusion flag summary:")
print(f"    Total rows           : {len(master_df):,}")
print(f"    Excluded             : {n_excluded:,} ({n_excluded/len(master_df)*100:.1f}%)")
print(f"    Available for regr.  : {len(master_df) - n_excluded:,}")
print(f"    Breakdown:")
print(f"      Roll dates         : {master_df['is_roll_date'].sum()}")
print(f"      Near expiry (<5d)  : {(master_df['delta_T'] < 5/365).sum()}")
print(f"      NaN fwd rate       : {master_df['fwd_rate_interp'].isna().sum()}")"""))

# ═══════════════════════════════════════════════════════════════════
# SECTION 11: Daily Differenced Series
# ═══════════════════════════════════════════════════════════════════
new_cells.append(md("""## Section 11 — Daily Differenced Series for Regression

Compute first-differences for the regression target and regressors:
- `delta_spot` = daily change in spot
- `delta_efp_raw` = daily change in raw EFP (basis)
- `delta_efp_adj` = daily change in adjusted EFP

Differences are set to NaN on roll dates and exclusion-flagged dates
to prevent spurious jumps from contaminating the regression."""))

new_cells.append(code("""# ── Compute daily differences per metal ───────────────────────────
for metal in config['metals']:
    mask = master_df['metal'] == metal
    idx = master_df.loc[mask].index

    master_df.loc[idx, 'delta_spot']    = master_df.loc[idx, 'spot'].diff()
    master_df.loc[idx, 'delta_efp_raw'] = master_df.loc[idx, 'EFP_raw'].diff()
    master_df.loc[idx, 'delta_efp_adj'] = master_df.loc[idx, 'EFP_adj'].diff()

# ── Null out differences on excluded dates ───────────────────────
exclude_mask = master_df['exclude_from_regression']
master_df.loc[exclude_mask, ['delta_spot', 'delta_efp_raw', 'delta_efp_adj']] = np.nan

# Also null out the day AFTER a roll (the diff would span the roll)
for metal in config['metals']:
    mask = master_df['metal'] == metal
    roll_idx = master_df.loc[mask & master_df['is_roll_date']].index
    # The next business day after each roll also has a tainted diff
    for ri in roll_idx:
        pos = master_df.index.get_loc(ri)
        if pos + 1 < len(master_df) and master_df.iloc[pos + 1]['metal'] == metal:
            next_idx = master_df.index[pos + 1]
            master_df.loc[next_idx, ['delta_spot', 'delta_efp_raw', 'delta_efp_adj']] = np.nan

# ── Summary ──────────────────────────────────────────────────────
print("=" * 70)
print("  DAILY DIFFERENCED SERIES SUMMARY")
print("=" * 70)

for metal in config['metals']:
    sub = master_df[master_df['metal'] == metal]
    print(f"\\n  {metal.upper()}:")
    for col in ['delta_spot', 'delta_efp_raw', 'delta_efp_adj']:
        s = sub[col].dropna()
        print(f"    {col:18s}  n={len(s):>5d}  mean={s.mean():+.4f}  "
              f"std={s.std():.4f}  range=[{s.min():+.4f}, {s.max():+.4f}]")

    # Correlation check
    clean = sub[['delta_spot', 'delta_efp_raw', 'delta_efp_adj']].dropna()
    if len(clean) > 20:
        corr_raw = clean['delta_spot'].corr(clean['delta_efp_raw'])
        corr_adj = clean['delta_spot'].corr(clean['delta_efp_adj'])
        print(f"    corr(delta_spot, delta_efp_raw) = {corr_raw:.4f}")
        print(f"    corr(delta_spot, delta_efp_adj) = {corr_adj:.4f}")"""))

# ═══════════════════════════════════════════════════════════════════
# SECTION 12: EFP Time Series Visualisations
# ═══════════════════════════════════════════════════════════════════
new_cells.append(md("""## Section 12 — EFP Time Series Visualisations

Four-panel view:
1. Gold: EFP\_raw and EFP\_adj ($/oz)
2. Silver: EFP\_raw and EFP\_adj ($/oz)
3. Gold: EFP\_adj\_ann (annualised adjusted EFP)
4. Silver: EFP\_adj\_ann

Roll dates shaded in light grey. Major EFP events annotated."""))

new_cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for col_idx, metal in enumerate(config['metals']):
    sub = master_df[master_df['metal'] == metal].sort_values('date').copy()

    # ── Top row: EFP_raw and EFP_adj ─────────────────────────────
    ax = axes[0, col_idx]
    ax.plot(sub['date'], sub['EFP_raw'],
            linewidth=0.9, color='#7f8c8d', alpha=0.7, label='EFP_raw (F-S)')
    ax.plot(sub['date'], sub['EFP_adj'],
            linewidth=1.2, color='#2c3e50', label='EFP_adj (F-F_OTC)')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

    # Shade roll dates
    roll_dates = sub[sub['is_roll_date']]['date']
    for rd in roll_dates:
        ax.axvline(x=rd, color='lightgrey', linewidth=0.3, alpha=0.5)

    # Annotate tariff period (Jan-Feb 2025) if in range
    tariff_date = pd.Timestamp('2025-01-15')
    if sub['date'].min() <= tariff_date <= sub['date'].max():
        ax.axvline(x=tariff_date, color='#e74c3c', linestyle='--',
                   linewidth=1, alpha=0.7)
        y_pos = sub['EFP_adj'].max() * 0.85
        ax.annotate('Tariff risk\\nJan 2025', xy=(tariff_date, y_pos),
                    fontsize=8, color='#e74c3c', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#e74c3c', alpha=0.8))

    unit = '$/oz'
    ax.set_title(f'{metal.upper()} — EFP Raw vs Adjusted ({unit})', fontsize=11)
    ax.set_ylabel(f'EFP ({unit})')
    ax.legend(loc='best', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.sca(ax)
    plt.xticks(rotation=45, fontsize=8)

    # ── Bottom row: EFP_adj_ann ──────────────────────────────────
    ax2 = axes[1, col_idx]
    ann = sub['EFP_adj_ann'].dropna()
    ax2.plot(sub['date'], sub['EFP_adj_ann'] * 100,
             linewidth=1.0, color='#8e44ad')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

    for rd in roll_dates:
        ax2.axvline(x=rd, color='lightgrey', linewidth=0.3, alpha=0.5)

    if sub['date'].min() <= tariff_date <= sub['date'].max():
        ax2.axvline(x=tariff_date, color='#e74c3c', linestyle='--',
                    linewidth=1, alpha=0.7)

    ax2.set_title(f'{metal.upper()} — Annualised Adjusted EFP (%)', fontsize=11)
    ax2.set_ylabel('EFP_adj_ann (%)')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.sca(ax2)
    plt.xticks(rotation=45, fontsize=8)

plt.tight_layout()
plt.show()"""))

new_cells.append(md("""### EFP Summary Table"""))

new_cells.append(code("""# ── Summary table ─────────────────────────────────────────────────
print("=" * 80)
print("  EFP SUMMARY TABLE")
print("=" * 80)
print(f"\\n  {'Metal':8s}  {'Mean Raw':>10s}  {'Mean Adj':>10s}  {'Std Adj':>10s}  "
      f"{'Max Adj':>10s}  {'Min Adj':>10s}  {'%>0 Adj':>8s}")
print(f"  {'-'*72}")

for metal in config['metals']:
    sub = master_df[master_df['metal'] == metal]
    raw = sub['EFP_raw'].dropna()
    adj = sub['EFP_adj'].dropna()
    pct_pos = (adj > 0).mean() * 100 if len(adj) > 0 else 0

    print(f"  {metal.upper():8s}  "
          f"${raw.mean():>+8.2f}  "
          f"${adj.mean():>+8.4f}  "
          f"${adj.std():>8.4f}  "
          f"${adj.max():>+8.4f}  "
          f"${adj.min():>+8.4f}  "
          f"{pct_pos:>6.1f}%")

# ── Latest values ────────────────────────────────────────────────
print(f"\\n  Latest values:")
for metal in config['metals']:
    sub = master_df[master_df['metal'] == metal].sort_values('date')
    if len(sub) == 0:
        continue
    latest = sub.iloc[-1]
    print(f"\\n  {metal.upper()} ({latest['date']:%Y-%m-%d}):")
    print(f"    Spot           : ${latest['spot']:>10,.2f}")
    print(f"    COMEX Futures  : ${latest['futures_price']:>10,.2f}")
    print(f"    OTC Forward    : ${latest['F_OTC']:>10,.2f}")
    print(f"    EFP_raw        : ${latest['EFP_raw']:>+10.2f}")
    print(f"    EFP_adj        : ${latest['EFP_adj']:>+10.4f}")
    if pd.notna(latest.get('EFP_adj_ann')):
        print(f"    EFP_adj_ann    : {latest['EFP_adj_ann']*100:>+10.4f}%")"""))

# ── Save updated master_df ──
new_cells.append(md("""### Export Updated Master DataFrame"""))

new_cells.append(code("""# ── Save master_df with EFP columns ──────────────────────────────
output_path = 'efp_with_spreads.csv'
master_df.to_csv(output_path, index=False)

print(f"Saved to: {output_path}")
print(f"  {master_df.shape[0]:,} rows x {master_df.shape[1]} columns")
print(f"\\nColumns: {list(master_df.columns)}")

# EFP columns added in Prompt B
efp_cols = ['fwd_rate_interp', 'F_OTC', 'EFP_raw', 'EFP_adj',
            'EFP_raw_pct', 'EFP_adj_pct', 'EFP_raw_ann', 'EFP_adj_ann',
            'exclude_from_regression', 'delta_spot', 'delta_efp_raw', 'delta_efp_adj']
print(f"\\nNew columns from Prompt B ({len(efp_cols)}):")
for col in efp_cols:
    n_valid = master_df[col].notna().sum() if col in master_df.columns else 0
    print(f"  {col:28s}: {n_valid:>6,} valid values")

print(f"\\nPrompt B complete: {datetime.now():%Y-%m-%d %H:%M}")
print("master_df is ready for regression in Prompt C.")"""))

# ═══════════════════════════════════════════════════════════════════
# Append to notebook and save
# ═══════════════════════════════════════════════════════════════════
nb["cells"].extend(new_cells)

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

# Verify
with open(nb_path) as f:
    nb2 = json.load(f)

n_cells = len(nb2["cells"])
n_md = sum(1 for c in nb2["cells"] if c["cell_type"] == "markdown")
n_code = sum(1 for c in nb2["cells"] if c["cell_type"] == "code")
n_new = n_cells - existing_count

print(f"Updated: {nb_path}")
print(f"Total cells: {n_cells} (markdown: {n_md}, code: {n_code})")
print(f"Existing (Prompt A): {existing_count}, New (Prompt B): {n_new}")

all_code_text = "".join("".join(c["source"]) for c in nb2["cells"] if c["cell_type"] == "code")
all_text = "".join("".join(c["source"]) for c in nb2["cells"])

checks = {
    "All source is list": all(isinstance(c["source"], list) for c in nb2["cells"]),
    "Has interpolate_fwd_rate": "interpolate_fwd_rate" in all_code_text,
    "Has compute_otc_forward": "compute_otc_forward" in all_code_text,
    "Has F_OTC": "F_OTC" in all_code_text,
    "Has EFP_raw": "EFP_raw" in all_code_text,
    "Has EFP_adj": "EFP_adj" in all_code_text,
    "Has EFP_adj_ann": "EFP_adj_ann" in all_code_text,
    "Has EFP_raw_ann": "EFP_raw_ann" in all_code_text,
    "Has EFP_raw_pct": "EFP_raw_pct" in all_code_text,
    "Has EFP_adj_pct": "EFP_adj_pct" in all_code_text,
    "Has exclude_from_regression": "exclude_from_regression" in all_code_text,
    "Has delta_spot": "delta_spot" in all_code_text,
    "Has delta_efp_raw": "delta_efp_raw" in all_code_text,
    "Has delta_efp_adj": "delta_efp_adj" in all_code_text,
    "Has scipy interp1d": "interp1d" in all_code_text,
    "Has simple interest formula": "1.0 + fwd_rate_dec * delta_T" in all_code_text,
    "Has bounds_error=False": "bounds_error=False" in all_code_text,
    "Has fill_value flat extrap": "fill_value=" in all_code_text,
    "Has efp_with_spreads.csv": "efp_with_spreads.csv" in all_code_text,
    "Has reload guard": "efp_master_data.csv" in all_code_text,
    "Has 4-panel chart": "2, 2" in all_code_text,
    "Has tariff annotation": "Tariff" in all_code_text or "tariff" in all_code_text,
    "Has roll date shading": "roll_dates" in all_code_text,
    "Has annualised guard (min_delta_T)": "min_delta_T_for_ann" in all_code_text,
    "Has 10 std dev outlier flag": "_efp_adj_zscore" in all_code_text,
    "Has post-roll NaN nulling": "next_idx" in all_code_text,
    "All code cells have outputs": all("outputs" in c for c in nb2["cells"] if c["cell_type"] == "code"),
}

for name, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'} | {name}")

print(f"\nAll {len(checks)} checks passed: {all(checks.values())}")

# Cell listing for new cells only
print(f"\nNew cells (Prompt B):")
for i in range(existing_count, n_cells):
    c = nb2["cells"][i]
    first = "".join(c["source"]).split("\n")[0][:85]
    print(f"  [{i:2d}] {c['cell_type']:8s} | {first}")
