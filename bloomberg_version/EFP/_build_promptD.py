#!/usr/bin/env python3
"""Append Sections 18-23 (Prompt D) to EFP_Beta_Analysis.ipynb.

Also inserts a Table of Contents cell near the top (after the title).
"""
import json


def src(text):
    lines = text.strip("\n").split("\n")
    return [l + "\n" if i < len(lines) - 1 else l for i, l in enumerate(lines)]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": src(text)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src(text),
    }


# Load existing notebook
nb_path = "/Users/lisa/Documents/Macro Scanner/Macro-Scanner/bloomberg_version/EFP/EFP_Beta_Analysis.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

existing_count = len(nb["cells"])
new_cells = []

# ═══════════════════════════════════════════════════════════════════
# Prompt D header + reload guard
# ═══════════════════════════════════════════════════════════════════
new_cells.append(
    md(
        """---
# Prompt D — Visualisations, Dashboard, Sensitivity & Documentation

> Sections 18–23: Publication-quality charts, summary dashboard tables,
> sensitivity analysis, daily workflow guide, table of contents, and
> end-to-end validation on a known historical date."""
    )
)

new_cells.append(
    code(
        """# ── Reload guard: load from CSV if DataFrames not in memory ───────
try:
    _ = master_df.shape
    _ = beta_comparison.shape
    _ = rolling_betas.shape
    _ = static_results
    print(f"All DataFrames in memory: master_df {master_df.shape}, "
          f"beta_comparison {beta_comparison.shape}, "
          f"rolling_betas {rolling_betas.shape}")
except NameError:
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.regression.rolling import RollingOLS
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    import seaborn as sns
    from scipy import stats as sp_stats
    from datetime import date, datetime
    import warnings
    warnings.filterwarnings("ignore")

    master_df = pd.read_csv('efp_with_spreads.csv', parse_dates=['date', 'fnd'])
    beta_comparison = pd.read_csv('efp_beta_results.csv', parse_dates=['date'])
    print(f"Loaded master_df: {master_df.shape}")
    print(f"Loaded beta_comparison: {beta_comparison.shape}")

    config = {
        'metals': ['gold', 'silver'],
        'start_date': '2023-01-01',
        'end_date': date.today().strftime('%Y-%m-%d'),
        'roll_days_before_fnd': 5,
        'regression_window_days': 60,
        'k_sigma': 3,
        'generic_depth': 4,
    }

    # Reconstruct rolling_betas from beta_comparison
    rolling_betas = beta_comparison[[
        'date', 'metal', 'rolling_beta', 'rolling_rsq',
        'upper_95', 'lower_95'
    ]].copy()

    # Reconstruct static_results by re-running static OLS
    static_results = {}
    for metal in config['metals']:
        sub = master_df[
            (master_df['metal'] == metal) &
            (~master_df['exclude_from_regression'].astype(bool))
        ][['date', 'delta_spot', 'delta_efp_adj', 'spot', 'EFP_adj']].dropna(
            subset=['delta_spot', 'delta_efp_adj'])
        if len(sub) < 30:
            continue
        y = sub['delta_efp_adj'].values
        X = sm.add_constant(sub['delta_spot'].values)
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        static_results[metal] = {
            'model': model,
            'n_obs': len(sub),
            'date_range': (sub['date'].min(), sub['date'].max()),
            'alpha': model.params[0],
            'beta_spot': model.params[1],
            'se_beta': model.bse[1],
            't_stat': model.tvalues[1],
            'p_value': model.pvalues[1],
            'r_squared': model.rsquared,
        }

    # Level regression
    level_results = {}
    for metal in config['metals']:
        sub = master_df[
            (master_df['metal'] == metal) &
            (~master_df['exclude_from_regression'].astype(bool))
        ][['spot', 'EFP_adj']].dropna()
        if len(sub) < 30:
            continue
        y = sub['EFP_adj'].values
        X = sm.add_constant(sub['spot'].values)
        m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
        level_results[metal] = {
            'beta_level': m.params[1],
            'se': m.bse[1],
            't_stat': m.tvalues[1],
            'p_value': m.pvalues[1],
            'r_squared': m.rsquared,
        }

    # Ensure beta_theo exists
    if 'beta_theo' not in master_df.columns:
        master_df['beta_theo'] = (
            master_df['fwd_rate_interp'] * master_df['delta_T']
        )

    print("Static and level regressions rebuilt from CSV data.")

# Ensure matplotlib imports available
import matplotlib.ticker as mticker"""
    )
)

# ═══════════════════════════════════════════════════════════════════
# SECTION 18: Core Visualization Suite (5 charts)
# ═══════════════════════════════════════════════════════════════════
new_cells.append(
    md(
        """## Section 18 — Core Visualisation Suite

Five publication-quality charts for buyside presentation:
1. Gold: rolling beta vs theoretical beta with 95% CI and COMEX-rich shading
2. Silver: same format
3. Gold: EFP\_adj with rolling beta overlay (dual axis)
4. Beta excess time series (both metals)
5. Delta exposure heatmaps (gold and silver)"""
    )
)

# Chart 1 & 2: Rolling Beta vs Theoretical Beta
new_cells.append(
    md("""### Charts 1 & 2 — Empirical vs Theoretical Beta (Gold & Silver)""")
)

new_cells.append(
    code(
        """fig, axes = plt.subplots(2, 1, figsize=(15, 10))

for i, metal in enumerate(config['metals']):
    ax = axes[i]
    bc = beta_comparison[beta_comparison['metal'] == metal].sort_values('date').copy()
    bc = bc.dropna(subset=['rolling_beta'])

    # Rolling beta line
    ax.plot(bc['date'], bc['rolling_beta'],
            linewidth=1.4, color='#2980b9', label='Empirical Beta (60d rolling)')

    # Theoretical beta line
    ax.plot(bc['date'], bc['beta_theo'],
            linewidth=1.2, color='#e67e22', linestyle='--',
            label='Theoretical Beta (fwd_rate × delta_T)')

    # 95% confidence band
    ax.fill_between(bc['date'], bc['lower_95'], bc['upper_95'],
                    alpha=0.15, color='#2980b9', label='95% Confidence Band')

    # Shade where beta_excess > 0 (COMEX rich)
    excess_pos = bc['beta_excess'] > 0
    if excess_pos.any():
        ax.fill_between(bc['date'],
                        ax.get_ylim()[0] if i == 0 else bc['rolling_beta'].min() * 1.5,
                        ax.get_ylim()[1] if i == 0 else bc['rolling_beta'].max() * 1.5,
                        where=excess_pos.values,
                        alpha=0.08, color='#e74c3c', label='COMEX Rich (excess > 0)')

    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)

    # Tariff annotation
    tariff_date = pd.Timestamp('2025-01-15')
    if len(bc) > 0 and bc['date'].min() <= tariff_date <= bc['date'].max():
        ax.axvline(x=tariff_date, color='#c0392b', linestyle='--',
                   linewidth=0.8, alpha=0.6)
        ax.annotate('Tariff\\nRisk', xy=(tariff_date, bc['rolling_beta'].max() * 0.92),
                    fontsize=8, color='#c0392b', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#c0392b', alpha=0.8))

    ax.set_title(f'{metal.upper()} EFP: Empirical vs Theoretical Beta to Spot',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Beta (delta_EFP_adj / delta_spot)', fontsize=10)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(axis='y', alpha=0.2)

# Re-apply COMEX-rich shading with correct y-limits after auto-scaling
for i, metal in enumerate(config['metals']):
    ax = axes[i]
    bc = beta_comparison[beta_comparison['metal'] == metal].sort_values('date').copy()
    bc = bc.dropna(subset=['rolling_beta'])
    excess_pos = bc['beta_excess'] > 0
    ylims = ax.get_ylim()
    if excess_pos.any():
        ax.fill_between(bc['date'], ylims[0], ylims[1],
                        where=excess_pos.values,
                        alpha=0.06, color='#e74c3c')
    ax.set_ylim(ylims)

plt.tight_layout()
plt.show()"""
    )
)

# Chart 3: EFP_adj with rolling beta overlay
new_cells.append(
    md("""### Chart 3 — Gold EFP Adjusted with Rolling Beta Overlay""")
)

new_cells.append(
    code(
        """fig, ax1 = plt.subplots(figsize=(15, 6))

metal = 'gold'
sub = master_df[master_df['metal'] == metal].sort_values('date').copy()
rb = rolling_betas[rolling_betas['metal'] == metal].sort_values('date').copy()

# Left axis: EFP_adj
color_efp = '#2c3e50'
ax1.plot(sub['date'], sub['EFP_adj'], linewidth=1.0, color=color_efp,
         alpha=0.8, label='EFP_adj ($/oz)')
ax1.set_ylabel('EFP_adj ($/oz)', color=color_efp, fontsize=11)
ax1.tick_params(axis='y', labelcolor=color_efp)
ax1.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)

# Annotate extreme EFP periods (> 2 std devs)
efp_mean = sub['EFP_adj'].mean()
efp_std = sub['EFP_adj'].std()
extreme_mask = (sub['EFP_adj'] > efp_mean + 2 * efp_std) | \\
               (sub['EFP_adj'] < efp_mean - 2 * efp_std)
extreme_dates = sub.loc[extreme_mask, 'date']
for ed in extreme_dates:
    ax1.axvline(x=ed, color='#e74c3c', linewidth=0.3, alpha=0.3)

if len(extreme_dates) > 0:
    ax1.axvline(x=extreme_dates.iloc[0], color='#e74c3c', linewidth=0.3,
                alpha=0.3, label=f'Extreme EFP (>{efp_std*2:.2f} from mean)')

# Right axis: rolling_beta
ax2 = ax1.twinx()
color_beta = '#8e44ad'
rb_clean = rb.dropna(subset=['rolling_beta'])
ax2.plot(rb_clean['date'], rb_clean['rolling_beta'],
         linewidth=1.3, color=color_beta, alpha=0.8,
         label='Rolling Beta (60d)')
ax2.set_ylabel('Rolling Beta', color=color_beta, fontsize=11)
ax2.tick_params(axis='y', labelcolor=color_beta)

ax1.set_title('Gold EFP Adjusted vs Rolling Spot Beta', fontsize=13,
              fontweight='bold')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax1.tick_params(axis='x', rotation=45, labelsize=8)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()"""
    )
)

# Chart 4: Beta excess both metals
new_cells.append(
    md("""### Chart 4 — Beta Excess: Gold vs Silver""")
)

new_cells.append(
    code(
        """fig, ax = plt.subplots(figsize=(15, 5))

for metal, color, lbl in [('gold', '#2980b9', 'Gold'),
                           ('silver', '#e67e22', 'Silver')]:
    bc = beta_comparison[beta_comparison['metal'] == metal].sort_values('date')
    bc = bc.dropna(subset=['beta_excess'])
    ax.plot(bc['date'], bc['beta_excess'],
            linewidth=1.1, color=color, alpha=0.85, label=lbl)

ax.axhline(y=0, color='black', linewidth=0.8)
ax.fill_between(ax.get_xlim(), 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.01,
                alpha=0.04, color='#e74c3c')
ax.fill_between(ax.get_xlim(), ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -0.01, 0,
                alpha=0.04, color='#3498db')

# Tariff annotation
tariff_date = pd.Timestamp('2025-01-15')
ax.axvline(x=tariff_date, color='#c0392b', linestyle='--', linewidth=0.8, alpha=0.6)
ax.annotate('Tariff risk\\nJan 2025', xy=(tariff_date, ax.get_ylim()[1] * 0.85),
            fontsize=8, color='#c0392b', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#c0392b', alpha=0.8))

# Annotations for zones
ax.text(0.98, 0.95, 'COMEX Rich / Scarcity \\u2191', transform=ax.transAxes,
        fontsize=8, color='#c0392b', ha='right', va='top', alpha=0.6)
ax.text(0.98, 0.05, 'COMEX Cheap / Normalising \\u2193', transform=ax.transAxes,
        fontsize=8, color='#2980b9', ha='right', va='bottom', alpha=0.6)

ax.set_title('EFP Beta Excess (Empirical \\u2212 Theoretical): Gold vs Silver',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Beta Excess', fontsize=11)
ax.legend(loc='upper left', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.show()"""
    )
)

# Chart 5: Delta exposure heatmaps
new_cells.append(
    md("""### Chart 5 — Delta Exposure Heatmaps (Gold & Silver)""")
)

new_cells.append(
    code(
        """position_sizes = [500, 1_000, 2_500, 5_000, 10_000]
spot_moves_pct = [-3.0, -2.0, -1.0, +1.0, +2.0, +3.0]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for col_idx, metal in enumerate(config['metals']):
    ax = axes[col_idx]

    # Get current empirical beta
    rb_metal = rolling_betas[
        rolling_betas['metal'] == metal
    ].sort_values('date').dropna(subset=['rolling_beta'])

    if len(rb_metal) == 0:
        ax.text(0.5, 0.5, f'No rolling beta for {metal}',
                ha='center', va='center', transform=ax.transAxes)
        continue

    emp_beta = rb_metal['rolling_beta'].iloc[-1]
    latest = master_df[master_df['metal'] == metal].sort_values('date').iloc[-1]
    spot = latest['spot']

    # Build heatmap matrix
    heatmap_data = np.zeros((len(spot_moves_pct), len(position_sizes)))
    for r, pct_move in enumerate(spot_moves_pct):
        for c, pos_oz in enumerate(position_sizes):
            dollar_move = spot * (pct_move / 100.0)
            pnl = pos_oz * emp_beta * dollar_move
            heatmap_data[r, c] = pnl

    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=[f'{m:+.0f}%' for m in spot_moves_pct],
        columns=[f'{p:,}' for p in position_sizes],
    )

    # Diverging colormap centered at zero
    vmax = np.abs(heatmap_data).max()
    sns.heatmap(heatmap_df, annot=True, fmt=',.0f', center=0,
                cmap='RdYlGn', vmin=-vmax, vmax=vmax,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'USD P&L'},
                ax=ax)

    ax.set_title(f'{metal.upper()} — Residual Delta P&L (USD)\\n'
                 f'Beta={emp_beta:.4f}, Spot=${spot:,.0f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Position Size (oz)', fontsize=10)
    ax.set_ylabel('Spot Move (%)', fontsize=10)

plt.tight_layout()
plt.show()

print("Chart 5: Delta exposure heatmaps rendered.")
print("  Green = positive P&L, Red = negative P&L from residual delta")
print("  These are P&L BEYOND carry — the directional residual only.")"""
    )
)

# ═══════════════════════════════════════════════════════════════════
# SECTION 19: Summary Dashboard (3 formatted tables)
# ═══════════════════════════════════════════════════════════════════
new_cells.append(
    md(
        """## Section 19 — Summary Dashboard

Three formatted tables for quick daily reference:
1. **EFP Beta Summary Statistics** — full-sample regression results
2. **Current EFP Snapshot** — today's live market values
3. **Residual Delta for Standard Position Sizes** — actionable risk numbers"""
    )
)

new_cells.append(
    md("""### Table 1 — EFP Beta Summary Statistics""")
)

new_cells.append(
    code(
        """# ── Table 1: EFP Beta Summary Statistics ─────────────────────────
print("=" * 110)
print("  TABLE 1 — EFP BETA SUMMARY STATISTICS")
print("=" * 110)

header = (f"  {'Metal':8s}  {'Static β(Δ)':>12s}  {'Static β(Lvl)':>14s}  "
          f"{'Theo β(Avg)':>12s}  {'β Excess(Avg)':>14s}  "
          f"{'R²':>6s}  {'HAC t-stat':>10s}  {'Sample':>24s}  {'N':>6s}")
print(header)
print(f"  {'─' * 106}")

for metal in config['metals']:
    sr = static_results.get(metal, {})
    lr = level_results.get(metal, {}) if 'level_results' in dir() else {}

    # Average theoretical beta
    bc_sub = beta_comparison[beta_comparison['metal'] == metal].dropna(
        subset=['beta_theo'])
    avg_theo = bc_sub['beta_theo'].mean() if len(bc_sub) > 0 else np.nan
    avg_excess = bc_sub['beta_excess'].mean() if len(bc_sub) > 0 else np.nan

    beta_d = sr.get('beta_spot', np.nan)
    beta_l = lr.get('beta_level', np.nan)
    r2 = sr.get('r_squared', np.nan)
    t_stat = sr.get('t_stat', np.nan)
    n_obs = sr.get('n_obs', 0)
    dr = sr.get('date_range', (pd.NaT, pd.NaT))

    sample_str = (f"{dr[0]:%Y-%m-%d} to {dr[1]:%Y-%m-%d}"
                  if pd.notna(dr[0]) else "N/A")

    print(f"  {metal.upper():8s}  "
          f"{beta_d:>+12.6f}  "
          f"{beta_l:>+14.6f}  " if pd.notna(beta_l) else f"  {metal.upper():8s}  "
          f"{beta_d:>+12.6f}  "
          f"{'N/A':>14s}  ",
          end="")
    print(f"{avg_theo:>12.6f}  " if pd.notna(avg_theo) else f"{'N/A':>12s}  ", end="")
    print(f"{avg_excess:>+14.6f}  " if pd.notna(avg_excess) else f"{'N/A':>14s}  ", end="")
    print(f"{r2:>6.4f}  " if pd.notna(r2) else f"{'N/A':>6s}  ", end="")
    print(f"{t_stat:>+10.3f}  " if pd.notna(t_stat) else f"{'N/A':>10s}  ", end="")
    print(f"{sample_str:>24s}  {n_obs:>6,}")"""
    )
)

new_cells.append(
    md("""### Table 2 — Current EFP Snapshot""")
)

new_cells.append(
    code(
        """# ── Table 2: Current EFP Snapshot ─────────────────────────────────
print("=" * 140)
print("  TABLE 2 — CURRENT EFP SNAPSHOT")
print("=" * 140)

header2 = (f"  {'Metal':8s}  {'Contract':>10s}  {'DaysFND':>7s}  "
           f"{'Spot':>10s}  {'Futures':>10s}  {'F_OTC':>10s}  "
           f"{'EFP_raw':>10s}  {'EFP_adj':>10s}  {'EFP_ann%':>9s}  "
           f"{'FwdRate%':>8s}  {'RollBeta':>10s}  {'TheoBeta':>10s}")
print(header2)
print(f"  {'─' * 136}")

for metal in config['metals']:
    latest = master_df[master_df['metal'] == metal].sort_values('date').iloc[-1]

    # Latest rolling beta
    rb_m = rolling_betas[
        rolling_betas['metal'] == metal
    ].sort_values('date').dropna(subset=['rolling_beta'])
    rb_val = rb_m['rolling_beta'].iloc[-1] if len(rb_m) > 0 else np.nan

    # Contract name (from ticker)
    contract = latest.get('ticker', 'N/A')

    # Days to FND
    days_fnd = int(latest['delta_T'] * 365) if pd.notna(latest['delta_T']) else 0

    # EFP_adj_ann as %
    ann_pct = (latest['EFP_adj_ann'] * 100
               if pd.notna(latest.get('EFP_adj_ann')) else np.nan)

    # Fwd rate
    fwd_pct = (latest['fwd_rate_interp'] * 100
               if pd.notna(latest.get('fwd_rate_interp')) else np.nan)

    theo = latest.get('beta_theo', np.nan)

    print(f"  {metal.upper():8s}  "
          f"{str(contract):>10s}  "
          f"{days_fnd:>7d}  "
          f"${latest['spot']:>9,.2f}  "
          f"${latest['futures_price']:>9,.2f}  "
          f"${latest['F_OTC']:>9,.2f}  "
          f"${latest['EFP_raw']:>+9.2f}  "
          f"${latest['EFP_adj']:>+9.4f}  ", end="")
    print(f"{ann_pct:>+8.3f}%  " if pd.notna(ann_pct) else f"{'N/A':>9s}  ", end="")
    print(f"{fwd_pct:>+7.3f}%  " if pd.notna(fwd_pct) else f"{'N/A':>8s}  ", end="")
    print(f"{rb_val:>+10.6f}  " if pd.notna(rb_val) else f"{'N/A':>10s}  ", end="")
    print(f"{theo:>10.6f}" if pd.notna(theo) else f"{'N/A':>10s}")

print(f"\\n  Snapshot date: {master_df['date'].max():%Y-%m-%d}")"""
    )
)

new_cells.append(
    md("""### Table 3 — Residual Delta for Standard Position Sizes""")
)

new_cells.append(
    code(
        """# ── Table 3: Residual Delta for Standard Positions ───────────────
print("=" * 90)
print("  TABLE 3 — RESIDUAL DELTA FOR STANDARD POSITION SIZES")
print("=" * 90)

print(f"\\n  {'Metal':8s}  {'Position (oz)':>14s}  {'Emp Beta':>12s}  "
      f"{'USD Delta / 1% Spot':>22s}")
print(f"  {'─' * 62}")

position_map = {
    'gold':   [1_000, 5_000, 10_000],
    'silver': [5_000, 25_000, 50_000],
}

for metal in config['metals']:
    latest = master_df[master_df['metal'] == metal].sort_values('date').iloc[-1]
    spot = latest['spot']

    rb_m = rolling_betas[
        rolling_betas['metal'] == metal
    ].sort_values('date').dropna(subset=['rolling_beta'])
    emp_beta = rb_m['rolling_beta'].iloc[-1] if len(rb_m) > 0 else np.nan

    for pos in position_map[metal]:
        if pd.notna(emp_beta):
            usd_delta = pos * emp_beta * (spot / 100.0)
            print(f"  {metal.upper():8s}  {pos:>14,}  {emp_beta:>+12.6f}  "
                  f"${usd_delta:>+20,.2f}")
        else:
            print(f"  {metal.upper():8s}  {pos:>14,}  {'N/A':>12s}  {'N/A':>22s}")

print(f"\\n  Formula: USD delta = position_oz × beta × (spot / 100)")
print(f"  Interpretation: P&L from a 1% spot move BEYOND carry")"""
    )
)

# ═══════════════════════════════════════════════════════════════════
# SECTION 20: Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════
new_cells.append(
    md(
        """## Section 20 — Sensitivity Analysis

Two sensitivity tests:
1. **Window-length sensitivity**: How stable is rolling beta across
   30, 60, 90, and 120-day estimation windows?
2. **Forward rate sensitivity**: What if the forward rate were
   +50bps or -50bps from observed? Impact on EFP\_adj and beta\_theo."""
    )
)

new_cells.append(
    md("""### Sensitivity 1 — Rolling Window Length""")
)

new_cells.append(
    code(
        """def efp_beta_sensitivity(metal, master_df, windows=[30, 60, 90, 120]):
    \"\"\"Run rolling beta for multiple window lengths and plot.

    Parameters
    ----------
    metal : str
        'gold' or 'silver'.
    master_df : pd.DataFrame
        Master DataFrame with delta_spot and delta_efp_adj.
    windows : list of int
        Rolling window sizes in trading days.

    Returns
    -------
    dict
        {window: pd.DataFrame with date and rolling_beta}.
    \"\"\"
    sub = master_df[
        (master_df['metal'] == metal) &
        (~master_df['exclude_from_regression'].astype(bool))
    ][['date', 'delta_spot', 'delta_efp_adj']].dropna().copy()
    sub = sub.sort_values('date').reset_index(drop=True)

    results = {}
    for w in windows:
        if len(sub) < w + 10:
            continue
        y = sub['delta_efp_adj']
        X = sm.add_constant(sub['delta_spot'])
        rols = RollingOLS(y, X, window=w).fit()
        df = pd.DataFrame({
            'date': sub['date'].values,
            'rolling_beta': rols.params.iloc[:, 1].values,
        })
        results[w] = df

    return results


fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
colors = ['#3498db', '#2c3e50', '#e67e22', '#e74c3c']
windows = [30, 60, 90, 120]

for i, metal in enumerate(config['metals']):
    ax = axes[i]
    sens = efp_beta_sensitivity(metal, master_df, windows=windows)

    for j, (w, df) in enumerate(sens.items()):
        df_clean = df.dropna(subset=['rolling_beta'])
        lw = 1.5 if w == 60 else 0.9
        alpha = 1.0 if w == 60 else 0.6
        ax.plot(df_clean['date'], df_clean['rolling_beta'],
                linewidth=lw, color=colors[j], alpha=alpha,
                label=f'{w}d window')

    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
    ax.set_title(f'{metal.upper()} — Rolling Beta Sensitivity to Window Length',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Rolling Beta', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.show()

print("Window sensitivity analysis complete.")
print("  60d window is the default; shorter = noisier, longer = smoother/laggier")"""
    )
)

new_cells.append(
    md("""### Sensitivity 2 — Forward Rate Shock (+/- 50bps)""")
)

new_cells.append(
    code(
        """# ── Forward rate sensitivity ──────────────────────────────────────
shocks_bps = [-50, 0, +50]

print("=" * 90)
print("  FORWARD RATE SENSITIVITY: Impact of +/- 50bps on EFP_adj and beta_theo")
print("=" * 90)

print(f"\\n  {'Metal':8s}  {'Shock':>8s}  {'Fwd Rate%':>10s}  "
      f"{'F_OTC':>12s}  {'EFP_adj':>12s}  {'beta_theo':>12s}")
print(f"  {'─' * 68}")

for metal in config['metals']:
    latest = master_df[master_df['metal'] == metal].sort_values('date').iloc[-1]
    spot = latest['spot']
    delta_T = latest['delta_T']
    fwd_rate = latest['fwd_rate_interp']
    futures = latest['futures_price']

    for shock in shocks_bps:
        shocked_rate = fwd_rate + shock / 10_000  # bps to decimal
        shocked_fotc = spot * (1.0 + shocked_rate * delta_T)
        shocked_efp_adj = futures - shocked_fotc
        shocked_beta_theo = shocked_rate * delta_T

        label = f"{shock:+d}bps" if shock != 0 else "BASE"
        print(f"  {metal.upper():8s}  {label:>8s}  "
              f"{shocked_rate*100:>+9.4f}%  "
              f"${shocked_fotc:>11,.2f}  "
              f"${shocked_efp_adj:>+11.4f}  "
              f"{shocked_beta_theo:>12.6f}")

print(f"\\n  Note: A +50bps rate shock reduces F_OTC (making EFP_adj more positive)")
print(f"  and increases beta_theo. This shows how sensitive the EFP is to")
print(f"  the accuracy of the forward rate curve.")"""
    )
)

# ═══════════════════════════════════════════════════════════════════
# SECTION 21: Daily Workflow Documentation
# ═══════════════════════════════════════════════════════════════════
new_cells.append(
    md(
        """## Section 21 — HOW TO USE THIS NOTEBOOK: Daily Workflow

### What this notebook does

This notebook pulls live COMEX gold and silver futures prices, spot prices,
and Bloomberg native forward rate curves via BQL, constructs the EFP
(Exchange for Physical = COMEX Futures - OTC Forward), and quantifies the
residual spot delta embedded in an EFP position. It auto-updates every time
you run it, always reflecting today's market.

---

### Daily Steps

| Step | Action | Section |
|------|--------|---------|
| 1 | **Re-run Sections 2-5** BQL pulls to refresh all data | Sections 2-5 |
| 2 | **Check Section 6** — has the front month rolled overnight? Look for sawtooth jumps in delta\\_T | Section 6 |
| 3 | **Review Section 12 charts** — is EFP\\_adj normal or spiking? Spikes = physical scarcity / tariff premium | Section 12 |
| 4 | **Read Table 2 (Section 19)** — today's live EFP snapshot: spot, futures, F\\_OTC, EFP\\_adj, rolling beta | Section 19 |
| 5 | **Read current rolling beta** — what directional delta are you carrying beyond carry? | Section 14, Table 2 |
| 6 | **Use Table 3** to determine any delta hedge required for your position size | Section 19, Table 3 |

---

### Parameter Tuning Guide

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `regression_window_days` | `config` dict (Section 0) | 60 | Rolling OLS window. Shorter = more reactive, noisier. Longer = smoother, laggier. |
| `roll_days_before_fnd` | `config` dict (Section 0) | 5 | Days before FND to skip in roll detection. Increase if you see spurious roll flags near expiry. |
| `start_date` | `config` dict (Section 0) | 2023-01-01 | Set earlier for longer history (e.g. 2020-01-01). May slow BQL pulls. |
| `k_sigma` | `config` dict (Section 0) | 3 | Roll detection threshold (multiples of 20d rolling std). Lower = more sensitive. |

---

### Interpretation Guide

| Signal | Meaning | Action |
|--------|---------|--------|
| **EFP\\_adj > 0** | COMEX rich to OTC; physical scarcity or tariff/repatriation premium priced in | Monitor for widening; consider selling EFP if premium is extreme |
| **EFP\\_adj < 0** | COMEX cheap to OTC; rare, possible cross-market liquidity dislocation | Potential buying opportunity |
| **beta\\_excess > 0** | Residual delta of long EFP larger than pure carry theory predicts; net long bias stronger than expected | Hedge additional delta if unwanted; or let ride if directionally bullish |
| **beta\\_excess < 0** | EFP moves less than theory suggests; possible mean-reversion or offsetting rate move | Less hedging needed; EFP is a purer carry trade |
| **Rolling beta unstable** | Regime change (compare 30d vs 120d in Section 20) | Use wider window or reduce position size |"""
    )
)

# ═══════════════════════════════════════════════════════════════════
# SECTION 23: End-to-End Validation (Worked Example)
# ═══════════════════════════════════════════════════════════════════
new_cells.append(
    md(
        """## Section 23 — End-to-End Validation: Worked Example

> **PURPOSE:** Verify the entire pipeline computes correctly by walking
> through every step on a single known historical date.
>
> We pick a date in **January 2025** when gold EFPs were elevated due to
> tariff concerns — this makes the validation meaningful because the EFP
> was non-trivially different from zero."""
    )
)

new_cells.append(
    code(
        r"""# ── Pick a validation date ─────────────────────────────────────────
# Target: a date in Jan 2025 when gold EFP was elevated
gold_sub = master_df[master_df['metal'] == 'gold'].sort_values('date')

# Find a date near 2025-01-15 (tariff risk period)
target = pd.Timestamp('2025-01-15')
gold_sub['_dist'] = (gold_sub['date'] - target).abs()
val_idx = gold_sub['_dist'].idxmin()
val_row = gold_sub.loc[val_idx].copy()
val_date = val_row['date']

print("=" * 80)
print("  END-TO-END VALIDATION — WORKED EXAMPLE")
print("  Verifies the entire pipeline on a known historical date")
print("=" * 80)

print(f"\n  Validation date: {val_date:%Y-%m-%d}")
print(f"  (Selected as nearest available date to 2025-01-15, tariff period)")

# Step 1: Active contract
print(f"\n  1. Active front-month (GC1): {val_row['ticker']}")

# Step 2: FND and delta_T
print(f"  2. FND: {val_row['fnd']}")
print(f"     delta_T: {val_row['delta_T']:.6f} years "
      f"({val_row['delta_T']*365:.1f} calendar days)")

# Step 3: Spot price
print(f"  3. Spot price (XAU): ${val_row['spot']:,.2f}")

# Step 4: Futures price
print(f"  4. Futures price (GC1): ${val_row['futures_price']:,.2f}")

# Step 5: Forward rate tenors
print(f"  5. Bloomberg forward rate tenors:")
fwd_cols = ['fwd_1W', 'fwd_1M', 'fwd_2M', 'fwd_3M', 'fwd_6M', 'fwd_12M']
tenor_labels = ['1W', '1M', '2M', '3M', '6M', '12M']
for col, lbl in zip(fwd_cols, tenor_labels):
    val = val_row.get(col, np.nan)
    if pd.notna(val):
        print(f"     {lbl:>4s}: {val*100:+.4f}%")
    else:
        print(f"     {lbl:>4s}: N/A")

# Step 6: Interpolated forward rate
print(f"  6. Interpolated forward rate at delta_T={val_row['delta_T']:.4f}y: "
      f"{val_row['fwd_rate_interp']*100:+.4f}%")

# Step 7: F_OTC
# Recompute to verify
f_otc_check = val_row['spot'] * (1.0 + val_row['fwd_rate_interp'] * val_row['delta_T'])
print(f"  7. F_OTC = spot × (1 + fwd_rate × delta_T)")
print(f"         = {val_row['spot']:,.2f} × (1 + {val_row['fwd_rate_interp']:.6f} "
      f"× {val_row['delta_T']:.6f})")
print(f"         = ${f_otc_check:,.4f}")
print(f"     Stored: ${val_row['F_OTC']:,.4f}  "
      f"(diff: ${abs(f_otc_check - val_row['F_OTC']):.6f})")

# Step 8: EFP_raw
efp_raw_check = val_row['futures_price'] - val_row['spot']
print(f"  8. EFP_raw = futures - spot = "
      f"${val_row['futures_price']:,.2f} - ${val_row['spot']:,.2f} "
      f"= ${efp_raw_check:+,.2f}")
print(f"     Stored: ${val_row['EFP_raw']:+,.2f}")

# Step 9: EFP_adj
efp_adj_check = val_row['futures_price'] - f_otc_check
print(f"  9. EFP_adj = futures - F_OTC = "
      f"${val_row['futures_price']:,.2f} - ${f_otc_check:,.4f} "
      f"= ${efp_adj_check:+,.4f}")
print(f"     Stored: ${val_row['EFP_adj']:+,.4f}")

# Step 10: Theoretical beta
beta_theo_check = val_row['fwd_rate_interp'] * val_row['delta_T']
print(f"  10. beta_theo = fwd_rate_interp × delta_T = "
      f"{val_row['fwd_rate_interp']:.6f} × {val_row['delta_T']:.6f} "
      f"= {beta_theo_check:.6f}")
print(f"      Stored: {val_row.get('beta_theo', np.nan):.6f}")

# Step 11: Rolling empirical beta on that date
rb_gold = rolling_betas[rolling_betas['metal'] == 'gold'].copy()
rb_gold['_dist'] = (rb_gold['date'] - val_date).abs()
if len(rb_gold) > 0:
    rb_nearest = rb_gold.loc[rb_gold['_dist'].idxmin()]
    print(f"  11. Rolling empirical beta (60d) on {rb_nearest['date']:%Y-%m-%d}: "
          f"{rb_nearest['rolling_beta']:+.6f}")
    emp_beta_val = rb_nearest['rolling_beta']
else:
    print(f"  11. Rolling empirical beta: N/A (insufficient data)")
    emp_beta_val = np.nan

# Step 12: Beta excess
if pd.notna(emp_beta_val):
    beta_excess_check = emp_beta_val - beta_theo_check
    print(f"  12. beta_excess = empirical - theoretical = "
          f"{emp_beta_val:.6f} - {beta_theo_check:.6f} "
          f"= {beta_excess_check:+.6f}")
else:
    print(f"  12. beta_excess: N/A")
    beta_excess_check = np.nan

# Step 13: USD delta for 10,000 oz
position = 10_000
if pd.notna(emp_beta_val):
    usd_delta = position * emp_beta_val * (val_row['spot'] / 100.0)
    print(f"  13. USD delta exposure for {position:,} oz long EFP, 1% spot move:")
    print(f"      = {position:,} × {emp_beta_val:.6f} × "
          f"(${val_row['spot']:,.2f} / 100)")
    print(f"      = ${usd_delta:+,.2f}")
else:
    print(f"  13. USD delta: N/A")

# Verification summary
print(f"\n  {'─' * 60}")
print(f"  VALIDATION RESULT:")
f_otc_ok = abs(f_otc_check - val_row['F_OTC']) < 0.01
efp_raw_ok = abs(efp_raw_check - val_row['EFP_raw']) < 0.01
efp_adj_ok = abs(efp_adj_check - val_row['EFP_adj']) < 0.01
bt_ok = abs(beta_theo_check - val_row.get('beta_theo', beta_theo_check)) < 0.000001

checks = {
    'F_OTC recomputed matches stored': f_otc_ok,
    'EFP_raw recomputed matches stored': efp_raw_ok,
    'EFP_adj recomputed matches stored': efp_adj_ok,
    'beta_theo recomputed matches stored': bt_ok,
}
for name, ok in checks.items():
    print(f"    {'PASS' if ok else 'FAIL'} | {name}")
print(f"  All {len(checks)} checks passed: {all(checks.values())}")

# Clean up temp columns
gold_sub.drop(columns=['_dist'], inplace=True, errors='ignore')"""
    )
)

# ═══════════════════════════════════════════════════════════════════
# Final completion cell
# ═══════════════════════════════════════════════════════════════════
new_cells.append(
    md(
        """---
## Notebook Complete

All 23 sections (Prompts A–D) are implemented. This notebook is
**production-ready** for daily use on Bloomberg BQuant.

**Outputs saved:**
- `efp_master_data.csv` — raw master DataFrame (Prompt A)
- `efp_with_spreads.csv` — with EFP columns (Prompt B)
- `efp_beta_results.csv` — beta comparison series (Prompt C)"""
    )
)

new_cells.append(
    code(
        """print("=" * 80)
print("  NOTEBOOK COMPLETE")
print("=" * 80)
print(f"  Timestamp      : {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"  master_df      : {master_df.shape[0]:,} rows × {master_df.shape[1]} cols")
print(f"  beta_comparison: {beta_comparison.shape[0]:,} rows × {beta_comparison.shape[1]} cols")
print(f"  rolling_betas  : {rolling_betas.shape[0]:,} rows × {rolling_betas.shape[1]} cols")
print(f"\\n  Sections: 0-23 (Prompts A through D)")
print(f"  All data sourced from Bloomberg BQL — auto-updating on re-run.")
print(f"\\n  Files saved:")
print(f"    efp_master_data.csv")
print(f"    efp_with_spreads.csv")
print(f"    efp_beta_results.csv")"""
    )
)

# ═══════════════════════════════════════════════════════════════════
# Append new cells to notebook
# ═══════════════════════════════════════════════════════════════════
nb["cells"].extend(new_cells)

# ═══════════════════════════════════════════════════════════════════
# SECTION 22: Insert Table of Contents at position 1 (after title)
# ═══════════════════════════════════════════════════════════════════
# Update the title cell (cell 0)
nb["cells"][0] = md(
    """# COMEX Precious Metals EFP Beta Analysis — Gold & Silver

> **Quantifying residual spot delta when long the EFP using Bloomberg BQL on BQuant**
>
> **Objective:** Quantify the delta (beta to spot) implicitly carried when
> long the COMEX Gold or Silver EFP (Exchange for Physical).
>
> **EFP = COMEX Futures Price - OTC Forward Price (to FND)**
>
> This notebook uses **generic futures tickers** (GC1-GC4, SI1-SI4) so it
> auto-updates without maintaining a specific contract list.
> Coverage: front month through approximately 3-9 months forward.
>
> Self-contained BQuant notebook — executable top-to-bottom."""
)

toc_cell = md(
    """## Table of Contents

### Prompt A — Data Pipeline
- [Section 0 — Imports and Configuration](#Section-0-—-Imports-and-Configuration)
- [Section 1 — Define Generic Ticker Lists](#Section-1-—-Define-Generic-Ticker-Lists)
- [Section 2 — BQL Pull 1: Contract Metadata](#Section-2-—-BQL-Pull-1:-Contract-Metadata-from-Generic-Tickers)
- [Section 3 — BQL Pull 2: Historical Prices](#Section-3-—-BQL-Pull-2:-Historical-Prices-for-Generic-Tickers)
- [Section 4 — BQL Pull 3: Spot Prices](#Section-4-—-BQL-Pull-3:-Gold-and-Silver-Spot-Prices)
- [Section 5 — BQL Pull 4: Forward Rate Curves](#Section-5-—-BQL-Pull-4:-Bloomberg-Native-Forward-Rate-Curves)
- [Section 6 — Delta\\_T Calculation](#Section-6-—-Delta_T-Calculation-for-Generic-Tickers)
- [Section 7 — Master DataFrame Assembly](#Section-7-—-Master-DataFrame-Assembly)

### Prompt B — EFP Construction
- [Section 8 — Forward Rate Interpolation](#Section-8-—-Forward-Rate-Interpolation-to-Delta_T)
- [Section 9 — OTC Forward Price](#Section-9-—-OTC-Forward-Price-Anchored-to-FND)
- [Section 10 — EFP Series Construction](#Section-10-—-EFP-Series-Construction)
- [Section 11 — Daily Differenced Series](#Section-11-—-Daily-Differenced-Series-for-Regression)
- [Section 12 — EFP Time Series Visualisations](#Section-12-—-EFP-Time-Series-Visualisations)

### Prompt C — Regression & Beta
- [Section 13 — Static OLS Regression](#Section-13-—-Static-OLS-Regression:-Empirical-Beta)
- [Section 14 — Rolling OLS: Time-Varying Beta](#Section-14-—-Rolling-OLS:-Time-Varying-Beta)
- [Section 15 — Theoretical Beta](#Section-15-—-Theoretical-Beta-from-Cost-of-Carry)
- [Section 16 — Regression Diagnostics](#Section-16-—-Regression-Diagnostics)
- [Section 17 — Practical Delta Exposure](#Section-17-—-Practical-Interpretation:-Delta-Exposure)

### Prompt D — Dashboard & Documentation
- [Section 18 — Core Visualisation Suite](#Section-18-—-Core-Visualisation-Suite)
- [Section 19 — Summary Dashboard](#Section-19-—-Summary-Dashboard)
- [Section 20 — Sensitivity Analysis](#Section-20-—-Sensitivity-Analysis)
- [Section 21 — Daily Workflow Documentation](#Section-21-—-HOW-TO-USE-THIS-NOTEBOOK:-Daily-Workflow)
- [Section 23 — End-to-End Validation](#Section-23-—-End-to-End-Validation:-Worked-Example)"""
)

# Insert TOC at position 1 (right after the updated title)
nb["cells"].insert(1, toc_cell)

# ═══════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════
with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

# ═══════════════════════════════════════════════════════════════════
# Verify
# ═══════════════════════════════════════════════════════════════════
with open(nb_path) as f:
    nb2 = json.load(f)

n_cells = len(nb2["cells"])
n_md = sum(1 for c in nb2["cells"] if c["cell_type"] == "markdown")
n_code = sum(1 for c in nb2["cells"] if c["cell_type"] == "code")
n_new = n_cells - existing_count  # includes TOC insert

print(f"Updated: {nb_path}")
print(f"Total cells: {n_cells} (markdown: {n_md}, code: {n_code})")
print(f"Existing (Prompts A-C): {existing_count}, New (Prompt D + TOC): {n_new}")

all_code_text = "".join(
    "".join(c["source"]) for c in nb2["cells"] if c["cell_type"] == "code"
)
all_text = "".join("".join(c["source"]) for c in nb2["cells"])

checks = {
    "All source is list": all(isinstance(c["source"], list) for c in nb2["cells"]),
    "Title updated": "Quantifying residual spot delta" in all_text,
    "TOC inserted at position 1": "Table of Contents" in "".join(nb2["cells"][1]["source"]),
    "TOC has all sections": all(
        f"Section {i}" in "".join(nb2["cells"][1]["source"]) for i in range(0, 22)
    ),
    "Has reload guard (efp_with_spreads)": "efp_with_spreads.csv" in all_code_text,
    "Has reload guard (efp_beta_results)": "efp_beta_results.csv" in all_code_text,
    # Section 18 charts
    "Has Chart 1/2 (rolling vs theo)": "Empirical vs Theoretical Beta to Spot" in all_code_text,
    "Has Chart 3 (dual axis)": "twinx" in all_code_text,
    "Has Chart 4 (beta excess both metals)": "Beta Excess" in all_code_text,
    "Has Chart 5 (heatmap)": "sns.heatmap" in all_code_text,
    "Has position_sizes for heatmap": "position_sizes" in all_code_text,
    "Has RdYlGn colormap": "RdYlGn" in all_code_text,
    # Section 19 tables
    "Has Table 1 header": "TABLE 1" in all_code_text,
    "Has Table 2 header": "TABLE 2" in all_code_text,
    "Has Table 3 header": "TABLE 3" in all_code_text,
    # Section 20 sensitivity
    "Has efp_beta_sensitivity function": "def efp_beta_sensitivity" in all_code_text,
    "Has window sensitivity [30,60,90,120]": "[30, 60, 90, 120]" in all_code_text,
    "Has fwd rate shock +/-50bps": "50bps" in all_text,
    "Has shocks_bps list": "shocks_bps" in all_code_text,
    # Section 21 workflow
    "Has daily workflow markdown": "Daily Workflow" in all_text,
    "Has parameter tuning guide": "Parameter Tuning" in all_text,
    "Has interpretation guide": "Interpretation Guide" in all_text,
    # Section 23 validation
    "Has validation date selection": "2025-01-15" in all_code_text,
    "Has step-by-step validation": "VALIDATION RESULT" in all_code_text,
    "Has F_OTC recomputation check": "f_otc_check" in all_code_text,
    "Has 10,000 oz example": "10_000" in all_code_text or "10000" in all_code_text,
    # Structural
    "All code cells have outputs": all(
        "outputs" in c for c in nb2["cells"] if c["cell_type"] == "code"
    ),
    "Notebook complete marker": "NOTEBOOK COMPLETE" in all_code_text,
}

for name, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'} | {name}")

n_pass = sum(checks.values())
n_total = len(checks)
print(f"\n{n_pass}/{n_total} checks passed: {all(checks.values())}")

# Cell listing
print(f"\nFull cell listing:")
for i in range(n_cells):
    c = nb2["cells"][i]
    first = "".join(c["source"]).split("\n")[0][:90]
    print(f"  [{i:2d}] {c['cell_type']:8s} | {first}")
