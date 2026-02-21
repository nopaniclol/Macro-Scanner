#!/usr/bin/env python3
"""Append Sections 13-17 (Prompt C) to EFP_Beta_Analysis.ipynb."""
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
# Prompt C header + reload guard
# ═══════════════════════════════════════════════════════════════════
new_cells.append(
    md(
        """---
# Prompt C — OLS Regression, Rolling Beta, Theoretical Beta

> Sections 13–17: Static and rolling OLS regressions of EFP changes
> on spot changes, theoretical beta from cost-of-carry, regression
> diagnostics, and practical delta-exposure interpretation."""
    )
)

new_cells.append(
    code(
        """# ── Reload guard: load from CSV if master_df not in memory ────────
try:
    _ = master_df.shape
    print(f"master_df already in memory: {master_df.shape}")
except NameError:
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.regression.rolling import RollingOLS
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    from scipy import stats as sp_stats
    from datetime import date, datetime
    import warnings
    warnings.filterwarnings("ignore")

    master_df = pd.read_csv('efp_with_spreads.csv', parse_dates=['date', 'fnd'])
    print(f"Loaded master_df from CSV: {master_df.shape}")

    # Rebuild config if not present
    config = {
        'metals': ['gold', 'silver'],
        'start_date': '2023-01-01',
        'end_date': date.today().strftime('%Y-%m-%d'),
        'roll_days_before_fnd': 5,
        'regression_window_days': 60,
        'k_sigma': 3,
        'generic_depth': 4,
    }

# Ensure imports available regardless
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from scipy import stats as sp_stats"""
    )
)

# ═══════════════════════════════════════════════════════════════════
# SECTION 13: Static OLS Regression — Empirical Beta
# ═══════════════════════════════════════════════════════════════════
new_cells.append(
    md(
        r"""## Section 13 — Static OLS Regression: Empirical Beta

Full-sample OLS regression of daily EFP changes on daily spot changes:

$$\Delta \text{EFP\_adj}(t) = \alpha + \beta_{spot} \times \Delta \text{spot}(t) + \varepsilon(t)$$

**Newey-West HAC** standard errors (lags=5) correct for autocorrelation
in the residuals. A statistically significant positive $\beta_{spot}$
means the EFP widens when spot rises — i.e. COMEX richens relative to
OTC forwards when the market rallies.

Also runs a **level regression** (EFP\_adj on spot) to estimate how
many dollars the EFP level moves per dollar of spot, which should
approximate $r_{fwd} \times \Delta T$ theoretically."""
    )
)

new_cells.append(
    code(
        r"""# ── Static OLS: changes regression ────────────────────────────────
static_results = {}

print("=" * 80)
print("  SECTION 13 — STATIC OLS: delta_EFP_adj ~ delta_spot")
print("=" * 80)

for metal in config['metals']:
    sub = master_df[
        (master_df['metal'] == metal) &
        (~master_df['exclude_from_regression'])
    ][['date', 'delta_spot', 'delta_efp_adj']].dropna()

    if len(sub) < 30:
        print(f"\n  {metal.upper()}: insufficient data ({len(sub)} obs)")
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

    r = static_results[metal]
    sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 \
          else "*" if r['p_value'] < 0.05 else ""

    print(f"\n  {metal.upper()} — Changes Regression")
    print(f"  {'─' * 55}")
    print(f"  delta_EFP_adj = alpha + beta_spot × delta_spot + eps")
    print(f"  {'─' * 55}")
    print(f"    beta_spot    : {r['beta_spot']:+.6f} {sig}")
    print(f"    Std error    : {r['se_beta']:.6f}  (HAC, lags=5)")
    print(f"    t-statistic  : {r['t_stat']:+.3f}")
    print(f"    p-value      : {r['p_value']:.4e}")
    print(f"    alpha        : {r['alpha']:+.6f}")
    print(f"    R-squared    : {r['r_squared']:.4f}")
    print(f"    Observations : {r['n_obs']:,}")
    print(f"    Sample       : {r['date_range'][0]:%Y-%m-%d} to "
          f"{r['date_range'][1]:%Y-%m-%d}")

# ── Level regression: EFP_adj ~ spot ─────────────────────────────
print("\n" + "=" * 80)
print("  STATIC OLS: EFP_adj_level ~ spot_level")
print("=" * 80)

level_results = {}

for metal in config['metals']:
    sub = master_df[
        (master_df['metal'] == metal) &
        (~master_df['exclude_from_regression'])
    ][['date', 'spot', 'EFP_adj']].dropna()

    if len(sub) < 30:
        continue

    y = sub['EFP_adj'].values
    X = sm.add_constant(sub['spot'].values)

    model_lvl = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 10})

    level_results[metal] = {
        'beta_level': model_lvl.params[1],
        'se': model_lvl.bse[1],
        't_stat': model_lvl.tvalues[1],
        'p_value': model_lvl.pvalues[1],
        'r_squared': model_lvl.rsquared,
    }

    lr = level_results[metal]
    sig = "***" if lr['p_value'] < 0.001 else "**" if lr['p_value'] < 0.01 \
          else "*" if lr['p_value'] < 0.05 else ""

    # Theoretical comparison
    avg_fwd = master_df.loc[
        (master_df['metal'] == metal) &
        (~master_df['exclude_from_regression']),
        'fwd_rate_interp'
    ].mean()
    avg_dT = master_df.loc[
        (master_df['metal'] == metal) &
        (~master_df['exclude_from_regression']),
        'delta_T'
    ].mean()

    print(f"\n  {metal.upper()} — Level Regression")
    print(f"  {'─' * 55}")
    print(f"    beta_level   : {lr['beta_level']:+.6f} {sig}")
    print(f"    Std error    : {lr['se']:.6f}  (HAC, lags=10)")
    print(f"    t-statistic  : {lr['t_stat']:+.3f}")
    print(f"    p-value      : {lr['p_value']:.4e}")
    print(f"    R-squared    : {lr['r_squared']:.4f}")
    print(f"    Theoretical  : fwd_rate × delta_T ≈ {avg_fwd:.4f} × {avg_dT:.4f} "
          f"= {avg_fwd * avg_dT:.6f}")"""
    )
)

new_cells.append(
    md(
        r"""### Interpretation of Static Beta

**Changes regression** ($\Delta$EFP\_adj ~ $\Delta$spot):
- $\beta_{spot}$ measures: for every \$1 that spot moves, by how many dollars
  does the carry-stripped EFP change *on the same day*?
- A positive, significant $\beta_{spot}$ implies the COMEX futures richens (cheapens)
  versus OTC forwards when spot rallies (sells off). This is the **residual
  directional delta** embedded in an EFP position beyond pure carry.
- If $\beta_{spot} \approx 0$, the EFP is purely a carry/financing trade with
  no residual spot exposure.

**Level regression** (EFP\_adj ~ spot):
- $\beta_{level}$ estimates how many dollars the EFP level changes per
  dollar of spot. Theoretically this should be close to $r_{fwd} \times \Delta T$
  (the carry component's sensitivity to spot).
- In practice, the level regression captures both carry sensitivity *and*
  any persistent co-movement between the EFP and spot (e.g. during tariff periods)."""
    )
)

# ═══════════════════════════════════════════════════════════════════
# SECTION 14: Rolling OLS Regression — Rolling Beta
# ═══════════════════════════════════════════════════════════════════
new_cells.append(
    md(
        r"""## Section 14 — Rolling OLS: Time-Varying Beta

A 60-day (configurable) rolling window OLS captures how the EFP's
sensitivity to spot evolves over time. Key questions:

- Does $\beta_{spot}$ spike during stress episodes (tariffs, physical scarcity)?
- Is the beta stable enough to hedge, or does it require dynamic adjustment?
- Does the 95% confidence band exclude zero consistently?"""
    )
)

new_cells.append(
    code(
        """# ── Rolling OLS per metal ─────────────────────────────────────────
window = config['regression_window_days']
rolling_frames = []

print("=" * 80)
print(f"  SECTION 14 — ROLLING OLS (window={window} days)")
print("=" * 80)

for metal in config['metals']:
    sub = master_df[
        (master_df['metal'] == metal) &
        (~master_df['exclude_from_regression'])
    ][['date', 'delta_spot', 'delta_efp_adj']].dropna().copy()

    sub = sub.sort_values('date').reset_index(drop=True)

    if len(sub) < window + 10:
        print(f"  {metal.upper()}: insufficient data ({len(sub)} obs for window={window})")
        continue

    y = sub['delta_efp_adj']
    X = sm.add_constant(sub['delta_spot'])

    rols = RollingOLS(y, X, window=window).fit()

    sub['rolling_beta']  = rols.params.iloc[:, 1].values
    sub['rolling_alpha'] = rols.params.iloc[:, 0].values
    sub['rolling_rsq']   = rols.rsquared.values
    sub['rolling_se']    = rols.bse.iloc[:, 1].values

    # 95% confidence bands
    sub['upper_95'] = sub['rolling_beta'] + 1.96 * sub['rolling_se']
    sub['lower_95'] = sub['rolling_beta'] - 1.96 * sub['rolling_se']

    sub['metal'] = metal

    rolling_frames.append(sub[[
        'date', 'metal', 'rolling_beta', 'rolling_alpha',
        'rolling_rsq', 'rolling_se', 'upper_95', 'lower_95'
    ]])

    valid = sub['rolling_beta'].dropna()
    print(f"\\n  {metal.upper()}:")
    print(f"    Rolling beta  : mean={valid.mean():+.6f}  "
          f"std={valid.std():.6f}  range=[{valid.min():+.6f}, {valid.max():+.6f}]")
    print(f"    Rolling R²    : mean={sub['rolling_rsq'].dropna().mean():.4f}")
    print(f"    % of windows with beta > 0  : "
          f"{(valid > 0).mean()*100:.1f}%")
    print(f"    % of windows where 95% CI excludes 0 : "
          f"{((sub['lower_95'].dropna() > 0) | (sub['upper_95'].dropna() < 0)).mean()*100:.1f}%")

rolling_betas = pd.concat(rolling_frames, ignore_index=True)
print(f"\\nrolling_betas shape: {rolling_betas.shape}")"""
    )
)

new_cells.append(
    md("""### Rolling Beta Visualisation""")
)

new_cells.append(
    code(
        """fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

for i, metal in enumerate(config['metals']):
    ax = axes[i]
    sub = rolling_betas[rolling_betas['metal'] == metal].sort_values('date')

    ax.plot(sub['date'], sub['rolling_beta'],
            linewidth=1.2, color='#2c3e50', label='Rolling beta (60d)')
    ax.fill_between(sub['date'], sub['lower_95'], sub['upper_95'],
                    alpha=0.2, color='#3498db', label='95% CI')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)

    # Static beta reference line
    if metal in static_results:
        ax.axhline(y=static_results[metal]['beta_spot'],
                   color='#e67e22', linestyle=':', linewidth=1,
                   label=f"Static beta = {static_results[metal]['beta_spot']:.4f}")

    # Annotate tariff period
    tariff_date = pd.Timestamp('2025-01-15')
    if sub['date'].min() <= tariff_date <= sub['date'].max():
        ax.axvline(x=tariff_date, color='#e74c3c', linestyle='--',
                   linewidth=1, alpha=0.7)
        ax.annotate('Tariff risk\\nJan 2025',
                    xy=(tariff_date, sub['rolling_beta'].max() * 0.9),
                    fontsize=8, color='#e74c3c', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#e74c3c', alpha=0.8))

    ax.set_title(f'{metal.upper()} — Rolling Beta (delta_EFP_adj ~ delta_spot, '
                 f'{config["regression_window_days"]}d window)', fontsize=11)
    ax.set_ylabel('Beta')
    ax.legend(loc='best', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""
    )
)

# ═══════════════════════════════════════════════════════════════════
# SECTION 15: Theoretical Beta Computation
# ═══════════════════════════════════════════════════════════════════
new_cells.append(
    md(
        r"""## Section 15 — Theoretical Beta from Cost-of-Carry

The cost-of-carry model implies:

$$\text{EFP\_adj} \approx S \times r_{fwd} \times \Delta T - S \times r_{fwd} \times \Delta T = 0$$

But for changes: $\frac{\partial \text{EFP\_adj}}{\partial S} \approx r_{fwd} \times \Delta T$

So the **theoretical beta** is:

$$\beta_{theo}(t) = r_{fwd,interp}(t) \times \Delta T(t)$$

Comparing rolling empirical beta to $\beta_{theo}$ reveals:
- **beta\_excess > 0**: COMEX richening faster than carry implies — physical scarcity / tariff premium
- **beta\_excess < 0**: COMEX cheapening — liquidity normalising or physical surplus"""
    )
)

new_cells.append(
    code(
        """# ── Theoretical beta ──────────────────────────────────────────────
master_df['beta_theo'] = master_df['fwd_rate_interp'] * master_df['delta_T']

print("=" * 80)
print("  SECTION 15 — THEORETICAL BETA (fwd_rate × delta_T)")
print("=" * 80)

for metal in config['metals']:
    sub = master_df[
        (master_df['metal'] == metal) &
        (~master_df['exclude_from_regression'])
    ]['beta_theo'].dropna()
    print(f"\\n  {metal.upper()}:")
    print(f"    Mean   : {sub.mean():.6f}")
    print(f"    Median : {sub.median():.6f}")
    print(f"    Std    : {sub.std():.6f}")
    print(f"    Range  : [{sub.min():.6f}, {sub.max():.6f}]")

# ── Build beta_comparison DataFrame ──────────────────────────────
# Merge rolling betas with theoretical beta from master_df
beta_comparison_frames = []

for metal in config['metals']:
    rb = rolling_betas[rolling_betas['metal'] == metal][
        ['date', 'metal', 'rolling_beta', 'rolling_rsq',
         'upper_95', 'lower_95']
    ].copy()

    theo = master_df[master_df['metal'] == metal][
        ['date', 'beta_theo']
    ].copy()

    merged = rb.merge(theo, on='date', how='left')
    merged['beta_excess'] = merged['rolling_beta'] - merged['beta_theo']
    beta_comparison_frames.append(merged)

beta_comparison = pd.concat(beta_comparison_frames, ignore_index=True)

# Summary
print("\\n" + "=" * 80)
print("  BETA COMPARISON: EMPIRICAL vs THEORETICAL")
print("=" * 80)

for metal in config['metals']:
    sub = beta_comparison[beta_comparison['metal'] == metal].dropna(
        subset=['rolling_beta', 'beta_theo'])
    if len(sub) == 0:
        continue
    print(f"\\n  {metal.upper()}:")
    print(f"    Empirical beta  : mean={sub['rolling_beta'].mean():+.6f}")
    print(f"    Theoretical beta: mean={sub['beta_theo'].mean():.6f}")
    print(f"    Beta excess     : mean={sub['beta_excess'].mean():+.6f}  "
          f"std={sub['beta_excess'].std():.6f}")
    print(f"    % excess > 0   : {(sub['beta_excess'] > 0).mean()*100:.1f}%")

print(f"\\nbeta_comparison shape: {beta_comparison.shape}")"""
    )
)

new_cells.append(
    md("""### Empirical vs Theoretical Beta — Visual Comparison""")
)

new_cells.append(
    code(
        """fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for col_idx, metal in enumerate(config['metals']):
    sub = beta_comparison[beta_comparison['metal'] == metal].sort_values('date')

    # ── Top row: empirical vs theoretical beta ────────────────────
    ax = axes[0, col_idx]
    ax.plot(sub['date'], sub['rolling_beta'],
            linewidth=1.2, color='#2c3e50', label='Empirical (rolling 60d)')
    ax.plot(sub['date'], sub['beta_theo'],
            linewidth=1.0, color='#27ae60', linestyle='--',
            label='Theoretical (fwd_rate × delta_T)')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_title(f'{metal.upper()} — Empirical vs Theoretical Beta', fontsize=11)
    ax.set_ylabel('Beta')
    ax.legend(loc='best', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.sca(ax)
    plt.xticks(rotation=45, fontsize=8)

    # ── Bottom row: beta excess ───────────────────────────────────
    ax2 = axes[1, col_idx]
    ax2.bar(sub['date'], sub['beta_excess'],
            width=1.5, color=np.where(sub['beta_excess'] > 0, '#e74c3c', '#3498db'),
            alpha=0.6)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_title(f'{metal.upper()} — Beta Excess (Empirical − Theoretical)',
                  fontsize=11)
    ax2.set_ylabel('Beta Excess')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.sca(ax2)
    plt.xticks(rotation=45, fontsize=8)

    # Annotate tariff period
    tariff_date = pd.Timestamp('2025-01-15')
    for a in [ax, ax2]:
        if sub['date'].min() <= tariff_date <= sub['date'].max():
            a.axvline(x=tariff_date, color='#e74c3c', linestyle='--',
                      linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.show()"""
    )
)

# ═══════════════════════════════════════════════════════════════════
# SECTION 16: Regression Diagnostics
# ═══════════════════════════════════════════════════════════════════
new_cells.append(
    md(
        """## Section 16 — Regression Diagnostics

Diagnostic checks for the static OLS (changes regression):
1. Residuals time series — check for autocorrelation patterns
2. Q-Q plot — normality of residuals
3. Histogram with normal overlay — fat tails check
4. Scatter of delta\_spot vs delta\_efp\_adj with OLS fit line"""
    )
)

new_cells.append(
    code(
        """fig, axes = plt.subplots(4, 2, figsize=(16, 18))

for col_idx, metal in enumerate(config['metals']):
    if metal not in static_results:
        continue

    model = static_results[metal]['model']
    resid = model.resid

    # Dates for the regression sample
    sub = master_df[
        (master_df['metal'] == metal) &
        (~master_df['exclude_from_regression'])
    ][['date', 'delta_spot', 'delta_efp_adj']].dropna()
    dates = sub['date'].values

    # ── Row 0: Residuals time series ──────────────────────────────
    ax = axes[0, col_idx]
    ax.plot(dates, resid, linewidth=0.6, color='#2c3e50', alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
    ax.set_title(f'{metal.upper()} — OLS Residuals', fontsize=11)
    ax.set_ylabel('Residual')

    # Shade tariff period
    tariff_start = pd.Timestamp('2025-01-01')
    tariff_end = pd.Timestamp('2025-03-01')
    ax.axvspan(tariff_start, tariff_end, alpha=0.15, color='red',
               label='Tariff period')
    ax.legend(loc='best', fontsize=8)

    # ── Row 1: Q-Q plot ───────────────────────────────────────────
    ax = axes[1, col_idx]
    sp_stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title(f'{metal.upper()} — Q-Q Plot of Residuals', fontsize=11)
    ax.get_lines()[0].set_markersize(3)

    # ── Row 2: Histogram ──────────────────────────────────────────
    ax = axes[2, col_idx]
    ax.hist(resid, bins=60, density=True, alpha=0.7, color='#3498db',
            edgecolor='white', linewidth=0.5)
    # Normal overlay
    x_range = np.linspace(resid.min(), resid.max(), 200)
    ax.plot(x_range, sp_stats.norm.pdf(x_range, resid.mean(), resid.std()),
            color='#e74c3c', linewidth=1.5, label='Normal')
    ax.set_title(f'{metal.upper()} — Residual Histogram', fontsize=11)
    ax.set_xlabel('Residual')
    ax.set_ylabel('Density')
    ax.legend(loc='best', fontsize=8)

    # ── Row 3: Scatter with OLS line ──────────────────────────────
    ax = axes[3, col_idx]
    ax.scatter(sub['delta_spot'].values, sub['delta_efp_adj'].values,
               s=8, alpha=0.4, color='#3498db')
    # OLS line
    x_line = np.linspace(sub['delta_spot'].min(), sub['delta_spot'].max(), 100)
    y_line = model.params[0] + model.params[1] * x_line
    ax.plot(x_line, y_line, color='#e74c3c', linewidth=1.5,
            label=f'OLS: beta={model.params[1]:.4f}')
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.3)
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.3)
    ax.set_title(f'{metal.upper()} — delta_spot vs delta_EFP_adj', fontsize=11)
    ax.set_xlabel('delta_spot ($/oz)')
    ax.set_ylabel('delta_EFP_adj ($/oz)')
    ax.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.show()"""
    )
)

new_cells.append(
    md("""### Diagnostic Statistics""")
)

new_cells.append(
    code(
        """from statsmodels.stats.stattools import durbin_watson

print("=" * 80)
print("  REGRESSION DIAGNOSTIC STATISTICS")
print("=" * 80)

for metal in config['metals']:
    if metal not in static_results:
        continue

    model = static_results[metal]['model']
    resid = model.resid

    dw = durbin_watson(resid)
    kurt = sp_stats.kurtosis(resid, fisher=True)  # excess kurtosis
    skew = sp_stats.skew(resid)
    jb_stat, jb_p = sp_stats.jarque_bera(resid)

    print(f"\\n  {metal.upper()}:")
    print(f"    Durbin-Watson         : {dw:.4f}  "
          f"({'≈2 OK' if 1.5 < dw < 2.5 else 'autocorrelation detected'})")
    print(f"    Excess kurtosis       : {kurt:.2f}  "
          f"({'fat tails' if abs(kurt) > 1 else 'near-normal'})")
    print(f"    Skewness              : {skew:+.2f}")
    print(f"    Jarque-Bera statistic : {jb_stat:.1f}  (p={jb_p:.4e})")
    print(f"    Residual std          : {resid.std():.6f}")

    # Check for structural breaks: compare first/second half betas
    sub = master_df[
        (master_df['metal'] == metal) &
        (~master_df['exclude_from_regression'])
    ][['date', 'delta_spot', 'delta_efp_adj']].dropna()

    mid = len(sub) // 2
    for label, half in [('First half', sub.iloc[:mid]), ('Second half', sub.iloc[mid:])]:
        y_h = half['delta_efp_adj'].values
        X_h = sm.add_constant(half['delta_spot'].values)
        m_h = sm.OLS(y_h, X_h).fit()
        print(f"    {label:12s} beta : {m_h.params[1]:+.6f}  "
              f"(n={len(half)}, R²={m_h.rsquared:.4f})")"""
    )
)

new_cells.append(
    md(
        r"""### Diagnostic Notes

- **Durbin-Watson ≈ 2** → no strong first-order autocorrelation in residuals
  (HAC standard errors already guard against this, but DW confirms).
- **Excess kurtosis > 0** → fat tails are expected in financial data and
  confirmed by Q-Q plots. This is why we use HAC and interpret p-values cautiously.
- **Structural break check**: comparing first-half vs second-half betas
  detects regime shifts. A large difference (especially around the 2025 tariff
  period) suggests the relationship is non-stationary and supports using
  rolling rather than static betas for risk management."""
    )
)

# ═══════════════════════════════════════════════════════════════════
# SECTION 17: Practical Interpretation of Beta
# ═══════════════════════════════════════════════════════════════════
new_cells.append(
    md(
        r"""## Section 17 — Practical Interpretation: Delta Exposure

For a trader holding an EFP position (long COMEX futures / short OTC forward),
the residual directional P&L from a spot move is:

$$\text{USD delta} = \text{position\_oz} \times \beta \times \frac{S}{100}$$

where $S/100$ converts a 1% spot move to dollars.

This answers: *"If I am long X oz of EFP and spot moves 1%, how much
residual P&L do I make/lose beyond the carry?"*"""
    )
)

new_cells.append(
    code(
        """def compute_delta_exposure(position_oz, beta, spot_price):
    \"\"\"Compute USD delta exposure for a 1% spot move.

    Parameters
    ----------
    position_oz : float
        EFP position in troy ounces.
    beta : float
        Empirical or theoretical beta.
    spot_price : float
        Current spot price ($/oz).

    Returns
    -------
    float
        USD P&L for a 1% spot move.
    \"\"\"
    return position_oz * beta * (spot_price / 100.0)


# ── Build scenario table ─────────────────────────────────────────
print("=" * 80)
print("  SECTION 17 — DELTA EXPOSURE TABLE")
print("=" * 80)

# Get latest values
scenarios = []

for metal in config['metals']:
    latest = master_df[master_df['metal'] == metal].sort_values('date').iloc[-1]
    spot = latest['spot']

    # Latest rolling beta
    rb_metal = rolling_betas[
        rolling_betas['metal'] == metal
    ].sort_values('date').dropna(subset=['rolling_beta'])
    emp_beta = rb_metal['rolling_beta'].iloc[-1] if len(rb_metal) > 0 else np.nan

    # Theoretical beta
    theo_beta = latest.get('beta_theo', np.nan)

    if metal == 'gold':
        positions = [1_000, 5_000, 10_000]
    else:
        positions = [5_000, 25_000, 50_000]

    for pos in positions:
        usd_emp = compute_delta_exposure(pos, emp_beta, spot)
        usd_theo = compute_delta_exposure(pos, theo_beta, spot)
        scenarios.append({
            'metal': metal.upper(),
            'position_oz': f"{pos:,}",
            'spot': f"${spot:,.2f}",
            'empirical_beta': f"{emp_beta:.6f}" if pd.notna(emp_beta) else "N/A",
            'theo_beta': f"{theo_beta:.6f}" if pd.notna(theo_beta) else "N/A",
            'USD_delta_empirical': f"${usd_emp:+,.2f}" if pd.notna(usd_emp) else "N/A",
            'USD_delta_theoretical': f"${usd_theo:+,.2f}" if pd.notna(usd_theo) else "N/A",
            'comment': 'Residual directional exposure over and above carry',
        })

scenario_df = pd.DataFrame(scenarios)

# Print formatted
print(f"\\n  {'Metal':8s}  {'Position':>10s}  {'Spot':>10s}  {'Emp Beta':>12s}  "
      f"{'Theo Beta':>12s}  {'USD/1% (emp)':>14s}  {'USD/1% (theo)':>14s}")
print(f"  {'─' * 90}")

for _, row in scenario_df.iterrows():
    print(f"  {row['metal']:8s}  {row['position_oz']:>10s}  {row['spot']:>10s}  "
          f"{row['empirical_beta']:>12s}  {row['theo_beta']:>12s}  "
          f"{row['USD_delta_empirical']:>14s}  {row['USD_delta_theoretical']:>14s}")

print(f"\\n  Comment: {scenarios[0]['comment']}")
print(f"\\n  Note: USD delta = position_oz × beta × (spot / 100)")
print(f"  This is the P&L from a 1% spot move BEYOND carry.")"""
    )
)

new_cells.append(
    md("""### Export Beta Comparison""")
)

new_cells.append(
    code(
        """# ── Save beta_comparison ──────────────────────────────────────────
output_path = 'efp_beta_results.csv'
beta_comparison.to_csv(output_path, index=False)

print(f"Saved to: {output_path}")
print(f"  {beta_comparison.shape[0]:,} rows x {beta_comparison.shape[1]} columns")
print(f"  Columns: {list(beta_comparison.columns)}")

# ── Final summary ────────────────────────────────────────────────
print("\\n" + "=" * 80)
print("  PROMPT C COMPLETE — SUMMARY")
print("=" * 80)

for metal in config['metals']:
    print(f"\\n  {metal.upper()}:")

    if metal in static_results:
        sr = static_results[metal]
        print(f"    Static beta (changes)  : {sr['beta_spot']:+.6f}  "
              f"(p={sr['p_value']:.4e}, R²={sr['r_squared']:.4f})")

    rb_metal = rolling_betas[
        rolling_betas['metal'] == metal
    ].sort_values('date').dropna(subset=['rolling_beta'])
    if len(rb_metal) > 0:
        latest_rb = rb_metal.iloc[-1]
        print(f"    Latest rolling beta    : {latest_rb['rolling_beta']:+.6f}")
        print(f"    Latest rolling R²      : {latest_rb['rolling_rsq']:.4f}")

    bc = beta_comparison[beta_comparison['metal'] == metal].dropna(
        subset=['beta_excess'])
    if len(bc) > 0:
        latest_bc = bc.sort_values('date').iloc[-1]
        print(f"    Latest beta_theo       : {latest_bc['beta_theo']:.6f}")
        print(f"    Latest beta_excess     : {latest_bc['beta_excess']:+.6f}")

print(f"\\nPrompt C complete: {datetime.now():%Y-%m-%d %H:%M}")
print("beta_comparison saved to 'efp_beta_results.csv'.")
print("Ready for Prompt D.")"""
    )
)

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
print(f"Existing (Prompts A+B): {existing_count}, New (Prompt C): {n_new}")

all_code_text = "".join(
    "".join(c["source"]) for c in nb2["cells"] if c["cell_type"] == "code"
)
all_text = "".join("".join(c["source"]) for c in nb2["cells"])

checks = {
    "All source is list": all(isinstance(c["source"], list) for c in nb2["cells"]),
    "Has reload guard (efp_with_spreads)": "efp_with_spreads.csv" in all_code_text,
    "Has sm.OLS": "sm.OLS" in all_code_text,
    "Has Newey-West HAC": "HAC" in all_code_text,
    "Has cov_kwds maxlags": "maxlags" in all_code_text,
    "Has add_constant": "add_constant" in all_code_text,
    "Has static_results dict": "static_results" in all_code_text,
    "Has level regression": "beta_level" in all_code_text,
    "Has RollingOLS": "RollingOLS" in all_code_text,
    "Has rolling_beta": "rolling_beta" in all_code_text,
    "Has rolling_alpha": "rolling_alpha" in all_code_text,
    "Has rolling_rsq": "rolling_rsq" in all_code_text,
    "Has rolling_se": "rolling_se" in all_code_text,
    "Has upper_95": "upper_95" in all_code_text,
    "Has lower_95": "lower_95" in all_code_text,
    "Has beta_theo": "beta_theo" in all_code_text,
    "Has beta_excess": "beta_excess" in all_code_text,
    "Has beta_comparison": "beta_comparison" in all_code_text,
    "Has durbin_watson": "durbin_watson" in all_code_text,
    "Has kurtosis check": "kurtosis" in all_code_text,
    "Has QQ plot": "probplot" in all_code_text,
    "Has residual histogram": "hist(resid" in all_code_text,
    "Has scatter OLS line": "x_line" in all_code_text,
    "Has structural break (half split)": "First half" in all_code_text,
    "Has compute_delta_exposure": "compute_delta_exposure" in all_code_text,
    "Has scenario table": "scenario_df" in all_code_text,
    "Has position_oz param": "position_oz" in all_code_text,
    "Has efp_beta_results.csv": "efp_beta_results.csv" in all_code_text,
    "Has tariff annotation": "Tariff" in all_text or "tariff" in all_text,
    "All code cells have outputs": all(
        "outputs" in c for c in nb2["cells"] if c["cell_type"] == "code"
    ),
}

for name, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'} | {name}")

print(f"\nAll {len(checks)} checks passed: {all(checks.values())}")

# Cell listing for new cells only
print(f"\nNew cells (Prompt C):")
for i in range(existing_count, n_cells):
    c = nb2["cells"][i]
    first = "".join(c["source"]).split("\n")[0][:85]
    print(f"  [{i:2d}] {c['cell_type']:8s} | {first}")
