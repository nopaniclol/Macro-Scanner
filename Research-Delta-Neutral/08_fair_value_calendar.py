"""
08_fair_value_calendar.py
==========================
Rate sensitivity analysis using actual SOFR OIS 1m rate (not proxy).
Builds SOFR-implied fair value model for GC and SI calendar spreads,
then backtests z-score strategy on the RATE-STRIPPED RESIDUAL.

Key improvement over naive z-score:
  naive:      z = (spread - rolling_mean) / rolling_std
  fair_value: z = (spread - sofr_fair_value) / rolling_residual_std

Outputs:
    08_fair_value/rate_sensitivity_sofr.csv
    08_fair_value/fair_value_features_gc.csv
    08_fair_value/fair_value_features_si.csv
    08_fair_value/fv_backtest_results.csv
    08_fair_value/fv_yearly_breakdown.csv
    08_fair_value/fv_vs_naive_comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

sys.path.insert(0, str(Path(__file__).parent))
from backtest_engine import BacktestConfig, run_backtest, summarise_results, yearly_breakdown

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE     = Path(__file__).parent
CLEAN    = BASE / "02_data_clean" / "data_clean.csv"
GC_FEAT  = BASE / "03_features_curve" / "gold_curve_features.csv"
SI_FEAT  = BASE / "03_features_curve" / "silver_curve_features.csv"
OUT_DIR  = BASE / "08_fair_value"

SOFR_COL = "sofr_ois_1m"    # 1-month SOFR OIS — matches ~30-day spread tenor

# Backtest grid (same as other modules for comparability)
GRID = dict(
    entry_zs  = [1.0, 1.5, 2.0, 2.5],
    exit_zs   = [0.0, 0.5],
    max_holds = [5, 10, 20, 40],
    lb_resid  = [20, 40, 60],   # lookback for residual std only (level is model-implied)
)
COST_GC = 0.50   # bps
COST_SI = 0.50   # bps


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    base = pd.read_csv(CLEAN, parse_dates=["date"])
    gc   = pd.read_csv(GC_FEAT, parse_dates=["date"])[
        ["date", "cal_12_norm", "fly_123_norm", "cal_12", "fly_123"]
    ]
    si   = pd.read_csv(SI_FEAT, parse_dates=["date"])[
        ["date", "si_cal_12_norm", "si_fly_123_norm"]
    ]
    df = base.merge(gc, on="date").merge(si, on="date")
    df = df.dropna(subset=[SOFR_COL, "cal_12_norm", "si_cal_12_norm"])
    return df


# ---------------------------------------------------------------------------
# Full-sample OLS: spread ~ SOFR  (for understanding, not for signal)
# ---------------------------------------------------------------------------
def full_sample_ols(df: pd.DataFrame) -> dict:
    """Fit full-sample OLS for all four spreads vs SOFR. Returns coefficients."""
    results = {}
    for col in ["cal_12_norm", "si_cal_12_norm", "fly_123_norm", "si_fly_123_norm"]:
        y = df[col].values
        x = df[SOFR_COL].values
        res = OLS(y, add_constant(x)).fit()
        a, b = res.params[0], res.params[1]
        r2 = res.rsquared
        resid_std = res.resid.std()
        results[col] = dict(alpha=a, beta=b, r2=r2, resid_std=resid_std)
    return results


# ---------------------------------------------------------------------------
# Rolling OLS fair value: estimate α and β using only past data
# Produces point-in-time fair value without lookahead.
# ---------------------------------------------------------------------------
def rolling_fair_value(
    spread: pd.Series,
    sofr: pd.Series,
    window: int = 252,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Rolling OLS: spread_t = α_t + β_t × SOFR_t
    Uses only past `window` days of data → no lookahead.

    Returns:
        fair_value:  model-implied spread level
        residual:    actual - fair_value (the tradeable component)
        beta_series: rolling beta (SOFR sensitivity)
    """
    n = len(spread)
    fair_value  = pd.Series(np.nan, index=spread.index)
    residual    = pd.Series(np.nan, index=spread.index)
    beta_series = pd.Series(np.nan, index=spread.index)

    for i in range(window, n):
        y_win = spread.iloc[i - window:i].values
        x_win = sofr.iloc[i - window:i].values
        mask  = ~(np.isnan(y_win) | np.isnan(x_win))
        if mask.sum() < window // 2:
            continue
        X = add_constant(x_win[mask])
        try:
            res   = OLS(y_win[mask], X).fit()
            a, b  = res.params[0], res.params[1]
            fv    = a + b * sofr.iloc[i]
            fair_value.iloc[i]  = fv
            residual.iloc[i]    = spread.iloc[i] - fv
            beta_series.iloc[i] = b
        except Exception:
            pass

    return fair_value, residual, beta_series


# ---------------------------------------------------------------------------
# Build fair-value z-score
# ---------------------------------------------------------------------------
def fair_value_zscore(
    residual: pd.Series,
    resid_window: int,
) -> pd.Series:
    """
    Z-score of residual relative to its own rolling std.
    Mean of residual should be ~0 by construction (model removes level).
    """
    sig = residual.rolling(resid_window, min_periods=resid_window).std()
    return residual / sig.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Backtest grid on fair-value z-score
# ---------------------------------------------------------------------------
def run_fv_grid(
    dates: pd.Series,
    spread: pd.Series,
    residual: pd.Series,
    label: str,
    cost: float,
) -> pd.DataFrame:
    rows = []
    for lb in GRID["lb_resid"]:
        z_series = fair_value_zscore(residual, lb)
        for ez in GRID["entry_zs"]:
            for xz in GRID["exit_zs"]:
                if xz >= ez:
                    continue
                for mh in GRID["max_holds"]:
                    cfg = BacktestConfig(
                        entry_z=ez, exit_z=xz, max_hold=mh,
                        cost_per_trade=cost, lookback=lb,
                        label=f"{label}_fv_lb{lb}_ez{ez}_xz{xz}_mh{mh}",
                    )
                    res = run_backtest(dates, spread, z_series, cfg)
                    stats = summarise_results(res, cfg.label)
                    stats.update({"spread": label, "lb_resid": lb,
                                  "entry_z": ez, "exit_z": xz, "max_hold": mh,
                                  "model": "fair_value"})
                    rows.append(stats)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading data...")
    df = load_data()
    print(f"  Shape: {df.shape}  SOFR range: {df[SOFR_COL].min():.3f}% – {df[SOFR_COL].max():.3f}%")

    # ── Full-sample rate sensitivity ──────────────────────────────────────
    print("\n[2/6] Full-sample rate sensitivity (SOFR OIS 1m) ...")
    ols_results = full_sample_ols(df)

    print()
    print(f"  {'Spread':25s}  {'α (intercept)':>14}  {'β (per 1% SOFR)':>16}  "
          f"{'R² (%)':>8}  {'Resid std':>10}  {'100bps impact':>14}")
    print("  " + "-" * 95)
    sens_rows = []
    for col, r in ols_results.items():
        impact = r['beta'] * 1.0   # bps shift per 100bps SOFR move
        print(f"  {col:25s}  {r['alpha']:>14.2f}  {r['beta']:>16.2f}  "
              f"{r['r2']*100:>7.1f}%  {r['resid_std']:>10.2f} bps  {impact:>+13.1f} bps")
        sens_rows.append({"spread": col, **r, "impact_per_100bps": impact})

    pd.DataFrame(sens_rows).to_csv(OUT_DIR / "rate_sensitivity_sofr.csv", index=False)
    print(f"\n  Saved: rate_sensitivity_sofr.csv")

    # ── Daily change sensitivity ──────────────────────────────────────────
    print("\n  Daily change sensitivity (ΔSpread ~ ΔSOFR):")
    df["d_sofr"] = df[SOFR_COL].diff()
    print(f"  {'Spread':25s}  {'β daily':>10}  {'R² (%)':>8}")
    for col in ["cal_12_norm", "si_cal_12_norm", "fly_123_norm", "si_fly_123_norm"]:
        d_spread = df[col].diff()
        sub = pd.DataFrame({"ds": d_spread, "dr": df["d_sofr"]}).dropna()
        res = OLS(sub["ds"].values, add_constant(sub["dr"].values)).fit()
        print(f"  {col:25s}  {res.params[1]:>10.2f}  {res.rsquared*100:>7.2f}%")

    # ── Rolling fair value features ───────────────────────────────────────
    print("\n[3/6] Building rolling fair value (252-day window, no lookahead)...")

    fv_features = {}
    for col, label, cost in [
        ("cal_12_norm",    "GC_cal12", COST_GC),
        ("si_cal_12_norm", "SI_cal12", COST_SI),
        ("fly_123_norm",   "GC_fly",   COST_GC),
        ("si_fly_123_norm","SI_fly",   COST_SI),
    ]:
        fv, resid, beta = rolling_fair_value(df[col], df[SOFR_COL], window=252)
        fv_features[col] = {"fair_value": fv, "residual": resid, "beta": beta}

        resid_valid = resid.dropna()
        print(f"  {label:10s}  resid: mean={resid_valid.mean():+.2f}  std={resid_valid.std():.2f} bps  "
              f"half-life check (raw): {col}")

    # Save fair value features
    fv_df = df[["date", SOFR_COL, "cal_12_norm", "si_cal_12_norm",
                "fly_123_norm", "si_fly_123_norm"]].copy()
    for col in ["cal_12_norm", "si_cal_12_norm", "fly_123_norm", "si_fly_123_norm"]:
        fv_df[f"{col}_fv"]    = fv_features[col]["fair_value"]
        fv_df[f"{col}_resid"] = fv_features[col]["residual"]
        fv_df[f"{col}_beta"]  = fv_features[col]["beta"]

    fv_df.to_csv(OUT_DIR / "fair_value_features.csv", index=False)
    print(f"  Saved: fair_value_features.csv")

    # ── Backtest on fair-value z-scores ───────────────────────────────────
    print("\n[4/6] Backtesting fair-value z-score strategy...")

    all_grid = []
    best_rows = []
    yearly_rows = []

    for col, label, cost in [
        ("cal_12_norm",    "GC_cal12_fv", COST_GC),
        ("si_cal_12_norm", "SI_cal12_fv", COST_SI),
        ("fly_123_norm",   "GC_fly_fv",   COST_GC),
        ("si_fly_123_norm","SI_fly_fv",   COST_SI),
    ]:
        resid = fv_features[col]["residual"]
        grid_df = run_fv_grid(df["date"], df[col], resid, label, cost)
        all_grid.append(grid_df)

        valid = grid_df[grid_df["n_trades"] >= 10]
        if valid.empty:
            print(f"  {label}: no valid configs")
            continue

        best = valid.loc[valid["sharpe"].idxmax()]
        best_rows.append(best.to_dict())
        print(f"  {label:15s}  best: lb={int(best['lb_resid'])} ez={best['entry_z']} "
              f"xz={best['exit_z']} mh={int(best['max_hold'])} → "
              f"Sharpe={best['sharpe']:.2f}  n_trades={int(best['n_trades'])}  "
              f"hit_rate={best['hit_rate']:.0%}")

        # Yearly breakdown
        best_cfg = BacktestConfig(
            entry_z=float(best["entry_z"]), exit_z=float(best["exit_z"]),
            max_hold=int(best["max_hold"]), cost_per_trade=cost,
            lookback=int(best["lb_resid"]), label=label,
        )
        z_best = fair_value_zscore(resid, int(best["lb_resid"]))
        best_result = run_backtest(df["date"], df[col], z_best, best_cfg)
        yr_df = yearly_breakdown(best_result)
        yr_df["strategy"] = label
        yearly_rows.append(yr_df)

    # ── Save ──────────────────────────────────────────────────────────────
    print("\n[5/6] Saving outputs...")
    if all_grid:
        pd.concat(all_grid, ignore_index=True).to_csv(OUT_DIR / "fv_backtest_results.csv", index=False)
        print(f"  Saved fv_backtest_results.csv")
    if best_rows:
        pd.DataFrame(best_rows).to_csv(OUT_DIR / "fv_best_params.csv", index=False)
    if yearly_rows:
        pd.concat(yearly_rows, ignore_index=True).to_csv(OUT_DIR / "fv_yearly_breakdown.csv", index=False)
        print(f"  Saved fv_yearly_breakdown.csv")

    # ── Comparison: fair value vs naive z-score ───────────────────────────
    print("\n[6/6] Fair-value vs naive z-score comparison...")

    # Load naive best results
    naive_gc = pd.read_csv(BASE / "05_research_curve" / "curve_best_params.csv")
    naive_si = pd.read_csv(BASE / "05_research_curve" / "silver_best_params.csv")
    naive_all = pd.concat([naive_gc, naive_si], ignore_index=True)

    # Filter to normalised calendar spreads only
    naive_cal = naive_all[naive_all["spread_label"].str.contains("cal.*norm|norm.*cal", regex=True)]

    if best_rows:
        fv_cal = pd.DataFrame(best_rows)
        fv_cal = fv_cal[fv_cal["spread"].str.contains("cal12")]

        cmp_rows = []
        for naive_row in naive_cal.itertuples():
            metal = "GC" if "GC" in naive_row.spread_label or "gc" in str(getattr(naive_row, "spread", "")).lower() else "SI"
            fv_match = fv_cal[fv_cal["spread"].str.startswith(metal)]
            if fv_match.empty:
                continue
            fv_row = fv_match.iloc[0]
            cmp_rows.append({
                "metal": metal,
                "naive_sharpe":    naive_row.sharpe,
                "fv_sharpe":       fv_row["sharpe"],
                "sharpe_delta":    fv_row["sharpe"] - naive_row.sharpe,
                "naive_hit_rate":  naive_row.hit_rate,
                "fv_hit_rate":     fv_row["hit_rate"],
                "naive_trades":    naive_row.n_trades,
                "fv_trades":       fv_row["n_trades"],
                "naive_max_dd":    naive_row.max_dd,
                "fv_max_dd":       fv_row["max_dd"],
            })

        cmp_df = pd.DataFrame(cmp_rows)
        cmp_df.to_csv(OUT_DIR / "fv_vs_naive_comparison.csv", index=False)
        print()
        print(f"  {'Metal':6s}  {'Naive Sharpe':>13}  {'FV Sharpe':>10}  {'Delta':>7}  "
              f"{'Naive HR':>9}  {'FV HR':>6}  {'Naive DD':>9}  {'FV DD':>7}")
        print("  " + "-" * 80)
        for _, r in cmp_df.iterrows():
            flag = "↑ IMPROVED" if r["sharpe_delta"] > 0 else "↓ WORSE"
            print(f"  {r['metal']:6s}  {r['naive_sharpe']:>13.2f}  {r['fv_sharpe']:>10.2f}  "
                  f"{r['sharpe_delta']:>+7.2f}  {r['naive_hit_rate']:>9.0%}  "
                  f"{r['fv_hit_rate']:>6.0%}  {r['naive_max_dd']:>9.1f}  "
                  f"{r['fv_max_dd']:>7.1f}  {flag}")

    print("\n=== KEY FINDINGS ===")
    print(f"  Model: spread = α + β × SOFR_OIS_1m")
    print(f"  GC: R²={ols_results['cal_12_norm']['r2']*100:.0f}%  "
          f"β={ols_results['cal_12_norm']['beta']:.2f} bps/1%  "
          f"resid_std={ols_results['cal_12_norm']['resid_std']:.1f} bps")
    print(f"  SI: R²={ols_results['si_cal_12_norm']['r2']*100:.0f}%  "
          f"β={ols_results['si_cal_12_norm']['beta']:.2f} bps/1%  "
          f"resid_std={ols_results['si_cal_12_norm']['resid_std']:.1f} bps")
    print()
    print(f"  Fly R² vs SOFR: GC={ols_results['fly_123_norm']['r2']*100:.1f}%  "
          f"SI={ols_results['si_fly_123_norm']['r2']*100:.1f}%  ← near zero (carry-neutral)")


if __name__ == "__main__":
    main()
