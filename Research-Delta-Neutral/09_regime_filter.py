"""
09_regime_filter.py
===================
SOFR regime filter applied to gold/silver calendar spread strategies.

─── Why two signals? ────────────────────────────────────────────────────────
The 2022 losses occurred because a sustained SOFR rate-hike cycle shifted the
structural fair value of the calendar spread, making the rolling mean an
invalid anchor for mean-reversion.  Two complementary SOFR signals detect this:

  Signal 1 — OIS Momentum  (sofr_ois_1m, 20-day change)
      Uses: SOFR OIS 1-month (the SAME rate in the cost-of-carry formula,
            R²=85% with GC cal spread).  Measures how fast rates are moving NOW.
      Advantage: available for full 2018-2026 sample, directly tied to the
                 spread fair value driver.

  Signal 2 — Futures Curve Slope  (sofr_fut_front vs sofr_fut_fourth)
      Uses: SOFR futures prices.  Implied rate = 100 - price.
            Slope = implied_rate(4th contract) - implied_rate(front contract).
            Positive slope → market pricing in further rate hikes ahead.
      Advantage: FORWARD-LOOKING — reacts to rate expectations before they
                 materialise in OIS.  Leads the OIS momentum signal.
      Note: when futures data is unavailable (NaN), slope filter defaults to ON.

  Why NOT the Fed Dot Plot?
      Updated only quarterly, not machine-readable, misses intra-quarter
      surprises.  SOFR futures already price in consensus rate expectations
      continuously and react instantly to data (CPI, FOMC statements, etc.).

─── Combined regime gate (OR logic — conservative) ─────────────────────────
  Regime OFF → no new entries, existing positions run to natural exit when:
    sofr_ois_1m.diff(20) > MOM_THRESH (30 bps)   [rates hiking now]
    OR implied_rate(4th) - implied_rate(front) > SLOPE_THRESH (80 bps)  [hikes priced ahead]

─── Applied to ──────────────────────────────────────────────────────────────
  Best-param run for all 4 strategies × 2 signal models (naive z-score and
  fair-value z-score), comparing with/without regime filter.

Outputs:
    09_regime/regime_signals.csv              — daily regime on/off + both signals
    09_regime/regime_metrics_comparison.csv   — strategy metrics with/without filter
    09_regime/regime_yearly_comparison.csv    — year-by-year Sharpe comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_engine import BacktestConfig, run_backtest, summarise_results, yearly_breakdown

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE    = Path(__file__).parent
CLEAN   = BASE / "02_data_clean" / "data_clean.csv"
GC_FEAT = BASE / "03_features_curve" / "gold_curve_features.csv"
SI_FEAT = BASE / "03_features_curve" / "silver_curve_features.csv"
FV_FEAT = BASE / "08_fair_value" / "fair_value_features.csv"
OUT_DIR = BASE / "09_regime"

# ---------------------------------------------------------------------------
# Regime thresholds
# ---------------------------------------------------------------------------
MOM_THRESH   = 0.30   # 30 bps in 20 trading days  (sofr_ois_1m is in %, so 0.30 = 30 bps)
SLOPE_THRESH = 0.80   # 80 bps upward futures curve slope (also in % units)
MOM_WINDOW   = 20     # lookback days for momentum signal

# ---------------------------------------------------------------------------
# Best params from previous research
# (naive: from 05_research_curve; FV: from 08_fair_value)
# ---------------------------------------------------------------------------
STRATEGIES = [
    # label, spread_col, z_col (None=compute from FV resid), lb, ez, xz, mh, cost, model, use_regime
    # use_regime=True  → SOFR filter applied  (GC cal12 only: R²=85%, loses in rate-hike cycles)
    # use_regime=False → no filter            (SI and flies: silver decouples from SOFR in hike cycles)
    dict(label="GC_cal12_naive", spread_col="cal_12_norm",    z_col="cal_12_norm_z60",
         lb=60,  ez=2.5, xz=0.5, mh=20, cost=0.50, model="naive", use_regime=True),
    dict(label="SI_cal12_naive", spread_col="si_cal_12_norm", z_col="si_cal_12_norm_z60",
         lb=60,  ez=1.5, xz=0.0, mh=40, cost=0.50, model="naive", use_regime=False),
    dict(label="GC_fly_naive",   spread_col="fly_123_norm",   z_col="fly_123_norm_z120",
         lb=120, ez=1.0, xz=0.0, mh=40, cost=0.50, model="naive", use_regime=False),
    dict(label="SI_fly_naive",   spread_col="si_fly_123_norm",z_col="si_fly_123_norm_z120",
         lb=120, ez=1.0, xz=0.5, mh=40, cost=0.50, model="naive", use_regime=False),
    # FV strategies — z computed from residual column in fair_value_features.csv
    dict(label="GC_cal12_fv",    spread_col="cal_12_norm",    z_col=None,
         resid_col="cal_12_norm_resid",
         lb=40,  ez=1.0, xz=0.0, mh=40, cost=0.50, model="fv",    use_regime=True),
    dict(label="SI_cal12_fv",    spread_col="si_cal_12_norm", z_col=None,
         resid_col="si_cal_12_norm_resid",
         lb=40,  ez=2.5, xz=0.5, mh=40, cost=0.50, model="fv",    use_regime=False),
    dict(label="GC_fly_fv",      spread_col="fly_123_norm",   z_col=None,
         resid_col="fly_123_norm_resid",
         lb=20,  ez=2.5, xz=0.0, mh=40, cost=0.50, model="fv",    use_regime=False),
    dict(label="SI_fly_fv",      spread_col="si_fly_123_norm",z_col=None,
         resid_col="si_fly_123_norm_resid",
         lb=60,  ez=2.5, xz=0.0, mh=40, cost=0.50, model="fv",    use_regime=False),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fv_zscore(residual: pd.Series, window: int) -> pd.Series:
    """Rolling std of residual → z = residual / rolling_std (mean ~0 by construction)."""
    sig = residual.rolling(window, min_periods=window).std()
    return residual / sig.replace(0, np.nan)


def build_regime_mask(
    sofr_ois: pd.Series,
    fut_front: pd.Series,
    fut_fourth: pd.Series,
    mom_thresh: float = MOM_THRESH,
    slope_thresh: float = SLOPE_THRESH,
    mom_window: int = MOM_WINDOW,
) -> pd.Series:
    """
    Build daily boolean regime mask (True = regime ON, entries allowed).

    Signal 1 — OIS momentum:
        mom = sofr_ois_1m.diff(mom_window)   [in %, so 0.30 = 30 bps]
        Off when mom > mom_thresh

    Signal 2 — Futures curve slope:
        rate_front  = 100 - sofr_fut_front    [convert price → rate]
        rate_fourth = 100 - sofr_fut_fourth
        slope = rate_fourth - rate_front      [positive = market pricing hikes]
        Off when slope > slope_thresh
        When futures data is NaN → slope filter defaults to ON (don't penalise
        for missing data; OIS momentum still applies).
    """
    mom = sofr_ois.diff(mom_window)

    rate_front  = 100 - fut_front
    rate_fourth = 100 - fut_fourth
    slope = rate_fourth - rate_front

    # Individual gates (True = OK to trade)
    mom_ok   = (mom <= mom_thresh) | mom.isna()
    slope_ok = (slope <= slope_thresh) | slope.isna()   # NaN → OK (default to ON)

    return (mom_ok & slope_ok).fillna(True)             # any remaining NaN → ON


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("[1/5] Loading data...")
    base = pd.read_csv(CLEAN, parse_dates=["date"])
    gc   = pd.read_csv(GC_FEAT, parse_dates=["date"])
    si   = pd.read_csv(SI_FEAT, parse_dates=["date"])
    fv   = pd.read_csv(FV_FEAT, parse_dates=["date"])

    # Merge all features on date
    df = base.merge(gc, on="date", suffixes=("", "_gc"))
    df = df.merge(si, on="date", suffixes=("", "_si"))
    df = df.merge(fv[["date", "cal_12_norm_resid", "si_cal_12_norm_resid",
                       "fly_123_norm_resid", "si_fly_123_norm_resid"]],
                  on="date", how="left")

    print(f"  Rows: {len(df)}  Date range: {df['date'].min().date()} – {df['date'].max().date()}")

    # ── Build regime signals ─────────────────────────────────────────────────
    print("\n[2/5] Building SOFR regime signals...")

    sofr_ois   = df["sofr_ois_1m"]
    fut_front  = df["sofr_fut_front"]
    fut_fourth = df["sofr_fut_fourth"]

    mom_20d = sofr_ois.diff(MOM_WINDOW)

    rate_front  = 100 - fut_front
    rate_fourth = 100 - fut_fourth
    slope       = rate_fourth - rate_front

    regime_mask = build_regime_mask(sofr_ois, fut_front, fut_fourth)

    # Signal diagnostics
    n_total   = len(df)
    n_off     = (~regime_mask).sum()
    n_mom_off = (mom_20d > MOM_THRESH).sum()
    n_slp_off = (slope > SLOPE_THRESH).dropna().sum()

    print(f"\n  Regime signal thresholds: OIS momentum >{MOM_THRESH*100:.0f} bps/20d  |  "
          f"Futures slope >{SLOPE_THRESH*100:.0f} bps")
    print(f"  {'Days regime ON':30s}: {n_total - n_off:,}  ({(n_total-n_off)/n_total*100:.0f}%)")
    print(f"  {'Days regime OFF':30s}: {n_off:,}  ({n_off/n_total*100:.0f}%)")
    print(f"  {'  → triggered by OIS momentum':30s}: {n_mom_off:,} days")
    print(f"  {'  → triggered by futures slope':30s}: {n_slp_off:,} days")

    # Year-by-year regime breakdown
    df_reg = df.assign(
        regime_on=regime_mask,
        mom_20d=mom_20d,
        futures_slope=slope,
    )
    print(f"\n  {'Year':6s}  {'Days OFF':>9}  {'% OFF':>7}  {'Avg OIS Chg (bps)':>19}  {'Avg Slope (bps)':>16}")
    print("  " + "-" * 65)
    for yr, g in df_reg.assign(year=df["date"].dt.year).groupby("year"):
        off_pct = (~g["regime_on"]).mean() * 100
        avg_mom = g["mom_20d"].mean() * 100   # convert to bps
        avg_slp = g["futures_slope"].mean() * 100 if g["futures_slope"].notna().any() else float("nan")
        print(f"  {yr:6d}  {(~g['regime_on']).sum():>9,}  {off_pct:>6.0f}%  "
              f"{avg_mom:>+19.0f}  {avg_slp:>+16.0f}")

    # Save regime signals
    sig_df = df[["date"]].copy()
    sig_df["sofr_ois_1m"]      = sofr_ois.values
    sig_df["mom_20d_bps"]      = mom_20d.values * 100
    sig_df["fut_front_rate"]   = rate_front.values
    sig_df["fut_fourth_rate"]  = rate_fourth.values
    sig_df["futures_slope_bps"]= slope.values * 100
    sig_df["regime_on"]        = regime_mask.values
    sig_df.to_csv(OUT_DIR / "regime_signals.csv", index=False)
    print(f"\n  Saved: regime_signals.csv")

    # ── Run all strategies with and without regime filter ─────────────────────
    print("\n[3/5] Running backtests (with and without regime filter)...")

    metric_rows  = []
    yearly_rows  = []

    for s in STRATEGIES:
        spread = df[s["spread_col"]].copy()

        # Build z-score
        if s["model"] == "naive":
            zscore = df[s["z_col"]].copy()
        else:
            resid  = df[s["resid_col"]].copy()
            zscore = fv_zscore(resid, s["lb"])

        cfg = BacktestConfig(
            entry_z=s["ez"], exit_z=s["xz"], max_hold=s["mh"],
            cost_per_trade=s["cost"], lookback=s["lb"], label=s["label"],
        )

        # Apply regime mask only where research shows it helps (GC cal12 only)
        mask = regime_mask if s["use_regime"] else None
        tag  = "SOFR_filter" if s["use_regime"] else "no_filter"

        res   = run_backtest(df["date"], spread, zscore, cfg, regime_mask=mask)
        stats = summarise_results(res, label=s["label"])

        stats.update({
            "strategy":   s["label"],
            "model":      s["model"],
            "filter":     tag,
            "spread":     s["spread_col"],
            "use_regime": s["use_regime"],
        })
        metric_rows.append(stats)

        yr = yearly_breakdown(res)
        yr["strategy"] = s["label"]
        yr["filter"]   = tag
        yr["model"]    = s["model"]
        yearly_rows.append(yr)

    # ── Print comparison ──────────────────────────────────────────────────────
    print("\n[4/5] Results comparison...")

    mdf = pd.DataFrame(metric_rows)

    print()
    print("=" * 115)
    print("  FINAL STRATEGY METRICS  (GC cal12: SOFR regime filter ON | all others: no filter)")
    print("=" * 115)
    hdr = (f"  {'Strategy':22s}  {'Model':7s}  {'Filter':13s}  "
           f"{'Sharpe':>7}  {'Ann Ret':>8}  {'Max DD':>8}  "
           f"{'Trades':>7}  {'Hit %':>6}  {'Avg Hold':>9}")
    print(hdr)
    print("  " + "-" * 110)

    for s in STRATEGIES:
        row = mdf[mdf["strategy"] == s["label"]].iloc[0]
        print(f"  {s['label']:22s}  {row['model']:7s}  {row['filter']:13s}  "
              f"{row['sharpe']:>7.3f}  {row['ann_return']:>8.2f}  "
              f"{row['max_dd']:>8.2f}  {int(row['n_trades']):>7d}  "
              f"{row['hit_rate']:>5.0%}  {row['avg_hold_days']:>9.1f}")

    # ── Yearly breakdown ──────────────────────────────────────────────────────
    print()
    print("=" * 95)
    print("  YEARLY SHARPE  (GC cal12 with SOFR filter | SI cal12 no filter | flies no filter)")
    print("=" * 95)

    ydf = pd.concat(yearly_rows, ignore_index=True)

    col_strats = [
        ("GC_cal12_naive", "GC cal12"),
        ("GC_cal12_fv",    "GC cal12 FV"),
        ("SI_cal12_naive", "SI cal12"),
        ("SI_cal12_fv",    "SI cal12 FV"),
        ("GC_fly_naive",   "GC fly"),
        ("SI_fly_naive",   "SI fly"),
    ]

    years = sorted(ydf["year"].unique())
    col_w = 12

    header_cols = [f"{lbl:>{col_w}}" for _, lbl in col_strats]
    print(f"\n  {'Year':6s}" + "".join(header_cols))
    print("  " + "-" * (6 + col_w * len(col_strats) + 2))

    for yr in years:
        row_vals = []
        for strat, _ in col_strats:
            sub = ydf[(ydf["year"] == yr) & (ydf["strategy"] == strat)]
            val = sub["sharpe"].values[0] if len(sub) else float("nan")
            row_vals.append(f"{val:>{col_w}.2f}" if not np.isnan(val) else f"{'—':>{col_w}}")
        print(f"  {yr:6d}" + "".join(row_vals))

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n[5/5] Saving outputs...")

    mdf.to_csv(OUT_DIR / "regime_metrics_comparison.csv", index=False)
    print("  Saved: regime_metrics_comparison.csv")

    pd.concat(yearly_rows, ignore_index=True).to_csv(
        OUT_DIR / "regime_yearly_comparison.csv", index=False
    )
    print("  Saved: regime_yearly_comparison.csv")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=== FINAL RESULTS SUMMARY ===")
    print("  (GC cal12: SOFR regime filter applied | all others: unfiltered)")
    print()
    for s in STRATEGIES:
        row = mdf[mdf["strategy"] == s["label"]].iloc[0]
        regime_note = " ← SOFR filter ON" if s["use_regime"] else ""
        print(f"  {s['label']:22s}  Sharpe={row['sharpe']:.3f}  "
              f"AnnRet={row['ann_return']:.1f} bps  MaxDD={row['max_dd']:.1f}  "
              f"Trades={int(row['n_trades'])}  HitRate={row['hit_rate']:.0%}{regime_note}")


if __name__ == "__main__":
    main()
