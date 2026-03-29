"""
05_backtest_curve.py
====================
Run the gold curve RV backtest:
  - cal_12 (GC1-GC2 calendar spread)
  - cal_23 (GC2-GC3 calendar spread)
  - fly_123 (butterfly)

Parameter grid: lookback × entry_z × exit_z × max_hold

Outputs:
    05_research_curve/curve_grid_results.csv
    05_research_curve/curve_best_params.csv
    05_research_curve/curve_yearly_breakdown.csv
    05_research_curve/curve_sensitivity_sharpe.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_engine import (
    BacktestConfig,
    run_backtest,
    summarise_results,
    yearly_breakdown,
    grid_scan,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
FEAT_DIR = BASE / "03_features_curve"
OUT_DIR = BASE / "05_research_curve"

GRID = {
    "lookbacks": [20, 60, 120],
    "entry_zs": [1.0, 1.5, 2.0, 2.5],
    "exit_zs": [0.0, 0.5],
    "max_holds": [5, 10, 20, 40],
    "cost": 0.5,  # spread units per one-way leg
}

SPREADS_TO_TEST = [
    ("cal_12",    "GC1-GC2 calendar"),
    ("cal_23",    "GC2-GC3 calendar"),
    ("fly_123",   "GC butterfly"),
    ("cal_12_norm", "GC1-GC2 normalised (bps)"),
    ("fly_123_norm", "GC butterfly normalised (bps)"),
]

# Cost in bps for normalised; in $ for raw spreads
COST_RAW = 0.50    # ~$0.50/oz round-trip (1 tick = $0.10/oz; ≈ 2-3 ticks RT)
COST_BPS = 0.50    # same in bps terms (roughly equivalent)


# ---------------------------------------------------------------------------
# Helper: build z-score map from feature file
# ---------------------------------------------------------------------------
def build_zscore_map(features: pd.DataFrame, spread_col: str) -> dict[int, pd.Series]:
    """Return dict {lookback: z_series} from pre-computed columns."""
    zmap = {}
    for lb in GRID["lookbacks"]:
        zcol = f"{spread_col}_z{lb}"
        if zcol in features.columns:
            zmap[lb] = features[zcol]
    return zmap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading curve features...")
    features = pd.read_csv(FEAT_DIR / "gold_curve_features.csv", parse_dates=["date"])
    print(f"  Shape: {features.shape}, dates: {features['date'].min().date()} → {features['date'].max().date()}")

    all_grid_rows = []
    yearly_rows = []
    best_rows = []

    for spread_col, spread_label in SPREADS_TO_TEST:
        if spread_col not in features.columns:
            print(f"  SKIP {spread_col} — not in features")
            continue

        print(f"\n[2/4] Grid scan: {spread_label} ...")
        cost = COST_BPS if "norm" in spread_col else COST_RAW
        zmap = build_zscore_map(features, spread_col)

        if not zmap:
            print(f"  No z-score columns found for {spread_col}")
            continue

        grid_df = grid_scan(
            features["date"],
            features[spread_col],
            zmap,
            entry_zs=GRID["entry_zs"],
            exit_zs=GRID["exit_zs"],
            max_holds=GRID["max_holds"],
            cost=cost,
            base_label=spread_col,
        )
        grid_df["spread"] = spread_col
        grid_df["spread_label"] = spread_label
        all_grid_rows.append(grid_df)

        # Best parameter set: highest Sharpe with n_trades > 10
        valid = grid_df[grid_df["n_trades"] >= 10].copy()
        if valid.empty:
            print(f"  No valid configs (n_trades >= 10) for {spread_col}")
            continue

        best = valid.loc[valid["sharpe"].idxmax()]
        best_rows.append(best.to_dict())

        print(f"  Best: lb={int(best['lookback'])} ez={best['entry_z']} "
              f"xz={best['exit_z']} mh={int(best['max_hold'])} → "
              f"Sharpe={best['sharpe']:.2f} n_trades={int(best['n_trades'])} "
              f"hit_rate={best['hit_rate']:.0%}")

        # Re-run best config to get yearly breakdown
        best_cfg = BacktestConfig(
            entry_z=float(best["entry_z"]),
            exit_z=float(best["exit_z"]),
            max_hold=int(best["max_hold"]),
            cost_per_trade=cost,
            lookback=int(best["lookback"]),
            label=spread_label,
        )
        z_series = zmap[int(best["lookback"])]
        best_result = run_backtest(features["date"], features[spread_col], z_series, best_cfg)
        yr_df = yearly_breakdown(best_result)
        yr_df["spread"] = spread_col
        yr_df["spread_label"] = spread_label
        yearly_rows.append(yr_df)

    print("\n[3/4] Saving outputs...")
    # Full grid
    if all_grid_rows:
        full_grid = pd.concat(all_grid_rows, ignore_index=True)
        full_grid.to_csv(OUT_DIR / "curve_grid_results.csv", index=False)
        print(f"  Saved curve_grid_results.csv — {len(full_grid)} rows")

    # Best params
    if best_rows:
        best_df = pd.DataFrame(best_rows)
        best_df.to_csv(OUT_DIR / "curve_best_params.csv", index=False)
        print(f"  Saved curve_best_params.csv")

    # Yearly
    if yearly_rows:
        yearly_df = pd.concat(yearly_rows, ignore_index=True)
        yearly_df.to_csv(OUT_DIR / "curve_yearly_breakdown.csv", index=False)
        print(f"  Saved curve_yearly_breakdown.csv")

    # Sensitivity: Sharpe by (lookback, entry_z) for fly_123 — key heatmap
    if all_grid_rows:
        full_grid_all = pd.concat(all_grid_rows, ignore_index=True)
        for sc in ["fly_123", "cal_12"]:
            sub = full_grid_all[full_grid_all["spread"] == sc]
            if sub.empty:
                continue
            pivot = (
                sub[sub["exit_z"] == 0.0]
                .groupby(["lookback", "entry_z"])["sharpe"]
                .mean()
                .unstack("entry_z")
                .round(2)
            )
            fname = f"sensitivity_sharpe_{sc}.csv"
            pivot.to_csv(OUT_DIR / fname)
            print(f"  Saved {fname}")

    print("\n[4/4] Summary table:")
    if best_rows:
        summary_cols = ["spread_label", "sharpe", "max_dd", "hit_rate",
                        "avg_hold_days", "n_trades", "lookback", "entry_z"]
        best_df_display = pd.DataFrame(best_rows)[summary_cols].sort_values("sharpe", ascending=False)
        print(best_df_display.to_string(index=False))


if __name__ == "__main__":
    main()
