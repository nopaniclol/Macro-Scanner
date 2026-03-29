"""
05b_backtest_silver_curve.py
============================
Run the silver curve RV backtest:
  - si_cal_12 (SI1-SI2 calendar spread)
  - si_cal_23 (SI2-SI3 calendar spread)
  - si_fly_123 (silver butterfly)
  - normalised versions in bps

Same parameter grid as gold curve backtest for direct comparability.

Outputs:
    05_research_curve/silver_grid_results.csv
    05_research_curve/silver_best_params.csv
    05_research_curve/silver_yearly_breakdown.csv
    05_research_curve/sensitivity_sharpe_si_fly_123.csv
    05_research_curve/sensitivity_sharpe_si_cal_12.csv
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
    "cost_raw": 0.005,   # $0.005/oz round-trip for silver (1 tick = $0.005/oz)
    "cost_bps": 0.50,    # same expressed in bps for normalised spreads
}

SPREADS_TO_TEST = [
    ("si_cal_12",      "SI1-SI2 calendar"),
    ("si_cal_23",      "SI2-SI3 calendar"),
    ("si_fly_123",     "SI butterfly"),
    ("si_cal_12_norm", "SI1-SI2 normalised (bps)"),
    ("si_fly_123_norm","SI butterfly normalised (bps)"),
]


# ---------------------------------------------------------------------------
# Helper: z-score map from pre-computed columns
# ---------------------------------------------------------------------------
def build_zscore_map(features: pd.DataFrame, spread_col: str) -> dict[int, pd.Series]:
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

    print("[1/4] Loading silver curve features...")
    features = pd.read_csv(FEAT_DIR / "silver_curve_features.csv", parse_dates=["date"])
    print(f"  Shape: {features.shape}, dates: {features['date'].min().date()} → {features['date'].max().date()}")

    all_grid_rows = []
    yearly_rows = []
    best_rows = []

    for spread_col, spread_label in SPREADS_TO_TEST:
        if spread_col not in features.columns:
            print(f"  SKIP {spread_col} — not in features")
            continue

        cost = GRID["cost_bps"] if "norm" in spread_col else GRID["cost_raw"]
        print(f"\n[2/4] Grid scan: {spread_label} (cost={cost}) ...")
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
            base_label=f"si_{spread_col}",
        )
        grid_df["spread"] = spread_col
        grid_df["spread_label"] = spread_label
        grid_df["metal"] = "silver"
        all_grid_rows.append(grid_df)

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

        # Yearly breakdown for best config
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
        yr_df["metal"] = "silver"
        yearly_rows.append(yr_df)

    print("\n[3/4] Saving outputs...")
    if all_grid_rows:
        full_grid = pd.concat(all_grid_rows, ignore_index=True)
        full_grid.to_csv(OUT_DIR / "silver_grid_results.csv", index=False)
        print(f"  Saved silver_grid_results.csv — {len(full_grid)} rows")

    if best_rows:
        best_df = pd.DataFrame(best_rows)
        best_df.to_csv(OUT_DIR / "silver_best_params.csv", index=False)
        print(f"  Saved silver_best_params.csv")

    if yearly_rows:
        yearly_df = pd.concat(yearly_rows, ignore_index=True)
        yearly_df.to_csv(OUT_DIR / "silver_yearly_breakdown.csv", index=False)
        print(f"  Saved silver_yearly_breakdown.csv")

    # Sensitivity heatmaps
    if all_grid_rows:
        full_grid_all = pd.concat(all_grid_rows, ignore_index=True)
        for sc in ["si_fly_123", "si_cal_12"]:
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

    print("\n[4/4] Silver summary table:")
    if best_rows:
        summary_cols = ["spread_label", "sharpe", "max_dd", "hit_rate",
                        "avg_hold_days", "n_trades", "lookback", "entry_z"]
        best_df_display = pd.DataFrame(best_rows)[summary_cols].sort_values("sharpe", ascending=False)
        print(best_df_display.to_string(index=False))


if __name__ == "__main__":
    main()
