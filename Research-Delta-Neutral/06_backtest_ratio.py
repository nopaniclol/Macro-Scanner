"""
06_backtest_ratio.py
====================
Run the gold-silver ratio RV backtest:
  - gs_ratio_fut (raw ratio)
  - gs_log_ratio (log ratio — more stationary)

Sizing is notional-neutral via equal %-weight (long GC% / short SI%).

Outputs:
    06_research_ratio/ratio_grid_results.csv
    06_research_ratio/ratio_best_params.csv
    06_research_ratio/ratio_yearly_breakdown.csv
    06_research_ratio/sensitivity_sharpe_gs_log_ratio.csv
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
FEAT_DIR = BASE / "04_features_ratio"
OUT_DIR = BASE / "06_research_ratio"

GRID = {
    "lookbacks": [20, 60, 120],
    "entry_zs": [1.0, 1.5, 2.0, 2.5],
    "exit_zs": [0.0, 0.5],
    "max_holds": [5, 10, 20, 40],
    "cost": 0.002,   # ~20bps round-trip for ratio (both legs combined)
}

SPREADS_TO_TEST = [
    ("gs_ratio_fut",  "GS raw ratio"),
    ("gs_log_ratio",  "GS log ratio (preferred)"),
]


# ---------------------------------------------------------------------------
# Build z-score map
# ---------------------------------------------------------------------------
def build_zscore_map(features: pd.DataFrame, spread_col: str) -> dict[int, pd.Series]:
    zmap = {}
    for lb in GRID["lookbacks"]:
        zcol = f"{spread_col}_z{lb}"
        if zcol in features.columns:
            zmap[lb] = features[zcol]
    return zmap


# ---------------------------------------------------------------------------
# Convert ratio PnL to % terms for comparability
# ---------------------------------------------------------------------------
def ratio_pnl_to_pct(ratio_series: pd.Series) -> pd.Series:
    """Daily change in log ratio ≈ daily spread return in % terms."""
    return ratio_series.diff()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading ratio features...")
    features = pd.read_csv(FEAT_DIR / "gold_silver_ratio_features.csv", parse_dates=["date"])
    print(f"  Shape: {features.shape}")

    all_grid_rows = []
    yearly_rows = []
    best_rows = []

    for spread_col, spread_label in SPREADS_TO_TEST:
        if spread_col not in features.columns:
            print(f"  SKIP {spread_col}")
            continue

        print(f"\n[2/4] Grid scan: {spread_label} ...")
        zmap = build_zscore_map(features, spread_col)

        if not zmap:
            print(f"  No z-score cols for {spread_col}")
            continue

        # Use log-ratio changes as spread for PnL if log_ratio; else raw diff
        if spread_col == "gs_log_ratio":
            pnl_series = features["gs_log_ratio"]   # log diffs ≈ % returns
        else:
            pnl_series = features[spread_col]

        grid_df = grid_scan(
            features["date"],
            pnl_series,
            zmap,
            entry_zs=GRID["entry_zs"],
            exit_zs=GRID["exit_zs"],
            max_holds=GRID["max_holds"],
            cost=GRID["cost"],
            base_label=spread_col,
        )
        grid_df["spread"] = spread_col
        grid_df["spread_label"] = spread_label
        all_grid_rows.append(grid_df)

        valid = grid_df[grid_df["n_trades"] >= 10].copy()
        if valid.empty:
            print(f"  No valid configs for {spread_col}")
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
            cost_per_trade=GRID["cost"],
            lookback=int(best["lookback"]),
            label=spread_label,
        )
        z_series = zmap[int(best["lookback"])]
        best_result = run_backtest(features["date"], pnl_series, z_series, best_cfg)
        yr_df = yearly_breakdown(best_result)
        yr_df["spread"] = spread_col
        yr_df["spread_label"] = spread_label
        yearly_rows.append(yr_df)

    print("\n[3/4] Saving outputs...")
    if all_grid_rows:
        full_grid = pd.concat(all_grid_rows, ignore_index=True)
        full_grid.to_csv(OUT_DIR / "ratio_grid_results.csv", index=False)
        print(f"  Saved ratio_grid_results.csv — {len(full_grid)} rows")

    if best_rows:
        best_df = pd.DataFrame(best_rows)
        best_df.to_csv(OUT_DIR / "ratio_best_params.csv", index=False)
        print(f"  Saved ratio_best_params.csv")

    if yearly_rows:
        yearly_df = pd.concat(yearly_rows, ignore_index=True)
        yearly_df.to_csv(OUT_DIR / "ratio_yearly_breakdown.csv", index=False)
        print(f"  Saved ratio_yearly_breakdown.csv")

    # Sensitivity heatmap for log ratio
    if all_grid_rows:
        full_grid_all = pd.concat(all_grid_rows, ignore_index=True)
        sub = full_grid_all[full_grid_all["spread"] == "gs_log_ratio"]
        if not sub.empty:
            pivot = (
                sub[sub["exit_z"] == 0.0]
                .groupby(["lookback", "entry_z"])["sharpe"]
                .mean()
                .unstack("entry_z")
                .round(2)
            )
            pivot.to_csv(OUT_DIR / "sensitivity_sharpe_gs_log_ratio.csv")
            print(f"  Saved sensitivity_sharpe_gs_log_ratio.csv")

    print("\n[4/4] Summary:")
    if best_rows:
        summary_cols = ["spread_label", "sharpe", "max_dd", "hit_rate",
                        "avg_hold_days", "n_trades", "lookback", "entry_z"]
        best_df_display = pd.DataFrame(best_rows)[summary_cols].sort_values("sharpe", ascending=False)
        print(best_df_display.to_string(index=False))


if __name__ == "__main__":
    main()
