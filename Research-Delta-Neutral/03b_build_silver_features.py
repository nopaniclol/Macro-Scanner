"""
03b_build_silver_features.py
=============================
Build silver curve RV features: calendar spreads, fly, z-scores,
half-life estimates, and carry proxies.

Mirrors 03_build_curve_features.py for gold.

Outputs:
    03_features_curve/silver_curve_features.csv
    03_features_curve/silver_halflife_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
CLEAN_DIR = BASE / "02_data_clean"
OUT_DIR = BASE / "03_features_curve"

LOOKBACKS = [20, 60, 120]


# ---------------------------------------------------------------------------
# Rolling z-score (no lookahead)
# ---------------------------------------------------------------------------
def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window).mean()
    sig = s.rolling(window, min_periods=window).std()
    return (s - mu) / sig.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Half-life estimate
# ---------------------------------------------------------------------------
def estimate_halflife(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 30:
        return np.nan
    y = s.values
    y_lag = y[:-1]
    dy = np.diff(y)
    X = add_constant(y_lag)
    res = OLS(dy, X).fit()
    lam = res.params[1]
    if lam >= 0:
        return np.nan
    return -np.log(2) / lam


# ---------------------------------------------------------------------------
# Build silver curve features
# ---------------------------------------------------------------------------
def build_silver_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["date"]].copy()

    si1 = df["si_fut_front"]
    si2 = df["si_fut_second"]
    si3 = df["si_fut_third"]
    spot = df["xagusd_spot"]

    # --- Level spreads ---
    out["si_cal_12"] = si1 - si2
    out["si_cal_23"] = si2 - si3
    out["si_fly_123"] = si1 - 2 * si2 + si3

    # --- Price-normalised spreads (bps of spot) ---
    out["si_cal_12_norm"] = out["si_cal_12"] / spot * 10_000
    out["si_cal_23_norm"] = out["si_cal_23"] / spot * 10_000
    out["si_fly_123_norm"] = out["si_fly_123"] / spot * 10_000

    # --- Roll yield proxy ---
    out["si_roll_yield"] = si1 - spot

    # --- Carry context: silver swap rates (if available) ---
    for col in ["xag_swap_1m", "xag_swap_3m", "xag_swap_6m", "xag_swap_1y"]:
        if col in df.columns:
            out[col] = df[col]

    # --- Z-scores for each lookback ---
    for lb in LOOKBACKS:
        for spread_col in [
            "si_cal_12", "si_cal_23", "si_fly_123",
            "si_cal_12_norm", "si_cal_23_norm", "si_fly_123_norm",
        ]:
            out[f"{spread_col}_z{lb}"] = rolling_zscore(out[spread_col], lb)

    # --- Daily PnL inputs ---
    for col, name in [(si1, "si1_ret"), (si2, "si2_ret"), (si3, "si3_ret")]:
        out[name] = col.diff()

    for col in ["si_cal_12", "si_cal_23", "si_fly_123"]:
        out[f"{col}_chg"] = out[col].diff()

    # --- Realised vol of spread ---
    for lb in [20, 60]:
        for col in ["si_cal_12", "si_cal_23", "si_fly_123"]:
            out[f"{col}_vol{lb}"] = (
                out[f"{col}_chg"].rolling(lb, min_periods=lb).std() * np.sqrt(252)
            )

    return out


# ---------------------------------------------------------------------------
# Half-life summary
# ---------------------------------------------------------------------------
def compute_halflife_summary(features: pd.DataFrame) -> pd.DataFrame:
    spread_cols = [
        "si_cal_12", "si_cal_23", "si_fly_123",
        "si_cal_12_norm", "si_cal_23_norm", "si_fly_123_norm",
    ]
    rows = []
    for col in spread_cols:
        if col not in features.columns:
            continue
        s = features.set_index("date")[col].dropna()
        hl_full = estimate_halflife(s)
        rows.append({
            "spread": col,
            "period": "full",
            "n_obs": len(s),
            "halflife_days": round(hl_full, 1) if not np.isnan(hl_full) else "N/A (not MR)",
        })

        for start, end, label in [
            ("2018-01-01", "2020-01-01", "pre-COVID"),
            ("2020-01-01", "2022-01-01", "COVID"),
            ("2022-01-01", "2024-01-01", "hike-cycle"),
            ("2024-01-01", "2099-01-01", "recent"),
        ]:
            sub = s[start:end]
            hl = estimate_halflife(sub) if len(sub) > 30 else np.nan
            rows.append({
                "spread": col,
                "period": label,
                "n_obs": len(sub),
                "halflife_days": round(hl, 1) if not np.isnan(hl) else "N/A (not MR)",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading clean data...")
    df = pd.read_csv(CLEAN_DIR / "data_clean.csv", parse_dates=["date"])
    print(f"  Shape: {df.shape}")

    print("[2/3] Building silver curve features...")
    features = build_silver_features(df)
    print(f"  Features shape: {features.shape}")

    print("[3/3] Saving outputs...")
    features.to_csv(OUT_DIR / "silver_curve_features.csv", index=False)
    print(f"  Saved: {OUT_DIR / 'silver_curve_features.csv'} — {len(features)} rows, {len(features.columns)} cols")

    hl_df = compute_halflife_summary(features)
    hl_df.to_csv(OUT_DIR / "silver_halflife_summary.csv", index=False)
    print(f"  Saved: {OUT_DIR / 'silver_halflife_summary.csv'}")

    print("\n=== SILVER CURVE FEATURE SUMMARY ===")
    for col in ["si_cal_12", "si_cal_23", "si_fly_123",
                "si_cal_12_norm", "si_cal_23_norm", "si_fly_123_norm"]:
        if col not in features.columns:
            continue
        s = features[col].dropna()
        hl = estimate_halflife(features.set_index("date")[col].dropna())
        hl_str = f"{hl:.1f}d" if not np.isnan(hl) else "not MR"
        print(f"  {col:22s}  mean={s.mean():+.4f}  std={s.std():.4f}  "
              f"min={s.min():+.4f}  max={s.max():+.4f}  half-life={hl_str}")

    print("\n=== SILVER HALF-LIFE (full sample) ===")
    full = hl_df[hl_df["period"] == "full"]
    for _, r in full.iterrows():
        print(f"  {r['spread']:25s}  {r['halflife_days']}")

    print("\n=== COMPARISON: GOLD vs SILVER (full sample) ===")
    gc_hl = {
        "cal_12": 408.5, "cal_23": "not MR", "fly_123": 6.8,
        "cal_12_norm": 63.9, "fly_123_norm": 5.7,
    }
    si_full = {r["spread"]: r["halflife_days"] for _, r in full.iterrows()}
    print(f"  {'Spread':20s}  {'Gold':>12s}  {'Silver':>12s}")
    print(f"  {'-'*46}")
    for gc_key, si_key in [
        ("cal_12", "si_cal_12"),
        ("cal_23", "si_cal_23"),
        ("fly_123", "si_fly_123"),
        ("cal_12_norm", "si_cal_12_norm"),
        ("fly_123_norm", "si_fly_123_norm"),
    ]:
        gc_v = gc_hl.get(gc_key, "—")
        si_v = si_full.get(si_key, "—")
        print(f"  {gc_key:20s}  {str(gc_v):>12s}  {str(si_v):>12s}")


if __name__ == "__main__":
    main()
