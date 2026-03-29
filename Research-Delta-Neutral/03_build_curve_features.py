"""
03_build_curve_features.py
==========================
Build gold curve RV features: calendar spreads, fly, z-scores,
half-life estimates, and carry proxies.

Outputs:
    03_features_curve/gold_curve_features.csv
    03_features_curve/halflife_summary.csv
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

LOOKBACKS = [20, 60, 120]  # trading days


# ---------------------------------------------------------------------------
# Rolling z-score (no lookahead)
# ---------------------------------------------------------------------------
def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score using expanding-min(window) historical mean/std."""
    mu = s.rolling(window, min_periods=window).mean()
    sig = s.rolling(window, min_periods=window).std()
    return (s - mu) / sig.replace(0, np.nan)


def rolling_percentile(s: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank (0–100) using past window only."""
    return s.rolling(window, min_periods=window).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1] * 100),
        raw=False,
    )


# ---------------------------------------------------------------------------
# Half-life via Ornstein-Uhlenbeck (ADF-style OLS)
# ---------------------------------------------------------------------------
def estimate_halflife(s: pd.Series) -> float:
    """
    Estimate mean-reversion half-life via OLS:
        Δy_t = λ * y_{t-1} + ε
    half_life = -ln(2) / λ
    Returns np.nan if series is not mean-reverting.
    """
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
        return np.nan  # not mean-reverting
    return -np.log(2) / lam


# ---------------------------------------------------------------------------
# Build features
# ---------------------------------------------------------------------------
def build_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct all curve spread features and z-scores."""
    out = df[["date"]].copy()

    gc1 = df["gc_fut_front"]
    gc2 = df["gc_fut_second"]
    gc3 = df["gc_fut_third"]

    # --- Level spreads ---
    out["cal_12"] = gc1 - gc2          # front minus second (negative = contango)
    out["cal_23"] = gc2 - gc3          # second minus third
    out["fly_123"] = gc1 - 2 * gc2 + gc3  # butterfly

    # --- Price-normalised spreads (bps of spot price) ---
    spot = df["xauusd_spot"]
    out["cal_12_norm"] = out["cal_12"] / spot * 10_000  # bps
    out["cal_23_norm"] = out["cal_23"] / spot * 10_000
    out["fly_123_norm"] = out["fly_123"] / spot * 10_000

    # --- Roll yield proxy (EFP-like: front - spot) ---
    out["roll_yield_12"] = gc1 - spot   # basis: futures premium/discount to spot

    # --- Carry context: gold swap rates (annualised %) ---
    for col in ["xau_swap_1m", "xau_swap_3m", "xau_swap_6m", "xau_swap_1y"]:
        if col in df.columns:
            out[col] = df[col]

    # --- Z-scores for each lookback ---
    for lb in LOOKBACKS:
        for spread_col in ["cal_12", "cal_23", "fly_123",
                           "cal_12_norm", "cal_23_norm", "fly_123_norm"]:
            out[f"{spread_col}_z{lb}"] = rolling_zscore(out[spread_col], lb)

    # --- Returns of each leg (for hedge-ratio and PnL calcs) ---
    for col, name in [(gc1, "gc1_ret"), (gc2, "gc2_ret"), (gc3, "gc3_ret")]:
        out[name] = col.diff()

    # --- Spread daily change ---
    for col in ["cal_12", "cal_23", "fly_123"]:
        out[f"{col}_chg"] = out[col].diff()

    # --- Realised vol of spread (for sizing context) ---
    for lb in [20, 60]:
        for col in ["cal_12", "cal_23", "fly_123"]:
            out[f"{col}_vol{lb}"] = (
                out[f"{col}_chg"].rolling(lb, min_periods=lb).std() * np.sqrt(252)
            )

    return out


# ---------------------------------------------------------------------------
# Half-life summary
# ---------------------------------------------------------------------------
def compute_halflife_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Compute half-life for all spreads on full sample and sub-periods."""
    spread_cols = ["cal_12", "cal_23", "fly_123",
                   "cal_12_norm", "cal_23_norm", "fly_123_norm"]
    rows = []
    for col in spread_cols:
        s = features.set_index("date")[col].dropna()
        hl_full = estimate_halflife(s)

        # Sub-periods
        for start, end, label in [
            ("2018-01-01", "2020-01-01", "pre-COVID"),
            ("2020-01-01", "2022-01-01", "COVID"),
            ("2022-01-01", "2024-01-01", "hike-cycle"),
            ("2024-01-01", "2099-01-01", "recent"),
        ]:
            sub = s[start:end]
            hl_sub = estimate_halflife(sub) if len(sub) > 30 else np.nan
            rows.append({
                "spread": col,
                "period": label,
                "n_obs": len(sub),
                "halflife_days": round(hl_sub, 1) if not np.isnan(hl_sub) else "N/A (not MR)",
            })

        rows_full = {
            "spread": col,
            "period": "full",
            "n_obs": len(s),
            "halflife_days": round(hl_full, 1) if not np.isnan(hl_full) else "N/A (not MR)",
        }
        rows.insert(-4, rows_full)  # insert full before sub-periods

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading clean data...")
    df = pd.read_csv(CLEAN_DIR / "data_clean.csv", parse_dates=["date"])
    print(f"  Shape: {df.shape}")

    print("[2/3] Building curve features...")
    features = build_curve_features(df)

    print("[3/3] Saving outputs...")
    features.to_csv(OUT_DIR / "gold_curve_features.csv", index=False)
    print(f"  Saved: {OUT_DIR / 'gold_curve_features.csv'} — {len(features)} rows, {len(features.columns)} cols")

    # Half-lives
    hl_df = compute_halflife_summary(features)
    hl_df.to_csv(OUT_DIR / "halflife_summary.csv", index=False)
    print(f"  Saved: {OUT_DIR / 'halflife_summary.csv'}")

    # Quick summary
    print("\n=== CURVE FEATURE SUMMARY ===")
    for col in ["cal_12", "cal_23", "fly_123"]:
        s = features[col].dropna()
        hl = estimate_halflife(features.set_index("date")[col].dropna())
        hl_str = f"{hl:.1f}d" if not np.isnan(hl) else "not MR"
        print(f"  {col:15s}  mean={s.mean():+.3f}  std={s.std():.3f}  "
              f"min={s.min():+.2f}  max={s.max():+.2f}  half-life={hl_str}")

    print("\n=== HALF-LIFE TABLE (full sample) ===")
    full = hl_df[hl_df["period"] == "full"]
    for _, r in full.iterrows():
        print(f"  {r['spread']:20s}  {r['halflife_days']}")


if __name__ == "__main__":
    main()
