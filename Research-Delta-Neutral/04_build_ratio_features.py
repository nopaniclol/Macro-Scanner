"""
04_build_ratio_features.py
==========================
Build gold-silver ratio RV features: raw ratio, notional-neutral
hedge ratios, z-scores, carry context.

Outputs:
    04_features_ratio/gold_silver_ratio_features.csv
    04_features_ratio/ratio_halflife_summary.csv
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
OUT_DIR = BASE / "04_features_ratio"

LOOKBACKS = [20, 60, 120]
ROLL_BETA_WINDOW = 60  # days for rolling beta hedge ratio
ROLL_VOL_WINDOW = 60   # days for rolling vol ratio


# ---------------------------------------------------------------------------
# Rolling z-score
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
# Rolling beta (GC returns ~ beta * SI returns)
# ---------------------------------------------------------------------------
def rolling_beta(gc_ret: pd.Series, si_ret: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta: gc_ret = alpha + beta * si_ret."""
    beta = pd.Series(np.nan, index=gc_ret.index)
    for i in range(window, len(gc_ret)):
        y = gc_ret.iloc[i - window:i].values
        x = si_ret.iloc[i - window:i].values
        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < window // 2:
            continue
        X = add_constant(x[mask])
        try:
            res = OLS(y[mask], X).fit()
            beta.iloc[i] = res.params[1]
        except Exception:
            pass
    return beta


# ---------------------------------------------------------------------------
# Build features
# ---------------------------------------------------------------------------
def build_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["date"]].copy()

    gc = df["gc_fut_front"]
    si = df["si_fut_front"]
    spot_gc = df["xauusd_spot"]
    spot_si = df["xagusd_spot"]

    # --- Raw price ratio ---
    out["gs_ratio_fut"] = gc / si
    out["gs_ratio_spot"] = spot_gc / spot_si

    # --- Spread as log-ratio (more stationary) ---
    out["gs_log_ratio"] = np.log(gc / si)

    # --- Daily returns ---
    out["gc_ret"] = gc.diff()
    out["si_ret"] = si.diff()
    out["gc_pct"] = gc.pct_change()
    out["si_pct"] = si.pct_change()

    # --- Hedge ratio methods ---

    # 1. Fixed 1:50 (historical ~ oz ratio; GC is 100oz, SI is 5000oz contract)
    # Note: 1 GC contract = 100 troy oz gold; 1 SI contract = 5000 troy oz silver
    # Notional: GC_price * 100 vs SI_price * 5000
    # Notional-neutral contracts: n_SI / n_GC = (GC_price * 100) / (SI_price * 5000)
    # = GC_price / (SI_price * 50) = ratio / 50
    out["notional_hedge_ratio"] = gc / (si * 50)  # SI contracts per 1 GC contract

    # 2. Fixed 1:ratio (price-only, simpler)
    out["price_hedge_ratio"] = gc / si

    # 3. Rolling 60d vol ratio (sigma_GC / sigma_SI) — vol parity
    gc_vol = out["gc_pct"].rolling(ROLL_VOL_WINDOW, min_periods=20).std()
    si_vol = out["si_pct"].rolling(ROLL_VOL_WINDOW, min_periods=20).std()
    out["vol_hedge_ratio"] = gc_vol / si_vol * (gc / si)  # in price terms

    # 4. Rolling 60d beta
    out["beta_hedge_ratio"] = rolling_beta(out["gc_ret"], out["si_ret"], ROLL_BETA_WINDOW)

    # --- Hedged spread (log-ratio is the simplest notional-neutral spread) ---
    # The log-ratio is the cleanest: long GC / short SI ratio
    # Spread PnL proxy when long GC / short SI:
    # pnl = gc_pct - si_pct (equal % exposure)
    out["spread_pct"] = out["gc_pct"] - out["si_pct"]  # equal % weight
    out["spread_log"] = out["gs_log_ratio"].diff()  # equivalent

    # --- Z-scores for all lookbacks ---
    for lb in LOOKBACKS:
        out[f"gs_ratio_z{lb}"] = rolling_zscore(out["gs_ratio_fut"], lb)
        out[f"gs_ratio_fut_z{lb}"] = out[f"gs_ratio_z{lb}"]   # alias for backtest column lookup
        out[f"gs_log_ratio_z{lb}"] = rolling_zscore(out["gs_log_ratio"], lb)

    # --- Carry context: gold vs silver swap differential ---
    if "xau_swap_1m" in df.columns and "xag_swap_1m" in df.columns:
        out["carry_diff_1m"] = df["xau_swap_1m"] - df["xag_swap_1m"]
    if "xau_swap_3m" in df.columns and "xag_swap_3m" in df.columns:
        out["carry_diff_3m"] = df["xau_swap_3m"] - df["xag_swap_3m"]

    # --- Rolling correlation (GC vs SI daily returns) ---
    for lb in [20, 60]:
        out[f"gc_si_corr{lb}"] = (
            out["gc_pct"].rolling(lb, min_periods=lb).corr(out["si_pct"])
        )

    # --- Regime context ---
    out["vix"] = df["vix"]
    out["us_10y"] = df["us_10y_yield"] if "us_10y_yield" in df.columns else np.nan

    return out


# ---------------------------------------------------------------------------
# Half-life summary
# ---------------------------------------------------------------------------
def compute_halflife_summary(features: pd.DataFrame) -> pd.DataFrame:
    cols = ["gs_ratio_fut", "gs_ratio_spot", "gs_log_ratio"]
    rows = []
    for col in cols:
        if col not in features.columns:
            continue
        s = features.set_index("date")[col].dropna()
        hl_full = estimate_halflife(s)
        rows.append({
            "series": col,
            "period": "full",
            "n_obs": len(s),
            "halflife_days": round(hl_full, 1) if not np.isnan(hl_full) else "N/A",
            "mean": round(float(s.mean()), 2),
            "std": round(float(s.std()), 3),
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
                "series": col,
                "period": label,
                "n_obs": len(sub),
                "halflife_days": round(hl, 1) if not np.isnan(hl) else "N/A",
                "mean": round(float(sub.mean()), 2) if len(sub) else None,
                "std": round(float(sub.std()), 3) if len(sub) else None,
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

    print("[2/3] Building ratio features...")
    features = build_ratio_features(df)
    print(f"  Features shape: {features.shape}")

    print("[3/3] Saving outputs...")
    features.to_csv(OUT_DIR / "gold_silver_ratio_features.csv", index=False)
    print(f"  Saved: {OUT_DIR / 'gold_silver_ratio_features.csv'}")

    hl_df = compute_halflife_summary(features)
    hl_df.to_csv(OUT_DIR / "ratio_halflife_summary.csv", index=False)
    print(f"  Saved: {OUT_DIR / 'ratio_halflife_summary.csv'}")

    print("\n=== RATIO FEATURE SUMMARY ===")
    for col in ["gs_ratio_fut", "gs_log_ratio"]:
        if col not in features.columns:
            continue
        s = features[col].dropna()
        hl = estimate_halflife(features.set_index("date")[col].dropna())
        hl_str = f"{hl:.1f}d" if not np.isnan(hl) else "not MR"
        print(f"  {col:20s}  mean={s.mean():+.3f}  std={s.std():.3f}  "
              f"min={s.min():+.2f}  max={s.max():+.2f}  half-life={hl_str}")

    corr_60 = features[["gc_pct", "si_pct"]].dropna().corr().iloc[0, 1]
    print(f"\n  GC/SI 60d avg correlation: {corr_60:.3f}")
    print(f"  GC/SI ratio range: {features['gs_ratio_fut'].min():.1f} – {features['gs_ratio_fut'].max():.1f}")
    print(f"  Notional hedge ratio range: {features['notional_hedge_ratio'].min():.2f} – {features['notional_hedge_ratio'].max():.2f}")


if __name__ == "__main__":
    main()
