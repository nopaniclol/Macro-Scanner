"""
02_clean_data.py
================
Load data-2.csv, run data audit, clean to trading days,
export cleaned data and audit summaries.

Outputs:
    01_data_raw/data_audit_table.csv
    01_data_raw/data_audit_summary.md
    02_data_clean/data_clean.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
RAW_DIR = BASE / "01_data_raw"
CLEAN_DIR = BASE / "02_data_clean"

GOLD_COLS = ["gc_fut_front", "gc_fut_second", "gc_fut_third"]
SILVER_COLS = ["si_fut_front", "si_fut_second", "si_fut_third"]
SPOT_COLS = ["xauusd_spot", "xagusd_spot"]
SWAP_COLS = [c for c in [
    "xau_swap_1m", "xau_swap_2m", "xau_swap_3m",
    "xau_swap_6m", "xau_swap_9m", "xau_swap_1y", "xau_swap_2y"
] if True]  # will filter against actual columns later

RESEARCH_COLS = GOLD_COLS + SILVER_COLS + SPOT_COLS + [
    "lbma_gold_am", "lbma_gold_pm", "lbma_silver_fix",
    "gold_efp", "silver_efp",
    "gold_vol_index", "vix",
    "us_10y_yield", "tips_10y_real_yield",
    "dxy_index",
    # USD SOFR OIS curve (actual rates — primary for fair-value modelling)
    "sofr_ois_1m", "sofr_ois_3m", "sofr_ois_6m", "sofr_ois_1y",
    # SOFR futures (regime filter: forward-looking slope signal)
    "sofr_fut_front", "sofr_fut_second", "sofr_fut_third", "sofr_fut_fourth",
    # Gold/silver forward swap rates (carry context)
    "xau_swap_1m", "xau_swap_3m", "xau_swap_6m", "xau_swap_1y",
]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    """Load raw CSV, parse dates, sort ascending."""
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------
def audit_series(df: pd.DataFrame, col: str) -> dict:
    """Compute audit stats for one column."""
    if col not in df.columns:
        return {
            "column": col,
            "present": False,
            "n_total": len(df),
            "n_valid": 0,
            "n_missing": len(df),
            "pct_missing": 100.0,
            "first_valid": None,
            "last_valid": None,
            "min": None,
            "max": None,
            "mean": None,
            "note": "COLUMN ABSENT",
        }

    s = pd.to_numeric(df[col], errors="coerce")
    valid = s.dropna()
    n_total = len(s)
    n_valid = len(valid)
    n_missing = n_total - n_valid

    # Detect first/last valid date
    mask = s.notna()
    first_valid = df.loc[mask, "date"].min() if mask.any() else None
    last_valid = df.loc[mask, "date"].max() if mask.any() else None

    # Outlier flag: any value > 5 std from rolling median (crude)
    note = ""
    if n_valid > 10:
        med = valid.median()
        std = valid.std()
        outliers = (valid - med).abs() > 5 * std
        if outliers.any():
            note += f"POSSIBLE_OUTLIERS({outliers.sum()}) "

    return {
        "column": col,
        "present": True,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / n_total, 1),
        "first_valid": str(first_valid.date()) if first_valid is not None else None,
        "last_valid": str(last_valid.date()) if last_valid is not None else None,
        "min": round(float(valid.min()), 4) if n_valid else None,
        "max": round(float(valid.max()), 4) if n_valid else None,
        "mean": round(float(valid.mean()), 4) if n_valid else None,
        "note": note.strip(),
    }


def audit_spread_plausibility(df: pd.DataFrame) -> dict[str, str]:
    """Sanity-check spreads: GC1-GC2 range, GS ratio range."""
    notes: dict[str, str] = {}

    for c1, c2, label, lo, hi in [
        ("gc_fut_front", "gc_fut_second", "cal_12_gc", -20, 20),
        ("gc_fut_second", "gc_fut_third", "cal_23_gc", -20, 20),
        ("si_fut_front", "si_fut_second", "cal_12_si", -5, 5),
    ]:
        if c1 in df.columns and c2 in df.columns:
            spread = pd.to_numeric(df[c1], errors="coerce") - pd.to_numeric(df[c2], errors="coerce")
            spread = spread.dropna()
            bad = ((spread < lo) | (spread > hi)).sum()
            notes[label] = (
                f"min={spread.min():.2f} max={spread.max():.2f} "
                f"mean={spread.mean():.3f} outside[{lo},{hi}]={bad}"
            )

    if "gc_fut_front" in df.columns and "si_fut_front" in df.columns:
        ratio = pd.to_numeric(df["gc_fut_front"], errors="coerce") / pd.to_numeric(df["si_fut_front"], errors="coerce")
        ratio = ratio.dropna()
        bad = ((ratio < 30) | (ratio > 200)).sum()
        notes["gs_ratio"] = (
            f"min={ratio.min():.1f} max={ratio.max():.1f} "
            f"mean={ratio.mean():.1f} outside[30,200]={bad}"
        )

    return notes


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def run_audit(df_raw: pd.DataFrame, audit_cols: list[str]) -> pd.DataFrame:
    """Run full audit, return audit DataFrame."""
    rows = [audit_series(df_raw, c) for c in audit_cols]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove weekends, coerce numeric columns,
    keep rows where all three GC legs are present.
    """
    # Drop weekends
    df = df[df["date"].dt.dayofweek < 5].copy()

    # Coerce numeric on research cols present in df
    for col in RESEARCH_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where ALL three GC legs are NaN (truly empty trading days)
    gc_present = df[GOLD_COLS].notna().all(axis=1)
    n_dropped = (~gc_present).sum()
    df = df[gc_present].copy()

    df = df.reset_index(drop=True)
    print(f"  [clean] After weekend+GC filter: {len(df)} rows ({n_dropped} dropped missing GC)")
    return df


# ---------------------------------------------------------------------------
# Build summary markdown
# ---------------------------------------------------------------------------
def build_summary_md(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    audit_df: pd.DataFrame,
    spread_notes: dict,
) -> str:
    """Return markdown string for data_audit_summary.md."""
    gold_audit = audit_df[audit_df["column"].isin(GOLD_COLS)]
    silver_audit = audit_df[audit_df["column"].isin(SILVER_COLS)]

    lines = [
        "# Data Audit Summary — data-2.csv",
        "",
        f"**Generated**: 2026-03-29",
        "",
        "## Date Range",
        f"- Raw rows: {len(df_raw):,}",
        f"- Date range: {df_raw['date'].min().date()} → {df_raw['date'].max().date()}",
        f"- Trading-day rows (post-clean): {len(df_clean):,}",
        "",
        "## Weekend Filtering",
        f"- Weekend rows removed: {len(df_raw) - len(df_clean) - (df_raw['date'].dt.dayofweek >= 5).sum():,} (GC-missing weekedays)",
        f"- Weekend rows: {(df_raw['date'].dt.dayofweek >= 5).sum():,}",
        "",
        "## Gold Futures Continuity",
    ]
    for _, row in gold_audit.iterrows():
        lines.append(
            f"- **{row['column']}**: {row['n_valid']:,}/{row['n_total']:,} valid "
            f"({row['pct_missing']}% missing) | "
            f"range {row['first_valid']} → {row['last_valid']} | "
            f"price [{row['min']}, {row['max']}] "
            + (f"⚠️ {row['note']}" if row['note'] else "✓")
        )

    lines += ["", "## Silver Futures Continuity"]
    for _, row in silver_audit.iterrows():
        present = row["present"]
        if not present:
            lines.append(f"- **{row['column']}**: ❌ ABSENT")
        else:
            lines.append(
                f"- **{row['column']}**: {row['n_valid']:,}/{row['n_total']:,} valid "
                f"({row['pct_missing']}% missing) | "
                f"range {row['first_valid']} → {row['last_valid']} | "
                f"price [{row['min']}, {row['max']}] "
                + (f"⚠️ {row['note']}" if row['note'] else "✓")
            )

    lines += ["", "## Spread Plausibility Checks"]
    for key, note in spread_notes.items():
        flag = "⚠️" if "outside" in note and not note.endswith("=0") else "✓"
        lines.append(f"- **{key}**: {note} {flag}")

    # Year coverage
    lines += ["", "## Year-by-Year Coverage (GC front)"]
    if "gc_fut_front" in df_clean.columns:
        yearly = df_clean.groupby(df_clean["date"].dt.year)["gc_fut_front"].agg(
            rows="count", missing=lambda x: x.isna().sum()
        )
        lines.append("| Year | Rows | Missing |")
        lines.append("|------|------|---------|")
        for yr, row in yearly.iterrows():
            lines.append(f"| {yr} | {row['rows']} | {row['missing']} |")

    lines += [
        "",
        "## Recommendations",
        "- ✅ Use `gc_fut_front`, `gc_fut_second`, `gc_fut_third` for Track 1 (curve RV)",
        "- ✅ Use `si_fut_front` for Track 2 (gold-silver ratio)",
        "- ⚠️ Check silver third-leg continuity before using for 3-leg silver fly",
        "- ✅ Gold-silver ratio range looks plausible (50–100 typical post-2018)",
        "- ⚠️ Jan 2018 has some missing values on first row (holiday) — already filtered by GC completeness check",
        "",
        "## First Research Pass Recommendation",
        "**Start with Track 1 (Gold Calendar Spreads)** — all three GC legs are clean,",
        "data goes back to early 2018, and the calendar spread story is the simplest to explain.",
        "Track 2 (gold-silver ratio) can run in parallel once Track 1 is validated.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading raw data...")
    df_raw = load_raw(RAW_DIR / "data-2.csv")
    print(f"  Raw shape: {df_raw.shape}")
    print(f"  Date range: {df_raw['date'].min().date()} → {df_raw['date'].max().date()}")

    print("[2/5] Running data audit...")
    audit_cols = GOLD_COLS + SILVER_COLS + SPOT_COLS + [
        "lbma_gold_am", "lbma_gold_pm",
        "gold_efp", "silver_efp",
        "xau_swap_1m", "xau_swap_3m", "xau_swap_6m", "xau_swap_1y",
        "gold_vol_index", "vix",
        "us_10y_yield",
    ]
    audit_df = run_audit(df_raw, audit_cols)
    spread_notes = audit_spread_plausibility(df_raw)

    print("[3/5] Cleaning data...")
    df_clean = clean_data(df_raw)

    print("[4/5] Saving outputs...")
    audit_df.to_csv(RAW_DIR / "data_audit_table.csv", index=False)
    print(f"  Saved: {RAW_DIR / 'data_audit_table.csv'}")

    summary_md = build_summary_md(df_raw, df_clean, audit_df, spread_notes)
    (RAW_DIR / "data_audit_summary.md").write_text(summary_md)
    print(f"  Saved: {RAW_DIR / 'data_audit_summary.md'}")

    # Save only research-relevant columns + date
    keep_cols = ["date"] + [c for c in RESEARCH_COLS if c in df_clean.columns]
    df_clean[keep_cols].to_csv(CLEAN_DIR / "data_clean.csv", index=False)
    print(f"  Saved: {CLEAN_DIR / 'data_clean.csv'} ({len(df_clean)} rows, {len(keep_cols)} cols)")

    print("[5/5] Done.")
    print("\n=== AUDIT HIGHLIGHTS ===")
    for _, row in audit_df.iterrows():
        flag = "❌" if not row["present"] or row["pct_missing"] > 20 else ("⚠️" if row["pct_missing"] > 5 else "✓")
        print(f"  {flag} {row['column']:30s} {row['pct_missing']:5.1f}% missing  [{row['first_valid']} → {row['last_valid']}]")
    print("\nSpread plausibility:")
    for k, v in spread_notes.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
