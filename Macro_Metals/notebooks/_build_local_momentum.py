#!/usr/bin/env python3
"""Build 01_momentum_research_LOCAL.ipynb from the BQuant version.

Strategy: copy the original notebook, then surgically replace:
  - Cell 0 (title): update to LOCAL mode
  - Cell 1 (imports): remove BQL, add local imports
  - Cell 3 (config): add YAML fallback
  - Cell 5 (universe): add resolve_ticker fallback
  - Cell 7 (BQL data loader): replace with LocalCSVLoader + data inspection
  - Cell 59 (export): add _local suffixes

And INSERT new cells:
  - After imports: MODE banner + data inspection
  - Before export: local debug cells
"""
import json, copy

def src(text):
    lines = text.strip("\n").split("\n")
    return [l + "\n" if i < len(lines)-1 else l for i, l in enumerate(lines)]

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": src(text)}

def code(text):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src(text)}

# Load original notebook
orig_path = "01_momentum_research.ipynb"
with open(orig_path) as f:
    nb = json.load(f)

cells = nb["cells"]

# We'll build the new notebook cell by cell
new_cells = []

# ═══════════════════════════════════════════════════════════════════
# CELL 0: Updated title
# ═══════════════════════════════════════════════════════════════════
new_cells.append(md("""# 01 Momentum Research — Macro Metals System (LOCAL MODE)

> **Strategy:** Canonical Time-Series Momentum (TSMOM)
> **Reference:** Moskowitz, Ooi & Pedersen (2012); Quantpedia; JPM/CME report
> **Scope:** In-sample development (2015–2022), monthly rebalance
> **Universe:** Multi-asset futures — commodities (70% risk budget), rates, equities, FX (30%)
> **Signal:** sign(12-month return), monthly rebalance, inverse-vol position sizing
> **Overlay:** Sector risk budgets + portfolio-level vol targeting (10% annual)
>
> **LOCAL MODE** — reads from CSV files instead of Bloomberg BQL.
> Strategy logic is IDENTICAL to the BQuant version.
> All outputs suffixed with `_local` to distinguish from BQuant runs."""))

# ═══════════════════════════════════════════════════════════════════
# CELL 1: Imports (no BQL)
# ═══════════════════════════════════════════════════════════════════
new_cells.append(code("""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
from datetime import datetime
from math import sqrt
import os
import glob

# NO Bloomberg BQL — LOCAL MODE
print(f"Session started : {datetime.now():%Y-%m-%d %H:%M}")
print(f"Mode            : LOCAL (CSV)")
print(f"NumPy {np.__version__}  |  pandas {pd.__version__}")"""))

# ═══════════════════════════════════════════════════════════════════
# MODE banner cell (new)
# ═══════════════════════════════════════════════════════════════════
new_cells.append(md("""## Mode Selection"""))

new_cells.append(code("""MODE = "LOCAL"  # "LOCAL" or "BQUANT"
DATA_DIR = "data/"
print(f"Running in {MODE} mode")
print(f"Data directory: {DATA_DIR}")
print(f"Expected: CSV file(s) with daily price data for multi-asset futures universe")"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 2 (orig): Config markdown — keep as-is
# ═══════════════════════════════════════════════════════════════════
new_cells.append(copy.deepcopy(cells[2]))

# ═══════════════════════════════════════════════════════════════════
# CELL 3 (orig): Config code — add YAML fallback
# ═══════════════════════════════════════════════════════════════════
new_cells.append(code(r"""CONFIG_DIR = Path("config")

# ── YAML loading with fallback ───────────────────────────────────
DEFAULT_PARAMS = {
    "global": {
        "in_sample_start": "2015-01-01",
        "in_sample_end": "2022-12-31",
        "target_portfolio_vol_annual": 0.10,
        "vol_decay_lambda": 0.94,
        "vol_cap_multiplier": 2.0,
    },
    "performance_targets": {
        "sharpe_per_strategy": 0.5,
        "vol_range_annual": [0.08, 0.12],
        "max_drawdown_pct": 15.0,
        "calmar_ratio": 0.7,
        "hit_rate_daily": 0.52,
        "max_turnover_annual": 30.0,
    },
}

try:
    with open(CONFIG_DIR / "parameters.yaml") as f:
        params = yaml.safe_load(f)
    print("Loaded: config/parameters.yaml")
except FileNotFoundError:
    params = DEFAULT_PARAMS
    print("WARNING: config/parameters.yaml not found — using hardcoded defaults.")

try:
    with open(CONFIG_DIR / "tickers.yaml") as f:
        tickers = yaml.safe_load(f)
    print("Loaded: config/tickers.yaml")
except FileNotFoundError:
    tickers = {}
    print("WARNING: config/tickers.yaml not found — will use fallback ticker mapping.")

gcfg    = params["global"]
targets = params["performance_targets"]

# ── Canonical TSMOM parameters ───────────────────────────────────
LOOKBACK_DAYS = 252
TARGET_VOL    = gcfg.get("target_portfolio_vol_annual", 0.10)
VOL_LAMBDA    = gcfg.get("vol_decay_lambda", 0.94)
LEV_CAP       = gcfg.get("vol_cap_multiplier", 2.0)
TC_BP         = 2.0

# Sector risk budgets (commodity-focused)
COMMODITY_RISK_BUDGET    = 0.70
DIVERSIFIER_RISK_BUDGET  = 0.30
PORTFOLIO_VOL_TARGET     = 0.10
PORTFOLIO_VOL_LOOKBACK   = 60

# ── FIX 2: Lower portfolio overlay scale floor ───────────────────
PORTFOLIO_SCALE_BOUNDS   = (0.05, 2.0)   # was (0.5, 2.0)

# ── FIX 1: Hard absolute weight caps per bucket ─────────────────
ABS_WEIGHT_CAP_BY_BUCKET = {
    "commodities": 1.0,
    "equities":    0.8,
    "fx":          0.8,
    "rates":       0.5,
}

# Optional per-instrument overrides (Fix 6: energy)
ABS_WEIGHT_CAP_BY_INSTRUMENT = {
    "cl_fut_front": 0.5,
    "ng_fut_front": 0.5,
}

# Winsorization threshold for return cleaning (Fix 5)
WINSOR_Z_THRESHOLD = 5.0

IS_START = gcfg.get("in_sample_start", "2015-01-01")
IS_END   = gcfg.get("in_sample_end", "2022-12-31")

print("\nCanonical TSMOM parameters:")
print(f"  Lookback           : {LOOKBACK_DAYS}d (12 months)")
print(f"  Rebalance          : monthly")
print(f"  Per-inst vol target: {TARGET_VOL:.0%}")
print(f"  EWMA lambda        : {VOL_LAMBDA}")
print(f"  Leverage cap (rel) : {LEV_CAP:.1f}x")
print(f"  TC per side        : {TC_BP:.1f} bp")
print(f"  Risk budget        : {COMMODITY_RISK_BUDGET:.0%} commodities / {DIVERSIFIER_RISK_BUDGET:.0%} diversifiers")
print(f"  Portfolio vol tgt  : {PORTFOLIO_VOL_TARGET:.0%}")
print(f"  Overlay scale floor: {PORTFOLIO_SCALE_BOUNDS[0]} (FIX 2)")
print(f"  Abs weight caps    : {ABS_WEIGHT_CAP_BY_BUCKET} (FIX 1)")
print(f"  Energy overrides   : {ABS_WEIGHT_CAP_BY_INSTRUMENT} (FIX 6)")
print(f"  Winsor z-threshold : {WINSOR_Z_THRESHOLD} (FIX 5)")
print(f"  IS period          : {IS_START} to {IS_END}")"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 4 (orig): Universe expansion markdown — keep as-is
# ═══════════════════════════════════════════════════════════════════
new_cells.append(copy.deepcopy(cells[4]))

# ═══════════════════════════════════════════════════════════════════
# CELL 5 (orig): Universe code — add fallback resolve
# ═══════════════════════════════════════════════════════════════════
new_cells.append(code("""# ── Multi-asset TSMOM universe ────────────────────────────────────
universe = {
    "commodities": [
        "gc_fut_front", "si_fut_front", "pl_fut_front",
        "hg_fut_front",
        "cl_fut_front", "ng_fut_front",
        "w_fut_front", "c_fut_front", "s_fut_front",
    ],
    "rates": [
        "sofr_fut_front", "sofr_fut_second", "sofr_fut_third", "sofr_fut_fourth",
        "ust_2y_fut", "ust_5y_fut", "ust_10y_fut", "ust_30y_fut",
    ],
    "equities": [
        "es_fut_front", "nq_fut_front", "stoxx50_fut_front", "nikkei_fut_front",
    ],
    "fx": [
        "eurusd_spot", "usdjpy_spot", "gbpusd_spot", "audusd_spot",
        "nzdusd_spot", "usdcad_spot",
    ],
}

LABELS = {
    "gc_fut_front": "Gold (GC)", "si_fut_front": "Silver (SI)",
    "pl_fut_front": "Platinum (PL)", "hg_fut_front": "Copper (HG)",
    "cl_fut_front": "Crude Oil (CL)", "ng_fut_front": "Nat Gas (NG)",
    "w_fut_front": "Wheat (W)", "c_fut_front": "Corn (C)", "s_fut_front": "Soybeans (S)",
    "sofr_fut_front": "SOFR 1st", "sofr_fut_second": "SOFR 2nd",
    "sofr_fut_third": "SOFR 3rd", "sofr_fut_fourth": "SOFR 4th",
    "ust_2y_fut": "UST 2Y (TU)", "ust_5y_fut": "UST 5Y (FV)",
    "ust_10y_fut": "UST 10Y (TY)", "ust_30y_fut": "UST 30Y (US)",
    "es_fut_front": "S&P 500 (ES)", "nq_fut_front": "Nasdaq (NQ)",
    "stoxx50_fut_front": "EuroStoxx 50", "nikkei_fut_front": "Nikkei 225",
    "eurusd_spot": "EURUSD", "usdjpy_spot": "USDJPY", "gbpusd_spot": "GBPUSD",
    "audusd_spot": "AUDUSD", "nzdusd_spot": "NZDUSD", "usdcad_spot": "USDCAD",
}

INST_BUCKET = {}
for bucket, instruments in universe.items():
    for inst in instruments:
        INST_BUCKET[inst] = bucket

# ── Ticker resolution (Bloomberg ticker <-> logical name) ────────
# Build REVERSE map: Bloomberg ticker -> logical name (for CSV column matching)
FALLBACK_MAP = {
    "GC1 Comdty": "gc_fut_front", "SI1 Comdty": "si_fut_front",
    "PL1 Comdty": "pl_fut_front", "HG1 Comdty": "hg_fut_front",
    "CL1 Comdty": "cl_fut_front", "NG1 Comdty": "ng_fut_front",
    "W 1 Comdty": "w_fut_front", "C 1 Comdty": "c_fut_front",
    "S 1 Comdty": "s_fut_front",
    "SFR Comdty": "sofr_fut_front", "SR31 Comdty": "sofr_fut_front",
    "SR32 Comdty": "sofr_fut_second",
    "SR33 Comdty": "sofr_fut_third", "SR34 Comdty": "sofr_fut_fourth",
    "TU1 Comdty": "ust_2y_fut", "FV1 Comdty": "ust_5y_fut",
    "TY1 Comdty": "ust_10y_fut", "US1 Comdty": "ust_30y_fut",
    "ES1 Index": "es_fut_front", "NQ1 Index": "nq_fut_front",
    "VG1 Index": "stoxx50_fut_front", "NK1 Index": "nikkei_fut_front",
    "EURUSD Curncy": "eurusd_spot", "USDJPY Curncy": "usdjpy_spot",
    "GBPUSD Curncy": "gbpusd_spot", "AUDUSD Curncy": "audusd_spot",
    "NZDUSD Curncy": "nzdusd_spot", "USDCAD Curncy": "usdcad_spot",
}

# Build forward map from tickers.yaml if available, else from FALLBACK
def build_ticker_maps(tickers_yaml, fallback):
    \"\"\"Build forward (logical->bbg) and reverse (bbg->logical) maps.\"\"\"
    fwd = {}
    rev = {}
    if tickers_yaml:
        for group in tickers_yaml.values():
            if isinstance(group, dict):
                for logical, bbg in group.items():
                    if isinstance(bbg, str):
                        fwd[logical] = bbg
                        rev[bbg] = logical
    # Add fallback entries not already covered
    for bbg, logical in fallback.items():
        if logical not in fwd:
            fwd[logical] = bbg
        if bbg not in rev:
            rev[bbg] = logical
    return fwd, rev

TICKER_FWD, TICKER_REV = build_ticker_maps(tickers, FALLBACK_MAP)

# Resolve universe
all_universe_insts = [i for insts in universe.values() for i in insts]
resolved = {inst: TICKER_FWD.get(inst, inst) for inst in all_universe_insts if inst in TICKER_FWD}
missing = [inst for inst in all_universe_insts if inst not in TICKER_FWD]

print(f"Universe: {len(all_universe_insts)} instruments across {len(universe)} buckets")
print(f"Resolved: {len(resolved)} | Missing: {len(missing)}")
if missing:
    print(f"\\n  Missing ticker mappings (will try direct CSV column match): {missing}")

print("\\nBucket breakdown:")
for bucket, instruments in universe.items():
    n_resolved = sum(1 for i in instruments if i in resolved)
    print(f"  {bucket:15s}: {n_resolved}/{len(instruments)} resolved")"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 6 (orig): Data Pipeline markdown — update for LOCAL
# ═══════════════════════════════════════════════════════════════════
new_cells.append(md("""## Data Pipeline (LOCAL CSV)

Read daily prices from CSV file(s) in the `data/` directory.
Auto-detects wide vs long format, maps columns to logical instrument names.

**Fix 3:** After panel construction, each instrument is masked to its
valid date range only. Pre-listing NaN periods are NOT forward-filled."""))

# ═══════════════════════════════════════════════════════════════════
# DATA INSPECTION (Part 1) — new cells
# ═══════════════════════════════════════════════════════════════════
new_cells.append(md("""## Data Inspection — Raw CSV Preview

Scan the `data/` directory, auto-detect format, print structure and readiness."""))

new_cells.append(code(r"""# ── PART 1: Data Inspection ───────────────────────────────────────
data_dir = Path(DATA_DIR)
csv_files = sorted(data_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in '{data_dir.resolve()}/'. "
        f"Please place your price data CSV(s) in the data/ directory."
    )

print(f"Found {len(csv_files)} CSV file(s) in {data_dir.resolve()}/:")
for f in csv_files:
    print(f"  {f.name}")

# ── Inspect each file ────────────────────────────────────────────
file_info = []

for csv_path in csv_files:
    print(f"\n{'='*70}")
    print(f"  FILE: {csv_path.name}")
    print(f"{'='*70}")

    # Try reading with common options
    raw_df = None
    for kwargs in [
        {},
        {"skiprows": 1},
        {"header": [0, 1]},
    ]:
        try:
            raw_df = pd.read_csv(csv_path, nrows=5, **kwargs)
            if raw_df.shape[1] >= 2:
                # Re-read full file
                raw_df = pd.read_csv(csv_path, **kwargs)
                break
        except Exception:
            continue

    if raw_df is None:
        print(f"  ERROR: Could not parse {csv_path.name}")
        # Show first 5 raw lines
        with open(csv_path) as fh:
            for i, line in enumerate(fh):
                if i >= 5: break
                print(f"  RAW LINE {i}: {line.rstrip()}")
        continue

    print(f"  Shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
    print(f"  Columns: {list(raw_df.columns)}")
    print(f"  Dtypes:\n{raw_df.dtypes.to_string()}")

    # Detect date column
    date_col = None
    for candidate in ['date', 'Date', 'DATE', 'Dates', 'datetime', 'Datetime', 'DATETIME']:
        if candidate in raw_df.columns:
            date_col = candidate
            break
    if date_col is None:
        # Try first column
        first_col = raw_df.columns[0]
        try:
            pd.to_datetime(raw_df[first_col].iloc[:5])
            date_col = first_col
        except Exception:
            pass
    if date_col is None and raw_df.index.dtype == 'object':
        try:
            pd.to_datetime(raw_df.index[:5])
            date_col = '__index__'
        except Exception:
            pass

    print(f"  Date column: {date_col}")

    # Parse dates
    if date_col and date_col != '__index__':
        for fmt in [None, "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"]:
            try:
                if fmt:
                    raw_df[date_col] = pd.to_datetime(raw_df[date_col], format=fmt)
                else:
                    raw_df[date_col] = pd.to_datetime(raw_df[date_col], infer_datetime_format=True)
                break
            except Exception:
                continue
        date_min = raw_df[date_col].min()
        date_max = raw_df[date_col].max()
    elif date_col == '__index__':
        raw_df.index = pd.to_datetime(raw_df.index)
        date_min = raw_df.index.min()
        date_max = raw_df.index.max()
    else:
        date_min = date_max = None

    print(f"  Date range: {date_min} to {date_max}")

    # NaN counts
    nan_counts = raw_df.isna().sum()
    print(f"  NaN counts (top 10):")
    for col, cnt in nan_counts.nlargest(10).items():
        print(f"    {col}: {cnt}")

    # First/last rows
    print(f"\n  First 3 rows:")
    print(raw_df.head(3).to_string())
    print(f"\n  Last 3 rows:")
    print(raw_df.tail(3).to_string())

    # Detect format
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    non_date_cols = [c for c in raw_df.columns if c != date_col]

    if len(csv_files) > 1 and len(numeric_cols) <= 2:
        fmt = "C (one file per instrument)"
    elif any(c in [c.lower() for c in raw_df.columns] for c in ['ticker', 'instrument', 'symbol']):
        fmt = "B (long format)"
    elif len(numeric_cols) >= 3:
        fmt = "A (wide format)"
    else:
        fmt = "Unknown"

    print(f"\n  Detected format: {fmt}")

    file_info.append({
        "file": csv_path.name,
        "format": fmt,
        "rows": raw_df.shape[0],
        "cols": raw_df.shape[1],
        "date_col": date_col,
        "date_min": date_min,
        "date_max": date_max,
        "numeric_cols": len(numeric_cols),
    })

# ── Data readiness summary ────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  DATA READINESS SUMMARY")
print(f"{'='*70}")
for info in file_info:
    print(f"  {info['file']:30s}  format={info['format'][:20]:20s}  "
          f"rows={info['rows']:>6d}  cols={info['cols']:>3d}  "
          f"range={info['date_min']} to {info['date_max']}")"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 7 replacement: LocalCSVLoader + data loading
# ═══════════════════════════════════════════════════════════════════
new_cells.append(md("""## LocalCSVLoader — Flexible CSV Data Pipeline

Replaces `BQuantDataLoader`. Auto-detects CSV format (wide/long/multi-file),
maps columns to logical instrument names, handles date parsing, forward-fill,
and gap reporting."""))

new_cells.append(code(r'''class LocalCSVLoader:
    """Load price data from local CSV files."""

    def __init__(self, ticker_fwd: dict = None, ticker_rev: dict = None,
                 fallback_map: dict = None):
        self._fwd = ticker_fwd or {}
        self._rev = ticker_rev or {}
        self._fallback = fallback_map or {}
        self._coverage = None

    def _parse_dates(self, df, date_col):
        """Try multiple date formats."""
        for fmt in [None, "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"]:
            try:
                if fmt:
                    return pd.to_datetime(df[date_col], format=fmt)
                else:
                    return pd.to_datetime(df[date_col], infer_datetime_format=True)
            except Exception:
                continue
        raise ValueError(f"Could not parse dates in column '{date_col}'")

    def _find_date_col(self, df):
        """Detect date column."""
        for candidate in ['date', 'Date', 'DATE', 'Dates', 'datetime', 'Datetime']:
            if candidate in df.columns:
                return candidate
        # Try first column
        try:
            pd.to_datetime(df.iloc[:5, 0])
            return df.columns[0]
        except Exception:
            pass
        return None

    def _map_column(self, col_name):
        """Map a CSV column name to a logical instrument name."""
        col_stripped = col_name.strip()
        # Direct match in reverse map (Bloomberg ticker -> logical)
        if col_stripped in self._rev:
            return self._rev[col_stripped]
        # Case-insensitive match
        for bbg, logical in self._rev.items():
            if bbg.lower() == col_stripped.lower():
                return logical
        for bbg, logical in self._fallback.items():
            if bbg.lower() == col_stripped.lower():
                return logical
        # Direct match as logical name
        all_logical = set()
        for insts in universe.values():
            all_logical.update(insts)
        if col_stripped in all_logical:
            return col_stripped
        if col_stripped.lower().replace(" ", "_") in all_logical:
            return col_stripped.lower().replace(" ", "_")
        return None

    def load_prices(self, data_dir: str = "data/") -> pd.DataFrame:
        """Load and return wide-format price DataFrame."""
        data_path = Path(data_dir)
        csv_files = sorted(data_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files in {data_path.resolve()}")

        all_dfs = []

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"  Could not parse CSV: {csv_path.name}. Error: {e}")
                with open(csv_path) as fh:
                    for i, line in enumerate(fh):
                        if i >= 5: break
                        print(f"  RAW LINE {i}: {line.rstrip()}")
                continue

            date_col = self._find_date_col(df)
            if date_col is None:
                print(f"  WARNING: No date column found in {csv_path.name}")
                continue

            df[date_col] = self._parse_dates(df, date_col)
            df = df.set_index(date_col)
            df.index.name = 'date'
            df.index = pd.to_datetime(df.index)

            # Detect long format
            long_cols = {'ticker', 'instrument', 'symbol'}
            found_long = long_cols.intersection(set(c.lower() for c in df.columns))
            if found_long:
                # Long format: pivot
                ticker_col = [c for c in df.columns if c.lower() in long_cols][0]
                value_cols = [c for c in df.columns if c.lower() in
                              {'price', 'value', 'close', 'px_last', 'last'}]
                if not value_cols:
                    value_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:1]
                if value_cols:
                    df_wide = df.pivot_table(index=df.index, columns=ticker_col,
                                            values=value_cols[0])
                    all_dfs.append(df_wide)
                continue

            # Wide format: use numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                all_dfs.append(numeric_df)

        if not all_dfs:
            raise ValueError("No valid price data found in any CSV file")

        # Combine all frames
        if len(all_dfs) == 1:
            raw_prices = all_dfs[0]
        else:
            raw_prices = pd.concat(all_dfs, axis=1)

        raw_prices = raw_prices.sort_index()
        raw_prices = raw_prices[~raw_prices.index.duplicated(keep='last')]

        # ── Map columns ──────────────────────────────────────────────
        mapped = {}
        unmapped = []
        for col in raw_prices.columns:
            logical = self._map_column(str(col))
            if logical:
                mapped[col] = logical
                print(f"  Mapped: {col} -> {logical}")
            else:
                unmapped.append(col)

        if unmapped:
            print(f"\n  Unmapped columns (will be skipped): {unmapped}")

        prices_df = raw_prices.rename(columns=mapped)
        # Keep only mapped columns that are in our universe
        all_logical = set()
        for insts in universe.values():
            all_logical.update(insts)
        keep_cols = [c for c in prices_df.columns if c in all_logical]
        prices_df = prices_df[keep_cols]

        # ── Filter to IS period ──────────────────────────────────────
        prices_df = prices_df.loc[IS_START:IS_END]

        # ── Business-day reindex + ffill (max 5 days) ────────────────
        bday_idx = pd.bdate_range(prices_df.index.min(), prices_df.index.max())
        prices_df = prices_df.reindex(bday_idx)
        prices_df = prices_df.ffill(limit=5)
        prices_df.index.name = 'date'

        # ── Report gaps > 5 business days ────────────────────────────
        for col in prices_df.columns:
            s = prices_df[col]
            nan_runs = s.isna()
            if nan_runs.any():
                groups = (nan_runs != nan_runs.shift()).cumsum()
                gap_lengths = nan_runs.groupby(groups).sum()
                long_gaps = gap_lengths[gap_lengths > 5]
                if len(long_gaps) > 0:
                    label = LABELS.get(col, col)
                    print(f"  WARNING: {label} has {len(long_gaps)} gap(s) > 5 business days")

        print(f"\nFinal prices_df: {prices_df.shape[0]} days x {prices_df.shape[1]} instruments")
        print(f"Date range: {prices_df.index[0]:%Y-%m-%d} to {prices_df.index[-1]:%Y-%m-%d}")
        print(f"Instruments: {list(prices_df.columns)}")

        self._prices = prices_df
        return prices_df

    def get_coverage_report(self) -> pd.DataFrame:
        """Generate coverage report matching BQL version."""
        if self._prices is None:
            return pd.DataFrame()

        rows = []
        for col in self._prices.columns:
            s = self._prices[col]
            n_total = len(s)
            n_nan = s.isna().sum()
            fv = s.first_valid_index()
            lv = s.last_valid_index()
            rows.append({
                "instrument": col, "label": LABELS.get(col, col),
                "bucket": INST_BUCKET.get(col, "?"),
                "n_obs": n_total, "n_nan": n_nan,
                "pct_missing": n_nan / max(n_total, 1) * 100,
                "first_valid": str(fv)[:10] if fv else "N/A",
                "last_valid": str(lv)[:10] if lv else "N/A",
                "status": "OK" if n_nan / max(n_total, 1) < 0.10 else "HIGH_MISSING",
            })
        return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)


# ── Load data ────────────────────────────────────────────────────
loader = LocalCSVLoader(
    ticker_fwd=TICKER_FWD,
    ticker_rev=TICKER_REV,
    fallback_map=FALLBACK_MAP,
)
prices_df = loader.load_prices(data_dir=DATA_DIR)
coverage_report = loader.get_coverage_report()

# ── FIX 3: Record first valid date per instrument ────────────────
FIRST_VALID_DATE = {}
for col in prices_df.columns:
    fvd = prices_df[col].first_valid_index()
    FIRST_VALID_DATE[col] = fvd

# Only ffill WITHIN each instrument's valid range (not before first valid)
for col in prices_df.columns:
    fvd = FIRST_VALID_DATE[col]
    if fvd is not None:
        valid_mask = prices_df.index >= fvd
        prices_df.loc[valid_mask, col] = prices_df.loc[valid_mask, col].ffill()

# Drop instruments with insufficient history
MIN_OBS = LOOKBACK_DAYS + 30
sufficient = prices_df.count() >= MIN_OBS
dropped_insts = prices_df.columns[~sufficient].tolist()
if dropped_insts:
    print(f"\nDropped (< {MIN_OBS} obs): {[LABELS.get(i,i) for i in dropped_insts]}")
prices_df = prices_df[prices_df.columns[sufficient]]

active_instruments = list(prices_df.columns)
active_buckets = {}
for inst in active_instruments:
    b = INST_BUCKET.get(inst)
    if b:
        active_buckets.setdefault(b, []).append(inst)

print(f"\nFinal panel: {prices_df.shape[0]} days x {prices_df.shape[1]} instruments")

# ── FIX 3: Print first valid dates ───────────────────────────────
print("\n" + "=" * 65)
print("  INSTRUMENT ACTIVE DATE RANGES (FIX 3)")
print("=" * 65)
for col in prices_df.columns:
    fvd = FIRST_VALID_DATE.get(col)
    n_pre = 0
    if fvd is not None:
        n_pre = (prices_df.index < fvd).sum()
    label = LABELS.get(col, col)
    print(f"  {label:22s}  active from: {str(fvd)[:10] if fvd else 'N/A':12s}  "
          f"({n_pre} days zeroed pre-listing)")

# Breadth report
print("\n" + "=" * 65)
print("  BREADTH REPORT")
print("=" * 65)
print(coverage_report.to_string(index=False))'''))

# ═══════════════════════════════════════════════════════════════════
# LOCAL DEBUG cells (Part 6) — inserted after data loading
# ═══════════════════════════════════════════════════════════════════
new_cells.append(md("""## LOCAL DEBUG — Raw CSV Preview

Normalised price plot and flat-series detection for visual data verification."""))

new_cells.append(code(r"""output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Head/tail of raw data
print("=" * 70)
print("  RAW CSV PREVIEW (head + tail)")
print("=" * 70)
print("\nHead (10 rows):")
print(prices_df.head(10).to_string())
print("\nTail (10 rows):")
print(prices_df.tail(10).to_string())
print(f"\nDtypes:\n{prices_df.dtypes.to_string()}")

# Normalised price plot
norm_prices = prices_df / prices_df.bfill().iloc[0] * 100
fig_raw = go.Figure()
for col in norm_prices.columns:
    label = LABELS.get(col, col)
    bucket = INST_BUCKET.get(col, "?")
    fig_raw.add_trace(go.Scatter(
        x=norm_prices.index, y=norm_prices[col], name=label,
        line=dict(width=1.2), legendgroup=bucket,
    ))
fig_raw.update_layout(
    title="LOCAL DEBUG: Normalised Price Levels (base=100 at start)",
    template="plotly_white", height=550, hovermode="x unified",
    yaxis_title="Index (100 = start)",
)
fig_raw.show()

# Flag flat series
print("\n" + "=" * 70)
print("  FLAT SERIES DETECTION")
print("=" * 70)
for col in prices_df.columns:
    s = prices_df[col].dropna()
    if len(s) < 50:
        continue
    cv = s.std() / s.mean() if s.mean() != 0 else 0
    label = LABELS.get(col, col)
    if cv < 0.001:
        print(f"  *** FLAT: {label:22s}  CV={cv:.6f}  (likely yield/rate series) ***")
    else:
        print(f"  OK:   {label:22s}  CV={cv:.4f}")

prices_df.head(100).to_csv(output_dir / "debug_raw_csv_preview.csv")
print(f"\nExported: debug_raw_csv_preview.csv")"""))

new_cells.append(md("""## LOCAL DEBUG — Return Distribution Sanity

Per-instrument return statistics to catch bad data before strategy runs."""))

new_cells.append(code(r"""ret_check = prices_df.pct_change()

print("=" * 85)
print("  RETURN DISTRIBUTION SANITY CHECK")
print("=" * 85)
print(f"\n  {'Instrument':22s}  {'p1%':>8s}  {'p50%':>8s}  {'p99%':>8s}  "
      f"{'Max 1d':>8s}  {'Ann Vol':>8s}  {'>1%':>5s}  {'>2%':>5s}  {'>5%':>5s}  {'Flag':>15s}")
print(f"  {'─'*115}")

ret_dist_rows = []
for col in prices_df.columns:
    r = ret_check[col].dropna()
    if len(r) == 0:
        continue
    p01 = r.quantile(0.01)
    p50 = r.quantile(0.50)
    p99 = r.quantile(0.99)
    max_1d = r.abs().max()
    ann_vol = r.std() * sqrt(252)
    gt1 = (r.abs() > 0.01).sum()
    gt2 = (r.abs() > 0.02).sum()
    gt5 = (r.abs() > 0.05).sum()
    label = LABELS.get(col, col)

    flags = []
    if ann_vol > 1.0:
        flags.append("VOL>100%")
    if max_1d > 0.20:
        flags.append("MAX>20%")
    if abs(p50) > 0.001:
        flags.append("MED_BIAS")
    flag_str = ", ".join(flags) if flags else ""

    print(f"  {label:22s}  {p01:+8.4f}  {p50:+8.4f}  {p99:+8.4f}  "
          f"{max_1d:8.4f}  {ann_vol:8.2%}  {gt1:5d}  {gt2:5d}  {gt5:5d}  {flag_str:>15s}")

    ret_dist_rows.append({
        "instrument": col, "label": label, "p01": p01, "p50": p50,
        "p99": p99, "max_1d": max_1d, "ann_vol": ann_vol,
        "gt_1pct": gt1, "gt_2pct": gt2, "gt_5pct": gt5, "flags": flag_str,
    })

ret_dist_df = pd.DataFrame(ret_dist_rows)
ret_dist_df.to_csv(output_dir / "debug_local_return_distribution.csv", index=False)
print(f"\nExported: debug_local_return_distribution.csv")"""))

# ═══════════════════════════════════════════════════════════════════
# Now copy cells 8-58 (DEBUG 1 through Visualisations) EXACTLY
# These are cells 8..57 in the original (indices 8-57 inclusive)
# ═══════════════════════════════════════════════════════════════════
for i in range(8, 58):
    new_cells.append(copy.deepcopy(cells[i]))

# ═══════════════════════════════════════════════════════════════════
# CELL 58-59 (orig): Export markdown + code — update with _local suffix
# ═══════════════════════════════════════════════════════════════════
new_cells.append(md("""## Export

Save signals, weights, equity, bucket risk, and summary (post-fix versions).
All outputs suffixed with `_local` to distinguish from BQuant runs."""))

new_cells.append(code(r"""output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

datestamp = datetime.now().strftime("%Y%m%d")

sig_path = output_dir / f"tsmom_signals_monthly_local_{datestamp}.csv"
signals.rename(columns=LABELS).to_csv(sig_path)

w_path = output_dir / f"tsmom_weights_daily_local_{datestamp}.csv"
weights_scaled.rename(columns=LABELS).to_csv(w_path)

wf_path = output_dir / f"tsmom_weights_final_daily_local_{datestamp}.csv"
final_weights.rename(columns=LABELS).to_csv(wf_path)

eq_path = output_dir / f"tsmom_portfolio_equity_local_{datestamp}.csv"
eq_export = pd.DataFrame({
    "portfolio_equity": port_equity, "portfolio_ret_net": port_ret_net,
    "portfolio_ret_gross": port_ret_gross, "turnover": turnover_daily,
})
eq_export.to_csv(eq_path)

br_path = output_dir / f"tsmom_bucket_risk_shares_local_{datestamp}.csv"
bucket_risk_shares.to_csv(br_path)

# Post-fix metrics
mf_path = output_dir / f"tsmom_metrics_post_fix_local_{datestamp}.csv"
metrics_df.to_csv(mf_path)

# Post-fix equity
eqf_path = output_dir / f"tsmom_portfolio_equity_post_fix_local_{datestamp}.csv"
eq_export.to_csv(eqf_path)

# Summary HTML
html_path = output_dir / f"tsmom_summary_post_fix_local_{datestamp}.html"
html_content = (
    "<h2>Canonical TSMOM — IS Performance (2015-2022, POST-FIX, LOCAL MODE)</h2>\n"
    "<p>Signal: sign(12M return) | Monthly rebalance | EWMA vol (lambda=0.94) | "
    f"Commodity budget: {COMMODITY_RISK_BUDGET:.0%} | Portfolio vol target: {PORTFOLIO_VOL_TARGET:.0%}<br>"
    f"Overlay scale floor: {PORTFOLIO_SCALE_BOUNDS[0]} | Abs weight caps: bucket-specific | "
    f"Winsorisation: {WINSOR_Z_THRESHOLD}z<br>"
    f"<b>Data source: LOCAL CSV (not Bloomberg BQL)</b></p>\n"
    "<h3>Per-instrument + Portfolio Metrics</h3>\n"
    + fmt_df.to_html()
    + "<br><h3>vs Memory File Targets (section 6.3)</h3>\n"
    + comparison.to_html()
    + "<br><h3>Control Tests</h3>\n"
    + controls_df.to_html()
)
with open(html_path, "w") as f:
    f.write(html_content)

print(f"Exported to {output_dir.resolve()}/")
for p in [sig_path, w_path, wf_path, eq_path, br_path, mf_path, eqf_path, html_path]:
    print(f"  {p.name}")

# Debug exports reminder
print(f"\nDebug exports (from earlier cells):")
debug_files = ["debug_coverage.csv", "debug_units_audit.csv", "debug_return_stats.csv",
               "debug_top_shocks.csv", "debug_roll_flags.csv", "debug_vol_summary.csv",
               "debug_weights_summary.csv", "debug_turnover.csv", "debug_controls.csv",
               "debug_sensitivity_grid.csv", "debug_universe_caps.csv",
               "debug_drawdown_decomposition.csv",
               "debug_raw_csv_preview.csv", "debug_local_return_distribution.csv"]
for f in debug_files:
    print(f"  {f}")

print(f"\n" + "=" * 65)
print(f"  ALL FIXES APPLIED. LOCAL notebook complete: {datetime.now():%Y-%m-%d %H:%M}")
print(f"=" * 65)"""))

# ═══════════════════════════════════════════════════════════════════
# Assemble notebook
# ═══════════════════════════════════════════════════════════════════
nb_out = {
    "cells": new_cells,
    "metadata": nb.get("metadata", {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    }),
    "nbformat": nb.get("nbformat", 4),
    "nbformat_minor": nb.get("nbformat_minor", 5),
}

out_path = "01_momentum_research_LOCAL.ipynb"
with open(out_path, "w") as f:
    json.dump(nb_out, f, indent=1)

# ═══════════════════════════════════════════════════════════════════
# Verify
# ═══════════════════════════════════════════════════════════════════
with open(out_path) as f:
    nb2 = json.load(f)

n_cells = len(nb2["cells"])
n_md = sum(1 for c in nb2["cells"] if c["cell_type"] == "markdown")
n_code = sum(1 for c in nb2["cells"] if c["cell_type"] == "code")

print(f"\nCreated: {out_path}")
print(f"Total cells: {n_cells} (markdown: {n_md}, code: {n_code})")

all_code = "".join(
    "".join(c["source"]) for c in nb2["cells"] if c["cell_type"] == "code"
)
all_text = "".join("".join(c["source"]) for c in nb2["cells"])

checks = {
    "All source is list": all(isinstance(c["source"], list) for c in nb2["cells"]),
    "No BQL import": "import bql" not in all_code,
    "No bq = bql.Service": "bql.Service()" not in all_code,
    "No BQuantDataLoader": "BQuantDataLoader" not in all_code,
    "Has LocalCSVLoader": "class LocalCSVLoader" in all_code,
    "Has MODE = LOCAL": 'MODE = "LOCAL"' in all_code,
    "Has DATA_DIR": "DATA_DIR" in all_code,
    "Has FALLBACK_MAP": "FALLBACK_MAP" in all_code,
    "Has YAML fallback": "hardcoded defaults" in all_code,
    "Has data inspection": "Data Inspection" in all_text,
    "Has format detection": "Detected format" in all_code,
    "Has LocalCSVLoader.load_prices": "def load_prices" in all_code,
    "Has LocalCSVLoader.get_coverage_report": "def get_coverage_report" in all_code,
    "Has raw CSV preview": "debug_raw_csv_preview" in all_code,
    "Has return distribution": "debug_local_return_distribution" in all_code,
    "Has flat series detection": "FLAT" in all_code,
    "Has normalised price plot": "Normalised Price" in all_code,
    # Strategy logic preserved
    "Has CanonicalTSMOMStrategy": "class CanonicalTSMOMStrategy" in all_code,
    "Has ex_ante_vol_ewma": "def ex_ante_vol_ewma" in all_code,
    "Has compute_bucket_scaling": "def compute_bucket_scaling" in all_code,
    "Has quick_portfolio_metrics": "def quick_portfolio_metrics" in all_code,
    "Has FIX 1 (abs weight cap)": "FIX 1" in all_text,
    "Has FIX 2 (overlay floor)": "FIX 2" in all_text,
    "Has FIX 3 (pre-listing)": "FIX 3" in all_text,
    "Has FIX 4 (lookahead)": "FIX 4" in all_text,
    "Has FIX 5 (winsorize)": "FIX 5" in all_text,
    "Has FIX 6 (CL)": "FIX 6" in all_text,
    "Has FIX 7 (bucket risk)": "FIX 7" in all_text,
    "Has FIX 8 (drawdown)": "FIX 8" in all_text,
    "Has FIX 9 (turnover)": "FIX 9" in all_text,
    "Has FIX 10 (metrics)": "FIX 10" in all_text,
    "Has DEBUG 1": "DEBUG 1" in all_text,
    "Has DEBUG 5": "DEBUG 5" in all_text,
    "Has DEBUG 8": "DEBUG 8" in all_text,
    "Has DEBUG 10": "DEBUG 10" in all_text,
    "Has sensitivity grid": "SENSITIVITY GRID" in all_code,
    "Has control tests": "CONTROL TESTS" in all_code,
    "Has Plotly visualisations": "plotly" in all_code.lower(),
    "Has _local suffix in exports": "_local_" in all_code,
    "Has LOCAL MODE in HTML": "LOCAL MODE" in all_code,
    "All code cells have outputs key": all(
        "outputs" in c for c in nb2["cells"] if c["cell_type"] == "code"
    ),
}

for name, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'} | {name}")

n_pass = sum(checks.values())
n_total = len(checks)
print(f"\n{n_pass}/{n_total} checks passed: {all(checks.values())}")

# Cell listing
print(f"\nFull cell listing:")
for i in range(n_cells):
    c = nb2["cells"][i]
    first = "".join(c["source"]).split("\n")[0][:90]
    print(f"  [{i:2d}] {c['cell_type']:8s} | {first}")
