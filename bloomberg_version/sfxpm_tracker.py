"""
Positioning & Momentum Tracker (BQuant / BQL)

Builds, for each configured instrument:
  POSITIONING = weighted(CFTC positioning, options skew)
  MOMENTUM    = weighted(RSI, 20d MA deviation, 100d MA deviation)

...plus a full daily signal history, forward-return analysis conditional on
signal buckets, an RSI heatmap, and CSV exports.

All scores are on a -10..+10 scale.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from datetime import date
from typing import Dict, Optional, Tuple

import bql
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

bq = bql.Service()

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 40)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# INSTRUMENT CONFIGURATION
# =============================================================================
#
# CFTC
#   Bloomberg CFTC tickers are passed to BQL verbatim -- the code does not
#   append "CFTC"/"NCL"/"NCS"/"OIN" to them.
#
#   If cftc_long_ticker + cftc_short_ticker + cftc_oi_ticker are all set, the
#   scored metric is:
#       (non-comm long - non-comm short) / open interest * 100
#   Otherwise cftc_net_ticker is scored directly (raw contract counts).
#
#   cftc_sign = -1.0 flips the series so the score reads in the direction of
#   the instrument as labelled (e.g. CFTC reports JPY; we track USD/JPY).
#
# RISK REVERSALS
#   Store only the root (e.g. "XAUUSD"). The code builds:
#       XAUUSD25R1M BGN Curncy / XAUUSD25R6M BGN Curncy
#   rr_sign = -1.0 flips the Bloomberg RR convention so that positive is
#   always bullish for the instrument as labelled.

INSTRUMENTS: Dict[str, dict] = {
    "XAU": {
        "label": "Gold",
        "asset_class": "metal",
        "price_ticker": "GCA Comdty",
        "cftc_net_ticker": "CEI1GNCN Index",
        "cftc_long_ticker": "CEI1GNCL Index",
        "cftc_short_ticker": "CEI1GNCS Index",
        "cftc_oi_ticker": "CEI1GOIN Index",
        "cftc_sign": 1.0,
        "rr_root": "XAUUSD",
        "rr_sign": 1.0,
    },
    "XAG": {
        "label": "Silver",
        "asset_class": "metal",
        "price_ticker": "SIA Comdty",
        "cftc_net_ticker": "CEI1SNCN Index",
        "cftc_long_ticker": "CEI1SNCL Index",
        "cftc_short_ticker": "CEI1SNCS Index",
        "cftc_oi_ticker": "CEI1SOIN Index",
        "cftc_sign": 1.0,
        "rr_root": "XAGUSD",
        "rr_sign": 1.0,
    },
    "XPT": {
        "label": "Platinum",
        "asset_class": "metal",
        "price_ticker": "PLA Comdty",
        "cftc_net_ticker": "NYM3PNCN Index",
        "cftc_long_ticker": "NYM3PNCL Index",
        "cftc_short_ticker": "NYM3PNCS Index",
        "cftc_oi_ticker": "NYM3POIN Index",
        "cftc_sign": 1.0,
        "rr_root": "XPTUSD",
        "rr_sign": 1.0,
    },
    "XPD": {
        "label": "Palladium",
        "asset_class": "metal",
        "price_ticker": "PAA Comdty",
        "cftc_net_ticker": "NYM2PNCN Index",
        "cftc_long_ticker": "NYM2PNCL Index",
        "cftc_short_ticker": "NYM2PNCS Index",
        "cftc_oi_ticker": "NYM2POIN Index",
        "cftc_sign": 1.0,
        "rr_root": "XPDUSD",
        "rr_sign": 1.0,
    },
    "DXY": {
        "label": "USD Index",
        "asset_class": "fx",
        "price_ticker": "DXY Index",
        "cftc_net_ticker": None,
        "cftc_long_ticker": None,
        "cftc_short_ticker": None,
        "cftc_oi_ticker": None,
        "cftc_sign": 1.0,
        "rr_root": None,
        "rr_sign": 1.0,
    },
    "EUR": {
        "label": "EUR/USD",
        "asset_class": "fx",
        "price_ticker": "EURUSD Curncy",
        "cftc_net_ticker": "IMMBENCN Index",
        "cftc_long_ticker": "IMMBENCL Index",
        "cftc_short_ticker": "IMMBENCS Index",
        "cftc_oi_ticker": "IMMBEOIN Index",
        "cftc_sign": 1.0,
        "rr_root": "EURUSD",
        "rr_sign": 1.0,
    },
    "JPY": {
        "label": "USD/JPY",
        "asset_class": "fx",
        "price_ticker": "USDJPY Curncy",
        "cftc_net_ticker": "IMM5JNCN Index",
        "cftc_long_ticker": "IMM5JNCL Index",
        "cftc_short_ticker": "IMM5JNCS Index",
        "cftc_oi_ticker": "IMM5JOIN Index",
        "cftc_sign": -1.0,   # CFTC reports JPY; invert to read as USD/JPY
        "rr_root": "USDJPY",
        "rr_sign": -1.0,     # invert BBG RR convention -> positive = bullish USD/JPY
    },
    "OIL": {
        "label": "WTI Crude Oil",
        "asset_class": "energy",
        "price_ticker": "CLA Comdty",
        "cftc_net_ticker": "NYM1CNCN Index",
        "cftc_long_ticker": "NYM1CNCL Index",
        "cftc_short_ticker": "NYM1CNCS Index",
        "cftc_oi_ticker": "NYM1COIN Index",
        "cftc_sign": 1.0,
        "rr_root": None,
        "rr_sign": 1.0,
    },
    "HG": {
        "label": "Copper",
        "asset_class": "metal",
        "price_ticker": "HGA Comdty",
        "cftc_net_ticker": "CEI1CNCN Index",
        "cftc_long_ticker": "CEI1CNCL Index",
        "cftc_short_ticker": "CEI1CNCS Index",
        "cftc_oi_ticker": "CEI1COIN Index",
        "cftc_sign": 1.0,
        "rr_root": None,
        "rr_sign": 1.0,
    },
    "AUD": {
        "label": "AUD/USD",
        "asset_class": "fx",
        "price_ticker": "AUDUSD Curncy",
        "cftc_net_ticker": "IMM6ANCN Index",
        "cftc_long_ticker": "IMM6ANCL Index",
        "cftc_short_ticker": "IMM6ANCS Index",
        "cftc_oi_ticker": "IMM6AOIN Index",
        "cftc_sign": 1.0,
        "rr_root": "AUDUSD",
        "rr_sign": 1.0,
    },
    "GBP": {
        "label": "GBP/USD",
        "asset_class": "fx",
        "price_ticker": "GBPUSD Curncy",
        "cftc_net_ticker": "IMM5PNCN Index",
        "cftc_long_ticker": "IMM5PNCL Index",
        "cftc_short_ticker": "IMM5PNCS Index",
        "cftc_oi_ticker": "IMM5POIN Index",
        "cftc_sign": 1.0,
        "rr_root": "GBPUSD",
        "rr_sign": 1.0,
    },
    "BTC": {
        "label": "Bitcoin",
        "asset_class": "crypto",
        "price_ticker": "XBT Curncy",
        "cftc_net_ticker": "CFF5RNCN Index",
        "cftc_long_ticker": "CFF5RNCL Index",
        "cftc_short_ticker": "CFF5RNCS Index",
        "cftc_oi_ticker": "CFF5ROIN Index",
        "cftc_sign": 1.0,
        "rr_root": None,
        "rr_sign": 1.0,
    },
    "TU": {
        "label": "2Y Treasury",
        "asset_class": "rates",
        "price_ticker": "TUA Comdty",
        "cftc_net_ticker": "CBT42NCN Index",
        "cftc_long_ticker": "CBT42NCL Index",
        "cftc_short_ticker": "CBT42NCS Index",
        "cftc_oi_ticker": "CBT42OIN Index",
        "cftc_sign": 1.0,
        "rr_root": None,
        "rr_sign": 1.0,
    },
    "FV": {
        "label": "5Y Treasury",
        "asset_class": "rates",
        "price_ticker": "FVA Comdty",
        "cftc_net_ticker": "CBT55NCN Index",
        "cftc_long_ticker": "CBT55NCL Index",
        "cftc_short_ticker": "CBT55NCS Index",
        "cftc_oi_ticker": "CBT55OIN Index",
        "cftc_sign": 1.0,
        "rr_root": None,
        "rr_sign": 1.0,
    },
    "TY": {
        "label": "10Y Treasury",
        "asset_class": "rates",
        "price_ticker": "TYA Comdty",
        "cftc_net_ticker": "CBT4TNCN Index",
        "cftc_long_ticker": "CBT4TNCL Index",
        "cftc_short_ticker": "CBT4TNCS Index",
        "cftc_oi_ticker": "CBT4TOIN Index",
        "cftc_sign": 1.0,
        "rr_root": None,
        "rr_sign": 1.0,
    },
}


# =============================================================================
# MODEL PARAMETERS
# =============================================================================

PRICE_HISTORY_START = "2010-01-01"
CFTC_HISTORY_START = "2010-01-01"
RR_HISTORY_START = "2018-01-01"

ZSCORE_CAP = 3.0

DEVIATION_ZSCORE_LOOKBACK = 504        # ~2 trading years
RR_ZSCORE_LOOKBACK = 756               # ~3 trading years
CFTC_CHANGE_ZSCORE_LOOKBACK = 260      # ~5 years of weekly observations
ROLLING_PERCENTILE_YEARS = 10          # rolling window for the CFTC level score

MIN_POSITIONING_COMPONENTS = 1
MIN_MOMENTUM_COMPONENTS = 2

# COT is surveyed Tuesday and released the following Friday afternoon. Shifting
# the weekly series by one release keeps a score dated to when the data was
# actually public, rather than when it was surveyed. Set to 0 to disable.
CFTC_RELEASE_LAG_WEEKS = 1

POSITIONING_WEIGHTS = {"cftc_positioning": 0.65, "options_skew": 0.35}
MOMENTUM_WEIGHTS = {"rsi": 1 / 3, "dev_20d": 1 / 3, "dev_100d": 1 / 3}


# -----------------------------------------------------------------------------
# CACHING
# -----------------------------------------------------------------------------
# Every BQL series is cached to disk. On subsequent runs only the tail is
# re-downloaded, which turns a multi-minute pull into a few seconds.
#
# The overlap re-fetches the last N calendar days on every run rather than
# resuming exactly where the cache ended. CFTC figures are revised, and the
# most recent price bar can be provisional, so the overlap lets corrections
# overwrite what was cached. New data always wins on conflict.

USE_CACHE = True
CACHE_DIR = "bql_cache"
CACHE_REFRESH_OVERLAP_DAYS = 30

# Set True to ignore existing cache and re-download everything once.
FORCE_CACHE_REFRESH = False


# -----------------------------------------------------------------------------
# SCORE HISTORY
# -----------------------------------------------------------------------------
# A running log of the report, appended once per run. This is what you actually
# saw on a given date, as distinct from the recomputed daily signal history --
# useful for spotting a regime building rather than just its endpoint.
#
# Re-running on the same date overwrites that date's snapshot rather than
# duplicating it.

SCORE_HISTORY_PATH = "sfxpm_score_history"

# Also save the full daily score series recomputed from history each run.
# Optional -- the tracker does not need it; it just lets you chart how a score
# evolved without waiting for snapshots to accumulate. Roughly 12 MB per run.
SAVE_SIGNAL_HISTORY = True

SCORE_HISTORY_COLUMNS = [
    "label", "asset_class",
    "POSITIONING", "cftc_positioning", "cftc_level_rolling", "cftc_level_full",
    "cftc_4wk_chg", "options_skew", "rr_1m", "rr_6m",
    "MOMENTUM", "rsi", "rsi_raw", "dev_20d", "dev_100d",
]


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def clean_series(series: pd.Series) -> pd.Series:
    """Coerce to numeric, drop NaNs and duplicate dates, sort chronologically."""
    out = pd.to_numeric(series, errors="coerce").dropna()
    out.index = pd.to_datetime(out.index)
    return out[~out.index.duplicated(keep="last")].sort_index()


def available_count(values) -> int:
    """Count non-null values."""
    return sum(v is not None and not pd.isna(v) for v in values)


def _round_or_none(value) -> Optional[float]:
    """Round a scalar to 2dp, or return None if it is missing."""
    return None if value is None or pd.isna(value) else round(float(value), 2)


def weighted_mean(
    items: Dict[str, Optional[float]],
    weights: Dict[str, float],
    min_components: int = 1,
) -> Optional[float]:
    """
    Weighted average over available components only.

    Missing components drop out and the remaining weights are re-normalised,
    so a partially-available composite stays on the same -10..+10 scale.
    """
    valid = {
        k: float(v)
        for k, v in items.items()
        if v is not None and not pd.isna(v) and weights.get(k, 0.0) > 0
    }

    if len(valid) < min_components:
        return None

    denominator = sum(weights[k] for k in valid)
    if denominator == 0:
        return None

    return round(sum(valid[k] * weights[k] for k in valid) / denominator, 2)


def weighted_mean_frame(
    frame: pd.DataFrame,
    weights: Dict[str, float],
    min_components: int = 1,
) -> pd.Series:
    """
    Vectorised row-wise equivalent of weighted_mean over a whole DataFrame.

    Same re-normalisation behaviour, but computed column-wise instead of with
    .apply(axis=1) -- the latter is orders of magnitude slower across ~6,000
    daily rows per instrument.
    """
    columns = [c for c in weights if c in frame.columns]
    if not columns:
        return pd.Series(np.nan, index=frame.index)

    values = frame[columns]
    weight_row = pd.Series({c: weights[c] for c in columns})

    present = values.notna()
    weight_matrix = present.mul(weight_row, axis=1)
    denominator = weight_matrix.sum(axis=1)
    numerator = (values.fillna(0.0) * weight_matrix).sum(axis=1)

    result = numerator / denominator.replace(0, np.nan)
    return result.where(present.sum(axis=1) >= min_components)


# =============================================================================
# SCORE FUNCTIONS
# =============================================================================

def _percentile_to_score(values: np.ndarray) -> float:
    """Percentile rank of the last element, mapped to -10..+10."""
    if len(values) < 20 or pd.isna(values[-1]):
        return np.nan
    percentile = np.mean(values <= values[-1]) * 100.0
    return (percentile - 50.0) / 50.0 * 10.0


def zscore_series(
    series: pd.Series,
    lookback: Optional[int] = None,
    cap: float = ZSCORE_CAP,
) -> pd.Series:
    """
    Capped z-score of a series against its own history, rescaled to -10..+10.

    lookback=None uses expanding statistics; otherwise rolling.
    """
    series = clean_series(series)

    if lookback is None:
        mean = series.expanding(min_periods=20).mean()
        std = series.expanding(min_periods=20).std()
    else:
        mean = series.rolling(lookback, min_periods=20).mean()
        std = series.rolling(lookback, min_periods=20).std()

    std = std.replace(0, np.nan)
    z = ((series - mean) / std).clip(lower=-cap, upper=cap)
    return z / cap * 10.0


def expanding_percentile_score(series: pd.Series) -> pd.Series:
    """Percentile score against all history to date (no lookahead)."""
    series = clean_series(series)
    return series.expanding(min_periods=20).apply(_percentile_to_score, raw=True)


def rolling_percentile_score(
    series: pd.Series,
    years: int = ROLLING_PERCENTILE_YEARS,
) -> pd.Series:
    """Percentile score against a rolling window, for a weekly series."""
    series = clean_series(series)
    window = max(int(years * 52), 20)
    return series.rolling(window, min_periods=20).apply(_percentile_to_score, raw=True)


# =============================================================================
# DISK IO (parquet where available, pickle otherwise)
# =============================================================================

try:
    import pyarrow  # noqa: F401
    _PARQUET_AVAILABLE = True
except ImportError:
    _PARQUET_AVAILABLE = False


def _storage_path(stem: str) -> str:
    """Append the extension for whichever storage format is available."""
    return f"{stem}.parquet" if _PARQUET_AVAILABLE else f"{stem}.pkl"


def save_frame(frame: pd.DataFrame, stem: str) -> str:
    """Write a DataFrame to parquet, falling back to pickle."""
    path = _storage_path(stem)
    if _PARQUET_AVAILABLE:
        frame.to_parquet(path)
    else:
        frame.to_pickle(path)
    return path


def load_frame(stem: str) -> Optional[pd.DataFrame]:
    """Read a DataFrame written by save_frame, or None if absent/unreadable."""
    path = _storage_path(stem)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path) if _PARQUET_AVAILABLE else pd.read_pickle(path)
    except Exception as exc:
        print(f"  [warn] Could not read {path}: {exc}")
        return None


# =============================================================================
# BQL DOWNLOAD (with disk cache)
# =============================================================================

_MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")


def _cache_stem(ticker: str) -> str:
    """Filesystem-safe stem for a ticker's cache file."""
    return os.path.join(CACHE_DIR, re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_"))


def _load_manifest() -> dict:
    """Read the cache manifest, which records the earliest start each ticker
    was requested with -- so a legitimately short history (e.g. BTC) is not
    mistaken for an incomplete cache and re-downloaded every run."""
    if not os.path.exists(_MANIFEST_PATH):
        return {}
    try:
        with open(_MANIFEST_PATH) as handle:
            return json.load(handle)
    except Exception:
        return {}


def _save_manifest(manifest: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_MANIFEST_PATH, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _extract_date_value(frame: pd.DataFrame) -> pd.Series:
    """
    Pull a date-indexed value series out of whatever shape BQL returns.

    Column naming varies by field and BQL version, so resolve by name where
    possible and fall back to the frame's own index for the dates.
    """
    lookup = {str(c).lower(): c for c in frame.columns}

    value_column = next(
        (lookup[name] for name in ("value", "px_last") if name in lookup), None
    )
    if value_column is None:
        raise ValueError(f"No value column in BQL output. Columns: {list(frame.columns)}")

    date_column = lookup.get("date")
    if date_column is not None:
        series = frame.set_index(date_column)[value_column]
    else:
        series = frame[value_column]

    if isinstance(series, pd.DataFrame):  # duplicate column names
        series = series.iloc[:, 0]

    return series


def _download_series_bql(ticker: str, start_date: str) -> pd.Series:
    """
    Download PX_LAST history for one Bloomberg ticker.

    Follows the BQuant request pattern: dates are passed via with_params as an
    absolute start/end range, and the response is read with each item's .df()
    rather than the deprecated bql.combined_df.
    """
    end_date = date.today().strftime("%Y-%m-%d")
    field = bq.data.px_last()

    request = bql.Request(
        ticker,
        {"Date": field["DATE"], "Value": field["value"]},
        with_params={"fill": "na", "dates": bq.func.range(start_date, end_date)},
    )

    response = bq.execute(request)
    frame = pd.concat([item.df() for item in response], axis=1)

    if frame.empty:
        raise ValueError(
            f"BQL returned no rows for '{ticker}' over {start_date} to {end_date}. "
            "Check the ticker is valid and has history in that window."
        )

    series = clean_series(_extract_date_value(frame))

    if series.empty:
        raise ValueError(
            f"BQL returned rows for '{ticker}' but no usable numeric values."
        )

    return series


def fetch_series_bql(ticker: str, start_date: str) -> pd.Series:
    """
    PX_LAST history for one ticker, served from disk cache where possible.

    Only the tail beyond the cached data (plus an overlap window) is
    downloaded. Falls back to a full download if the cache is missing, is
    unreadable, or does not reach far enough back.
    """
    if not USE_CACHE:
        return _download_series_bql(ticker, start_date)

    os.makedirs(CACHE_DIR, exist_ok=True)
    stem = _cache_stem(ticker)
    manifest = _load_manifest()
    entry = manifest.get(ticker, {})

    cached_frame = None if FORCE_CACHE_REFRESH else load_frame(stem)
    cached = (
        clean_series(cached_frame["px_last"])
        if cached_frame is not None and "px_last" in cached_frame.columns
        else None
    )

    requested = pd.Timestamp(start_date)
    cached_from = pd.Timestamp(entry["requested_start"]) if entry.get("requested_start") else None

    # Re-download in full if there is no usable cache, or if this run asks for
    # history earlier than the cache was ever built from.
    if cached is None or cached.empty or cached_from is None or requested < cached_from:
        series = _download_series_bql(ticker, start_date)
        effective_start = requested
    else:
        resume_from = cached.index.max() - pd.Timedelta(days=CACHE_REFRESH_OVERLAP_DAYS)
        try:
            fresh = _download_series_bql(ticker, resume_from.strftime("%Y-%m-%d"))
            # Fresh data wins on overlap, so revisions overwrite stale values.
            series = fresh.combine_first(cached).sort_index()
        except Exception as exc:
            print(f"  [warn] Incremental fetch failed for {ticker}, using cache: {exc}")
            series = cached
        effective_start = min(requested, cached_from)

    if not series.empty:
        save_frame(series.to_frame("px_last"), stem)
        manifest[ticker] = {
            "requested_start": effective_start.strftime("%Y-%m-%d"),
            "last_date": series.index.max().strftime("%Y-%m-%d"),
            "observations": int(series.size),
        }
        _save_manifest(manifest)

    return series.loc[series.index >= requested]


def clear_cache() -> None:
    """Delete every cached series and the manifest."""
    if not os.path.isdir(CACHE_DIR):
        print("No cache directory to clear.")
        return

    removed = 0
    for name in os.listdir(CACHE_DIR):
        os.remove(os.path.join(CACHE_DIR, name))
        removed += 1
    print(f"Cleared {removed} file(s) from {CACHE_DIR}/")


# =============================================================================
# MOMENTUM
# =============================================================================

EMPTY_MOMENTUM_SUMMARY = {
    "rsi": None,
    "rsi_raw": None,
    "dev_20d": None,
    "dev_100d": None,
    "MOMENTUM": None,
    "momentum_component_count": 0,
}


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI."""
    prices = clean_series(prices)
    delta = prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    rsi.loc[avg_loss == 0] = 100.0
    rsi.loc[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return rsi


def rsi_to_score(rsi: pd.Series) -> pd.Series:
    """Map RSI (0-100) to -10..+10."""
    return (rsi - 50.0) / 50.0 * 10.0


def deviation_from_ma(prices: pd.Series, window: int) -> pd.Series:
    """Percentage deviation of price from its own moving average."""
    prices = clean_series(prices)
    ma = prices.rolling(window, min_periods=window).mean()
    return (prices - ma) / ma * 100.0


def build_momentum_history(prices: pd.Series) -> Tuple[pd.DataFrame, dict]:
    """Daily momentum history plus a current-state summary."""
    if prices is None or len(prices) == 0:
        return pd.DataFrame(), dict(EMPTY_MOMENTUM_SUMMARY)

    rsi_raw = compute_rsi(prices, period=14)

    history = pd.concat(
        [
            prices.rename("price"),
            rsi_raw.rename("rsi_raw"),
            rsi_to_score(rsi_raw).rename("rsi"),
            zscore_series(
                deviation_from_ma(prices, 20), DEVIATION_ZSCORE_LOOKBACK
            ).rename("dev_20d"),
            zscore_series(
                deviation_from_ma(prices, 100), DEVIATION_ZSCORE_LOOKBACK
            ).rename("dev_100d"),
        ],
        axis=1,
    )

    history["MOMENTUM"] = weighted_mean_frame(
        history, MOMENTUM_WEIGHTS, MIN_MOMENTUM_COMPONENTS
    )

    if history.empty:
        return history, dict(EMPTY_MOMENTUM_SUMMARY)

    latest = history.iloc[-1]
    summary = {
        key: _round_or_none(latest[key])
        for key in ("rsi", "rsi_raw", "dev_20d", "dev_100d", "MOMENTUM")
    }
    summary["momentum_component_count"] = available_count(
        [latest["rsi"], latest["dev_20d"], latest["dev_100d"]]
    )
    return history, summary


# =============================================================================
# CFTC POSITIONING
# =============================================================================

EMPTY_CFTC_SUMMARY = {
    "cftc_metric": None,
    "cftc_level_full": None,
    "cftc_level_rolling": None,
    "cftc_4wk_chg": None,
    "cftc_positioning": None,
}


def fetch_cftc_history_bql(
    instrument: dict,
    start_date: str = CFTC_HISTORY_START,
) -> Tuple[pd.DataFrame, str]:
    """
    Download CFTC history.

    Prefers net non-commercial as a percentage of open interest (requires the
    long/short/OI tickers). Falls back to the direct net-position series,
    which is raw contract counts and therefore not normalised for growth in
    open interest over time -- percentile comparisons across two decades are
    considerably less meaningful on that basis.
    """
    long_ticker = instrument.get("cftc_long_ticker")
    short_ticker = instrument.get("cftc_short_ticker")
    oi_ticker = instrument.get("cftc_oi_ticker")
    net_ticker = instrument.get("cftc_net_ticker")

    if long_ticker and short_ticker and oi_ticker:
        frame = pd.concat(
            [
                fetch_series_bql(long_ticker, start_date).rename("long"),
                fetch_series_bql(short_ticker, start_date).rename("short"),
                fetch_series_bql(oi_ticker, start_date).rename("open_interest"),
            ],
            axis=1,
        ).dropna(how="all")

        frame["position_metric"] = (
            (frame["long"] - frame["short"])
            / frame["open_interest"].replace(0, np.nan)
            * 100.0
        )
        metric_name = "net_noncomm_pct_oi"

    elif net_ticker:
        frame = fetch_series_bql(net_ticker, start_date).rename("position_metric").to_frame()
        metric_name = "direct_net_series"

    else:
        return pd.DataFrame(), "none"

    frame["position_metric"] *= float(instrument.get("cftc_sign", 1.0))
    return frame.sort_index(), metric_name


def to_weekly_release_series(series: pd.Series) -> pd.Series:
    """
    Collapse CFTC data to one observation per Friday.

    Handles both genuine weekly series and daily forward-filled Bloomberg
    series. CFTC_RELEASE_LAG_WEEKS then shifts the series so a value is only
    used from the release after it was surveyed.
    """
    weekly = clean_series(series).resample("W-FRI").last().dropna()

    if CFTC_RELEASE_LAG_WEEKS:
        weekly = weekly.shift(CFTC_RELEASE_LAG_WEEKS).dropna()

    return weekly


def build_positioning_history(symbol: str, instrument: dict) -> Tuple[pd.DataFrame, dict]:
    """Weekly CFTC positioning scores plus a current-state summary."""
    try:
        raw_history, metric_name = fetch_cftc_history_bql(instrument)
    except Exception as exc:
        print(f"  [warn] CFTC fetch failed for {symbol}: {exc}")
        return pd.DataFrame(), dict(EMPTY_CFTC_SUMMARY)

    if raw_history.empty:
        return pd.DataFrame(), dict(EMPTY_CFTC_SUMMARY)

    weekly = to_weekly_release_series(raw_history["position_metric"])
    if weekly.empty:
        return pd.DataFrame(), dict(EMPTY_CFTC_SUMMARY)

    history = pd.concat(
        [
            weekly.rename("cftc_metric_raw"),
            expanding_percentile_score(weekly).rename("cftc_level_full"),
            rolling_percentile_score(weekly, ROLLING_PERCENTILE_YEARS).rename(
                "cftc_level_rolling"
            ),
            zscore_series(weekly.diff(4), CFTC_CHANGE_ZSCORE_LOOKBACK).rename(
                "cftc_4wk_chg"
            ),
        ],
        axis=1,
    )

    history["cftc_positioning"] = history[
        ["cftc_level_rolling", "cftc_4wk_chg"]
    ].mean(axis=1)

    if history.empty:
        return pd.DataFrame(), dict(EMPTY_CFTC_SUMMARY)

    latest = history.iloc[-1]
    summary = {"cftc_metric": metric_name}
    summary.update(
        {
            key: _round_or_none(latest[key])
            for key in (
                "cftc_level_full",
                "cftc_level_rolling",
                "cftc_4wk_chg",
                "cftc_positioning",
            )
        }
    )
    return history, summary


# =============================================================================
# RISK REVERSALS / OPTIONS SKEW
# =============================================================================

EMPTY_RR_SUMMARY = {
    "rr_1m": None,
    "rr_6m": None,
    "options_skew": None,
    "rr_component_count": 0,
}


def fetch_risk_reversal_history(symbol: str, instrument: dict) -> Tuple[pd.DataFrame, dict]:
    """Download and score 1M/6M 25-delta risk reversals."""
    root = instrument.get("rr_root")
    if not root:
        return pd.DataFrame(), dict(EMPTY_RR_SUMMARY)

    rr_sign = float(instrument.get("rr_sign", 1.0))
    collected, summary = [], {}

    for tenor, key in (("1M", "rr_1m"), ("6M", "rr_6m")):
        ticker = f"{root}25R{tenor} BGN Curncy"
        try:
            raw = fetch_series_bql(ticker, RR_HISTORY_START) * rr_sign
            score = zscore_series(raw, RR_ZSCORE_LOOKBACK)

            collected.append(raw.rename(f"{key}_raw"))
            collected.append(score.rename(key))

            summary[key] = _round_or_none(score.iloc[-1]) if len(score) else None
        except Exception as exc:
            print(f"  [warn] RR fetch failed for {symbol} {tenor} ({ticker}): {exc}")
            summary[key] = None

    history = pd.concat(collected, axis=1) if collected else pd.DataFrame()

    score_columns = [c for c in ("rr_1m", "rr_6m") if c in history.columns]
    if score_columns:
        history["options_skew"] = history[score_columns].mean(axis=1)
        latest_skew = history["options_skew"].dropna()
        summary["options_skew"] = (
            _round_or_none(latest_skew.iloc[-1]) if len(latest_skew) else None
        )
    else:
        summary["options_skew"] = None

    summary["rr_component_count"] = available_count(
        [summary.get("rr_1m"), summary.get("rr_6m")]
    )
    return history, summary


# =============================================================================
# COMPOSITE HISTORY
# =============================================================================

def build_composite_history(
    momentum_history: pd.DataFrame,
    cftc_history: pd.DataFrame,
    risk_reversal_history: pd.DataFrame,
) -> pd.DataFrame:
    """Join daily momentum, weekly CFTC (forward-filled) and daily skew."""
    out = momentum_history.copy()

    if not cftc_history.empty:
        out = out.join(cftc_history.reindex(out.index, method="ffill"), how="left")

    if not risk_reversal_history.empty:
        out = out.join(risk_reversal_history, how="left")

    for column in ("cftc_positioning", "options_skew"):
        if column not in out.columns:
            out[column] = np.nan

    out["POSITIONING"] = weighted_mean_frame(
        out, POSITIONING_WEIGHTS, MIN_POSITIONING_COMPONENTS
    )
    out["positioning_component_count"] = (
        out[["cftc_positioning", "options_skew"]].notna().sum(axis=1)
    )
    return out


# =============================================================================
# INTERPRETATION
# =============================================================================

def concentration_label(value) -> str:
    """Describe positioning crowding, without implying a directional forecast."""
    if value is None or pd.isna(value):
        return "unavailable"
    if value >= 5:
        return "crowded long"
    if value <= -5:
        return "crowded short"
    return "not extreme"


def momentum_label(value) -> str:
    """Describe momentum direction."""
    if value is None or pd.isna(value):
        return "unavailable"
    if value >= 5:
        return "strong positive momentum"
    if value <= -5:
        return "strong negative momentum"
    return "not extreme"


def skew_label(value) -> str:
    """Describe options skew in the direction of the named instrument."""
    if value is None or pd.isna(value):
        return "unavailable"
    if value >= 5:
        return "strong bullish skew for named instrument"
    if value <= -5:
        return "strong bearish skew for named instrument"
    return "not extreme"


LABEL_FUNCTIONS = {
    "POSITIONING": concentration_label,
    "OPTIONS_SKEW": skew_label,
    "MOMENTUM": momentum_label,
}

EXTREME_COLUMNS = {
    "POSITIONING": "POSITIONING",
    "OPTIONS_SKEW": "options_skew",
    "MOMENTUM": "MOMENTUM",
}


def flag_extremes(frame: pd.DataFrame, threshold: float = 5.0) -> None:
    """Print positioning, skew and momentum scores beyond the threshold."""
    print(f"--- Extremes (|score| >= {threshold}) ---")
    found = False

    for symbol, row in frame.iterrows():
        for name, column in EXTREME_COLUMNS.items():
            value = row.get(column)
            if value is not None and not pd.isna(value) and abs(value) >= threshold:
                print(f"  {symbol} {name}: {value:+.1f} ({LABEL_FUNCTIONS[name](value)})")
                found = True

    if not found:
        print("  No composite extremes at the selected threshold.")


# =============================================================================
# RSI HEATMAP
# =============================================================================

def plot_rsi_heatmap(frame: pd.DataFrame, ncols: int = 5, cell_size=(1.9, 1.3)) -> None:
    """Current 14-day RSI as a grid: low RSI blue, mid yellow, high RSI red."""
    data = (
        frame[["label", "rsi_raw"]]
        .dropna(subset=["rsi_raw"])
        .sort_values("rsi_raw", ascending=False)
    )

    if data.empty:
        print("No RSI data available to plot.")
        return

    nrows = -(-len(data) // ncols)
    _, axis = plt.subplots(figsize=(cell_size[0] * ncols, cell_size[1] * nrows))

    colour_map = plt.get_cmap("RdYlBu_r")
    normalise = mcolors.Normalize(vmin=0, vmax=100)

    for index, (symbol, row) in enumerate(data.iterrows()):
        row_number, column_number = divmod(index, ncols)
        y = nrows - row_number - 1
        colour = colour_map(normalise(row["rsi_raw"]))

        axis.add_patch(
            patches.Rectangle(
                (column_number, y), 1, 1,
                facecolor=colour, edgecolor="white", linewidth=2,
            )
        )

        luminance = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]
        text_colour = "white" if luminance < 0.55 else "black"
        label = row["label"] if pd.notna(row["label"]) else symbol

        axis.text(
            column_number + 0.5, y + 0.62, label,
            ha="center", va="center", fontsize=10,
            fontweight="bold", color=text_colour,
        )
        axis.text(
            column_number + 0.5, y + 0.30, f"{row['rsi_raw']:.0f}",
            ha="center", va="center", fontsize=11, color=text_colour,
        )

    axis.set_xlim(0, ncols)
    axis.set_ylim(0, nrows)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")
    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.set_title(
        "Tracked-Instrument RSI Heatmap\nCurrent 14-day RSI",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# =============================================================================
# SCORE HISTORY
# =============================================================================

def append_score_history(
    report: pd.DataFrame,
    run_date: Optional[str] = None,
    path_stem: str = SCORE_HISTORY_PATH,
) -> pd.DataFrame:
    """
    Append the current report to the running score log and save it.

    Keyed on (run_date, symbol), so re-running on the same day replaces that
    day's snapshot rather than duplicating it.
    """
    run_date = run_date or date.today().isoformat()

    columns = [c for c in SCORE_HISTORY_COLUMNS if c in report.columns]
    snapshot = report[columns].reset_index()
    snapshot.insert(0, "run_date", run_date)

    existing = load_frame(path_stem)
    combined = (
        pd.concat([existing, snapshot], ignore_index=True)
        if existing is not None and not existing.empty
        else snapshot
    )

    combined = (
        combined.drop_duplicates(subset=["run_date", "symbol"], keep="last")
        .sort_values(["run_date", "symbol"])
        .reset_index(drop=True)
    )

    save_frame(combined, path_stem)
    return combined


def load_score_history(path_stem: str = SCORE_HISTORY_PATH) -> pd.DataFrame:
    """Load the full score log, with run_date parsed as a datetime."""
    history = load_frame(path_stem)
    if history is None or history.empty:
        return pd.DataFrame()

    history = history.copy()
    history["run_date"] = pd.to_datetime(history["run_date"])
    return history.sort_values(["run_date", "symbol"])


def score_history_for(
    symbol: str,
    columns=("POSITIONING", "MOMENTUM"),
    path_stem: str = SCORE_HISTORY_PATH,
) -> pd.DataFrame:
    """One instrument's score log as a time series, indexed by run date."""
    history = load_score_history(path_stem)
    if history.empty or symbol not in set(history["symbol"]):
        return pd.DataFrame()

    subset = history[history["symbol"] == symbol].set_index("run_date")
    return subset[[c for c in columns if c in subset.columns]]


def plot_score_history(
    symbol: str,
    columns=("POSITIONING", "MOMENTUM"),
    path_stem: str = SCORE_HISTORY_PATH,
) -> None:
    """Plot how an instrument's scores have evolved across runs."""
    series = score_history_for(symbol, columns, path_stem)

    if series.empty:
        print(f"No score history stored for {symbol} yet.")
        return
    if len(series) < 2:
        print(f"Only one snapshot stored for {symbol} -- nothing to plot yet.")
        return

    _, axis = plt.subplots(figsize=(10, 4))
    for column in series.columns:
        axis.plot(series.index, series[column], marker="o", markersize=3, label=column)

    axis.axhline(0, color="grey", linewidth=0.8)
    axis.axhline(5, color="grey", linewidth=0.6, linestyle="--")
    axis.axhline(-5, color="grey", linewidth=0.6, linestyle="--")

    axis.set_ylim(-10.5, 10.5)
    axis.set_ylabel("Score")
    axis.set_title(f"{symbol} score history", fontsize=12, fontweight="bold")
    axis.legend(loc="upper left", fontsize=9)
    axis.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


# =============================================================================
# PER-INSTRUMENT PROCESSING
# =============================================================================


def process_instrument(symbol: str, instrument: dict) -> Tuple[dict, Optional[pd.DataFrame]]:
    """Fetch, score and assemble everything for one instrument."""
    print(f"Fetching {symbol} ({instrument['label']})...")

    row = {
        "symbol": symbol,
        "label": instrument["label"],
        "asset_class": instrument["asset_class"],
    }

    try:
        prices = fetch_series_bql(instrument["price_ticker"], PRICE_HISTORY_START)
        momentum_history, momentum_summary = build_momentum_history(prices)
    except Exception as exc:
        print(f"  [warn] Price/momentum fetch failed for {symbol}: {exc}")
        momentum_history = pd.DataFrame()
        momentum_summary = dict(EMPTY_MOMENTUM_SUMMARY)

    cftc_history, cftc_summary = build_positioning_history(symbol, instrument)
    rr_history, rr_summary = fetch_risk_reversal_history(symbol, instrument)

    row.update(cftc_summary)
    row.update(rr_summary)
    row.update(momentum_summary)

    row["POSITIONING"] = weighted_mean(
        {
            "cftc_positioning": row.get("cftc_positioning"),
            "options_skew": row.get("options_skew"),
        },
        POSITIONING_WEIGHTS,
        MIN_POSITIONING_COMPONENTS,
    )
    row["positioning_component_count"] = available_count(
        [row.get("cftc_positioning"), row.get("options_skew")]
    )

    row["positioning_interpretation"] = concentration_label(row["POSITIONING"])
    row["momentum_interpretation"] = momentum_label(row["MOMENTUM"])
    row["skew_interpretation"] = skew_label(row.get("options_skew"))

    if momentum_history.empty:
        return row, None

    full_history = build_composite_history(momentum_history, cftc_history, rr_history)
    full_history["symbol"] = symbol
    return row, full_history


REPORT_COLUMNS = [
    "label", "asset_class",
    "POSITIONING", "positioning_component_count", "positioning_interpretation",
    "cftc_positioning", "cftc_metric", "cftc_level_rolling", "cftc_level_full",
    "cftc_4wk_chg",
    "options_skew", "rr_component_count", "rr_1m", "rr_6m", "skew_interpretation",
    "MOMENTUM", "momentum_component_count", "momentum_interpretation",
    "rsi", "rsi_raw", "dev_20d", "dev_100d",
]

# =============================================================================
# ENHANCED TRADER UI
# Paste this block immediately after REPORT_COLUMNS and before your old main().
# This does not change the underlying positioning or momentum calculations.
# =============================================================================

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    _WIDGETS_AVAILABLE = True
except ImportError:
    _WIDGETS_AVAILABLE = False


# =============================================================================
# UI PARAMETERS
# =============================================================================

UI_FORWARD_HORIZONS = {
    "1W": 5,
    "1M": 21,
    "3M": 63,
}

UI_BUCKET_EDGES = [
    -np.inf,
    -5.0,
    -2.0,
    2.0,
    5.0,
    np.inf,
]

UI_BUCKET_LABELS = [
    "<-5",
    "-5 to -2",
    "-2 to +2",
    "+2 to +5",
    ">+5",
]

UI_CHANGE_WINDOWS = {
    "1W": 5,
    "1M": 21,
    "3M": 63,
}

UI_MIN_FORWARD_OBSERVATIONS = 20


# =============================================================================
# UI HELPERS
# =============================================================================

def _score_change(series: pd.Series, periods: int) -> float:
    """
    Change in a score across a specified number of observations.

    For the daily composite history:
        5 observations  = approximately 1 week
        21 observations = approximately 1 month
        63 observations = approximately 3 months
    """
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) <= periods:
        return np.nan

    return float(values.iloc[-1] - values.iloc[-periods - 1])


def _current_percentile(series: pd.Series) -> float:
    """
    Percentile rank of the latest score against its available history.

    Returns a value between 0 and 100.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) < 20:
        return np.nan

    latest = values.iloc[-1]
    return float((values <= latest).mean() * 100.0)


def _bucket_series(series: pd.Series) -> pd.Series:
    """
    Assign a -10 to +10 score into one of five signal buckets.
    """
    return pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=UI_BUCKET_EDGES,
        labels=UI_BUCKET_LABELS,
        include_lowest=True,
        right=False,
    )


def _quadrant_signal(
    positioning: Optional[float],
    momentum: Optional[float],
) -> str:
    """
    Convert the positioning and momentum combination into a trading regime.
    """
    if pd.isna(positioning) or pd.isna(momentum):
        return "Insufficient data"

    if positioning >= 2 and momentum >= 2:
        return "Trend long / crowded"

    if positioning <= -2 and momentum <= -2:
        return "Bear trend / crowded short"

    if positioning >= 2 and momentum <= -2:
        return "Crowded-long unwind risk"

    if positioning <= -2 and momentum >= 2:
        return "Short-squeeze watch"

    if momentum >= 2:
        return "Positive momentum"

    if momentum <= -2:
        return "Negative momentum"

    return "Neutral / transition"


def enrich_report_for_ui(
    report: pd.DataFrame,
    histories: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Add UI and trader-focused metrics to the current report.

    Added fields:
        POSITIONING_PCTL
        POSITIONING_CHG_1W
        POSITIONING_CHG_1M
        POSITIONING_CHG_3M

        MOMENTUM_PCTL
        MOMENTUM_CHG_1W
        MOMENTUM_CHG_1M
        MOMENTUM_CHG_3M

        COMBINED
        SIGNAL
        ABS_EXTREME
        RANK
    """
    out = report.copy()

    for symbol, history in histories.items():

        if symbol not in out.index:
            continue

        if history is None or history.empty:
            continue

        for score in ("POSITIONING", "MOMENTUM"):

            if score not in history.columns:
                continue

            out.loc[
                symbol,
                f"{score}_PCTL",
            ] = _current_percentile(history[score])

            for window_label, periods in UI_CHANGE_WINDOWS.items():

                out.loc[
                    symbol,
                    f"{score}_CHG_{window_label}",
                ] = _score_change(
                    history[score],
                    periods,
                )

    # Equal-weight display score.
    # This is for ranking and UI display only.
    # It does not replace the existing model composites.
    out["COMBINED"] = out[
        ["POSITIONING", "MOMENTUM"]
    ].mean(
        axis=1,
        skipna=True,
    )

    no_composite_data = (
        out[["POSITIONING", "MOMENTUM"]]
        .notna()
        .sum(axis=1)
        == 0
    )

    out.loc[no_composite_data, "COMBINED"] = np.nan

    out["SIGNAL"] = [
        _quadrant_signal(positioning, momentum)
        for positioning, momentum in zip(
            out["POSITIONING"],
            out["MOMENTUM"],
        )
    ]

    # Rank by the most extreme of positioning or momentum.
    out["ABS_EXTREME"] = out[
        ["POSITIONING", "MOMENTUM"]
    ].abs().max(axis=1)

    out["RANK"] = out["ABS_EXTREME"].rank(
        method="min",
        ascending=False,
    )

    return out.sort_values(
        ["RANK", "COMBINED"],
        ascending=[True, False],
    )


# =============================================================================
# FORWARD RETURN ANALYSIS
# =============================================================================

def forward_return_analysis(
    history: pd.DataFrame,
    min_observations: int = UI_MIN_FORWARD_OBSERVATIONS,
) -> pd.DataFrame:
    """
    Calculate historical forward returns conditional on the instrument's
    current positioning and momentum buckets.

    The current state is defined by:
        current positioning bucket
        current momentum bucket

    Historical dates matching both buckets are selected.

    Forward returns are calculated for:
        1W = 5 trading days
        1M = 21 trading days
        3M = 63 trading days

    The historical observations overlap. This is intentional for descriptive
    analysis, but it means the observation count should not be interpreted as
    a count of fully independent events.
    """
    required_columns = {
        "price",
        "POSITIONING",
        "MOMENTUM",
    }

    if history is None or history.empty:
        return pd.DataFrame()

    if not required_columns.issubset(history.columns):
        return pd.DataFrame()

    work = history[
        [
            "price",
            "POSITIONING",
            "MOMENTUM",
        ]
    ].copy().sort_index()

    work["positioning_bucket"] = _bucket_series(
        work["POSITIONING"]
    )

    work["momentum_bucket"] = _bucket_series(
        work["MOMENTUM"]
    )

    valid_states = work[
        [
            "positioning_bucket",
            "momentum_bucket",
        ]
    ].dropna()

    if valid_states.empty:
        return pd.DataFrame()

    current_positioning_bucket = (
        valid_states.iloc[-1]["positioning_bucket"]
    )

    current_momentum_bucket = (
        valid_states.iloc[-1]["momentum_bucket"]
    )

    matching_state = (
        (
            work["positioning_bucket"]
            == current_positioning_bucket
        )
        &
        (
            work["momentum_bucket"]
            == current_momentum_bucket
        )
    )

    rows = []

    for horizon_label, trading_days in UI_FORWARD_HORIZONS.items():

        forward_returns = (
            (
                work["price"].shift(-trading_days)
                / work["price"]
            )
            - 1.0
        ) * 100.0

        sample = forward_returns[
            matching_state
        ].dropna()

        observation_count = int(len(sample))

        rows.append(
            {
                "Horizon": horizon_label,

                "Avg %": (
                    float(sample.mean())
                    if observation_count
                    else np.nan
                ),

                "Median %": (
                    float(sample.median())
                    if observation_count
                    else np.nan
                ),

                "Hit rate %": (
                    float((sample > 0).mean() * 100.0)
                    if observation_count
                    else np.nan
                ),

                "Best %": (
                    float(sample.max())
                    if observation_count
                    else np.nan
                ),

                "Worst %": (
                    float(sample.min())
                    if observation_count
                    else np.nan
                ),

                "Observations": observation_count,

                "Reliable": (
                    observation_count
                    >= min_observations
                ),

                "Positioning bucket": str(
                    current_positioning_bucket
                ),

                "Momentum bucket": str(
                    current_momentum_bucket
                ),
            }
        )

    return pd.DataFrame(rows).set_index("Horizon")


def build_expectancy_table(
    histories: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build one combined forward-return table for all instruments.
    """
    rows = []

    for symbol, history in histories.items():

        statistics = forward_return_analysis(history)

        if statistics.empty:
            continue

        for horizon, row in statistics.iterrows():

            rows.append(
                {
                    "symbol": symbol,
                    "Horizon": horizon,
                    **row.to_dict(),
                }
            )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .set_index(["symbol", "Horizon"])
        .sort_index()
    )


# =============================================================================
# AUTOMATIC INTERPRETATION
# =============================================================================

def trader_interpretation(
    symbol: str,
    row: pd.Series,
    statistics: pd.DataFrame,
) -> str:
    """
    Generate a concise interpretation using the current scores and historical
    conditional return analysis.
    """
    signal = row.get(
        "SIGNAL",
        "Insufficient data",
    )

    positioning = row.get(
        "POSITIONING",
        np.nan,
    )

    momentum = row.get(
        "MOMENTUM",
        np.nan,
    )

    positioning_change_1m = row.get(
        "POSITIONING_CHG_1M",
        np.nan,
    )

    momentum_change_1m = row.get(
        "MOMENTUM_CHG_1M",
        np.nan,
    )

    text = (
        f"{symbol}: {signal}. "
        f"Positioning {positioning:+.1f} and "
        f"momentum {momentum:+.1f}."
    )

    if pd.notna(positioning_change_1m):

        if positioning_change_1m > 0:
            positioning_direction = "building"
        elif positioning_change_1m < 0:
            positioning_direction = "unwinding"
        else:
            positioning_direction = "unchanged"

        text += (
            f" Positioning is {positioning_direction} "
            f"over 1M ({positioning_change_1m:+.1f} points)."
        )

    if pd.notna(momentum_change_1m):

        if momentum_change_1m > 0:
            momentum_direction = "strengthening"
        elif momentum_change_1m < 0:
            momentum_direction = "weakening"
        else:
            momentum_direction = "unchanged"

        text += (
            f" Momentum is {momentum_direction} "
            f"over 1M ({momentum_change_1m:+.1f} points)."
        )

    if (
        statistics is not None
        and not statistics.empty
        and "1M" in statistics.index
    ):

        one_month = statistics.loc["1M"]

        observations = int(
            one_month["Observations"]
        )

        if bool(one_month["Reliable"]):

            text += (
                f" Matching historical score buckets produced "
                f"an average 1M return of "
                f"{one_month['Avg %']:+.2f}% "
                f"with a "
                f"{one_month['Hit rate %']:.0f}% "
                f"positive hit rate across "
                f"{observations} observations."
            )

        else:

            text += (
                f" The matching 1M historical sample contains "
                f"only {observations} observations, "
                f"so treat the expectancy cautiously."
            )

    return text


# =============================================================================
# COMPOSITE HEATMAP
# =============================================================================

def plot_composite_heatmap(
    ui_report: pd.DataFrame,
    axis=None,
) -> None:
    """
    Plot current positioning and momentum scores as a heatmap.

    Red   = negative
    Yellow = neutral
    Green = positive
    """
    columns = [
        "POSITIONING",
        "MOMENTUM",
    ]

    data = (
        ui_report[columns]
        .copy()
        .sort_values(
            "POSITIONING",
            ascending=False,
        )
    )

    if data.empty:
        if axis is None:
            print("No composite data available to plot.")
        return

    if axis is None:
        _, axis = plt.subplots(
            figsize=(
                6,
                max(4, len(data) * 0.42),
            )
        )

    matrix = data.fillna(0.0).to_numpy()

    image = axis.imshow(
        matrix,
        cmap="RdYlGn",
        vmin=-10,
        vmax=10,
        aspect="auto",
    )

    axis.set_xticks(
        range(len(columns)),
        [
            "Positioning",
            "Momentum",
        ],
    )

    axis.set_yticks(
        range(len(data)),
        data.index,
    )

    for row_number in range(len(data)):

        for column_number in range(len(columns)):

            value = data.iloc[
                row_number,
                column_number,
            ]

            label = (
                "NA"
                if pd.isna(value)
                else f"{value:+.1f}"
            )

            text_colour = (
                "white"
                if pd.notna(value)
                and abs(value) > 5
                else "black"
            )

            axis.text(
                column_number,
                row_number,
                label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=text_colour,
            )

    axis.set_title(
        "Composite Score Heatmap",
        fontweight="bold",
    )

    if axis.figure:
        colour_bar = axis.figure.colorbar(
            image,
            ax=axis,
            fraction=0.035,
            pad=0.03,
        )

        colour_bar.set_label(
            "Score",
            rotation=270,
            labelpad=12,
        )


# =============================================================================
# POSITIONING VS MOMENTUM QUADRANT
# =============================================================================

def plot_positioning_momentum_quadrant(
    ui_report: pd.DataFrame,
    axis=None,
) -> None:
    """
    Positioning on the x-axis and momentum on the y-axis.

    Upper right:
        positive positioning and positive momentum

    Upper left:
        negative positioning and positive momentum
        potential short-squeeze setup

    Lower right:
        positive positioning and negative momentum
        potential crowded-long unwind

    Lower left:
        negative positioning and negative momentum
    """
    data = ui_report.dropna(
        subset=[
            "POSITIONING",
            "MOMENTUM",
        ]
    )

    if data.empty:
        if axis is None:
            print("No positioning/momentum data available.")
        return

    if axis is None:
        _, axis = plt.subplots(
            figsize=(8, 7)
        )

    asset_colours = {
        "metal": "#D4A017",
        "fx": "#4472C4",
        "energy": "#70AD47",
        "rates": "#7030A0",
        "crypto": "#ED7D31",
    }

    plotted_asset_classes = set()

    for symbol, row in data.iterrows():

        asset_class = row.get(
            "asset_class",
            "other",
        )

        colour = asset_colours.get(
            asset_class,
            "#666666",
        )

        legend_label = (
            asset_class.title()
            if asset_class not in plotted_asset_classes
            else None
        )

        plotted_asset_classes.add(asset_class)

        axis.scatter(
            row["POSITIONING"],
            row["MOMENTUM"],
            s=100,
            color=colour,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label=legend_label,
        )

        axis.annotate(
            symbol,
            (
                row["POSITIONING"],
                row["MOMENTUM"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    axis.axvspan(
        -10.5,
        0,
        ymin=0.5,
        ymax=1.0,
        color="#E2F0D9",
        alpha=0.18,
    )

    axis.axvspan(
        0,
        10.5,
        ymin=0.5,
        ymax=1.0,
        color="#C6E0B4",
        alpha=0.18,
    )

    axis.axvspan(
        -10.5,
        0,
        ymin=0.0,
        ymax=0.5,
        color="#F4CCCC",
        alpha=0.18,
    )

    axis.axvspan(
        0,
        10.5,
        ymin=0.0,
        ymax=0.5,
        color="#FCE4D6",
        alpha=0.22,
    )

    axis.axvline(
        0,
        color="black",
        linewidth=0.8,
    )

    axis.axhline(
        0,
        color="black",
        linewidth=0.8,
    )

    for threshold in (-5, 5):

        axis.axvline(
            threshold,
            color="grey",
            linewidth=0.6,
            linestyle="--",
        )

        axis.axhline(
            threshold,
            color="grey",
            linewidth=0.6,
            linestyle="--",
        )

    axis.set_xlim(
        -10.5,
        10.5,
    )

    axis.set_ylim(
        -10.5,
        10.5,
    )

    axis.set_xlabel(
        "POSITIONING  ← short / long →"
    )

    axis.set_ylabel(
        "MOMENTUM  ← negative / positive →"
    )

    axis.set_title(
        "Positioning vs Momentum",
        fontweight="bold",
    )

    axis.grid(alpha=0.15)

    axis.text(
        9.8,
        9.5,
        "TREND LONG",
        ha="right",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="#267326",
    )

    axis.text(
        -9.8,
        9.5,
        "SHORT SQUEEZE",
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="#267326",
    )

    axis.text(
        9.8,
        -9.5,
        "LONG UNWIND",
        ha="right",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#9C0006",
    )

    axis.text(
        -9.8,
        -9.5,
        "BEAR TREND",
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#9C0006",
    )

    axis.legend(
        loc="lower left",
        fontsize=8,
        frameon=True,
    )


# =============================================================================
# OPPORTUNITY TABLES
# =============================================================================

def opportunity_tables(
    ui_report: pd.DataFrame,
    number_to_show: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Build ranked opportunity and risk tables.
    """
    columns = [
        "label",
        "POSITIONING",
        "POSITIONING_CHG_1M",
        "MOMENTUM",
        "MOMENTUM_CHG_1M",
        "COMBINED",
        "SIGNAL",
    ]

    available_columns = [
        column
        for column in columns
        if column in ui_report.columns
    ]

    strongest_bullish = (
        ui_report
        .nlargest(
            number_to_show,
            "COMBINED",
        )
        [available_columns]
    )

    strongest_bearish = (
        ui_report
        .nsmallest(
            number_to_show,
            "COMBINED",
        )
        [available_columns]
    )

    short_squeeze = (
        ui_report[
            (ui_report["POSITIONING"] <= -2)
            &
            (ui_report["MOMENTUM"] >= 2)
        ]
        [available_columns]
        .sort_values(
            "MOMENTUM",
            ascending=False,
        )
    )

    crowded_long_risk = (
        ui_report[
            (ui_report["POSITIONING"] >= 2)
            &
            (ui_report["MOMENTUM"] <= -2)
        ]
        [available_columns]
        .sort_values(
            "POSITIONING",
            ascending=False,
        )
    )

    strongest_momentum = (
        ui_report
        .nlargest(
            number_to_show,
            "MOMENTUM",
        )
        [available_columns]
    )

    weakest_momentum = (
        ui_report
        .nsmallest(
            number_to_show,
            "MOMENTUM",
        )
        [available_columns]
    )

    return {
        "Strongest bullish": strongest_bullish,
        "Strongest bearish": strongest_bearish,
        "Short-squeeze watch": short_squeeze,
        "Crowded-long risk": crowded_long_risk,
        "Strongest momentum": strongest_momentum,
        "Weakest momentum": weakest_momentum,
    }


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def plot_main_dashboard(
    ui_report: pd.DataFrame,
) -> None:
    """
    Display the one-screen morning dashboard.

    Layout:
        Left:
            positioning vs momentum quadrant

        Upper right:
            composite heatmap

        Lower right:
            ranked actionable signals
    """
    figure = plt.figure(
        figsize=(17, 10)
    )

    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[
            1.2,
            1.0,
        ],
        height_ratios=[
            1,
            1,
        ],
    )

    quadrant_axis = figure.add_subplot(
        grid[:, 0]
    )

    heatmap_axis = figure.add_subplot(
        grid[0, 1]
    )

    summary_axis = figure.add_subplot(
        grid[1, 1]
    )

    plot_positioning_momentum_quadrant(
        ui_report,
        quadrant_axis,
    )

    plot_composite_heatmap(
        ui_report,
        heatmap_axis,
    )

    ranked = (
        ui_report
        .sort_values(
            "ABS_EXTREME",
            ascending=False,
        )
        .head(8)
    )

    summary_axis.axis("off")

    summary_lines = [
        "MOST ACTIONABLE CURRENT STATES",
        "",
    ]

    for symbol, row in ranked.iterrows():

        positioning = row.get(
            "POSITIONING",
            np.nan,
        )

        momentum = row.get(
            "MOMENTUM",
            np.nan,
        )

        positioning_text = (
            "   NA"
            if pd.isna(positioning)
            else f"{positioning:+5.1f}"
        )

        momentum_text = (
            "   NA"
            if pd.isna(momentum)
            else f"{momentum:+5.1f}"
        )

        summary_lines.append(
            f"{symbol:<5}  "
            f"Pos {positioning_text}  "
            f"Mom {momentum_text}   "
            f"{row['SIGNAL']}"
        )

    summary_axis.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        va="top",
        family="monospace",
        fontsize=11,
    )

    figure.suptitle(
        (
            "Positioning & Momentum Trader Dashboard"
            f" | {date.today().isoformat()}"
        ),
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(
        rect=(0, 0, 1, 0.96)
    )

    plt.show()


# =============================================================================
# INSTRUMENT DETAIL VIEW
# =============================================================================

def plot_instrument_detail(
    symbol: str,
    ui_report: pd.DataFrame,
    histories: Dict[str, pd.DataFrame],
) -> None:
    """
    Plot price, positioning and momentum history for one instrument.
    """
    history = histories.get(symbol)

    if history is None or history.empty:
        print(
            f"No signal history available for {symbol}."
        )
        return

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1.3,
                1,
                1,
            ]
        },
    )

    price_axis = axes[0]
    positioning_axis = axes[1]
    momentum_axis = axes[2]

    if "price" in history.columns:

        history["price"].plot(
            ax=price_axis,
            color="black",
            linewidth=1.2,
        )

    price_axis.set_title(
        f"{symbol} | Price and Signal History",
        fontweight="bold",
    )

    price_axis.set_ylabel("Price")

    positioning_lines = (
        (
            "POSITIONING",
            "#4472C4",
        ),
        (
            "cftc_positioning",
            "#70AD47",
        ),
        (
            "options_skew",
            "#ED7D31",
        ),
    )

    for column, colour in positioning_lines:

        if column in history.columns:

            history[column].plot(
                ax=positioning_axis,
                label=column,
                color=colour,
                linewidth=1.0,
            )

    positioning_axis.set_ylabel(
        "Positioning"
    )

    positioning_axis.set_ylim(
        -10.5,
        10.5,
    )

    positioning_axis.legend(
        loc="upper left",
        ncol=3,
        fontsize=8,
    )

    momentum_lines = (
        (
            "MOMENTUM",
            "#C00000",
        ),
        (
            "rsi",
            "#7030A0",
        ),
        (
            "dev_20d",
            "#5B9BD5",
        ),
        (
            "dev_100d",
            "#A5A5A5",
        ),
    )

    for column, colour in momentum_lines:

        if column in history.columns:

            history[column].plot(
                ax=momentum_axis,
                label=column,
                color=colour,
                linewidth=1.0,
            )

    momentum_axis.set_ylabel(
        "Momentum"
    )

    momentum_axis.set_ylim(
        -10.5,
        10.5,
    )

    momentum_axis.legend(
        loc="upper left",
        ncol=4,
        fontsize=8,
    )

    for axis in (
        positioning_axis,
        momentum_axis,
    ):

        axis.axhline(
            0,
            color="grey",
            linewidth=0.7,
        )

        axis.axhline(
            5,
            color="grey",
            linewidth=0.5,
            linestyle="--",
        )

        axis.axhline(
            -5,
            color="grey",
            linewidth=0.5,
            linestyle="--",
        )

    for axis in axes:
        axis.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


# =============================================================================
# STYLED DATAFRAME DISPLAY
# =============================================================================

def _display_styled_frame(
    frame: pd.DataFrame,
    decimals: int = 2,
) -> None:
    """
    Display a DataFrame with score heatmap formatting in BQuant/Jupyter.
    """
    if frame is None or frame.empty:
        display(
            pd.DataFrame(
                {"Message": ["No matching data available."]}
            )
        )
        return

    numeric_columns = frame.select_dtypes(
        include=[np.number]
    ).columns

    format_dictionary = {
        column: f"{{:.{decimals}f}}"
        for column in numeric_columns
    }

    score_columns = [
        column
        for column in (
            "POSITIONING",
            "MOMENTUM",
            "COMBINED",
        )
        if column in frame.columns
    ]

    styled = frame.style.format(
        format_dictionary,
        na_rep="-",
    )

    if score_columns:

        styled = styled.background_gradient(
            cmap="RdYlGn",
            vmin=-10,
            vmax=10,
            subset=score_columns,
        )

    display(styled)


# =============================================================================
# BQUANT FOUR-TAB UI
# =============================================================================

def launch_tracker_ui(
    report: pd.DataFrame,
    histories: Dict[str, pd.DataFrame],
):
    """
    Launch the four-tab BQuant/Jupyter interface.

    Tabs:
        Dashboard
        Instrument detail
        Forward returns
        Opportunities

    Falls back to a static matplotlib dashboard if ipywidgets is unavailable.
    """
    ui_report = enrich_report_for_ui(
        report,
        histories,
    )

    expectancy = build_expectancy_table(
        histories
    )

    if not _WIDGETS_AVAILABLE:

        print(
            "ipywidgets is unavailable. "
            "Showing the static dashboard instead."
        )

        plot_main_dashboard(ui_report)

        return ui_report, expectancy

    symbols = [
        symbol
        for symbol in ui_report.index
        if symbol in histories
    ]

    if not symbols:

        print(
            "No instrument histories are available "
            "for the interactive interface."
        )

        plot_main_dashboard(ui_report)

        return ui_report, expectancy

    instrument_selector = widgets.Dropdown(
        options=symbols,
        value=symbols[0],
        description="Instrument:",
        layout=widgets.Layout(
            width="320px"
        ),
    )

    refresh_button = widgets.Button(
        description="Refresh selected",
        button_style="primary",
        icon="refresh",
    )

    dashboard_output = widgets.Output()
    detail_output = widgets.Output()
    forward_output = widgets.Output()
    opportunities_output = widgets.Output()

    def render_dashboard() -> None:

        with dashboard_output:

            clear_output(wait=True)

            plot_main_dashboard(
                ui_report
            )

            dashboard_columns = [
                "RANK",
                "label",
                "asset_class",

                "POSITIONING",
                "POSITIONING_PCTL",
                "POSITIONING_CHG_1W",
                "POSITIONING_CHG_1M",
                "POSITIONING_CHG_3M",

                "MOMENTUM",
                "MOMENTUM_PCTL",
                "MOMENTUM_CHG_1W",
                "MOMENTUM_CHG_1M",
                "MOMENTUM_CHG_3M",

                "COMBINED",
                "SIGNAL",
            ]

            dashboard_columns = [
                column
                for column in dashboard_columns
                if column in ui_report.columns
            ]

            _display_styled_frame(
                ui_report[dashboard_columns]
            )

    def render_selected(*_) -> None:

        symbol = instrument_selector.value

        if symbol is None:
            return

        history = histories.get(symbol)

        statistics = forward_return_analysis(
            history
        )

        with detail_output:

            clear_output(wait=True)

            plot_instrument_detail(
                symbol,
                ui_report,
                histories,
            )

            interpretation = trader_interpretation(
                symbol,
                ui_report.loc[symbol],
                statistics,
            )

            print(interpretation)

        with forward_output:

            clear_output(wait=True)

            print(
                f"{symbol} | Forward Returns Conditional "
                f"on Current Score Buckets"
            )

            print()

            print(
                "Returns are historical outcomes rather than forecasts."
            )

            print(
                "Reliable=True requires at least "
                f"{UI_MIN_FORWARD_OBSERVATIONS} observations."
            )

            print(
                "Historical observations can overlap, so the count "
                "is not a count of fully independent events."
            )

            print()

            _display_styled_frame(
                statistics
            )

    def render_opportunities() -> None:

        with opportunities_output:

            clear_output(wait=True)

            tables = opportunity_tables(
                ui_report
            )

            for heading, table in tables.items():

                print()
                print(heading.upper())
                print("=" * len(heading))

                _display_styled_frame(
                    table
                )

    instrument_selector.observe(
        render_selected,
        names="value",
    )

    refresh_button.on_click(
        render_selected
    )

    render_dashboard()
    render_selected()
    render_opportunities()

    tabs = widgets.Tab(
        children=[
            dashboard_output,
            detail_output,
            forward_output,
            opportunities_output,
        ]
    )

    tab_titles = [
        "Dashboard",
        "Instrument detail",
        "Forward returns",
        "Opportunities",
    ]

    for index, title in enumerate(tab_titles):

        tabs.set_title(
            index,
            title,
        )

    controls = widgets.HBox(
        [
            instrument_selector,
            refresh_button,
        ]
    )

    display(
        widgets.VBox(
            [
                controls,
                tabs,
            ]
        )
    )

    return ui_report, expectancy


# =============================================================================
# REPLACEMENT MAIN FUNCTION
# =============================================================================

def main() -> None:
    """
    Run the tracker, export the results and display the enhanced UI.
    """
    rows = []
    histories = {}

    for symbol, instrument in INSTRUMENTS.items():

        row, history = process_instrument(
            symbol,
            instrument,
        )

        rows.append(row)

        if history is not None:
            histories[symbol] = history

    report = (
        pd.DataFrame(rows)
        .set_index("symbol")
        .reindex(columns=REPORT_COLUMNS)
    )

    ui_report = enrich_report_for_ui(
        report,
        histories,
    )

    expectancy = build_expectancy_table(
        histories
    )

    print("\nCurrent ranked report:")

    console_columns = [
        "RANK",
        "label",

        "POSITIONING",
        "POSITIONING_PCTL",
        "POSITIONING_CHG_1W",
        "POSITIONING_CHG_1M",
        "POSITIONING_CHG_3M",

        "MOMENTUM",
        "MOMENTUM_PCTL",
        "MOMENTUM_CHG_1W",
        "MOMENTUM_CHG_1M",
        "MOMENTUM_CHG_3M",

        "COMBINED",
        "SIGNAL",
    ]

    console_columns = [
        column
        for column in console_columns
        if column in ui_report.columns
    ]

    print(
        ui_report[
            console_columns
        ].round(2)
    )

    print()

    flag_extremes(report)

    run_date = date.today().isoformat()

    # -------------------------------------------------------------------------
    # ENHANCED CURRENT REPORT EXPORT
    # -------------------------------------------------------------------------

    report_path = (
        f"sfxpm_report_{run_date}.csv"
    )

    ui_report.to_csv(
        report_path
    )

    print(
        f"Saved {report_path}"
    )

    # -------------------------------------------------------------------------
    # FORWARD EXPECTANCY EXPORT
    # -------------------------------------------------------------------------

    expectancy_path = (
        f"sfxpm_forward_expectancy_{run_date}.csv"
    )

    expectancy.to_csv(
        expectancy_path
    )

    print(
        f"Saved {expectancy_path}"
    )

    # -------------------------------------------------------------------------
    # SCORE SNAPSHOT HISTORY
    # -------------------------------------------------------------------------

    score_log = append_score_history(
        report,
        run_date,
    )

    print(
        f"Saved {_storage_path(SCORE_HISTORY_PATH)} "
        f"({score_log['run_date'].nunique()} snapshot date(s), "
        f"{len(score_log)} rows)"
    )

    # -------------------------------------------------------------------------
    # FULL DAILY SIGNAL HISTORY
    # -------------------------------------------------------------------------

    if SAVE_SIGNAL_HISTORY and histories:

        combined_history = pd.concat(
            histories,
            names=[
                "instrument",
                "date",
            ],
        )

        history_path = save_frame(
            combined_history.reset_index(),
            f"sfxpm_daily_signal_history_{run_date}",
        )

        print(
            f"Saved {history_path}"
        )

    # -------------------------------------------------------------------------
    # LAUNCH INTERACTIVE UI
    # -------------------------------------------------------------------------

    launch_tracker_ui(
        report,
        histories,
    )


if __name__ == "__main__":
    main()
