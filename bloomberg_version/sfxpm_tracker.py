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

PRICE_HISTORY_START = "2000-01-01"
CFTC_HISTORY_START = "2000-01-01"
RR_HISTORY_START = "2018-01-01"

ZSCORE_CAP = 3.0

DEVIATION_ZSCORE_LOOKBACK = 504        # ~2 trading years
RR_ZSCORE_LOOKBACK = 756               # ~3 trading years
CFTC_CHANGE_ZSCORE_LOOKBACK = 260      # ~5 years of weekly observations
ROLLING_PERCENTILE_YEARS = 10          # rolling window for the CFTC level score

MIN_POSITIONING_COMPONENTS = 1
MIN_MOMENTUM_COMPONENTS = 2

# COT is surveyed Tuesday and released the following Friday afternoon. The
# weekly series is labelled with the Friday, but treating it as tradable at
# that Friday's close is optimistic (release is 15:30 ET). Shifting one extra
# business day keeps the forward-return study honest. Set to 0 to disable.
CFTC_RELEASE_LAG_WEEKS = 1

FORWARD_HORIZONS = (1, 5, 20, 60)

POSITIONING_WEIGHTS = {"cftc_positioning": 0.65, "options_skew": 0.35}
MOMENTUM_WEIGHTS = {"rsi": 1 / 3, "dev_20d": 1 / 3, "dev_100d": 1 / 3}

SCORE_BUCKET_BINS = [-np.inf, -5.0, -2.0, 2.0, 5.0, np.inf]
SCORE_BUCKET_LABELS = ["very_negative", "negative", "neutral", "positive", "very_positive"]


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


def score_bucket(series: pd.Series) -> pd.Categorical:
    """Bucket -10..+10 scores into five directional bands."""
    return pd.cut(
        series,
        bins=SCORE_BUCKET_BINS,
        labels=SCORE_BUCKET_LABELS,
        include_lowest=True,
    )


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


def _download_series_bql(ticker: str, start_date: str) -> pd.Series:
    """Download PX_LAST history for one Bloomberg ticker."""
    date_range = bq.func.range(start_date, "0d")
    request = bql.Request(ticker, {"px_last": bq.data.px_last(dates=date_range)})
    frame = bql.combined_df(bq.execute(request))

    if not {"DATE", "px_last"}.issubset(frame.columns):
        raise ValueError(
            f"Unexpected BQL output for {ticker}. Columns: {list(frame.columns)}"
        )

    return clean_series(frame[["DATE", "px_last"]].set_index("DATE")["px_last"])


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
# FORWARD-RETURN ANALYSIS
# =============================================================================

def add_forward_returns(history: pd.DataFrame, horizons=FORWARD_HORIZONS) -> pd.DataFrame:
    """Add forward percentage returns for each trading-day horizon."""
    out = history.copy()
    for horizon in horizons:
        out[f"fwd_{horizon}d_return_pct"] = (
            out["price"].shift(-horizon) / out["price"] - 1.0
        ) * 100.0
    return out


def _summarise_returns(values: pd.Series) -> dict:
    """Mean/median/hit-rate/std for one set of forward returns."""
    values = values.dropna()
    return {
        "observations": int(values.count()),
        "mean_return_pct": values.mean() if len(values) else np.nan,
        "median_return_pct": values.median() if len(values) else np.nan,
        "positive_hit_rate_pct": (values > 0).mean() * 100.0 if len(values) else np.nan,
        "return_std_pct": values.std() if len(values) else np.nan,
    }


def conditional_forward_return_table(
    history: pd.DataFrame,
    signal: str,
    horizons=FORWARD_HORIZONS,
) -> pd.DataFrame:
    """Forward returns conditional on a single signal's bucket."""
    forward_columns = [f"fwd_{h}d_return_pct" for h in horizons]
    data = history[[signal] + forward_columns].copy()
    data["bucket"] = score_bucket(data[signal])

    rows = []
    for bucket, group in data.groupby("bucket", observed=False):
        for horizon in horizons:
            rows.append(
                {
                    "signal": signal,
                    "bucket": str(bucket),
                    "horizon_days": horizon,
                    **_summarise_returns(group[f"fwd_{horizon}d_return_pct"]),
                }
            )
    return pd.DataFrame(rows)


def combined_regime_table(history: pd.DataFrame, horizons=FORWARD_HORIZONS) -> pd.DataFrame:
    """Forward returns conditional on the joint positioning/momentum regime."""
    data = history.copy()
    data["positioning_bucket"] = score_bucket(data["POSITIONING"])
    data["momentum_bucket"] = score_bucket(data["MOMENTUM"])

    rows = []
    grouped = data.groupby(["positioning_bucket", "momentum_bucket"], observed=False)
    for (positioning_bucket, momentum_bucket), group in grouped:
        for horizon in horizons:
            stats = _summarise_returns(group[f"fwd_{horizon}d_return_pct"])
            stats.pop("return_std_pct", None)
            rows.append(
                {
                    "positioning_bucket": str(positioning_bucket),
                    "momentum_bucket": str(momentum_bucket),
                    "horizon_days": horizon,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


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

EMPTY_MOMENTUM_SUMMARY = {
    "rsi": None,
    "rsi_raw": None,
    "dev_20d": None,
    "dev_100d": None,
    "MOMENTUM": None,
    "momentum_component_count": 0,
}

BACKTEST_SIGNALS = ("POSITIONING", "MOMENTUM", "cftc_positioning", "options_skew")


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

    full_history = add_forward_returns(
        build_composite_history(momentum_history, cftc_history, rr_history)
    )
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


def main() -> None:
    rows, histories, backtests, regimes = [], {}, [], []

    for symbol, instrument in INSTRUMENTS.items():
        row, history = process_instrument(symbol, instrument)
        rows.append(row)

        if history is None:
            continue

        histories[symbol] = history

        for signal in BACKTEST_SIGNALS:
            if signal in history.columns and history[signal].notna().any():
                table = conditional_forward_return_table(history, signal)
                table.insert(0, "symbol", symbol)
                backtests.append(table)

        if history["POSITIONING"].notna().any() and history["MOMENTUM"].notna().any():
            table = combined_regime_table(history)
            table.insert(0, "symbol", symbol)
            regimes.append(table)

    report = pd.DataFrame(rows).set_index("symbol").reindex(columns=REPORT_COLUMNS)

    print("\nCurrent report:")
    print(report.round(2))
    print()
    flag_extremes(report)
    plot_rsi_heatmap(report, ncols=5)

    backtest_report = pd.concat(backtests, ignore_index=True) if backtests else pd.DataFrame()
    regime_report = pd.concat(regimes, ignore_index=True) if regimes else pd.DataFrame()
    combined_history = (
        pd.concat(histories, names=["instrument", "date"]) if histories else pd.DataFrame()
    )

    if not backtest_report.empty:
        print("\nSingle-signal forward-return analysis:")
        print(backtest_report.round(2).to_string(index=False))

    if not regime_report.empty:
        print("\nCombined positioning and momentum regimes:")
        print(regime_report.round(2).to_string(index=False))

    run_date = date.today().isoformat()

    print()
    for frame, path, keep_index in [
        (report, f"sfxpm_report_{run_date}.csv", True),
        (backtest_report, f"sfxpm_signal_backtest_{run_date}.csv", False),
        (regime_report, f"sfxpm_combined_regimes_{run_date}.csv", False),
    ]:
        frame.to_csv(path, index=keep_index)
        print(f"Saved {path}")

    # Daily signal history is large (tens of MB as CSV) -- store it columnar.
    if not combined_history.empty:
        path = save_frame(
            combined_history.reset_index(),
            f"sfxpm_daily_signal_history_{run_date}",
        )
        print(f"Saved {path}")

    score_log = append_score_history(report, run_date)
    print(
        f"Saved {_storage_path(SCORE_HISTORY_PATH)} "
        f"({score_log['run_date'].nunique()} snapshot date(s), {len(score_log)} rows)"
    )


if __name__ == "__main__":
    main()
