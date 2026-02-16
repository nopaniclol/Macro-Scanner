# data/loaders.py
"""
Macro Metals System — Data Loading Layer.

Provides a unified DataLoader interface that routes requests to:
  1. A local cache (CSV / Parquet) when data already exists on disk.
  2. A Bloomberg BQL placeholder (to be wired up inside BQNT).

Reference: macro_metals_system_memory.md §5, §7.1
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"


def _load_yaml(path: Path) -> dict:
    """Read and parse a YAML file.

    Args:
        path: Absolute or relative ``Path`` to the YAML file.

    Returns:
        Parsed dict.
    """
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Bloomberg BQL placeholder
# ---------------------------------------------------------------------------

def _bql_fetch(
    ticker: str,
    fields: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch historical data from Bloomberg via BQL.

    **Placeholder** — returns an empty DataFrame with the expected schema.
    Replace the body with a real ``bql.Service()`` call when running
    inside the BQNT environment.

    Args:
        ticker: Bloomberg ticker string (e.g. ``"GC1 Comdty"``).
        fields: Bloomberg field names (e.g. ``["PX_LAST", "PX_VOLUME"]``).
        start:  ISO-format start date (``"YYYY-MM-DD"``).
        end:    ISO-format end date   (``"YYYY-MM-DD"``).

    Returns:
        DataFrame indexed by ``date`` with one column per field.
    """
    logger.warning("_bql_fetch() is a placeholder — no data returned for %s", ticker)
    idx = pd.DatetimeIndex([], name="date")
    return pd.DataFrame(index=idx, columns=fields, dtype=float)


# ---------------------------------------------------------------------------
# Local-cache helpers
# ---------------------------------------------------------------------------

def _cache_key(ticker: str, fields: list[str]) -> str:
    """Build a deterministic filename stem for a ticker + field set."""
    safe = ticker.replace(" ", "_").replace("/", "_")
    return f"{safe}__{'_'.join(sorted(fields))}"


def _read_cache(cache_dir: Path, key: str) -> Optional[pd.DataFrame]:
    """Return cached DataFrame (parquet preferred, CSV fallback) or ``None``."""
    for ext, reader in [
        (".parquet", pd.read_parquet),
        (".csv", lambda p: pd.read_csv(p, index_col=0, parse_dates=True)),
    ]:
        path = cache_dir / f"{key}{ext}"
        if path.exists():
            logger.info("Cache hit (%s): %s", ext.lstrip("."), path.name)
            return reader(path)
    return None


def _write_cache(
    cache_dir: Path, key: str, df: pd.DataFrame, fmt: str = "parquet"
) -> None:
    """Persist a DataFrame to the local cache directory.

    Args:
        cache_dir: Directory for cached files.
        key: Filename stem.
        df:  DataFrame to persist.
        fmt: ``"parquet"`` (default) or ``"csv"``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.{fmt}"
    if fmt == "parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path)
    logger.info("Cached %d rows → %s", len(df), path.name)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """Unified data access for the Macro Metals system.

    Resolves logical ticker names via ``config/tickers.yaml``, checks the
    local file cache, and falls back to the Bloomberg BQL placeholder.

    Args:
        config:   Parsed dict **or** ``Path`` / str to ``parameters.yaml``.
                  ``None`` loads the default config.
        data_dir: Root directory for the on-disk cache.  Defaults to
                  ``data/cache/`` under the project root.

    Example::

        loader = DataLoader()
        gold = loader.get_history("gc_fut_front", "2020-01-01", "2024-12-31")
        eur  = loader.get_fx_cross("EUR", "USD", "2020-01-01", "2024-12-31")
    """

    def __init__(
        self,
        config: dict | str | Path | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        if config is None:
            self._params = _load_yaml(_CONFIG_DIR / "parameters.yaml")
        elif isinstance(config, (str, Path)):
            self._params = _load_yaml(Path(config))
        else:
            self._params = config

        self._tickers: dict = _load_yaml(_CONFIG_DIR / "tickers.yaml")
        self._cache_dir = Path(data_dir) if data_dir else _DEFAULT_CACHE_DIR

    # ------------------------------------------------------------------
    # Ticker resolution
    # ------------------------------------------------------------------

    def resolve_ticker(self, logical_name: str) -> str:
        """Map a logical snake_case name to a Bloomberg ticker string.

        Searches every top-level group in ``tickers.yaml``.

        Args:
            logical_name: e.g. ``"gc_fut_front"``.

        Returns:
            Bloomberg ticker, e.g. ``"GC1 Comdty"``.

        Raises:
            KeyError: If the logical name is not found in any group.
        """
        for group in self._tickers.values():
            if isinstance(group, dict) and logical_name in group:
                return group[logical_name]
        raise KeyError(
            f"Logical ticker '{logical_name}' not found in tickers.yaml"
        )

    # ------------------------------------------------------------------
    # Core loaders
    # ------------------------------------------------------------------

    def get_history(
        self,
        ticker: str,
        start: str,
        end: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Retrieve historical data for a single instrument.

        Accepts either a **logical name** (``"gc_fut_front"``) or a raw
        Bloomberg ticker (``"GC1 Comdty"``).

        Args:
            ticker: Logical name or Bloomberg ticker.
            start:  ISO start date (``"YYYY-MM-DD"``).
            end:    ISO end date   (``"YYYY-MM-DD"``).
            fields: Bloomberg fields.  Defaults to ``["PX_LAST"]``.

        Returns:
            DataFrame indexed by date with one column per field.
        """
        if fields is None:
            fields = ["PX_LAST"]

        try:
            bbg_ticker = self.resolve_ticker(ticker)
        except KeyError:
            bbg_ticker = ticker

        key = _cache_key(bbg_ticker, fields)
        cached = _read_cache(self._cache_dir, key)
        if cached is not None:
            mask = (cached.index >= start) & (cached.index <= end)
            return cached.loc[mask]

        df = _bql_fetch(bbg_ticker, fields, start, end)
        if not df.empty:
            _write_cache(self._cache_dir, key, df)
        return df

    def get_fx_cross(
        self,
        base: str,
        quote: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Retrieve an FX spot rate.

        Constructs ``"{base}{quote} Curncy"`` per §5.3 convention (derive
        cross rates from USD pairs only).

        Args:
            base:  ISO 4217 currency code (e.g. ``"EUR"``).
            quote: ISO 4217 currency code (e.g. ``"USD"``).
            start: ISO start date.
            end:   ISO end date.

        Returns:
            DataFrame with ``PX_LAST`` column indexed by date.
        """
        bbg_ticker = f"{base}{quote} Curncy"
        return self.get_history(bbg_ticker, start, end, fields=["PX_LAST"])

    # ------------------------------------------------------------------
    # Bulk helper
    # ------------------------------------------------------------------

    def get_group(
        self,
        group_name: str,
        start: str,
        end: str,
        fields: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch history for every ticker in a ``tickers.yaml`` group.

        Args:
            group_name: Top-level key (e.g. ``"metals"``, ``"fx"``).
            start: ISO start date.
            end:   ISO end date.
            fields: Bloomberg fields.

        Returns:
            Dict mapping logical names → DataFrames.
        """
        group: dict = self._tickers.get(group_name, {})
        if not isinstance(group, dict):
            raise ValueError(f"Group '{group_name}' is not a dict of tickers")

        results: dict[str, pd.DataFrame] = {}
        for name, bbg in group.items():
            if not isinstance(bbg, str):
                continue
            results[name] = self.get_history(bbg, start, end, fields)
        return results
