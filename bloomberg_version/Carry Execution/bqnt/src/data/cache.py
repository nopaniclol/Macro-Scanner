"""
In-memory caching for BQNT Carry Execution.

Handles session timeouts by caching BQL query results.
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional
from ..config.parameters import DATA_PARAMS


class BQNTCache:
    """
    Simple in-memory cache with TTL (time-to-live).

    Designed to reduce BQL queries and handle BQNT session timeouts.
    """

    def __init__(self, ttl_minutes: Optional[int] = None):
        """
        Initialize cache.

        Args:
            ttl_minutes: Time-to-live in minutes (default from DATA_PARAMS)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl_seconds = (ttl_minutes or DATA_PARAMS['cache_ttl_minutes']) * 60
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0,
        }

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            self._stats['misses'] += 1
            return None

        entry = self._cache[key]
        if self._is_expired(entry):
            self._evict(key)
            self._stats['misses'] += 1
            return None

        self._stats['hits'] += 1
        return entry['value']

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional custom TTL for this entry
        """
        self._cache[key] = {
            'value': value,
            'timestamp': time.time(),
            'ttl': ttl_seconds or self._ttl_seconds,
        }
        self._stats['sets'] += 1

    def has(self, key: str) -> bool:
        """
        Check if key exists and is not expired.

        Args:
            key: Cache key

        Returns:
            True if key exists and is valid
        """
        if key not in self._cache:
            return False

        entry = self._cache[key]
        if self._is_expired(entry):
            self._evict(key)
            return False

        return True

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0,
        }

    def cleanup(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if self._is_expired(entry)
        ]

        for key in expired_keys:
            self._evict(key)

        return len(expired_keys)

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        age = time.time() - entry['timestamp']
        return age > entry['ttl']

    def _evict(self, key: str) -> None:
        """Evict key from cache."""
        if key in self._cache:
            del self._cache[key]
            self._stats['evictions'] += 1

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0

        return {
            **self._stats,
            'size': len(self._cache),
            'hit_rate': hit_rate,
        }

    @property
    def size(self) -> int:
        """Get number of cached items."""
        return len(self._cache)

    def keys(self) -> list:
        """Get all cache keys."""
        return list(self._cache.keys())

    def info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        self.cleanup()  # Clean up expired entries first

        return {
            'size': self.size,
            'stats': self.stats,
            'keys': self.keys(),
            'ttl_minutes': self._ttl_seconds / 60,
        }


class DataCache(BQNTCache):
    """
    Specialized cache for market data with date-aware keys.
    """

    def __init__(self, ttl_minutes: Optional[int] = None):
        super().__init__(ttl_minutes)

    def get_price_key(self, tickers: list, days: int) -> str:
        """Generate cache key for price data."""
        ticker_str = '-'.join(sorted(tickers))
        return f"prices_{ticker_str}_{days}"

    def get_ohlcv_key(self, ticker: str, days: int) -> str:
        """Generate cache key for OHLCV data."""
        return f"ohlcv_{ticker}_{days}"

    def get_signal_key(self, signal_type: str, date: datetime) -> str:
        """Generate cache key for signal data."""
        date_str = date.strftime('%Y-%m-%d')
        return f"signal_{signal_type}_{date_str}"

    def invalidate_prices(self) -> int:
        """Invalidate all price data cache entries."""
        price_keys = [k for k in self.keys() if k.startswith('prices_')]
        for key in price_keys:
            self.delete(key)
        return len(price_keys)

    def invalidate_ohlcv(self) -> int:
        """Invalidate all OHLCV data cache entries."""
        ohlcv_keys = [k for k in self.keys() if k.startswith('ohlcv_')]
        for key in ohlcv_keys:
            self.delete(key)
        return len(ohlcv_keys)

    def invalidate_signals(self) -> int:
        """Invalidate all signal cache entries."""
        signal_keys = [k for k in self.keys() if k.startswith('signal_')]
        for key in signal_keys:
            self.delete(key)
        return len(signal_keys)


# =============================================================================
# SINGLETON CACHE INSTANCE
# =============================================================================
_global_cache: Optional[DataCache] = None


def get_cache() -> DataCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = DataCache()
    return _global_cache


def reset_cache() -> None:
    """Reset global cache instance."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
    _global_cache = DataCache()
