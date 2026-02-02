"""
API Client for Stock Scanner

Integrates multiple free data sources:
- Finnhub: Earnings calendar, earnings surprises, real-time quotes
- Alpha Vantage: Earnings data, fundamentals
- yfinance: OHLCV data, historical prices (primary source for price data)
- Financial Modeling Prep: PEAD tracking (optional)

Rate limiting and caching are implemented to stay within free tier limits.
"""

import os
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from functools import wraps
import requests
import pandas as pd

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Run: pip install yfinance")


# =============================================================================
# CONFIGURATION
# =============================================================================

API_CONFIG = {
    'finnhub': {
        'base_url': 'https://finnhub.io/api/v1',
        'rate_limit': 60,  # calls per minute
        'endpoints': {
            'earnings_calendar': '/calendar/earnings',
            'earnings_surprise': '/stock/earnings',
            'quote': '/quote',
            'candles': '/stock/candle',
            'company_profile': '/stock/profile2',
        },
    },
    'alpha_vantage': {
        'base_url': 'https://www.alphavantage.co/query',
        'rate_limit': 5,  # calls per minute (free tier)
        'endpoints': {
            'earnings': 'EARNINGS',
            'daily': 'TIME_SERIES_DAILY_ADJUSTED',
            'overview': 'OVERVIEW',
        },
    },
    'fmp': {
        'base_url': 'https://financialmodelingprep.com/api/v3',
        'rate_limit': 250,  # calls per day (free tier)
        'endpoints': {
            'earnings_surprise': '/earnings-surprises',
            'earnings_calendar': '/earning_calendar',
            'quote': '/quote',
        },
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EarningsData:
    """Earnings data for a stock."""
    ticker: str
    report_date: datetime
    eps_actual: Optional[float]
    eps_estimate: Optional[float]
    eps_surprise: Optional[float]
    eps_surprise_pct: Optional[float]
    revenue_actual: Optional[float]
    revenue_estimate: Optional[float]
    revenue_surprise_pct: Optional[float]


@dataclass
class QuoteData:
    """Real-time quote data."""
    ticker: str
    price: float
    change: float
    change_pct: float
    volume: int
    timestamp: datetime


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = []

    def wait_if_needed(self):
        """Wait if we've exceeded the rate limit."""
        now = time.time()
        minute_ago = now - 60

        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if t > minute_ago]

        if len(self.calls) >= self.calls_per_minute:
            sleep_time = self.calls[0] - minute_ago + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.calls.append(time.time())


# =============================================================================
# CACHE
# =============================================================================

class SimpleCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self.cache = {}
        self.default_ttl = default_ttl

    def _make_key(self, *args, **kwargs) -> str:
        """Create a cache key from arguments."""
        key_str = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            value, expiry = self.cache[key]
            if datetime.now() < expiry:
                return value
            del self.cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with TTL."""
        ttl = ttl or self.default_ttl
        expiry = datetime.now() + timedelta(seconds=ttl)
        self.cache[key] = (value, expiry)

    def clear(self):
        """Clear all cached values."""
        self.cache = {}


def cached(ttl: int = 300):
    """Decorator to cache function results."""
    def decorator(func):
        cache = SimpleCache(ttl)

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = cache._make_key(func.__name__, *args, **kwargs)
            result = cache.get(key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            if result is not None:
                cache.set(key, result, ttl)
            return result
        return wrapper
    return decorator


# =============================================================================
# API CLIENT
# =============================================================================

class APIClient:
    """
    Unified API client for stock data.

    Handles multiple data sources with rate limiting and caching.
    """

    def __init__(self, api_keys: Optional[Dict[str, str]] = None,
                 finnhub_key: str = None,
                 alpha_vantage_key: str = None,
                 fmp_key: str = None):
        """
        Initialize API client.

        Args:
            api_keys: Dictionary with API keys (alternative to individual keys)
            finnhub_key: Finnhub API key
            alpha_vantage_key: Alpha Vantage API key
            fmp_key: Financial Modeling Prep API key
        """
        self.api_keys = api_keys or {}

        # Accept individual keys as well
        if finnhub_key:
            self.api_keys['finnhub'] = finnhub_key
        if alpha_vantage_key:
            self.api_keys['alpha_vantage'] = alpha_vantage_key
        if fmp_key:
            self.api_keys['fmp'] = fmp_key

        # Try to load from environment if not provided
        if 'finnhub' not in self.api_keys:
            self.api_keys['finnhub'] = os.environ.get('FINNHUB_API_KEY', '')
        if 'alpha_vantage' not in self.api_keys:
            self.api_keys['alpha_vantage'] = os.environ.get('ALPHA_VANTAGE_API_KEY', '')
        if 'fmp' not in self.api_keys:
            self.api_keys['fmp'] = os.environ.get('FMP_API_KEY', '')

        # Rate limiters
        self.rate_limiters = {
            'finnhub': RateLimiter(API_CONFIG['finnhub']['rate_limit']),
            'alpha_vantage': RateLimiter(API_CONFIG['alpha_vantage']['rate_limit']),
            'fmp': RateLimiter(API_CONFIG['fmp']['rate_limit']),
        }

        # Cache
        self.cache = SimpleCache(default_ttl=300)

    # =========================================================================
    # YFINANCE METHODS (Primary for price data)
    # =========================================================================

    def fetch_ohlcv(self, ticker: str, period: str = '6mo',
                    interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data using yfinance.

        Args:
            ticker: Stock ticker symbol
            period: Data period (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1d, 1wk, 1mo)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        """
        if not YFINANCE_AVAILABLE:
            print("yfinance not available")
            return None

        cache_key = f"ohlcv_{ticker}_{period}_{interval}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                return None

            # Standardize column names
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]

            # Ensure we have required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                return None

            self.cache.set(cache_key, df, ttl=300)
            return df

        except Exception as e:
            print(f"Error fetching OHLCV for {ticker}: {e}")
            return None

    def fetch_price_history(self, ticker: str, days: int = 252) -> Optional[pd.DataFrame]:
        """
        Fetch price history for a specific number of days.

        Args:
            ticker: Stock ticker symbol
            days: Number of trading days of history

        Returns:
            DataFrame with OHLCV data
        """
        # Map days to yfinance period
        if days <= 30:
            period = '1mo'
        elif days <= 90:
            period = '3mo'
        elif days <= 180:
            period = '6mo'
        elif days <= 365:
            period = '1y'
        elif days <= 730:
            period = '2y'
        else:
            period = '5y'

        df = self.fetch_ohlcv(ticker, period=period)
        if df is not None and len(df) > days:
            df = df.tail(days)
        return df

    def fetch_quote(self, ticker: str) -> Optional[QuoteData]:
        """
        Fetch real-time quote using yfinance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            QuoteData object
        """
        if not YFINANCE_AVAILABLE:
            return None

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return QuoteData(
                ticker=ticker,
                price=info.get('currentPrice', info.get('regularMarketPrice', 0)),
                change=info.get('regularMarketChange', 0),
                change_pct=info.get('regularMarketChangePercent', 0),
                volume=info.get('regularMarketVolume', 0),
                timestamp=datetime.now(),
            )
        except Exception as e:
            print(f"Error fetching quote for {ticker}: {e}")
            return None

    def fetch_info(self, ticker: str) -> Optional[Dict]:
        """
        Fetch company info using yfinance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with company information
        """
        if not YFINANCE_AVAILABLE:
            return None

        cache_key = f"info_{ticker}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            self.cache.set(cache_key, info, ttl=3600)  # Cache for 1 hour
            return info
        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")
            return None

    # =========================================================================
    # FINNHUB METHODS
    # =========================================================================

    def _finnhub_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a request to Finnhub API."""
        if not self.api_keys.get('finnhub'):
            return None

        self.rate_limiters['finnhub'].wait_if_needed()

        url = API_CONFIG['finnhub']['base_url'] + endpoint
        params = params or {}
        params['token'] = self.api_keys['finnhub']

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Finnhub API error: {e}")
            return None

    def fetch_earnings_calendar(self, from_date: str = None,
                                to_date: str = None) -> Optional[List[Dict]]:
        """
        Fetch earnings calendar from Finnhub.

        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            List of earnings events
        """
        if from_date is None:
            from_date = datetime.now().strftime('%Y-%m-%d')
        if to_date is None:
            to_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')

        cache_key = f"earnings_cal_{from_date}_{to_date}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        data = self._finnhub_request(
            API_CONFIG['finnhub']['endpoints']['earnings_calendar'],
            {'from': from_date, 'to': to_date}
        )

        if data and 'earningsCalendar' in data:
            result = data['earningsCalendar']
            self.cache.set(cache_key, result, ttl=3600)
            return result
        return None

    def fetch_earnings_surprise(self, ticker: str, limit: int = 4) -> Optional[List[EarningsData]]:
        """
        Fetch historical earnings surprises from Finnhub.

        Args:
            ticker: Stock ticker symbol
            limit: Number of quarters to fetch

        Returns:
            List of EarningsData objects
        """
        cache_key = f"earnings_surprise_{ticker}_{limit}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        data = self._finnhub_request(
            API_CONFIG['finnhub']['endpoints']['earnings_surprise'],
            {'symbol': ticker, 'limit': limit}
        )

        if not data:
            return None

        results = []
        for item in data:
            try:
                actual = item.get('actual')
                estimate = item.get('estimate')

                if actual is not None and estimate is not None and estimate != 0:
                    surprise = actual - estimate
                    surprise_pct = (surprise / abs(estimate)) * 100
                else:
                    surprise = None
                    surprise_pct = None

                results.append(EarningsData(
                    ticker=ticker,
                    report_date=datetime.strptime(item.get('period', ''), '%Y-%m-%d') if item.get('period') else None,
                    eps_actual=actual,
                    eps_estimate=estimate,
                    eps_surprise=surprise,
                    eps_surprise_pct=surprise_pct,
                    revenue_actual=None,  # Finnhub doesn't provide revenue in this endpoint
                    revenue_estimate=None,
                    revenue_surprise_pct=None,
                ))
            except Exception as e:
                print(f"Error parsing earnings data: {e}")
                continue

        if results:
            self.cache.set(cache_key, results, ttl=3600)
        return results

    # =========================================================================
    # ALPHA VANTAGE METHODS
    # =========================================================================

    def _alpha_vantage_request(self, function: str, params: Dict = None) -> Optional[Dict]:
        """Make a request to Alpha Vantage API."""
        if not self.api_keys.get('alpha_vantage'):
            return None

        self.rate_limiters['alpha_vantage'].wait_if_needed()

        url = API_CONFIG['alpha_vantage']['base_url']
        params = params or {}
        params['function'] = function
        params['apikey'] = self.api_keys['alpha_vantage']

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            # Check for API limit message
            if 'Note' in data or 'Information' in data:
                print(f"Alpha Vantage limit reached: {data.get('Note', data.get('Information'))}")
                return None

            return data
        except Exception as e:
            print(f"Alpha Vantage API error: {e}")
            return None

    def fetch_earnings_alpha_vantage(self, ticker: str) -> Optional[List[EarningsData]]:
        """
        Fetch earnings data from Alpha Vantage.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of EarningsData objects
        """
        cache_key = f"av_earnings_{ticker}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        data = self._alpha_vantage_request('EARNINGS', {'symbol': ticker})

        if not data or 'quarterlyEarnings' not in data:
            return None

        results = []
        for item in data['quarterlyEarnings'][:8]:  # Last 8 quarters
            try:
                actual = float(item.get('reportedEPS', 0)) if item.get('reportedEPS') != 'None' else None
                estimate = float(item.get('estimatedEPS', 0)) if item.get('estimatedEPS') != 'None' else None

                if actual is not None and estimate is not None and estimate != 0:
                    surprise = actual - estimate
                    surprise_pct = float(item.get('surprisePercentage', 0)) if item.get('surprisePercentage') != 'None' else (surprise / abs(estimate)) * 100
                else:
                    surprise = None
                    surprise_pct = None

                results.append(EarningsData(
                    ticker=ticker,
                    report_date=datetime.strptime(item.get('reportedDate', ''), '%Y-%m-%d') if item.get('reportedDate') else None,
                    eps_actual=actual,
                    eps_estimate=estimate,
                    eps_surprise=surprise,
                    eps_surprise_pct=surprise_pct,
                    revenue_actual=None,
                    revenue_estimate=None,
                    revenue_surprise_pct=None,
                ))
            except Exception as e:
                print(f"Error parsing AV earnings: {e}")
                continue

        if results:
            self.cache.set(cache_key, results, ttl=3600)
        return results

    # =========================================================================
    # COMBINED / FALLBACK METHODS
    # =========================================================================

    def get_earnings_surprise(self, ticker: str) -> Optional[EarningsData]:
        """
        Get the most recent earnings surprise using available APIs.

        Falls back through: Finnhub -> Alpha Vantage -> yfinance

        Args:
            ticker: Stock ticker symbol

        Returns:
            Most recent EarningsData or None
        """
        # Try Finnhub first
        finnhub_data = self.fetch_earnings_surprise(ticker, limit=1)
        if finnhub_data:
            return finnhub_data[0]

        # Try Alpha Vantage
        av_data = self.fetch_earnings_alpha_vantage(ticker)
        if av_data:
            return av_data[0]

        # Try yfinance as last resort
        if YFINANCE_AVAILABLE:
            try:
                stock = yf.Ticker(ticker)
                earnings = stock.earnings_dates
                if earnings is not None and len(earnings) > 0:
                    # yfinance doesn't provide surprise data directly
                    return None
            except:
                pass

        return None

    def get_stocks_reporting_today(self) -> List[str]:
        """
        Get list of stocks reporting earnings today.

        Returns:
            List of ticker symbols
        """
        today = datetime.now().strftime('%Y-%m-%d')
        calendar = self.fetch_earnings_calendar(from_date=today, to_date=today)

        if not calendar:
            return []

        return [item.get('symbol') for item in calendar if item.get('symbol')]

    def get_recent_earnings_stocks(self, days: int = 5) -> List[Dict]:
        """
        Get stocks that reported earnings in the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of dictionaries with ticker and report date
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        calendar = self.fetch_earnings_calendar(
            from_date=start_date.strftime('%Y-%m-%d'),
            to_date=end_date.strftime('%Y-%m-%d')
        )

        if not calendar:
            return []

        return [
            {'ticker': item.get('symbol'), 'date': item.get('date')}
            for item in calendar
            if item.get('symbol')
        ]

    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================

    def fetch_multiple_ohlcv(self, tickers: List[str], period: str = '6mo') -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            period: Data period

        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        for ticker in tickers:
            df = self.fetch_ohlcv(ticker, period=period)
            if df is not None:
                results[ticker] = df
        return results

    def fetch_multiple_quotes(self, tickers: List[str]) -> Dict[str, QuoteData]:
        """
        Fetch quotes for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to QuoteData
        """
        results = {}
        for ticker in tickers:
            quote = self.fetch_quote(ticker)
            if quote is not None:
                results[ticker] = quote
        return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_api_client(finnhub_key: str = None, alpha_vantage_key: str = None,
                      fmp_key: str = None) -> APIClient:
    """
    Create an API client with the provided keys.

    Args:
        finnhub_key: Finnhub API key
        alpha_vantage_key: Alpha Vantage API key
        fmp_key: Financial Modeling Prep API key

    Returns:
        Configured APIClient instance
    """
    keys = {}
    if finnhub_key:
        keys['finnhub'] = finnhub_key
    if alpha_vantage_key:
        keys['alpha_vantage'] = alpha_vantage_key
    if fmp_key:
        keys['fmp'] = fmp_key

    return APIClient(api_keys=keys)


if __name__ == '__main__':
    # Test the API client
    print("Testing API Client...")

    client = APIClient()

    # Test yfinance
    print("\n1. Testing yfinance OHLCV fetch for NVDA...")
    df = client.fetch_ohlcv('NVDA', period='1mo')
    if df is not None:
        print(f"   Got {len(df)} days of data")
        print(f"   Latest close: ${df['close'].iloc[-1]:.2f}")
    else:
        print("   Failed to fetch data")

    # Test quote
    print("\n2. Testing quote fetch for AAPL...")
    quote = client.fetch_quote('AAPL')
    if quote:
        print(f"   Price: ${quote.price:.2f}")
        print(f"   Change: {quote.change_pct:.2f}%")
    else:
        print("   Failed to fetch quote")

    # Test earnings calendar (requires API key)
    print("\n3. Testing earnings calendar...")
    calendar = client.fetch_earnings_calendar()
    if calendar:
        print(f"   Found {len(calendar)} earnings events")
    else:
        print("   No data (API key required)")

    print("\nAPI Client test complete.")
