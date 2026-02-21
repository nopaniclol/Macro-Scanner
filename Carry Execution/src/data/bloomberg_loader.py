"""
Bloomberg Data Loader
BQL-based data fetching for FX, Metals, Rates, and Macro data.

Uses Bloomberg Query Language (BQL) syntax as per BQUANT environment.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod
import time
import logging

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class BaseDataLoader(ABC):
    """Abstract base class for all data loaders."""

    @abstractmethod
    def fetch_spot(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Fetch spot price data."""
        pass

    @abstractmethod
    def fetch_ohlcv(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Fetch OHLCV data."""
        pass

    @abstractmethod
    def fetch_multiple(self, tickers: List[str], days: int = 252) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers."""
        pass


# =============================================================================
# BLOOMBERG CONNECTION
# =============================================================================

class BloombergConnection:
    """
    Manages Bloomberg BQL connection with retry logic and caching.
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize Bloomberg connection.

        Args:
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._bq = None
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._cache_timestamps = {}

    @property
    def bq(self):
        """Lazy initialization of BQL service."""
        if self._bq is None:
            try:
                import bql
                self._bq = bql.Service()
                logger.info("Bloomberg BQL service initialized successfully")
            except ImportError:
                raise ImportError(
                    "BQL package not available. Install via Bloomberg Terminal or use FreeDataLoader."
                )
            except Exception as e:
                raise ConnectionError(f"Failed to initialize BQL service: {e}")
        return self._bq

    def _get_cache_key(self, ticker: str, fields: tuple, days: int) -> str:
        """Generate cache key."""
        return f"{ticker}:{fields}:{days}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        elapsed = time.time() - self._cache_timestamps[cache_key]
        return elapsed < self._cache_ttl

    def execute_with_retry(self, request) -> pd.DataFrame:
        """
        Execute BQL request with retry logic.

        Args:
            request: BQL request object

        Returns:
            DataFrame with results
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.bq.execute(request)
                # Combine response data items into single DataFrame
                df = pd.concat([item.df() for item in response], axis=1)
                return df

            except Exception as e:
                last_error = e
                logger.warning(f"BQL request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff

        raise RuntimeError(f"BQL request failed after {self.max_retries} attempts: {last_error}")

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._cache_timestamps.clear()


# =============================================================================
# FX SPOT LOADER
# =============================================================================

class FXSpotLoader(BaseDataLoader):
    """
    Loader for FX spot prices.

    Handles currency pairs like EURUSD, USDJPY, etc.
    """

    def __init__(self, connection: BloombergConnection, ticker_config: Dict):
        """
        Initialize FX spot loader.

        Args:
            connection: Bloomberg connection instance
            ticker_config: Ticker configuration from tickers.yaml
        """
        self.conn = connection
        self.config = ticker_config

    def _get_bloomberg_ticker(self, pair: str) -> str:
        """Convert pair name to Bloomberg ticker."""
        # Check in config first
        fx_spot = self.config.get('fx', {}).get('spot', {})
        if pair in fx_spot:
            return fx_spot[pair]
        # Default conversion
        return f"{pair} Curncy"

    def fetch_spot(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """
        Fetch FX spot prices.

        Args:
            ticker: Currency pair (e.g., 'EURUSD' or 'EURUSD Curncy')
            days: Number of days of history

        Returns:
            DataFrame with Date and Close columns
        """
        import bql

        bb_ticker = self._get_bloomberg_ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        request = bql.Request(
            bb_ticker,
            {
                'Date': self.conn.bq.data.px_last()['DATE'],
                'Close': self.conn.bq.data.px_last()['value'],
            },
            with_params={
                'fill': 'prev',
                'dates': self.conn.bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            }
        )

        df = self.conn.execute_with_retry(request)
        df = df.sort_values('Date').reset_index(drop=True)
        df['Ticker'] = ticker

        return df

    def fetch_ohlcv(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """
        Fetch FX OHLCV data.

        Args:
            ticker: Currency pair
            days: Number of days of history

        Returns:
            DataFrame with OHLCV columns
        """
        import bql

        bb_ticker = self._get_bloomberg_ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        request = bql.Request(
            bb_ticker,
            {
                'Date': self.conn.bq.data.px_open()['DATE'],
                'Open': self.conn.bq.data.px_open()['value'],
                'High': self.conn.bq.data.px_high()['value'],
                'Low': self.conn.bq.data.px_low()['value'],
                'Close': self.conn.bq.data.px_last()['value'],
                'Volume': self.conn.bq.data.px_volume()['value'],
            },
            with_params={
                'fill': 'prev',
                'dates': self.conn.bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            }
        )

        df = self.conn.execute_with_retry(request)
        df = df.sort_values('Date').reset_index(drop=True)
        df['Ticker'] = ticker

        return df

    def fetch_multiple(self, tickers: List[str], days: int = 252) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple FX pairs.

        Args:
            tickers: List of currency pairs
            days: Number of days of history

        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch_spot(ticker, days)
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                results[ticker] = pd.DataFrame()
        return results


# =============================================================================
# FX FORWARD LOADER
# =============================================================================

class FXForwardLoader(BaseDataLoader):
    """
    Loader for FX forward points and outright forwards.

    Used for carry calculation via forward points.
    """

    def __init__(self, connection: BloombergConnection, ticker_config: Dict):
        self.conn = connection
        self.config = ticker_config

    def _get_forward_ticker(self, pair: str, tenor: str) -> str:
        """Get Bloomberg ticker for forward points."""
        forwards = self.config.get('fx', {}).get('forwards', {})
        pair_forwards = forwards.get(pair, {})
        if tenor in pair_forwards:
            return pair_forwards[tenor]
        # Default pattern
        return f"{pair}{tenor} Curncy"

    def fetch_forward_points(self, pair: str, tenor: str, days: int = 252) -> pd.DataFrame:
        """
        Fetch forward points for a currency pair and tenor.

        Args:
            pair: Currency pair (e.g., 'EURUSD')
            tenor: Forward tenor (e.g., '3M')
            days: Number of days of history

        Returns:
            DataFrame with forward points
        """
        import bql

        bb_ticker = self._get_forward_ticker(pair, tenor)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        request = bql.Request(
            bb_ticker,
            {
                'Date': self.conn.bq.data.px_last()['DATE'],
                'ForwardPoints': self.conn.bq.data.px_last()['value'],
            },
            with_params={
                'fill': 'prev',
                'dates': self.conn.bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            }
        )

        df = self.conn.execute_with_retry(request)
        df = df.sort_values('Date').reset_index(drop=True)
        df['Pair'] = pair
        df['Tenor'] = tenor

        return df

    def fetch_spot(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Fetch spot via FXSpotLoader - delegate."""
        raise NotImplementedError("Use FXSpotLoader for spot data")

    def fetch_ohlcv(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Not applicable for forwards."""
        raise NotImplementedError("OHLCV not available for forwards")

    def fetch_multiple(self, tickers: List[str], days: int = 252) -> Dict[str, pd.DataFrame]:
        """Not directly applicable - use fetch_forward_points instead."""
        raise NotImplementedError("Use fetch_forward_points with pair and tenor")


# =============================================================================
# SWAP CURVE LOADER
# =============================================================================

class SwapCurveLoader(BaseDataLoader):
    """
    Loader for SOFR IRS and foreign OIS curves.

    Fetches swap rates across the term structure.
    """

    def __init__(self, connection: BloombergConnection, ticker_config: Dict):
        self.conn = connection
        self.config = ticker_config

    def _get_swap_ticker(self, currency: str, tenor: str) -> str:
        """Get Bloomberg ticker for swap rate."""
        if currency == 'USD':
            sofr = self.config.get('rates', {}).get('sofr_swaps', {})
            if tenor in sofr:
                return sofr[tenor]
        else:
            ois = self.config.get('rates', {}).get('ois_curves', {}).get(currency, {})
            if tenor in ois:
                return ois[tenor]
        return None

    def fetch_curve(self, currency: str, tenors: List[str], days: int = 252) -> pd.DataFrame:
        """
        Fetch swap curve for a currency.

        Args:
            currency: Currency code (e.g., 'USD', 'EUR')
            tenors: List of tenors (e.g., ['1Y', '2Y', '5Y', '10Y'])
            days: Number of days of history

        Returns:
            DataFrame with swap rates for each tenor
        """
        import bql

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        results = []

        for tenor in tenors:
            bb_ticker = self._get_swap_ticker(currency, tenor)
            if bb_ticker is None:
                logger.warning(f"No ticker found for {currency} {tenor}")
                continue

            try:
                request = bql.Request(
                    bb_ticker,
                    {
                        'Date': self.conn.bq.data.px_last()['DATE'],
                        f'{tenor}': self.conn.bq.data.px_last()['value'],
                    },
                    with_params={
                        'fill': 'prev',
                        'dates': self.conn.bq.func.range(
                            start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d')
                        )
                    }
                )

                df = self.conn.execute_with_retry(request)
                results.append(df)

            except Exception as e:
                logger.error(f"Failed to fetch {currency} {tenor}: {e}")

        if not results:
            return pd.DataFrame()

        # Merge all tenors
        merged = results[0]
        for df in results[1:]:
            merged = merged.merge(df, on='Date', how='outer')

        merged = merged.sort_values('Date').reset_index(drop=True)
        merged['Currency'] = currency

        return merged

    def fetch_spot(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Fetch single swap rate."""
        import bql

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        request = bql.Request(
            ticker,
            {
                'Date': self.conn.bq.data.px_last()['DATE'],
                'Rate': self.conn.bq.data.px_last()['value'],
            },
            with_params={
                'fill': 'prev',
                'dates': self.conn.bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            }
        )

        df = self.conn.execute_with_retry(request)
        df = df.sort_values('Date').reset_index(drop=True)
        df.rename(columns={'Rate': 'Close'}, inplace=True)

        return df

    def fetch_ohlcv(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Not applicable for swap rates."""
        raise NotImplementedError("OHLCV not available for swap rates")

    def fetch_multiple(self, tickers: List[str], days: int = 252) -> Dict[str, pd.DataFrame]:
        """Fetch multiple swap rates."""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch_spot(ticker, days)
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                results[ticker] = pd.DataFrame()
        return results


# =============================================================================
# METALS LOADER
# =============================================================================

class MetalsLoader(BaseDataLoader):
    """
    Loader for precious metals (Gold, Silver) spot prices.
    """

    def __init__(self, connection: BloombergConnection, ticker_config: Dict):
        self.conn = connection
        self.config = ticker_config

    def _get_metal_ticker(self, metal: str) -> str:
        """Get Bloomberg ticker for metal."""
        metals = self.config.get('metals', {}).get('spot', {})
        if metal in metals:
            return metals[metal]
        # Default mappings
        defaults = {
            'XAUUSD': 'XAU Curncy',
            'XAGUSD': 'XAG Curncy',
            'Gold': 'XAU Curncy',
            'Silver': 'XAG Curncy',
        }
        return defaults.get(metal, f"{metal} Curncy")

    def fetch_spot(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Fetch metal spot price."""
        import bql

        bb_ticker = self._get_metal_ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        request = bql.Request(
            bb_ticker,
            {
                'Date': self.conn.bq.data.px_last()['DATE'],
                'Close': self.conn.bq.data.px_last()['value'],
            },
            with_params={
                'fill': 'prev',
                'dates': self.conn.bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            }
        )

        df = self.conn.execute_with_retry(request)
        df = df.sort_values('Date').reset_index(drop=True)
        df['Ticker'] = ticker

        return df

    def fetch_ohlcv(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Fetch metal OHLCV data."""
        import bql

        bb_ticker = self._get_metal_ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        request = bql.Request(
            bb_ticker,
            {
                'Date': self.conn.bq.data.px_open()['DATE'],
                'Open': self.conn.bq.data.px_open()['value'],
                'High': self.conn.bq.data.px_high()['value'],
                'Low': self.conn.bq.data.px_low()['value'],
                'Close': self.conn.bq.data.px_last()['value'],
                'Volume': self.conn.bq.data.px_volume()['value'],
            },
            with_params={
                'fill': 'prev',
                'dates': self.conn.bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            }
        )

        df = self.conn.execute_with_retry(request)
        df = df.sort_values('Date').reset_index(drop=True)
        df['Ticker'] = ticker

        return df

    def fetch_multiple(self, tickers: List[str], days: int = 252) -> Dict[str, pd.DataFrame]:
        """Fetch multiple metals."""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch_spot(ticker, days)
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                results[ticker] = pd.DataFrame()
        return results


# =============================================================================
# IMPLIED VOLATILITY LOADER
# =============================================================================

class ImpliedVolLoader(BaseDataLoader):
    """
    Loader for FX implied volatility surfaces.
    """

    def __init__(self, connection: BloombergConnection, ticker_config: Dict):
        self.conn = connection
        self.config = ticker_config

    def _get_vol_ticker(self, pair: str, tenor: str, delta: str = 'ATM') -> str:
        """Get Bloomberg ticker for implied vol."""
        vol_config = self.config.get('fx', {}).get('implied_vol', {})
        pair_vols = vol_config.get(pair, {})

        # Try to find specific tenor/delta combo
        key = f"{tenor}_{delta}"
        if key in pair_vols:
            return pair_vols[key]

        # Default pattern: e.g., EURUSDV1M Curncy for 1M ATM vol
        return f"{pair}V{tenor} Curncy"

    def fetch_spot(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Fetch single vol ticker."""
        import bql

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        request = bql.Request(
            ticker,
            {
                'Date': self.conn.bq.data.px_last()['DATE'],
                'ImpliedVol': self.conn.bq.data.px_last()['value'],
            },
            with_params={
                'fill': 'prev',
                'dates': self.conn.bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            }
        )

        df = self.conn.execute_with_retry(request)
        df = df.sort_values('Date').reset_index(drop=True)
        df.rename(columns={'ImpliedVol': 'Close'}, inplace=True)

        return df

    def fetch_vol_surface(self, pair: str, tenors: List[str], days: int = 252) -> pd.DataFrame:
        """
        Fetch ATM vol surface for a currency pair.

        Args:
            pair: Currency pair (e.g., 'EURUSD')
            tenors: List of tenors (e.g., ['1W', '1M', '3M', '6M', '1Y'])
            days: Number of days of history

        Returns:
            DataFrame with vol for each tenor
        """
        import bql

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        results = []

        for tenor in tenors:
            bb_ticker = self._get_vol_ticker(pair, tenor)

            try:
                request = bql.Request(
                    bb_ticker,
                    {
                        'Date': self.conn.bq.data.px_last()['DATE'],
                        f'Vol_{tenor}': self.conn.bq.data.px_last()['value'],
                    },
                    with_params={
                        'fill': 'prev',
                        'dates': self.conn.bq.func.range(
                            start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d')
                        )
                    }
                )

                df = self.conn.execute_with_retry(request)
                results.append(df)

            except Exception as e:
                logger.error(f"Failed to fetch {pair} {tenor} vol: {e}")

        if not results:
            return pd.DataFrame()

        # Merge all tenors
        merged = results[0]
        for df in results[1:]:
            merged = merged.merge(df, on='Date', how='outer')

        merged = merged.sort_values('Date').reset_index(drop=True)
        merged['Pair'] = pair

        return merged

    def fetch_ohlcv(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Not applicable for vols."""
        raise NotImplementedError("OHLCV not available for implied vols")

    def fetch_multiple(self, tickers: List[str], days: int = 252) -> Dict[str, pd.DataFrame]:
        """Fetch multiple vol tickers."""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch_spot(ticker, days)
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                results[ticker] = pd.DataFrame()
        return results


# =============================================================================
# MACRO DATA LOADER
# =============================================================================

class MacroDataLoader(BaseDataLoader):
    """
    Loader for macroeconomic data (GDP, CPI, PMI, etc.).
    """

    def __init__(self, connection: BloombergConnection, ticker_config: Dict):
        self.conn = connection
        self.config = ticker_config

    def _get_macro_ticker(self, indicator: str, country: str = 'US') -> str:
        """Get Bloomberg ticker for macro indicator."""
        macro_config = self.config.get('macro', {}).get(country, {})
        if indicator in macro_config:
            return macro_config[indicator]
        return None

    def fetch_spot(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Fetch macro indicator."""
        import bql

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        request = bql.Request(
            ticker,
            {
                'Date': self.conn.bq.data.px_last()['DATE'],
                'Value': self.conn.bq.data.px_last()['value'],
            },
            with_params={
                'fill': 'prev',
                'dates': self.conn.bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            }
        )

        df = self.conn.execute_with_retry(request)
        df = df.sort_values('Date').reset_index(drop=True)
        df.rename(columns={'Value': 'Close'}, inplace=True)

        return df

    def fetch_ohlcv(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Not applicable for macro data."""
        raise NotImplementedError("OHLCV not available for macro indicators")

    def fetch_multiple(self, tickers: List[str], days: int = 252) -> Dict[str, pd.DataFrame]:
        """Fetch multiple macro indicators."""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch_spot(ticker, days)
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                results[ticker] = pd.DataFrame()
        return results


# =============================================================================
# RISK INDICATOR LOADER
# =============================================================================

class RiskIndicatorLoader(BaseDataLoader):
    """
    Loader for risk indicators (VIX, credit spreads, etc.).
    """

    def __init__(self, connection: BloombergConnection, ticker_config: Dict):
        self.conn = connection
        self.config = ticker_config

    def _get_risk_ticker(self, indicator: str) -> str:
        """Get Bloomberg ticker for risk indicator."""
        risk_config = self.config.get('volatility', {})
        if indicator in risk_config:
            return risk_config[indicator]
        # Defaults
        defaults = {
            'VIX': 'VIX Index',
            'MOVE': 'MOVE Index',
            'CVIX': 'CVIX Index',
        }
        return defaults.get(indicator, f"{indicator} Index")

    def fetch_spot(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Fetch risk indicator."""
        import bql

        bb_ticker = self._get_risk_ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        request = bql.Request(
            bb_ticker,
            {
                'Date': self.conn.bq.data.px_last()['DATE'],
                'Close': self.conn.bq.data.px_last()['value'],
            },
            with_params={
                'fill': 'prev',
                'dates': self.conn.bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            }
        )

        df = self.conn.execute_with_retry(request)
        df = df.sort_values('Date').reset_index(drop=True)
        df['Ticker'] = ticker

        return df

    def fetch_ohlcv(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Fetch OHLCV for risk indicator."""
        import bql

        bb_ticker = self._get_risk_ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        request = bql.Request(
            bb_ticker,
            {
                'Date': self.conn.bq.data.px_open()['DATE'],
                'Open': self.conn.bq.data.px_open()['value'],
                'High': self.conn.bq.data.px_high()['value'],
                'Low': self.conn.bq.data.px_low()['value'],
                'Close': self.conn.bq.data.px_last()['value'],
            },
            with_params={
                'fill': 'prev',
                'dates': self.conn.bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            }
        )

        df = self.conn.execute_with_retry(request)
        df = df.sort_values('Date').reset_index(drop=True)
        df['Ticker'] = ticker

        return df

    def fetch_multiple(self, tickers: List[str], days: int = 252) -> Dict[str, pd.DataFrame]:
        """Fetch multiple risk indicators."""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch_spot(ticker, days)
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                results[ticker] = pd.DataFrame()
        return results


# =============================================================================
# UNIFIED BLOOMBERG LOADER
# =============================================================================

class BloombergDataLoader:
    """
    Unified Bloomberg data loader that provides access to all specialized loaders.

    This is the main entry point for Bloomberg data access.
    """

    def __init__(self, ticker_config: Optional[Dict] = None):
        """
        Initialize unified Bloomberg loader.

        Args:
            ticker_config: Ticker configuration dictionary (from tickers.yaml)
        """
        self.ticker_config = ticker_config or {}
        self.connection = BloombergConnection()

        # Initialize specialized loaders
        self.fx_spot = FXSpotLoader(self.connection, self.ticker_config)
        self.fx_forward = FXForwardLoader(self.connection, self.ticker_config)
        self.swap_curve = SwapCurveLoader(self.connection, self.ticker_config)
        self.metals = MetalsLoader(self.connection, self.ticker_config)
        self.implied_vol = ImpliedVolLoader(self.connection, self.ticker_config)
        self.macro = MacroDataLoader(self.connection, self.ticker_config)
        self.risk = RiskIndicatorLoader(self.connection, self.ticker_config)

    def fetch_spot(self, ticker: str, asset_type: str = 'auto', days: int = 252) -> pd.DataFrame:
        """
        Fetch spot price for any asset type.

        Args:
            ticker: Asset ticker
            asset_type: 'fx', 'metal', 'rate', 'risk', or 'auto' (detect from ticker)
            days: Number of days of history

        Returns:
            DataFrame with Date and Close columns
        """
        if asset_type == 'auto':
            asset_type = self._detect_asset_type(ticker)

        loader_map = {
            'fx': self.fx_spot,
            'metal': self.metals,
            'rate': self.swap_curve,
            'risk': self.risk,
            'macro': self.macro,
        }

        loader = loader_map.get(asset_type)
        if loader is None:
            raise ValueError(f"Unknown asset type: {asset_type}")

        return loader.fetch_spot(ticker, days)

    def fetch_ohlcv(self, ticker: str, asset_type: str = 'auto', days: int = 252) -> pd.DataFrame:
        """
        Fetch OHLCV data for any asset type.

        Args:
            ticker: Asset ticker
            asset_type: 'fx', 'metal', 'risk', or 'auto' (detect from ticker)
            days: Number of days of history

        Returns:
            DataFrame with OHLCV columns
        """
        if asset_type == 'auto':
            asset_type = self._detect_asset_type(ticker)

        loader_map = {
            'fx': self.fx_spot,
            'metal': self.metals,
            'risk': self.risk,
        }

        loader = loader_map.get(asset_type)
        if loader is None:
            raise ValueError(f"OHLCV not available for asset type: {asset_type}")

        return loader.fetch_ohlcv(ticker, days)

    def _detect_asset_type(self, ticker: str) -> str:
        """Auto-detect asset type from ticker."""
        ticker_upper = ticker.upper()

        # FX pairs
        fx_pairs = ['EUR', 'USD', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'NZD', 'CNH']
        if any(fx in ticker_upper for fx in fx_pairs) and len(ticker) <= 10:
            if 'XAU' in ticker_upper or 'XAG' in ticker_upper:
                return 'metal'
            return 'fx'

        # Metals
        if any(metal in ticker_upper for metal in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            return 'metal'

        # Risk indicators
        if any(risk in ticker_upper for risk in ['VIX', 'MOVE', 'CVIX']):
            return 'risk'

        # Default to macro
        return 'macro'

    def is_available(self) -> bool:
        """Check if Bloomberg connection is available."""
        try:
            _ = self.connection.bq
            return True
        except:
            return False
