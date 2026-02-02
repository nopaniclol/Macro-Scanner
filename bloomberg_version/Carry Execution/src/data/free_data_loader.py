"""
Free Data Loader
Data fetching using yfinance for FX/metals and FRED for rates/macro data.

Provides the same interface as BloombergDataLoader for swap-in-swap-out usage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
import time

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# TICKER MAPPINGS
# =============================================================================

# Map internal tickers to yfinance format
YFINANCE_FX_MAP = {
    'EURUSD': 'EURUSD=X',
    'USDJPY': 'USDJPY=X',
    'GBPUSD': 'GBPUSD=X',
    'AUDUSD': 'AUDUSD=X',
    'USDCAD': 'USDCAD=X',
    'USDCHF': 'USDCHF=X',
    'NZDUSD': 'NZDUSD=X',
    'USDCNH': 'CNY=X',  # Onshore CNY as proxy
    'DXY': 'DX-Y.NYB',  # Dollar Index
}

YFINANCE_METALS_MAP = {
    'XAUUSD': 'GC=F',   # Gold futures
    'XAGUSD': 'SI=F',   # Silver futures
    'Gold': 'GC=F',
    'Silver': 'SI=F',
}

YFINANCE_RISK_MAP = {
    'VIX': '^VIX',
}

# FRED series IDs for rates
FRED_RATES_MAP = {
    # US Treasury yields
    'US1M': 'DGS1MO',
    'US3M': 'DGS3MO',
    'US6M': 'DGS6MO',
    'US1Y': 'DGS1',
    'US2Y': 'DGS2',
    'US3Y': 'DGS3',
    'US5Y': 'DGS5',
    'US7Y': 'DGS7',
    'US10Y': 'DGS10',
    'US20Y': 'DGS20',
    'US30Y': 'DGS30',
    # SOFR
    'SOFR': 'SOFR',
    # Fed Funds
    'FEDFUNDS': 'FEDFUNDS',
    # TIPS (real yields)
    'TIPS5Y': 'DFII5',
    'TIPS10Y': 'DFII10',
    # Breakevens
    'BE5Y': 'T5YIE',
    'BE10Y': 'T10YIE',
}

FRED_MACRO_MAP = {
    # CPI
    'CPI': 'CPIAUCSL',
    'CPI_YOY': 'CPIAUCSL',
    'CORE_CPI': 'CPILFESL',
    # GDP
    'GDP': 'GDP',
    'GDPC1': 'GDPC1',
    # Employment
    'UNRATE': 'UNRATE',
    'PAYEMS': 'PAYEMS',
    # ISM
    'ISM_MFG': 'MANEMP',
}


# =============================================================================
# BASE FREE DATA LOADER
# =============================================================================

class FreeDataLoader:
    """
    Free data loader using yfinance and FRED APIs.

    Provides the same interface as BloombergDataLoader for swap-in-swap-out usage.
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize free data loader.

        Args:
            fred_api_key: FRED API key (optional, will try to fetch without if not provided)
        """
        self.fred_api_key = fred_api_key
        self._yf = None
        self._fred = None

    @property
    def yf(self):
        """Lazy initialization of yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
                logger.info("yfinance initialized successfully")
            except ImportError:
                raise ImportError("yfinance not installed. Run: pip install yfinance")
        return self._yf

    @property
    def fred(self):
        """Lazy initialization of FRED API."""
        if self._fred is None:
            try:
                from fredapi import Fred
                if self.fred_api_key:
                    self._fred = Fred(api_key=self.fred_api_key)
                else:
                    # Try without API key (limited functionality)
                    self._fred = Fred()
                logger.info("FRED API initialized successfully")
            except ImportError:
                logger.warning("fredapi not installed. FRED data will not be available.")
                self._fred = None
            except Exception as e:
                logger.warning(f"FRED API initialization failed: {e}")
                self._fred = None
        return self._fred

    # =========================================================================
    # FX DATA
    # =========================================================================

    def _get_yf_fx_ticker(self, pair: str) -> str:
        """Convert pair to yfinance ticker."""
        # Clean up ticker
        pair_clean = pair.upper().replace(' CURNCY', '').replace('CURNCY', '').strip()

        if pair_clean in YFINANCE_FX_MAP:
            return YFINANCE_FX_MAP[pair_clean]

        # Default: append =X
        return f"{pair_clean}=X"

    def fetch_fx_spot(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """
        Fetch FX spot price data.

        Args:
            ticker: Currency pair (e.g., 'EURUSD')
            days: Number of days of history

        Returns:
            DataFrame with Date and Close columns
        """
        yf_ticker = self._get_yf_fx_ticker(ticker)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)  # Extra buffer for weekends

        try:
            data = self.yf.download(
                yf_ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )

            if data.empty:
                logger.warning(f"No data returned for {ticker} ({yf_ticker})")
                return pd.DataFrame()

            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = data[['Close']].copy()
            df = df.reset_index()
            df.columns = ['Date', 'Close']
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df['Ticker'] = ticker

            return df.tail(days)

        except Exception as e:
            logger.error(f"Failed to fetch FX data for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_fx_ohlcv(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """
        Fetch FX OHLCV data.

        Args:
            ticker: Currency pair
            days: Number of days of history

        Returns:
            DataFrame with OHLCV columns
        """
        yf_ticker = self._get_yf_fx_ticker(ticker)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)

        try:
            data = self.yf.download(
                yf_ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )

            if data.empty:
                logger.warning(f"No data returned for {ticker} ({yf_ticker})")
                return pd.DataFrame()

            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = df.reset_index()
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df['Ticker'] = ticker

            return df.tail(days)

        except Exception as e:
            logger.error(f"Failed to fetch FX OHLCV for {ticker}: {e}")
            return pd.DataFrame()

    # =========================================================================
    # METALS DATA
    # =========================================================================

    def _get_yf_metal_ticker(self, metal: str) -> str:
        """Convert metal to yfinance ticker."""
        metal_clean = metal.upper().replace(' CURNCY', '').strip()

        if metal_clean in YFINANCE_METALS_MAP:
            return YFINANCE_METALS_MAP[metal_clean]

        return metal_clean

    def fetch_metal_spot(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """
        Fetch metal spot price data.

        Args:
            ticker: Metal ticker (e.g., 'XAUUSD', 'Gold')
            days: Number of days of history

        Returns:
            DataFrame with Date and Close columns
        """
        yf_ticker = self._get_yf_metal_ticker(ticker)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)

        try:
            data = self.yf.download(
                yf_ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )

            if data.empty:
                logger.warning(f"No data returned for {ticker} ({yf_ticker})")
                return pd.DataFrame()

            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = data[['Close']].copy()
            df = df.reset_index()
            df.columns = ['Date', 'Close']
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df['Ticker'] = ticker

            return df.tail(days)

        except Exception as e:
            logger.error(f"Failed to fetch metal data for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_metal_ohlcv(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Fetch metal OHLCV data."""
        yf_ticker = self._get_yf_metal_ticker(ticker)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)

        try:
            data = self.yf.download(
                yf_ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )

            if data.empty:
                return pd.DataFrame()

            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = df.reset_index()
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df['Ticker'] = ticker

            return df.tail(days)

        except Exception as e:
            logger.error(f"Failed to fetch metal OHLCV for {ticker}: {e}")
            return pd.DataFrame()

    # =========================================================================
    # RATES DATA (FRED)
    # =========================================================================

    def _get_fred_series(self, series_id: str) -> str:
        """Get FRED series ID."""
        if series_id in FRED_RATES_MAP:
            return FRED_RATES_MAP[series_id]
        if series_id in FRED_MACRO_MAP:
            return FRED_MACRO_MAP[series_id]
        return series_id

    def fetch_rate(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """
        Fetch interest rate data from FRED.

        Args:
            ticker: Rate ticker (e.g., 'US10Y', 'SOFR')
            days: Number of days of history

        Returns:
            DataFrame with Date and Close columns
        """
        if self.fred is None:
            logger.warning("FRED API not available")
            return pd.DataFrame()

        fred_id = self._get_fred_series(ticker)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)  # Extra buffer

        try:
            series = self.fred.get_series(
                fred_id,
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=end_date.strftime('%Y-%m-%d')
            )

            if series.empty:
                logger.warning(f"No data returned for {ticker} ({fred_id})")
                return pd.DataFrame()

            df = series.reset_index()
            df.columns = ['Date', 'Close']
            df['Date'] = pd.to_datetime(df['Date'])
            df['Ticker'] = ticker

            # Forward fill missing values (weekends/holidays)
            df['Close'] = df['Close'].ffill()

            return df.tail(days)

        except Exception as e:
            logger.error(f"Failed to fetch rate data for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_yield_curve(self, tenors: List[str] = None, days: int = 252) -> pd.DataFrame:
        """
        Fetch US Treasury yield curve.

        Args:
            tenors: List of tenors (e.g., ['US2Y', 'US5Y', 'US10Y'])
            days: Number of days of history

        Returns:
            DataFrame with yields for each tenor
        """
        if tenors is None:
            tenors = ['US1M', 'US3M', 'US6M', 'US1Y', 'US2Y', 'US3Y',
                     'US5Y', 'US7Y', 'US10Y', 'US20Y', 'US30Y']

        results = []
        for tenor in tenors:
            df = self.fetch_rate(tenor, days)
            if not df.empty:
                df = df.rename(columns={'Close': tenor})
                df = df[['Date', tenor]]
                results.append(df)

        if not results:
            return pd.DataFrame()

        # Merge all tenors
        merged = results[0]
        for df in results[1:]:
            merged = merged.merge(df, on='Date', how='outer')

        merged = merged.sort_values('Date').reset_index(drop=True)

        return merged

    # =========================================================================
    # MACRO DATA (FRED)
    # =========================================================================

    def fetch_macro(self, indicator: str, days: int = 252) -> pd.DataFrame:
        """
        Fetch macroeconomic data from FRED.

        Args:
            indicator: Macro indicator (e.g., 'CPI', 'GDP', 'UNRATE')
            days: Number of days of history

        Returns:
            DataFrame with Date and Value columns
        """
        if self.fred is None:
            logger.warning("FRED API not available")
            return pd.DataFrame()

        fred_id = self._get_fred_series(indicator)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 60)  # Extra buffer for monthly data

        try:
            series = self.fred.get_series(
                fred_id,
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=end_date.strftime('%Y-%m-%d')
            )

            if series.empty:
                logger.warning(f"No data returned for {indicator} ({fred_id})")
                return pd.DataFrame()

            df = series.reset_index()
            df.columns = ['Date', 'Close']
            df['Date'] = pd.to_datetime(df['Date'])
            df['Ticker'] = indicator

            return df

        except Exception as e:
            logger.error(f"Failed to fetch macro data for {indicator}: {e}")
            return pd.DataFrame()

    # =========================================================================
    # RISK INDICATORS
    # =========================================================================

    def fetch_risk_indicator(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """
        Fetch risk indicator (VIX, etc.) from yfinance.

        Args:
            ticker: Risk indicator ticker (e.g., 'VIX')
            days: Number of days of history

        Returns:
            DataFrame with Date and Close columns
        """
        yf_ticker = YFINANCE_RISK_MAP.get(ticker.upper(), f'^{ticker}')

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)

        try:
            data = self.yf.download(
                yf_ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )

            if data.empty:
                logger.warning(f"No data returned for {ticker} ({yf_ticker})")
                return pd.DataFrame()

            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = data[['Close']].copy()
            df = df.reset_index()
            df.columns = ['Date', 'Close']
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df['Ticker'] = ticker

            return df.tail(days)

        except Exception as e:
            logger.error(f"Failed to fetch risk indicator for {ticker}: {e}")
            return pd.DataFrame()

    # =========================================================================
    # UNIFIED INTERFACE (matches BloombergDataLoader)
    # =========================================================================

    def fetch_spot(self, ticker: str, asset_type: str = 'auto', days: int = 252) -> pd.DataFrame:
        """
        Fetch spot price for any asset type.

        Args:
            ticker: Asset ticker
            asset_type: 'fx', 'metal', 'rate', 'risk', 'macro', or 'auto'
            days: Number of days of history

        Returns:
            DataFrame with Date and Close columns
        """
        if asset_type == 'auto':
            asset_type = self._detect_asset_type(ticker)

        if asset_type == 'fx':
            return self.fetch_fx_spot(ticker, days)
        elif asset_type == 'metal':
            return self.fetch_metal_spot(ticker, days)
        elif asset_type == 'rate':
            return self.fetch_rate(ticker, days)
        elif asset_type == 'risk':
            return self.fetch_risk_indicator(ticker, days)
        elif asset_type == 'macro':
            return self.fetch_macro(ticker, days)
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")

    def fetch_ohlcv(self, ticker: str, asset_type: str = 'auto', days: int = 252) -> pd.DataFrame:
        """
        Fetch OHLCV data for any asset type.

        Args:
            ticker: Asset ticker
            asset_type: 'fx', 'metal', 'risk', or 'auto'
            days: Number of days of history

        Returns:
            DataFrame with OHLCV columns
        """
        if asset_type == 'auto':
            asset_type = self._detect_asset_type(ticker)

        if asset_type == 'fx':
            return self.fetch_fx_ohlcv(ticker, days)
        elif asset_type == 'metal':
            return self.fetch_metal_ohlcv(ticker, days)
        elif asset_type == 'risk':
            # For risk indicators, try OHLCV from yfinance
            yf_ticker = YFINANCE_RISK_MAP.get(ticker.upper(), f'^{ticker}')
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 10)

            data = self.yf.download(
                yf_ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )

            if data.empty:
                return pd.DataFrame()

            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = data[['Open', 'High', 'Low', 'Close']].copy()
            df['Volume'] = 0  # VIX doesn't have volume
            df = df.reset_index()
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df['Ticker'] = ticker

            return df.tail(days)
        else:
            raise ValueError(f"OHLCV not available for asset type: {asset_type}")

    def fetch_multiple(self, tickers: List[str], asset_type: str = 'auto',
                       days: int = 252) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers.

        Args:
            tickers: List of tickers
            asset_type: Asset type or 'auto'
            days: Number of days of history

        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch_spot(ticker, asset_type, days)
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                results[ticker] = pd.DataFrame()
        return results

    def _detect_asset_type(self, ticker: str) -> str:
        """Auto-detect asset type from ticker."""
        ticker_upper = ticker.upper()

        # FX pairs
        fx_pairs = ['EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'NZD', 'CNH', 'DXY']
        if any(fx in ticker_upper for fx in fx_pairs):
            if 'XAU' in ticker_upper or 'XAG' in ticker_upper:
                return 'metal'
            return 'fx'

        # Metals
        if any(metal in ticker_upper for metal in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            return 'metal'

        # Risk indicators
        if any(risk in ticker_upper for risk in ['VIX', 'MOVE', 'CVIX']):
            return 'risk'

        # Rates
        if any(rate in ticker_upper for rate in ['US1', 'US2', 'US3', 'US5', 'US7', 'US10',
                                                   'US20', 'US30', 'SOFR', 'TIPS', 'BE']):
            return 'rate'

        # Macro
        if any(macro in ticker_upper for macro in ['CPI', 'GDP', 'UNRATE', 'ISM', 'PAYEMS']):
            return 'macro'

        # Default to FX
        return 'fx'

    def is_available(self) -> bool:
        """Check if free data sources are available."""
        try:
            _ = self.yf
            return True
        except:
            return False


# =============================================================================
# DATA LOADER FACTORY
# =============================================================================

def get_data_loader(config: Dict) -> Union['BloombergDataLoader', FreeDataLoader]:
    """
    Factory function to get the appropriate data loader based on config.

    Args:
        config: Configuration dictionary with 'data_source' key
                ('bloomberg' or 'free')

    Returns:
        BloombergDataLoader or FreeDataLoader instance
    """
    data_source = config.get('project', {}).get('data_mode', 'free')

    if data_source == 'bloomberg':
        try:
            from .bloomberg_loader import BloombergDataLoader
            ticker_config = config.get('tickers', {})
            loader = BloombergDataLoader(ticker_config)
            if loader.is_available():
                logger.info("Using Bloomberg data loader")
                return loader
            else:
                logger.warning("Bloomberg not available, falling back to free data")
        except ImportError:
            logger.warning("Bloomberg loader not available, falling back to free data")

    # Default to free data loader
    fred_key = config.get('api_keys', {}).get('fred')
    loader = FreeDataLoader(fred_api_key=fred_key)
    logger.info("Using free data loader (yfinance + FRED)")
    return loader
