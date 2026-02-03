"""
BQL Data Loader for BQNT Carry Execution.

Handles all data fetching from Bloomberg using BQL queries.
Replaces yfinance/FRED data sources from the proof-of-concept.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

# BQL import - only available in BQNT environment
try:
    import bql
    BQL_AVAILABLE = True
except ImportError:
    BQL_AVAILABLE = False
    bql = None

from ..config.tickers import (
    ALL_TICKERS,
    FX_SPOT_TICKERS,
    FX_FORWARD_TICKERS,
    METALS_TICKERS,
    TREASURY_TICKERS,
    VOLATILITY_TICKERS,
    get_bloomberg_ticker,
    get_forward_ticker,
)
from ..config.parameters import DATA_PARAMS


class BQLDataLoader:
    """
    Unified data loader using Bloomberg BQL.

    Provides methods to fetch:
    - Historical OHLCV data
    - Close prices for multiple assets
    - Forward points for carry calculation
    - Yields and volatility indices
    """

    def __init__(self, cache=None):
        """
        Initialize BQL data loader.

        Args:
            cache: Optional BQNTCache instance for caching results
        """
        if not BQL_AVAILABLE:
            raise ImportError(
                "BQL not available. This module requires BQNT environment."
            )

        self.bq = bql.Service()
        self.cache = cache
        self._default_lookback = DATA_PARAMS['default_lookback']
        self._fill_method = DATA_PARAMS['fill_method']

    def fetch_ohlcv(
        self,
        ticker: str,
        days: int = 252,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker.

        Args:
            ticker: Internal ticker key (e.g., 'EURUSD') or Bloomberg ticker
            days: Number of days of history
            end_date: End date (default: today)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        # Check cache first
        cache_key = f"ohlcv_{ticker}_{days}"
        if self.cache and self.cache.has(cache_key):
            return self.cache.get(cache_key)

        # Convert to Bloomberg ticker if needed
        bbg_ticker = get_bloomberg_ticker(ticker)

        # Calculate date range
        if end_date is None:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Build BQL request
        request = bql.Request(
            bbg_ticker,
            {
                'Date': self.bq.data.px_last()['DATE'],
                'Open': self.bq.data.px_open()['value'],
                'High': self.bq.data.px_high()['value'],
                'Low': self.bq.data.px_low()['value'],
                'Close': self.bq.data.px_last()['value'],
                'Volume': self.bq.data.px_volume()['value'],
            },
            with_params={
                'dates': self.bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                ),
                'fill': self._fill_method,
            }
        )

        # Execute query
        response = self.bq.execute(request)
        df = pd.concat([d.df() for d in response], axis=1)

        # Clean up DataFrame
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        df = df.sort_index()

        # Cache result
        if self.cache:
            self.cache.set(cache_key, df)

        return df

    def fetch_prices(
        self,
        tickers: Union[str, List[str]],
        days: int = 252,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch close prices for multiple tickers.

        Args:
            tickers: Single ticker or list of internal ticker keys
            days: Number of days of history
            end_date: End date (default: today)

        Returns:
            DataFrame with tickers as columns, dates as index
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        # Check cache
        cache_key = f"prices_{'-'.join(sorted(tickers))}_{days}"
        if self.cache and self.cache.has(cache_key):
            return self.cache.get(cache_key)

        # Convert to Bloomberg tickers
        bbg_tickers = [get_bloomberg_ticker(t) for t in tickers]

        # Calculate date range
        if end_date is None:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Build BQL request
        request = bql.Request(
            bbg_tickers,
            {
                'Date': self.bq.data.px_last()['DATE'],
                'Close': self.bq.data.px_last()['value'],
            },
            with_params={
                'dates': self.bq.func.range(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                ),
                'fill': self._fill_method,
            }
        )

        # Execute query
        response = self.bq.execute(request)

        # Parse response into DataFrame
        dfs = []
        for i, data_item in enumerate(response):
            item_df = data_item.df()
            if not item_df.empty:
                item_df = item_df.rename(columns={'Close': tickers[i]})
                if 'Date' in item_df.columns:
                    item_df['Date'] = pd.to_datetime(item_df['Date'])
                    item_df.set_index('Date', inplace=True)
                dfs.append(item_df[[tickers[i]]])

        if dfs:
            df = pd.concat(dfs, axis=1)
            df = df.sort_index()
        else:
            df = pd.DataFrame()

        # Cache result
        if self.cache:
            self.cache.set(cache_key, df)

        return df

    def fetch_fx_data(self, days: int = 252) -> Dict[str, pd.DataFrame]:
        """
        Fetch FX spot and forward data for carry calculation.

        Args:
            days: Number of days of history

        Returns:
            Dict with 'spot' and 'forward' DataFrames
        """
        cache_key = f"fx_data_{days}"
        if self.cache and self.cache.has(cache_key):
            return self.cache.get(cache_key)

        # Fetch spot prices
        spot_tickers = list(FX_SPOT_TICKERS.keys())
        spot_df = self.fetch_prices(spot_tickers, days)

        # Fetch forward prices
        forward_tickers = list(FX_FORWARD_TICKERS.keys())
        forward_df = self.fetch_prices(forward_tickers, days)

        result = {
            'spot': spot_df,
            'forward': forward_df,
        }

        if self.cache:
            self.cache.set(cache_key, result)

        return result

    def fetch_metals_data(self, days: int = 252) -> pd.DataFrame:
        """
        Fetch precious metals price data.

        Args:
            days: Number of days of history

        Returns:
            DataFrame with metal prices
        """
        metal_tickers = list(METALS_TICKERS.keys())
        return self.fetch_prices(metal_tickers, days)

    def fetch_treasury_yields(self, days: int = 252) -> pd.DataFrame:
        """
        Fetch Treasury yield data.

        Args:
            days: Number of days of history

        Returns:
            DataFrame with Treasury yields
        """
        yield_tickers = list(TREASURY_TICKERS.keys())
        return self.fetch_prices(yield_tickers, days)

    def fetch_volatility_data(self, days: int = 252) -> pd.DataFrame:
        """
        Fetch volatility indices (VIX, FX Vol).

        Args:
            days: Number of days of history

        Returns:
            DataFrame with volatility indices
        """
        vol_tickers = list(VOLATILITY_TICKERS.keys())
        return self.fetch_prices(vol_tickers, days)

    def fetch_all_universe(self, days: int = 252) -> pd.DataFrame:
        """
        Fetch price data for entire trading universe.

        Args:
            days: Number of days of history

        Returns:
            DataFrame with all asset prices
        """
        cache_key = f"universe_{days}"
        if self.cache and self.cache.has(cache_key):
            return self.cache.get(cache_key)

        all_tickers = list(ALL_TICKERS.keys())
        df = self.fetch_prices(all_tickers, days)

        if self.cache:
            self.cache.set(cache_key, df)

        return df

    def fetch_latest(self, tickers: Union[str, List[str]]) -> pd.Series:
        """
        Fetch latest prices for tickers.

        Args:
            tickers: Single ticker or list of tickers

        Returns:
            Series with latest prices
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        bbg_tickers = [get_bloomberg_ticker(t) for t in tickers]

        request = bql.Request(
            bbg_tickers,
            {'Close': self.bq.data.px_last()['value']},
        )

        response = self.bq.execute(request)

        prices = {}
        for i, data_item in enumerate(response):
            df = data_item.df()
            if not df.empty:
                prices[tickers[i]] = df['Close'].iloc[-1]

        return pd.Series(prices)

    def calculate_returns(
        self,
        prices: pd.DataFrame,
        periods: int = 1,
        log_returns: bool = False
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.

        Args:
            prices: DataFrame of prices
            periods: Number of periods for return calculation
            log_returns: If True, calculate log returns

        Returns:
            DataFrame of returns
        """
        if log_returns:
            returns = np.log(prices / prices.shift(periods))
        else:
            returns = prices.pct_change(periods)

        return returns.dropna()

    def calculate_volatility(
        self,
        prices: pd.DataFrame,
        window: int = 21,
        annualize: bool = True
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility.

        Args:
            prices: DataFrame of prices
            window: Rolling window size
            annualize: If True, annualize volatility

        Returns:
            DataFrame of rolling volatility
        """
        returns = self.calculate_returns(prices)
        vol = returns.rolling(window).std()

        if annualize:
            vol = vol * np.sqrt(252)

        return vol


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def create_loader(cache=None) -> BQLDataLoader:
    """Create a BQL data loader instance."""
    return BQLDataLoader(cache=cache)
