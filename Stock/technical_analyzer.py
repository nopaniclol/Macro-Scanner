"""
Technical Analyzer for EGM/PEAD Scanner

Provides technical analysis metrics including:
- ADR% (Average Daily Range Percentage) - minimum 6% required
- ATR% (Average True Range Percentage) - minimum 7% required
- Moving Average Cluster Detection - 10/20/50 SMA convergence before catalyst
- Basic technical indicators (SMA, EMA, RSI, etc.)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

VOLATILITY_CRITERIA = {
    'min_adr_percent': 6.0,      # Minimum 6% Average Daily Range
    'min_atr_percent': 7.0,      # Minimum 7% Average True Range %
    'adr_lookback': 14,          # 14-day ADR calculation
    'atr_lookback': 14,          # 14-day ATR calculation
}

MA_CLUSTER_CRITERIA = {
    'ma_periods': [10, 20, 50],  # SMAs to check
    'max_spread_percent': 3.0,   # Max 3% spread between highest and lowest MA
    'lookback_for_cluster': 5,   # MAs must be clustered within last 5 days before gap
    'cluster_days_required': 3,  # At least 3 days of tight clustering
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VolatilityMetrics:
    """Volatility analysis results."""
    ticker: str
    adr_percent: float
    atr_percent: float
    adr_meets_criteria: bool
    atr_meets_criteria: bool
    meets_all_criteria: bool
    adr_lookback: int
    atr_lookback: int
    analysis_date: datetime


@dataclass
class MAClusterResult:
    """Moving average cluster analysis results."""
    ticker: str
    is_clustered: bool
    spread_percent: float
    cluster_quality: str  # 'tight', 'moderate', 'loose'
    sma_10: float
    sma_20: float
    sma_50: float
    days_clustered: int
    analysis_date: datetime
    gap_date: Optional[datetime] = None


@dataclass
class TechnicalAnalysis:
    """Complete technical analysis results."""
    ticker: str
    volatility: VolatilityMetrics
    ma_cluster: MAClusterResult
    current_price: float
    sma_10: float
    sma_20: float
    sma_50: float
    rsi_14: float
    above_10sma: bool
    above_20sma: bool
    above_50sma: bool
    analysis_date: datetime


# =============================================================================
# TECHNICAL ANALYZER CLASS
# =============================================================================

class TechnicalAnalyzer:
    """
    Technical analysis engine for EGM/PEAD scanner.

    Provides volatility metrics (ADR%, ATR%) and MA cluster detection
    to filter stocks for the EGM strategy.
    """

    def __init__(self, adr_lookback: int = 14, atr_lookback: int = 14):
        """
        Initialize technical analyzer.

        Args:
            adr_lookback: Lookback period for ADR calculation
            atr_lookback: Lookback period for ATR calculation
        """
        self.adr_lookback = adr_lookback
        self.atr_lookback = atr_lookback

    # =========================================================================
    # ADR% CALCULATION
    # =========================================================================

    def calculate_adr_percent(self, df: pd.DataFrame,
                              lookback: int = None) -> float:
        """
        Calculate Average Daily Range as a percentage.

        ADR% = Average((High - Low) / Low) * 100 over lookback period

        This measures the typical daily price swing as a percentage,
        indicating how much the stock moves on average each day.

        Args:
            df: DataFrame with 'high' and 'low' columns
            lookback: Number of days for calculation (default: self.adr_lookback)

        Returns:
            ADR percentage (e.g., 6.5 means 6.5%)
        """
        lookback = lookback or self.adr_lookback

        if len(df) < lookback:
            lookback = len(df)

        if lookback == 0:
            return 0.0

        # Get the most recent data
        recent = df.tail(lookback)

        # Calculate daily range as percentage
        # Using (High - Low) / Low * 100 for each day
        high = recent['high'].values
        low = recent['low'].values

        # Avoid division by zero
        low = np.where(low == 0, np.nan, low)
        daily_range_pct = ((high - low) / low) * 100

        # Return average, excluding NaN values
        return float(np.nanmean(daily_range_pct))

    # =========================================================================
    # ATR% CALCULATION
    # =========================================================================

    def calculate_atr(self, df: pd.DataFrame, lookback: int = None) -> pd.Series:
        """
        Calculate Average True Range.

        True Range = max(H-L, |H-Prev_C|, |L-Prev_C|)
        ATR = Rolling mean of True Range

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            lookback: ATR period (default: self.atr_lookback)

        Returns:
            Series of ATR values
        """
        lookback = lookback or self.atr_lookback

        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)

        # Calculate True Range components
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        # True Range is the maximum of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is the rolling mean
        atr = tr.rolling(window=lookback).mean()

        return atr

    def calculate_atr_percent(self, df: pd.DataFrame,
                              lookback: int = None) -> float:
        """
        Calculate ATR as a percentage of price.

        ATR% = (ATR / Close) * 100

        This normalizes ATR to account for different price levels,
        making it comparable across stocks.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            lookback: ATR period (default: self.atr_lookback)

        Returns:
            ATR percentage (e.g., 7.5 means 7.5%)
        """
        lookback = lookback or self.atr_lookback

        if len(df) < lookback:
            return 0.0

        atr = self.calculate_atr(df, lookback)
        close = df['close']

        # Calculate ATR%
        atr_pct = (atr / close) * 100

        # Return the most recent value
        return float(atr_pct.iloc[-1]) if not np.isnan(atr_pct.iloc[-1]) else 0.0

    # =========================================================================
    # VOLATILITY ANALYSIS
    # =========================================================================

    def analyze_volatility(self, df: pd.DataFrame,
                           ticker: str = '') -> VolatilityMetrics:
        """
        Perform complete volatility analysis.

        Checks if stock meets minimum ADR% and ATR% requirements:
        - ADR% >= 6% (stock moves enough daily)
        - ATR% >= 7% (accounts for gaps, better volatility measure)

        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol

        Returns:
            VolatilityMetrics with analysis results
        """
        adr_pct = self.calculate_adr_percent(df)
        atr_pct = self.calculate_atr_percent(df)

        adr_meets = adr_pct >= VOLATILITY_CRITERIA['min_adr_percent']
        atr_meets = atr_pct >= VOLATILITY_CRITERIA['min_atr_percent']

        return VolatilityMetrics(
            ticker=ticker,
            adr_percent=round(adr_pct, 2),
            atr_percent=round(atr_pct, 2),
            adr_meets_criteria=adr_meets,
            atr_meets_criteria=atr_meets,
            meets_all_criteria=adr_meets and atr_meets,
            adr_lookback=self.adr_lookback,
            atr_lookback=self.atr_lookback,
            analysis_date=datetime.now(),
        )

    # =========================================================================
    # MOVING AVERAGE CALCULATIONS
    # =========================================================================

    def calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period).mean()

    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    def add_moving_averages(self, df: pd.DataFrame,
                            periods: List[int] = None) -> pd.DataFrame:
        """
        Add moving averages to DataFrame.

        Args:
            df: DataFrame with 'close' column
            periods: List of MA periods (default: [10, 20, 50])

        Returns:
            DataFrame with added SMA columns
        """
        periods = periods or MA_CLUSTER_CRITERIA['ma_periods']
        df = df.copy()

        for period in periods:
            df[f'sma_{period}'] = self.calculate_sma(df['close'], period)

        return df

    # =========================================================================
    # MA CLUSTER DETECTION
    # =========================================================================

    def detect_ma_cluster(self, df: pd.DataFrame, gap_date: datetime = None,
                          ticker: str = '') -> MAClusterResult:
        """
        Detect if 10/20/50 SMAs were tightly clustered before a catalyst.

        Tight MAs indicate consolidation/coiling before an explosive move.
        When 10/20/50 SMAs converge within 3%, a breakout often follows.

        Args:
            df: DataFrame with OHLCV data
            gap_date: Date of the gap/catalyst (uses last date if None)
            ticker: Stock ticker symbol

        Returns:
            MAClusterResult with cluster analysis
        """
        # Add moving averages
        df = self.add_moving_averages(df)

        # Determine analysis window
        if gap_date is not None:
            # Get data before the gap
            pre_gap = df[df.index < gap_date]
            if len(pre_gap) < MA_CLUSTER_CRITERIA['lookback_for_cluster']:
                pre_gap = df.head(len(df) - 1)  # Use all but last day
        else:
            # Use recent data (last 5 days before latest)
            pre_gap = df.iloc[-6:-1] if len(df) > 5 else df

        if len(pre_gap) == 0:
            return MAClusterResult(
                ticker=ticker,
                is_clustered=False,
                spread_percent=999.0,
                cluster_quality='loose',
                sma_10=0,
                sma_20=0,
                sma_50=0,
                days_clustered=0,
                analysis_date=datetime.now(),
                gap_date=gap_date,
            )

        # Get the lookback window
        lookback = min(MA_CLUSTER_CRITERIA['lookback_for_cluster'], len(pre_gap))
        analysis_window = pre_gap.tail(lookback)

        # Calculate spread between MAs for each day
        spreads = []
        days_tight = 0

        for idx in range(len(analysis_window)):
            row = analysis_window.iloc[idx]
            sma_10 = row.get('sma_10', np.nan)
            sma_20 = row.get('sma_20', np.nan)
            sma_50 = row.get('sma_50', np.nan)

            if pd.isna(sma_10) or pd.isna(sma_20) or pd.isna(sma_50):
                continue

            ma_values = [sma_10, sma_20, sma_50]
            ma_high = max(ma_values)
            ma_low = min(ma_values)

            if ma_low > 0:
                spread = ((ma_high - ma_low) / ma_low) * 100
                spreads.append(spread)
                if spread <= MA_CLUSTER_CRITERIA['max_spread_percent']:
                    days_tight += 1

        # Calculate average spread
        avg_spread = np.mean(spreads) if spreads else 999.0

        # Get latest MA values
        latest = df.iloc[-1]
        sma_10 = latest.get('sma_10', 0)
        sma_20 = latest.get('sma_20', 0)
        sma_50 = latest.get('sma_50', 0)

        # Determine cluster quality
        if avg_spread < 2.0:
            quality = 'tight'
        elif avg_spread < 3.0:
            quality = 'moderate'
        else:
            quality = 'loose'

        is_clustered = (avg_spread <= MA_CLUSTER_CRITERIA['max_spread_percent'] and
                        days_tight >= MA_CLUSTER_CRITERIA['cluster_days_required'])

        return MAClusterResult(
            ticker=ticker,
            is_clustered=is_clustered,
            spread_percent=round(avg_spread, 2),
            cluster_quality=quality,
            sma_10=round(sma_10, 2) if not pd.isna(sma_10) else 0,
            sma_20=round(sma_20, 2) if not pd.isna(sma_20) else 0,
            sma_50=round(sma_50, 2) if not pd.isna(sma_50) else 0,
            days_clustered=days_tight,
            analysis_date=datetime.now(),
            gap_date=gap_date,
        )

    # =========================================================================
    # RSI CALCULATION
    # =========================================================================

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        Args:
            series: Price series (typically close)
            period: RSI period (default: 14)

        Returns:
            Series of RSI values (0-100)
        """
        delta = series.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Avoid division by zero
        avg_loss = avg_loss.replace(0, np.nan)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    # =========================================================================
    # COMPLETE TECHNICAL ANALYSIS
    # =========================================================================

    def analyze(self, df: pd.DataFrame, ticker: str = '',
                gap_date: datetime = None) -> TechnicalAnalysis:
        """
        Perform complete technical analysis.

        Combines volatility analysis, MA cluster detection, and
        basic technical indicators.

        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol
            gap_date: Date of gap/catalyst for MA cluster analysis

        Returns:
            TechnicalAnalysis with all metrics
        """
        # Volatility analysis
        volatility = self.analyze_volatility(df, ticker)

        # MA cluster detection
        ma_cluster = self.detect_ma_cluster(df, gap_date, ticker)

        # Add indicators
        df = self.add_moving_averages(df)
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)

        # Get latest values
        latest = df.iloc[-1]
        current_price = latest['close']
        sma_10 = latest.get('sma_10', 0)
        sma_20 = latest.get('sma_20', 0)
        sma_50 = latest.get('sma_50', 0)
        rsi_14 = latest.get('rsi_14', 50)

        return TechnicalAnalysis(
            ticker=ticker,
            volatility=volatility,
            ma_cluster=ma_cluster,
            current_price=round(current_price, 2),
            sma_10=round(sma_10, 2) if not pd.isna(sma_10) else 0,
            sma_20=round(sma_20, 2) if not pd.isna(sma_20) else 0,
            sma_50=round(sma_50, 2) if not pd.isna(sma_50) else 0,
            rsi_14=round(rsi_14, 2) if not pd.isna(rsi_14) else 50,
            above_10sma=current_price > sma_10 if sma_10 > 0 else False,
            above_20sma=current_price > sma_20 if sma_20 > 0 else False,
            above_50sma=current_price > sma_50 if sma_50 > 0 else False,
            analysis_date=datetime.now(),
        )

    # =========================================================================
    # BATCH ANALYSIS
    # =========================================================================

    def analyze_multiple(self, data: Dict[str, pd.DataFrame],
                         gap_dates: Dict[str, datetime] = None) -> Dict[str, TechnicalAnalysis]:
        """
        Analyze multiple stocks.

        Args:
            data: Dictionary mapping ticker to DataFrame
            gap_dates: Dictionary mapping ticker to gap date

        Returns:
            Dictionary mapping ticker to TechnicalAnalysis
        """
        gap_dates = gap_dates or {}
        results = {}

        for ticker, df in data.items():
            gap_date = gap_dates.get(ticker)
            results[ticker] = self.analyze(df, ticker, gap_date)

        return results

    def filter_by_volatility(self, data: Dict[str, pd.DataFrame],
                             min_adr: float = None,
                             min_atr: float = None) -> Dict[str, TechnicalAnalysis]:
        """
        Filter stocks by volatility requirements.

        Args:
            data: Dictionary mapping ticker to DataFrame
            min_adr: Minimum ADR% (default: 6%)
            min_atr: Minimum ATR% (default: 7%)

        Returns:
            Dictionary of stocks meeting volatility criteria
        """
        min_adr = min_adr or VOLATILITY_CRITERIA['min_adr_percent']
        min_atr = min_atr or VOLATILITY_CRITERIA['min_atr_percent']

        results = {}

        for ticker, df in data.items():
            analysis = self.analyze(df, ticker)
            if (analysis.volatility.adr_percent >= min_adr and
                    analysis.volatility.atr_percent >= min_atr):
                results[ticker] = analysis

        return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_volatility_requirements(df: pd.DataFrame,
                                  min_adr: float = 6.0,
                                  min_atr: float = 7.0) -> Tuple[bool, Dict]:
    """
    Quick check if a stock meets volatility requirements.

    Args:
        df: DataFrame with OHLCV data
        min_adr: Minimum ADR%
        min_atr: Minimum ATR%

    Returns:
        Tuple of (meets_criteria, details_dict)
    """
    analyzer = TechnicalAnalyzer()
    volatility = analyzer.analyze_volatility(df)

    meets = volatility.adr_percent >= min_adr and volatility.atr_percent >= min_atr

    return meets, {
        'adr_percent': volatility.adr_percent,
        'atr_percent': volatility.atr_percent,
        'meets_criteria': meets,
    }


def check_ma_cluster(df: pd.DataFrame, gap_date: datetime = None) -> Tuple[bool, Dict]:
    """
    Quick check if MAs were clustered before a catalyst.

    Args:
        df: DataFrame with OHLCV data
        gap_date: Date of the gap/catalyst

    Returns:
        Tuple of (is_clustered, details_dict)
    """
    analyzer = TechnicalAnalyzer()
    result = analyzer.detect_ma_cluster(df, gap_date)

    return result.is_clustered, {
        'spread_percent': result.spread_percent,
        'cluster_quality': result.cluster_quality,
        'days_clustered': result.days_clustered,
    }


if __name__ == '__main__':
    # Test the technical analyzer
    print("Testing Technical Analyzer...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    # Generate sample OHLCV data with some volatility
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
        'high': prices * (1 + np.random.uniform(0.01, 0.04, 100)),
        'low': prices * (1 + np.random.uniform(-0.04, -0.01, 100)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100),
    }, index=dates)

    # Test analyzer
    analyzer = TechnicalAnalyzer()

    print("\n1. Testing ADR% calculation...")
    adr = analyzer.calculate_adr_percent(df)
    print(f"   ADR%: {adr:.2f}%")

    print("\n2. Testing ATR% calculation...")
    atr = analyzer.calculate_atr_percent(df)
    print(f"   ATR%: {atr:.2f}%")

    print("\n3. Testing volatility analysis...")
    volatility = analyzer.analyze_volatility(df, 'TEST')
    print(f"   ADR%: {volatility.adr_percent}% (meets: {volatility.adr_meets_criteria})")
    print(f"   ATR%: {volatility.atr_percent}% (meets: {volatility.atr_meets_criteria})")
    print(f"   Overall: {'PASS' if volatility.meets_all_criteria else 'FAIL'}")

    print("\n4. Testing MA cluster detection...")
    ma_cluster = analyzer.detect_ma_cluster(df, ticker='TEST')
    print(f"   Spread: {ma_cluster.spread_percent}%")
    print(f"   Quality: {ma_cluster.cluster_quality}")
    print(f"   Clustered: {ma_cluster.is_clustered}")

    print("\n5. Testing complete analysis...")
    analysis = analyzer.analyze(df, 'TEST')
    print(f"   Price: ${analysis.current_price}")
    print(f"   SMA 10/20/50: ${analysis.sma_10}/${analysis.sma_20}/${analysis.sma_50}")
    print(f"   RSI(14): {analysis.rsi_14}")
    print(f"   Above SMAs: 10={analysis.above_10sma}, 20={analysis.above_20sma}, 50={analysis.above_50sma}")

    print("\nTechnical Analyzer test complete.")
