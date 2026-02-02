"""
Gap Detector for EGM/PEAD Scanner

Identifies price gaps meeting the EGM criteria:
- Minimum 10% gap up or down
- Volume spike (3x+ average daily volume)
- Gap occurred within last N days
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

GAP_CRITERIA = {
    'min_gap_percent': 10.0,        # Minimum 10% gap up/down
    'min_volume_multiple': 3.0,     # 3x average daily volume minimum
    'lookback_avg_volume': 20,      # 20-day average volume baseline
    'gap_recency_days': 5,          # Scan stocks that gapped within last 5 days
}

VOLUME_CRITERIA = {
    'gap_day_volume': 5.0,          # Ideal: 5x+ avg volume on gap day
    'good_volume_multiple': 3.0,    # Minimum: 3x avg volume
    'exceptional_volume': 10.0,     # Exceptional: 10x+ avg volume
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GapEvent:
    """Represents a detected price gap."""
    ticker: str
    gap_date: datetime
    gap_percent: float
    gap_direction: str  # 'up' or 'down'
    open_price: float
    prev_close: float
    close_price: float
    high_price: float
    low_price: float
    volume: int
    avg_volume: float
    volume_multiple: float
    gap_filled: bool
    gap_fill_percent: float
    meets_criteria: bool


@dataclass
class GapAnalysis:
    """Complete gap analysis for a stock."""
    ticker: str
    gaps: List[GapEvent]
    recent_gap: Optional[GapEvent]
    has_qualifying_gap: bool
    total_gaps_found: int
    analysis_date: datetime


# =============================================================================
# GAP DETECTOR CLASS
# =============================================================================

class GapDetector:
    """
    Detects price gaps meeting EGM/PEAD criteria.

    A gap is defined as:
    - Open significantly above/below previous close
    - Must be >= minimum gap percentage (default: 10%)
    - Should have volume confirmation (3x+ average)
    """

    def __init__(self, min_gap_percent: float = None,
                 min_volume_multiple: float = None,
                 volume_lookback: int = None):
        """
        Initialize gap detector.

        Args:
            min_gap_percent: Minimum gap percentage to qualify
            min_volume_multiple: Minimum volume multiple vs average
            volume_lookback: Days for average volume calculation
        """
        self.min_gap_percent = min_gap_percent or GAP_CRITERIA['min_gap_percent']
        self.min_volume_multiple = min_volume_multiple or GAP_CRITERIA['min_volume_multiple']
        self.volume_lookback = volume_lookback or GAP_CRITERIA['lookback_avg_volume']

    def calculate_average_volume(self, df: pd.DataFrame, end_idx: int,
                                 lookback: int = None) -> float:
        """
        Calculate average volume over lookback period.

        Args:
            df: DataFrame with 'volume' column
            end_idx: End index (exclusive) for calculation
            lookback: Number of days for average

        Returns:
            Average volume
        """
        lookback = lookback or self.volume_lookback
        start_idx = max(0, end_idx - lookback)

        if start_idx >= end_idx:
            return 0.0

        volumes = df['volume'].iloc[start_idx:end_idx]
        return float(volumes.mean()) if len(volumes) > 0 else 0.0

    def calculate_gap_percent(self, open_price: float,
                              prev_close: float) -> float:
        """
        Calculate gap percentage.

        Args:
            open_price: Opening price on gap day
            prev_close: Previous day's closing price

        Returns:
            Gap percentage (positive for gap up, negative for gap down)
        """
        if prev_close == 0:
            return 0.0
        return ((open_price - prev_close) / prev_close) * 100

    def check_gap_filled(self, gap_direction: str, open_price: float,
                         prev_close: float, low_price: float,
                         high_price: float) -> Tuple[bool, float]:
        """
        Check if gap was filled during the day.

        A gap is filled if price trades back to previous close.

        Args:
            gap_direction: 'up' or 'down'
            open_price: Opening price
            prev_close: Previous close
            low_price: Day's low
            high_price: Day's high

        Returns:
            Tuple of (gap_filled, fill_percent)
        """
        if gap_direction == 'up':
            # Gap up: check if low went below previous close
            gap_filled = low_price <= prev_close
            gap_size = open_price - prev_close
            if gap_size > 0:
                filled_amount = open_price - low_price
                fill_percent = min(100, (filled_amount / gap_size) * 100)
            else:
                fill_percent = 0
        else:
            # Gap down: check if high went above previous close
            gap_filled = high_price >= prev_close
            gap_size = prev_close - open_price
            if gap_size > 0:
                filled_amount = high_price - open_price
                fill_percent = min(100, (filled_amount / gap_size) * 100)
            else:
                fill_percent = 0

        return gap_filled, fill_percent

    def detect_gap(self, df: pd.DataFrame, idx: int,
                   ticker: str = '') -> Optional[GapEvent]:
        """
        Detect if a specific day has a qualifying gap.

        Args:
            df: DataFrame with OHLCV data
            idx: Index of the day to check
            ticker: Stock ticker symbol

        Returns:
            GapEvent if gap detected, None otherwise
        """
        if idx < 1 or idx >= len(df):
            return None

        current = df.iloc[idx]
        previous = df.iloc[idx - 1]

        open_price = current['open']
        prev_close = previous['close']
        close_price = current['close']
        high_price = current['high']
        low_price = current['low']
        volume = current['volume']

        # Calculate gap percentage
        gap_pct = self.calculate_gap_percent(open_price, prev_close)

        # Check if it meets minimum gap requirement
        if abs(gap_pct) < self.min_gap_percent:
            return None

        gap_direction = 'up' if gap_pct > 0 else 'down'

        # Calculate volume multiple
        avg_volume = self.calculate_average_volume(df, idx, self.volume_lookback)
        volume_multiple = volume / avg_volume if avg_volume > 0 else 0

        # Check if gap was filled
        gap_filled, fill_percent = self.check_gap_filled(
            gap_direction, open_price, prev_close, low_price, high_price
        )

        # Determine if gap meets all criteria
        meets_criteria = (
            abs(gap_pct) >= self.min_gap_percent and
            volume_multiple >= self.min_volume_multiple
        )

        # Get date from index
        gap_date = df.index[idx] if hasattr(df.index[idx], 'to_pydatetime') else datetime.now()
        if hasattr(gap_date, 'to_pydatetime'):
            gap_date = gap_date.to_pydatetime()

        return GapEvent(
            ticker=ticker,
            gap_date=gap_date,
            gap_percent=round(gap_pct, 2),
            gap_direction=gap_direction,
            open_price=round(open_price, 2),
            prev_close=round(prev_close, 2),
            close_price=round(close_price, 2),
            high_price=round(high_price, 2),
            low_price=round(low_price, 2),
            volume=int(volume),
            avg_volume=round(avg_volume, 0),
            volume_multiple=round(volume_multiple, 2),
            gap_filled=gap_filled,
            gap_fill_percent=round(fill_percent, 2),
            meets_criteria=meets_criteria,
        )

    def detect_gaps(self, df: pd.DataFrame, ticker: str = '',
                    min_gap_pct: float = None) -> List[GapEvent]:
        """
        Detect all gaps in a price series.

        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol
            min_gap_pct: Override minimum gap percentage

        Returns:
            List of GapEvent objects
        """
        min_gap = min_gap_pct if min_gap_pct is not None else self.min_gap_percent
        original_min = self.min_gap_percent
        self.min_gap_percent = min_gap

        gaps = []
        for i in range(1, len(df)):
            gap = self.detect_gap(df, i, ticker)
            if gap is not None:
                gaps.append(gap)

        self.min_gap_percent = original_min
        return gaps

    def detect_recent_gaps(self, df: pd.DataFrame, ticker: str = '',
                           days: int = None) -> List[GapEvent]:
        """
        Detect gaps within the last N days.

        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol
            days: Number of days to look back

        Returns:
            List of recent GapEvent objects
        """
        days = days or GAP_CRITERIA['gap_recency_days']

        # Only check the last N days
        if len(df) > days:
            recent_df = df.tail(days + 1)  # +1 because we need previous day for gap calc
            start_idx = len(df) - days
        else:
            recent_df = df
            start_idx = 1

        gaps = []
        for i in range(1, len(recent_df)):
            # Adjust index to original dataframe for volume calculation
            original_idx = start_idx + i - 1 if len(df) > days else i
            gap = self.detect_gap(df, original_idx, ticker)
            if gap is not None:
                gaps.append(gap)

        return gaps

    def analyze(self, df: pd.DataFrame, ticker: str = '',
                recent_days: int = None) -> GapAnalysis:
        """
        Perform complete gap analysis for a stock.

        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol
            recent_days: Days to look back for recent gaps

        Returns:
            GapAnalysis with all detected gaps
        """
        recent_days = recent_days or GAP_CRITERIA['gap_recency_days']

        # Detect all gaps
        all_gaps = self.detect_gaps(df, ticker)

        # Get recent gaps
        recent_gaps = self.detect_recent_gaps(df, ticker, recent_days)

        # Find most recent qualifying gap
        qualifying_gaps = [g for g in recent_gaps if g.meets_criteria]
        recent_gap = qualifying_gaps[-1] if qualifying_gaps else None

        return GapAnalysis(
            ticker=ticker,
            gaps=all_gaps,
            recent_gap=recent_gap,
            has_qualifying_gap=recent_gap is not None,
            total_gaps_found=len(all_gaps),
            analysis_date=datetime.now(),
        )

    def find_best_gaps(self, data: Dict[str, pd.DataFrame],
                       recent_days: int = None,
                       min_gaps: int = 1) -> List[GapAnalysis]:
        """
        Find stocks with the best qualifying gaps.

        Args:
            data: Dictionary mapping ticker to DataFrame
            recent_days: Days to look back
            min_gaps: Minimum qualifying gaps required

        Returns:
            List of GapAnalysis objects, sorted by gap magnitude
        """
        results = []

        for ticker, df in data.items():
            analysis = self.analyze(df, ticker, recent_days)
            if analysis.has_qualifying_gap:
                results.append(analysis)

        # Sort by gap magnitude (largest first)
        results.sort(
            key=lambda x: abs(x.recent_gap.gap_percent) if x.recent_gap else 0,
            reverse=True
        )

        return results

    def get_volume_quality(self, volume_multiple: float) -> str:
        """
        Categorize volume quality.

        Args:
            volume_multiple: Volume vs average multiple

        Returns:
            Volume quality string
        """
        if volume_multiple >= VOLUME_CRITERIA['exceptional_volume']:
            return 'exceptional'
        elif volume_multiple >= VOLUME_CRITERIA['gap_day_volume']:
            return 'strong'
        elif volume_multiple >= VOLUME_CRITERIA['good_volume_multiple']:
            return 'good'
        else:
            return 'weak'


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_gaps_in_ticker(df: pd.DataFrame, ticker: str = '',
                        min_gap_pct: float = 10.0,
                        min_volume_mult: float = 3.0) -> List[GapEvent]:
    """
    Quick function to find gaps in a single ticker.

    Args:
        df: DataFrame with OHLCV data
        ticker: Stock ticker symbol
        min_gap_pct: Minimum gap percentage
        min_volume_mult: Minimum volume multiple

    Returns:
        List of GapEvent objects
    """
    detector = GapDetector(
        min_gap_percent=min_gap_pct,
        min_volume_multiple=min_volume_mult
    )
    return detector.detect_gaps(df, ticker)


def has_recent_gap(df: pd.DataFrame, days: int = 5,
                   min_gap_pct: float = 10.0) -> Tuple[bool, Optional[GapEvent]]:
    """
    Quick check if stock has a recent qualifying gap.

    Args:
        df: DataFrame with OHLCV data
        days: Days to look back
        min_gap_pct: Minimum gap percentage

    Returns:
        Tuple of (has_gap, gap_event)
    """
    detector = GapDetector(min_gap_percent=min_gap_pct)
    gaps = detector.detect_recent_gaps(df, days=days)
    qualifying = [g for g in gaps if g.meets_criteria]

    if qualifying:
        return True, qualifying[-1]
    return False, None


def calculate_gap_score(gap: GapEvent) -> float:
    """
    Calculate a score for a gap event (0-100).

    Higher scores for:
    - Larger gap percentage
    - Higher volume
    - Gap not filled

    Args:
        gap: GapEvent object

    Returns:
        Score from 0-100
    """
    score = 0.0

    # Gap magnitude score (0-40 points)
    # 10% gap = 20 points, 20% gap = 40 points
    gap_score = min(40, (abs(gap.gap_percent) / 20) * 40)
    score += gap_score

    # Volume score (0-35 points)
    # 3x = 15 points, 5x = 25 points, 10x = 35 points
    if gap.volume_multiple >= 10:
        vol_score = 35
    elif gap.volume_multiple >= 5:
        vol_score = 25 + (gap.volume_multiple - 5) * 2
    elif gap.volume_multiple >= 3:
        vol_score = 15 + (gap.volume_multiple - 3) * 5
    else:
        vol_score = gap.volume_multiple * 5
    score += vol_score

    # Gap fill penalty (0-25 points)
    # Unfilled gap = 25 points, fully filled = 0 points
    fill_score = 25 * (1 - gap.gap_fill_percent / 100)
    score += fill_score

    return min(100, score)


if __name__ == '__main__':
    # Test the gap detector
    print("Testing Gap Detector...")

    # Create sample data with a gap
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')

    # Generate base prices
    base_price = 100
    prices = [base_price]
    for i in range(29):
        if i == 14:  # Create a 15% gap on day 15
            prices.append(prices[-1] * 1.15)
        else:
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))

    prices = np.array(prices)

    # Generate OHLCV
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, 30)),
        'high': prices * (1 + np.random.uniform(0.005, 0.02, 30)),
        'low': prices * (1 + np.random.uniform(-0.02, -0.005, 30)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 30),
    }, index=dates)

    # Make the gap day have high volume
    df.iloc[15, df.columns.get_loc('volume')] = 30000000  # 3x normal

    # Adjust open price to create the gap
    df.iloc[15, df.columns.get_loc('open')] = df.iloc[14, df.columns.get_loc('close')] * 1.12

    # Test detector
    detector = GapDetector(min_gap_percent=10.0, min_volume_multiple=2.0)

    print("\n1. Testing gap detection...")
    gaps = detector.detect_gaps(df, 'TEST')
    print(f"   Found {len(gaps)} gaps")

    for gap in gaps:
        print(f"\n   Gap on {gap.gap_date.strftime('%Y-%m-%d')}:")
        print(f"   - Direction: {gap.gap_direction}")
        print(f"   - Gap %: {gap.gap_percent}%")
        print(f"   - Volume: {gap.volume_multiple:.1f}x average")
        print(f"   - Filled: {gap.gap_filled} ({gap.gap_fill_percent}%)")
        print(f"   - Meets criteria: {gap.meets_criteria}")

    print("\n2. Testing recent gap detection...")
    recent = detector.detect_recent_gaps(df, 'TEST', days=20)
    print(f"   Found {len(recent)} recent gaps")

    print("\n3. Testing full analysis...")
    analysis = detector.analyze(df, 'TEST')
    print(f"   Total gaps: {analysis.total_gaps_found}")
    print(f"   Has qualifying gap: {analysis.has_qualifying_gap}")
    if analysis.recent_gap:
        print(f"   Recent gap: {analysis.recent_gap.gap_percent}% on {analysis.recent_gap.gap_date.strftime('%Y-%m-%d')}")

    print("\n4. Testing gap scoring...")
    if gaps:
        score = calculate_gap_score(gaps[0])
        print(f"   Gap score: {score:.1f}/100")

    print("\nGap Detector test complete.")
