"""
Industry Classifier for EGM/PEAD Scanner

Provides industry classification and relative strength analysis:
- Classify stocks into granular sub-industries
- Calculate relative strength vs industry peers
- Identify industry leaders and laggards
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .industry_data import (
    INDUSTRY_CLASSIFICATION,
    TICKER_TO_INDUSTRY,
    get_industry_for_ticker,
    get_peers_for_ticker,
    search_by_keyword,
)
from .api_client import APIClient


# =============================================================================
# CONFIGURATION
# =============================================================================

RELATIVE_STRENGTH_CRITERIA = {
    'comparison_period_days': 20,    # Compare returns over 20 days
    'min_percentile_rank': 75,       # Must be in top 25% of industry
    'min_peers_for_comparison': 5,   # Need at least 5 peers for valid comparison
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class IndustryInfo:
    """Industry classification information."""
    ticker: str
    sector: str
    industry: str
    description: str
    peers: List[str]
    peer_count: int
    classification_source: str  # 'manual', 'api', 'fallback'


@dataclass
class RelativeStrength:
    """Relative strength analysis results."""
    ticker: str
    industry: str
    stock_return: float
    industry_avg_return: float
    outperformance: float
    percentile_rank: float
    rank: int
    total_peers: int
    is_leader: bool
    comparison_period_days: int
    peer_returns: Dict[str, float]
    analysis_date: datetime


@dataclass
class IndustryAnalysis:
    """Complete industry analysis for a stock."""
    ticker: str
    info: IndustryInfo
    relative_strength: Optional[RelativeStrength]
    meets_criteria: bool
    analysis_date: datetime


# =============================================================================
# INDUSTRY CLASSIFIER CLASS
# =============================================================================

class IndustryClassifier:
    """
    Classifies stocks into industries and calculates relative strength.

    Uses the granular industry classification from industry_data.py
    for meaningful peer comparison.
    """

    def __init__(self, api_client: APIClient = None,
                 comparison_period: int = None,
                 min_percentile: float = None,
                 min_peers: int = None):
        """
        Initialize industry classifier.

        Args:
            api_client: APIClient for fetching price data
            comparison_period: Days for return comparison
            min_percentile: Minimum percentile rank to be a leader
            min_peers: Minimum peers for valid comparison
        """
        self.api_client = api_client or APIClient()
        self.comparison_period = comparison_period or RELATIVE_STRENGTH_CRITERIA['comparison_period_days']
        self.min_percentile = min_percentile or RELATIVE_STRENGTH_CRITERIA['min_percentile_rank']
        self.min_peers = min_peers or RELATIVE_STRENGTH_CRITERIA['min_peers_for_comparison']

        # Cache for price data
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def get_industry_info(self, ticker: str) -> IndustryInfo:
        """
        Get industry classification for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            IndustryInfo with classification details
        """
        ticker = ticker.upper()

        # Check manual classification first
        info = get_industry_for_ticker(ticker)

        if info:
            peers = get_peers_for_ticker(ticker)
            return IndustryInfo(
                ticker=ticker,
                sector=info['sector'],
                industry=info['industry'],
                description=info['description'],
                peers=peers,
                peer_count=len(peers),
                classification_source='manual',
            )

        # Fallback: try to get from yfinance
        try:
            yf_info = self.api_client.fetch_info(ticker)
            if yf_info:
                sector = yf_info.get('sector', 'unknown')
                industry = yf_info.get('industry', 'unknown')

                return IndustryInfo(
                    ticker=ticker,
                    sector=sector.lower().replace(' ', '_'),
                    industry=industry.lower().replace(' ', '_'),
                    description=industry,
                    peers=[],  # Can't get peers without manual classification
                    peer_count=0,
                    classification_source='api',
                )
        except Exception:
            pass

        # Ultimate fallback
        return IndustryInfo(
            ticker=ticker,
            sector='unknown',
            industry='unknown',
            description='Unknown industry',
            peers=[],
            peer_count=0,
            classification_source='fallback',
        )

    def _get_price_data(self, ticker: str, days: int = None) -> Optional[pd.DataFrame]:
        """
        Get price data for a ticker (with caching).

        Args:
            ticker: Stock ticker symbol
            days: Number of days of history

        Returns:
            DataFrame with price data
        """
        days = days or self.comparison_period + 10  # Extra buffer

        cache_key = f"{ticker}_{days}"
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        df = self.api_client.fetch_price_history(ticker, days=days)
        if df is not None:
            self._price_cache[cache_key] = df

        return df

    def calculate_return(self, df: pd.DataFrame, days: int = None) -> float:
        """
        Calculate return over a period.

        Args:
            df: DataFrame with 'close' column
            days: Number of days for return calculation

        Returns:
            Return percentage
        """
        days = days or self.comparison_period

        if df is None or len(df) < 2:
            return 0.0

        if len(df) < days:
            days = len(df)

        recent = df.tail(days)
        if len(recent) < 2:
            return 0.0

        start_price = recent['close'].iloc[0]
        end_price = recent['close'].iloc[-1]

        if start_price == 0:
            return 0.0

        return ((end_price / start_price) - 1) * 100

    def calculate_relative_strength(self, ticker: str,
                                    industry_info: IndustryInfo = None,
                                    period_days: int = None) -> Optional[RelativeStrength]:
        """
        Calculate relative strength vs industry peers.

        Args:
            ticker: Stock ticker symbol
            industry_info: Pre-fetched IndustryInfo (optional)
            period_days: Days for return comparison

        Returns:
            RelativeStrength or None if insufficient peers
        """
        ticker = ticker.upper()
        period_days = period_days or self.comparison_period

        # Get industry info if not provided
        if industry_info is None:
            industry_info = self.get_industry_info(ticker)

        # Need peers for comparison
        if not industry_info.peers or len(industry_info.peers) < self.min_peers:
            return None

        # Get returns for all peers (including target)
        all_tickers = [ticker] + [p for p in industry_info.peers if p != ticker]
        returns = {}

        for t in all_tickers:
            df = self._get_price_data(t, period_days + 10)
            if df is not None:
                returns[t] = self.calculate_return(df, period_days)

        # Need minimum peers with valid data
        if len(returns) < self.min_peers:
            return None

        # Get target stock return
        stock_return = returns.get(ticker, 0.0)

        # Calculate industry average (excluding target)
        peer_returns = {k: v for k, v in returns.items() if k != ticker}
        if not peer_returns:
            return None

        industry_avg = sum(peer_returns.values()) / len(peer_returns)

        # Calculate percentile rank
        all_returns = sorted(returns.values())
        rank_position = all_returns.index(stock_return) + 1 if stock_return in all_returns else len(all_returns)
        percentile_rank = (rank_position / len(all_returns)) * 100

        # Calculate rank (1 = best)
        sorted_by_return = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        rank = next((i + 1 for i, (t, _) in enumerate(sorted_by_return) if t == ticker), len(sorted_by_return))

        return RelativeStrength(
            ticker=ticker,
            industry=industry_info.industry,
            stock_return=round(stock_return, 2),
            industry_avg_return=round(industry_avg, 2),
            outperformance=round(stock_return - industry_avg, 2),
            percentile_rank=round(percentile_rank, 1),
            rank=rank,
            total_peers=len(returns),
            is_leader=percentile_rank >= self.min_percentile,
            comparison_period_days=period_days,
            peer_returns=peer_returns,
            analysis_date=datetime.now(),
        )

    def analyze(self, ticker: str, period_days: int = None) -> IndustryAnalysis:
        """
        Perform complete industry analysis.

        Args:
            ticker: Stock ticker symbol
            period_days: Days for return comparison

        Returns:
            IndustryAnalysis with all metrics
        """
        ticker = ticker.upper()

        # Get industry classification
        info = self.get_industry_info(ticker)

        # Calculate relative strength
        rel_strength = None
        if info.peers:
            rel_strength = self.calculate_relative_strength(ticker, info, period_days)

        # Determine if meets criteria
        meets = False
        if rel_strength:
            meets = rel_strength.is_leader

        return IndustryAnalysis(
            ticker=ticker,
            info=info,
            relative_strength=rel_strength,
            meets_criteria=meets,
            analysis_date=datetime.now(),
        )

    def get_industry_leaders(self, sector: str, industry: str,
                             top_n: int = 5,
                             period_days: int = None) -> List[Tuple[str, float]]:
        """
        Get top performers in an industry.

        Args:
            sector: Sector name
            industry: Industry name
            top_n: Number of top performers to return
            period_days: Days for return calculation

        Returns:
            List of (ticker, return) tuples, sorted by return
        """
        period_days = period_days or self.comparison_period

        # Get all tickers in industry
        industry_data = INDUSTRY_CLASSIFICATION.get(sector, {}).get(industry, {})
        tickers = industry_data.get('examples', [])

        if not tickers:
            return []

        # Calculate returns
        returns = []
        for ticker in tickers:
            df = self._get_price_data(ticker, period_days + 10)
            if df is not None:
                ret = self.calculate_return(df, period_days)
                returns.append((ticker, ret))

        # Sort by return and return top N
        returns.sort(key=lambda x: x[1], reverse=True)
        return returns[:top_n]

    def get_industry_laggards(self, sector: str, industry: str,
                              bottom_n: int = 5,
                              period_days: int = None) -> List[Tuple[str, float]]:
        """
        Get worst performers in an industry.

        Args:
            sector: Sector name
            industry: Industry name
            bottom_n: Number of bottom performers to return
            period_days: Days for return calculation

        Returns:
            List of (ticker, return) tuples, sorted by return (ascending)
        """
        period_days = period_days or self.comparison_period

        # Get all tickers in industry
        industry_data = INDUSTRY_CLASSIFICATION.get(sector, {}).get(industry, {})
        tickers = industry_data.get('examples', [])

        if not tickers:
            return []

        # Calculate returns
        returns = []
        for ticker in tickers:
            df = self._get_price_data(ticker, period_days + 10)
            if df is not None:
                ret = self.calculate_return(df, period_days)
                returns.append((ticker, ret))

        # Sort by return ascending and return bottom N
        returns.sort(key=lambda x: x[1])
        return returns[:bottom_n]

    def compare_to_peers(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Create a comparison table of ticker vs all peers.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with peer comparison
        """
        analysis = self.analyze(ticker)

        if not analysis.relative_strength:
            return None

        rs = analysis.relative_strength

        # Create DataFrame
        data = []
        for peer, ret in rs.peer_returns.items():
            data.append({
                'ticker': peer,
                'return_pct': ret,
                'vs_avg': ret - rs.industry_avg_return,
            })

        # Add target ticker
        data.append({
            'ticker': ticker,
            'return_pct': rs.stock_return,
            'vs_avg': rs.outperformance,
        })

        df = pd.DataFrame(data)
        df = df.sort_values('return_pct', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)

        return df

    def clear_cache(self):
        """Clear the price data cache."""
        self._price_cache = {}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_relative_strength_score(rel_strength: RelativeStrength) -> float:
    """
    Calculate a score for relative strength (0-100).

    Args:
        rel_strength: RelativeStrength object

    Returns:
        Score from 0-100
    """
    if rel_strength is None:
        return 50.0  # Neutral if no comparison available

    # Percentile rank is already 0-100
    return rel_strength.percentile_rank


def is_industry_leader(ticker: str, classifier: IndustryClassifier = None) -> Tuple[bool, Optional[RelativeStrength]]:
    """
    Quick check if a stock is an industry leader.

    Args:
        ticker: Stock ticker symbol
        classifier: IndustryClassifier instance (optional)

    Returns:
        Tuple of (is_leader, RelativeStrength)
    """
    classifier = classifier or IndustryClassifier()
    analysis = classifier.analyze(ticker)

    if analysis.relative_strength:
        return analysis.relative_strength.is_leader, analysis.relative_strength

    return False, None


def get_sector_performance(sector: str, period_days: int = 20) -> Dict[str, float]:
    """
    Calculate average returns for all industries in a sector.

    Args:
        sector: Sector name
        period_days: Days for return calculation

    Returns:
        Dictionary mapping industry to average return
    """
    classifier = IndustryClassifier()
    results = {}

    industries = INDUSTRY_CLASSIFICATION.get(sector, {})
    for industry, data in industries.items():
        tickers = data.get('examples', [])
        if not tickers:
            continue

        returns = []
        for ticker in tickers[:10]:  # Limit to 10 for speed
            df = classifier._get_price_data(ticker, period_days + 10)
            if df is not None:
                ret = classifier.calculate_return(df, period_days)
                returns.append(ret)

        if returns:
            results[industry] = sum(returns) / len(returns)

    return results


if __name__ == '__main__':
    # Test the industry classifier
    print("Testing Industry Classifier...")

    classifier = IndustryClassifier()

    print("\n1. Testing industry info lookup...")
    info = classifier.get_industry_info('NVDA')
    print(f"   Ticker: {info.ticker}")
    print(f"   Sector: {info.sector}")
    print(f"   Industry: {info.industry}")
    print(f"   Description: {info.description}")
    print(f"   Peers: {info.peers[:5]}... ({info.peer_count} total)")
    print(f"   Source: {info.classification_source}")

    print("\n2. Testing unknown ticker...")
    info = classifier.get_industry_info('FAKE123')
    print(f"   Ticker: {info.ticker}")
    print(f"   Industry: {info.industry}")
    print(f"   Source: {info.classification_source}")

    print("\n3. Testing relative strength calculation...")
    print("   (This requires API access and may take a moment...)")

    # Create sample data for testing without API
    print("\n4. Testing with sample data...")

    # Simulate relative strength
    sample_rs = RelativeStrength(
        ticker='NVDA',
        industry='semiconductors_logic',
        stock_return=18.5,
        industry_avg_return=5.8,
        outperformance=12.7,
        percentile_rank=94.0,
        rank=1,
        total_peers=10,
        is_leader=True,
        comparison_period_days=20,
        peer_returns={'AMD': 12.3, 'INTC': -3.2, 'QCOM': 4.1},
        analysis_date=datetime.now(),
    )

    print(f"   Stock Return: {sample_rs.stock_return}%")
    print(f"   Industry Avg: {sample_rs.industry_avg_return}%")
    print(f"   Outperformance: {sample_rs.outperformance}%")
    print(f"   Percentile Rank: {sample_rs.percentile_rank}")
    print(f"   Is Leader: {sample_rs.is_leader}")

    print("\n5. Testing relative strength score...")
    score = get_relative_strength_score(sample_rs)
    print(f"   RS Score: {score}/100")

    print("\nIndustry Classifier test complete.")
