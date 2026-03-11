"""
Earnings Validator for EGM/PEAD Scanner

Validates earnings surprise criteria:
- EPS surprise >= 25% (actual vs estimate)
- Revenue surprise >= 5% (optional)
- Raised guidance (bonus)
- Recent earnings report (within N days)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

from .api_client import APIClient, EarningsData


# =============================================================================
# CONFIGURATION
# =============================================================================

EARNINGS_CRITERIA = {
    'min_eps_surprise_pct': 25.0,   # >25% EPS beat vs consensus
    'min_revenue_surprise_pct': 5.0, # >5% revenue beat (optional)
    'days_since_earnings': 5,        # Must have reported within 5 days
    'max_days_lookback': 90,         # Maximum days to look back for earnings
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EarningsValidation:
    """Results of earnings validation."""
    ticker: str
    has_recent_earnings: bool
    days_since_report: Optional[int]
    eps_actual: Optional[float]
    eps_estimate: Optional[float]
    eps_surprise: Optional[float]
    eps_surprise_pct: Optional[float]
    revenue_actual: Optional[float]
    revenue_estimate: Optional[float]
    revenue_surprise_pct: Optional[float]
    meets_eps_criteria: bool
    meets_revenue_criteria: bool
    meets_all_criteria: bool
    report_date: Optional[datetime]
    validation_date: datetime
    source: str  # 'finnhub', 'alpha_vantage', 'manual', etc.


@dataclass
class EarningsCalendarEntry:
    """Entry from earnings calendar."""
    ticker: str
    report_date: datetime
    report_time: str  # 'bmo' (before market open), 'amc' (after market close)
    eps_estimate: Optional[float]
    revenue_estimate: Optional[float]


# =============================================================================
# EARNINGS VALIDATOR CLASS
# =============================================================================

class EarningsValidator:
    """
    Validates earnings data against EGM/PEAD criteria.

    The PEAD (Post-Earnings Announcement Drift) strategy requires
    a significant earnings surprise (typically 25%+) to generate
    the expected price drift over the following 60 days.
    """

    def __init__(self, api_client: APIClient = None,
                 min_eps_surprise: float = None,
                 min_revenue_surprise: float = None,
                 max_days_since_report: int = None):
        """
        Initialize earnings validator.

        Args:
            api_client: APIClient for fetching earnings data
            min_eps_surprise: Minimum EPS surprise percentage
            min_revenue_surprise: Minimum revenue surprise percentage
            max_days_since_report: Maximum days since earnings report
        """
        self.api_client = api_client or APIClient()
        self.min_eps_surprise = min_eps_surprise or EARNINGS_CRITERIA['min_eps_surprise_pct']
        self.min_revenue_surprise = min_revenue_surprise or EARNINGS_CRITERIA['min_revenue_surprise_pct']
        self.max_days_since_report = max_days_since_report or EARNINGS_CRITERIA['days_since_earnings']

    def calculate_surprise_percent(self, actual: float,
                                   estimate: float) -> Optional[float]:
        """
        Calculate surprise percentage.

        Args:
            actual: Actual reported value
            estimate: Consensus estimate

        Returns:
            Surprise percentage (positive = beat, negative = miss)
        """
        if actual is None or estimate is None:
            return None
        if estimate == 0:
            # If estimate was 0 and actual is positive, treat as infinite beat
            # If estimate was 0 and actual is negative, treat as miss
            if actual > 0:
                return 100.0  # Cap at 100%
            elif actual < 0:
                return -100.0
            else:
                return 0.0

        return ((actual - estimate) / abs(estimate)) * 100

    def get_earnings_surprise(self, ticker: str) -> Optional[EarningsData]:
        """
        Fetch earnings surprise data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            EarningsData or None
        """
        return self.api_client.get_earnings_surprise(ticker)

    def validate_earnings(self, ticker: str,
                          earnings_data: EarningsData = None,
                          reference_date: datetime = None) -> EarningsValidation:
        """
        Validate earnings against EGM criteria.

        Args:
            ticker: Stock ticker symbol
            earnings_data: Pre-fetched earnings data (optional)
            reference_date: Date to compare against (default: today)

        Returns:
            EarningsValidation with results
        """
        reference_date = reference_date or datetime.now()

        # Fetch earnings if not provided
        if earnings_data is None:
            earnings_data = self.get_earnings_surprise(ticker)

        # Handle missing data
        if earnings_data is None:
            return EarningsValidation(
                ticker=ticker,
                has_recent_earnings=False,
                days_since_report=None,
                eps_actual=None,
                eps_estimate=None,
                eps_surprise=None,
                eps_surprise_pct=None,
                revenue_actual=None,
                revenue_estimate=None,
                revenue_surprise_pct=None,
                meets_eps_criteria=False,
                meets_revenue_criteria=False,
                meets_all_criteria=False,
                report_date=None,
                validation_date=datetime.now(),
                source='none',
            )

        # Calculate days since report
        days_since = None
        if earnings_data.report_date:
            days_since = (reference_date.date() - earnings_data.report_date.date()).days

        # Check if recent
        has_recent = days_since is not None and days_since <= self.max_days_since_report

        # Calculate surprise if not already present
        eps_surprise_pct = earnings_data.eps_surprise_pct
        if eps_surprise_pct is None and earnings_data.eps_actual is not None:
            eps_surprise_pct = self.calculate_surprise_percent(
                earnings_data.eps_actual,
                earnings_data.eps_estimate
            )

        # Calculate revenue surprise
        rev_surprise_pct = earnings_data.revenue_surprise_pct
        if rev_surprise_pct is None and earnings_data.revenue_actual is not None:
            rev_surprise_pct = self.calculate_surprise_percent(
                earnings_data.revenue_actual,
                earnings_data.revenue_estimate
            )

        # Check criteria
        meets_eps = (eps_surprise_pct is not None and
                     eps_surprise_pct >= self.min_eps_surprise)
        meets_revenue = (rev_surprise_pct is not None and
                         rev_surprise_pct >= self.min_revenue_surprise)

        # EPS is required, revenue is optional bonus
        meets_all = has_recent and meets_eps

        return EarningsValidation(
            ticker=ticker,
            has_recent_earnings=has_recent,
            days_since_report=days_since,
            eps_actual=earnings_data.eps_actual,
            eps_estimate=earnings_data.eps_estimate,
            eps_surprise=earnings_data.eps_surprise,
            eps_surprise_pct=round(eps_surprise_pct, 2) if eps_surprise_pct else None,
            revenue_actual=earnings_data.revenue_actual,
            revenue_estimate=earnings_data.revenue_estimate,
            revenue_surprise_pct=round(rev_surprise_pct, 2) if rev_surprise_pct else None,
            meets_eps_criteria=meets_eps,
            meets_revenue_criteria=meets_revenue,
            meets_all_criteria=meets_all,
            report_date=earnings_data.report_date,
            validation_date=datetime.now(),
            source='api',
        )

    def validate_manual(self, ticker: str,
                        eps_actual: float,
                        eps_estimate: float,
                        report_date: datetime,
                        revenue_actual: float = None,
                        revenue_estimate: float = None,
                        reference_date: datetime = None) -> EarningsValidation:
        """
        Validate manually provided earnings data.

        Useful when API data is unavailable or for backtesting.

        Args:
            ticker: Stock ticker symbol
            eps_actual: Actual EPS
            eps_estimate: EPS estimate
            report_date: Earnings report date
            revenue_actual: Actual revenue (optional)
            revenue_estimate: Revenue estimate (optional)
            reference_date: Date to compare against

        Returns:
            EarningsValidation with results
        """
        reference_date = reference_date or datetime.now()

        # Create EarningsData object
        eps_surprise = eps_actual - eps_estimate if eps_estimate else None
        eps_surprise_pct = self.calculate_surprise_percent(eps_actual, eps_estimate)
        rev_surprise_pct = self.calculate_surprise_percent(revenue_actual, revenue_estimate)

        earnings_data = EarningsData(
            ticker=ticker,
            report_date=report_date,
            eps_actual=eps_actual,
            eps_estimate=eps_estimate,
            eps_surprise=eps_surprise,
            eps_surprise_pct=eps_surprise_pct,
            revenue_actual=revenue_actual,
            revenue_estimate=revenue_estimate,
            revenue_surprise_pct=rev_surprise_pct,
        )

        validation = self.validate_earnings(ticker, earnings_data, reference_date)
        validation.source = 'manual'
        return validation

    def get_upcoming_earnings(self, days_ahead: int = 7) -> List[EarningsCalendarEntry]:
        """
        Get stocks reporting earnings in the next N days.

        Args:
            days_ahead: Days to look ahead

        Returns:
            List of EarningsCalendarEntry objects
        """
        today = datetime.now()
        end_date = today + timedelta(days=days_ahead)

        calendar = self.api_client.fetch_earnings_calendar(
            from_date=today.strftime('%Y-%m-%d'),
            to_date=end_date.strftime('%Y-%m-%d')
        )

        if not calendar:
            return []

        entries = []
        for item in calendar:
            try:
                entries.append(EarningsCalendarEntry(
                    ticker=item.get('symbol', ''),
                    report_date=datetime.strptime(item.get('date', ''), '%Y-%m-%d') if item.get('date') else None,
                    report_time=item.get('hour', 'unknown'),
                    eps_estimate=item.get('epsEstimate'),
                    revenue_estimate=item.get('revenueEstimate'),
                ))
            except Exception:
                continue

        return entries

    def get_stocks_that_just_reported(self, days_back: int = 5) -> List[Dict]:
        """
        Get stocks that reported earnings in the last N days.

        Args:
            days_back: Days to look back

        Returns:
            List of dictionaries with ticker and report info
        """
        return self.api_client.get_recent_earnings_stocks(days=days_back)

    def validate_multiple(self, tickers: List[str],
                          reference_date: datetime = None) -> Dict[str, EarningsValidation]:
        """
        Validate earnings for multiple tickers.

        Args:
            tickers: List of ticker symbols
            reference_date: Date to compare against

        Returns:
            Dictionary mapping ticker to EarningsValidation
        """
        results = {}
        for ticker in tickers:
            results[ticker] = self.validate_earnings(ticker, reference_date=reference_date)
        return results

    def filter_by_surprise(self, validations: Dict[str, EarningsValidation],
                           min_surprise: float = None) -> Dict[str, EarningsValidation]:
        """
        Filter validations by minimum surprise threshold.

        Args:
            validations: Dictionary of EarningsValidation objects
            min_surprise: Minimum EPS surprise percentage

        Returns:
            Filtered dictionary
        """
        min_surprise = min_surprise or self.min_eps_surprise

        return {
            ticker: v for ticker, v in validations.items()
            if v.eps_surprise_pct is not None and v.eps_surprise_pct >= min_surprise
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_earnings_score(validation: EarningsValidation) -> float:
    """
    Calculate a score for earnings quality (0-100).

    Higher scores for:
    - Larger EPS surprise
    - Revenue beat
    - Recent report

    Args:
        validation: EarningsValidation object

    Returns:
        Score from 0-100
    """
    if not validation.has_recent_earnings:
        return 0.0

    score = 0.0

    # EPS surprise score (0-60 points)
    # 25% = 30 points, 50% = 60 points
    if validation.eps_surprise_pct is not None:
        eps_score = min(60, (validation.eps_surprise_pct / 50) * 60)
        score += max(0, eps_score)

    # Revenue surprise score (0-25 points)
    # 5% = 10 points, 15% = 25 points
    if validation.revenue_surprise_pct is not None and validation.revenue_surprise_pct > 0:
        rev_score = min(25, (validation.revenue_surprise_pct / 15) * 25)
        score += max(0, rev_score)

    # Recency score (0-15 points)
    # Same day = 15 points, 5 days ago = 5 points
    if validation.days_since_report is not None:
        recency_score = max(0, 15 - (validation.days_since_report * 2))
        score += recency_score

    return min(100, score)


def meets_pead_criteria(validation: EarningsValidation) -> bool:
    """
    Check if earnings meet PEAD strategy criteria.

    Args:
        validation: EarningsValidation object

    Returns:
        True if meets criteria
    """
    return (
        validation.has_recent_earnings and
        validation.meets_eps_criteria
    )


def get_surprise_quality(eps_surprise_pct: float) -> str:
    """
    Categorize earnings surprise quality.

    Args:
        eps_surprise_pct: EPS surprise percentage

    Returns:
        Quality string
    """
    if eps_surprise_pct is None:
        return 'unknown'
    elif eps_surprise_pct >= 50:
        return 'exceptional'
    elif eps_surprise_pct >= 35:
        return 'strong'
    elif eps_surprise_pct >= 25:
        return 'good'
    elif eps_surprise_pct >= 10:
        return 'moderate'
    elif eps_surprise_pct >= 0:
        return 'slight_beat'
    else:
        return 'miss'


if __name__ == '__main__':
    # Test the earnings validator
    print("Testing Earnings Validator...")

    # Create validator (without API keys for testing)
    validator = EarningsValidator()

    print("\n1. Testing manual validation...")
    validation = validator.validate_manual(
        ticker='TEST',
        eps_actual=1.25,
        eps_estimate=0.95,
        report_date=datetime.now() - timedelta(days=2),
        revenue_actual=5.2e9,
        revenue_estimate=4.8e9,
    )

    print(f"   Ticker: {validation.ticker}")
    print(f"   EPS: ${validation.eps_actual} vs ${validation.eps_estimate} estimate")
    print(f"   EPS Surprise: {validation.eps_surprise_pct}%")
    print(f"   Revenue Surprise: {validation.revenue_surprise_pct}%")
    print(f"   Days Since Report: {validation.days_since_report}")
    print(f"   Meets EPS Criteria: {validation.meets_eps_criteria}")
    print(f"   Meets All Criteria: {validation.meets_all_criteria}")

    print("\n2. Testing earnings score...")
    score = calculate_earnings_score(validation)
    print(f"   Earnings Score: {score:.1f}/100")

    print("\n3. Testing surprise quality...")
    quality = get_surprise_quality(validation.eps_surprise_pct)
    print(f"   Surprise Quality: {quality}")

    print("\n4. Testing PEAD criteria check...")
    meets = meets_pead_criteria(validation)
    print(f"   Meets PEAD Criteria: {meets}")

    print("\n5. Testing with miss (negative surprise)...")
    miss_validation = validator.validate_manual(
        ticker='MISS',
        eps_actual=0.80,
        eps_estimate=1.00,
        report_date=datetime.now() - timedelta(days=1),
    )
    print(f"   EPS Surprise: {miss_validation.eps_surprise_pct}%")
    print(f"   Quality: {get_surprise_quality(miss_validation.eps_surprise_pct)}")
    print(f"   Meets Criteria: {miss_validation.meets_all_criteria}")

    print("\nEarnings Validator test complete.")
