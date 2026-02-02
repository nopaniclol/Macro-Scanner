"""
EGM/PEAD Scanner - Main Orchestrator

Combines all components to scan for EGM (Episodic Gap Momentum) /
PEAD (Post-Earnings Announcement Drift) setups.

Workflow:
1. Fetch stocks that reported earnings recently
2. Detect qualifying gaps (10%+, 3x+ volume)
3. Validate earnings surprise (25%+ EPS beat)
4. Analyze technical setup (ADR%, ATR%, MA cluster)
5. Check consolidation pattern and entry triggers
6. Calculate industry relative strength
7. Generate weighted composite signals
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from .api_client import APIClient
from .gap_detector import GapDetector, GapEvent, GapAnalysis
from .earnings_validator import EarningsValidator, EarningsValidation
from .technical_analyzer import TechnicalAnalyzer, TechnicalAnalysis
from .setup_analyzer import SetupAnalyzer, SetupAnalysis
from .industry_classifier import IndustryClassifier, RelativeStrength
from .signal_generator import SignalGenerator, SignalResult, ScanResult, format_scan_summary


# =============================================================================
# CONFIGURATION
# =============================================================================

SCANNER_CONFIG = {
    'days_lookback': 5,              # Look back 5 days for recent earnings
    'min_market_cap': 500_000_000,   # $500M minimum market cap
    'min_avg_volume': 500_000,       # 500K minimum average volume
    'max_stocks_to_scan': 100,       # Limit for API rate management
    'price_history_days': 90,        # Days of price history to fetch
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ScannerStats:
    """Scanner execution statistics."""
    total_candidates: int
    gaps_detected: int
    earnings_validated: int
    setups_analyzed: int
    signals_generated: int
    strong_buy_count: int
    buy_count: int
    watch_count: int
    disqualified_count: int
    execution_time_seconds: float
    scan_date: datetime


# =============================================================================
# EGM SCANNER CLASS
# =============================================================================

class EGMScanner:
    """
    Main scanner orchestrating all components for EGM/PEAD strategy.

    The scanner identifies stocks with:
    1. Recent earnings-driven gap (10%+)
    2. Significant earnings surprise (25%+ EPS)
    3. Strong volume confirmation (3x+)
    4. Proper volatility profile (ADR 6%+, ATR 7%+)
    5. Tight MA cluster pre-gap
    6. Quality consolidation pattern
    7. Industry leadership
    """

    def __init__(self, api_client: APIClient = None,
                 finnhub_key: str = None,
                 alpha_vantage_key: str = None):
        """
        Initialize EGM Scanner.

        Args:
            api_client: Pre-configured APIClient (optional)
            finnhub_key: Finnhub API key
            alpha_vantage_key: Alpha Vantage API key
        """
        # Initialize API client
        if api_client:
            self.api_client = api_client
        else:
            self.api_client = APIClient(
                finnhub_key=finnhub_key,
                alpha_vantage_key=alpha_vantage_key
        )

        # Initialize component analyzers
        self.gap_detector = GapDetector()
        self.earnings_validator = EarningsValidator(self.api_client)
        self.technical_analyzer = TechnicalAnalyzer()
        self.setup_analyzer = SetupAnalyzer()
        self.industry_classifier = IndustryClassifier(self.api_client)
        self.signal_generator = SignalGenerator()

        # Cache for price data
        self._price_cache: Dict[str, pd.DataFrame] = {}

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def _fetch_price_data(self, ticker: str, days: int = None) -> Optional[pd.DataFrame]:
        """
        Fetch price data with caching.

        Args:
            ticker: Stock ticker symbol
            days: Number of days of history

        Returns:
            DataFrame with OHLCV data
        """
        days = days or SCANNER_CONFIG['price_history_days']
        cache_key = f"{ticker}_{days}"

        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        df = self.api_client.fetch_price_history(ticker, days=days)
        if df is not None:
            self._price_cache[cache_key] = df

        return df

    def get_candidates_from_earnings(self, days_back: int = None) -> List[str]:
        """
        Get candidate stocks that reported earnings recently.

        Args:
            days_back: Days to look back for earnings

        Returns:
            List of ticker symbols
        """
        days_back = days_back or SCANNER_CONFIG['days_lookback']

        stocks = self.earnings_validator.get_stocks_that_just_reported(days_back)

        if not stocks:
            return []

        # Extract tickers
        tickers = [s.get('ticker', s.get('symbol', '')) for s in stocks]
        return [t for t in tickers if t]

    # =========================================================================
    # INDIVIDUAL STOCK ANALYSIS
    # =========================================================================

    def analyze_stock(self, ticker: str,
                      price_df: pd.DataFrame = None) -> Optional[ScanResult]:
        """
        Perform complete analysis on a single stock.

        Args:
            ticker: Stock ticker symbol
            price_df: Pre-fetched price data (optional)

        Returns:
            ScanResult with all analysis components
        """
        ticker = ticker.upper()

        # Fetch price data if not provided
        if price_df is None:
            price_df = self._fetch_price_data(ticker)

        if price_df is None or len(price_df) < 20:
            return None

        # Step 1: Detect gaps
        gap_analysis = self.gap_detector.analyze(price_df, ticker)
        gap = gap_analysis.recent_gap

        # Step 2: Validate earnings
        earnings = self.earnings_validator.validate_earnings(ticker)

        # Step 3: Technical analysis (with gap date if available)
        gap_date = gap.gap_date if gap else None
        technicals = self.technical_analyzer.analyze(price_df, ticker, gap_date)

        # Step 4: Setup analysis (if we have a gap)
        setup = None
        if gap and gap.meets_criteria:
            setup = self.setup_analyzer.analyze(
                price_df,
                gap.gap_date,
                gap.high_price,
                gap.volume,
                ticker
            )

        # Step 5: Industry relative strength
        rel_strength = None
        industry_info = self.industry_classifier.get_industry_info(ticker)
        if industry_info.peers:
            rel_strength = self.industry_classifier.calculate_relative_strength(
                ticker, industry_info
            )

        # Step 6: Generate signal
        result = self.signal_generator.generate_scan_result(
            ticker,
            gap=gap,
            earnings=earnings,
            technicals=technicals,
            setup=setup,
            rel_strength=rel_strength
        )

        return result

    # =========================================================================
    # BATCH SCANNING
    # =========================================================================

    def scan_tickers(self, tickers: List[str],
                     show_progress: bool = True) -> List[ScanResult]:
        """
        Scan a list of tickers.

        Args:
            tickers: List of ticker symbols
            show_progress: Print progress updates

        Returns:
            List of ScanResult objects
        """
        results = []
        total = len(tickers)

        for i, ticker in enumerate(tickers):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Scanning... {i + 1}/{total}")

            try:
                result = self.analyze_stock(ticker)
                if result:
                    results.append(result)
            except Exception as e:
                if show_progress:
                    print(f"  Error scanning {ticker}: {e}")
                continue

        return results

    def scan_earnings_today(self, days_back: int = 5,
                            show_progress: bool = True) -> Tuple[List[ScanResult], ScannerStats]:
        """
        Scan stocks that reported earnings recently.

        Args:
            days_back: Days to look back for earnings
            show_progress: Print progress updates

        Returns:
            Tuple of (results, stats)
        """
        start_time = datetime.now()

        if show_progress:
            print("\n" + "=" * 60)
            print("EGM/PEAD SCANNER")
            print("=" * 60)
            print(f"\nFetching stocks that reported in last {days_back} days...")

        # Get candidates
        candidates = self.get_candidates_from_earnings(days_back)

        if not candidates:
            if show_progress:
                print("No earnings candidates found.")
            return [], self._create_empty_stats(start_time)

        # Limit for rate management
        candidates = candidates[:SCANNER_CONFIG['max_stocks_to_scan']]

        if show_progress:
            print(f"Found {len(candidates)} candidates")
            print("\nScanning...")

        # Scan all candidates
        results = self.scan_tickers(candidates, show_progress)

        # Calculate stats
        stats = self._calculate_stats(results, len(candidates), start_time)

        if show_progress:
            print(f"\nScan complete in {stats.execution_time_seconds:.1f}s")

        return results, stats

    def scan_universe(self, tickers: List[str] = None,
                      show_progress: bool = True) -> Tuple[List[ScanResult], ScannerStats]:
        """
        Scan a custom universe of stocks.

        Args:
            tickers: List of tickers to scan
            show_progress: Print progress updates

        Returns:
            Tuple of (results, stats)
        """
        start_time = datetime.now()

        if not tickers:
            print("No tickers provided.")
            return [], self._create_empty_stats(start_time)

        if show_progress:
            print("\n" + "=" * 60)
            print("EGM/PEAD SCANNER - Custom Universe")
            print("=" * 60)
            print(f"\nScanning {len(tickers)} stocks...")

        results = self.scan_tickers(tickers, show_progress)
        stats = self._calculate_stats(results, len(tickers), start_time)

        if show_progress:
            print(f"\nScan complete in {stats.execution_time_seconds:.1f}s")

        return results, stats

    def scan_for_gaps(self, tickers: List[str],
                      show_progress: bool = True) -> List[GapAnalysis]:
        """
        Scan tickers for recent gaps (without full analysis).

        Useful for finding gap candidates quickly.

        Args:
            tickers: List of tickers to scan
            show_progress: Print progress updates

        Returns:
            List of GapAnalysis objects with qualifying gaps
        """
        results = []

        for ticker in tickers:
            try:
                df = self._fetch_price_data(ticker)
                if df is not None:
                    gap_analysis = self.gap_detector.analyze(df, ticker)
                    if gap_analysis.has_qualifying_gap:
                        results.append(gap_analysis)
            except Exception:
                continue

        return results

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def _calculate_stats(self, results: List[ScanResult],
                         total_candidates: int,
                         start_time: datetime) -> ScannerStats:
        """Calculate scanner statistics."""
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        strong_buy = len([r for r in results if r.signal.recommendation == 'strong_buy'])
        buy = len([r for r in results if r.signal.recommendation == 'buy'])
        watch = len([r for r in results if r.signal.recommendation == 'watch'])
        disqualified = len([r for r in results if r.signal.is_disqualified])

        gaps_detected = len([r for r in results if r.gap is not None])
        earnings_validated = len([r for r in results
                                  if r.earnings and r.earnings.meets_eps_criteria])
        setups_analyzed = len([r for r in results if r.setup is not None])

        return ScannerStats(
            total_candidates=total_candidates,
            gaps_detected=gaps_detected,
            earnings_validated=earnings_validated,
            setups_analyzed=setups_analyzed,
            signals_generated=len(results),
            strong_buy_count=strong_buy,
            buy_count=buy,
            watch_count=watch,
            disqualified_count=disqualified,
            execution_time_seconds=execution_time,
            scan_date=datetime.now(),
        )

    def _create_empty_stats(self, start_time: datetime) -> ScannerStats:
        """Create empty stats object."""
        return ScannerStats(
            total_candidates=0,
            gaps_detected=0,
            earnings_validated=0,
            setups_analyzed=0,
            signals_generated=0,
            strong_buy_count=0,
            buy_count=0,
            watch_count=0,
            disqualified_count=0,
            execution_time_seconds=0.0,
            scan_date=datetime.now(),
        )

    # =========================================================================
    # REPORTING
    # =========================================================================

    def print_results(self, results: List[ScanResult], stats: ScannerStats = None):
        """
        Print formatted results to console.

        Args:
            results: List of ScanResult objects
            stats: Optional ScannerStats
        """
        print(format_scan_summary(results))

        if stats:
            print("\n" + "-" * 60)
            print("SCANNER STATISTICS")
            print("-" * 60)
            print(f"Candidates Scanned:  {stats.total_candidates}")
            print(f"Gaps Detected:       {stats.gaps_detected}")
            print(f"Earnings Validated:  {stats.earnings_validated}")
            print(f"Setups Analyzed:     {stats.setups_analyzed}")
            print(f"Signals Generated:   {stats.signals_generated}")
            print(f"Execution Time:      {stats.execution_time_seconds:.1f}s")
            print("-" * 60)

    def export_results(self, results: List[ScanResult],
                       filename: str = None) -> pd.DataFrame:
        """
        Export results to DataFrame and optionally CSV.

        Args:
            results: List of ScanResult objects
            filename: Optional CSV filename

        Returns:
            DataFrame with results
        """
        data = []

        for r in results:
            row = {
                'ticker': r.ticker,
                'score': r.signal.total_score,
                'recommendation': r.signal.recommendation,
                'entry_price': r.signal.entry_price,
                'stop_loss': r.signal.stop_loss,
                'target_price': r.signal.target_price,
                'risk_reward': r.signal.risk_reward,
                'position_size_pct': r.signal.position_size_percent,
                'gap_percent': r.gap.gap_percent if r.gap else None,
                'volume_multiple': r.gap.volume_multiple if r.gap else None,
                'eps_surprise_pct': r.earnings.eps_surprise_pct if r.earnings else None,
                'adr_percent': r.technicals.volatility.adr_percent if r.technicals else None,
                'atr_percent': r.technicals.volatility.atr_percent if r.technicals else None,
                'ma_cluster_quality': r.technicals.ma_cluster.cluster_quality if r.technicals else None,
                'industry_rank_pct': r.relative_strength.percentile_rank if r.relative_strength else None,
                'is_disqualified': r.signal.is_disqualified,
                'disqualification_reason': r.signal.disqualification_reason,
                'scan_date': r.scan_date,
            }
            data.append(row)

        df = pd.DataFrame(data)

        if filename:
            df.to_csv(filename, index=False)
            print(f"Results exported to {filename}")

        return df

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def clear_cache(self):
        """Clear all cached data."""
        self._price_cache = {}
        self.industry_classifier.clear_cache()

    def get_actionable_signals(self, results: List[ScanResult],
                               min_score: float = 65) -> List[ScanResult]:
        """
        Filter results to only actionable signals.

        Args:
            results: List of ScanResult objects
            min_score: Minimum score threshold

        Returns:
            Filtered list of actionable signals
        """
        return [r for r in results
                if not r.signal.is_disqualified
                and r.signal.total_score >= min_score]

    def get_watchlist(self, results: List[ScanResult]) -> List[str]:
        """
        Get tickers for watchlist (watch or better).

        Args:
            results: List of ScanResult objects

        Returns:
            List of ticker symbols
        """
        return [r.ticker for r in results
                if r.signal.recommendation in ['strong_buy', 'buy', 'watch']]


# =============================================================================
# QUICK SCAN FUNCTIONS
# =============================================================================

def quick_scan(tickers: List[str], verbose: bool = True) -> List[ScanResult]:
    """
    Quick scan a list of tickers.

    Args:
        tickers: List of ticker symbols
        verbose: Print progress

    Returns:
        List of ScanResult objects
    """
    scanner = EGMScanner()
    results, stats = scanner.scan_universe(tickers, show_progress=verbose)

    if verbose:
        scanner.print_results(results, stats)

    return results


def scan_earnings(days_back: int = 5, verbose: bool = True) -> List[ScanResult]:
    """
    Scan stocks with recent earnings.

    Args:
        days_back: Days to look back
        verbose: Print progress

    Returns:
        List of ScanResult objects
    """
    scanner = EGMScanner()
    results, stats = scanner.scan_earnings_today(days_back, show_progress=verbose)

    if verbose:
        scanner.print_results(results, stats)

    return results


def analyze_single(ticker: str, verbose: bool = True) -> Optional[ScanResult]:
    """
    Analyze a single stock.

    Args:
        ticker: Stock ticker symbol
        verbose: Print detailed report

    Returns:
        ScanResult object
    """
    scanner = EGMScanner()
    result = scanner.analyze_stock(ticker)

    if result and verbose:
        from .signal_generator import format_signal_report
        print(format_signal_report(result.signal))

    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Example usage
    print("EGM/PEAD Scanner - Example Usage")
    print("=" * 60)

    # Test with a few well-known stocks
    test_tickers = ['NVDA', 'AMD', 'AAPL', 'MSFT', 'GOOGL']

    print(f"\nTesting with: {test_tickers}")

    # Initialize scanner
    scanner = EGMScanner()

    # Scan the test tickers
    results, stats = scanner.scan_universe(test_tickers, show_progress=True)

    # Print results
    scanner.print_results(results, stats)

    # Get actionable signals
    actionable = scanner.get_actionable_signals(results)
    if actionable:
        print(f"\nActionable signals: {[r.ticker for r in actionable]}")

    # Get watchlist
    watchlist = scanner.get_watchlist(results)
    if watchlist:
        print(f"Watchlist: {watchlist}")

    print("\n" + "=" * 60)
    print("Scanner test complete.")
