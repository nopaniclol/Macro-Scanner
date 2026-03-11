"""
Signal Generator for EGM/PEAD Scanner

Generates weighted composite signals combining:
- Gap magnitude (15%)
- Volume spike (15%)
- Earnings surprise (20%)
- Volatility profile (12%)
- MA cluster quality (12%)
- Consolidation quality (10%)
- Industry relative strength (16%)

Provides buy/watch/pass recommendations with entry/exit levels.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .gap_detector import GapEvent
from .earnings_validator import EarningsValidation
from .technical_analyzer import TechnicalAnalysis, VolatilityMetrics, MAClusterResult
from .setup_analyzer import SetupAnalysis, ConsolidationPattern, EntrySignal
from .industry_classifier import RelativeStrength


# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNAL_WEIGHTS = {
    'gap_magnitude': 0.15,          # Larger gaps = stronger signal
    'volume_spike': 0.15,           # Higher volume = more conviction
    'earnings_surprise': 0.20,      # Bigger surprise = stronger drift
    'volatility_profile': 0.12,     # ADR% + ATR% meeting thresholds
    'ma_cluster_quality': 0.12,     # Tight 10/20/50 SMA pre-catalyst
    'consolidation_quality': 0.10,  # Tighter consolidation = better
    'industry_relative_strength': 0.16,  # Top performer in industry
}

SCORE_THRESHOLDS = {
    'strong_buy': 80,
    'buy': 65,
    'watch': 50,
    'pass': 0,
}

# Hard requirements - instant disqualification
HARD_REQUIREMENTS = {
    'min_adr_percent': 6.0,
    'min_atr_percent': 7.0,
    'min_gap_percent': 10.0,
    'min_volume_multiple': 3.0,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ComponentScores:
    """Individual component scores."""
    gap_score: float = 0.0
    volume_score: float = 0.0
    earnings_score: float = 0.0
    volatility_score: float = 0.0
    ma_cluster_score: float = 0.0
    consolidation_score: float = 0.0
    relative_strength_score: float = 0.0


@dataclass
class SignalResult:
    """Complete signal generation result."""
    ticker: str
    total_score: float
    component_scores: ComponentScores
    recommendation: str  # 'strong_buy', 'buy', 'watch', 'pass'
    is_disqualified: bool
    disqualification_reason: Optional[str]
    entry_price: Optional[float]
    stop_loss: Optional[float]
    target_price: Optional[float]
    risk_reward: Optional[float]
    position_size_percent: float  # Suggested position size as % of portfolio
    signal_date: datetime
    notes: List[str] = field(default_factory=list)


@dataclass
class ScanResult:
    """Complete scan result for a single stock."""
    ticker: str
    signal: SignalResult
    gap: Optional[GapEvent]
    earnings: Optional[EarningsValidation]
    technicals: Optional[TechnicalAnalysis]
    setup: Optional[SetupAnalysis]
    relative_strength: Optional[RelativeStrength]
    scan_date: datetime


# =============================================================================
# SIGNAL GENERATOR CLASS
# =============================================================================

class SignalGenerator:
    """
    Generates composite EGM/PEAD signals with weighted scoring.

    The signal score combines multiple factors to identify
    high-probability setups following earnings-driven gaps.
    """

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize signal generator.

        Args:
            weights: Custom signal weights (default: SIGNAL_WEIGHTS)
        """
        self.weights = weights or SIGNAL_WEIGHTS.copy()

        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight != 1.0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    # =========================================================================
    # COMPONENT SCORE CALCULATIONS
    # =========================================================================

    def calculate_gap_score(self, gap: GapEvent) -> float:
        """
        Calculate gap magnitude score (0-100).

        Scoring:
        - 10% gap = 50 points (minimum qualifying)
        - 15% gap = 75 points
        - 20%+ gap = 100 points

        Args:
            gap: GapEvent object

        Returns:
            Score from 0-100
        """
        if gap is None:
            return 0.0

        gap_pct = abs(gap.gap_percent)

        if gap_pct < 10:
            return 0.0
        elif gap_pct >= 20:
            return 100.0
        else:
            # Linear scale from 10% (50) to 20% (100)
            return 50 + ((gap_pct - 10) / 10) * 50

    def calculate_volume_score(self, gap: GapEvent) -> float:
        """
        Calculate volume spike score (0-100).

        Scoring:
        - 3x volume = 50 points (minimum qualifying)
        - 5x volume = 75 points
        - 10x+ volume = 100 points

        Args:
            gap: GapEvent object

        Returns:
            Score from 0-100
        """
        if gap is None:
            return 0.0

        vol_mult = gap.volume_multiple

        if vol_mult < 3:
            return 0.0
        elif vol_mult >= 10:
            return 100.0
        else:
            # Linear scale from 3x (50) to 10x (100)
            return 50 + ((vol_mult - 3) / 7) * 50

    def calculate_earnings_score(self, earnings: EarningsValidation) -> float:
        """
        Calculate earnings surprise score (0-100).

        Scoring:
        - 25% surprise = 50 points (minimum qualifying)
        - 50% surprise = 100 points
        - Revenue beat adds bonus points

        Args:
            earnings: EarningsValidation object

        Returns:
            Score from 0-100
        """
        if earnings is None or earnings.eps_surprise_pct is None:
            return 0.0

        eps_surprise = earnings.eps_surprise_pct

        if eps_surprise < 0:
            return 0.0  # Miss = no score

        # EPS score (0-80 points)
        if eps_surprise >= 50:
            eps_score = 80
        elif eps_surprise >= 25:
            # Linear scale from 25% (40) to 50% (80)
            eps_score = 40 + ((eps_surprise - 25) / 25) * 40
        else:
            # Below threshold but still positive
            eps_score = (eps_surprise / 25) * 40

        # Revenue bonus (0-20 points)
        rev_score = 0
        if earnings.revenue_surprise_pct is not None and earnings.revenue_surprise_pct > 0:
            if earnings.revenue_surprise_pct >= 15:
                rev_score = 20
            elif earnings.revenue_surprise_pct >= 5:
                rev_score = 10 + ((earnings.revenue_surprise_pct - 5) / 10) * 10
            else:
                rev_score = (earnings.revenue_surprise_pct / 5) * 10

        return min(100, eps_score + rev_score)

    def calculate_volatility_score(self, technicals: TechnicalAnalysis) -> float:
        """
        Calculate volatility profile score (0-100).

        Requirements:
        - ADR% >= 6%
        - ATR% >= 7%

        Both must be met. Higher values earn more points.

        Args:
            technicals: TechnicalAnalysis object

        Returns:
            Score from 0-100
        """
        if technicals is None:
            return 0.0

        vol = technicals.volatility

        # Hard requirement check
        if not vol.meets_all_criteria:
            return 0.0

        # ADR score (0-50)
        adr = vol.adr_percent
        if adr >= 10:
            adr_score = 50
        else:
            # Linear from 6% (30) to 10% (50)
            adr_score = 30 + ((adr - 6) / 4) * 20

        # ATR score (0-50)
        atr = vol.atr_percent
        if atr >= 12:
            atr_score = 50
        else:
            # Linear from 7% (30) to 12% (50)
            atr_score = 30 + ((atr - 7) / 5) * 20

        return adr_score + atr_score

    def calculate_ma_cluster_score(self, technicals: TechnicalAnalysis) -> float:
        """
        Calculate MA cluster quality score (0-100).

        Scoring:
        - Tight (<2% spread) = 100 points
        - Moderate (2-3% spread) = 70 points
        - Loose (>3% spread) = 30 points

        Args:
            technicals: TechnicalAnalysis object

        Returns:
            Score from 0-100
        """
        if technicals is None:
            return 50.0  # Neutral if no data

        cluster = technicals.ma_cluster

        quality_scores = {
            'tight': 100,
            'moderate': 70,
            'loose': 30,
        }

        base_score = quality_scores.get(cluster.cluster_quality, 30)

        # Bonus for more clustered days
        if cluster.days_clustered >= 3:
            base_score = min(100, base_score + 10)

        return base_score

    def calculate_consolidation_score(self, setup: SetupAnalysis) -> float:
        """
        Calculate consolidation quality score (0-100).

        Based on:
        - Pullback percentage (less = better)
        - Range tightening
        - Volume declining
        - Above key MAs

        Args:
            setup: SetupAnalysis object

        Returns:
            Score from 0-100
        """
        if setup is None:
            return 50.0  # Neutral if no setup data

        cons = setup.consolidation

        # Pattern quality base score
        quality_scores = {
            'excellent': 80,
            'good': 60,
            'fair': 40,
            'poor': 20,
        }
        base_score = quality_scores.get(cons.pattern_quality, 20)

        # Pullback bonus (0-10)
        if cons.pullback_percent <= 20:
            pullback_bonus = 10
        elif cons.pullback_percent <= 30:
            pullback_bonus = 5
        else:
            pullback_bonus = 0

        # Range/volume bonus (0-10)
        extra_bonus = 0
        if cons.range_tightening:
            extra_bonus += 5
        if cons.volume_declining:
            extra_bonus += 5

        return min(100, base_score + pullback_bonus + extra_bonus)

    def calculate_relative_strength_score(self, rel_strength: RelativeStrength) -> float:
        """
        Calculate industry relative strength score (0-100).

        Based on percentile rank within industry.

        Args:
            rel_strength: RelativeStrength object

        Returns:
            Score from 0-100
        """
        if rel_strength is None:
            return 50.0  # Neutral if no comparison available

        # Percentile rank is already 0-100
        return rel_strength.percentile_rank

    # =========================================================================
    # DISQUALIFICATION CHECK
    # =========================================================================

    def check_disqualification(self, gap: GapEvent,
                               technicals: TechnicalAnalysis) -> Tuple[bool, Optional[str]]:
        """
        Check if stock fails hard requirements.

        Hard requirements (instant disqualification):
        - ADR% < 6%
        - ATR% < 7%
        - Gap < 10%
        - Volume < 3x average

        Args:
            gap: GapEvent object
            technicals: TechnicalAnalysis object

        Returns:
            Tuple of (is_disqualified, reason)
        """
        reasons = []

        # Check volatility
        if technicals is not None:
            vol = technicals.volatility
            if vol.adr_percent < HARD_REQUIREMENTS['min_adr_percent']:
                reasons.append(f"ADR% {vol.adr_percent:.1f}% < {HARD_REQUIREMENTS['min_adr_percent']}% minimum")
            if vol.atr_percent < HARD_REQUIREMENTS['min_atr_percent']:
                reasons.append(f"ATR% {vol.atr_percent:.1f}% < {HARD_REQUIREMENTS['min_atr_percent']}% minimum")

        # Check gap
        if gap is not None:
            if abs(gap.gap_percent) < HARD_REQUIREMENTS['min_gap_percent']:
                reasons.append(f"Gap {abs(gap.gap_percent):.1f}% < {HARD_REQUIREMENTS['min_gap_percent']}% minimum")
            if gap.volume_multiple < HARD_REQUIREMENTS['min_volume_multiple']:
                reasons.append(f"Volume {gap.volume_multiple:.1f}x < {HARD_REQUIREMENTS['min_volume_multiple']}x minimum")

        if reasons:
            return True, "; ".join(reasons)

        return False, None

    # =========================================================================
    # ENTRY/EXIT CALCULATIONS
    # =========================================================================

    def calculate_entry_levels(self, setup: SetupAnalysis,
                               technicals: TechnicalAnalysis,
                               gap: GapEvent) -> Dict[str, float]:
        """
        Calculate entry, stop, and target levels.

        Args:
            setup: SetupAnalysis object
            technicals: TechnicalAnalysis object
            gap: GapEvent object

        Returns:
            Dictionary with entry, stop, target, risk_reward
        """
        result = {
            'entry_price': None,
            'stop_loss': None,
            'target_price': None,
            'risk_reward': None,
        }

        # Use setup entry signal if available
        if setup and setup.entry_signal:
            signal = setup.entry_signal
            result['entry_price'] = signal.entry_price
            result['stop_loss'] = signal.stop_loss
            result['target_price'] = signal.target_price
            result['risk_reward'] = signal.risk_reward
            return result

        # Calculate from technicals
        if technicals:
            current_price = technicals.current_price
            sma_10 = technicals.sma_10
            sma_20 = technicals.sma_20

            if current_price > 0:
                result['entry_price'] = current_price

                # Stop below 10-SMA or 20-SMA
                if sma_20 > 0:
                    result['stop_loss'] = round(sma_20 * 0.98, 2)
                elif sma_10 > 0:
                    result['stop_loss'] = round(sma_10 * 0.95, 2)
                else:
                    result['stop_loss'] = round(current_price * 0.90, 2)

                # Target: 15-20% above entry
                result['target_price'] = round(current_price * 1.18, 2)

                # Calculate R:R
                if result['stop_loss']:
                    risk = current_price - result['stop_loss']
                    reward = result['target_price'] - current_price
                    if risk > 0:
                        result['risk_reward'] = round(reward / risk, 2)

        return result

    def calculate_position_size(self, score: float, risk_reward: float = None) -> float:
        """
        Calculate suggested position size as percentage of portfolio.

        Based on signal strength and risk/reward.

        Args:
            score: Total signal score (0-100)
            risk_reward: Risk/reward ratio

        Returns:
            Position size as percentage (e.g., 5.0 = 5%)
        """
        # Base size based on score
        if score >= 80:
            base_size = 5.0
        elif score >= 65:
            base_size = 3.0
        elif score >= 50:
            base_size = 2.0
        else:
            base_size = 0.0

        # Adjust for R:R
        if risk_reward and risk_reward >= 3:
            base_size *= 1.2
        elif risk_reward and risk_reward >= 2:
            base_size *= 1.0
        elif risk_reward and risk_reward < 1.5:
            base_size *= 0.7

        return min(10.0, round(base_size, 1))

    # =========================================================================
    # MAIN SIGNAL GENERATION
    # =========================================================================

    def generate_signal(self, ticker: str,
                        gap: GapEvent = None,
                        earnings: EarningsValidation = None,
                        technicals: TechnicalAnalysis = None,
                        setup: SetupAnalysis = None,
                        rel_strength: RelativeStrength = None) -> SignalResult:
        """
        Generate composite signal for a stock.

        Args:
            ticker: Stock ticker symbol
            gap: GapEvent object
            earnings: EarningsValidation object
            technicals: TechnicalAnalysis object
            setup: SetupAnalysis object
            rel_strength: RelativeStrength object

        Returns:
            SignalResult with score and recommendation
        """
        notes = []

        # Check for disqualification
        is_disqualified, disq_reason = self.check_disqualification(gap, technicals)

        if is_disqualified:
            return SignalResult(
                ticker=ticker,
                total_score=0.0,
                component_scores=ComponentScores(),
                recommendation='pass',
                is_disqualified=True,
                disqualification_reason=disq_reason,
                entry_price=None,
                stop_loss=None,
                target_price=None,
                risk_reward=None,
                position_size_percent=0.0,
                signal_date=datetime.now(),
                notes=[f"DISQUALIFIED: {disq_reason}"],
            )

        # Calculate component scores
        gap_score = self.calculate_gap_score(gap)
        volume_score = self.calculate_volume_score(gap)
        earnings_score = self.calculate_earnings_score(earnings)
        volatility_score = self.calculate_volatility_score(technicals)
        ma_cluster_score = self.calculate_ma_cluster_score(technicals)
        consolidation_score = self.calculate_consolidation_score(setup)
        rs_score = self.calculate_relative_strength_score(rel_strength)

        component_scores = ComponentScores(
            gap_score=round(gap_score, 1),
            volume_score=round(volume_score, 1),
            earnings_score=round(earnings_score, 1),
            volatility_score=round(volatility_score, 1),
            ma_cluster_score=round(ma_cluster_score, 1),
            consolidation_score=round(consolidation_score, 1),
            relative_strength_score=round(rs_score, 1),
        )

        # Calculate weighted total
        total_score = (
            gap_score * self.weights['gap_magnitude'] +
            volume_score * self.weights['volume_spike'] +
            earnings_score * self.weights['earnings_surprise'] +
            volatility_score * self.weights['volatility_profile'] +
            ma_cluster_score * self.weights['ma_cluster_quality'] +
            consolidation_score * self.weights['consolidation_quality'] +
            rs_score * self.weights['industry_relative_strength']
        )

        # Determine recommendation
        if total_score >= SCORE_THRESHOLDS['strong_buy']:
            recommendation = 'strong_buy'
            notes.append("High conviction setup")
        elif total_score >= SCORE_THRESHOLDS['buy']:
            recommendation = 'buy'
            notes.append("Qualifying setup")
        elif total_score >= SCORE_THRESHOLDS['watch']:
            recommendation = 'watch'
            notes.append("Monitor for better entry")
        else:
            recommendation = 'pass'
            notes.append("Does not meet criteria")

        # Calculate entry levels
        levels = self.calculate_entry_levels(setup, technicals, gap)

        # Calculate position size
        position_size = self.calculate_position_size(total_score, levels.get('risk_reward'))

        # Add component notes
        if gap_score >= 75:
            notes.append(f"Strong gap: {gap.gap_percent:.1f}%" if gap else "")
        if earnings_score >= 75:
            notes.append(f"Big earnings beat: {earnings.eps_surprise_pct:.0f}%" if earnings else "")
        if rel_strength and rel_strength.is_leader:
            notes.append(f"Industry leader: {rel_strength.percentile_rank:.0f}th percentile")
        if technicals and technicals.ma_cluster.cluster_quality == 'tight':
            notes.append("Tight MA cluster pre-gap")

        return SignalResult(
            ticker=ticker,
            total_score=round(total_score, 1),
            component_scores=component_scores,
            recommendation=recommendation,
            is_disqualified=False,
            disqualification_reason=None,
            entry_price=levels.get('entry_price'),
            stop_loss=levels.get('stop_loss'),
            target_price=levels.get('target_price'),
            risk_reward=levels.get('risk_reward'),
            position_size_percent=position_size,
            signal_date=datetime.now(),
            notes=[n for n in notes if n],  # Filter empty notes
        )

    def generate_scan_result(self, ticker: str,
                             gap: GapEvent = None,
                             earnings: EarningsValidation = None,
                             technicals: TechnicalAnalysis = None,
                             setup: SetupAnalysis = None,
                             rel_strength: RelativeStrength = None) -> ScanResult:
        """
        Generate complete scan result including signal and all components.

        Args:
            ticker: Stock ticker symbol
            gap: GapEvent object
            earnings: EarningsValidation object
            technicals: TechnicalAnalysis object
            setup: SetupAnalysis object
            rel_strength: RelativeStrength object

        Returns:
            ScanResult with signal and all component data
        """
        signal = self.generate_signal(
            ticker, gap, earnings, technicals, setup, rel_strength
        )

        return ScanResult(
            ticker=ticker,
            signal=signal,
            gap=gap,
            earnings=earnings,
            technicals=technicals,
            setup=setup,
            relative_strength=rel_strength,
            scan_date=datetime.now(),
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_signal_report(result: SignalResult) -> str:
    """
    Format signal result as a text report.

    Args:
        result: SignalResult object

    Returns:
        Formatted string report
    """
    lines = [
        f"{'='*60}",
        f"SIGNAL: {result.ticker}",
        f"{'='*60}",
        f"",
        f"Total Score: {result.total_score}/100",
        f"Recommendation: {result.recommendation.upper()}",
        f"",
    ]

    if result.is_disqualified:
        lines.append(f"DISQUALIFIED: {result.disqualification_reason}")
    else:
        # Component scores
        lines.extend([
            "Component Scores:",
            f"  Gap Magnitude:      {result.component_scores.gap_score:.1f}/100",
            f"  Volume Spike:       {result.component_scores.volume_score:.1f}/100",
            f"  Earnings Surprise:  {result.component_scores.earnings_score:.1f}/100",
            f"  Volatility:         {result.component_scores.volatility_score:.1f}/100",
            f"  MA Cluster:         {result.component_scores.ma_cluster_score:.1f}/100",
            f"  Consolidation:      {result.component_scores.consolidation_score:.1f}/100",
            f"  Relative Strength:  {result.component_scores.relative_strength_score:.1f}/100",
            "",
        ])

        # Entry levels
        if result.entry_price:
            lines.extend([
                "Entry Levels:",
                f"  Entry:  ${result.entry_price:.2f}",
                f"  Stop:   ${result.stop_loss:.2f}",
                f"  Target: ${result.target_price:.2f}",
                f"  R:R:    {result.risk_reward:.1f}:1",
                f"  Size:   {result.position_size_percent:.1f}% of portfolio",
                "",
            ])

        # Notes
        if result.notes:
            lines.append("Notes:")
            for note in result.notes:
                lines.append(f"  - {note}")

    lines.append(f"{'='*60}")
    return "\n".join(lines)


def format_scan_summary(results: List[ScanResult]) -> str:
    """
    Format multiple scan results as a summary report.

    Args:
        results: List of ScanResult objects

    Returns:
        Formatted summary string
    """
    # Sort by score descending
    results = sorted(results, key=lambda x: x.signal.total_score, reverse=True)

    lines = [
        "=" * 80,
        f"EGM/PEAD SCANNER RESULTS - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 80,
        "",
    ]

    # Strong setups
    strong = [r for r in results if r.signal.recommendation == 'strong_buy']
    if strong:
        lines.append("STRONG SETUPS (Score >= 80)")
        lines.append("-" * 40)
        for r in strong:
            lines.append(_format_result_line(r))
        lines.append("")

    # Buy setups
    buy = [r for r in results if r.signal.recommendation == 'buy']
    if buy:
        lines.append("BUY SETUPS (Score 65-79)")
        lines.append("-" * 40)
        for r in buy:
            lines.append(_format_result_line(r))
        lines.append("")

    # Watch list
    watch = [r for r in results if r.signal.recommendation == 'watch']
    if watch:
        lines.append("WATCH LIST (Score 50-64)")
        lines.append("-" * 40)
        for r in watch:
            lines.append(_format_result_line(r))
        lines.append("")

    # Disqualified
    disq = [r for r in results if r.signal.is_disqualified]
    if disq:
        lines.append("DISQUALIFIED")
        lines.append("-" * 40)
        for r in disq:
            lines.append(f"{r.ticker}: {r.signal.disqualification_reason}")
        lines.append("")

    lines.append("=" * 80)
    lines.append(f"Total: {len(results)} stocks scanned")
    lines.append(f"Strong Buy: {len(strong)} | Buy: {len(buy)} | Watch: {len(watch)} | Disqualified: {len(disq)}")
    lines.append("=" * 80)

    return "\n".join(lines)


def _format_result_line(r: ScanResult) -> str:
    """Format a single result as a one-line summary."""
    gap_str = f"Gap: {r.gap.gap_percent:+.1f}%" if r.gap else "Gap: N/A"
    vol_str = f"Vol: {r.gap.volume_multiple:.1f}x" if r.gap else "Vol: N/A"

    eps_str = "EPS: N/A"
    if r.earnings and r.earnings.eps_surprise_pct:
        eps_str = f"EPS: +{r.earnings.eps_surprise_pct:.0f}%"

    adr_str = "ADR: N/A"
    if r.technicals:
        adr_str = f"ADR: {r.technicals.volatility.adr_percent:.1f}%"

    return f"{r.ticker:6} | {gap_str:12} | {vol_str:10} | {eps_str:12} | {adr_str:10} | Score: {r.signal.total_score:.0f}"


def rank_signals(results: List[ScanResult]) -> List[ScanResult]:
    """
    Rank scan results by signal quality.

    Args:
        results: List of ScanResult objects

    Returns:
        Sorted list with best signals first
    """
    # Filter out disqualified
    valid = [r for r in results if not r.signal.is_disqualified]

    # Sort by total score descending
    return sorted(valid, key=lambda x: x.signal.total_score, reverse=True)


def get_actionable_signals(results: List[ScanResult],
                           min_score: float = 65) -> List[ScanResult]:
    """
    Get only actionable signals meeting minimum score.

    Args:
        results: List of ScanResult objects
        min_score: Minimum score threshold

    Returns:
        Filtered list of actionable signals
    """
    return [r for r in results
            if not r.signal.is_disqualified
            and r.signal.total_score >= min_score]


if __name__ == '__main__':
    # Test the signal generator
    print("Testing Signal Generator...")

    # Create mock data for testing
    from datetime import datetime, timedelta

    # Mock GapEvent
    mock_gap = GapEvent(
        ticker='TEST',
        gap_date=datetime.now() - timedelta(days=3),
        gap_percent=12.5,
        gap_direction='up',
        open_price=112.50,
        prev_close=100.00,
        close_price=115.00,
        high_price=118.00,
        low_price=111.00,
        volume=15000000,
        avg_volume=2500000,
        volume_multiple=6.0,
        gap_filled=False,
        gap_fill_percent=0.0,
        meets_criteria=True,
    )

    # Mock EarningsValidation
    mock_earnings = EarningsValidation(
        ticker='TEST',
        has_recent_earnings=True,
        days_since_report=3,
        eps_actual=1.25,
        eps_estimate=0.95,
        eps_surprise=0.30,
        eps_surprise_pct=31.6,
        revenue_actual=5.2e9,
        revenue_estimate=4.8e9,
        revenue_surprise_pct=8.3,
        meets_eps_criteria=True,
        meets_revenue_criteria=True,
        meets_all_criteria=True,
        report_date=datetime.now() - timedelta(days=3),
        validation_date=datetime.now(),
        source='test',
    )

    # Mock VolatilityMetrics
    mock_volatility = VolatilityMetrics(
        ticker='TEST',
        adr_percent=7.5,
        atr_percent=8.2,
        adr_meets_criteria=True,
        atr_meets_criteria=True,
        meets_all_criteria=True,
        adr_lookback=14,
        atr_lookback=14,
        analysis_date=datetime.now(),
    )

    # Mock MAClusterResult
    mock_cluster = MAClusterResult(
        ticker='TEST',
        is_clustered=True,
        spread_percent=1.8,
        cluster_quality='tight',
        sma_10=100.50,
        sma_20=99.80,
        sma_50=98.50,
        days_clustered=4,
        analysis_date=datetime.now(),
        gap_date=datetime.now() - timedelta(days=3),
    )

    # Mock TechnicalAnalysis
    mock_technicals = TechnicalAnalysis(
        ticker='TEST',
        volatility=mock_volatility,
        ma_cluster=mock_cluster,
        current_price=115.00,
        sma_10=112.00,
        sma_20=108.00,
        sma_50=100.00,
        rsi_14=68.5,
        above_10sma=True,
        above_20sma=True,
        above_50sma=True,
        analysis_date=datetime.now(),
    )

    # Mock RelativeStrength
    mock_rs = RelativeStrength(
        ticker='TEST',
        industry='technology_software',
        stock_return=18.5,
        industry_avg_return=5.2,
        outperformance=13.3,
        percentile_rank=92.0,
        rank=2,
        total_peers=15,
        is_leader=True,
        comparison_period_days=20,
        peer_returns={'PEER1': 12.0, 'PEER2': 8.5, 'PEER3': 3.2},
        analysis_date=datetime.now(),
    )

    # Mock ConsolidationPattern
    mock_consolidation = ConsolidationPattern(
        ticker='TEST',
        gap_date=datetime.now() - timedelta(days=3),
        consolidation_days=3,
        is_valid_consolidation=True,
        pullback_percent=15.0,
        within_pullback_limit=True,
        is_above_10sma=True,
        is_above_20sma=True,
        range_tightening=True,
        volume_declining=True,
        pattern_quality='excellent',
        analysis_date=datetime.now(),
    )

    # Mock EntrySignal
    mock_entry = EntrySignal(
        ticker='TEST',
        signal_type='fishhook',
        entry_price=112.00,
        stop_loss=106.00,
        target_price=130.00,
        risk_reward=3.0,
        signal_strength='strong',
        trigger_date=datetime.now(),
        is_valid=True,
        notes='Pullback to 10-SMA',
    )

    # Mock SetupAnalysis
    mock_setup = SetupAnalysis(
        ticker='TEST',
        consolidation=mock_consolidation,
        entry_signal=mock_entry,
        is_actionable=True,
        setup_score=85.0,
        analysis_date=datetime.now(),
    )

    # Test signal generation
    generator = SignalGenerator()

    print("\n1. Testing signal generation...")
    signal = generator.generate_signal(
        'TEST',
        gap=mock_gap,
        earnings=mock_earnings,
        technicals=mock_technicals,
        setup=mock_setup,
        rel_strength=mock_rs,
    )

    print(f"   Total Score: {signal.total_score}")
    print(f"   Recommendation: {signal.recommendation}")
    print(f"   Position Size: {signal.position_size_percent}%")

    print("\n2. Testing component scores...")
    print(f"   Gap:          {signal.component_scores.gap_score}")
    print(f"   Volume:       {signal.component_scores.volume_score}")
    print(f"   Earnings:     {signal.component_scores.earnings_score}")
    print(f"   Volatility:   {signal.component_scores.volatility_score}")
    print(f"   MA Cluster:   {signal.component_scores.ma_cluster_score}")
    print(f"   Consolidation:{signal.component_scores.consolidation_score}")
    print(f"   Rel Strength: {signal.component_scores.relative_strength_score}")

    print("\n3. Testing disqualification...")
    # Create low volatility technicals
    low_vol = VolatilityMetrics(
        ticker='LOWVOL',
        adr_percent=4.0,  # Below 6% threshold
        atr_percent=5.0,  # Below 7% threshold
        adr_meets_criteria=False,
        atr_meets_criteria=False,
        meets_all_criteria=False,
        adr_lookback=14,
        atr_lookback=14,
        analysis_date=datetime.now(),
    )
    low_vol_tech = TechnicalAnalysis(
        ticker='LOWVOL',
        volatility=low_vol,
        ma_cluster=mock_cluster,
        current_price=50.00,
        sma_10=49.00,
        sma_20=48.00,
        sma_50=45.00,
        rsi_14=55.0,
        above_10sma=True,
        above_20sma=True,
        above_50sma=True,
        analysis_date=datetime.now(),
    )

    disq_signal = generator.generate_signal(
        'LOWVOL',
        gap=mock_gap,
        technicals=low_vol_tech,
    )
    print(f"   Disqualified: {disq_signal.is_disqualified}")
    print(f"   Reason: {disq_signal.disqualification_reason}")

    print("\n4. Testing report formatting...")
    report = format_signal_report(signal)
    print(report)

    print("\nSignal Generator test complete.")
