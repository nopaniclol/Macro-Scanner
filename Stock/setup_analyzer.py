"""
Setup Analyzer for EGM/PEAD Scanner

Analyzes post-gap consolidation patterns and identifies entry triggers:
- Consolidation pattern detection (2-10 days post-gap)
- Entry methods: Opening Range Breakout (ORB) and Fishhook
- Exit rules based on moving averages
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

SETUP_CRITERIA = {
    'consolidation_days': (2, 10),   # 2-10 days of consolidation
    'max_pullback_pct': 50.0,        # Max 50% retracement of gap
    'above_vwap': True,              # Trading above session VWAP
    'above_10sma': True,             # Above 10-day SMA
    'tight_range': True,             # Tight daily ranges during consolidation
}

ORB_ENTRY = {
    'range_period_minutes': 15,      # First 15 minutes
    'breakout_trigger': 'high_break', # Buy above opening range high
    'stop_loss': 'range_low',        # Stop at opening range low
    'min_range_percent': 0.5,        # Minimum 0.5% opening range
    'max_range_percent': 3.0,        # Maximum 3% opening range
}

FISHHOOK_ENTRY = {
    'pullback_days': (2, 4),         # Wait 2-4 days for pullback
    'entry_level': '10_sma',         # Enter at 10-day SMA touch
    'stop_loss_level': '20_sma',     # Stop below 20-day SMA
    'higher_low_required': True,     # Must form higher low
}

EXIT_RULES = {
    'partial_exit_1': {
        'trigger': 'below_10sma',
        'size': 0.50,                # Sell 50% below 10-SMA
    },
    'full_exit': {
        'trigger': 'below_20sma',
        'size': 1.00,                # Full exit below 20-SMA
    },
    'time_stop': {
        'max_hold_days': 60,         # PEAD effect fades after 60 days
    },
    'profit_target': {
        'target_percent': 20.0,      # Optional: Take profits at 20%
        'trailing_stop': 0.10,       # Or use 10% trailing stop
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ConsolidationPattern:
    """Consolidation pattern analysis results."""
    ticker: str
    gap_date: datetime
    consolidation_days: int
    is_valid_consolidation: bool
    pullback_percent: float
    within_pullback_limit: bool
    is_above_10sma: bool
    is_above_20sma: bool
    range_tightening: bool
    volume_declining: bool
    pattern_quality: str  # 'excellent', 'good', 'fair', 'poor'
    analysis_date: datetime


@dataclass
class EntrySignal:
    """Entry signal details."""
    ticker: str
    signal_type: str  # 'orb', 'fishhook', 'breakout'
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward: float
    signal_strength: str  # 'strong', 'moderate', 'weak'
    trigger_date: datetime
    is_valid: bool
    notes: str


@dataclass
class ExitSignal:
    """Exit signal details."""
    ticker: str
    exit_type: str  # 'partial', 'full', 'stop', 'target', 'time'
    exit_price: float
    exit_percent: float  # 0.5 for 50%, 1.0 for 100%
    trigger_date: datetime
    notes: str


@dataclass
class SetupAnalysis:
    """Complete setup analysis."""
    ticker: str
    consolidation: ConsolidationPattern
    entry_signal: Optional[EntrySignal]
    is_actionable: bool
    setup_score: float
    analysis_date: datetime


# =============================================================================
# SETUP ANALYZER CLASS
# =============================================================================

class SetupAnalyzer:
    """
    Analyzes post-gap setups for EGM/PEAD strategy.

    After a qualifying gap, we look for:
    1. 2-10 days of consolidation
    2. Pullback not exceeding 50% of the gap
    3. Price holding above key moving averages
    4. Declining volume (drying up)
    5. Tight daily ranges
    """

    def __init__(self, max_pullback_pct: float = None,
                 min_consolidation_days: int = None,
                 max_consolidation_days: int = None):
        """
        Initialize setup analyzer.

        Args:
            max_pullback_pct: Maximum allowed pullback percentage
            min_consolidation_days: Minimum consolidation days
            max_consolidation_days: Maximum consolidation days
        """
        self.max_pullback_pct = max_pullback_pct or SETUP_CRITERIA['max_pullback_pct']
        self.min_consolidation_days = min_consolidation_days or SETUP_CRITERIA['consolidation_days'][0]
        self.max_consolidation_days = max_consolidation_days or SETUP_CRITERIA['consolidation_days'][1]

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame."""
        df = df.copy()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['avg_volume_20'] = df['volume'].rolling(window=20).mean()
        df['daily_range'] = df['high'] - df['low']
        df['daily_range_pct'] = (df['high'] - df['low']) / df['low'] * 100
        return df

    def calculate_pullback(self, df: pd.DataFrame, gap_date: datetime,
                           gap_high: float) -> Tuple[float, float]:
        """
        Calculate pullback from gap high.

        Args:
            df: DataFrame with OHLCV data
            gap_date: Date of the gap
            gap_high: High price on gap day

        Returns:
            Tuple of (pullback_points, pullback_percent)
        """
        # Get data after gap
        post_gap = df[df.index > gap_date]

        if len(post_gap) == 0:
            return 0.0, 0.0

        # Find lowest low since gap
        lowest_low = post_gap['low'].min()

        # Calculate pullback
        pullback_points = gap_high - lowest_low
        pullback_pct = (pullback_points / gap_high) * 100

        return pullback_points, pullback_pct

    def check_range_tightening(self, df: pd.DataFrame, gap_date: datetime,
                               lookback: int = 5) -> bool:
        """
        Check if daily ranges are tightening (narrowing).

        Args:
            df: DataFrame with 'daily_range_pct' column
            gap_date: Date of the gap
            lookback: Days to check

        Returns:
            True if ranges are tightening
        """
        post_gap = df[df.index > gap_date]

        if len(post_gap) < lookback:
            return False

        recent = post_gap.tail(lookback)

        if 'daily_range_pct' not in recent.columns:
            return False

        # Compare first half to second half
        mid = len(recent) // 2
        first_half_avg = recent['daily_range_pct'].iloc[:mid].mean()
        second_half_avg = recent['daily_range_pct'].iloc[mid:].mean()

        return second_half_avg < first_half_avg

    def check_volume_declining(self, df: pd.DataFrame, gap_date: datetime,
                               gap_volume: int, lookback: int = 5) -> bool:
        """
        Check if volume is declining (drying up).

        Args:
            df: DataFrame with 'volume' column
            gap_date: Date of the gap
            gap_volume: Volume on gap day
            lookback: Days to check

        Returns:
            True if volume is declining
        """
        post_gap = df[df.index > gap_date]

        if len(post_gap) < lookback:
            return False

        recent = post_gap.tail(lookback)
        recent_avg_volume = recent['volume'].mean()

        # Volume should be less than 50% of gap day volume
        return recent_avg_volume < gap_volume * 0.5

    def analyze_consolidation(self, df: pd.DataFrame, gap_date: datetime,
                              gap_high: float, gap_volume: int,
                              ticker: str = '') -> ConsolidationPattern:
        """
        Analyze consolidation pattern after a gap.

        Args:
            df: DataFrame with OHLCV data
            gap_date: Date of the gap
            gap_high: High price on gap day
            gap_volume: Volume on gap day
            ticker: Stock ticker symbol

        Returns:
            ConsolidationPattern with analysis results
        """
        df = self._add_indicators(df)

        # Get post-gap data
        post_gap = df[df.index > gap_date]
        consolidation_days = len(post_gap)

        # Calculate pullback
        _, pullback_pct = self.calculate_pullback(df, gap_date, gap_high)

        # Check moving averages (latest data)
        if len(post_gap) > 0:
            latest = post_gap.iloc[-1]
            is_above_10sma = latest['close'] > latest.get('sma_10', 0)
            is_above_20sma = latest['close'] > latest.get('sma_20', 0)
        else:
            is_above_10sma = False
            is_above_20sma = False

        # Check range tightening and volume
        range_tightening = self.check_range_tightening(df, gap_date)
        volume_declining = self.check_volume_declining(df, gap_date, gap_volume)

        # Determine if valid consolidation
        is_valid = (
            self.min_consolidation_days <= consolidation_days <= self.max_consolidation_days and
            pullback_pct <= self.max_pullback_pct and
            is_above_10sma
        )

        # Determine pattern quality
        quality_score = 0
        if pullback_pct <= 30:
            quality_score += 2
        elif pullback_pct <= 50:
            quality_score += 1

        if is_above_10sma:
            quality_score += 2
        if is_above_20sma:
            quality_score += 1
        if range_tightening:
            quality_score += 2
        if volume_declining:
            quality_score += 2

        if quality_score >= 8:
            pattern_quality = 'excellent'
        elif quality_score >= 6:
            pattern_quality = 'good'
        elif quality_score >= 4:
            pattern_quality = 'fair'
        else:
            pattern_quality = 'poor'

        return ConsolidationPattern(
            ticker=ticker,
            gap_date=gap_date,
            consolidation_days=consolidation_days,
            is_valid_consolidation=is_valid,
            pullback_percent=round(pullback_pct, 2),
            within_pullback_limit=pullback_pct <= self.max_pullback_pct,
            is_above_10sma=is_above_10sma,
            is_above_20sma=is_above_20sma,
            range_tightening=range_tightening,
            volume_declining=volume_declining,
            pattern_quality=pattern_quality,
            analysis_date=datetime.now(),
        )

    def check_fishhook_entry(self, df: pd.DataFrame, gap_date: datetime,
                             ticker: str = '') -> Optional[EntrySignal]:
        """
        Check for Fishhook entry (pullback to 10-SMA).

        The Fishhook entry is triggered when:
        1. Price pulls back to 10-day SMA
        2. Forms a higher low
        3. Stop loss below 20-day SMA

        Args:
            df: DataFrame with OHLCV data
            gap_date: Date of the gap
            ticker: Stock ticker symbol

        Returns:
            EntrySignal if triggered, None otherwise
        """
        df = self._add_indicators(df)
        post_gap = df[df.index > gap_date]

        if len(post_gap) < FISHHOOK_ENTRY['pullback_days'][0]:
            return None

        # Check if price touched 10-SMA recently
        recent = post_gap.tail(5)

        for i in range(len(recent)):
            row = recent.iloc[i]
            sma_10 = row.get('sma_10', 0)
            sma_20 = row.get('sma_20', 0)

            if sma_10 == 0 or sma_20 == 0:
                continue

            # Check if low touched or dipped below 10-SMA
            if row['low'] <= sma_10 * 1.01:  # Within 1% of 10-SMA
                # Entry signal
                entry_price = sma_10
                stop_loss = sma_20 * 0.99  # Just below 20-SMA
                target_price = entry_price * 1.15  # 15% target

                risk = entry_price - stop_loss
                reward = target_price - entry_price
                risk_reward = reward / risk if risk > 0 else 0

                return EntrySignal(
                    ticker=ticker,
                    signal_type='fishhook',
                    entry_price=round(entry_price, 2),
                    stop_loss=round(stop_loss, 2),
                    target_price=round(target_price, 2),
                    risk_reward=round(risk_reward, 2),
                    signal_strength='strong' if risk_reward >= 2 else 'moderate',
                    trigger_date=recent.index[i].to_pydatetime() if hasattr(recent.index[i], 'to_pydatetime') else datetime.now(),
                    is_valid=True,
                    notes='Pullback to 10-SMA entry',
                )

        return None

    def check_breakout_entry(self, df: pd.DataFrame, gap_date: datetime,
                             consolidation_high: float,
                             ticker: str = '') -> Optional[EntrySignal]:
        """
        Check for breakout from consolidation range.

        Args:
            df: DataFrame with OHLCV data
            gap_date: Date of the gap
            consolidation_high: High of consolidation range
            ticker: Stock ticker symbol

        Returns:
            EntrySignal if triggered, None otherwise
        """
        df = self._add_indicators(df)
        latest = df.iloc[-1]

        # Check if price broke above consolidation high
        if latest['high'] > consolidation_high:
            sma_10 = latest.get('sma_10', consolidation_high * 0.95)
            sma_20 = latest.get('sma_20', consolidation_high * 0.90)

            entry_price = consolidation_high
            stop_loss = max(sma_10 * 0.98, sma_20)
            target_price = entry_price * 1.20

            risk = entry_price - stop_loss
            reward = target_price - entry_price
            risk_reward = reward / risk if risk > 0 else 0

            return EntrySignal(
                ticker=ticker,
                signal_type='breakout',
                entry_price=round(entry_price, 2),
                stop_loss=round(stop_loss, 2),
                target_price=round(target_price, 2),
                risk_reward=round(risk_reward, 2),
                signal_strength='strong' if risk_reward >= 2 else 'moderate',
                trigger_date=df.index[-1].to_pydatetime() if hasattr(df.index[-1], 'to_pydatetime') else datetime.now(),
                is_valid=True,
                notes='Breakout above consolidation range',
            )

        return None

    def check_exit_signals(self, df: pd.DataFrame, entry_price: float,
                           entry_date: datetime,
                           ticker: str = '') -> List[ExitSignal]:
        """
        Check for exit signals based on exit rules.

        Args:
            df: DataFrame with OHLCV data
            entry_price: Entry price
            entry_date: Entry date
            ticker: Stock ticker symbol

        Returns:
            List of ExitSignal objects
        """
        df = self._add_indicators(df)
        latest = df.iloc[-1]
        signals = []

        sma_10 = latest.get('sma_10', 0)
        sma_20 = latest.get('sma_20', 0)
        current_price = latest['close']

        # Check partial exit (below 10-SMA)
        if sma_10 > 0 and current_price < sma_10:
            signals.append(ExitSignal(
                ticker=ticker,
                exit_type='partial',
                exit_price=round(current_price, 2),
                exit_percent=0.50,
                trigger_date=datetime.now(),
                notes='Price below 10-SMA - sell 50%',
            ))

        # Check full exit (below 20-SMA)
        if sma_20 > 0 and current_price < sma_20:
            signals.append(ExitSignal(
                ticker=ticker,
                exit_type='full',
                exit_price=round(current_price, 2),
                exit_percent=1.00,
                trigger_date=datetime.now(),
                notes='Price below 20-SMA - full exit',
            ))

        # Check profit target
        profit_target = entry_price * (1 + EXIT_RULES['profit_target']['target_percent'] / 100)
        if current_price >= profit_target:
            signals.append(ExitSignal(
                ticker=ticker,
                exit_type='target',
                exit_price=round(current_price, 2),
                exit_percent=1.00,
                trigger_date=datetime.now(),
                notes=f'Hit {EXIT_RULES["profit_target"]["target_percent"]}% profit target',
            ))

        # Check time stop
        if entry_date:
            days_held = (datetime.now() - entry_date).days
            if days_held >= EXIT_RULES['time_stop']['max_hold_days']:
                signals.append(ExitSignal(
                    ticker=ticker,
                    exit_type='time',
                    exit_price=round(current_price, 2),
                    exit_percent=1.00,
                    trigger_date=datetime.now(),
                    notes=f'Time stop - {days_held} days held',
                ))

        return signals

    def analyze(self, df: pd.DataFrame, gap_date: datetime,
                gap_high: float, gap_volume: int,
                ticker: str = '') -> SetupAnalysis:
        """
        Perform complete setup analysis.

        Args:
            df: DataFrame with OHLCV data
            gap_date: Date of the gap
            gap_high: High price on gap day
            gap_volume: Volume on gap day
            ticker: Stock ticker symbol

        Returns:
            SetupAnalysis with all metrics
        """
        # Analyze consolidation
        consolidation = self.analyze_consolidation(
            df, gap_date, gap_high, gap_volume, ticker
        )

        # Check for entry signals
        entry_signal = None

        # First check fishhook
        if consolidation.is_valid_consolidation:
            entry_signal = self.check_fishhook_entry(df, gap_date, ticker)

            # If no fishhook, check breakout
            if entry_signal is None:
                post_gap = df[df.index > gap_date]
                if len(post_gap) > 0:
                    consolidation_high = post_gap['high'].max()
                    entry_signal = self.check_breakout_entry(
                        df, gap_date, consolidation_high, ticker
                    )

        # Calculate setup score
        setup_score = self._calculate_setup_score(consolidation, entry_signal)

        # Determine if actionable
        is_actionable = (
            consolidation.is_valid_consolidation and
            entry_signal is not None and
            setup_score >= 60
        )

        return SetupAnalysis(
            ticker=ticker,
            consolidation=consolidation,
            entry_signal=entry_signal,
            is_actionable=is_actionable,
            setup_score=setup_score,
            analysis_date=datetime.now(),
        )

    def _calculate_setup_score(self, consolidation: ConsolidationPattern,
                               entry_signal: Optional[EntrySignal]) -> float:
        """
        Calculate overall setup score (0-100).

        Args:
            consolidation: ConsolidationPattern object
            entry_signal: EntrySignal object (optional)

        Returns:
            Score from 0-100
        """
        score = 0.0

        # Consolidation quality (0-50 points)
        quality_scores = {'excellent': 50, 'good': 40, 'fair': 25, 'poor': 10}
        score += quality_scores.get(consolidation.pattern_quality, 0)

        # Pullback score (0-20 points)
        if consolidation.within_pullback_limit:
            pullback_score = 20 * (1 - consolidation.pullback_percent / 50)
            score += max(0, pullback_score)

        # Entry signal score (0-30 points)
        if entry_signal:
            if entry_signal.signal_strength == 'strong':
                score += 30
            elif entry_signal.signal_strength == 'moderate':
                score += 20
            else:
                score += 10

        return min(100, score)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_risk_reward(entry: float, stop: float, target: float) -> float:
    """
    Calculate risk/reward ratio.

    Args:
        entry: Entry price
        stop: Stop loss price
        target: Target price

    Returns:
        Risk/reward ratio
    """
    risk = abs(entry - stop)
    reward = abs(target - entry)

    if risk == 0:
        return 0.0

    return reward / risk


def get_entry_recommendation(setup: SetupAnalysis) -> str:
    """
    Get entry recommendation based on setup analysis.

    Args:
        setup: SetupAnalysis object

    Returns:
        Recommendation string
    """
    if not setup.is_actionable:
        return 'WAIT'

    if setup.setup_score >= 80:
        return 'STRONG BUY'
    elif setup.setup_score >= 60:
        return 'BUY'
    else:
        return 'WATCH'


if __name__ == '__main__':
    # Test the setup analyzer
    print("Testing Setup Analyzer...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')

    base_price = 100
    prices = [base_price]

    # Create a gap on day 10
    for i in range(29):
        if i == 9:  # Gap day
            prices.append(prices[-1] * 1.12)  # 12% gap
        else:
            # Consolidation after gap
            if i > 9:
                prices.append(prices[-1] * (1 + np.random.uniform(-0.01, 0.01)))
            else:
                prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))

    prices = np.array(prices)

    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, 30)),
        'high': prices * (1 + np.random.uniform(0.005, 0.02, 30)),
        'low': prices * (1 + np.random.uniform(-0.02, -0.005, 30)),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 30),
    }, index=dates)

    # Gap day has high volume
    df.iloc[10, df.columns.get_loc('volume')] = 15000000

    gap_date = dates[10]
    gap_high = df.iloc[10]['high']
    gap_volume = df.iloc[10]['volume']

    # Test analyzer
    analyzer = SetupAnalyzer()

    print("\n1. Testing consolidation analysis...")
    consolidation = analyzer.analyze_consolidation(
        df, gap_date, gap_high, gap_volume, 'TEST'
    )
    print(f"   Days: {consolidation.consolidation_days}")
    print(f"   Pullback: {consolidation.pullback_percent}%")
    print(f"   Above 10-SMA: {consolidation.is_above_10sma}")
    print(f"   Range Tightening: {consolidation.range_tightening}")
    print(f"   Volume Declining: {consolidation.volume_declining}")
    print(f"   Quality: {consolidation.pattern_quality}")
    print(f"   Valid: {consolidation.is_valid_consolidation}")

    print("\n2. Testing complete analysis...")
    setup = analyzer.analyze(df, gap_date, gap_high, gap_volume, 'TEST')
    print(f"   Setup Score: {setup.setup_score:.1f}")
    print(f"   Actionable: {setup.is_actionable}")
    if setup.entry_signal:
        print(f"   Entry Type: {setup.entry_signal.signal_type}")
        print(f"   Entry: ${setup.entry_signal.entry_price}")
        print(f"   Stop: ${setup.entry_signal.stop_loss}")
        print(f"   Target: ${setup.entry_signal.target_price}")
        print(f"   R:R: {setup.entry_signal.risk_reward}")

    print("\n3. Testing entry recommendation...")
    rec = get_entry_recommendation(setup)
    print(f"   Recommendation: {rec}")

    print("\nSetup Analyzer test complete.")
