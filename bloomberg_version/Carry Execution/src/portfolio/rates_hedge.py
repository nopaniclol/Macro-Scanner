"""
Rates Hedge Engine

FX risk analysis and SOFR IRS hedging for carry trades.
Computes DV01 exposure from FX forwards and optimizes hedge using 1-3 SOFR tenors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FXPosition:
    """Single FX position with forward exposure."""
    ticker: str
    notional_usd: float           # Position size in USD
    spot_rate: float              # Current spot
    forward_rate: float           # Forward outright
    forward_points: float         # Forward points
    tenor_days: int               # Days to forward delivery
    direction: str                # 'long' or 'short' base currency


@dataclass
class DV01Exposure:
    """DV01 risk exposure for a position."""
    ticker: str
    dv01_usd: float               # DV01 in USD per 1bp
    pv01_usd: float               # PV01 (present value of 1bp)
    bucket_exposures: Dict[str, float]  # DV01 by tenor bucket
    weighted_maturity: float      # Weighted average maturity in years


@dataclass
class HedgeTrade:
    """Single hedge trade recommendation."""
    instrument: str               # e.g., 'SOFR_2Y', 'SFRU25'
    instrument_type: str          # 'swap' or 'future'
    tenor_years: float            # Tenor in years
    notional_usd: float           # Hedge notional
    dv01_hedge: float             # DV01 contribution
    direction: str                # 'pay' or 'receive' fixed (swaps)
    rate_bps: float               # Current rate in bps


@dataclass
class HedgeResult:
    """Complete hedge recommendation."""
    fx_dv01_total: float          # Total FX DV01 to hedge
    hedge_ratio: float            # Target hedge ratio
    target_dv01: float            # Target DV01 to hedge
    trades: List[HedgeTrade]      # List of hedge trades
    residual_dv01: float          # Unhedged DV01
    hedge_efficiency: float       # % of target hedged
    basis_risk_score: float       # 0-1 basis risk metric
    timestamp: datetime


@dataclass
class SOFRCurve:
    """SOFR swap curve with DV01 per tenor."""
    tenors: List[str]             # ['1W', '1M', '3M', '6M', '1Y', '2Y', ...]
    rates_bps: Dict[str, float]   # Rate in bps by tenor
    dv01_per_mm: Dict[str, float] # DV01 per $1M notional by tenor


# =============================================================================
# SOFR CURVE DEFINITIONS
# =============================================================================

# Full SOFR curve tenors from 1W to 30Y
SOFR_TENORS = [
    '1W', '2W', '1M', '2M', '3M', '4M', '5M', '6M',
    '9M', '1Y', '18M', '2Y', '3Y', '4Y', '5Y',
    '7Y', '10Y', '12Y', '15Y', '20Y', '25Y', '30Y'
]

# Approximate DV01 per $1MM notional for each SOFR tenor
# (These are approximate values - production would use actual curve)
SOFR_DV01_PER_MM = {
    '1W': 0.19,
    '2W': 0.38,
    '1M': 0.83,
    '2M': 1.67,
    '3M': 2.50,
    '4M': 3.33,
    '5M': 4.17,
    '6M': 5.00,
    '9M': 7.50,
    '1Y': 10.00,
    '18M': 15.00,
    '2Y': 19.50,
    '3Y': 29.00,
    '4Y': 38.00,
    '5Y': 47.00,
    '7Y': 64.00,
    '10Y': 88.00,
    '12Y': 102.00,
    '15Y': 120.00,
    '20Y': 145.00,
    '25Y': 163.00,
    '30Y': 175.00,
}

# SOFR futures contract specs (front-end hedging)
SOFR_FUTURES = {
    'SR1': {'tenor_months': 1, 'dv01': 41.67},    # 1-month SOFR
    'SR3': {'tenor_months': 3, 'dv01': 25.00},    # 3-month SOFR
}

# Tenor string to years mapping
TENOR_TO_YEARS = {
    '1W': 1/52, '2W': 2/52, '1M': 1/12, '2M': 2/12, '3M': 0.25,
    '4M': 4/12, '5M': 5/12, '6M': 0.5, '9M': 0.75, '1Y': 1.0,
    '18M': 1.5, '2Y': 2.0, '3Y': 3.0, '4Y': 4.0, '5Y': 5.0,
    '7Y': 7.0, '10Y': 10.0, '12Y': 12.0, '15Y': 15.0,
    '20Y': 20.0, '25Y': 25.0, '30Y': 30.0
}


# =============================================================================
# FX RISK ANALYZER
# =============================================================================

class FXRiskAnalyzer:
    """
    Analyzes FX forward positions to compute DV01 exposure.

    FX forwards embed interest rate risk through covered interest parity:
    Forward = Spot * (1 + r_foreign * T) / (1 + r_domestic * T)

    DV01 of an FX forward is approximately:
    DV01 = Notional * Spot * T / 10000

    Where T = time to maturity in years
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FX risk analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.base_currency = self.config.get('base_currency', 'USD')

    def compute_fx_dv01(self,
                        fx_positions: Dict[str, FXPosition],
                        spot_rates: Dict[str, float] = None,
                        forward_curves: Dict[str, pd.Series] = None) -> Dict[str, DV01Exposure]:
        """
        Compute DV01 exposure for FX forward positions.

        The DV01 of an FX forward reflects sensitivity to interest rate
        differentials. When you buy FX forward, you're essentially:
        - Borrowing domestic currency (USD)
        - Lending foreign currency

        Rate sensitivity is approximately:
        DV01 ≈ Notional × (Forward/Spot - 1) × T / 10000

        For simplification, we use:
        DV01 ≈ Notional × T / 10000 (assuming flat 1% rate differential)

        Args:
            fx_positions: Dictionary of ticker -> FXPosition
            spot_rates: Optional current spot rates
            forward_curves: Optional forward curve data

        Returns:
            Dictionary of ticker -> DV01Exposure
        """
        exposures = {}

        for ticker, pos in fx_positions.items():
            try:
                # Time to maturity in years
                T = pos.tenor_days / 365

                # Direction multiplier (long base = receive foreign rate)
                direction_mult = 1.0 if pos.direction == 'long' else -1.0

                # Simple DV01 calculation
                # DV01 = sensitivity to 1bp move in rate differential
                dv01 = pos.notional_usd * T / 10000 * direction_mult

                # PV01 (present value basis) with simple discounting
                discount_factor = 1 / (1 + 0.05 * T)  # Assume 5% discount rate
                pv01 = dv01 * discount_factor

                # Bucket exposures by maturity
                bucket = self._get_maturity_bucket(T)
                bucket_exposures = {bucket: dv01}

                exposures[ticker] = DV01Exposure(
                    ticker=ticker,
                    dv01_usd=dv01,
                    pv01_usd=pv01,
                    bucket_exposures=bucket_exposures,
                    weighted_maturity=T
                )

                logger.debug(f"{ticker}: DV01=${dv01:,.0f}, maturity={T:.2f}Y")

            except Exception as e:
                logger.error(f"Error computing DV01 for {ticker}: {e}")

        return exposures

    def _get_maturity_bucket(self, years: float) -> str:
        """Map maturity to standard bucket."""
        if years <= 0.25:
            return '0-3M'
        elif years <= 0.5:
            return '3-6M'
        elif years <= 1.0:
            return '6-12M'
        elif years <= 2.0:
            return '1-2Y'
        elif years <= 5.0:
            return '2-5Y'
        elif years <= 10.0:
            return '5-10Y'
        else:
            return '10Y+'

    def aggregate_dv01(self, exposures: Dict[str, DV01Exposure]) -> Dict:
        """
        Aggregate DV01 across all positions.

        Args:
            exposures: Dictionary of DV01 exposures

        Returns:
            Aggregated risk metrics
        """
        total_dv01 = sum(exp.dv01_usd for exp in exposures.values())
        total_pv01 = sum(exp.pv01_usd for exp in exposures.values())

        # Aggregate bucket exposures
        bucket_totals = {}
        for exp in exposures.values():
            for bucket, dv01 in exp.bucket_exposures.items():
                bucket_totals[bucket] = bucket_totals.get(bucket, 0) + dv01

        # Weighted average maturity
        total_notional_weighted = sum(
            abs(exp.dv01_usd) * exp.weighted_maturity
            for exp in exposures.values()
        )
        total_abs_dv01 = sum(abs(exp.dv01_usd) for exp in exposures.values())
        weighted_maturity = total_notional_weighted / total_abs_dv01 if total_abs_dv01 > 0 else 0

        return {
            'total_dv01_usd': total_dv01,
            'total_pv01_usd': total_pv01,
            'bucket_exposures': bucket_totals,
            'weighted_maturity_years': weighted_maturity,
            'position_count': len(exposures)
        }


# =============================================================================
# SOFR HEDGE OPTIMIZER
# =============================================================================

class SOFRHedgeOptimizer:
    """
    Optimizes SOFR hedge to minimize basis risk.

    Features:
    - Full SOFR curve flexibility (1W-30Y)
    - Selects 1-3 optimal tenors to minimize basis risk
    - Can use SOFR futures for front-end (<2Y)
    - Supports odd-dated swaps if needed
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize SOFR hedge optimizer.

        Args:
            config: Configuration with hedge parameters
        """
        self.config = config or {}
        self.max_tenors = self.config.get('max_hedge_tenors', 3)
        self.min_notional = self.config.get('min_hedge_notional', 1_000_000)
        self.use_futures_frontend = self.config.get('use_futures_frontend', True)
        self.futures_cutoff_years = self.config.get('futures_cutoff_years', 2.0)

        # Current SOFR curve (would be fetched from Bloomberg in production)
        self.sofr_curve = self._get_sofr_curve()

    def _get_sofr_curve(self) -> SOFRCurve:
        """Get current SOFR curve (mock for now)."""
        # In production, this would fetch from Bloomberg
        # USSOC Curncy (SOFR OIS curve)
        mock_rates = {
            '1W': 532, '2W': 531, '1M': 530, '2M': 528, '3M': 525,
            '4M': 520, '5M': 515, '6M': 510, '9M': 495, '1Y': 480,
            '18M': 455, '2Y': 435, '3Y': 400, '4Y': 380, '5Y': 365,
            '7Y': 350, '10Y': 340, '12Y': 335, '15Y': 332,
            '20Y': 330, '25Y': 328, '30Y': 325
        }

        return SOFRCurve(
            tenors=SOFR_TENORS,
            rates_bps=mock_rates,
            dv01_per_mm=SOFR_DV01_PER_MM
        )

    def select_optimal_hedge_tenors(self,
                                     target_dv01: float,
                                     weighted_maturity: float,
                                     bucket_exposures: Dict[str, float] = None,
                                     max_tenors: int = None) -> List[str]:
        """
        Select optimal SOFR tenors to hedge FX DV01 exposure.

        Strategy:
        1. Match weighted average maturity of FX book
        2. If concentrated in specific buckets, match those
        3. Minimize basis risk by using tenors close to FX maturities
        4. Use futures for front-end if enabled

        Args:
            target_dv01: Total DV01 to hedge
            weighted_maturity: Weighted avg maturity of FX book
            bucket_exposures: DV01 by maturity bucket
            max_tenors: Maximum number of tenors (default: self.max_tenors)

        Returns:
            List of optimal tenor strings (e.g., ['3M', '1Y', '5Y'])
        """
        max_tenors = max_tenors or self.max_tenors

        # Find tenor closest to weighted maturity
        primary_tenor = self._find_closest_tenor(weighted_maturity)
        selected = [primary_tenor]

        # If we have bucket exposures, try to match them
        if bucket_exposures and len(bucket_exposures) > 1:
            # Sort buckets by absolute DV01
            sorted_buckets = sorted(
                bucket_exposures.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            for bucket, dv01 in sorted_buckets[:max_tenors-1]:
                bucket_maturity = self._bucket_to_years(bucket)
                tenor = self._find_closest_tenor(bucket_maturity)
                if tenor not in selected:
                    selected.append(tenor)

        # Fill remaining slots with diversified tenors
        while len(selected) < max_tenors:
            # Add short and long ends for curve exposure
            if '3M' not in selected and weighted_maturity > 0.5:
                selected.append('3M')
            elif '2Y' not in selected and weighted_maturity < 2.0:
                selected.append('2Y')
            elif '5Y' not in selected:
                selected.append('5Y')
            else:
                break

        # Use futures for front-end if enabled
        if self.use_futures_frontend:
            selected = self._substitute_futures(selected)

        return selected[:max_tenors]

    def _find_closest_tenor(self, target_years: float) -> str:
        """Find SOFR tenor closest to target maturity."""
        min_diff = float('inf')
        closest = '1Y'

        for tenor, years in TENOR_TO_YEARS.items():
            diff = abs(years - target_years)
            if diff < min_diff:
                min_diff = diff
                closest = tenor

        return closest

    def _bucket_to_years(self, bucket: str) -> float:
        """Convert bucket string to approximate years."""
        bucket_map = {
            '0-3M': 0.125,
            '3-6M': 0.375,
            '6-12M': 0.75,
            '1-2Y': 1.5,
            '2-5Y': 3.5,
            '5-10Y': 7.5,
            '10Y+': 15.0
        }
        return bucket_map.get(bucket, 1.0)

    def _substitute_futures(self, tenors: List[str]) -> List[str]:
        """Substitute short-dated swaps with futures if beneficial."""
        result = []
        for tenor in tenors:
            years = TENOR_TO_YEARS.get(tenor, 1.0)
            if years <= self.futures_cutoff_years:
                # Could use SOFR futures instead
                # For now, keep as swap (futures logic would check liquidity)
                result.append(tenor)
            else:
                result.append(tenor)
        return result

    def generate_hedge_trades(self,
                               fx_dv01_total: float,
                               hedge_ratio: float,
                               selected_tenors: List[str],
                               weighted_maturity: float = None) -> List[HedgeTrade]:
        """
        Generate specific hedge trades to achieve target DV01 offset.

        Args:
            fx_dv01_total: Total FX DV01 exposure (positive = receive rate)
            hedge_ratio: Target hedge ratio (e.g., 0.6 = 60%)
            selected_tenors: Tenors to use for hedging
            weighted_maturity: Weighted maturity for allocation

        Returns:
            List of HedgeTrade objects
        """
        target_dv01 = fx_dv01_total * hedge_ratio

        if abs(target_dv01) < 100:  # Minimum threshold
            logger.info(f"Target DV01 ${target_dv01:,.0f} below threshold, no hedge needed")
            return []

        # Allocate hedge across tenors
        trades = []

        if len(selected_tenors) == 1:
            # Single tenor hedge
            allocations = {selected_tenors[0]: 1.0}
        elif len(selected_tenors) == 2:
            # Two-tenor hedge (bullet/barbell)
            allocations = {selected_tenors[0]: 0.6, selected_tenors[1]: 0.4}
        else:
            # Multi-tenor hedge (ladder)
            allocations = {}
            weight = 1.0 / len(selected_tenors)
            for tenor in selected_tenors:
                allocations[tenor] = weight

        for tenor, alloc in allocations.items():
            tenor_dv01_target = target_dv01 * alloc
            dv01_per_mm = self.sofr_curve.dv01_per_mm.get(tenor, 10.0)

            # Calculate notional needed
            notional = abs(tenor_dv01_target) / dv01_per_mm * 1_000_000
            notional = max(notional, self.min_notional)  # Floor

            # Round to nearest $1M
            notional = round(notional / 1_000_000) * 1_000_000

            if notional < self.min_notional:
                continue

            # Direction: if FX DV01 is positive (receive rates), we pay fixed to hedge
            direction = 'pay' if fx_dv01_total > 0 else 'receive'

            # Actual DV01 from this trade
            actual_dv01 = notional / 1_000_000 * dv01_per_mm
            if direction == 'pay':
                actual_dv01 = -actual_dv01  # Pay fixed = short duration

            trades.append(HedgeTrade(
                instrument=f'SOFR_{tenor}',
                instrument_type='swap',
                tenor_years=TENOR_TO_YEARS.get(tenor, 1.0),
                notional_usd=notional,
                dv01_hedge=actual_dv01,
                direction=direction,
                rate_bps=self.sofr_curve.rates_bps.get(tenor, 500)
            ))

        return trades

    def calculate_basis_risk(self,
                              fx_bucket_exposures: Dict[str, float],
                              hedge_trades: List[HedgeTrade]) -> float:
        """
        Calculate basis risk score for the hedge.

        Basis risk arises when:
        1. Hedge maturity doesn't match FX forward maturity
        2. SOFR rate moves differently than FX rate differential

        Score 0-1 where 0 = perfect hedge, 1 = maximum basis risk

        Args:
            fx_bucket_exposures: FX DV01 by maturity bucket
            hedge_trades: Proposed hedge trades

        Returns:
            Basis risk score 0-1
        """
        if not hedge_trades:
            return 1.0  # No hedge = max basis risk

        # Calculate weighted maturity of FX exposure
        total_fx_dv01 = sum(abs(v) for v in fx_bucket_exposures.values())
        if total_fx_dv01 == 0:
            return 0.0

        fx_weighted_mat = sum(
            abs(dv01) * self._bucket_to_years(bucket)
            for bucket, dv01 in fx_bucket_exposures.items()
        ) / total_fx_dv01

        # Calculate weighted maturity of hedge
        total_hedge_dv01 = sum(abs(t.dv01_hedge) for t in hedge_trades)
        if total_hedge_dv01 == 0:
            return 1.0

        hedge_weighted_mat = sum(
            abs(t.dv01_hedge) * t.tenor_years
            for t in hedge_trades
        ) / total_hedge_dv01

        # Maturity mismatch component (max 0.5)
        mat_mismatch = abs(fx_weighted_mat - hedge_weighted_mat)
        mat_score = min(mat_mismatch / 5.0, 0.5)  # 5Y mismatch = 0.5 score

        # DV01 coverage component (max 0.5)
        coverage = total_hedge_dv01 / total_fx_dv01 if total_fx_dv01 > 0 else 0
        coverage_score = abs(1.0 - coverage) * 0.5

        return min(mat_score + coverage_score, 1.0)


# =============================================================================
# RATES OVERLAY ENGINE
# =============================================================================

class RatesOverlayEngine:
    """
    Master coordinator for rates overlay on FX/Metals carry trades.

    Integrates:
    - FXRiskAnalyzer: DV01 computation
    - SOFRHedgeOptimizer: Optimal hedge selection
    - Trade generation and risk monitoring
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize rates overlay engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Component initialization
        self.fx_analyzer = FXRiskAnalyzer(config)
        self.hedge_optimizer = SOFRHedgeOptimizer(config)

        # Default parameters
        self.default_hedge_ratio = self.config.get('default_hedge_ratio', 0.6)
        self.hedge_threshold_dv01 = self.config.get('hedge_threshold_dv01', 5000)

    def analyze_and_hedge(self,
                          fx_positions: Dict[str, FXPosition],
                          hedge_ratio: float = None,
                          max_tenors: int = None) -> HedgeResult:
        """
        Complete workflow: analyze FX risk and generate optimal hedge.

        Args:
            fx_positions: Dictionary of FX positions
            hedge_ratio: Target hedge ratio (default: 0.6)
            max_tenors: Maximum hedge tenors (default: 3)

        Returns:
            HedgeResult with trades and risk metrics
        """
        hedge_ratio = hedge_ratio or self.default_hedge_ratio

        # Step 1: Compute FX DV01 exposures
        logger.info("Computing FX DV01 exposures...")
        dv01_exposures = self.fx_analyzer.compute_fx_dv01(fx_positions)

        # Step 2: Aggregate risk
        agg_risk = self.fx_analyzer.aggregate_dv01(dv01_exposures)
        total_dv01 = agg_risk['total_dv01_usd']
        weighted_maturity = agg_risk['weighted_maturity_years']
        bucket_exposures = agg_risk['bucket_exposures']

        logger.info(f"Total FX DV01: ${total_dv01:,.0f}")
        logger.info(f"Weighted maturity: {weighted_maturity:.2f}Y")

        # Step 3: Check if hedge is needed
        if abs(total_dv01) < self.hedge_threshold_dv01:
            logger.info(f"DV01 ${total_dv01:,.0f} below threshold ${self.hedge_threshold_dv01:,.0f}, no hedge")
            return HedgeResult(
                fx_dv01_total=total_dv01,
                hedge_ratio=hedge_ratio,
                target_dv01=total_dv01 * hedge_ratio,
                trades=[],
                residual_dv01=total_dv01,
                hedge_efficiency=0.0,
                basis_risk_score=1.0,
                timestamp=datetime.now()
            )

        # Step 4: Select optimal hedge tenors
        logger.info("Selecting optimal hedge tenors...")
        selected_tenors = self.hedge_optimizer.select_optimal_hedge_tenors(
            target_dv01=total_dv01 * hedge_ratio,
            weighted_maturity=weighted_maturity,
            bucket_exposures=bucket_exposures,
            max_tenors=max_tenors
        )
        logger.info(f"Selected tenors: {selected_tenors}")

        # Step 5: Generate hedge trades
        logger.info("Generating hedge trades...")
        trades = self.hedge_optimizer.generate_hedge_trades(
            fx_dv01_total=total_dv01,
            hedge_ratio=hedge_ratio,
            selected_tenors=selected_tenors,
            weighted_maturity=weighted_maturity
        )

        # Step 6: Calculate hedge metrics
        hedge_dv01_total = sum(t.dv01_hedge for t in trades)
        target_dv01 = total_dv01 * hedge_ratio
        residual_dv01 = total_dv01 + hedge_dv01_total  # Hedge DV01 is opposite sign

        hedge_efficiency = 1.0 - abs(residual_dv01) / abs(total_dv01) if total_dv01 != 0 else 0

        # Step 7: Calculate basis risk
        basis_risk = self.hedge_optimizer.calculate_basis_risk(
            bucket_exposures, trades
        )

        # Log results
        logger.info(f"Hedge DV01: ${hedge_dv01_total:,.0f}")
        logger.info(f"Residual DV01: ${residual_dv01:,.0f}")
        logger.info(f"Hedge efficiency: {hedge_efficiency:.1%}")
        logger.info(f"Basis risk score: {basis_risk:.2f}")

        return HedgeResult(
            fx_dv01_total=total_dv01,
            hedge_ratio=hedge_ratio,
            target_dv01=target_dv01,
            trades=trades,
            residual_dv01=residual_dv01,
            hedge_efficiency=hedge_efficiency,
            basis_risk_score=basis_risk,
            timestamp=datetime.now()
        )

    def get_hedge_summary(self, result: HedgeResult) -> str:
        """
        Generate human-readable hedge summary.

        Args:
            result: HedgeResult from analyze_and_hedge

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 60,
            "RATES HEDGE SUMMARY",
            "=" * 60,
            f"FX DV01 Exposure:     ${result.fx_dv01_total:>12,.0f}",
            f"Hedge Ratio:          {result.hedge_ratio:>12.0%}",
            f"Target DV01:          ${result.target_dv01:>12,.0f}",
            f"Residual DV01:        ${result.residual_dv01:>12,.0f}",
            f"Hedge Efficiency:     {result.hedge_efficiency:>12.1%}",
            f"Basis Risk Score:     {result.basis_risk_score:>12.2f}",
            "",
            "HEDGE TRADES:",
            "-" * 60,
        ]

        if not result.trades:
            lines.append("  No trades required (below threshold)")
        else:
            for i, trade in enumerate(result.trades, 1):
                lines.extend([
                    f"  Trade {i}:",
                    f"    Instrument:  {trade.instrument}",
                    f"    Type:        {trade.instrument_type}",
                    f"    Direction:   {trade.direction} fixed",
                    f"    Notional:    ${trade.notional_usd:,.0f}",
                    f"    Tenor:       {trade.tenor_years:.1f}Y",
                    f"    Rate:        {trade.rate_bps:.0f} bps",
                    f"    DV01:        ${trade.dv01_hedge:,.0f}",
                    ""
                ])

        lines.append("=" * 60)
        return "\n".join(lines)

    def create_fx_position(self,
                           ticker: str,
                           notional_usd: float,
                           direction: str,
                           tenor_days: int = 91,
                           spot_rate: float = None,
                           forward_points: float = None) -> FXPosition:
        """
        Helper to create FXPosition object.

        Args:
            ticker: Currency pair (e.g., 'EURUSD')
            notional_usd: Position size in USD
            direction: 'long' or 'short' base currency
            tenor_days: Days to forward delivery
            spot_rate: Current spot rate (optional)
            forward_points: Forward points in pips (optional)

        Returns:
            FXPosition object
        """
        # Default spot rates if not provided
        default_spots = {
            'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'AUDUSD': 0.6550,
            'USDJPY': 148.50, 'USDCAD': 1.3650, 'USDCHF': 0.8850,
            'USDCNH': 7.2500, 'XAUUSD': 2650.0, 'XAGUSD': 30.50
        }

        # Default forward points (pips)
        default_fwd_pts = {
            'EURUSD': -25, 'GBPUSD': -15, 'AUDUSD': 35,
            'USDJPY': 120, 'USDCAD': 45, 'USDCHF': -30,
            'USDCNH': 150, 'XAUUSD': 15, 'XAGUSD': 0.05
        }

        spot = spot_rate or default_spots.get(ticker, 1.0)
        fwd_pts = forward_points or default_fwd_pts.get(ticker, 0)

        # Calculate forward outright
        if ticker in ['USDJPY', 'USDCAD', 'USDCHF', 'USDCNH']:
            fwd_outright = spot + fwd_pts / 100  # JPY pairs
        else:
            fwd_outright = spot + fwd_pts / 10000  # Standard pairs

        return FXPosition(
            ticker=ticker,
            notional_usd=notional_usd,
            spot_rate=spot,
            forward_rate=fwd_outright,
            forward_points=fwd_pts,
            tenor_days=tenor_days,
            direction=direction
        )
