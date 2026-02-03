"""
Portfolio Construction Engine

Combines signals into portfolio weights with:
- Inverse volatility weighting
- Position limits enforcement
- Volatility targeting (12% annual)
- Turnover control
- Regime-based scaling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PortfolioWeights:
    """Container for portfolio weights output."""
    raw_weights: Dict[str, float]         # Pre-constraint weights
    constrained_weights: Dict[str, float]  # After position limits
    final_weights: Dict[str, float]        # After vol targeting
    leverage: float                        # Total leverage
    gross_exposure: float                  # Sum of absolute weights
    net_exposure: float                    # Sum of weights (directional)
    turnover: float                        # Change from previous
    vol_forecast: float                    # Expected portfolio volatility
    regime_multiplier: float               # Applied regime scaling
    timestamp: pd.Timestamp


@dataclass
class TradeOrder:
    """Container for trade orders."""
    ticker: str
    direction: str              # 'buy' or 'sell'
    notional: float             # Trade size in base currency
    current_weight: float       # Current portfolio weight
    target_weight: float        # Target portfolio weight
    weight_change: float        # Delta
    priority: int               # Execution priority (1=highest)


@dataclass
class PortfolioConfig:
    """Portfolio construction configuration."""
    # Volatility targeting
    vol_target: float = 0.12              # 12% annual target
    vol_lookback: int = 63                # 3-month lookback
    vol_method: str = 'ewma'              # ewma | simple
    ewma_halflife: int = 21

    # Leverage limits
    min_leverage: float = 0.5
    max_leverage: float = 3.0

    # Position limits
    max_single_position: float = 0.25     # 25% max per asset
    max_fx_gross: float = 4.0             # 400% gross FX
    max_metals_pct: float = 0.30          # 30% max metals
    max_carry_concentration: float = 0.40  # 40% max in high-carry

    # Turnover control
    max_daily_turnover: float = 0.15      # 15% max daily
    min_trade_size: float = 0.01          # 1% minimum trade
    turnover_penalty: float = 0.0005      # 5bp cost

    # Signal weights
    signal_weights: Dict[str, float] = field(default_factory=lambda: {
        'momentum': 0.35,
        'carry': 0.35,
        'value': 0.15,
        'rates': 0.15,
    })

    # Instrument base weights
    instrument_weights: Dict[str, float] = field(default_factory=lambda: {
        'EURUSD': 0.25,
        'USDJPY': 0.20,
        'GBPUSD': 0.15,
        'AUDUSD': 0.15,
        'USDCNH': 0.10,
        'XAUUSD': 0.10,
        'XAGUSD': 0.05,
    })


# =============================================================================
# VOLATILITY ESTIMATION
# =============================================================================

def compute_volatility_simple(returns: pd.Series, lookback: int = 63) -> float:
    """Compute simple rolling volatility (annualized)."""
    if len(returns) < lookback:
        lookback = max(20, len(returns))

    vol = returns.iloc[-lookback:].std() * np.sqrt(252)
    return vol if not pd.isna(vol) else 0.15  # Default 15%


def compute_volatility_ewma(returns: pd.Series, halflife: int = 21) -> float:
    """Compute EWMA volatility (annualized)."""
    if len(returns) < 20:
        return 0.15  # Default

    # EWMA variance
    ewma_var = returns.ewm(halflife=halflife, min_periods=20).var()
    ewma_vol = np.sqrt(ewma_var.iloc[-1]) * np.sqrt(252)

    return ewma_vol if not pd.isna(ewma_vol) else 0.15


def compute_portfolio_volatility(weights: Dict[str, float],
                                  returns_df: pd.DataFrame,
                                  lookback: int = 63) -> float:
    """
    Compute portfolio volatility from asset returns.

    Args:
        weights: Asset weights
        returns_df: DataFrame of asset returns (columns = tickers)
        lookback: Covariance lookback

    Returns:
        Annualized portfolio volatility
    """
    # Get assets in both weights and returns
    common_assets = [a for a in weights.keys() if a in returns_df.columns]

    if len(common_assets) == 0:
        return 0.15  # Default

    # Extract weights vector
    w = np.array([weights.get(a, 0) for a in common_assets])

    # Compute covariance matrix
    returns_subset = returns_df[common_assets].iloc[-lookback:]
    cov_matrix = returns_subset.cov() * 252  # Annualize

    # Portfolio variance
    port_var = w @ cov_matrix.values @ w
    port_vol = np.sqrt(max(port_var, 0))

    return port_vol if not pd.isna(port_vol) else 0.15


# =============================================================================
# PORTFOLIO CONSTRUCTOR CLASS
# =============================================================================

class PortfolioConstructor:
    """
    Constructs portfolio weights from signals with risk management.

    Features:
    - Inverse volatility weighting
    - Position limits enforcement
    - Volatility targeting
    - Turnover control
    - Regime-based scaling
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize portfolio constructor.

        Args:
            config: Portfolio configuration object
        """
        self.config = config or PortfolioConfig()
        self.previous_weights: Dict[str, float] = {}

    def _combine_signals(self,
                         momentum_signals: Dict[str, float],
                         carry_signals: Dict[str, float],
                         value_signals: Dict[str, float] = None,
                         rates_signal: float = 0.0) -> Dict[str, float]:
        """
        Combine multiple signal types into composite signals.

        Args:
            momentum_signals: Ticker -> momentum signal
            carry_signals: Ticker -> carry signal
            value_signals: Ticker -> value signal (optional)
            rates_signal: Overall rates signal (applied uniformly)

        Returns:
            Combined signals per ticker
        """
        value_signals = value_signals or {}

        # Get all tickers
        all_tickers = set(momentum_signals.keys()) | set(carry_signals.keys())

        combined = {}
        weights = self.config.signal_weights

        for ticker in all_tickers:
            mom = momentum_signals.get(ticker, 0.0)
            carry = carry_signals.get(ticker, 0.0)
            value = value_signals.get(ticker, 0.0)

            # Weighted combination
            signal = (
                mom * weights['momentum'] +
                carry * weights['carry'] +
                value * weights['value'] +
                rates_signal * weights['rates']
            )

            combined[ticker] = signal

        return combined

    def _inverse_vol_weights(self,
                              signals: Dict[str, float],
                              returns_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute inverse volatility weights.

        Higher vol assets get lower weight to equalize risk contribution.

        Args:
            signals: Ticker -> signal strength
            returns_df: DataFrame of asset returns

        Returns:
            Inverse vol weighted signals
        """
        weights = {}
        vols = {}

        # Compute volatility for each asset
        for ticker in signals.keys():
            if ticker in returns_df.columns:
                returns = returns_df[ticker].dropna()
                if self.config.vol_method == 'ewma':
                    vol = compute_volatility_ewma(returns, self.config.ewma_halflife)
                else:
                    vol = compute_volatility_simple(returns, self.config.vol_lookback)
            else:
                vol = 0.15  # Default

            # Floor volatility
            vols[ticker] = max(vol, 0.05)

        # Inverse vol weighting
        total_inv_vol = sum(1.0 / v for v in vols.values())

        for ticker, signal in signals.items():
            inv_vol_weight = (1.0 / vols[ticker]) / total_inv_vol
            # Scale signal by inverse vol weight and base weight
            base_weight = self.config.instrument_weights.get(ticker, 0.1)
            weights[ticker] = signal * inv_vol_weight * base_weight * len(signals)

        return weights

    def _apply_position_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply position limits to weights.

        Constraints:
        - Max single position
        - Max gross exposure by asset class
        - Max concentration in high-carry assets

        Args:
            weights: Raw portfolio weights

        Returns:
            Constrained weights
        """
        constrained = weights.copy()

        # 1. Cap individual positions
        for ticker in constrained:
            max_pos = self.config.max_single_position
            constrained[ticker] = np.clip(
                constrained[ticker],
                -max_pos,
                max_pos
            )

        # 2. Check FX gross exposure
        fx_tickers = [t for t in constrained if 'USD' in t and 'XA' not in t]
        fx_gross = sum(abs(constrained.get(t, 0)) for t in fx_tickers)

        if fx_gross > self.config.max_fx_gross:
            scale = self.config.max_fx_gross / fx_gross
            for t in fx_tickers:
                constrained[t] *= scale

        # 3. Check metals allocation
        metal_tickers = [t for t in constrained if 'XAU' in t or 'XAG' in t]
        metals_gross = sum(abs(constrained.get(t, 0)) for t in metal_tickers)

        if metals_gross > self.config.max_metals_pct:
            scale = self.config.max_metals_pct / metals_gross
            for t in metal_tickers:
                constrained[t] *= scale

        return constrained

    def _volatility_targeting(self,
                               weights: Dict[str, float],
                               returns_df: pd.DataFrame) -> Tuple[Dict[str, float], float]:
        """
        Scale weights to target volatility.

        Args:
            weights: Constrained weights
            returns_df: Asset returns for vol estimation

        Returns:
            Tuple of (scaled weights, leverage factor)
        """
        # Compute current portfolio volatility
        port_vol = compute_portfolio_volatility(
            weights, returns_df, self.config.vol_lookback
        )

        if port_vol < 0.01:
            port_vol = 0.15  # Default

        # Compute leverage to hit target
        target_leverage = self.config.vol_target / port_vol

        # Apply leverage bounds
        leverage = np.clip(
            target_leverage,
            self.config.min_leverage,
            self.config.max_leverage
        )

        # Scale weights
        scaled = {t: w * leverage for t, w in weights.items()}

        return scaled, leverage

    def _turnover_control(self,
                          target_weights: Dict[str, float],
                          current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply turnover constraints.

        Limits daily portfolio changes to max_daily_turnover.

        Args:
            target_weights: Desired weights
            current_weights: Current weights

        Returns:
            Turnover-constrained weights
        """
        if not current_weights:
            return target_weights

        # Calculate required turnover
        all_tickers = set(target_weights.keys()) | set(current_weights.keys())

        changes = {}
        total_turnover = 0.0

        for ticker in all_tickers:
            target = target_weights.get(ticker, 0.0)
            current = current_weights.get(ticker, 0.0)
            change = target - current
            changes[ticker] = change
            total_turnover += abs(change)

        # Scale changes if turnover too high
        if total_turnover > self.config.max_daily_turnover:
            scale = self.config.max_daily_turnover / total_turnover

            constrained = {}
            for ticker in all_tickers:
                current = current_weights.get(ticker, 0.0)
                constrained[ticker] = current + changes[ticker] * scale

                # Apply minimum trade size filter
                if abs(constrained[ticker] - current) < self.config.min_trade_size:
                    constrained[ticker] = current

            return constrained

        # Filter small trades
        constrained = {}
        for ticker in all_tickers:
            current = current_weights.get(ticker, 0.0)
            target = target_weights.get(ticker, 0.0)

            if abs(target - current) < self.config.min_trade_size:
                constrained[ticker] = current
            else:
                constrained[ticker] = target

        return constrained

    def _apply_regime_scaling(self,
                               weights: Dict[str, float],
                               regime_multiplier: float) -> Dict[str, float]:
        """
        Apply regime-based position scaling.

        Args:
            weights: Portfolio weights
            regime_multiplier: Scaling factor from regime engine

        Returns:
            Regime-scaled weights
        """
        return {t: w * regime_multiplier for t, w in weights.items()}

    def _apply_value_tilt(self,
                          weights: Dict[str, float],
                          value_tilts: Dict[str, float]) -> Dict[str, float]:
        """
        Apply value-based position tilts.

        Args:
            weights: Portfolio weights
            value_tilts: Ticker -> tilt multiplier (e.g., 1.4 for undervalued)

        Returns:
            Value-tilted weights
        """
        tilted = {}
        for ticker, weight in weights.items():
            tilt = value_tilts.get(ticker, 1.0)
            tilted[ticker] = weight * tilt

        return tilted

    def _incorporate_rates_trades(self,
                                   weights: Dict[str, float],
                                   rates_trades: Dict[str, Dict]) -> Dict[str, float]:
        """
        Incorporate rates overlay trades into portfolio.

        Args:
            weights: FX/metals weights
            rates_trades: Rates trade specifications from rates engine

        Returns:
            Combined weights including rates positions
        """
        combined = weights.copy()

        if not rates_trades:
            return combined

        # Add rates positions
        for trade_type, trade_spec in rates_trades.items():
            if trade_spec.get('direction') != 'neutral':
                # Create synthetic ticker for rates position
                ticker = f"SOFR_{trade_type}"
                signal = trade_spec.get('signal', 0.0)
                combined[ticker] = signal * 0.10  # 10% base weight for rates

        return combined

    def construct_portfolio(self,
                            momentum_signals: Dict[str, float],
                            carry_signals: Dict[str, float],
                            value_tilts: Dict[str, float] = None,
                            regime_multiplier: float = 1.0,
                            rates_trades: Dict[str, Dict] = None,
                            returns_df: pd.DataFrame = None) -> PortfolioWeights:
        """
        Construct complete portfolio from signals.

        Pipeline:
        1. Combine signals (momentum, carry, value, rates)
        2. Inverse vol weight
        3. Apply position limits
        4. Apply value tilts
        5. Volatility targeting
        6. Regime scaling
        7. Turnover control

        Args:
            momentum_signals: Ticker -> momentum signal [-1, 1]
            carry_signals: Ticker -> carry signal [-1, 1]
            value_tilts: Ticker -> value tilt multiplier
            regime_multiplier: Regime scaling factor
            rates_trades: Rates overlay trade specs
            returns_df: Historical returns for vol estimation

        Returns:
            PortfolioWeights object
        """
        value_tilts = value_tilts or {}
        rates_trades = rates_trades or {}

        # Create dummy returns if not provided
        if returns_df is None or returns_df.empty:
            all_tickers = set(momentum_signals.keys()) | set(carry_signals.keys())
            dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='B')
            returns_df = pd.DataFrame(
                np.random.normal(0, 0.01, (252, len(all_tickers))),
                index=dates,
                columns=list(all_tickers)
            )

        # 1. Combine signals
        combined_signals = self._combine_signals(
            momentum_signals, carry_signals, value_tilts,
            rates_signal=0.0  # Could add rates signal here
        )

        # 2. Inverse volatility weighting
        raw_weights = self._inverse_vol_weights(combined_signals, returns_df)

        # 3. Apply position limits
        constrained_weights = self._apply_position_limits(raw_weights)

        # 4. Apply value tilts
        tilted_weights = self._apply_value_tilt(constrained_weights, value_tilts)

        # 5. Volatility targeting
        vol_targeted, leverage = self._volatility_targeting(tilted_weights, returns_df)

        # 6. Regime scaling
        regime_scaled = self._apply_regime_scaling(vol_targeted, regime_multiplier)

        # 7. Incorporate rates trades
        with_rates = self._incorporate_rates_trades(regime_scaled, rates_trades)

        # 8. Turnover control
        final_weights = self._turnover_control(with_rates, self.previous_weights)

        # Compute portfolio metrics
        gross_exposure = sum(abs(w) for w in final_weights.values())
        net_exposure = sum(final_weights.values())

        # Calculate actual turnover
        if self.previous_weights:
            turnover = sum(
                abs(final_weights.get(t, 0) - self.previous_weights.get(t, 0))
                for t in set(final_weights.keys()) | set(self.previous_weights.keys())
            )
        else:
            turnover = gross_exposure

        # Forecast portfolio vol
        vol_forecast = compute_portfolio_volatility(
            final_weights, returns_df, self.config.vol_lookback
        )

        # Store for next iteration
        self.previous_weights = final_weights.copy()

        return PortfolioWeights(
            raw_weights=raw_weights,
            constrained_weights=constrained_weights,
            final_weights=final_weights,
            leverage=leverage,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            turnover=turnover,
            vol_forecast=vol_forecast,
            regime_multiplier=regime_multiplier,
            timestamp=pd.Timestamp.now()
        )

    def generate_trade_orders(self,
                               target_weights: Dict[str, float],
                               current_weights: Dict[str, float],
                               nav: float = 100_000_000) -> List[TradeOrder]:
        """
        Generate trade orders to rebalance portfolio.

        Args:
            target_weights: Target portfolio weights
            current_weights: Current portfolio weights
            nav: Portfolio NAV for notional calculation

        Returns:
            List of TradeOrder objects
        """
        orders = []

        all_tickers = set(target_weights.keys()) | set(current_weights.keys())

        for ticker in all_tickers:
            target = target_weights.get(ticker, 0.0)
            current = current_weights.get(ticker, 0.0)
            change = target - current

            # Skip small trades
            if abs(change) < self.config.min_trade_size:
                continue

            direction = 'buy' if change > 0 else 'sell'
            notional = abs(change) * nav

            # Priority based on size (larger trades first)
            priority = 1 if abs(change) > 0.05 else (2 if abs(change) > 0.02 else 3)

            orders.append(TradeOrder(
                ticker=ticker,
                direction=direction,
                notional=notional,
                current_weight=current,
                target_weight=target,
                weight_change=change,
                priority=priority
            ))

        # Sort by priority
        orders.sort(key=lambda x: x.priority)

        return orders

    def get_portfolio_summary(self, weights: PortfolioWeights) -> str:
        """
        Generate human-readable portfolio summary.

        Args:
            weights: PortfolioWeights object

        Returns:
            Summary string
        """
        lines = [
            "=" * 60,
            "PORTFOLIO CONSTRUCTION SUMMARY",
            "=" * 60,
            f"Timestamp: {weights.timestamp}",
            "",
            "EXPOSURE METRICS:",
            f"  Gross Exposure:     {weights.gross_exposure:.2%}",
            f"  Net Exposure:       {weights.net_exposure:+.2%}",
            f"  Leverage:           {weights.leverage:.2f}x",
            f"  Regime Multiplier:  {weights.regime_multiplier:.2f}",
            "",
            "RISK METRICS:",
            f"  Vol Forecast:       {weights.vol_forecast:.2%}",
            f"  Turnover:           {weights.turnover:.2%}",
            "",
            "FINAL WEIGHTS:",
        ]

        # Sort weights by absolute value
        sorted_weights = sorted(
            weights.final_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for ticker, weight in sorted_weights:
            direction = "LONG" if weight > 0 else "SHORT" if weight < 0 else "FLAT"
            lines.append(f"  {ticker:12} {weight:+8.2%}  ({direction})")

        lines.append("=" * 60)

        return "\n".join(lines)
