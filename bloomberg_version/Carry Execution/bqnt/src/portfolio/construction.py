"""
Portfolio Construction for BQNT Carry Execution.

Combines signals and applies risk management.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from ..config.parameters import (
    PORTFOLIO_PARAMS,
    SIGNAL_WEIGHTS,
    get_signal_weight,
)


@dataclass
class PortfolioConfig:
    """Configuration for portfolio construction."""
    target_vol: float = PORTFOLIO_PARAMS['target_vol']
    max_position: float = PORTFOLIO_PARAMS['max_position']
    min_position: float = PORTFOLIO_PARAMS['min_position']
    max_gross_exposure: float = PORTFOLIO_PARAMS['max_gross_exposure']
    max_net_exposure: float = PORTFOLIO_PARAMS['max_net_exposure']
    use_risk_parity: bool = PORTFOLIO_PARAMS['use_risk_parity']
    vol_lookback: int = PORTFOLIO_PARAMS['vol_lookback']
    vol_floor: float = PORTFOLIO_PARAMS['vol_floor']
    vol_cap: float = PORTFOLIO_PARAMS['vol_cap']


class PortfolioConstructor:
    """
    Constructs portfolio weights from signals.

    Steps:
    1. Combine weighted signals
    2. Apply volatility targeting
    3. Apply risk parity (optional)
    4. Apply regime multiplier
    5. Enforce position limits
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize portfolio constructor.

        Args:
            config: Portfolio configuration
        """
        self.config = config or PortfolioConfig()

    def combine_signals(
        self,
        momentum_signals: pd.DataFrame,
        carry_signals: pd.DataFrame,
        value_signals: Optional[pd.DataFrame] = None,
        rates_signals: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Combine weighted signals into composite signal.

        Args:
            momentum_signals: Momentum signal DataFrame
            carry_signals: Carry signal DataFrame
            value_signals: Value signal DataFrame (optional)
            rates_signals: Rates signal DataFrame (optional)

        Returns:
            Combined signal DataFrame
        """
        # Get common assets and dates
        assets = set(momentum_signals.columns) & set(carry_signals.columns)
        if value_signals is not None:
            assets &= set(value_signals.columns)
        if rates_signals is not None:
            assets &= set(rates_signals.columns)

        assets = list(assets)

        # Get common dates
        dates = momentum_signals.index.intersection(carry_signals.index)
        if value_signals is not None:
            dates = dates.intersection(value_signals.index)
        if rates_signals is not None:
            dates = dates.intersection(rates_signals.index)

        # Initialize combined signals
        combined = pd.DataFrame(index=dates, columns=assets, dtype=float)
        combined[:] = 0.0

        # Add weighted signals
        weights = SIGNAL_WEIGHTS

        # Momentum
        mom_weight = weights.get('momentum', 0.35)
        for asset in assets:
            if asset in momentum_signals.columns:
                combined[asset] += momentum_signals.loc[dates, asset].fillna(0) * mom_weight

        # Carry
        carry_weight = weights.get('carry', 0.35)
        for asset in assets:
            if asset in carry_signals.columns:
                combined[asset] += carry_signals.loc[dates, asset].fillna(0) * carry_weight

        # Value (if provided)
        if value_signals is not None:
            value_weight = weights.get('value', 0.15)
            for asset in assets:
                if asset in value_signals.columns:
                    combined[asset] += value_signals.loc[dates, asset].fillna(0) * value_weight

        # Rates (if provided)
        if rates_signals is not None:
            rates_weight = weights.get('rates', 0.15)
            for asset in assets:
                if asset in rates_signals.columns:
                    combined[asset] += rates_signals.loc[dates, asset].fillna(0) * rates_weight

        return combined

    def calculate_volatility_weights(
        self,
        returns: pd.DataFrame,
        lookback: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate inverse volatility weights for risk parity.

        Args:
            returns: DataFrame of asset returns
            lookback: Volatility lookback period

        Returns:
            DataFrame of volatility-based weights
        """
        lookback = lookback or self.config.vol_lookback

        # Calculate rolling volatility
        vol = returns.rolling(lookback).std() * np.sqrt(252)

        # Apply floor and cap
        vol = vol.clip(lower=self.config.vol_floor, upper=self.config.vol_cap)

        # Inverse volatility weights
        inv_vol = 1 / vol

        # Normalize to sum to 1 (cross-sectionally)
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)

        return weights

    def apply_volatility_targeting(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        target_vol: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Scale weights to target portfolio volatility.

        Args:
            weights: Raw portfolio weights
            returns: Asset returns
            target_vol: Target annualized volatility

        Returns:
            Scaled weights
        """
        target_vol = target_vol or self.config.target_vol
        lookback = self.config.vol_lookback

        # Calculate portfolio volatility
        # Simplified: use weighted average of individual vols
        asset_vol = returns.rolling(lookback).std() * np.sqrt(252)
        asset_vol = asset_vol.clip(lower=self.config.vol_floor)

        # Portfolio vol approximation (ignoring correlations for simplicity)
        port_vol = (weights.abs() * asset_vol).sum(axis=1)

        # Scale factor
        scale = target_vol / port_vol.replace(0, np.nan)
        scale = scale.clip(lower=0.1, upper=3.0)  # Reasonable bounds

        # Apply scaling
        scaled_weights = weights.mul(scale, axis=0)

        return scaled_weights

    def apply_regime_multiplier(
        self,
        weights: pd.DataFrame,
        regime_multiplier: pd.Series
    ) -> pd.DataFrame:
        """
        Apply regime-based position multiplier.

        Args:
            weights: Portfolio weights
            regime_multiplier: Series of multipliers (0.2 to 1.2)

        Returns:
            Adjusted weights
        """
        # Align regime multiplier to weights
        multiplier = regime_multiplier.reindex(weights.index).fillna(1.0)

        # Apply multiplier
        adjusted = weights.mul(multiplier, axis=0)

        return adjusted

    def enforce_limits(
        self,
        weights: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Enforce position and exposure limits.

        Args:
            weights: Raw weights

        Returns:
            Constrained weights
        """
        # Individual position limits
        constrained = weights.clip(
            lower=self.config.min_position,
            upper=self.config.max_position
        )

        # Gross exposure limit
        gross = constrained.abs().sum(axis=1)
        gross_scale = self.config.max_gross_exposure / gross.replace(0, 1)
        gross_scale = gross_scale.clip(upper=1.0)
        constrained = constrained.mul(gross_scale, axis=0)

        # Net exposure limit
        net = constrained.sum(axis=1).abs()
        net_scale = self.config.max_net_exposure / net.replace(0, 1)
        net_scale = net_scale.clip(upper=1.0)

        # Only scale if net exposure exceeds limit
        for date in constrained.index:
            if net.loc[date] > self.config.max_net_exposure:
                constrained.loc[date] = constrained.loc[date] * net_scale.loc[date]

        return constrained

    def construct_portfolio(
        self,
        momentum_signals: pd.DataFrame,
        carry_signals: pd.DataFrame,
        returns: pd.DataFrame,
        regime_multiplier: Optional[pd.Series] = None,
        value_signals: Optional[pd.DataFrame] = None,
        rates_signals: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Full portfolio construction pipeline.

        Args:
            momentum_signals: Momentum signals
            carry_signals: Carry signals
            returns: Asset returns (for vol calculation)
            regime_multiplier: VIX-based multiplier
            value_signals: Value signals (optional)
            rates_signals: Rates signals (optional)

        Returns:
            Final portfolio weights
        """
        # Step 1: Combine signals
        combined = self.combine_signals(
            momentum_signals,
            carry_signals,
            value_signals,
            rates_signals,
        )

        # Step 2: Apply risk parity weighting (optional)
        if self.config.use_risk_parity:
            rp_weights = self.calculate_volatility_weights(returns)
            # Align to combined signals
            rp_aligned = rp_weights.reindex(
                index=combined.index,
                columns=combined.columns
            ).fillna(1.0 / len(combined.columns))
            combined = combined * rp_aligned

        # Step 3: Apply volatility targeting
        weights = self.apply_volatility_targeting(combined, returns)

        # Step 4: Apply regime multiplier
        if regime_multiplier is not None:
            weights = self.apply_regime_multiplier(weights, regime_multiplier)

        # Step 5: Enforce limits
        weights = self.enforce_limits(weights)

        return weights

    def get_latest_weights(
        self,
        momentum_signals: pd.Series,
        carry_signals: pd.Series,
        volatility: pd.Series,
        regime_multiplier: float = 1.0,
        value_signals: Optional[pd.Series] = None,
        rates_signals: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Calculate latest portfolio weights from current signals.

        Args:
            momentum_signals: Current momentum signals
            carry_signals: Current carry signals
            volatility: Current asset volatilities
            regime_multiplier: Current VIX multiplier
            value_signals: Current value signals (optional)
            rates_signals: Current rates signals (optional)

        Returns:
            Series of portfolio weights
        """
        # Get common assets
        assets = set(momentum_signals.index) & set(carry_signals.index)
        assets = list(assets)

        # Combine signals
        weights = pd.Series(0.0, index=assets)

        mom_weight = get_signal_weight('momentum')
        carry_weight = get_signal_weight('carry')
        value_weight = get_signal_weight('value')
        rates_weight = get_signal_weight('rates')

        for asset in assets:
            signal = 0.0
            signal += momentum_signals.get(asset, 0) * mom_weight
            signal += carry_signals.get(asset, 0) * carry_weight

            if value_signals is not None:
                signal += value_signals.get(asset, 0) * value_weight
            if rates_signals is not None:
                signal += rates_signals.get(asset, 0) * rates_weight

            weights[asset] = signal

        # Vol targeting (simple)
        asset_vols = volatility.reindex(assets).fillna(self.config.vol_floor)
        inv_vol = 1 / asset_vols
        vol_scale = self.config.target_vol / asset_vols.mean()
        weights = weights * inv_vol / inv_vol.sum() * vol_scale

        # Regime multiplier
        weights = weights * regime_multiplier

        # Enforce limits
        weights = weights.clip(
            lower=self.config.min_position,
            upper=self.config.max_position
        )

        return weights
