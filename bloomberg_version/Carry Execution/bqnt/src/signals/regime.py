"""
Regime Signal Engine for BQNT Carry Execution.

Detects market regime and adjusts position multipliers.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from ..config.parameters import REGIME_PARAMS, get_vix_regime, get_regime_multiplier


class RegimeSignalEngine:
    """
    Detects market regime and provides position size adjustments.

    Regimes based on VIX levels:
    - Low (<12): Increase exposure 1.20x
    - Normal (12-20): Full exposure 1.00x
    - Elevated (20-25): Reduce to 0.70x
    - High (25-30): Reduce to 0.40x
    - Extreme (>30): Reduce to 0.20x
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize regime signal engine.

        Args:
            params: Optional parameter overrides
        """
        self.params = {**REGIME_PARAMS, **(params or {})}
        self._vix_thresholds = self.params['vix_thresholds']
        self._vix_multipliers = self.params['vix_multipliers']
        self._fxvol_thresholds = self.params['fxvol_thresholds']

    def detect_vix_regime(self, vix_level: float) -> str:
        """
        Determine VIX regime from current level.

        Args:
            vix_level: Current VIX value

        Returns:
            Regime name: 'low', 'normal', 'elevated', 'high', 'extreme'
        """
        return get_vix_regime(vix_level)

    def get_position_multiplier(self, vix_level: float) -> float:
        """
        Get position size multiplier based on VIX regime.

        Args:
            vix_level: Current VIX value

        Returns:
            Multiplier (0.20 to 1.20)
        """
        return get_regime_multiplier(vix_level)

    def calculate_regime_series(
        self,
        vix_data: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate regime and multiplier for time series.

        Args:
            vix_data: Series of VIX values

        Returns:
            Tuple of (regime_series, multiplier_series)
        """
        regimes = vix_data.apply(self.detect_vix_regime)
        multipliers = vix_data.apply(self.get_position_multiplier)

        return regimes, multipliers

    def detect_fxvol_regime(self, fxvol_level: float) -> str:
        """
        Determine FX volatility regime.

        Args:
            fxvol_level: Current FX vol index value

        Returns:
            Regime name
        """
        thresholds = self._fxvol_thresholds

        if fxvol_level < thresholds['low']:
            return 'low'
        elif fxvol_level < thresholds['normal']:
            return 'normal'
        elif fxvol_level < thresholds['elevated']:
            return 'elevated'
        else:
            return 'high'

    def calculate_combined_regime(
        self,
        vix_level: float,
        fxvol_level: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Calculate combined regime from VIX and FX vol.

        Args:
            vix_level: Current VIX value
            fxvol_level: Current FX vol value (optional)

        Returns:
            Dict with regime info
        """
        vix_regime = self.detect_vix_regime(vix_level)
        vix_multiplier = self.get_position_multiplier(vix_level)

        result = {
            'vix_level': vix_level,
            'vix_regime': vix_regime,
            'vix_multiplier': vix_multiplier,
        }

        if fxvol_level is not None:
            fxvol_regime = self.detect_fxvol_regime(fxvol_level)
            result['fxvol_level'] = fxvol_level
            result['fxvol_regime'] = fxvol_regime

            # Average the multipliers if both available
            fxvol_multiplier = self._get_fxvol_multiplier(fxvol_regime)
            result['fxvol_multiplier'] = fxvol_multiplier
            result['combined_multiplier'] = (vix_multiplier + fxvol_multiplier) / 2
        else:
            result['combined_multiplier'] = vix_multiplier

        return result

    def _get_fxvol_multiplier(self, regime: str) -> float:
        """Get multiplier for FX vol regime."""
        multipliers = {
            'low': 1.10,
            'normal': 1.00,
            'elevated': 0.80,
            'high': 0.60,
        }
        return multipliers.get(regime, 1.00)

    def detect_regime_change(
        self,
        vix_data: pd.Series,
        lookback: int = 5
    ) -> pd.DataFrame:
        """
        Detect regime changes over time.

        Args:
            vix_data: Series of VIX values
            lookback: Days to check for change

        Returns:
            DataFrame with regime change signals
        """
        regimes, multipliers = self.calculate_regime_series(vix_data)

        # Detect changes
        regime_changed = regimes != regimes.shift(1)
        multiplier_changed = multipliers != multipliers.shift(1)

        # Direction of change
        multiplier_direction = np.sign(multipliers.diff())

        return pd.DataFrame({
            'regime': regimes,
            'multiplier': multipliers,
            'regime_changed': regime_changed,
            'multiplier_changed': multiplier_changed,
            'direction': multiplier_direction,
        })

    def get_regime_stats(
        self,
        vix_data: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate regime statistics over a period.

        Args:
            vix_data: Series of VIX values

        Returns:
            Dict with regime statistics
        """
        regimes, multipliers = self.calculate_regime_series(vix_data)

        # Time in each regime
        regime_counts = regimes.value_counts(normalize=True)

        # Average multiplier
        avg_multiplier = multipliers.mean()

        # Current vs average
        current_multiplier = multipliers.iloc[-1] if len(multipliers) > 0 else 1.0

        return {
            'current_vix': vix_data.iloc[-1] if len(vix_data) > 0 else None,
            'current_regime': regimes.iloc[-1] if len(regimes) > 0 else None,
            'current_multiplier': current_multiplier,
            'avg_multiplier': avg_multiplier,
            'time_in_regime': regime_counts.to_dict(),
            'vix_percentile': (vix_data < vix_data.iloc[-1]).mean() if len(vix_data) > 0 else None,
        }


class MacroQuadrantDetector:
    """
    Detects macro quadrant from equity and bond returns.

    Quadrants:
    1. Goldilocks: Equities up, Bonds up (disinflation growth)
    2. Reflation: Equities up, Bonds down (inflationary growth)
    3. Stagflation: Equities down, Bonds down (inflationary slowdown)
    4. Risk-Off: Equities down, Bonds up (deflationary slowdown)
    """

    def __init__(self, lookback: int = 21):
        """
        Initialize quadrant detector.

        Args:
            lookback: Days for return calculation
        """
        self.lookback = lookback

    def detect_quadrant(
        self,
        equity_returns: float,
        bond_returns: float
    ) -> int:
        """
        Detect current macro quadrant.

        Args:
            equity_returns: Equity return (e.g., SPX)
            bond_returns: Bond return (inverse of yield change)

        Returns:
            Quadrant number (1-4)
        """
        eq_positive = equity_returns > 0
        bd_positive = bond_returns > 0

        if eq_positive and bd_positive:
            return 1  # Goldilocks
        elif eq_positive and not bd_positive:
            return 2  # Reflation
        elif not eq_positive and not bd_positive:
            return 3  # Stagflation
        else:
            return 4  # Risk-Off

    def get_quadrant_name(self, quadrant: int) -> str:
        """Get quadrant name."""
        names = {
            1: 'Goldilocks',
            2: 'Reflation',
            3: 'Stagflation',
            4: 'Risk-Off',
        }
        return names.get(quadrant, 'Unknown')

    def calculate_quadrant_series(
        self,
        equity_prices: pd.Series,
        bond_prices: pd.Series
    ) -> pd.Series:
        """
        Calculate quadrant over time.

        Args:
            equity_prices: Series of equity prices
            bond_prices: Series of bond prices

        Returns:
            Series of quadrant numbers
        """
        eq_returns = equity_prices.pct_change(self.lookback)
        bd_returns = bond_prices.pct_change(self.lookback)

        quadrants = []
        for eq_ret, bd_ret in zip(eq_returns, bd_returns):
            if pd.isna(eq_ret) or pd.isna(bd_ret):
                quadrants.append(np.nan)
            else:
                quadrants.append(self.detect_quadrant(eq_ret, bd_ret))

        return pd.Series(quadrants, index=equity_prices.index)
