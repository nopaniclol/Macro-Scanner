# backtest/engine.py
"""
Macro Metals System — Backtest Engine.

Vectorised daily-frequency backtest for a single asset.  Given a price
series and a signal series, produces positions, gross/net returns, and
an equity curve with parameterised transaction costs.

Reference: macro_metals_system_memory.md §6
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "parameters.yaml"


def _load_parameters() -> dict:
    """Load global + backtest parameters from ``parameters.yaml``."""
    with open(_CONFIG_PATH, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Vectorised single-asset backtest runner.

    Args:
        capital:    Starting notional in USD (default 1 000 000).
        tc_bp:      Round-trip transaction cost in basis points, applied
                    per side to absolute notional turnover.
                    ``None`` → defaults to 2.0 bp.
        vol_target: Annualised volatility target for position scaling.
                    ``None`` → reads ``target_portfolio_vol_annual``
                    from ``parameters.yaml``.

    Example::

        engine = BacktestEngine(capital=1_000_000, tc_bp=2.0)
        result = engine.run_single_asset_strategy(prices, signals)
        print(result[["equity"]].tail())
    """

    def __init__(
        self,
        capital: float = 1_000_000.0,
        tc_bp: float | None = None,
        vol_target: float | None = None,
    ) -> None:
        params = _load_parameters()
        gcfg = params.get("global", {})

        self.capital = capital
        self.tc_bp = tc_bp if tc_bp is not None else 2.0
        self.vol_target: float = (
            vol_target
            if vol_target is not None
            else gcfg.get("target_portfolio_vol_annual", 0.10)
        )
        self.vol_lookback: int = gcfg.get("vol_lookback_days", 30)
        self.vol_cap_mult: float = gcfg.get("vol_cap_multiplier", 2.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ewma_vol_scale(self, returns: pd.Series) -> pd.Series:
        """Compute a vol-scaling factor from EWMA realised vol.

        Returns target_vol / realised_vol, capped at ``vol_cap_mult``.
        """
        ann_vol = returns.ewm(halflife=self.vol_lookback).std() * np.sqrt(252)
        ann_vol = ann_vol.replace(0, np.nan)
        scale = (self.vol_target / ann_vol).clip(upper=self.vol_cap_mult)
        return scale.fillna(1.0)

    # ------------------------------------------------------------------
    # Core runner
    # ------------------------------------------------------------------

    def run_single_asset_strategy(
        self,
        prices: pd.Series,
        signals: pd.Series,
        tc_bp: float | None = None,
    ) -> pd.DataFrame:
        """Run a single-asset back-test.

        Args:
            prices:  Daily price series (DatetimeIndex, float).
            signals: Daily signal in ``[-1, +1]`` representing desired
                     position as a fraction of capital.  Applied with a
                     **one-day lag** (signal on *t* → position on *t+1*).
            tc_bp:   Override transaction cost (bp per side) for this run.
                     Falls back to instance default.

        Returns:
            DataFrame with columns:

            - ``position``  — vol-scaled position (fraction of capital).
            - ``price``     — input price.
            - ``ret_gross`` — daily gross return (before costs).
            - ``ret_net``   — daily net return  (after costs).
            - ``equity``    — cumulative equity curve starting at *capital*.
        """
        tc = tc_bp if tc_bp is not None else self.tc_bp

        prices = prices.sort_index().dropna()
        signals = signals.reindex(prices.index).fillna(0.0)

        daily_ret = prices.pct_change().fillna(0.0)

        # Vol-scale and lag the signal by one day
        vol_scale = self._ewma_vol_scale(daily_ret)
        position = (signals.shift(1).fillna(0.0) * vol_scale).clip(
            -self.vol_cap_mult, self.vol_cap_mult
        )

        # Gross return: position × instrument return
        ret_gross = position * daily_ret

        # Transaction cost: tc_bp per side on absolute turnover
        turnover = position.diff().abs().fillna(0.0)
        cost = turnover * (tc / 10_000)

        ret_net = ret_gross - cost
        equity = self.capital * (1 + ret_net).cumprod()

        return pd.DataFrame(
            {
                "position": position,
                "price": prices,
                "ret_gross": ret_gross,
                "ret_net": ret_net,
                "equity": equity,
            },
            index=prices.index,
        )
