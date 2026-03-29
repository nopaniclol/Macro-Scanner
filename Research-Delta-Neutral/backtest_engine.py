"""
backtest_engine.py
==================
Shared mean-reversion backtest engine.
No lookahead bias: signals are generated from rolling stats computed on past data only.

Usage:
    from backtest_engine import run_backtest, summarise_results, yearly_breakdown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    """All parameters in one place — no magic numbers."""
    entry_z: float = 2.0          # enter when |z| > this
    exit_z: float = 0.0           # exit when |z| < this (or sign change)
    max_hold: int = 20            # time-stop: max trading days in position
    cost_per_trade: float = 0.5   # one-way cost in spread units (applied at entry + exit)
    lookback: int = 60            # z-score lookback (trading days)
    label: str = ""               # strategy label for reporting


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------
def run_backtest(
    dates: pd.Series,
    spread: pd.Series,
    zscore: pd.Series,
    config: BacktestConfig,
    regime_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Mean-reversion backtest on a spread time series.

    Parameters
    ----------
    dates : daily date series
    spread : spread level (used for PnL calculation)
    zscore : rolling z-score (pre-computed, no lookahead)
    config : BacktestConfig
    regime_mask : optional boolean Series (True = regime ON, new entries allowed).
                  When False, no new entries are opened; existing positions run to
                  their natural exit (z-reversion or time-stop).

    Returns
    -------
    DataFrame with daily position, PnL, and trade metadata.
    """
    n = len(dates)
    position = np.zeros(n, dtype=float)   # +1 long spread / -1 short spread
    daily_pnl = np.zeros(n, dtype=float)
    trade_entry = np.full(n, False)
    trade_exit = np.full(n, False)
    hold_days = np.zeros(n, dtype=int)

    spread_arr = spread.values
    z_arr = zscore.values
    regime_arr = (regime_mask.values if regime_mask is not None
                  else np.ones(n, dtype=bool))

    pos = 0.0
    days_held = 0
    entry_cost_accum = 0.0  # accumulated round-trip cost to apply at exit

    for i in range(1, n):
        # Carry forward position
        if pos != 0:
            days_held += 1
            # Spread PnL: pos * (spread[i] - spread[i-1])
            daily_pnl[i] = pos * (spread_arr[i] - spread_arr[i - 1])

        z = z_arr[i]
        if np.isnan(z):
            position[i] = pos
            hold_days[i] = days_held
            continue

        # --- Exit logic ---
        should_exit = False
        if pos != 0:
            # z-score reversion exit
            if pos > 0 and z >= -config.exit_z:  # long spread → exit when z normalises
                should_exit = True
            elif pos < 0 and z <= config.exit_z:  # short spread → exit when z normalises
                should_exit = True
            # Time-stop
            if days_held >= config.max_hold:
                should_exit = True

        if should_exit:
            # Apply round-trip cost (one-way entry + one-way exit)
            daily_pnl[i] -= config.cost_per_trade  # exit leg
            daily_pnl[i] -= entry_cost_accum        # entry leg (was deferred)
            entry_cost_accum = 0.0
            trade_exit[i] = True
            pos = 0.0
            days_held = 0

        # --- Entry logic (blocked when regime is OFF) ---
        if pos == 0 and bool(regime_arr[i]):
            if z < -config.entry_z:   # spread too cheap → buy it
                pos = 1.0
                entry_cost_accum = config.cost_per_trade
                trade_entry[i] = True
                days_held = 0
            elif z > config.entry_z:  # spread too rich → sell it
                pos = -1.0
                entry_cost_accum = config.cost_per_trade
                trade_entry[i] = True
                days_held = 0

        position[i] = pos
        hold_days[i] = days_held

    # Build result DataFrame
    result = pd.DataFrame({
        "date": dates.values,
        "spread": spread_arr,
        "zscore": z_arr,
        "position": position,
        "daily_pnl": daily_pnl,
        "trade_entry": trade_entry,
        "trade_exit": trade_exit,
        "hold_days": hold_days,
    })
    result["cum_pnl"] = result["daily_pnl"].cumsum()
    return result


# ---------------------------------------------------------------------------
# Trade-level summary
# ---------------------------------------------------------------------------
def extract_trades(result: pd.DataFrame) -> pd.DataFrame:
    """Extract individual trade records from daily result."""
    trades = []
    in_trade = False
    entry_date = None
    entry_pnl_cum = 0.0
    direction = 0

    for _, row in result.iterrows():
        if row["trade_entry"] and not in_trade:
            in_trade = True
            entry_date = row["date"]
            entry_pnl_cum = row["cum_pnl"]
            direction = int(row["position"])

        if row["trade_exit"] and in_trade:
            trades.append({
                "entry_date": entry_date,
                "exit_date": row["date"],
                "hold_days": (row["date"] - entry_date).days,
                "direction": direction,
                "trade_pnl": row["cum_pnl"] - entry_pnl_cum,
            })
            in_trade = False

    df = pd.DataFrame(trades)
    return df


# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------
def summarise_results(result: pd.DataFrame, label: str = "") -> dict:
    """Compute strategy-level performance metrics."""
    pnl = result["daily_pnl"]
    cum_pnl = result["cum_pnl"]

    n_days = len(pnl)
    total_pnl = cum_pnl.iloc[-1]
    ann_return = total_pnl / (n_days / 252)

    vol = pnl.std() * np.sqrt(252)
    sharpe = ann_return / vol if vol > 0 else np.nan

    # Max drawdown
    roll_max = cum_pnl.cummax()
    drawdown = cum_pnl - roll_max
    max_dd = drawdown.min()

    # Trades
    entries = result["trade_entry"].sum()
    exits = result["trade_exit"].sum()
    trades_df = extract_trades(result)
    n_trades = len(trades_df)

    if n_trades > 0:
        wins = trades_df[trades_df["trade_pnl"] > 0]
        losses = trades_df[trades_df["trade_pnl"] <= 0]
        hit_rate = len(wins) / n_trades
        avg_win = wins["trade_pnl"].mean() if len(wins) else 0.0
        avg_loss = losses["trade_pnl"].mean() if len(losses) else 0.0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
        avg_hold = trades_df["hold_days"].mean()

        # Win/loss streaks
        outcomes = (trades_df["trade_pnl"] > 0).astype(int).values
        max_win_streak = _max_streak(outcomes, 1)
        max_loss_streak = _max_streak(outcomes, 0)
    else:
        hit_rate = avg_win = avg_loss = win_loss_ratio = avg_hold = np.nan
        max_win_streak = max_loss_streak = 0

    return {
        "label": label,
        "n_days": n_days,
        "total_pnl": round(total_pnl, 2),
        "ann_return": round(ann_return, 2),
        "ann_vol": round(vol, 3),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 2),
        "n_trades": n_trades,
        "hit_rate": round(hit_rate, 3) if not np.isnan(hit_rate) else np.nan,
        "avg_win": round(avg_win, 3),
        "avg_loss": round(avg_loss, 3),
        "win_loss_ratio": round(win_loss_ratio, 2) if not np.isnan(win_loss_ratio) else np.nan,
        "avg_hold_days": round(avg_hold, 1) if not np.isnan(avg_hold) else np.nan,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }


def _max_streak(arr: np.ndarray, value: int) -> int:
    """Find maximum consecutive run of `value` in array."""
    max_s = cur_s = 0
    for x in arr:
        if x == value:
            cur_s += 1
            max_s = max(max_s, cur_s)
        else:
            cur_s = 0
    return max_s


# ---------------------------------------------------------------------------
# Yearly breakdown
# ---------------------------------------------------------------------------
def yearly_breakdown(result: pd.DataFrame) -> pd.DataFrame:
    """Compute year-by-year performance metrics."""
    result = result.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["year"] = result["date"].dt.year

    rows = []
    for yr, grp in result.groupby("year"):
        pnl = grp["daily_pnl"]
        total = pnl.sum()
        vol = pnl.std() * np.sqrt(252)
        sharpe = (total / (len(pnl) / 252)) / vol if vol > 0 else np.nan
        cum = pnl.cumsum()
        max_dd = (cum - cum.cummax()).min()
        trades = grp["trade_entry"].sum()
        rows.append({
            "year": yr,
            "total_pnl": round(total, 2),
            "ann_vol": round(vol, 3),
            "sharpe": round(sharpe, 2),
            "max_dd": round(max_dd, 2),
            "n_entries": int(trades),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parameter grid scan
# ---------------------------------------------------------------------------
def grid_scan(
    dates: pd.Series,
    spread: pd.Series,
    zscore_map: dict[int, pd.Series],
    entry_zs: list[float],
    exit_zs: list[float],
    max_holds: list[int],
    cost: float = 0.5,
    base_label: str = "",
) -> pd.DataFrame:
    """
    Run backtest over a parameter grid.
    zscore_map: {lookback: z_series}
    Returns DataFrame with one row per parameter combination.
    """
    rows = []
    for lb, z_series in zscore_map.items():
        for entry_z in entry_zs:
            for exit_z in exit_zs:
                if exit_z >= entry_z:
                    continue
                for mh in max_holds:
                    cfg = BacktestConfig(
                        entry_z=entry_z,
                        exit_z=exit_z,
                        max_hold=mh,
                        cost_per_trade=cost,
                        lookback=lb,
                        label=f"{base_label}_lb{lb}_ez{entry_z}_xz{exit_z}_mh{mh}",
                    )
                    res = run_backtest(dates, spread, z_series, cfg)
                    stats = summarise_results(res, cfg.label)
                    stats["lookback"] = lb
                    stats["entry_z"] = entry_z
                    stats["exit_z"] = exit_z
                    stats["max_hold"] = mh
                    rows.append(stats)
    return pd.DataFrame(rows)
