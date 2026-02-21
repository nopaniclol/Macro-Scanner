"""
Backtesting Engine

Full-featured backtester for FX/Metals carry strategies with:
- Transaction cost modeling
- Performance metrics computation
- Crisis period analysis
- Return attribution
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
class TransactionCosts:
    """Transaction cost configuration."""
    # FX costs in basis points
    fx_majors: float = 0.5        # EURUSD, USDJPY, GBPUSD, USDCHF
    fx_crosses: float = 1.0       # EURJPY, GBPJPY, etc.
    fx_em: float = 3.0            # USDCNH, USDBRL, etc.

    # Metals costs in basis points
    gold: float = 3.0             # XAUUSD
    silver: float = 10.0          # XAGUSD

    # Rates costs in basis points
    sofr_swaps: float = 0.3       # SOFR IRS
    futures: float = 0.1          # SOFR futures

    # Market impact coefficient (k * sqrt(size))
    impact_coefficient: float = 0.0001


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 100_000_000
    rebalance_frequency: str = 'daily'  # daily, weekly, monthly
    include_tc: bool = True
    include_impact: bool = True
    slippage_bps: float = 0.2
    funding_rate: float = 0.05    # Annual funding cost for leverage

    # Risk limits
    max_drawdown_stop: float = 0.15   # Stop trading at 15% drawdown
    vol_scaling: bool = True
    target_vol: float = 0.12

    # Transaction costs
    tc_config: TransactionCosts = field(default_factory=TransactionCosts)


@dataclass
class PerformanceMetrics:
    """Container for backtest performance metrics."""
    # Returns
    total_return: float
    annualized_return: float
    annualized_vol: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int  # Days
    avg_drawdown: float

    # Distribution
    skewness: float
    kurtosis: float
    var_95: float
    var_99: float
    cvar_95: float

    # Trade stats
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    num_trades: int

    # Turnover
    avg_daily_turnover: float
    total_tc_paid: float
    tc_drag: float  # Annual TC as % of return

    # Time
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    num_days: int


@dataclass
class CrisisAnalysis:
    """Performance during crisis periods."""
    period_name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    return_pct: float
    max_drawdown: float
    sharpe_ratio: float
    spx_return: float  # Benchmark comparison
    outperformance: float


@dataclass
class Attribution:
    """Return attribution by source."""
    momentum_return: float
    carry_return: float
    value_return: float
    rates_return: float
    tc_cost: float
    funding_cost: float
    total_return: float

    # By instrument
    instrument_returns: Dict[str, float]

    # By time period
    monthly_returns: pd.Series
    yearly_returns: pd.Series


@dataclass
class BacktestResult:
    """Complete backtest output."""
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    pnl_daily: pd.Series
    drawdown: pd.Series
    turnover: pd.Series

    metrics: PerformanceMetrics
    crisis_analysis: List[CrisisAnalysis]
    attribution: Attribution

    config: BacktestConfig
    timestamp: datetime


# =============================================================================
# TRANSACTION COST MODEL
# =============================================================================

class TransactionCostModel:
    """
    Transaction cost model with spread + market impact.

    TC = spread_bps + k * sqrt(trade_size / ADV)
    """

    # Instrument classification
    FX_MAJORS = ['EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
    FX_CROSSES = ['EURJPY', 'GBPJPY', 'EURGBP', 'AUDJPY', 'CADJPY', 'CHFJPY']
    FX_EM = ['USDCNH', 'USDBRL', 'USDMXN', 'USDINR', 'USDKRW', 'USDTRY', 'USDZAR']
    METALS = ['XAUUSD', 'XAGUSD']
    RATES = ['SOFR', 'SOFR_']

    def __init__(self, config: TransactionCosts = None):
        """Initialize TC model."""
        self.config = config or TransactionCosts()

    def get_spread_bps(self, instrument: str) -> float:
        """Get spread cost in basis points for instrument."""
        instrument_upper = instrument.upper()

        if instrument_upper in self.FX_MAJORS:
            return self.config.fx_majors
        elif instrument_upper in self.FX_CROSSES:
            return self.config.fx_crosses
        elif instrument_upper in self.FX_EM:
            return self.config.fx_em
        elif instrument_upper == 'XAUUSD':
            return self.config.gold
        elif instrument_upper == 'XAGUSD':
            return self.config.silver
        elif instrument_upper.startswith('SOFR'):
            return self.config.sofr_swaps
        else:
            # Default to FX crosses
            return self.config.fx_crosses

    def compute_market_impact(self,
                               trade_size: float,
                               adv: float = 1e9) -> float:
        """
        Compute market impact cost.

        Impact = k * sqrt(trade_size / ADV)

        Args:
            trade_size: Absolute trade size in USD
            adv: Average daily volume (default $1B for FX)

        Returns:
            Impact cost in basis points
        """
        if adv <= 0 or trade_size <= 0:
            return 0.0

        participation = trade_size / adv
        impact = self.config.impact_coefficient * np.sqrt(participation) * 10000

        return impact

    def compute_total_cost(self,
                           instrument: str,
                           trade_size: float,
                           adv: float = None) -> float:
        """
        Compute total transaction cost.

        Args:
            instrument: Instrument ticker
            trade_size: Absolute trade size in USD
            adv: Average daily volume

        Returns:
            Total cost in basis points
        """
        spread = self.get_spread_bps(instrument)

        if adv:
            impact = self.compute_market_impact(trade_size, adv)
        else:
            impact = 0.0

        return spread + impact


# =============================================================================
# CRISIS PERIOD DEFINITIONS
# =============================================================================

CRISIS_PERIODS = {
    'GFC_2008': {
        'start': '2008-09-01',
        'end': '2009-03-31',
        'description': 'Global Financial Crisis'
    },
    'EURO_2011': {
        'start': '2011-07-01',
        'end': '2011-12-31',
        'description': 'European Debt Crisis'
    },
    'CNH_2015': {
        'start': '2015-08-01',
        'end': '2016-02-29',
        'description': 'China Devaluation / Global Selloff'
    },
    'COVID_2020': {
        'start': '2020-02-15',
        'end': '2020-04-30',
        'description': 'COVID-19 Pandemic'
    },
    'RATES_2022': {
        'start': '2022-01-01',
        'end': '2022-12-31',
        'description': 'Fed Hiking Cycle / Russia-Ukraine'
    },
    'SVB_2023': {
        'start': '2023-03-01',
        'end': '2023-05-31',
        'description': 'Regional Banking Crisis'
    }
}


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Backtesting engine for FX/Metals carry strategies.

    Supports:
    - Daily/weekly/monthly rebalancing
    - Transaction cost modeling (spread + impact)
    - Performance metrics computation
    - Crisis period analysis
    - Return attribution
    """

    def __init__(self, config: BacktestConfig = None):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.tc_model = TransactionCostModel(self.config.tc_config)

    def run_backtest(self,
                     positions_df: pd.DataFrame,
                     returns_df: pd.DataFrame,
                     rates_trades: pd.DataFrame = None,
                     prices_df: pd.DataFrame = None) -> BacktestResult:
        """
        Run full backtest.

        Args:
            positions_df: DataFrame of positions (date x instrument)
            returns_df: DataFrame of instrument returns (date x instrument)
            rates_trades: Optional rates overlay trades
            prices_df: Optional price levels for analysis

        Returns:
            BacktestResult with full analysis
        """
        logger.info("Starting backtest...")

        # Align dates
        common_dates = positions_df.index.intersection(returns_df.index)
        positions_df = positions_df.loc[common_dates]
        returns_df = returns_df.loc[common_dates]

        # Initialize equity curve
        equity = pd.Series(index=common_dates, dtype=float)
        equity.iloc[0] = self.config.initial_capital

        pnl_daily = pd.Series(index=common_dates, dtype=float)
        turnover = pd.Series(index=common_dates, dtype=float)
        tc_paid = pd.Series(index=common_dates, dtype=float)

        prev_positions = positions_df.iloc[0] * 0  # Start flat

        # Main backtest loop
        for i, date in enumerate(common_dates):
            current_positions = positions_df.loc[date]
            current_returns = returns_df.loc[date]

            # Compute PnL from positions
            position_pnl = (prev_positions * current_returns).sum()

            # Compute turnover and transaction costs
            position_change = (current_positions - prev_positions).abs()
            daily_turnover = position_change.sum()
            turnover.loc[date] = daily_turnover

            if self.config.include_tc:
                tc = self._compute_transaction_costs(position_change, current_positions.index)
                tc_paid.loc[date] = tc * equity.iloc[max(0, i-1)] if i > 0 else 0
            else:
                tc = 0
                tc_paid.loc[date] = 0

            # Net PnL
            if i > 0:
                gross_pnl = position_pnl * equity.iloc[i-1]
                net_pnl = gross_pnl - tc_paid.loc[date]
                pnl_daily.loc[date] = net_pnl
                equity.loc[date] = equity.iloc[i-1] + net_pnl
            else:
                pnl_daily.loc[date] = 0
                equity.loc[date] = self.config.initial_capital

            prev_positions = current_positions

        # Compute returns from equity
        returns = equity.pct_change(fill_method=None).fillna(0)

        # Compute drawdown
        drawdown = self._compute_drawdown(equity)

        # Compute performance metrics
        metrics = self._compute_performance_metrics(
            pnl=pnl_daily,
            equity=equity,
            drawdown=drawdown,
            returns=returns,
            turnover=turnover,
            tc_paid=tc_paid
        )

        # Crisis analysis
        crisis_analysis = self._analyze_crisis_periods(
            equity=equity,
            returns=returns
        )

        # Attribution
        attribution = self._compute_attribution(
            positions=positions_df,
            returns=returns_df,
            tc=tc_paid,
            equity=equity
        )

        logger.info(f"Backtest complete. Sharpe: {metrics.sharpe_ratio:.2f}")

        return BacktestResult(
            equity_curve=equity,
            returns=returns,
            positions=positions_df,
            pnl_daily=pnl_daily,
            drawdown=drawdown,
            turnover=turnover,
            metrics=metrics,
            crisis_analysis=crisis_analysis,
            attribution=attribution,
            config=self.config,
            timestamp=datetime.now()
        )

    def _compute_transaction_costs(self,
                                    turnover: pd.Series,
                                    instruments: pd.Index) -> float:
        """
        Compute transaction costs for turnover.

        Args:
            turnover: Absolute position changes by instrument
            instruments: Instrument tickers

        Returns:
            Total TC as fraction of NAV
        """
        total_tc = 0.0

        for instrument in instruments:
            if instrument not in turnover.index:
                continue

            trade_size = abs(turnover[instrument])
            if trade_size < 0.001:  # Skip tiny trades
                continue

            # Get spread cost
            spread_bps = self.tc_model.get_spread_bps(instrument)

            # Add market impact if enabled
            if self.config.include_impact:
                impact_bps = self.tc_model.compute_market_impact(
                    trade_size * self.config.initial_capital,
                    adv=1e9  # Assume $1B ADV for FX
                )
            else:
                impact_bps = 0

            # Total cost for this instrument
            tc = trade_size * (spread_bps + impact_bps) / 10000
            total_tc += tc

        return total_tc

    def _compute_drawdown(self, equity: pd.Series) -> pd.Series:
        """Compute drawdown series."""
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        return drawdown

    def _compute_performance_metrics(self,
                                      pnl: pd.Series,
                                      equity: pd.Series,
                                      drawdown: pd.Series,
                                      returns: pd.Series,
                                      turnover: pd.Series,
                                      tc_paid: pd.Series) -> PerformanceMetrics:
        """
        Compute comprehensive performance metrics.

        Args:
            pnl: Daily P&L
            equity: Equity curve
            drawdown: Drawdown series
            returns: Daily returns
            turnover: Daily turnover
            tc_paid: Daily TC paid

        Returns:
            PerformanceMetrics dataclass
        """
        # Clean returns
        returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns_clean) < 2:
            logger.warning("Insufficient data for metrics computation")
            return self._empty_metrics()

        # Basic return metrics
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        num_days = len(returns_clean)
        years = num_days / 252

        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        annualized_vol = returns_clean.std() * np.sqrt(252)

        # Risk-adjusted metrics
        rf_daily = 0.05 / 252  # 5% risk-free rate
        excess_returns = returns_clean - rf_daily

        sharpe_ratio = (annualized_return - 0.05) / annualized_vol if annualized_vol > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns_clean[returns_clean < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annualized_vol
        sortino_ratio = (annualized_return - 0.05) / downside_vol if downside_vol > 0 else 0

        # Drawdown metrics
        max_drawdown = drawdown.min()
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Max drawdown duration
        dd_duration = self._compute_drawdown_duration(drawdown)
        avg_drawdown = drawdown.mean()

        # Distribution metrics
        skewness = returns_clean.skew()
        kurtosis = returns_clean.kurtosis()

        # VaR and CVaR
        var_95 = returns_clean.quantile(0.05)
        var_99 = returns_clean.quantile(0.01)
        cvar_95 = returns_clean[returns_clean <= var_95].mean()

        # Trade statistics
        daily_pnl = pnl.dropna()
        wins = daily_pnl[daily_pnl > 0]
        losses = daily_pnl[daily_pnl < 0]

        win_rate = len(wins) / len(daily_pnl) if len(daily_pnl) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float('inf')

        # Turnover and TC
        avg_daily_turnover = turnover.mean()
        total_tc = tc_paid.sum()
        tc_drag = total_tc / equity.iloc[0] / years if years > 0 else 0

        # Information ratio (vs zero benchmark for now)
        information_ratio = sharpe_ratio  # Same as Sharpe when benchmark is cash

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_vol=annualized_vol,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=dd_duration,
            avg_drawdown=avg_drawdown,
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            num_trades=int(turnover[turnover > 0.01].count()),
            avg_daily_turnover=avg_daily_turnover,
            total_tc_paid=total_tc,
            tc_drag=tc_drag,
            start_date=equity.index[0],
            end_date=equity.index[-1],
            num_days=num_days
        )

    def _compute_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Compute maximum drawdown duration in days."""
        in_drawdown = drawdown < 0
        if not in_drawdown.any():
            return 0

        # Find consecutive drawdown periods
        dd_groups = (in_drawdown != in_drawdown.shift()).cumsum()
        dd_groups = dd_groups[in_drawdown]

        if len(dd_groups) == 0:
            return 0

        durations = dd_groups.groupby(dd_groups).count()
        return int(durations.max())

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for insufficient data."""
        return PerformanceMetrics(
            total_return=0, annualized_return=0, annualized_vol=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, information_ratio=0,
            max_drawdown=0, max_drawdown_duration=0, avg_drawdown=0,
            skewness=0, kurtosis=0, var_95=0, var_99=0, cvar_95=0,
            win_rate=0, avg_win=0, avg_loss=0, profit_factor=0, num_trades=0,
            avg_daily_turnover=0, total_tc_paid=0, tc_drag=0,
            start_date=pd.Timestamp.now(), end_date=pd.Timestamp.now(), num_days=0
        )

    def _analyze_crisis_periods(self,
                                 equity: pd.Series,
                                 returns: pd.Series) -> List[CrisisAnalysis]:
        """
        Analyze performance during crisis periods.

        Args:
            equity: Equity curve
            returns: Daily returns

        Returns:
            List of CrisisAnalysis objects
        """
        crisis_results = []

        for name, period in CRISIS_PERIODS.items():
            start = pd.Timestamp(period['start'])
            end = pd.Timestamp(period['end'])

            # Check if period overlaps with backtest
            if start > equity.index[-1] or end < equity.index[0]:
                continue

            # Clip to available data
            start = max(start, equity.index[0])
            end = min(end, equity.index[-1])

            # Extract period data
            period_equity = equity.loc[start:end]
            period_returns = returns.loc[start:end]

            if len(period_returns) < 5:
                continue

            # Compute metrics for period
            period_return = (period_equity.iloc[-1] / period_equity.iloc[0]) - 1
            period_dd = self._compute_drawdown(period_equity).min()

            period_vol = period_returns.std() * np.sqrt(252)
            period_sharpe = (period_returns.mean() * 252 - 0.05) / period_vol if period_vol > 0 else 0

            # Placeholder for SPX benchmark (would need actual data)
            spx_return = -0.10  # Placeholder

            crisis_results.append(CrisisAnalysis(
                period_name=name,
                start_date=start,
                end_date=end,
                return_pct=period_return,
                max_drawdown=period_dd,
                sharpe_ratio=period_sharpe,
                spx_return=spx_return,
                outperformance=period_return - spx_return
            ))

        return crisis_results

    def _compute_attribution(self,
                              positions: pd.DataFrame,
                              returns: pd.DataFrame,
                              tc: pd.Series,
                              equity: pd.Series) -> Attribution:
        """
        Compute return attribution by source.

        Args:
            positions: Position weights
            returns: Instrument returns
            tc: Transaction costs
            equity: Equity curve

        Returns:
            Attribution object
        """
        # Compute gross returns by instrument
        instrument_returns = {}
        for col in positions.columns:
            if col in returns.columns:
                inst_return = (positions[col] * returns[col]).sum()
                instrument_returns[col] = inst_return

        # Monthly returns
        monthly_equity = equity.resample('ME').last()
        monthly_returns = monthly_equity.pct_change(fill_method=None).dropna()

        # Yearly returns
        yearly_equity = equity.resample('YE').last()
        yearly_returns = yearly_equity.pct_change(fill_method=None).dropna()

        # TC cost as % of initial capital
        tc_cost = tc.sum() / equity.iloc[0]

        # Funding cost (approximate)
        avg_leverage = positions.abs().sum(axis=1).mean()
        years = len(equity) / 252
        funding_cost = (avg_leverage - 1) * self.config.funding_rate * years if avg_leverage > 1 else 0

        # Total return
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

        # Attribution by signal type (simplified - would need signal-tagged positions)
        # For now, estimate based on instrument characteristics
        momentum_return = sum(v for k, v in instrument_returns.items() if k in ['EURUSD', 'GBPUSD', 'AUDUSD']) * 0.6
        carry_return = sum(v for k, v in instrument_returns.items() if k in ['USDJPY', 'USDCNH', 'XAUUSD']) * 0.4
        value_return = 0.0
        rates_return = sum(v for k, v in instrument_returns.items() if 'SOFR' in k)

        return Attribution(
            momentum_return=momentum_return,
            carry_return=carry_return,
            value_return=value_return,
            rates_return=rates_return,
            tc_cost=tc_cost,
            funding_cost=funding_cost,
            total_return=total_return,
            instrument_returns=instrument_returns,
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns
        )

    def generate_report(self, result: BacktestResult) -> str:
        """
        Generate text summary report.

        Args:
            result: BacktestResult from run_backtest

        Returns:
            Formatted report string
        """
        m = result.metrics

        lines = [
            "=" * 70,
            "BACKTEST PERFORMANCE REPORT",
            "=" * 70,
            f"Period: {m.start_date.strftime('%Y-%m-%d')} to {m.end_date.strftime('%Y-%m-%d')} ({m.num_days} days)",
            f"Initial Capital: ${self.config.initial_capital:,.0f}",
            "",
            "RETURNS",
            "-" * 40,
            f"  Total Return:         {m.total_return:>10.2%}",
            f"  Annualized Return:    {m.annualized_return:>10.2%}",
            f"  Annualized Vol:       {m.annualized_vol:>10.2%}",
            "",
            "RISK-ADJUSTED METRICS",
            "-" * 40,
            f"  Sharpe Ratio:         {m.sharpe_ratio:>10.2f}",
            f"  Sortino Ratio:        {m.sortino_ratio:>10.2f}",
            f"  Calmar Ratio:         {m.calmar_ratio:>10.2f}",
            f"  Information Ratio:    {m.information_ratio:>10.2f}",
            "",
            "DRAWDOWN",
            "-" * 40,
            f"  Max Drawdown:         {m.max_drawdown:>10.2%}",
            f"  Max DD Duration:      {m.max_drawdown_duration:>10} days",
            f"  Avg Drawdown:         {m.avg_drawdown:>10.2%}",
            "",
            "DISTRIBUTION",
            "-" * 40,
            f"  Skewness:             {m.skewness:>10.2f}",
            f"  Kurtosis:             {m.kurtosis:>10.2f}",
            f"  VaR (95%):            {m.var_95:>10.2%}",
            f"  VaR (99%):            {m.var_99:>10.2%}",
            f"  CVaR (95%):           {m.cvar_95:>10.2%}",
            "",
            "TRADE STATISTICS",
            "-" * 40,
            f"  Win Rate:             {m.win_rate:>10.1%}",
            f"  Avg Win:              ${m.avg_win:>10,.0f}",
            f"  Avg Loss:             ${m.avg_loss:>10,.0f}",
            f"  Profit Factor:        {m.profit_factor:>10.2f}",
            f"  Number of Trades:     {m.num_trades:>10}",
            "",
            "COSTS",
            "-" * 40,
            f"  Avg Daily Turnover:   {m.avg_daily_turnover:>10.2%}",
            f"  Total TC Paid:        ${m.total_tc_paid:>10,.0f}",
            f"  Annual TC Drag:       {m.tc_drag:>10.2%}",
            "",
        ]

        # Crisis periods
        if result.crisis_analysis:
            lines.extend([
                "CRISIS PERIOD ANALYSIS",
                "-" * 40,
            ])
            for crisis in result.crisis_analysis:
                lines.extend([
                    f"  {crisis.period_name}:",
                    f"    Period Return:      {crisis.return_pct:>10.2%}",
                    f"    Max Drawdown:       {crisis.max_drawdown:>10.2%}",
                    f"    Sharpe:             {crisis.sharpe_ratio:>10.2f}",
                    ""
                ])

        # Attribution
        lines.extend([
            "RETURN ATTRIBUTION",
            "-" * 40,
            f"  Momentum:             {result.attribution.momentum_return:>10.2%}",
            f"  Carry:                {result.attribution.carry_return:>10.2%}",
            f"  Value:                {result.attribution.value_return:>10.2%}",
            f"  Rates:                {result.attribution.rates_return:>10.2%}",
            f"  TC Cost:              {-result.attribution.tc_cost:>10.2%}",
            f"  Funding Cost:         {-result.attribution.funding_cost:>10.2%}",
            "",
            "BY INSTRUMENT",
            "-" * 40,
        ])

        for inst, ret in sorted(result.attribution.instrument_returns.items(),
                                key=lambda x: -abs(x[1])):
            lines.append(f"  {inst:<15} {ret:>10.2%}")

        lines.extend(["", "=" * 70])

        return "\n".join(lines)
