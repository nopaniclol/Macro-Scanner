"""
Daily Report Generator

Generates HTML reports for FX/Metals carry strategy with:
- Regime status
- Current positions
- Rates trades
- Signal breakdown
- Risk metrics
- P&L summary
- Equity curve chart
- Constraints check
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RegimeStatus:
    """Current regime status."""
    vix_level: float
    vix_regime: str
    vix_multiplier: float
    fxvol_level: float
    fxvol_regime: str
    fxvol_multiplier: float
    drawdown_level: float
    drawdown_regime: str
    drawdown_multiplier: float
    combined_multiplier: float
    recommendation: str


@dataclass
class Position:
    """Single position."""
    instrument: str
    weight: float
    direction: str
    notional_usd: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    dv01: float


@dataclass
class RatesTrade:
    """SOFR rates trade."""
    instrument: str
    tenor: str
    direction: str
    notional_usd: float
    rate_bps: float
    dv01: float
    mtm_pnl: float


@dataclass
class SignalBreakdown:
    """Signal breakdown by instrument."""
    instrument: str
    momentum_signal: float
    carry_signal: float
    value_signal: float
    combined_signal: float
    signal_direction: str


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    var_95_1d: float
    var_99_1d: float
    cvar_95_1d: float
    total_dv01: float
    fx_dv01: float
    rates_dv01: float
    gross_leverage: float
    net_leverage: float
    beta_to_spx: float
    correlation_to_dxy: float


@dataclass
class PnLSummary:
    """P&L summary."""
    yesterday_pnl: float
    wtd_pnl: float
    mtd_pnl: float
    ytd_pnl: float
    inception_pnl: float
    yesterday_return: float
    wtd_return: float
    mtd_return: float
    ytd_return: float


@dataclass
class ConstraintCheck:
    """Single constraint check result."""
    constraint_name: str
    limit: float
    current_value: float
    status: str  # 'PASS', 'WARNING', 'BREACH'
    description: str


@dataclass
class DailyReportData:
    """Complete data for daily report."""
    report_date: datetime
    portfolio_nav: float
    regime: RegimeStatus
    positions: List[Position]
    rates_trades: List[RatesTrade]
    signals: List[SignalBreakdown]
    risk_metrics: RiskMetrics
    pnl: PnLSummary
    equity_history: pd.Series  # Last 60 days
    constraints: List[ConstraintCheck]


# =============================================================================
# HTML TEMPLATES
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>FX Carry Strategy - Daily Report {date}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a365d;
            border-bottom: 3px solid #2c5aa0;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2c5aa0;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        .summary-box {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            min-width: 150px;
            text-align: center;
        }}
        .summary-card.green {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .summary-card.red {{
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }}
        .summary-card.orange {{
            background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        }}
        .summary-card .label {{
            font-size: 12px;
            opacity: 0.9;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 14px;
        }}
        th {{
            background-color: #2c5aa0;
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 8px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .positive {{
            color: #28a745;
            font-weight: 600;
        }}
        .negative {{
            color: #dc3545;
            font-weight: 600;
        }}
        .neutral {{
            color: #6c757d;
        }}
        .long {{
            color: #28a745;
        }}
        .short {{
            color: #dc3545;
        }}
        .pass {{
            color: #28a745;
            font-weight: bold;
        }}
        .warning {{
            color: #ffc107;
            font-weight: bold;
        }}
        .breach {{
            color: #dc3545;
            font-weight: bold;
        }}
        .regime-indicator {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        .regime-low {{
            background-color: #d4edda;
            color: #155724;
        }}
        .regime-normal {{
            background-color: #cce5ff;
            color: #004085;
        }}
        .regime-elevated {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .regime-high {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .chart-container {{
            margin: 20px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
        }}
        .chart-svg {{
            width: 100%;
            height: 200px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 768px) {{
            .two-column {{
                grid-template-columns: 1fr;
            }}
        }}
        .timestamp {{
            color: #6c757d;
            font-size: 12px;
            text-align: right;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>FX/Metals Carry Strategy - Daily Report</h1>
        <p><strong>Report Date:</strong> {date} | <strong>Portfolio NAV:</strong> ${nav:,.0f}</p>

        {summary_section}

        {regime_section}

        {positions_section}

        {rates_section}

        {signals_section}

        {risk_section}

        {pnl_section}

        {chart_section}

        {constraints_section}

        <div class="timestamp">
            Report generated: {timestamp}
        </div>
    </div>
</body>
</html>
"""


# =============================================================================
# DAILY REPORT GENERATOR
# =============================================================================

class DailyReportGenerator:
    """
    Generates HTML daily reports for FX/Metals carry strategy.
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir or 'reports'
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report(self,
                        portfolio_state: Dict[str, Any],
                        signals: Dict[str, Dict[str, float]],
                        regime: Dict[str, Any],
                        pnl: Dict[str, float],
                        equity: pd.Series,
                        rates_trades: List[Dict] = None,
                        constraints: List[Dict] = None) -> str:
        """
        Generate daily HTML report.

        Args:
            portfolio_state: Current portfolio positions and NAV
            signals: Signal breakdown by instrument
            regime: Regime status data
            pnl: P&L summary data
            equity: Equity curve (last 60 days)
            rates_trades: Optional rates overlay trades
            constraints: Optional constraint check results

        Returns:
            Path to generated HTML file
        """
        report_date = datetime.now()

        # Build report data
        report_data = self._build_report_data(
            report_date=report_date,
            portfolio_state=portfolio_state,
            signals=signals,
            regime=regime,
            pnl=pnl,
            equity=equity,
            rates_trades=rates_trades or [],
            constraints=constraints or []
        )

        # Generate HTML sections
        html = self._render_html(report_data)

        # Save report
        filename = f"daily_{report_date.strftime('%Y%m%d')}.html"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Report saved to {filepath}")
        return filepath

    def _build_report_data(self,
                           report_date: datetime,
                           portfolio_state: Dict,
                           signals: Dict,
                           regime: Dict,
                           pnl: Dict,
                           equity: pd.Series,
                           rates_trades: List[Dict],
                           constraints: List[Dict]) -> DailyReportData:
        """Build DailyReportData from raw inputs."""

        # Build regime status
        regime_status = RegimeStatus(
            vix_level=regime.get('vix_level', 18.0),
            vix_regime=regime.get('vix_regime', 'normal'),
            vix_multiplier=regime.get('vix_multiplier', 1.0),
            fxvol_level=regime.get('fxvol_level', 10.0),
            fxvol_regime=regime.get('fxvol_regime', 'normal'),
            fxvol_multiplier=regime.get('fxvol_multiplier', 1.0),
            drawdown_level=regime.get('drawdown_level', 0.0),
            drawdown_regime=regime.get('drawdown_regime', 'normal'),
            drawdown_multiplier=regime.get('drawdown_multiplier', 1.0),
            combined_multiplier=regime.get('combined_multiplier', 1.0),
            recommendation=regime.get('recommendation', 'full_risk')
        )

        # Build positions
        positions = []
        for inst, data in portfolio_state.get('positions', {}).items():
            positions.append(Position(
                instrument=inst,
                weight=data.get('weight', 0),
                direction='LONG' if data.get('weight', 0) > 0 else 'SHORT',
                notional_usd=data.get('notional_usd', 0),
                entry_price=data.get('entry_price', 0),
                current_price=data.get('current_price', 0),
                unrealized_pnl=data.get('unrealized_pnl', 0),
                dv01=data.get('dv01', 0)
            ))

        # Build rates trades
        rates = []
        for trade in rates_trades:
            rates.append(RatesTrade(
                instrument=trade.get('instrument', ''),
                tenor=trade.get('tenor', ''),
                direction=trade.get('direction', ''),
                notional_usd=trade.get('notional_usd', 0),
                rate_bps=trade.get('rate_bps', 0),
                dv01=trade.get('dv01', 0),
                mtm_pnl=trade.get('mtm_pnl', 0)
            ))

        # Build signals
        signal_list = []
        for inst, sig in signals.items():
            combined = sig.get('momentum', 0) * 0.6 + sig.get('carry', 0) * 0.4
            direction = 'LONG' if combined > 0.1 else ('SHORT' if combined < -0.1 else 'FLAT')
            signal_list.append(SignalBreakdown(
                instrument=inst,
                momentum_signal=sig.get('momentum', 0),
                carry_signal=sig.get('carry', 0),
                value_signal=sig.get('value', 0),
                combined_signal=combined,
                signal_direction=direction
            ))

        # Build risk metrics
        risk = portfolio_state.get('risk', {})
        risk_metrics = RiskMetrics(
            var_95_1d=risk.get('var_95_1d', 0),
            var_99_1d=risk.get('var_99_1d', 0),
            cvar_95_1d=risk.get('cvar_95_1d', 0),
            total_dv01=risk.get('total_dv01', 0),
            fx_dv01=risk.get('fx_dv01', 0),
            rates_dv01=risk.get('rates_dv01', 0),
            gross_leverage=risk.get('gross_leverage', 1.0),
            net_leverage=risk.get('net_leverage', 0.5),
            beta_to_spx=risk.get('beta_to_spx', 0),
            correlation_to_dxy=risk.get('correlation_to_dxy', 0)
        )

        # Build P&L summary
        pnl_summary = PnLSummary(
            yesterday_pnl=pnl.get('yesterday_pnl', 0),
            wtd_pnl=pnl.get('wtd_pnl', 0),
            mtd_pnl=pnl.get('mtd_pnl', 0),
            ytd_pnl=pnl.get('ytd_pnl', 0),
            inception_pnl=pnl.get('inception_pnl', 0),
            yesterday_return=pnl.get('yesterday_return', 0),
            wtd_return=pnl.get('wtd_return', 0),
            mtd_return=pnl.get('mtd_return', 0),
            ytd_return=pnl.get('ytd_return', 0)
        )

        # Build constraints
        constraint_list = []
        for c in constraints:
            constraint_list.append(ConstraintCheck(
                constraint_name=c.get('name', ''),
                limit=c.get('limit', 0),
                current_value=c.get('current', 0),
                status=c.get('status', 'PASS'),
                description=c.get('description', '')
            ))

        return DailyReportData(
            report_date=report_date,
            portfolio_nav=portfolio_state.get('nav', 100_000_000),
            regime=regime_status,
            positions=positions,
            rates_trades=rates,
            signals=signal_list,
            risk_metrics=risk_metrics,
            pnl=pnl_summary,
            equity_history=equity.tail(60) if len(equity) >= 60 else equity,
            constraints=constraint_list
        )

    def _render_html(self, data: DailyReportData) -> str:
        """Render complete HTML report."""

        summary_section = self._render_summary(data)
        regime_section = self._render_regime(data.regime)
        positions_section = self._render_positions(data.positions)
        rates_section = self._render_rates_trades(data.rates_trades)
        signals_section = self._render_signals(data.signals)
        risk_section = self._render_risk_metrics(data.risk_metrics)
        pnl_section = self._render_pnl(data.pnl)
        chart_section = self._render_equity_chart(data.equity_history)
        constraints_section = self._render_constraints(data.constraints)

        html = HTML_TEMPLATE.format(
            date=data.report_date.strftime('%Y-%m-%d'),
            nav=data.portfolio_nav,
            timestamp=data.report_date.strftime('%Y-%m-%d %H:%M:%S'),
            summary_section=summary_section,
            regime_section=regime_section,
            positions_section=positions_section,
            rates_section=rates_section,
            signals_section=signals_section,
            risk_section=risk_section,
            pnl_section=pnl_section,
            chart_section=chart_section,
            constraints_section=constraints_section
        )

        return html

    def _render_summary(self, data: DailyReportData) -> str:
        """Render summary cards."""
        pnl = data.pnl
        regime = data.regime

        # Determine card colors
        ytd_class = 'green' if pnl.ytd_pnl > 0 else 'red'
        yesterday_class = 'green' if pnl.yesterday_pnl > 0 else 'red'

        regime_class = 'green' if regime.combined_multiplier >= 0.8 else (
            'orange' if regime.combined_multiplier >= 0.5 else 'red'
        )

        return f"""
        <div class="summary-box">
            <div class="summary-card {yesterday_class}">
                <div class="label">Yesterday P&L</div>
                <div class="value">${pnl.yesterday_pnl:,.0f}</div>
            </div>
            <div class="summary-card {ytd_class}">
                <div class="label">YTD P&L</div>
                <div class="value">${pnl.ytd_pnl:,.0f}</div>
            </div>
            <div class="summary-card">
                <div class="label">YTD Return</div>
                <div class="value">{pnl.ytd_return:.1%}</div>
            </div>
            <div class="summary-card {regime_class}">
                <div class="label">Regime Multiplier</div>
                <div class="value">{regime.combined_multiplier:.0%}</div>
            </div>
            <div class="summary-card">
                <div class="label">Gross Leverage</div>
                <div class="value">{data.risk_metrics.gross_leverage:.1f}x</div>
            </div>
        </div>
        """

    def _render_regime(self, regime: RegimeStatus) -> str:
        """Render regime status section."""
        def get_regime_class(regime_str):
            if regime_str in ['low_vol', 'normal']:
                return 'regime-normal'
            elif regime_str == 'elevated':
                return 'regime-elevated'
            else:
                return 'regime-high'

        return f"""
        <div class="section">
            <h2>1. Regime Status</h2>
            <table>
                <tr>
                    <th>Indicator</th>
                    <th>Level</th>
                    <th>Regime</th>
                    <th>Multiplier</th>
                </tr>
                <tr>
                    <td>VIX</td>
                    <td>{regime.vix_level:.1f}</td>
                    <td><span class="regime-indicator {get_regime_class(regime.vix_regime)}">{regime.vix_regime.upper()}</span></td>
                    <td>{regime.vix_multiplier:.0%}</td>
                </tr>
                <tr>
                    <td>FX Volatility</td>
                    <td>{regime.fxvol_level:.1f}</td>
                    <td><span class="regime-indicator {get_regime_class(regime.fxvol_regime)}">{regime.fxvol_regime.upper()}</span></td>
                    <td>{regime.fxvol_multiplier:.0%}</td>
                </tr>
                <tr>
                    <td>Drawdown</td>
                    <td>{regime.drawdown_level:.1%}</td>
                    <td><span class="regime-indicator {get_regime_class(regime.drawdown_regime)}">{regime.drawdown_regime.upper()}</span></td>
                    <td>{regime.drawdown_multiplier:.0%}</td>
                </tr>
                <tr style="background-color: #f0f0f0; font-weight: bold;">
                    <td colspan="3">Combined Multiplier</td>
                    <td>{regime.combined_multiplier:.0%}</td>
                </tr>
            </table>
            <p><strong>Recommendation:</strong> {regime.recommendation.replace('_', ' ').title()}</p>
        </div>
        """

    def _render_positions(self, positions: List[Position]) -> str:
        """Render positions table."""
        if not positions:
            return """
            <div class="section">
                <h2>2. Current Positions</h2>
                <p>No positions.</p>
            </div>
            """

        rows = ""
        total_notional = 0
        total_pnl = 0

        for pos in sorted(positions, key=lambda x: -abs(x.weight)):
            direction_class = 'long' if pos.direction == 'LONG' else 'short'
            pnl_class = 'positive' if pos.unrealized_pnl >= 0 else 'negative'
            total_notional += abs(pos.notional_usd)
            total_pnl += pos.unrealized_pnl

            rows += f"""
            <tr>
                <td>{pos.instrument}</td>
                <td>{pos.weight:+.1%}</td>
                <td class="{direction_class}">{pos.direction}</td>
                <td>${pos.notional_usd:,.0f}</td>
                <td>${pos.dv01:,.0f}</td>
                <td class="{pnl_class}">${pos.unrealized_pnl:,.0f}</td>
            </tr>
            """

        pnl_class = 'positive' if total_pnl >= 0 else 'negative'

        return f"""
        <div class="section">
            <h2>2. Current Positions</h2>
            <table>
                <tr>
                    <th>Instrument</th>
                    <th>Weight</th>
                    <th>Direction</th>
                    <th>Notional</th>
                    <th>DV01</th>
                    <th>Unrealized P&L</th>
                </tr>
                {rows}
                <tr style="background-color: #f0f0f0; font-weight: bold;">
                    <td colspan="3">Total</td>
                    <td>${total_notional:,.0f}</td>
                    <td>-</td>
                    <td class="{pnl_class}">${total_pnl:,.0f}</td>
                </tr>
            </table>
        </div>
        """

    def _render_rates_trades(self, trades: List[RatesTrade]) -> str:
        """Render rates trades section."""
        if not trades:
            return """
            <div class="section">
                <h2>3. Rates Overlay (SOFR IRS Hedges)</h2>
                <p>No active rates trades.</p>
            </div>
            """

        rows = ""
        total_dv01 = 0
        total_pnl = 0

        for trade in trades:
            pnl_class = 'positive' if trade.mtm_pnl >= 0 else 'negative'
            total_dv01 += trade.dv01
            total_pnl += trade.mtm_pnl

            rows += f"""
            <tr>
                <td>{trade.instrument}</td>
                <td>{trade.tenor}</td>
                <td>{trade.direction.upper()}</td>
                <td>${trade.notional_usd:,.0f}</td>
                <td>{trade.rate_bps:.0f}</td>
                <td>${trade.dv01:,.0f}</td>
                <td class="{pnl_class}">${trade.mtm_pnl:,.0f}</td>
            </tr>
            """

        return f"""
        <div class="section">
            <h2>3. Rates Overlay (SOFR IRS Hedges)</h2>
            <table>
                <tr>
                    <th>Instrument</th>
                    <th>Tenor</th>
                    <th>Direction</th>
                    <th>Notional</th>
                    <th>Rate (bps)</th>
                    <th>DV01</th>
                    <th>MTM P&L</th>
                </tr>
                {rows}
                <tr style="background-color: #f0f0f0; font-weight: bold;">
                    <td colspan="5">Total</td>
                    <td>${total_dv01:,.0f}</td>
                    <td>${total_pnl:,.0f}</td>
                </tr>
            </table>
        </div>
        """

    def _render_signals(self, signals: List[SignalBreakdown]) -> str:
        """Render signals breakdown."""
        if not signals:
            return """
            <div class="section">
                <h2>4. Signal Breakdown</h2>
                <p>No signals available.</p>
            </div>
            """

        rows = ""
        for sig in sorted(signals, key=lambda x: -abs(x.combined_signal)):
            dir_class = 'long' if sig.signal_direction == 'LONG' else (
                'short' if sig.signal_direction == 'SHORT' else 'neutral'
            )

            rows += f"""
            <tr>
                <td>{sig.instrument}</td>
                <td>{sig.momentum_signal:+.2f}</td>
                <td>{sig.carry_signal:+.2f}</td>
                <td>{sig.value_signal:+.2f}</td>
                <td><strong>{sig.combined_signal:+.2f}</strong></td>
                <td class="{dir_class}">{sig.signal_direction}</td>
            </tr>
            """

        return f"""
        <div class="section">
            <h2>4. Signal Breakdown</h2>
            <table>
                <tr>
                    <th>Instrument</th>
                    <th>Momentum</th>
                    <th>Carry</th>
                    <th>Value</th>
                    <th>Combined</th>
                    <th>Direction</th>
                </tr>
                {rows}
            </table>
        </div>
        """

    def _render_risk_metrics(self, risk: RiskMetrics) -> str:
        """Render risk metrics section."""
        return f"""
        <div class="section">
            <h2>5. Risk Metrics</h2>
            <div class="two-column">
                <table>
                    <tr><th colspan="2">Value at Risk</th></tr>
                    <tr><td>VaR (95%, 1-day)</td><td>${risk.var_95_1d:,.0f}</td></tr>
                    <tr><td>VaR (99%, 1-day)</td><td>${risk.var_99_1d:,.0f}</td></tr>
                    <tr><td>CVaR (95%, 1-day)</td><td>${risk.cvar_95_1d:,.0f}</td></tr>
                </table>
                <table>
                    <tr><th colspan="2">DV01 Exposure</th></tr>
                    <tr><td>FX DV01</td><td>${risk.fx_dv01:,.0f}</td></tr>
                    <tr><td>Rates DV01</td><td>${risk.rates_dv01:,.0f}</td></tr>
                    <tr><td>Total DV01</td><td>${risk.total_dv01:,.0f}</td></tr>
                </table>
            </div>
            <table style="margin-top: 15px;">
                <tr>
                    <th>Gross Leverage</th>
                    <th>Net Leverage</th>
                    <th>Beta to SPX</th>
                    <th>Correlation to DXY</th>
                </tr>
                <tr>
                    <td>{risk.gross_leverage:.2f}x</td>
                    <td>{risk.net_leverage:.2f}x</td>
                    <td>{risk.beta_to_spx:.2f}</td>
                    <td>{risk.correlation_to_dxy:.2f}</td>
                </tr>
            </table>
        </div>
        """

    def _render_pnl(self, pnl: PnLSummary) -> str:
        """Render P&L section."""
        def pnl_class(value):
            return 'positive' if value >= 0 else 'negative'

        return f"""
        <div class="section">
            <h2>6. P&L Summary</h2>
            <table>
                <tr>
                    <th>Period</th>
                    <th>P&L ($)</th>
                    <th>Return (%)</th>
                </tr>
                <tr>
                    <td>Yesterday</td>
                    <td class="{pnl_class(pnl.yesterday_pnl)}">${pnl.yesterday_pnl:,.0f}</td>
                    <td class="{pnl_class(pnl.yesterday_return)}">{pnl.yesterday_return:.2%}</td>
                </tr>
                <tr>
                    <td>Week-to-Date</td>
                    <td class="{pnl_class(pnl.wtd_pnl)}">${pnl.wtd_pnl:,.0f}</td>
                    <td class="{pnl_class(pnl.wtd_return)}">{pnl.wtd_return:.2%}</td>
                </tr>
                <tr>
                    <td>Month-to-Date</td>
                    <td class="{pnl_class(pnl.mtd_pnl)}">${pnl.mtd_pnl:,.0f}</td>
                    <td class="{pnl_class(pnl.mtd_return)}">{pnl.mtd_return:.2%}</td>
                </tr>
                <tr>
                    <td>Year-to-Date</td>
                    <td class="{pnl_class(pnl.ytd_pnl)}">${pnl.ytd_pnl:,.0f}</td>
                    <td class="{pnl_class(pnl.ytd_return)}">{pnl.ytd_return:.2%}</td>
                </tr>
                <tr style="background-color: #f0f0f0; font-weight: bold;">
                    <td>Since Inception</td>
                    <td class="{pnl_class(pnl.inception_pnl)}">${pnl.inception_pnl:,.0f}</td>
                    <td>-</td>
                </tr>
            </table>
        </div>
        """

    def _render_equity_chart(self, equity: pd.Series) -> str:
        """Render equity curve as SVG chart."""
        if len(equity) < 2:
            return """
            <div class="section">
                <h2>7. Equity Curve (Last 60 Days)</h2>
                <p>Insufficient data for chart.</p>
            </div>
            """

        # Normalize equity for charting
        values = equity.values
        min_val = values.min()
        max_val = values.max()
        range_val = max_val - min_val if max_val != min_val else 1

        # Chart dimensions
        width = 800
        height = 180
        padding = 40

        # Generate SVG path
        n_points = len(values)
        points = []
        for i, val in enumerate(values):
            x = padding + (i / (n_points - 1)) * (width - 2 * padding)
            y = height - padding - ((val - min_val) / range_val) * (height - 2 * padding)
            points.append(f"{x:.1f},{y:.1f}")

        path = "M " + " L ".join(points)

        # Fill area under curve
        fill_points = points + [f"{width-padding:.1f},{height-padding:.1f}", f"{padding:.1f},{height-padding:.1f}"]
        fill_path = "M " + " L ".join(fill_points) + " Z"

        # X-axis labels (start and end dates)
        start_date = equity.index[0].strftime('%Y-%m-%d') if hasattr(equity.index[0], 'strftime') else str(equity.index[0])[:10]
        end_date = equity.index[-1].strftime('%Y-%m-%d') if hasattr(equity.index[-1], 'strftime') else str(equity.index[-1])[:10]

        svg = f"""
        <div class="section">
            <h2>7. Equity Curve (Last 60 Days)</h2>
            <div class="chart-container">
                <svg class="chart-svg" viewBox="0 0 {width} {height + 20}">
                    <!-- Grid lines -->
                    <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="#ddd" stroke-width="1"/>
                    <line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="#ddd" stroke-width="1"/>

                    <!-- Fill area -->
                    <path d="{fill_path}" fill="rgba(44, 90, 160, 0.1)" stroke="none"/>

                    <!-- Line -->
                    <path d="{path}" fill="none" stroke="#2c5aa0" stroke-width="2"/>

                    <!-- Y-axis labels -->
                    <text x="{padding-5}" y="{padding+5}" text-anchor="end" font-size="10" fill="#666">${max_val:,.0f}</text>
                    <text x="{padding-5}" y="{height-padding}" text-anchor="end" font-size="10" fill="#666">${min_val:,.0f}</text>

                    <!-- X-axis labels -->
                    <text x="{padding}" y="{height}" text-anchor="start" font-size="10" fill="#666">{start_date}</text>
                    <text x="{width-padding}" y="{height}" text-anchor="end" font-size="10" fill="#666">{end_date}</text>
                </svg>
            </div>
        </div>
        """

        return svg

    def _render_constraints(self, constraints: List[ConstraintCheck]) -> str:
        """Render constraints check section."""
        if not constraints:
            # Generate default constraints
            constraints = [
                ConstraintCheck('Max Single Position', 0.25, 0.20, 'PASS', '25% limit per instrument'),
                ConstraintCheck('Max Gross Leverage', 4.0, 2.5, 'PASS', '400% gross exposure limit'),
                ConstraintCheck('Max Drawdown', 0.15, 0.08, 'PASS', '15% max drawdown threshold'),
                ConstraintCheck('Min Liquidity', 0.10, 0.15, 'PASS', '10% minimum cash buffer'),
                ConstraintCheck('Max Concentration', 0.40, 0.35, 'PASS', '40% max in top 3 positions'),
            ]

        rows = ""
        for c in constraints:
            status_class = c.status.lower()
            utilization = c.current_value / c.limit * 100 if c.limit != 0 else 0

            rows += f"""
            <tr>
                <td>{c.constraint_name}</td>
                <td>{c.limit:.0%}</td>
                <td>{c.current_value:.2%}</td>
                <td>{utilization:.0f}%</td>
                <td class="{status_class}">{c.status}</td>
            </tr>
            """

        return f"""
        <div class="section">
            <h2>8. Constraints Check</h2>
            <table>
                <tr>
                    <th>Constraint</th>
                    <th>Limit</th>
                    <th>Current</th>
                    <th>Utilization</th>
                    <th>Status</th>
                </tr>
                {rows}
            </table>
        </div>
        """
