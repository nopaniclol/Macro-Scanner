"""
Daily Report Generator for BQNT Carry Execution.

Generates HTML reports for display in BQNT notebooks.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from IPython.display import HTML, display
from ..config.parameters import REPORTING_PARAMS


class DailyReportGenerator:
    """
    Generates daily HTML reports for the carry strategy.

    Reports include:
    - Current positions and signals
    - Regime status
    - Performance summary
    - Risk metrics
    """

    def __init__(self):
        """Initialize report generator."""
        self._decimal_places = REPORTING_PARAMS['decimal_places']
        self._pct_places = REPORTING_PARAMS['pct_decimal_places']

    def generate_signal_report(
        self,
        momentum_signals: pd.Series,
        carry_signals: pd.Series,
        combined_weights: pd.Series,
        regime_info: Dict,
    ) -> str:
        """
        Generate HTML report for current signals.

        Args:
            momentum_signals: Current momentum signals
            carry_signals: Current carry signals
            combined_weights: Portfolio weights
            regime_info: Regime information dict

        Returns:
            HTML string
        """
        date_str = datetime.now().strftime('%Y-%m-%d %H:%M')

        # Build signals table
        signals_df = pd.DataFrame({
            'Momentum': momentum_signals,
            'Carry': carry_signals,
            'Weight': combined_weights,
        }).round(self._decimal_places)

        signals_html = signals_df.to_html(classes='signal-table')

        # Regime status
        vix_level = regime_info.get('vix_level', 'N/A')
        vix_regime = regime_info.get('vix_regime', 'N/A')
        multiplier = regime_info.get('combined_multiplier', 1.0)

        regime_color = self._get_regime_color(vix_regime)

        html = f"""
        <style>
            .report-container {{
                font-family: Arial, sans-serif;
                padding: 20px;
                max-width: 1000px;
            }}
            .report-header {{
                background-color: #1a1a2e;
                color: white;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
            }}
            .section {{
                margin-bottom: 25px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }}
            .section-title {{
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #333;
            }}
            .signal-table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .signal-table th, .signal-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: right;
            }}
            .signal-table th {{
                background-color: #4a4a6a;
                color: white;
            }}
            .signal-table tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .regime-box {{
                display: inline-block;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                color: white;
                background-color: {regime_color};
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
            }}
            .metric-card {{
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-label {{
                font-size: 12px;
                color: #666;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }}
        </style>

        <div class="report-container">
            <div class="report-header">
                <h2>FX/Metals Carry Strategy - Daily Report</h2>
                <p>Generated: {date_str}</p>
            </div>

            <div class="section">
                <div class="section-title">Regime Status</div>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-label">VIX Level</div>
                        <div class="metric-value">{vix_level:.1f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Regime</div>
                        <div class="regime-box">{vix_regime.upper()}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Position Multiplier</div>
                        <div class="metric-value">{multiplier:.2f}x</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <div class="section-title">Current Signals & Positions</div>
                {signals_html}
            </div>
        </div>
        """

        return html

    def generate_performance_report(
        self,
        metrics: Dict[str, float],
        positions: pd.DataFrame,
        drawdowns: pd.Series,
    ) -> str:
        """
        Generate HTML report for performance.

        Args:
            metrics: Performance metrics dict
            positions: Current positions
            drawdowns: Drawdown series

        Returns:
            HTML string
        """
        # Format metrics
        sharpe = metrics.get('sharpe_ratio', 0)
        ann_return = metrics.get('ann_return', 0) * 100
        ann_vol = metrics.get('ann_volatility', 0) * 100
        max_dd = metrics.get('max_drawdown', 0) * 100
        win_rate = metrics.get('win_rate', 0) * 100
        calmar = metrics.get('calmar_ratio', 0)

        # Color coding
        sharpe_color = '#28a745' if sharpe > 0.5 else '#dc3545' if sharpe < 0 else '#ffc107'
        return_color = '#28a745' if ann_return > 0 else '#dc3545'

        html = f"""
        <style>
            .perf-container {{
                font-family: Arial, sans-serif;
                padding: 20px;
            }}
            .perf-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin-bottom: 25px;
            }}
            .perf-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .perf-card.highlight {{
                background: linear-gradient(135deg, {sharpe_color} 0%, {sharpe_color}dd 100%);
            }}
            .perf-label {{
                font-size: 14px;
                opacity: 0.9;
            }}
            .perf-value {{
                font-size: 32px;
                font-weight: bold;
                margin-top: 10px;
            }}
            .dd-section {{
                background-color: #fff3cd;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #ffc107;
            }}
        </style>

        <div class="perf-container">
            <h3>Performance Summary</h3>

            <div class="perf-grid">
                <div class="perf-card highlight">
                    <div class="perf-label">Sharpe Ratio</div>
                    <div class="perf-value">{sharpe:.2f}</div>
                </div>
                <div class="perf-card">
                    <div class="perf-label">Annual Return</div>
                    <div class="perf-value">{ann_return:+.1f}%</div>
                </div>
                <div class="perf-card">
                    <div class="perf-label">Annual Volatility</div>
                    <div class="perf-value">{ann_vol:.1f}%</div>
                </div>
                <div class="perf-card">
                    <div class="perf-label">Max Drawdown</div>
                    <div class="perf-value">{max_dd:.1f}%</div>
                </div>
                <div class="perf-card">
                    <div class="perf-label">Win Rate</div>
                    <div class="perf-value">{win_rate:.0f}%</div>
                </div>
                <div class="perf-card">
                    <div class="perf-label">Calmar Ratio</div>
                    <div class="perf-value">{calmar:.2f}</div>
                </div>
            </div>

            <div class="dd-section">
                <strong>Current Drawdown:</strong> {drawdowns.iloc[-1] * 100:.1f}%
            </div>
        </div>
        """

        return html

    def generate_position_table(
        self,
        weights: pd.Series,
        prices: pd.Series,
        pnl: Optional[pd.Series] = None,
    ) -> str:
        """
        Generate HTML table of current positions.

        Args:
            weights: Portfolio weights
            prices: Current prices
            pnl: Position P&L (optional)

        Returns:
            HTML string
        """
        df = pd.DataFrame({
            'Weight': weights,
            'Price': prices,
        })

        if pnl is not None:
            df['P&L'] = pnl

        # Add direction indicator
        df['Direction'] = df['Weight'].apply(
            lambda x: '🟢 LONG' if x > 0.01 else ('🔴 SHORT' if x < -0.01 else '⚪ FLAT')
        )

        # Format
        df['Weight'] = df['Weight'].apply(lambda x: f'{x:.1%}')
        df['Price'] = df['Price'].apply(lambda x: f'{x:,.2f}')

        html = df.to_html(classes='position-table', escape=False)

        return f"""
        <style>
            .position-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            .position-table th {{
                background-color: #343a40;
                color: white;
                padding: 10px;
            }}
            .position-table td {{
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }}
        </style>
        <h4>Current Positions</h4>
        {html}
        """

    def display_report(
        self,
        momentum_signals: pd.Series,
        carry_signals: pd.Series,
        weights: pd.Series,
        regime_info: Dict,
        metrics: Optional[Dict] = None,
        drawdowns: Optional[pd.Series] = None,
    ) -> None:
        """
        Display full report in notebook.

        Args:
            momentum_signals: Momentum signals
            carry_signals: Carry signals
            weights: Portfolio weights
            regime_info: Regime information
            metrics: Performance metrics (optional)
            drawdowns: Drawdown series (optional)
        """
        # Signal report
        signal_html = self.generate_signal_report(
            momentum_signals,
            carry_signals,
            weights,
            regime_info,
        )
        display(HTML(signal_html))

        # Performance report (if metrics provided)
        if metrics is not None and drawdowns is not None:
            perf_html = self.generate_performance_report(
                metrics,
                weights.to_frame(),
                drawdowns,
            )
            display(HTML(perf_html))

    def _get_regime_color(self, regime: str) -> str:
        """Get color for regime status."""
        colors = {
            'low': '#28a745',      # Green
            'normal': '#17a2b8',   # Blue
            'elevated': '#ffc107', # Yellow
            'high': '#fd7e14',     # Orange
            'extreme': '#dc3545',  # Red
        }
        return colors.get(regime.lower(), '#6c757d')


def display_signal_summary(
    signals: pd.DataFrame,
    title: str = "Signal Summary"
) -> None:
    """
    Display quick signal summary.

    Args:
        signals: DataFrame of signals
        title: Report title
    """
    html = f"""
    <h3>{title}</h3>
    <p>Latest signals as of {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    {signals.round(3).to_html()}
    """
    display(HTML(html))
