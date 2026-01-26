"""
EFP (Exchange for Physical) Charting Module

This module provides visualization functions for EFP analysis:
- EFP Time Series (Traded vs Swap-Implied)
- Premium/Discount analysis
- COMEX Inventory and Coverage Ratio
- Swap Rate term structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from typing import Optional, Tuple, Dict
from datetime import datetime


# =============================================================================
# STYLING CONFIGURATION
# =============================================================================

COLORS = {
    'traded_efp': '#1f77b4',      # Blue
    'implied_efp': '#2ca02c',     # Green
    'premium': '#d62728',         # Red
    'discount': '#2ca02c',        # Green
    'threshold_1': '#ff7f0e',     # Orange
    'threshold_2': '#d62728',     # Red
    'threshold_3': '#9467bd',     # Purple
    'comex_stocks': '#17becf',    # Cyan
    'open_interest': '#bcbd22',   # Yellow-green
    'coverage_ratio': '#e377c2',  # Pink
    'gold': '#FFD700',            # Gold
    'silver': '#C0C0C0',          # Silver
}

plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# EFP TIME SERIES CHARTS
# =============================================================================

def plot_efp_timeseries(
    df: pd.DataFrame,
    metal: str = 'GOLD',
    figsize: Tuple[int, int] = (14, 8),
    show_thresholds: bool = True
) -> plt.Figure:
    """
    Plot EFP time series showing Traded EFP vs Swap-Implied EFP.

    Parameters:
        df: DataFrame with columns ['traded_efp', 'swap_implied_efp', 'threshold_1', 'threshold_2', 'threshold_3']
        metal: 'GOLD' or 'SILVER' for threshold labeling
        figsize: Figure size tuple
        show_thresholds: Whether to show threshold lines

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot EFP lines
    ax.plot(df.index, df['traded_efp'], label='Traded EFP',
            color=COLORS['traded_efp'], linewidth=2)
    ax.plot(df.index, df['swap_implied_efp'], label='Swap-Implied EFP',
            color=COLORS['implied_efp'], linewidth=2, linestyle='--')

    # Fill area between
    ax.fill_between(df.index, df['traded_efp'], df['swap_implied_efp'],
                    where=df['traded_efp'] >= df['swap_implied_efp'],
                    alpha=0.3, color=COLORS['premium'], label='Premium')
    ax.fill_between(df.index, df['traded_efp'], df['swap_implied_efp'],
                    where=df['traded_efp'] < df['swap_implied_efp'],
                    alpha=0.3, color=COLORS['discount'], label='Discount')

    # Plot thresholds
    if show_thresholds and 'threshold_1' in df.columns:
        ax.axhline(y=df['threshold_1'].iloc[0], color=COLORS['threshold_1'],
                   linestyle=':', linewidth=1.5, label=f"Threshold 1 (${df['threshold_1'].iloc[0]:.2f})")
        ax.axhline(y=df['threshold_2'].iloc[0], color=COLORS['threshold_2'],
                   linestyle=':', linewidth=1.5, label=f"Threshold 2 (${df['threshold_2'].iloc[0]:.2f})")
        ax.axhline(y=df['threshold_3'].iloc[0], color=COLORS['threshold_3'],
                   linestyle=':', linewidth=1.5, label=f"Threshold 3 (${df['threshold_3'].iloc[0]:.2f})")

    # Formatting
    ax.set_title(f'{metal} EFP Analysis: Traded vs Swap-Implied', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('EFP ($)', fontsize=12)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_efp_comparison(
    gold_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot Gold and Silver EFP side by side.

    Parameters:
        gold_df: DataFrame with Gold EFP data
        silver_df: DataFrame with Silver EFP data
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Gold EFP
    ax1 = axes[0]
    ax1.plot(gold_df.index, gold_df['traded_efp'], label='Traded EFP',
             color=COLORS['gold'], linewidth=2)
    ax1.plot(gold_df.index, gold_df['swap_implied_efp'], label='Swap-Implied EFP',
             color=COLORS['implied_efp'], linewidth=2, linestyle='--')
    ax1.fill_between(gold_df.index, gold_df['traded_efp'], gold_df['swap_implied_efp'],
                     alpha=0.3, color=COLORS['gold'])
    ax1.set_title('GOLD EFP Analysis', fontsize=12, fontweight='bold')
    ax1.set_ylabel('EFP ($)', fontsize=10)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Silver EFP (in cents)
    ax2 = axes[1]
    ax2.plot(silver_df.index, silver_df['traded_efp'] * 100, label='Traded EFP',
             color=COLORS['silver'], linewidth=2)
    ax2.plot(silver_df.index, silver_df['swap_implied_efp'] * 100, label='Swap-Implied EFP',
             color=COLORS['implied_efp'], linewidth=2, linestyle='--')
    ax2.fill_between(silver_df.index, silver_df['traded_efp'] * 100,
                     silver_df['swap_implied_efp'] * 100,
                     alpha=0.3, color=COLORS['silver'])
    ax2.set_title('SILVER EFP Analysis', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('EFP (cents)', fontsize=10)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig


# =============================================================================
# PREMIUM/DISCOUNT CHARTS
# =============================================================================

def plot_premium_discount(
    df: pd.DataFrame,
    metal: str = 'GOLD',
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot premium/discount as bar chart with color coding.

    Parameters:
        df: DataFrame with 'premium_discount' column
        metal: 'GOLD' or 'SILVER' for labeling
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Color bars based on premium/discount
    colors = [COLORS['premium'] if x >= 0 else COLORS['discount']
              for x in df['premium_discount']]

    # For large datasets, use fill instead of bars
    if len(df) > 100:
        ax.fill_between(df.index, 0, df['premium_discount'],
                        where=df['premium_discount'] >= 0,
                        color=COLORS['premium'], alpha=0.7, label='Premium')
        ax.fill_between(df.index, 0, df['premium_discount'],
                        where=df['premium_discount'] < 0,
                        color=COLORS['discount'], alpha=0.7, label='Discount')
    else:
        ax.bar(df.index, df['premium_discount'], color=colors, alpha=0.7, width=1)

    # Zero line
    ax.axhline(y=0, color='black', linewidth=1)

    # Thresholds
    if 'threshold_1' in df.columns:
        ax.axhline(y=df['threshold_1'].iloc[0], color=COLORS['threshold_1'],
                   linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=-df['threshold_1'].iloc[0], color=COLORS['threshold_1'],
                   linestyle='--', linewidth=1, alpha=0.7)

    # Formatting
    unit = 'cents' if metal.upper() == 'SILVER' else '$'
    ax.set_title(f'{metal} EFP Premium/Discount Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'Premium/Discount ({unit})', fontsize=12)

    # Add legend for bar colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['premium'], alpha=0.7, label='Premium'),
                       Patch(facecolor=COLORS['discount'], alpha=0.7, label='Discount')]
    ax.legend(handles=legend_elements, loc='upper left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_premium_discount_histogram(
    df: pd.DataFrame,
    metal: str = 'GOLD',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot histogram of premium/discount distribution.

    Parameters:
        df: DataFrame with 'premium_discount' column
        metal: 'GOLD' or 'SILVER' for labeling
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    data = df['premium_discount'].dropna()

    # Create histogram with color split
    n, bins, patches = ax.hist(data, bins=50, edgecolor='white', alpha=0.7)

    # Color bins based on whether they're premium or discount
    for patch, bin_left in zip(patches, bins[:-1]):
        if bin_left >= 0:
            patch.set_facecolor(COLORS['premium'])
        else:
            patch.set_facecolor(COLORS['discount'])

    # Add vertical line at 0
    ax.axvline(x=0, color='black', linewidth=2, linestyle='-')

    # Add mean line
    mean_val = data.mean()
    ax.axvline(x=mean_val, color='blue', linewidth=2, linestyle='--',
               label=f'Mean: ${mean_val:.2f}')

    # Threshold lines
    if 'threshold_1' in df.columns:
        threshold = df['threshold_1'].iloc[0]
        ax.axvline(x=threshold, color=COLORS['threshold_1'], linewidth=1.5,
                   linestyle=':', label=f'Threshold: ${threshold:.2f}')
        ax.axvline(x=-threshold, color=COLORS['threshold_1'], linewidth=1.5,
                   linestyle=':')

    # Formatting
    unit = 'cents' if metal.upper() == 'SILVER' else '$'
    ax.set_title(f'{metal} EFP Premium/Discount Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Premium/Discount ({unit})', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


# =============================================================================
# COMEX INVENTORY CHARTS
# =============================================================================

def plot_comex_coverage_ratio(
    df: pd.DataFrame,
    metal: str = 'GOLD',
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot COMEX stocks and coverage ratio over time.

    Parameters:
        df: DataFrame with 'comex_stocks', 'open_interest', 'coverage_ratio' columns
        metal: 'GOLD' or 'SILVER' for labeling
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot coverage ratio on primary axis
    ax1.plot(df.index, df['coverage_ratio'] * 100, color=COLORS['coverage_ratio'],
             linewidth=2, label='Coverage Ratio (%)')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Coverage Ratio (%)', fontsize=12, color=COLORS['coverage_ratio'])
    ax1.tick_params(axis='y', labelcolor=COLORS['coverage_ratio'])

    # Add warning level
    ax1.axhline(y=50, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='50% Warning Level')
    ax1.axhline(y=25, color='red', linestyle='--', linewidth=1, alpha=0.7, label='25% Critical Level')

    # Create secondary axis for COMEX stocks
    ax2 = ax1.twinx()
    ax2.fill_between(df.index, 0, df['comex_stocks'] / 1e6,
                     color=COLORS['comex_stocks'], alpha=0.3, label='COMEX Stocks')
    ax2.set_ylabel('COMEX Stocks (Million oz)', fontsize=12, color=COLORS['comex_stocks'])
    ax2.tick_params(axis='y', labelcolor=COLORS['comex_stocks'])

    # Formatting
    ax1.set_title(f'{metal} COMEX Coverage Ratio', fontsize=14, fontweight='bold')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_comex_stocks_vs_oi(
    df: pd.DataFrame,
    metal: str = 'GOLD',
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot COMEX stocks vs Open Interest over time.

    Parameters:
        df: DataFrame with 'comex_stocks', 'oi_oz' columns
        metal: 'GOLD' or 'SILVER' for labeling
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Top: Stocks and OI
    ax1 = axes[0]
    ax1.plot(df.index, df['comex_stocks'] / 1e6, color=COLORS['comex_stocks'],
             linewidth=2, label='COMEX Stocks')
    ax1.plot(df.index, df['oi_oz'] / 1e6, color=COLORS['open_interest'],
             linewidth=2, label='Open Interest (oz)')
    ax1.set_ylabel('Million Ounces', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.set_title(f'{metal} COMEX Stocks vs Open Interest', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Bottom: Coverage ratio
    ax2 = axes[1]
    ax2.fill_between(df.index, 0, df['coverage_ratio'] * 100,
                     color=COLORS['coverage_ratio'], alpha=0.5)
    ax2.axhline(y=100, color='green', linestyle='--', linewidth=1,
                label='100% Coverage', alpha=0.7)
    ax2.axhline(y=50, color='orange', linestyle='--', linewidth=1,
                label='50% Warning', alpha=0.7)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Coverage Ratio (%)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig


# =============================================================================
# SWAP RATE CHARTS
# =============================================================================

def plot_swap_rate_term_structure(
    swap_rates: Dict[int, float],
    metal: str = 'GOLD',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot swap rate term structure (1M to 7M).

    Parameters:
        swap_rates: Dictionary {tenor_months: rate}
        metal: 'GOLD' or 'SILVER' for labeling
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    tenors = list(swap_rates.keys())
    rates = [swap_rates[t] * 100 for t in tenors]  # Convert to percentage

    color = COLORS['gold'] if metal.upper() == 'GOLD' else COLORS['silver']

    ax.plot(tenors, rates, marker='o', markersize=10, linewidth=2, color=color)
    ax.fill_between(tenors, 0, rates, alpha=0.3, color=color)

    # Formatting
    ax.set_title(f'{metal} Swap Rate Term Structure', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tenor (Months)', fontsize=12)
    ax.set_ylabel('Swap Rate (%)', fontsize=12)
    ax.set_xticks(tenors)
    ax.set_xticklabels([f'{t}M' for t in tenors])
    ax.grid(True, alpha=0.3)

    # Annotate each point
    for t, r in zip(tenors, rates):
        ax.annotate(f'{r:.2f}%', (t, r), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=10)

    plt.tight_layout()
    return fig


def plot_swap_rate_history(
    df: pd.DataFrame,
    metal: str = 'GOLD',
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot historical swap rate.

    Parameters:
        df: DataFrame with 'swap_rate' column
        metal: 'GOLD' or 'SILVER' for labeling
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    color = COLORS['gold'] if metal.upper() == 'GOLD' else COLORS['silver']

    ax.plot(df.index, df['swap_rate'] * 100, color=color, linewidth=2)
    ax.fill_between(df.index, 0, df['swap_rate'] * 100, alpha=0.3, color=color)

    # Formatting
    ax.set_title(f'{metal} Swap Rate History', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Swap Rate (%)', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# COMPREHENSIVE DASHBOARD
# =============================================================================

def plot_efp_dashboard(
    efp_df: pd.DataFrame,
    comex_df: pd.DataFrame,
    metal: str = 'GOLD',
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create comprehensive EFP dashboard with multiple panels.

    Parameters:
        efp_df: DataFrame with EFP data (traded_efp, swap_implied_efp, premium_discount)
        comex_df: DataFrame with COMEX data (comex_stocks, oi_oz, coverage_ratio)
        metal: 'GOLD' or 'SILVER'
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.25)

    # Panel 1: EFP Time Series (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(efp_df.index, efp_df['traded_efp'], label='Traded EFP',
             color=COLORS['traded_efp'], linewidth=2)
    ax1.plot(efp_df.index, efp_df['swap_implied_efp'], label='Swap-Implied EFP',
             color=COLORS['implied_efp'], linewidth=2, linestyle='--')
    ax1.fill_between(efp_df.index, efp_df['traded_efp'], efp_df['swap_implied_efp'],
                     alpha=0.3, color=COLORS['premium'])

    if 'threshold_3' in efp_df.columns:
        ax1.axhline(y=efp_df['threshold_3'].iloc[0], color=COLORS['threshold_3'],
                    linestyle=':', linewidth=1.5, alpha=0.7)

    ax1.set_title(f'{metal} EFP Analysis Dashboard', fontsize=16, fontweight='bold')
    ax1.set_ylabel('EFP ($)', fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Premium/Discount (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    colors = [COLORS['premium'] if x >= 0 else COLORS['discount']
              for x in efp_df['premium_discount']]
    ax2.fill_between(efp_df.index, 0, efp_df['premium_discount'],
                     where=efp_df['premium_discount'] >= 0,
                     color=COLORS['premium'], alpha=0.7)
    ax2.fill_between(efp_df.index, 0, efp_df['premium_discount'],
                     where=efp_df['premium_discount'] < 0,
                     color=COLORS['discount'], alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_title('Premium/Discount', fontsize=12, fontweight='bold')
    ax2.set_ylabel('$ Value', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Swap Rate (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    color = COLORS['gold'] if metal.upper() == 'GOLD' else COLORS['silver']
    ax3.plot(efp_df.index, efp_df['swap_rate'] * 100, color=color, linewidth=2)
    ax3.fill_between(efp_df.index, 0, efp_df['swap_rate'] * 100, alpha=0.3, color=color)
    ax3.set_title('Swap Rate', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Rate (%)', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Panel 4: COMEX Stocks (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.fill_between(comex_df.index, 0, comex_df['comex_stocks'] / 1e6,
                     color=COLORS['comex_stocks'], alpha=0.7)
    ax4.set_title('COMEX Stocks', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=10)
    ax4.set_ylabel('Million oz', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Coverage Ratio (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.fill_between(comex_df.index, 0, comex_df['coverage_ratio'] * 100,
                     color=COLORS['coverage_ratio'], alpha=0.7)
    ax5.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax5.axhline(y=50, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax5.set_title('Coverage Ratio', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Date', fontsize=10)
    ax5.set_ylabel('Ratio (%)', fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Format all x-axes
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    return fig


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_chart(fig: plt.Figure, filename: str, dpi: int = 150) -> None:
    """Save chart to file."""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {filename}")


def plot_all_efp_charts(
    efp_df: pd.DataFrame,
    comex_df: pd.DataFrame,
    metal: str = 'GOLD',
    save_path: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Generate all EFP charts for a given metal.

    Parameters:
        efp_df: DataFrame with EFP data
        comex_df: DataFrame with COMEX inventory data
        metal: 'GOLD' or 'SILVER'
        save_path: Optional path to save charts

    Returns:
        Dictionary of figure names to Figure objects
    """
    figures = {}

    # 1. EFP Time Series
    figures['efp_timeseries'] = plot_efp_timeseries(efp_df, metal)

    # 2. Premium/Discount
    figures['premium_discount'] = plot_premium_discount(efp_df, metal)

    # 3. Premium/Discount Histogram
    figures['premium_histogram'] = plot_premium_discount_histogram(efp_df, metal)

    # 4. COMEX Coverage
    figures['comex_coverage'] = plot_comex_coverage_ratio(comex_df, metal)

    # 5. Stocks vs OI
    figures['stocks_vs_oi'] = plot_comex_stocks_vs_oi(comex_df, metal)

    # 6. Swap Rate History
    figures['swap_rate'] = plot_swap_rate_history(efp_df, metal)

    # 7. Dashboard
    figures['dashboard'] = plot_efp_dashboard(efp_df, comex_df, metal)

    # Save if path provided
    if save_path:
        for name, fig in figures.items():
            save_chart(fig, f"{save_path}/{metal.lower()}_{name}.png")

    return figures


if __name__ == "__main__":
    # Demo with sample data
    import numpy as np

    # Create sample data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')
    n = len(dates)

    efp_df = pd.DataFrame({
        'traded_efp': np.random.normal(8, 2, n).cumsum() / 50 + 5,
        'swap_implied_efp': np.random.normal(7, 1, n).cumsum() / 50 + 4,
        'swap_rate': np.random.uniform(0.04, 0.05, n),
        'threshold_1': 4.0,
        'threshold_2': 6.0,
        'threshold_3': 8.0,
    }, index=dates)
    efp_df['premium_discount'] = efp_df['traded_efp'] - efp_df['swap_implied_efp']

    comex_df = pd.DataFrame({
        'comex_stocks': np.random.uniform(20e6, 25e6, n),
        'oi_oz': np.random.uniform(50e6, 60e6, n),
    }, index=dates)
    comex_df['coverage_ratio'] = comex_df['comex_stocks'] / comex_df['oi_oz']

    # Generate dashboard
    fig = plot_efp_dashboard(efp_df, comex_df, 'GOLD')
    plt.show()
