#!/usr/bin/env python3
"""
TOPIX-COMEX Gold Arbitrage Analysis
High-Frequency Trading Spread Analysis for Gold Futures

This module analyzes arbitrage opportunities between:
- TOCOM/JPX Gold Standard Futures (Tokyo)
- COMEX Gold Futures (New York)

Key arbitrage drivers:
1. Tick size asymmetry (COMEX: $0.10/oz vs TOCOM: JPY 1/g)
2. Currency fluctuation (USDJPY)
3. Trading hours overlap windows
4. Price discovery leadership patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONTRACT SPECIFICATIONS
# ============================================================================

CONTRACT_SPECS = {
    'COMEX_GC': {
        'name': 'COMEX Gold Futures',
        'exchange': 'CME/COMEX',
        'contract_size_oz': 100,          # 100 troy ounces
        'contract_size_kg': 3.11035,      # ~3.11 kg
        'tick_size_per_oz': 0.10,         # $0.10 per troy ounce
        'tick_value_usd': 10.00,          # $10.00 per contract
        'currency': 'USD',
        'trading_hours_ct': '17:00-16:00',  # Sunday-Friday, 23 hours
        'initial_margin_usd': 12100,
        'maintenance_margin_usd': 11000,
    },
    'TOCOM_GOLD': {
        'name': 'TOCOM Gold Standard Futures',
        'exchange': 'JPX/TOCOM',
        'contract_size_oz': 32.15,        # ~32.15 troy ounces (1 kg)
        'contract_size_kg': 1.0,          # 1 kg
        'tick_size_per_g': 1,             # JPY 1 per gram
        'tick_value_jpy': 1000,           # JPY 1,000 per contract
        'currency': 'JPY',
        'trading_hours_jst': '08:45-15:45, 17:00-06:00',
        'trading_hours_ct': '18:45-01:45, 03:00-16:00',  # Converted to CT
    },
    'TOCOM_MINI': {
        'name': 'TOCOM Gold Mini Futures',
        'exchange': 'JPX/TOCOM',
        'contract_size_oz': 3.215,        # ~3.215 troy ounces (100g)
        'contract_size_kg': 0.1,          # 100g
        'tick_size_per_g': 0.5,           # JPY 0.5 per gram
        'tick_value_jpy': 50,             # JPY 50 per contract
        'currency': 'JPY',
    }
}

# Conversion constants
TROY_OZ_PER_KG = 32.1507
GRAMS_PER_KG = 1000
GRAMS_PER_TROY_OZ = 31.1035


# ============================================================================
# UNIT CONVERSION UTILITIES
# ============================================================================

def usd_per_oz_to_jpy_per_gram(price_usd_oz: float, usdjpy: float) -> float:
    """Convert COMEX price (USD/oz) to TOCOM units (JPY/gram)"""
    price_usd_gram = price_usd_oz / GRAMS_PER_TROY_OZ
    return price_usd_gram * usdjpy


def jpy_per_gram_to_usd_per_oz(price_jpy_gram: float, usdjpy: float) -> float:
    """Convert TOCOM price (JPY/gram) to COMEX units (USD/oz)"""
    price_usd_gram = price_jpy_gram / usdjpy
    return price_usd_gram * GRAMS_PER_TROY_OZ


def calculate_equivalent_tick_sizes(usdjpy: float) -> Dict:
    """
    Calculate equivalent tick sizes across both exchanges
    in common units for comparison
    """
    # COMEX tick in various units
    comex_tick_usd_oz = 0.10
    comex_tick_usd_kg = comex_tick_usd_oz * TROY_OZ_PER_KG  # ~$3.22/kg
    comex_tick_jpy_kg = comex_tick_usd_kg * usdjpy
    comex_tick_jpy_g = comex_tick_jpy_kg / GRAMS_PER_KG

    # TOCOM tick in various units
    tocom_tick_jpy_g = 1.0
    tocom_tick_jpy_kg = tocom_tick_jpy_g * GRAMS_PER_KG  # JPY 1000/kg
    tocom_tick_usd_kg = tocom_tick_jpy_kg / usdjpy
    tocom_tick_usd_oz = tocom_tick_usd_kg / TROY_OZ_PER_KG

    return {
        'comex': {
            'usd_per_oz': comex_tick_usd_oz,
            'usd_per_kg': comex_tick_usd_kg,
            'jpy_per_kg': comex_tick_jpy_kg,
            'jpy_per_g': comex_tick_jpy_g,
        },
        'tocom': {
            'jpy_per_g': tocom_tick_jpy_g,
            'jpy_per_kg': tocom_tick_jpy_kg,
            'usd_per_kg': tocom_tick_usd_kg,
            'usd_per_oz': tocom_tick_usd_oz,
        },
        'tick_ratio_tocom_vs_comex': tocom_tick_jpy_g / comex_tick_jpy_g,
        'usdjpy_rate': usdjpy,
    }


# ============================================================================
# ARBITRAGE SPREAD CALCULATIONS
# ============================================================================

def calculate_basis_spread(
    comex_price_usd_oz: float,
    tocom_price_jpy_g: float,
    usdjpy: float
) -> Dict:
    """
    Calculate the basis spread between COMEX and TOCOM gold

    Returns spread in multiple units for analysis
    """
    # Convert both prices to common units
    comex_in_jpy_g = usd_per_oz_to_jpy_per_gram(comex_price_usd_oz, usdjpy)
    tocom_in_usd_oz = jpy_per_gram_to_usd_per_oz(tocom_price_jpy_g, usdjpy)

    # Absolute spreads
    spread_jpy_g = tocom_price_jpy_g - comex_in_jpy_g
    spread_usd_oz = tocom_in_usd_oz - comex_price_usd_oz

    # Percentage spread (basis points)
    spread_pct = (spread_jpy_g / comex_in_jpy_g) * 100
    spread_bps = spread_pct * 100

    # Per-contract values
    comex_contract_value = comex_price_usd_oz * CONTRACT_SPECS['COMEX_GC']['contract_size_oz']
    tocom_contract_value = (tocom_price_jpy_g * GRAMS_PER_KG) / usdjpy * TROY_OZ_PER_KG

    spread_per_comex_contract = spread_usd_oz * CONTRACT_SPECS['COMEX_GC']['contract_size_oz']
    spread_per_tocom_contract = spread_jpy_g * GRAMS_PER_KG

    return {
        'spread_jpy_per_g': spread_jpy_g,
        'spread_usd_per_oz': spread_usd_oz,
        'spread_pct': spread_pct,
        'spread_bps': spread_bps,
        'spread_per_comex_contract_usd': spread_per_comex_contract,
        'spread_per_tocom_contract_jpy': spread_per_tocom_contract,
        'comex_price_jpy_g': comex_in_jpy_g,
        'tocom_price_usd_oz': tocom_in_usd_oz,
        'comex_price_usd_oz': comex_price_usd_oz,
        'tocom_price_jpy_g': tocom_price_jpy_g,
        'usdjpy': usdjpy,
    }


def calculate_arbitrage_threshold(
    usdjpy: float,
    comex_fee_usd: float = 2.50,    # Round-trip commission per contract
    tocom_fee_jpy: float = 500,      # Round-trip commission per contract
    slippage_ticks: int = 1,         # Expected slippage in ticks
    latency_buffer_bps: float = 0.5  # Buffer for execution latency
) -> Dict:
    """
    Calculate minimum spread required for profitable arbitrage
    accounting for transaction costs, slippage, and latency
    """
    # Get tick sizes
    tick_info = calculate_equivalent_tick_sizes(usdjpy)

    # COMEX costs per oz
    comex_slippage_usd = slippage_ticks * tick_info['comex']['usd_per_oz']
    comex_fee_per_oz = comex_fee_usd / CONTRACT_SPECS['COMEX_GC']['contract_size_oz']
    comex_total_cost_usd_oz = comex_slippage_usd + comex_fee_per_oz

    # TOCOM costs per gram
    tocom_slippage_jpy = slippage_ticks * tick_info['tocom']['jpy_per_g']
    tocom_fee_per_g = tocom_fee_jpy / (CONTRACT_SPECS['TOCOM_GOLD']['contract_size_kg'] * GRAMS_PER_KG)
    tocom_total_cost_jpy_g = tocom_slippage_jpy + tocom_fee_per_g

    # Convert to common units (USD/oz)
    tocom_total_cost_usd_oz = jpy_per_gram_to_usd_per_oz(tocom_total_cost_jpy_g, usdjpy)

    # Total round-trip cost
    total_cost_usd_oz = comex_total_cost_usd_oz + tocom_total_cost_usd_oz

    # Add latency buffer
    latency_buffer_usd = (latency_buffer_bps / 100) * 100  # Assuming ~$2000/oz gold
    min_spread_usd_oz = total_cost_usd_oz + (latency_buffer_usd / 100)

    return {
        'min_spread_usd_per_oz': min_spread_usd_oz,
        'min_spread_jpy_per_g': usd_per_oz_to_jpy_per_gram(min_spread_usd_oz, usdjpy),
        'comex_costs_usd_per_oz': comex_total_cost_usd_oz,
        'tocom_costs_usd_per_oz': tocom_total_cost_usd_oz,
        'total_round_trip_cost_usd': total_cost_usd_oz * CONTRACT_SPECS['COMEX_GC']['contract_size_oz'],
        'breakeven_bps': (min_spread_usd_oz / 2000) * 10000,  # Assuming ~$2000/oz gold
    }


# ============================================================================
# TICK SIZE ARBITRAGE ANALYSIS
# ============================================================================

def analyze_tick_granularity_advantage(usdjpy: float, gold_price_usd_oz: float = 2000) -> Dict:
    """
    Analyze the tick size granularity difference between exchanges

    COMEX has finer granularity in USD terms, while TOCOM has coarser
    ticks. This creates opportunities when:
    1. TOCOM price jumps by 1 tick while COMEX moves less
    2. Currency moves create sub-tick price differences
    """
    tick_info = calculate_equivalent_tick_sizes(usdjpy)

    # Calculate tick sizes as percentage of price
    comex_tick_pct = (tick_info['comex']['usd_per_oz'] / gold_price_usd_oz) * 100
    tocom_tick_pct = (tick_info['tocom']['usd_per_oz'] / gold_price_usd_oz) * 100

    # Granularity ratio
    granularity_ratio = tick_info['tocom']['usd_per_oz'] / tick_info['comex']['usd_per_oz']

    # USDJPY move required to shift TOCOM by 1 tick (in JPY/g terms)
    tocom_tick_jpy_g = 1.0
    gold_price_jpy_g = usd_per_oz_to_jpy_per_gram(gold_price_usd_oz, usdjpy)
    usdjpy_sensitivity = tocom_tick_jpy_g / (gold_price_usd_oz / GRAMS_PER_TROY_OZ)

    return {
        'comex_tick_pct_of_price': comex_tick_pct,
        'tocom_tick_pct_of_price': tocom_tick_pct,
        'granularity_ratio': granularity_ratio,
        'tocom_ticks_per_comex_tick': granularity_ratio,
        'usdjpy_move_for_1_tocom_tick': usdjpy_sensitivity,
        'tick_info': tick_info,
        'analysis': f"TOCOM tick is {granularity_ratio:.1f}x coarser than COMEX. "
                   f"A {usdjpy_sensitivity:.3f} move in USDJPY shifts TOCOM by 1 tick."
    }


# ============================================================================
# TRADING HOURS OVERLAP ANALYSIS
# ============================================================================

def get_trading_session_overlap() -> Dict:
    """
    Analyze trading hours overlap between COMEX and TOCOM
    All times in Central Time (CT) for consistency
    """
    sessions = {
        'comex': {
            'start': '17:00',  # Previous day
            'end': '16:00',    # Current day
            'duration_hours': 23,
        },
        'tocom_day': {
            'start': '18:45',  # CT (8:45 JST)
            'end': '01:45',    # CT (15:45 JST)
            'duration_hours': 7,
        },
        'tocom_night': {
            'start': '03:00',  # CT (17:00 JST)
            'end': '16:00',    # CT (6:00 JST next day)
            'duration_hours': 13,
        },
    }

    # Key overlap windows
    overlap_windows = [
        {
            'name': 'Asian Morning (TOCOM Day Open)',
            'ct_time': '18:45-01:45',
            'jst_time': '08:45-15:45',
            'characteristics': 'Tokyo price discovery, lower COMEX liquidity',
            'arbitrage_potential': 'HIGH - TOCOM leads, COMEX adjusts',
        },
        {
            'name': 'Asian Night (TOCOM Night Session)',
            'ct_time': '03:00-08:30',
            'jst_time': '17:00-22:30',
            'characteristics': 'Dual market liquidity building',
            'arbitrage_potential': 'MEDIUM - Both markets active',
        },
        {
            'name': 'US Pre-Market',
            'ct_time': '08:30-09:30',
            'jst_time': '22:30-23:30',
            'characteristics': 'US economic data releases',
            'arbitrage_potential': 'HIGH - Volatility spike, COMEX leads',
        },
        {
            'name': 'US Trading Hours',
            'ct_time': '09:30-16:00',
            'jst_time': '23:30-06:00',
            'characteristics': 'Maximum liquidity, COMEX dominant',
            'arbitrage_potential': 'MEDIUM - Tight spreads, high competition',
        },
    ]

    return {
        'sessions': sessions,
        'overlap_windows': overlap_windows,
        'total_overlap_hours': 20,  # ~20 hours of overlap
        'best_arb_windows': ['Asian Morning', 'US Pre-Market'],
    }


# ============================================================================
# SPREAD TIME SERIES ANALYSIS
# ============================================================================

class SpreadAnalyzer:
    """
    Analyzes historical spread data for mean reversion and
    arbitrage opportunity detection
    """

    def __init__(self, lookback_periods: int = 100):
        self.lookback = lookback_periods
        self.spread_history = []
        self.stats = {}

    def update(self, spread_bps: float, timestamp: datetime):
        """Add new spread observation"""
        self.spread_history.append({
            'timestamp': timestamp,
            'spread_bps': spread_bps
        })

        # Keep only lookback period
        if len(self.spread_history) > self.lookback:
            self.spread_history = self.spread_history[-self.lookback:]

        self._update_stats()

    def _update_stats(self):
        """Calculate rolling statistics"""
        if len(self.spread_history) < 20:
            return

        spreads = [s['spread_bps'] for s in self.spread_history]

        self.stats = {
            'mean': np.mean(spreads),
            'std': np.std(spreads),
            'min': np.min(spreads),
            'max': np.max(spreads),
            'current': spreads[-1],
            'z_score': (spreads[-1] - np.mean(spreads)) / np.std(spreads) if np.std(spreads) > 0 else 0,
            'percentile': np.percentile(spreads, 50),
        }

    def get_signal(self, threshold_z: float = 2.0) -> Dict:
        """
        Generate trading signal based on spread deviation

        Returns:
            signal: 'BUY_COMEX_SELL_TOCOM', 'BUY_TOCOM_SELL_COMEX', or 'NEUTRAL'
        """
        if not self.stats:
            return {'signal': 'INSUFFICIENT_DATA', 'confidence': 0}

        z = self.stats['z_score']

        if z > threshold_z:
            # TOCOM expensive vs COMEX - sell TOCOM, buy COMEX
            return {
                'signal': 'BUY_COMEX_SELL_TOCOM',
                'z_score': z,
                'confidence': min(abs(z) / 3, 1.0),
                'expected_reversion_bps': self.stats['current'] - self.stats['mean'],
            }
        elif z < -threshold_z:
            # COMEX expensive vs TOCOM - sell COMEX, buy TOCOM
            return {
                'signal': 'BUY_TOCOM_SELL_COMEX',
                'z_score': z,
                'confidence': min(abs(z) / 3, 1.0),
                'expected_reversion_bps': self.stats['mean'] - self.stats['current'],
            }
        else:
            return {
                'signal': 'NEUTRAL',
                'z_score': z,
                'confidence': 0,
                'expected_reversion_bps': 0,
            }

    def get_half_life(self) -> Optional[float]:
        """
        Estimate mean reversion half-life using OLS regression
        on spread changes
        """
        if len(self.spread_history) < 30:
            return None

        spreads = np.array([s['spread_bps'] for s in self.spread_history])

        # Regress spread(t) on spread(t-1)
        y = spreads[1:]
        x = spreads[:-1]

        # OLS: y = a + b*x
        n = len(y)
        mean_x, mean_y = np.mean(x), np.mean(y)

        beta = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x)**2)

        if beta >= 1 or beta <= 0:
            return None  # No mean reversion

        # Half-life = -ln(2) / ln(beta)
        half_life = -np.log(2) / np.log(beta)

        return half_life


# ============================================================================
# HIGH-FREQUENCY ARBITRAGE DETECTOR
# ============================================================================

class HFTArbitrageDetector:
    """
    Real-time arbitrage opportunity detector for HFT strategies
    """

    def __init__(
        self,
        usdjpy: float = 150.0,
        comex_fee_usd: float = 2.50,
        tocom_fee_jpy: float = 500,
        latency_ms: float = 50,  # Expected round-trip latency
    ):
        self.usdjpy = usdjpy
        self.threshold = calculate_arbitrage_threshold(
            usdjpy, comex_fee_usd, tocom_fee_jpy
        )
        self.latency_ms = latency_ms
        self.spread_analyzer = SpreadAnalyzer()

    def update_usdjpy(self, usdjpy: float):
        """Update FX rate and recalculate thresholds"""
        self.usdjpy = usdjpy
        self.threshold = calculate_arbitrage_threshold(self.usdjpy)

    def check_opportunity(
        self,
        comex_bid: float,
        comex_ask: float,
        tocom_bid_jpy_g: float,
        tocom_ask_jpy_g: float,
        timestamp: datetime
    ) -> Dict:
        """
        Check for arbitrage opportunity given current quotes

        Returns detailed opportunity analysis
        """
        # Convert TOCOM quotes to USD/oz
        tocom_bid_usd = jpy_per_gram_to_usd_per_oz(tocom_bid_jpy_g, self.usdjpy)
        tocom_ask_usd = jpy_per_gram_to_usd_per_oz(tocom_ask_jpy_g, self.usdjpy)

        # Calculate mid prices and spreads
        comex_mid = (comex_bid + comex_ask) / 2
        tocom_mid = (tocom_bid_usd + tocom_ask_usd) / 2

        mid_spread = tocom_mid - comex_mid
        mid_spread_bps = (mid_spread / comex_mid) * 10000

        # Update spread analyzer
        self.spread_analyzer.update(mid_spread_bps, timestamp)

        # Check arbitrage opportunities (crossing quotes)
        opportunities = []

        # Opportunity 1: Buy COMEX, Sell TOCOM
        if tocom_bid_usd > comex_ask:
            profit_usd = tocom_bid_usd - comex_ask
            profit_pct = (profit_usd / comex_ask) * 100

            if profit_usd > self.threshold['min_spread_usd_per_oz']:
                opportunities.append({
                    'direction': 'BUY_COMEX_SELL_TOCOM',
                    'entry_comex': comex_ask,
                    'exit_tocom': tocom_bid_usd,
                    'gross_profit_usd_oz': profit_usd,
                    'net_profit_usd_oz': profit_usd - self.threshold['min_spread_usd_per_oz'],
                    'profit_pct': profit_pct,
                    'contracts_comex': 1,
                    'contracts_tocom': 3,  # ~3 TOCOM = 1 COMEX (by oz)
                })

        # Opportunity 2: Buy TOCOM, Sell COMEX
        if comex_bid > tocom_ask_usd:
            profit_usd = comex_bid - tocom_ask_usd
            profit_pct = (profit_usd / tocom_ask_usd) * 100

            if profit_usd > self.threshold['min_spread_usd_per_oz']:
                opportunities.append({
                    'direction': 'BUY_TOCOM_SELL_COMEX',
                    'entry_tocom': tocom_ask_usd,
                    'exit_comex': comex_bid,
                    'gross_profit_usd_oz': profit_usd,
                    'net_profit_usd_oz': profit_usd - self.threshold['min_spread_usd_per_oz'],
                    'profit_pct': profit_pct,
                    'contracts_comex': 1,
                    'contracts_tocom': 3,
                })

        # Get mean-reversion signal
        mean_reversion_signal = self.spread_analyzer.get_signal()

        return {
            'timestamp': timestamp,
            'comex_bid': comex_bid,
            'comex_ask': comex_ask,
            'tocom_bid_usd': tocom_bid_usd,
            'tocom_ask_usd': tocom_ask_usd,
            'mid_spread_usd': mid_spread,
            'mid_spread_bps': mid_spread_bps,
            'opportunities': opportunities,
            'has_arbitrage': len(opportunities) > 0,
            'mean_reversion_signal': mean_reversion_signal,
            'threshold': self.threshold,
        }


# ============================================================================
# BLOOMBERG DATA INTEGRATION
# ============================================================================

def fetch_gold_data_bloomberg(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch gold futures and FX data from Bloomberg BQL

    Requires active Bloomberg Terminal connection
    """
    try:
        import bql
        bq = bql.Service()

        # Tickers
        tickers = [
            'GCA Comdty',      # COMEX Gold Active
            'JAU Comdty',      # TOCOM Gold (Bloomberg ticker)
            'USDJPY Curncy',   # USD/JPY
        ]

        # Build BQL query
        ticker_str = ",".join([f'"{t}"' for t in tickers])
        query = f"""
        get(PX_LAST, PX_BID, PX_ASK, PX_VOLUME)
        for([{ticker_str}])
        with(dates=range({start_date}, {end_date}), fill=prev)
        """

        response = bq.execute(query)
        df = response[0].df()

        return df

    except ImportError:
        print("Bloomberg BQL not available. Using sample data.")
        return None
    except Exception as e:
        print(f"Bloomberg connection error: {e}")
        return None


def fetch_gold_data_yfinance(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch gold futures and FX data from Yahoo Finance (free alternative)
    """
    try:
        import yfinance as yf

        # Note: Yahoo Finance doesn't have TOCOM data directly
        # We'll use COMEX and approximate
        tickers = {
            'GC=F': 'COMEX_Gold',
            'JPY=X': 'USDJPY',
        }

        data = {}
        for ticker, name in tickers.items():
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                # Handle multi-index columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[name] = df

        return data

    except ImportError:
        print("yfinance not available")
        return None


# ============================================================================
# ANALYSIS REPORTS
# ============================================================================

def generate_arbitrage_report(usdjpy: float = 150.0, gold_price_usd: float = 2000.0):
    """
    Generate comprehensive arbitrage analysis report
    """
    print("=" * 80)
    print("TOPIX-COMEX GOLD ARBITRAGE ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nAssumptions: USDJPY = {usdjpy}, Gold = ${gold_price_usd}/oz")
    print()

    # 1. Tick Size Analysis
    print("-" * 40)
    print("1. TICK SIZE ANALYSIS")
    print("-" * 40)
    tick_analysis = analyze_tick_granularity_advantage(usdjpy, gold_price_usd)
    print(f"   COMEX tick: ${tick_analysis['tick_info']['comex']['usd_per_oz']:.2f}/oz")
    print(f"              = {tick_analysis['comex_tick_pct_of_price']:.4f}% of price")
    print(f"   TOCOM tick: JPY {tick_analysis['tick_info']['tocom']['jpy_per_g']:.0f}/g")
    print(f"              = ${tick_analysis['tick_info']['tocom']['usd_per_oz']:.4f}/oz")
    print(f"              = {tick_analysis['tocom_tick_pct_of_price']:.4f}% of price")
    print(f"\n   Granularity Ratio: {tick_analysis['granularity_ratio']:.1f}x")
    print(f"   (TOCOM tick is {tick_analysis['granularity_ratio']:.1f}x coarser than COMEX)")
    print()

    # 2. Arbitrage Threshold
    print("-" * 40)
    print("2. ARBITRAGE THRESHOLD ANALYSIS")
    print("-" * 40)
    threshold = calculate_arbitrage_threshold(usdjpy)
    print(f"   Minimum profitable spread:")
    print(f"   - USD: ${threshold['min_spread_usd_per_oz']:.4f}/oz")
    print(f"   - JPY: JPY {threshold['min_spread_jpy_per_g']:.2f}/g")
    print(f"   - BPS: {threshold['breakeven_bps']:.2f} bps")
    print(f"\n   Cost breakdown (USD/oz):")
    print(f"   - COMEX costs: ${threshold['comex_costs_usd_per_oz']:.4f}")
    print(f"   - TOCOM costs: ${threshold['tocom_costs_usd_per_oz']:.4f}")
    print(f"   - Round-trip per COMEX contract: ${threshold['total_round_trip_cost_usd']:.2f}")
    print()

    # 3. Trading Hours
    print("-" * 40)
    print("3. TRADING HOURS OVERLAP")
    print("-" * 40)
    sessions = get_trading_session_overlap()
    print("   Best arbitrage windows:")
    for window in sessions['overlap_windows']:
        print(f"   - {window['name']}")
        print(f"     CT: {window['ct_time']} | JST: {window['jst_time']}")
        print(f"     Potential: {window['arbitrage_potential']}")
    print()

    # 4. Sample Spread Calculation
    print("-" * 40)
    print("4. SAMPLE SPREAD CALCULATION")
    print("-" * 40)
    # Assume TOCOM trades at small premium due to JPY weakness
    tocom_premium_pct = 0.05  # 5 bps premium
    tocom_price_jpy_g = usd_per_oz_to_jpy_per_gram(gold_price_usd, usdjpy) * (1 + tocom_premium_pct/100)

    spread = calculate_basis_spread(gold_price_usd, tocom_price_jpy_g, usdjpy)
    print(f"   COMEX: ${gold_price_usd}/oz")
    print(f"   TOCOM: JPY {tocom_price_jpy_g:.0f}/g (assumed {tocom_premium_pct} bps premium)")
    print(f"\n   Spread: {spread['spread_bps']:.2f} bps")
    print(f"   Spread: ${spread['spread_usd_per_oz']:.4f}/oz")
    print(f"   Per COMEX contract: ${spread['spread_per_comex_contract_usd']:.2f}")

    if spread['spread_bps'] > threshold['breakeven_bps']:
        print(f"\n   ✓ ARBITRAGE OPPORTUNITY EXISTS")
        print(f"     Net profit: {spread['spread_bps'] - threshold['breakeven_bps']:.2f} bps")
    else:
        print(f"\n   ✗ No arbitrage (spread below threshold)")
    print()

    # 5. Key Insights
    print("-" * 40)
    print("5. KEY INSIGHTS FOR HFT STRATEGY")
    print("-" * 40)
    print("""
   a) TICK SIZE ADVANTAGE:
      - COMEX has 10x finer price granularity
      - TOCOM price "rounds" to nearest JPY 1/g
      - Exploit when TOCOM overshoots on tick jumps

   b) CURRENCY CORRELATION:
      - USDJPY moves create spread dislocations
      - Monitor BoJ intervention windows
      - JPY weakness = TOCOM premium opportunity

   c) OPTIMAL EXECUTION:
      - Asian morning: TOCOM leads, fade extremes
      - US data releases: COMEX leads, follow momentum
      - Use TOCOM Mini for position sizing flexibility

   d) RISK FACTORS:
      - FX volatility can exceed spread profit
      - Latency disadvantage vs local HFT
      - Position limits and margin requirements
    """)

    print("=" * 80)
    return {
        'tick_analysis': tick_analysis,
        'threshold': threshold,
        'sessions': sessions,
        'sample_spread': spread,
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run analysis with current market assumptions
    # Update these values with live data for actual trading

    CURRENT_USDJPY = 156.0  # Update with live rate
    CURRENT_GOLD_USD = 2650.0  # Update with live price

    report = generate_arbitrage_report(
        usdjpy=CURRENT_USDJPY,
        gold_price_usd=CURRENT_GOLD_USD
    )

    print("\n\nQUICK REFERENCE:")
    print(f"Min spread for profit: {report['threshold']['breakeven_bps']:.2f} bps")
    print(f"Best windows: Asian Morning (18:45-01:45 CT), US Pre-Market (08:30-09:30 CT)")
