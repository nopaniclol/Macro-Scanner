"""
EFP (Exchange for Physical) Analysis Module

This module calculates and analyzes EFP metrics for Gold and Silver:
- Swap-Implied EFP: Theoretical cost based on financing rates
- Traded EFP: Actual market EFP (Closing Futures - Closing Spot)
- Premium/Discount: The difference between traded and implied
- COMEX inventory and coverage ratios

Bloomberg tickers used:
- Gold: XAU Curncy, GCA Comdty, XAUSR1M-7M Curncy, COMXGOLD Index
- Silver: XAG Curncy, SIA Comdty, XAGSR1M-7M Curncy, COMXSILV Index
"""

import bql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Union

# Initialize BQL service
bq = bql.Service()

# =============================================================================
# TICKER DEFINITIONS
# =============================================================================

GOLD_TICKERS = {
    'spot': 'XAU Curncy',
    'futures': 'GCA Comdty',
    'swap_1m': 'XAUSR1M Curncy',
    'swap_2m': 'XAUSR2M Curncy',
    'swap_3m': 'XAUSR3M Curncy',
    'swap_4m': 'XAUSR4M Curncy',
    'swap_5m': 'XAUSR5M Curncy',
    'swap_6m': 'XAUSR6M Curncy',
    'swap_7m': 'XAUSR7M Curncy',
    'comex_stocks': 'COMXGOLD Index',
}

SILVER_TICKERS = {
    'spot': 'XAG Curncy',
    'futures': 'SIA Comdty',
    'swap_1m': 'XAGSR1M Curncy',
    'swap_2m': 'XAGSR2M Curncy',
    'swap_3m': 'XAGSR3M Curncy',
    'swap_4m': 'XAGSR4M Curncy',
    'swap_5m': 'XAGSR5M Curncy',
    'swap_6m': 'XAGSR6M Curncy',
    'swap_7m': 'XAGSR7M Curncy',
    'comex_stocks': 'COMXSILV Index',
}

# Risk thresholds
GOLD_THRESHOLDS = {
    'level_1': 4.0,   # 50% cover trigger
    'level_2': 6.0,   # 75% cover trigger
    'level_3': 8.0,   # 100% cover trigger
}

SILVER_THRESHOLDS = {
    'level_1': 0.15,  # 15 cents - Call to assess
    'level_2': 0.20,  # 20 cents - 75% cover
    'level_3': 0.25,  # 25 cents - 100% cover
}

# Contract specifications
GOLD_CONTRACT_SIZE = 100  # oz per contract
SILVER_CONTRACT_SIZE = 5000  # oz per contract


# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def get_tickers(metal: str) -> Dict[str, str]:
    """Get ticker dictionary for specified metal."""
    metal = metal.upper()
    if metal == 'GOLD':
        return GOLD_TICKERS
    elif metal == 'SILVER':
        return SILVER_TICKERS
    else:
        raise ValueError(f"Unknown metal: {metal}. Use 'GOLD' or 'SILVER'.")


def fetch_spot_price(metal: str, date: Optional[str] = None) -> float:
    """
    Fetch spot price for Gold or Silver.

    Parameters:
        metal: 'GOLD' or 'SILVER'
        date: Optional date string (YYYY-MM-DD). If None, uses latest.

    Returns:
        Spot price as float
    """
    tickers = get_tickers(metal)
    ticker = tickers['spot']

    if date:
        request = bql.Request(ticker, bql.data.px_last(dates=date))
    else:
        request = bql.Request(ticker, bql.data.px_last())

    response = bq.execute(request)
    df = bql.combined_df(response)
    return df['px_last'].iloc[0]


def fetch_futures_price(metal: str, date: Optional[str] = None) -> float:
    """
    Fetch active futures contract price for Gold or Silver.

    Parameters:
        metal: 'GOLD' or 'SILVER'
        date: Optional date string (YYYY-MM-DD). If None, uses latest.

    Returns:
        Futures price as float
    """
    tickers = get_tickers(metal)
    ticker = tickers['futures']

    if date:
        request = bql.Request(ticker, bql.data.px_last(dates=date))
    else:
        request = bql.Request(ticker, bql.data.px_last())

    response = bq.execute(request)
    df = bql.combined_df(response)
    return df['px_last'].iloc[0]


def fetch_swap_rate(metal: str, tenor_months: int, date: Optional[str] = None) -> float:
    """
    Fetch swap rate for specified metal and tenor.

    Parameters:
        metal: 'GOLD' or 'SILVER'
        tenor_months: 1 to 7 months
        date: Optional date string (YYYY-MM-DD). If None, uses latest.

    Returns:
        Swap rate as decimal (e.g., 0.045 for 4.5%)
    """
    if tenor_months < 1 or tenor_months > 7:
        raise ValueError("Tenor must be between 1 and 7 months")

    tickers = get_tickers(metal)
    ticker = tickers[f'swap_{tenor_months}m']

    if date:
        request = bql.Request(ticker, bql.data.px_last(dates=date))
    else:
        request = bql.Request(ticker, bql.data.px_last())

    response = bq.execute(request)
    df = bql.combined_df(response)
    # Swap rates are typically quoted in percentage, convert to decimal
    rate = df['px_last'].iloc[0]
    return rate / 100 if rate > 1 else rate


def fetch_first_delivery_date(metal: str) -> datetime:
    """
    Fetch First Delivery Date (FDD) for Gold or First Notice Date (FND) for Silver.

    Parameters:
        metal: 'GOLD' or 'SILVER'

    Returns:
        First delivery/notice date as datetime
    """
    tickers = get_tickers(metal)
    ticker = tickers['futures']

    if metal.upper() == 'GOLD':
        request = bql.Request(ticker, bql.data.fut_dlv_dt_first())
        response = bq.execute(request)
        df = bql.combined_df(response)
        return pd.to_datetime(df['fut_dlv_dt_first'].iloc[0])
    else:
        request = bql.Request(ticker, bql.data.fut_notice_first())
        response = bq.execute(request)
        df = bql.combined_df(response)
        return pd.to_datetime(df['fut_notice_first'].iloc[0])


def fetch_comex_stocks(metal: str, date: Optional[str] = None) -> float:
    """
    Fetch COMEX warehouse stocks (in ounces).

    Parameters:
        metal: 'GOLD' or 'SILVER'
        date: Optional date string (YYYY-MM-DD). If None, uses latest.

    Returns:
        COMEX stocks in ounces
    """
    tickers = get_tickers(metal)
    ticker = tickers['comex_stocks']

    if date:
        request = bql.Request(ticker, bql.data.px_last(dates=date))
    else:
        request = bql.Request(ticker, bql.data.px_last())

    response = bq.execute(request)
    df = bql.combined_df(response)
    return df['px_last'].iloc[0]


def fetch_open_interest(metal: str, date: Optional[str] = None) -> int:
    """
    Fetch aggregate open interest for futures contract.

    Parameters:
        metal: 'GOLD' or 'SILVER'
        date: Optional date string (YYYY-MM-DD). If None, uses latest.

    Returns:
        Open interest in number of contracts
    """
    tickers = get_tickers(metal)
    ticker = tickers['futures']

    if date:
        request = bql.Request(ticker, bql.data.fut_aggte_open_int(dates=date))
    else:
        request = bql.Request(ticker, bql.data.fut_aggte_open_int())

    response = bq.execute(request)
    df = bql.combined_df(response)
    return int(df['fut_aggte_open_int'].iloc[0])


def fetch_active_contract_name(metal: str) -> str:
    """
    Get the name/ticker of the current active contract.

    Parameters:
        metal: 'GOLD' or 'SILVER'

    Returns:
        Contract name string (e.g., 'GCG5' for Gold Feb 2025)
    """
    tickers = get_tickers(metal)
    ticker = tickers['futures']

    request = bql.Request(ticker, bql.data.name())
    response = bq.execute(request)
    df = bql.combined_df(response)
    return df['name'].iloc[0]


# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def calculate_spot_date(trade_date: datetime) -> datetime:
    """
    Calculate spot date (T+2 business days).

    Parameters:
        trade_date: The trade date

    Returns:
        Spot date (T+2)
    """
    spot_date = trade_date
    business_days_added = 0
    while business_days_added < 2:
        spot_date += timedelta(days=1)
        if spot_date.weekday() < 5:  # Monday = 0, Friday = 4
            business_days_added += 1
    return spot_date


def calculate_business_days(start_date: datetime, end_date: datetime) -> int:
    """
    Calculate number of business days between two dates.

    Parameters:
        start_date: Start date
        end_date: End date

    Returns:
        Number of business days
    """
    days = 0
    current = start_date
    while current < end_date:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            days += 1
    return days


def get_appropriate_swap_tenor(days_to_fdd: int) -> int:
    """
    Determine the appropriate swap rate tenor based on days to FDD.

    Parameters:
        days_to_fdd: Number of business days to first delivery date

    Returns:
        Swap tenor in months (1-7)
    """
    if days_to_fdd <= 0:
        return 1
    tenor = (days_to_fdd // 30) + 1
    return min(7, max(1, tenor))


def calculate_swap_implied_efp(spot: float, swap_rate: float, days_to_fdd: int) -> float:
    """
    Calculate the swap-implied EFP.

    Formula: Spot × Swap Rate × (Days to FDD / 360)

    Parameters:
        spot: Spot price
        swap_rate: Annualized swap rate as decimal
        days_to_fdd: Number of days to first delivery date

    Returns:
        Swap-implied EFP value
    """
    return spot * swap_rate * (days_to_fdd / 360)


def calculate_traded_efp(futures_price: float, spot_price: float) -> float:
    """
    Calculate the traded EFP.

    Formula: Closing Futures - Closing Spot

    Parameters:
        futures_price: Futures closing price
        spot_price: Spot closing price

    Returns:
        Traded EFP value
    """
    return futures_price - spot_price


def calculate_premium_discount(traded_efp: float, swap_implied_efp: float) -> float:
    """
    Calculate the premium or discount.

    Formula: Traded EFP - Swap Implied EFP

    Parameters:
        traded_efp: Actual traded EFP
        swap_implied_efp: Theoretical swap-implied EFP

    Returns:
        Premium (positive) or discount (negative)
    """
    return traded_efp - swap_implied_efp


def calculate_coverage_ratio(comex_stocks: float, open_interest: int, metal: str) -> float:
    """
    Calculate COMEX coverage ratio.

    Formula: COMEX Stocks (oz) / (Open Interest × Contract Size)

    Parameters:
        comex_stocks: COMEX warehouse stocks in ounces
        open_interest: Number of open contracts
        metal: 'GOLD' or 'SILVER' (determines contract size)

    Returns:
        Coverage ratio as decimal
    """
    contract_size = GOLD_CONTRACT_SIZE if metal.upper() == 'GOLD' else SILVER_CONTRACT_SIZE
    oi_ounces = open_interest * contract_size
    if oi_ounces == 0:
        return 0
    return comex_stocks / oi_ounces


def get_threshold_status(premium_discount: float, metal: str) -> str:
    """
    Determine risk status based on premium/discount thresholds.

    Parameters:
        premium_discount: The premium/discount value
        metal: 'GOLD' or 'SILVER'

    Returns:
        Status string indicating risk level
    """
    thresholds = GOLD_THRESHOLDS if metal.upper() == 'GOLD' else SILVER_THRESHOLDS

    abs_value = abs(premium_discount)

    if abs_value >= thresholds['level_3']:
        return f"LEVEL 3 - 100% cover (>= ${thresholds['level_3']:.2f})"
    elif abs_value >= thresholds['level_2']:
        return f"LEVEL 2 - 75% cover (>= ${thresholds['level_2']:.2f})"
    elif abs_value >= thresholds['level_1']:
        return f"LEVEL 1 - 50% cover (>= ${thresholds['level_1']:.2f})"
    else:
        return "Within normal range"


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_daily_efp_report(date: Optional[str] = None) -> Dict:
    """
    Generate comprehensive daily EFP report for both Gold and Silver.

    Parameters:
        date: Optional date string (YYYY-MM-DD). If None, uses latest data.

    Returns:
        Dictionary with all EFP metrics for both metals
    """
    report = {
        'date': date or datetime.now().strftime('%Y-%m-%d'),
        'gold': {},
        'silver': {}
    }

    for metal in ['GOLD', 'SILVER']:
        try:
            # Fetch prices
            spot = fetch_spot_price(metal, date)
            futures = fetch_futures_price(metal, date)

            # Fetch delivery date
            fdd = fetch_first_delivery_date(metal)

            # Calculate dates
            trade_date = datetime.strptime(date, '%Y-%m-%d') if date else datetime.now()
            spot_date = calculate_spot_date(trade_date)
            days_to_fdd = calculate_business_days(spot_date, fdd)

            # Get appropriate swap rate
            tenor = get_appropriate_swap_tenor(days_to_fdd)
            swap_rate = fetch_swap_rate(metal, tenor, date)

            # Calculate EFP metrics
            swap_implied_efp = calculate_swap_implied_efp(spot, swap_rate, days_to_fdd)
            traded_efp = calculate_traded_efp(futures, spot)
            premium_discount = calculate_premium_discount(traded_efp, swap_implied_efp)

            # Fetch COMEX data
            comex_stocks = fetch_comex_stocks(metal, date)
            open_interest = fetch_open_interest(metal, date)
            coverage_ratio = calculate_coverage_ratio(comex_stocks, open_interest, metal)

            # Get contract name
            contract_name = fetch_active_contract_name(metal)

            # Get threshold status
            status = get_threshold_status(premium_discount, metal)

            # Store results
            metal_key = metal.lower()
            report[metal_key] = {
                'spot_price': spot,
                'futures_price': futures,
                'active_contract': contract_name,
                'first_delivery_date': fdd.strftime('%Y-%m-%d'),
                'spot_date': spot_date.strftime('%Y-%m-%d'),
                'days_to_fdd': days_to_fdd,
                'swap_rate': swap_rate,
                'swap_tenor_months': tenor,
                'swap_implied_efp': swap_implied_efp,
                'traded_efp': traded_efp,
                'premium_discount': premium_discount,
                'comex_stocks': comex_stocks,
                'open_interest': open_interest,
                'open_interest_oz': open_interest * (GOLD_CONTRACT_SIZE if metal == 'GOLD' else SILVER_CONTRACT_SIZE),
                'coverage_ratio': coverage_ratio,
                'risk_status': status
            }

        except Exception as e:
            report[metal.lower()] = {'error': str(e)}

    return report


def generate_historical_efp_data(
    metal: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Generate historical EFP data for charting.

    Parameters:
        metal: 'GOLD' or 'SILVER'
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with daily EFP metrics
    """
    tickers = get_tickers(metal)

    # Fetch historical spot and futures prices
    date_range = bql.func.range(start_date, end_date)

    spot_request = bql.Request(tickers['spot'], bql.data.px_last(dates=date_range))
    futures_request = bql.Request(tickers['futures'], bql.data.px_last(dates=date_range))

    spot_response = bq.execute(spot_request)
    futures_response = bq.execute(futures_request)

    spot_df = bql.combined_df(spot_response).reset_index()
    futures_df = bql.combined_df(futures_response).reset_index()

    # Merge data
    df = pd.merge(
        spot_df[['DATE', 'px_last']].rename(columns={'px_last': 'spot'}),
        futures_df[['DATE', 'px_last']].rename(columns={'px_last': 'futures'}),
        on='DATE'
    )

    # Calculate traded EFP
    df['traded_efp'] = df['futures'] - df['spot']

    # Fetch swap rates (using 1M as default for historical)
    swap_request = bql.Request(tickers['swap_1m'], bql.data.px_last(dates=date_range))
    swap_response = bq.execute(swap_request)
    swap_df = bql.combined_df(swap_response).reset_index()

    df = pd.merge(
        df,
        swap_df[['DATE', 'px_last']].rename(columns={'px_last': 'swap_rate'}),
        on='DATE',
        how='left'
    )

    # Convert swap rate to decimal
    df['swap_rate'] = df['swap_rate'].apply(lambda x: x / 100 if x > 1 else x)

    # Calculate swap implied EFP (using 30 days as default estimate)
    df['swap_implied_efp'] = df['spot'] * df['swap_rate'] * (30 / 360)

    # Calculate premium/discount
    df['premium_discount'] = df['traded_efp'] - df['swap_implied_efp']

    # Add thresholds
    thresholds = GOLD_THRESHOLDS if metal.upper() == 'GOLD' else SILVER_THRESHOLDS
    df['threshold_1'] = thresholds['level_1']
    df['threshold_2'] = thresholds['level_2']
    df['threshold_3'] = thresholds['level_3']

    df = df.rename(columns={'DATE': 'date'})
    df = df.set_index('date')

    return df


def generate_comex_inventory_history(
    metal: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Generate historical COMEX inventory data.

    Parameters:
        metal: 'GOLD' or 'SILVER'
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with COMEX stocks and open interest history
    """
    tickers = get_tickers(metal)
    date_range = bql.func.range(start_date, end_date)

    # Fetch COMEX stocks
    stocks_request = bql.Request(tickers['comex_stocks'], bql.data.px_last(dates=date_range))
    stocks_response = bq.execute(stocks_request)
    stocks_df = bql.combined_df(stocks_response).reset_index()

    # Fetch open interest
    oi_request = bql.Request(tickers['futures'], bql.data.fut_aggte_open_int(dates=date_range))
    oi_response = bq.execute(oi_request)
    oi_df = bql.combined_df(oi_response).reset_index()

    # Merge data
    df = pd.merge(
        stocks_df[['DATE', 'px_last']].rename(columns={'px_last': 'comex_stocks'}),
        oi_df[['DATE', 'fut_aggte_open_int']].rename(columns={'fut_aggte_open_int': 'open_interest'}),
        on='DATE',
        how='outer'
    )

    # Calculate open interest in ounces
    contract_size = GOLD_CONTRACT_SIZE if metal.upper() == 'GOLD' else SILVER_CONTRACT_SIZE
    df['oi_oz'] = df['open_interest'] * contract_size

    # Calculate coverage ratio
    df['coverage_ratio'] = df['comex_stocks'] / df['oi_oz']

    df = df.rename(columns={'DATE': 'date'})
    df = df.set_index('date')

    return df


def print_daily_report(report: Dict) -> None:
    """
    Print a formatted daily EFP report.

    Parameters:
        report: Report dictionary from generate_daily_efp_report()
    """
    print("=" * 80)
    print(f"                         EFP DAILY REPORT - {report['date']}")
    print("=" * 80)

    for metal in ['gold', 'silver']:
        data = report[metal]
        print(f"\n{metal.upper()}")
        print("-" * 40)

        if 'error' in data:
            print(f"  Error: {data['error']}")
            continue

        print(f"  Spot Price:           ${data['spot_price']:,.2f}")
        print(f"  Futures ({data['active_contract']}):  ${data['futures_price']:,.2f}")
        print(f"  First Delivery Date:  {data['first_delivery_date']}")
        print(f"  Days to FDD:          {data['days_to_fdd']}")
        print(f"  Swap Rate ({data['swap_tenor_months']}M):       {data['swap_rate']*100:.2f}%")
        print(f"  Swap Implied EFP:     ${data['swap_implied_efp']:.2f}")
        print(f"  Traded EFP:           ${data['traded_efp']:.2f}")

        prem_disc = data['premium_discount']
        prem_disc_label = "PREMIUM" if prem_disc > 0 else "DISCOUNT" if prem_disc < 0 else "FAIR VALUE"
        print(f"  Premium/Discount:     ${prem_disc:.2f} ({prem_disc_label})")

        print()
        print(f"  COMEX Stocks:         {data['comex_stocks']:,.0f} oz")
        print(f"  Open Interest:        {data['open_interest']:,} contracts ({data['open_interest_oz']:,.0f} oz)")
        print(f"  Coverage Ratio:       {data['coverage_ratio']*100:.1f}%")
        print()
        print(f"  Risk Status: {data['risk_status']}")

    print()
    print("=" * 80)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_efp_check(metal: str = 'GOLD') -> Tuple[float, float, float]:
    """
    Quick check of current EFP metrics.

    Parameters:
        metal: 'GOLD' or 'SILVER'

    Returns:
        Tuple of (swap_implied_efp, traded_efp, premium_discount)
    """
    spot = fetch_spot_price(metal)
    futures = fetch_futures_price(metal)
    fdd = fetch_first_delivery_date(metal)

    trade_date = datetime.now()
    spot_date = calculate_spot_date(trade_date)
    days_to_fdd = calculate_business_days(spot_date, fdd)

    tenor = get_appropriate_swap_tenor(days_to_fdd)
    swap_rate = fetch_swap_rate(metal, tenor)

    swap_implied = calculate_swap_implied_efp(spot, swap_rate, days_to_fdd)
    traded = calculate_traded_efp(futures, spot)
    premium = calculate_premium_discount(traded, swap_implied)

    return swap_implied, traded, premium


if __name__ == "__main__":
    # Example usage
    report = generate_daily_efp_report()
    print_daily_report(report)
