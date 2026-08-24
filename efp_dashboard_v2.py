# ─────────────────────────────────────────────────────────────
#  CELL 1 — Imports, BQL Service, Styling
# ─────────────────────────────────────────────────────────────

import bql
import os
import shutil
import numbers
import warnings

from datetime import datetime, date

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display, HTML, clear_output

import ipywidgets as widgets


warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
#  Styling
# ─────────────────────────────────────────────────────────────

BG_DARK = '#0d1117'
BG_PANEL = '#161b22'
BG_CARD = '#1c2128'

C_GOLD = '#FFD700'
C_SILVER = '#C0C0C0'
C_PLAT = '#58a6ff'
C_PALL = '#3fb950'

C_RED = '#f85149'
C_GREEN = '#3fb950'
C_AMBER = '#d29922'
C_BLUE = '#58a6ff'
C_MUTED = '#8b949e'
C_TEXT = '#e6edf3'
C_GRID = '#30363d'


METAL_COLORS = {
    'XAU': C_GOLD,
    'XAG': C_SILVER,
    'XPT': C_PLAT,
    'XPD': C_PALL,
}


METAL_NAMES = {
    'XAU': 'Gold',
    'XAG': 'Silver',
    'XPT': 'Platinum',
    'XPD': 'Palladium',
}


METALS = [
    'XAU',
    'XAG',
    'XPT',
    'XPD',
]


CONTRACTS = [
    'C1',
    'C2',
    'C3',
    'C4',
]


plt.rcParams.update({
    'figure.facecolor': BG_DARK,
    'axes.facecolor': BG_PANEL,
    'axes.edgecolor': C_GRID,
    'axes.labelcolor': C_MUTED,
    'xtick.color': C_MUTED,
    'ytick.color': C_MUTED,
    'text.color': C_TEXT,
    'grid.color': C_GRID,
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'lines.linewidth': 1.8,
    'font.family': 'monospace',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.titlecolor': C_TEXT,
    'axes.titleweight': 'bold',
    'figure.dpi': 120,
    'legend.framealpha': 0.3,
    'legend.facecolor': BG_CARD,
    'legend.edgecolor': C_GRID,
})


bq = bql.Service()


print(
    f"✅ BQL Service ready | "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)


# ─────────────────────────────────────────────────────────────
#  CELL 2 — Ticker Configuration
# ─────────────────────────────────────────────────────────────

SPOT_TICKERS = {
    'XAU': 'XAUUSD Curncy',
    'XAG': 'XAGUSD Curncy',
    'XPT': 'XPTUSD Curncy',
    'XPD': 'XPDUSD Curncy',
}


FUTURES_CHAIN = {
    'XAU': {
        'C1': 'GC1 Comdty',
        'C2': 'GC2 Comdty',
        'C3': 'GC3 Comdty',
        'C4': 'GC4 Comdty',
    },

    'XAG': {
        'C1': 'SI1 Comdty',
        'C2': 'SI2 Comdty',
        'C3': 'SI3 Comdty',
        'C4': 'SI4 Comdty',
    },

    'XPT': {
        'C1': 'PL1 Comdty',
        'C2': 'PL2 Comdty',
        'C3': 'PL3 Comdty',
        'C4': 'PL4 Comdty',
    },

    'XPD': {
        'C1': 'PA1 Comdty',
        'C2': 'PA2 Comdty',
        'C3': 'PA3 Comdty',
        'C4': 'PA4 Comdty',
    },
}


print("✅ Ticker configuration loaded")


for metal in METALS:

    ticker_text = ' | '.join(
        f"{contract}: {ticker}"
        for contract, ticker in FUTURES_CHAIN[metal].items()
    )

    print(
        f"  {metal} spot: {SPOT_TICKERS[metal]} "
        f"| futures: {ticker_text}"
    )


# ─────────────────────────────────────────────────────────────
#  CELL 3 — Tenors and SOFR Curve
# ─────────────────────────────────────────────────────────────

TENOR_DAYS = {
    '1W': 7,
    '2W': 14,

    '1M': 30,
    '2M': 60,
    '3M': 90,
    '4M': 120,
    '5M': 150,
    '6M': 180,
    '7M': 210,
    '8M': 240,
    '9M': 270,
    '10M': 300,
    '11M': 330,
    '12M': 360,

    '1Y': 360,
    '18M': 540,
    '2Y': 720,
    '3Y': 1080,
}


SOFR_CURVE_TICKERS = {
    '1W': 'USOSFR1Z Curncy',
    '2W': 'USOSFR2Z Curncy',

    '1M': 'USOSFRA Curncy',
    '2M': 'USOSFRB Curncy',
    '3M': 'USOSFRC Curncy',
    '4M': 'USOSFRD Curncy',
    '5M': 'USOSFRE Curncy',
    '6M': 'USOSFRF Curncy',
    '7M': 'USOSFRG Curncy',
    '8M': 'USOSFRH Curncy',
    '9M': 'USOSFRI Curncy',
    '10M': 'USOSFRJ Curncy',
    '11M': 'USOSFRK Curncy',

    '1Y': 'USOSFR1 Curncy',
    '18M': 'USOSFR1F Curncy',
    '2Y': 'USOSFR2 Curncy',
    '3Y': 'USOSFR3 Curncy',
}


DEFAULT_METAL_FORWARD_CURVES = {
    'XAU': {
        '1M': 1.00,
        '2M': 1.05,
        '3M': 1.10,
        '6M': 1.20,
        '9M': 1.28,
        '12M': 1.35,
    },

    'XAG': {
        '1M': 2.00,
        '2M': 2.10,
        '3M': 2.20,
        '6M': 2.35,
        '9M': 2.42,
        '12M': 2.50,
    },

    'XPT': {
        '1M': 1.50,
        '2M': 1.60,
        '3M': 1.70,
        '6M': 1.85,
        '9M': 1.92,
        '12M': 2.00,
    },

    'XPD': {
        '1M': 1.75,
        '2M': 1.85,
        '3M': 1.95,
        '6M': 2.10,
        '9M': 2.18,
        '12M': 2.25,
    },
}


METAL_FORWARD_TENORS = [
    '1M',
    '2M',
    '3M',
    '6M',
    '9M',
    '12M',
]


# 0.7bp per annum = 0.007%
INSURANCE_RATE_PCT = 0.007


# ─────────────────────────────────────────────────────────────
#  Physical arbitrage assumptions
# ─────────────────────────────────────────────────────────────

FREIGHT_USD_OZ = {
    'XAU': 0.50,
    'XAG': 0.10,
    'XPT': 0.50,
    'XPD': 0.50,
}


TRANSIT_DAYS = {
    'XAU': 3,
    'XAG': 15,
    'XPT': 3,
    'XPD': 3,
}


# Recast operation only.
# Gold freight is already included in FREIGHT_USD_OZ.
RECAST_USD_OZ = {
    'XAU': 0.50,
    'XAG': 0.00,
    'XPT': 0.00,
    'XPD': 0.00,
}


BUFFER_NOISE_MULT = 1.0

NOISE_FLOOR_PP = 0.55

CORROB_WARN_X = 3.0

LEASE_SANITY_MIN_PCT = -5.0

LEASE_SANITY_MAX_PCT = 60.0


# ─────────────────────────────────────────────────────────────
#  Persistence
#
#  BARE RELATIVE FILENAMES, exactly as the original working notebook.
#
#  In BQuant the notebook's working directory IS the persistent cloud
#  storage, so a bare filename lands in the right place and survives a
#  kernel restart. The v2 rewrite put these under an absolute
#  ./pm_efp_data/ subdirectory with migration machinery; that added risk
#  around an existing efp_history.csv for no benefit, so it is reverted.
#
#  Anything this notebook writes must use these constants, never an
#  absolute path and never a local machine path.
# ─────────────────────────────────────────────────────────────

HISTORY_FILE = 'efp_history.csv'

FORWARD_CURVE_LATEST_FILE = 'pm_forward_curves_latest.csv'
FORWARD_CURVE_HISTORY_FILE = 'pm_forward_curves_history.csv'

FUNDING_SETTINGS_LATEST_FILE = 'pm_funding_settings_latest.csv'
FUNDING_SETTINGS_HISTORY_FILE = 'pm_funding_settings_history.csv'

# Files owned by the go-live cell (Cell 12), same convention.
SNAP_FILE = 'pm_intraday_snap.csv'
RECON_FILE = 'pm_reconciliation.csv'
REGIME_FILE = 'pm_regime_state.csv'


print(
    "✅ Persistence: bare filenames in the notebook working directory"
)



# ─────────────────────────────────────────────────────────────
#  General numeric helper functions
# ─────────────────────────────────────────────────────────────

def is_valid_number(value):
    """
    Return True for finite Python or NumPy numeric values.

    Boolean values are excluded.
    """

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        )
    ):
        return False

    if not isinstance(
        value,
        numbers.Number
    ):
        return False

    try:

        return bool(
            pd.notna(value)
            and np.isfinite(float(value))
        )

    except Exception:

        return False


def safe_float(
    value,
    default=np.nan
):
    """
    Convert a value to a finite float.

    Return default if conversion fails or the result is not finite.
    """

    try:

        value = float(value)

        if np.isfinite(value):
            return value

        return default

    except Exception:

        return default


# ─────────────────────────────────────────────────────────────
#  Business-day helper functions
#
#  This is a weekday-only fallback.
#
#  Business-day logic is centralized here so a London/COMEX
#  holiday calendar can be inserted later without rewriting the
#  valuation functions.
# ─────────────────────────────────────────────────────────────

def add_business_days(
    start_date,
    business_days
):
    """
    Add weekday-only business days to a date.
    """

    if start_date is None:
        return None

    try:

        start_timestamp = pd.Timestamp(
            start_date
        )

        if pd.isna(start_timestamp):
            return None

        result = (
            start_timestamp
            + pd.offsets.BDay(
                int(business_days)
            )
        )

        return result.date()

    except Exception:

        return None


def get_spot_value_date(
    as_of_date=None
):
    """
    Spot value date using the dashboard's T+2 convention.
    """

    if as_of_date is None:
        as_of_date = date.today()

    return add_business_days(
        as_of_date,
        2
    )


def days_to_value(
    fnd_date,
    spot_dt
):
    """
    Day-count convention:

        Days = FND + 1 business day - spot value date

    The result is floored at one calendar day.
    """

    if fnd_date is None:
        return None

    if spot_dt is None:
        return None

    fnd_value_date = add_business_days(
        fnd_date,
        1
    )

    if fnd_value_date is None:
        return None

    try:

        day_count = (
            pd.Timestamp(
                fnd_value_date
            ).date()
            -
            pd.Timestamp(
                spot_dt
            ).date()
        ).days

        return max(
            int(day_count),
            1
        )

    except Exception:

        return None


print("✅ Assumptions and helper functions loaded")


# ─────────────────────────────────────────────────────────────
#  CELL 4 — Load Persistent Forward and Funding Settings
# ─────────────────────────────────────────────────────────────

def normalise_forward_curves(curves):
    """
    Return a complete metal/tenor curve dictionary.

    Invalid or missing values are replaced with defaults.
    """

    clean_curves = {}

    for metal in METALS:

        clean_curves[metal] = {}

        for tenor in METAL_FORWARD_TENORS:

            default_value = (
                DEFAULT_METAL_FORWARD_CURVES
                .get(
                    metal,
                    {}
                )
                .get(
                    tenor,
                    np.nan
                )
            )

            raw_value = (
                curves
                .get(
                    metal,
                    {}
                )
                .get(
                    tenor,
                    default_value
                )
            )

            clean_curves[metal][tenor] = safe_float(
                raw_value,
                default=default_value
            )

    return clean_curves


def load_saved_forward_curves():
    """
    Load the latest saved metal-forward inputs.
    """

    if not os.path.exists(
        FORWARD_CURVE_LATEST_FILE
    ):

        return normalise_forward_curves(
            DEFAULT_METAL_FORWARD_CURVES
        )

    try:

        dataframe = pd.read_csv(
            FORWARD_CURVE_LATEST_FILE
        )

        required_columns = {
            'Metal',
            'Tenor',
            'Rate (%)',
        }

        if not required_columns.issubset(
            dataframe.columns
        ):

            raise ValueError(
                "Required columns missing from "
                "forward-curve latest file"
            )

        loaded_curves = {
            metal: {}
            for metal in METALS
        }

        for metal in METALS:

            for tenor in METAL_FORWARD_TENORS:

                matching_rows = dataframe[
                    (
                        dataframe['Metal']
                        == metal
                    )
                    &
                    (
                        dataframe['Tenor']
                        == tenor
                    )
                ]

                if matching_rows.empty:

                    loaded_curves[metal][tenor] = (
                        DEFAULT_METAL_FORWARD_CURVES
                        [metal][tenor]
                    )

                else:

                    loaded_curves[metal][tenor] = (
                        matching_rows
                        .iloc[-1]
                        ['Rate (%)']
                    )

        return normalise_forward_curves(
            loaded_curves
        )

    except Exception as error:

        print(
            f"⚠ Could not load saved forward curves. "
            f"Using defaults. Error: {error}"
        )

        return normalise_forward_curves(
            DEFAULT_METAL_FORWARD_CURVES
        )


def load_saved_funding_settings():
    """
    Load the latest SOFR fallback and funding spread.
    """

    default_settings = {
        'SOFR Fallback (%)': 4.30,
        'SOFR Spread (%)': 0.00,
    }

    if not os.path.exists(
        FUNDING_SETTINGS_LATEST_FILE
    ):

        return default_settings

    try:

        dataframe = pd.read_csv(
            FUNDING_SETTINGS_LATEST_FILE
        )

        if dataframe.empty:
            return default_settings

        latest_row = dataframe.iloc[-1]

        return {
            'SOFR Fallback (%)': safe_float(
                latest_row.get(
                    'SOFR Fallback (%)',
                    default_settings[
                        'SOFR Fallback (%)'
                    ]
                ),
                default_settings[
                    'SOFR Fallback (%)'
                ]
            ),

            'SOFR Spread (%)': safe_float(
                latest_row.get(
                    'SOFR Spread (%)',
                    default_settings[
                        'SOFR Spread (%)'
                    ]
                ),
                default_settings[
                    'SOFR Spread (%)'
                ]
            ),
        }

    except Exception as error:

        print(
            f"⚠ Could not load funding settings. "
            f"Using defaults. Error: {error}"
        )

        return default_settings


SAVED_METAL_FORWARD_CURVES = (
    load_saved_forward_curves()
)


SAVED_FUNDING_SETTINGS = (
    load_saved_funding_settings()
)


# ─────────────────────────────────────────────────────────────
#  Forward and funding widgets
# ─────────────────────────────────────────────────────────────

SOFR_FALLBACK_WIDGET = widgets.FloatText(
    value=(
        SAVED_FUNDING_SETTINGS[
            'SOFR Fallback (%)'
        ]
    ),
    description='SOFR Fallback %',
    layout=widgets.Layout(
        width='220px'
    )
)


SOFR_SPREAD_WIDGET = widgets.FloatText(
    value=(
        SAVED_FUNDING_SETTINGS[
            'SOFR Spread (%)'
        ]
    ),
    description='SOFR Spread %',
    layout=widgets.Layout(
        width='220px'
    )
)


FWD_INPUTS = {}


for metal in METALS:

    FWD_INPUTS[metal] = {}

    for tenor in METAL_FORWARD_TENORS:

        FWD_INPUTS[metal][tenor] = (
            widgets.FloatText(
                value=(
                    SAVED_METAL_FORWARD_CURVES
                    [metal][tenor]
                ),
                description=(
                    f'{metal} {tenor}'
                ),
                layout=widgets.Layout(
                    width='145px'
                )
            )
        )


AUTO_SAVE_ENABLED = False


def get_current_forward_curves():
    """
    Read current metal-forward widget values.
    """

    current_curves = {}

    for metal in METALS:

        current_curves[metal] = {}

        for tenor in METAL_FORWARD_TENORS:

            current_curves[metal][tenor] = (
                float(
                    FWD_INPUTS
                    [metal][tenor]
                    .value
                )
            )

    return normalise_forward_curves(
        current_curves
    )


def get_sofr_spread():
    """
    Return the current funding spread in percentage points.
    """

    return float(
        SOFR_SPREAD_WIDGET.value
    )


def save_forward_curves(
    verbose=True
):
    """
    Save the current metal-forward and funding inputs.

    Latest files are overwritten.

    History files append one timestamped snapshot and remove exact
    timestamp/metal/tenor duplicates.
    """

    curves = get_current_forward_curves()

    timestamp = datetime.now().replace(
        second=0,
        microsecond=0
    )

    curve_rows = []

    for metal in METALS:

        for tenor in METAL_FORWARD_TENORS:

            curve_rows.append({
                'Timestamp': timestamp,
                'Date': timestamp.date(),
                'Metal': metal,
                'Metal Name': (
                    METAL_NAMES.get(
                        metal,
                        metal
                    )
                ),
                'Tenor': tenor,
                'Days': TENOR_DAYS.get(
                    tenor,
                    np.nan
                ),
                'Rate (%)': (
                    curves[metal][tenor]
                ),
            })

    latest_curve_dataframe = pd.DataFrame(
        curve_rows
    )

    latest_curve_dataframe.to_csv(
        FORWARD_CURVE_LATEST_FILE,
        index=False
    )

    try:

        old_curve_history = pd.read_csv(
            FORWARD_CURVE_HISTORY_FILE
        )

    except Exception:

        old_curve_history = pd.DataFrame()

    curve_history = pd.concat(
        [
            old_curve_history,
            latest_curve_dataframe,
        ],
        ignore_index=True
    )

    curve_history = (
        curve_history
        .drop_duplicates(
            subset=[
                'Timestamp',
                'Metal',
                'Tenor',
            ],
            keep='last'
        )
    )

    curve_history.to_csv(
        FORWARD_CURVE_HISTORY_FILE,
        index=False
    )

    funding_settings_row = pd.DataFrame([
        {
            'Timestamp': timestamp,
            'Date': timestamp.date(),

            'SOFR Fallback (%)': float(
                SOFR_FALLBACK_WIDGET.value
            ),

            'SOFR Spread (%)': float(
                SOFR_SPREAD_WIDGET.value
            ),

            'Insurance (%)': (
                INSURANCE_RATE_PCT
            ),
        }
    ])

    funding_settings_row.to_csv(
        FUNDING_SETTINGS_LATEST_FILE,
        index=False
    )

    try:

        old_funding_history = pd.read_csv(
            FUNDING_SETTINGS_HISTORY_FILE
        )

    except Exception:

        old_funding_history = pd.DataFrame()

    funding_history = pd.concat(
        [
            old_funding_history,
            funding_settings_row,
        ],
        ignore_index=True
    )

    funding_history = (
        funding_history
        .drop_duplicates(
            subset=[
                'Timestamp',
            ],
            keep='last'
        )
    )

    funding_history.to_csv(
        FUNDING_SETTINGS_HISTORY_FILE,
        index=False
    )

    if verbose:

        print(
            f"✅ Forward curves saved:\n"
            f"   {FORWARD_CURVE_LATEST_FILE}"
        )

        print(
            f"✅ Forward history updated:\n"
            f"   {FORWARD_CURVE_HISTORY_FILE}"
        )

        print(
            f"✅ Funding settings saved:\n"
            f"   {FUNDING_SETTINGS_LATEST_FILE}"
        )

        print(
            f"✅ Funding history updated:\n"
            f"   {FUNDING_SETTINGS_HISTORY_FILE}"
        )


def auto_save(change):
    """
    Save changed widget values and expose any write failure.
    """

    global AUTO_SAVE_ENABLED

    if not AUTO_SAVE_ENABLED:
        return

    try:

        save_forward_curves(
            verbose=False
        )

        if 'status_label' in globals():

            status_label.value = (
                f"✅ Inputs auto-saved at "
                f"{datetime.now().strftime('%H:%M:%S')}"
            )

    except Exception as error:

        error_message = (
            f"Auto-save failed: {error}"
        )

        if 'status_label' in globals():

            status_label.value = (
                f"❌ {error_message}"
            )

        print(
            f"⚠ {error_message}"
        )


for metal in METALS:

    for tenor in METAL_FORWARD_TENORS:

        FWD_INPUTS[metal][tenor].observe(
            auto_save,
            names='value'
        )


SOFR_SPREAD_WIDGET.observe(
    auto_save,
    names='value'
)


SOFR_FALLBACK_WIDGET.observe(
    auto_save,
    names='value'
)


AUTO_SAVE_ENABLED = True


def display_forward_inputs():
    """
    Display the persistent forward and funding input controls.
    """

    rows = []

    rows.append(
        widgets.HTML(
            value=f"""
            <div style="
                background-color:{BG_CARD};
                color:{C_TEXT};
                padding:10px;
                margin:10px 0;
            ">
                <b>Manual Inputs</b><br>

                Metal forward rates are annualised percentage
                inputs for 1M, 2M, 3M, 6M, 9M and 12M.<br>

                SOFR is pulled from the Bloomberg USOSFR OIS
                curve tickers.<br>

                <b>SOFR Spread</b> is used only in the Cash
                &amp; Carry monitor.<br>

                Fallback SOFR is used only if the Bloomberg
                SOFR curve cannot be fetched.<br><br>

                <b>EFP RV</b> = EFP Mid - OTC<br>

                <b>OTC</b> = Spot x Metal Forward x Days / 360<br>

                <b>Cash &amp; Carry Funding Cost</b> =
                SOFR + SOFR Spread<br>

                <b>Insurance</b> =
                {INSURANCE_RATE_PCT:.3f}% p.a.<br><br>

                <b>Persistence</b><br>
                {HISTORY_FILE} in the notebook working directory
            </div>
            """
        )
    )

    rows.append(
        widgets.HBox([
            SOFR_FALLBACK_WIDGET,
            SOFR_SPREAD_WIDGET,
        ])
    )

    for metal in METALS:

        rows.append(
            widgets.HTML(
                value=(
                    f"<div style='"
                    f"color:{METAL_COLORS[metal]};"
                    f"font-weight:bold;"
                    f"margin-top:8px;"
                    f"'>"
                    f"{METAL_NAMES[metal]} "
                    f"({metal}) Forward Curve"
                    f"</div>"
                )
            )
        )

        rows.append(
            widgets.HBox([
                FWD_INPUTS[metal][tenor]
                for tenor
                in METAL_FORWARD_TENORS
            ])
        )

    display(
        widgets.VBox(
            rows
        )
    )


print(
    "✅ Persistent metal-forward and funding "
    "input configuration loaded"
)


# ─────────────────────────────────────────────────────────────
#  CELL 5 — Market Data Helper Functions
# ─────────────────────────────────────────────────────────────

def fetch_bid_ask(tickers):

    if isinstance(tickers, str):
        tickers = [tickers]

    result = {
        t: {
            'bid': np.nan,
            'ask': np.nan,
            'mid': np.nan,
        }
        for t in tickers
    }

    try:

        ticker_string = ', '.join(
            f"'{t}'"
            for t in tickers
        )

        bid_response = bq.execute(
            f"get(px_bid) for([{ticker_string}])"
        )

        ask_response = bq.execute(
            f"get(px_ask) for([{ticker_string}])"
        )

        bid_df = bid_response[0].df()
        ask_df = ask_response[0].df()

        for ticker in tickers:

            try:

                bid = bid_df.loc[
                    ticker,
                    'px_bid'
                ]

                ask = ask_df.loc[
                    ticker,
                    'px_ask'
                ]

                bid = safe_float(bid)
                ask = safe_float(ask)

                result[ticker] = {
                    'bid': bid,
                    'ask': ask,
                    'mid': (
                        (bid + ask) / 2
                        if (
                            is_valid_number(bid)
                            and
                            is_valid_number(ask)
                        )
                        else np.nan
                    ),
                }

            except Exception:

                pass

    except Exception as error:

        print(
            f"⚠ fetch_bid_ask failed: "
            f"{error}"
        )

    return result


def fetch_last(tickers):

    if isinstance(tickers, str):
        tickers = [tickers]

    result = {
        ticker: np.nan
        for ticker in tickers
    }

    try:

        ticker_string = ', '.join(
            f"'{ticker}'"
            for ticker
            in tickers
        )

        response = bq.execute(
            f"get(px_last) for([{ticker_string}])"
        )

        dataframe = response[0].df()

        for ticker in tickers:

            try:

                result[ticker] = safe_float(
                    dataframe.loc[
                        ticker,
                        'px_last'
                    ]
                )

            except Exception:

                pass

    except Exception as error:

        print(
            f"⚠ fetch_last failed: "
            f"{error}"
        )

    return result


def fetch_fnd(tickers):

    result = {}

    try:

        ticker_string = ', '.join(
            f"'{ticker}'"
            for ticker
            in tickers
        )

        response = bq.execute(
            f"get(fut_notice_first) "
            f"for([{ticker_string}])"
        )

        dataframe = response[0].df()

        for ticker in tickers:

            try:

                value = dataframe.loc[
                    ticker,
                    'fut_notice_first'
                ]

                result[ticker] = (
                    pd.Timestamp(value).date()
                    if pd.notna(value)
                    else None
                )

            except Exception:

                result[ticker] = None

    except Exception as error:

        print(
            f"⚠ fetch_fnd failed: "
            f"{error}"
        )

    return result


def fetch_fut_month_yr(tickers):

    result = {}

    try:

        ticker_string = ', '.join(
            f"'{ticker}'"
            for ticker
            in tickers
        )

        response = bq.execute(
            f"get(fut_month_yr) "
            f"for([{ticker_string}])"
        )

        dataframe = response[0].df()

        for ticker in tickers:

            try:

                result[ticker] = str(
                    dataframe.loc[
                        ticker,
                        'fut_month_yr'
                    ]
                )

            except Exception:

                result[ticker] = '—'

    except Exception as error:

        print(
            f"⚠ fetch_fut_month_yr failed: "
            f"{error}"
        )

        for ticker in tickers:
            result[ticker] = '—'

    return result


print(
    "✅ Market data functions loaded"
)


# ─────────────────────────────────────────────────────────────
#  CELL 6 — SOFR / Forward Curve Functions
# ─────────────────────────────────────────────────────────────

def fetch_sofr_curve():

    px = fetch_last(
        list(
            SOFR_CURVE_TICKERS.values()
        )
    )

    curve = {}

    for tenor, ticker in SOFR_CURVE_TICKERS.items():

        value = px.get(
            ticker,
            np.nan
        )

        if is_valid_number(value):

            curve[tenor] = float(value)

    return curve


def fallback_sofr_curve():

    fallback = float(
        SOFR_FALLBACK_WIDGET.value
    )

    return {
        tenor: fallback
        for tenor
        in SOFR_CURVE_TICKERS
    }


def interpolate_curve_by_days(
    curve,
    days
):

    if not is_valid_number(days):
        return np.nan

    points = []

    for tenor, rate in curve.items():

        if tenor not in TENOR_DAYS:
            continue

        if not is_valid_number(rate):
            continue

        points.append(
            (
                TENOR_DAYS[tenor],
                float(rate)
            )
        )

    if not points:
        return np.nan

    points = sorted(
        points,
        key=lambda x: x[0]
    )

    x = [p[0] for p in points]
    y = [p[1] for p in points]

    return float(
        np.interp(
            float(days),
            x,
            y
        )
    )


current_sofr_curve = {}


def interpolate_sofr_rate(days):

    global current_sofr_curve

    if not current_sofr_curve:
        return np.nan

    return interpolate_curve_by_days(
        current_sofr_curve,
        days
    )


def interpolate_funding_cost_rate(days):

    sofr = interpolate_sofr_rate(
        days
    )

    if not is_valid_number(sofr):
        return np.nan

    return (
        sofr
        +
        get_sofr_spread()
    )


def interpolate_metal_forward_rate(
    metal,
    days
):

    return interpolate_curve_by_days(
        get_current_forward_curves().get(
            metal,
            {}
        ),
        days
    )


print(
    "✅ SOFR curve functions loaded"
)


# ─────────────────────────────────────────────────────────────
#  CELL 7 — OTC / Carry / Bound Functions
# ─────────────────────────────────────────────────────────────

def otc_value_dollar(
    spot_mid,
    metal_fwd_pct,
    days
):

    if not all(
        is_valid_number(v)
        for v in [
            spot_mid,
            metal_fwd_pct,
            days,
        ]
    ):
        return np.nan

    return (
        spot_mid
        *
        metal_fwd_pct
        / 100.0
        *
        days
        / 360.0
    )


def carry_cost_dollar(
    spot_mid,
    funding_cost_pct,
    insurance_pct,
    days
):

    if not all(
        is_valid_number(v)
        for v in [
            spot_mid,
            funding_cost_pct,
            insurance_pct,
            days,
        ]
    ):
        return np.nan

    return (
        spot_mid
        *
        (
            funding_cost_pct
            +
            insurance_pct
        )
        / 100.0
        *
        days
        / 360.0
    )


def derived_lease_pct(
    metal_fwd_pct,
    days
):

    sofr = interpolate_sofr_rate(
        days
    )

    if (
        not is_valid_number(sofr)
        or
        not is_valid_number(metal_fwd_pct)
    ):
        return np.nan

    return (
        sofr
        -
        metal_fwd_pct
    )


def arb_bound_dollar(
    metal,
    spot_mid,
    lease_pct,
    days=None
):

    if (
        not is_valid_number(spot_mid)
        or
        not is_valid_number(lease_pct)
    ):
        return np.nan

    return (
        FREIGHT_USD_OZ[metal]
        +
        spot_mid
        *
        lease_pct
        / 100.0
        *
        TRANSIT_DAYS[metal]
        / 360.0
    )


def noise_buffer_dollar(
    spot_mid,
    days,
    k=None
):

    if (
        not is_valid_number(spot_mid)
        or
        not is_valid_number(days)
    ):
        return np.nan

    if k is None:
        k = BUFFER_NOISE_MULT

    return (
        spot_mid
        *
        (
            k
            *
            NOISE_FLOOR_PP
            / 100.0
        )
        *
        days
        / 360.0
    )


def bound_thresholds(
    metal,
    bound,
    spot_mid=np.nan,
    days=np.nan
):

    if not is_valid_number(bound):
        return (
            np.nan,
            np.nan
        )

    noise_buffer = noise_buffer_dollar(
        spot_mid,
        days
    )

    if not is_valid_number(
        noise_buffer
    ):
        noise_buffer = 0.0

    sell_threshold = (
        bound
        +
        noise_buffer
    )

    buy_threshold = -(
        bound
        +
        noise_buffer
        +
        RECAST_USD_OZ.get(
            metal,
            0.0
        )
    )

    return (
        sell_threshold,
        buy_threshold
    )

# ─────────────────────────────────────────────────────────────
#  CELL 8 — Bound Signal Functions
# ─────────────────────────────────────────────────────────────

def classify_bound(
    metal,
    efp_rv,
    bound,
    spot_mid=np.nan,
    days=np.nan
):
    """
    Positive RV beyond the threshold:

        Futures rich
        Sell EFP

    Negative RV beyond the threshold:

        Futures cheap
        Buy EFP
    """

    if (
        not is_valid_number(efp_rv)
        or
        not is_valid_number(bound)
    ):
        return 'No bound'

    sell_threshold, buy_threshold = (
        bound_thresholds(
            metal,
            bound,
            spot_mid,
            days
        )
    )

    if (
        not is_valid_number(sell_threshold)
        or
        not is_valid_number(buy_threshold)
    ):
        return 'No bound'

    if efp_rv > sell_threshold:
        return 'SELL EFP'

    if efp_rv < buy_threshold:
        return 'BUY EFP'

    return 'Inside bound'


def bound_multiple(
    metal,
    efp_rv,
    bound,
    spot_mid=np.nan,
    days=np.nan
):
    """
    Signed threshold multiple.

    Positive:
        EFP rich
        Sell direction

    Negative:
        EFP cheap
        Buy direction

    Absolute value of 1.0:
        exactly at threshold
    """

    if (
        not is_valid_number(efp_rv)
        or
        not is_valid_number(bound)
    ):
        return np.nan

    sell_threshold, buy_threshold = (
        bound_thresholds(
            metal,
            bound,
            spot_mid,
            days
        )
    )

    if efp_rv >= 0:

        threshold_abs = sell_threshold

    else:

        threshold_abs = abs(
            buy_threshold
        )

    if (
        not is_valid_number(
            threshold_abs
        )
        or
        threshold_abs <= 0
    ):
        return np.nan

    return (
        float(efp_rv)
        /
        float(threshold_abs)
    )


# ─────────────────────────────────────────────────────────────
#  COMEX Forward Construction
# ─────────────────────────────────────────────────────────────

def comex_implied_forward(
    f1,
    f2,
    t1,
    t2
):
    """
    Futures-calendar implied annualised forward:

        F2/F1 =
            (1 + f*T2/360)
            /
            (1 + f*T1/360)
    """

    values = [
        safe_float(f1),
        safe_float(f2),
        safe_float(t1),
        safe_float(t2)
    ]

    if not all(
        is_valid_number(v)
        for v in values
    ):
        return np.nan

    f1, f2, t1, t2 = values

    if (
        f1 <= 0
        or
        f2 <= 0
        or
        t1 <= 0
        or
        t2 <= 0
    ):
        return np.nan

    incremental_days = (
        t2
        -
        t1
    )

    if not (
        20
        <
        incremental_days
        <
        200
    ):
        return np.nan

    denominator = (
        f1 * t2
        -
        f2 * t1
    )

    if abs(denominator) < 1e-12:
        return np.nan

    return (
        360.0
        *
        (
            f2
            -
            f1
        )
        /
        denominator
        *
        100.0
    )


def corroboration_x(
    entered_fwd_pct,
    implied_fwd_pct
):
    """
    Difference measured in noise-floor units.
    """

    if (
        not is_valid_number(
            entered_fwd_pct
        )
        or
        not is_valid_number(
            implied_fwd_pct
        )
    ):
        return np.nan

    if NOISE_FLOOR_PP <= 0:
        return np.nan

    return (
        abs(
            entered_fwd_pct
            -
            implied_fwd_pct
        )
        /
        NOISE_FLOOR_PP
    )


def add_comex_corroboration(
    dataframe
):
    """
    Correct maturity-matched corroboration:

        C2 uses C1-C2
        C3 uses C2-C3
        C4 uses C3-C4

    This replaces the original design where
    one C1-C2 value was applied everywhere.
    """

    dataframe = dataframe.copy()

    dataframe['COMEX Fwd (%)'] = np.nan

    dataframe['Corrob (x)'] = np.nan

    dataframe['Corrob Basis'] = '—'

    pairs = [
        ('C1', 'C2'),
        ('C2', 'C3'),
        ('C3', 'C4'),
    ]

    for near_contract, far_contract in pairs:

        if (
            near_contract
            not in dataframe.index
        ):
            continue

        if (
            far_contract
            not in dataframe.index
        ):
            continue

        implied_forward = (
            comex_implied_forward(
                dataframe.loc[
                    near_contract,
                    'Fut Mid'
                ],
                dataframe.loc[
                    far_contract,
                    'Fut Mid'
                ],
                dataframe.loc[
                    near_contract,
                    'Days'
                ],
                dataframe.loc[
                    far_contract,
                    'Days'
                ]
            )
        )

        dataframe.loc[
            far_contract,
            'COMEX Fwd (%)'
        ] = implied_forward

        dataframe.loc[
            far_contract,
            'Corrob (x)'
        ] = corroboration_x(
            dataframe.loc[
                far_contract,
                'Metal Fwd (%)'
            ],
            implied_forward
        )

        dataframe.loc[
            far_contract,
            'Corrob Basis'
        ] = (
            f'{near_contract}-{far_contract}'
        )

    return dataframe


print(
    "✅ Bound and corroboration functions loaded"
)


# ─────────────────────────────────────────────────────────────
#  CELL 9 — Colour Functions
# ─────────────────────────────────────────────────────────────

def bound_signal_color(signal):

    if signal == 'SELL EFP':
        return C_RED

    if signal == 'BUY EFP':
        return C_GREEN

    return C_MUTED


def signal_color(signal):

    if signal in [
        'Very Rich',
        'Rich'
    ]:
        return C_RED

    if signal in [
        'Very Cheap',
        'Cheap'
    ]:
        return C_GREEN

    return C_MUTED


def classify_rich_cheap(z_score):

    if not is_valid_number(
        z_score
    ):
        return 'No history'

    if z_score >= 2:
        return 'Very Rich'

    if z_score >= 1:
        return 'Rich'

    if z_score <= -2:
        return 'Very Cheap'

    if z_score <= -1:
        return 'Cheap'

    return 'Fair'


def _cell_color_rv(value):

    if not is_valid_number(value):
        return ''

    return (
        f'color:{C_RED}'
        if value > 0
        else
        f'color:{C_GREEN}'
    )


def _cell_color_z(value):

    if not is_valid_number(value):
        return ''

    if value >= 1:
        return f'color:{C_RED}'

    if value <= -1:
        return f'color:{C_GREEN}'

    return f'color:{C_MUTED}'


def _cell_color_bound_x(value):

    if not is_valid_number(value):
        return ''

    return (
        f'color:{C_AMBER}'
        if abs(value) > 1
        else
        f'color:{C_MUTED}'
    )


def _cell_color_corrob(value):

    if not is_valid_number(value):
        return ''

    return (
        f'color:{C_GREEN}'
        if value >= CORROB_WARN_X
        else
        f'color:{C_AMBER}'
    )


print(
    "✅ Styling functions loaded"
)


# ─────────────────────────────────────────────────────────────
#  CELL 10 — Historical Analytics
# ─────────────────────────────────────────────────────────────

def load_efp_history():

    try:

        history = pd.read_csv(
            HISTORY_FILE
        )

        if 'Timestamp' in history.columns:

            history['Timestamp'] = (
                pd.to_datetime(
                    history['Timestamp']
                )
            )

        return history

    except Exception:

        return pd.DataFrame()


def save_efp_history(
    new_rows
):

    new_dataframe = pd.DataFrame(
        new_rows
    )

    if new_dataframe.empty:
        return

    history = pd.concat(
        [
            load_efp_history(),
            new_dataframe
        ],
        ignore_index=True
    )

    dedupe_columns = [
        column
        for column in [
            'Timestamp',
            'Metal',
            'Contract',
            'Month/Year'
        ]
        if column in history.columns
    ]

    if dedupe_columns:

        history = (
            history
            .drop_duplicates(
                subset=dedupe_columns,
                keep='last'
            )
        )

    history.to_csv(
        HISTORY_FILE,
        index=False
    )


def get_history_stats(
    history,
    metal,
    contract,
    current_value,
    value_col='EFP RV ($/oz)',
    month_yr=None
):

    output = {

        'Stat Basis': 'none',

        'Hist Obs': np.nan,

        '1D Δ ($)': np.nan,
        '5D Δ ($)': np.nan,
        '20D Δ ($)': np.nan,

        'Hist Mean ($)': np.nan,
        'Hist Std ($)': np.nan,

        'Z-Score': np.nan,

        'Percentile': np.nan,

        'Signal': 'No history',
    }

    if (
        history is None
        or
        history.empty
    ):
        return output

    required_columns = [
        'Timestamp',
        'Metal',
        'Contract',
        value_col
    ]

    if not all(
        column in history.columns
        for column
        in required_columns
    ):
        return output

    if (
        month_yr is not None
        and
        'Month/Year'
        in history.columns
    ):

        subset = history[
            (
                history['Metal']
                ==
                metal
            )
            &
            (
                history['Month/Year']
                ==
                month_yr
            )
        ].copy()

        if len(subset) < 20:

            subset = history[
                (
                    history['Metal']
                    ==
                    metal
                )
                &
                (
                    history['Contract']
                    ==
                    contract
                )
            ].copy()

            output[
                'Stat Basis'
            ] = (
                'slot '
                '(thin contract history)'
            )

        else:

            output[
                'Stat Basis'
            ] = 'contract'

    else:

        subset = history[
            (
                history['Metal']
                ==
                metal
            )
            &
            (
                history['Contract']
                ==
                contract
            )
        ].copy()

        output[
            'Stat Basis'
        ] = 'slot'

    if (
        subset.empty
        or
        not is_valid_number(
            current_value
        )
    ):
        return output

    subset['Timestamp'] = (
        pd.to_datetime(
            subset['Timestamp']
        )
    )

    subset['Date'] = (
        subset['Timestamp']
        .dt.date
    )

    subset = subset.sort_values(
        'Timestamp'
    )

    daily = (
        subset
        .groupby(
            'Date',
            as_index=False
        )
        .tail(1)
    )

    series = (
        pd.to_numeric(
            daily[value_col],
            errors='coerce'
        )
        .dropna()
    )

    if series.empty:
        # Return the OUTPUT DICT, not None. Every other exit path returns
        # `output` and the caller does `stats.items()`; a bare return raises
        # AttributeError on the SECOND refresh, once a history file exists.
        return output

    output['Hist Obs'] = int(len(series))

    if len(series) >= 1:
        output['1D Δ ($)'] = current_value - series.iloc[-1]

    if len(series) >= 5:
        output['5D Δ ($)'] = current_value - series.iloc[-5]

    if len(series) >= 20:
        output['20D Δ ($)'] = current_value - series.iloc[-20]

    window = series.tail(252)

    mean_value = window.mean()
    std_value = window.std(ddof=1)

    output['Hist Mean ($)'] = mean_value
    output['Hist Std ($)'] = std_value

    if is_valid_number(std_value) and std_value != 0:
        z_score = (current_value - mean_value) / std_value
        output['Z-Score'] = z_score
        output['Signal'] = classify_rich_cheap(z_score)

    output['Percentile'] = float(
        (window <= current_value).mean() * 100.0
    )

    return output


print(
    "✅ Historical statistics loaded"
)


# ─────────────────────────────────────────────────────────────
#  CELL 11 — EFP RV Opportunity Table Builder
# ─────────────────────────────────────────────────────────────

def build_efp_rv_opportunity_table(
    efp_data
):
    """
    Build the cross-metal EFP RV opportunity table.

    EFP RV is pure paper relative value:

        EFP RV = EFP Mid - OTC

    Ranking is by absolute Bound x.

    A positive Bound x is the sell-EFP direction.
    A negative Bound x is the buy-EFP direction.
    """

    rows = []

    for metal in METALS:

        if metal not in efp_data:
            continue

        metal_dataframe = (
            efp_data[metal]
            .copy()
        )

        if metal_dataframe.empty:
            continue

        for contract, row in metal_dataframe.iterrows():

            rows.append({

                'Metal': metal,

                'Name': METAL_NAMES.get(
                    metal,
                    metal
                ),

                'Contract': contract,

                'Month/Year': row.get(
                    'Month/Year'
                ),

                'Days': row.get(
                    'Days'
                ),

                'EFP Bid ($/oz)': row.get(
                    'EFP Bid'
                ),

                'EFP Ask ($/oz)': row.get(
                    'EFP Ask'
                ),

                'EFP Mid ($/oz)': row.get(
                    'EFP Mid'
                ),

                'EFP Spread ($/oz)': row.get(
                    'EFP Spread'
                ),

                'OTC ($/oz)': row.get(
                    'OTC ($/oz)'
                ),

                'EFP RV ($/oz)': row.get(
                    'EFP RV ($/oz)'
                ),

                'Arb Bound ($/oz)': row.get(
                    'Arb Bound ($/oz)'
                ),

                'Noise Buf ($/oz)': row.get(
                    'Noise Buf ($/oz)'
                ),

                'Sell Thr ($/oz)': row.get(
                    'Sell Thr ($/oz)'
                ),

                'Buy Thr ($/oz)': row.get(
                    'Buy Thr ($/oz)'
                ),

                'Bound x': row.get(
                    'Bound x'
                ),

                'Bound Signal': row.get(
                    'Bound Signal'
                ),

                'Corrob (x)': row.get(
                    'Corrob (x)'
                ),

                'Corrob Basis': row.get(
                    'Corrob Basis'
                ),

                'Ann Mid (%)': row.get(
                    'Ann Mid (%)'
                ),

                'Metal Fwd (%)': row.get(
                    'Metal Fwd (%)'
                ),

                'COMEX Fwd (%)': row.get(
                    'COMEX Fwd (%)'
                ),

                'Lease (%)': row.get(
                    'Lease (%)'
                ),

                'Z-Score': row.get(
                    'Z-Score'
                ),

                'Percentile': row.get(
                    'Percentile'
                ),

                'Signal': row.get(
                    'Signal'
                ),
            })

    output = pd.DataFrame(
        rows
    )

    if output.empty:
        return output

    output['Sort Score'] = (
        pd.to_numeric(
            output['Bound x'],
            errors='coerce'
        )
        .abs()
    )

    output = output.sort_values(
        by=[
            'Sort Score',
            'EFP RV ($/oz)',
        ],
        ascending=[
            False,
            False,
        ],
        na_position='last'
    )

    output = output.drop(
        columns=[
            'Sort Score',
        ]
    )

    return output.reset_index(
        drop=True
    )


print(
    "✅ EFP RV opportunity-table builder loaded"
)


# ─────────────────────────────────────────────────────────────
#  CELL 12 — Cash & Carry Table Builder
# ─────────────────────────────────────────────────────────────

def build_cash_carry_table(
    efp_data
):
    """
    Cash-and-carry construction:

        1. Take delivery against the front future.
        2. Hold the warranted metal.
        3. Sell a deferred future.

    Gross Carry Spread:

        Deferred EFP - Front EFP

    Carry Cost:

        Spot
        x
        (SOFR + desk funding spread + insurance)
        x
        Hold Days / 360

    Net Cash Carry:

        Gross Carry Spread - Carry Cost

    Capital Used:

        Spot Mid + Front EFP

    Annualised ROC:

        Net Cash Carry
        / Capital Used
        x 360 / Hold Days
    """

    rows = []

    for metal in METALS:

        if metal not in efp_data:
            continue

        metal_dataframe = (
            efp_data[metal]
            .copy()
        )

        if (
            metal_dataframe.empty
            or
            'C1' not in metal_dataframe.index
        ):
            continue

        front_row = metal_dataframe.loc[
            'C1'
        ]

        front_efp = front_row.get(
            'EFP Mid'
        )

        front_days = front_row.get(
            'Days'
        )

        spot_mid = front_row.get(
            'Spot Mid'
        )

        delivery_trigger = (
            is_valid_number(front_efp)
            and
            front_efp < 0
        )

        capital_used = np.nan

        if (
            is_valid_number(spot_mid)
            and
            is_valid_number(front_efp)
        ):

            capital_used = (
                spot_mid
                +
                front_efp
            )

        for deferred_contract in [
            'C2',
            'C3',
            'C4',
        ]:

            if (
                deferred_contract
                not in metal_dataframe.index
            ):
                continue

            deferred_row = metal_dataframe.loc[
                deferred_contract
            ]

            deferred_efp = deferred_row.get(
                'EFP Mid'
            )

            deferred_days = deferred_row.get(
                'Days'
            )

            hold_days = np.nan

            gross_carry_spread = np.nan

            hold_sofr = np.nan

            hold_funding_cost = np.nan

            carry_cost = np.nan

            net_cash_carry = np.nan

            carry_per_day = np.nan

            roc_pct = np.nan

            annualised_roc_pct = np.nan

            if (
                is_valid_number(front_days)
                and
                is_valid_number(deferred_days)
            ):

                hold_days = (
                    deferred_days
                    -
                    front_days
                )

            if (
                is_valid_number(front_efp)
                and
                is_valid_number(deferred_efp)
            ):

                gross_carry_spread = (
                    deferred_efp
                    -
                    front_efp
                )

            if (
                is_valid_number(hold_days)
                and
                hold_days > 0
            ):

                hold_sofr = (
                    interpolate_sofr_rate(
                        hold_days
                    )
                )

                hold_funding_cost = (
                    interpolate_funding_cost_rate(
                        hold_days
                    )
                )

                carry_cost = carry_cost_dollar(
                    spot_mid=spot_mid,
                    funding_cost_pct=(
                        hold_funding_cost
                    ),
                    insurance_pct=(
                        INSURANCE_RATE_PCT
                    ),
                    days=hold_days
                )

                if (
                    is_valid_number(
                        gross_carry_spread
                    )
                    and
                    is_valid_number(
                        carry_cost
                    )
                ):

                    net_cash_carry = (
                        gross_carry_spread
                        -
                        carry_cost
                    )

                if is_valid_number(
                    net_cash_carry
                ):

                    carry_per_day = (
                        net_cash_carry
                        /
                        hold_days
                    )

                if (
                    is_valid_number(
                        net_cash_carry
                    )
                    and
                    is_valid_number(
                        capital_used
                    )
                    and
                    capital_used > 0
                ):

                    roc_pct = (
                        net_cash_carry
                        /
                        capital_used
                        *
                        100.0
                    )

                if is_valid_number(
                    roc_pct
                ):

                    annualised_roc_pct = (
                        roc_pct
                        *
                        360.0
                        /
                        hold_days
                    )

            if (
                is_valid_number(
                    net_cash_carry
                )
                and
                net_cash_carry > 0
                and
                delivery_trigger
            ):

                signal = 'Attractive'

            elif (
                is_valid_number(
                    net_cash_carry
                )
                and
                net_cash_carry > 0
                and
                not delivery_trigger
            ):

                signal = (
                    'Positive but no trigger'
                )

            else:

                signal = 'Not attractive'

            rows.append({

                'Metal': metal,

                'Name': METAL_NAMES.get(
                    metal,
                    metal
                ),

                'Front Contract': 'C1',

                'Deferred Contract': (
                    deferred_contract
                ),

                'Front Month': front_row.get(
                    'Month/Year'
                ),

                'Deferred Month': (
                    deferred_row.get(
                        'Month/Year'
                    )
                ),

                'Front EFP ($/oz)': (
                    front_efp
                ),

                'Deferred EFP ($/oz)': (
                    deferred_efp
                ),

                'Hold Days': hold_days,

                'Hold SOFR (%)': (
                    hold_sofr
                ),

                'SOFR Spread (%)': (
                    get_sofr_spread()
                ),

                'Hold Funding Cost (%)': (
                    hold_funding_cost
                ),

                'Insurance (%)': (
                    INSURANCE_RATE_PCT
                ),

                'Gross Carry Spread ($/oz)': (
                    gross_carry_spread
                ),

                'Carry Cost ($/oz)': (
                    carry_cost
                ),

                'Net Cash Carry ($/oz)': (
                    net_cash_carry
                ),

                'Carry Per Day ($/oz/day)': (
                    carry_per_day
                ),

                'Capital Used ($/oz)': (
                    capital_used
                ),

                'ROC (%)': roc_pct,

                'Annualised ROC (%)': (
                    annualised_roc_pct
                ),

                'Delivery Trigger': (
                    'Yes'
                    if delivery_trigger
                    else 'No'
                ),

                'Signal': signal,
            })

    output = pd.DataFrame(
        rows
    )

    if output.empty:
        return output

    output['Sort Score'] = (
        pd.to_numeric(
            output[
                'Annualised ROC (%)'
            ],
            errors='coerce'
        )
    )

    output = output.sort_values(
        by=[
            'Sort Score',
            'Net Cash Carry ($/oz)',
        ],
        ascending=[
            False,
            False,
        ],
        na_position='last'
    )

    output = output.drop(
        columns=[
            'Sort Score',
        ]
    )

    return output.reset_index(
        drop=True
    )


print(
    "✅ Cash-and-carry table builder loaded"
)


# ─────────────────────────────────────────────────────────────
#  CELL 13 — Switch Table Builder
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
#  DISPLAY COLUMN SETS
#
#  These four constants were referenced by the renderer but never defined,
#  so every render path raised NameError before printing anything. Each list
#  is derived from the columns its builder actually emits, so a rename in a
#  builder shows up here as a KeyError rather than silently dropping a column.
# ─────────────────────────────────────────────────────────────

EFP_OPPORTUNITY_DISPLAY_COLUMNS = [
    'Metal',
    'Month/Year',
    'Days',
    'EFP Mid ($/oz)',
    'EFP Spread ($/oz)',
    'OTC ($/oz)',
    'EFP RV ($/oz)',
    'Arb Bound ($/oz)',
    'Noise Buf ($/oz)',
    'Sell Thr ($/oz)',
    'Buy Thr ($/oz)',
    'Bound x',
    'Bound Signal',
    'Corrob (x)',
    'Corrob Basis',
    'Metal Fwd (%)',
    'COMEX Fwd (%)',
    'Lease (%)',
    'Z-Score',
    'Signal',
]

EFP_METAL_DISPLAY_COLUMNS = [
    'Month/Year',
    'Days',
    'EFP Bid',
    'EFP Ask',
    'EFP Mid',
    'EFP Spread',
    'OTC ($/oz)',
    'EFP RV ($/oz)',
    'Arb Bound ($/oz)',
    'Noise Buf ($/oz)',
    'Sell Thr ($/oz)',
    'Buy Thr ($/oz)',
    'Bound x',
    'Bound Signal',
    'Corrob (x)',
    'Metal Fwd (%)',
    'COMEX Fwd (%)',
    'Lease (%)',
]

CASH_CARRY_DISPLAY_COLUMNS = [
    'Metal',
    'Front Month',
    'Deferred Month',
    'Hold Days',
    'Front EFP ($/oz)',
    'Deferred EFP ($/oz)',
    'Hold SOFR (%)',
    'SOFR Spread (%)',
    'Hold Funding Cost (%)',
    'Insurance (%)',
    'Gross Carry Spread ($/oz)',
    'Carry Cost ($/oz)',
    'Net Cash Carry ($/oz)',
    'Carry Per Day ($/oz/day)',
    'Capital Used ($/oz)',
    'ROC (%)',
    'Annualised ROC (%)',
    'Signal',
]

SWITCH_DISPLAY_COLUMNS = [
    'Pair',
    'Near Month',
    'Far Month',
    'Days Gap',
    'Switch Bid',
    'Switch Ask',
    'Switch Mid',
    'Switch Spread',
    'Ann Bid (%)',
    'Ann Ask (%)',
    'Ann Mid (%)',
]

SWITCH_PAIRS = [
    (
        'C1',
        'C2',
    ),

    (
        'C2',
        'C3',
    ),

    (
        'C3',
        'C4',
    ),
]


def build_switch_tables(
    efp_data
):
    """
    Build adjacent EFP switch markets.

    Switch bid:

        Far EFP Bid - Near EFP Ask

    Switch ask:

        Far EFP Ask - Near EFP Bid

    Therefore the displayed switch incorporates the executable
    bid/ask legs rather than subtracting two mids.
    """

    switch_output = {}

    for metal in METALS:

        if metal not in efp_data:
            continue

        metal_dataframe = (
            efp_data[metal]
            .copy()
        )

        rows = []

        if metal_dataframe.empty:

            switch_output[metal] = (
                pd.DataFrame()
            )

            continue

        spot_mid = (
            metal_dataframe[
                'Spot Mid'
            ]
            .iloc[0]
        )

        for (
            near_contract,
            far_contract
        ) in SWITCH_PAIRS:

            if (
                near_contract
                not in metal_dataframe.index
            ):
                continue

            if (
                far_contract
                not in metal_dataframe.index
            ):
                continue

            near_row = metal_dataframe.loc[
                near_contract
            ]

            far_row = metal_dataframe.loc[
                far_contract
            ]

            near_bid = near_row.get(
                'EFP Bid'
            )

            near_ask = near_row.get(
                'EFP Ask'
            )

            far_bid = far_row.get(
                'EFP Bid'
            )

            far_ask = far_row.get(
                'EFP Ask'
            )

            switch_bid = np.nan

            switch_ask = np.nan

            switch_mid = np.nan

            switch_spread = np.nan

            if (
                is_valid_number(far_bid)
                and
                is_valid_number(near_ask)
            ):

                switch_bid = (
                    far_bid
                    -
                    near_ask
                )

            if (
                is_valid_number(far_ask)
                and
                is_valid_number(near_bid)
            ):

                switch_ask = (
                    far_ask
                    -
                    near_bid
                )

            if (
                is_valid_number(switch_bid)
                and
                is_valid_number(switch_ask)
            ):

                switch_mid = (
                    switch_bid + switch_ask
                ) / 2.0

                switch_spread = (
                    switch_ask - switch_bid
                )

            near_days = safe_float(near_row.get('Days'))
            far_days = safe_float(far_row.get('Days'))

            day_gap = np.nan

            if is_valid_number(near_days) and is_valid_number(far_days):
                day_gap = far_days - near_days

            def _annualise(value):
                """Switch expressed as an annualised rate over the leg gap."""
                if not (
                    is_valid_number(value)
                    and is_valid_number(spot_mid)
                    and is_valid_number(day_gap)
                    and day_gap > 0
                    and spot_mid != 0
                ):
                    return np.nan
                return value / spot_mid * (360.0 / day_gap) * 100.0

            rows.append({
                'Pair': f'{near_contract}/{far_contract}',
                'Near': near_contract,
                'Far': far_contract,
                'Near Month': near_row.get('Month/Year'),
                'Far Month': far_row.get('Month/Year'),
                'Days Gap': day_gap,
                'Switch Bid': switch_bid,
                'Switch Ask': switch_ask,
                'Switch Mid': switch_mid,
                'Switch Spread': switch_spread,
                'Ann Bid (%)': _annualise(switch_bid),
                'Ann Ask (%)': _annualise(switch_ask),
                'Ann Mid (%)': _annualise(switch_mid),
            })

        switch_output[metal] = (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(columns=SWITCH_DISPLAY_COLUMNS)
        )

    return switch_output


print(
    "✅ Switch table builder loaded"
)


# ─────────────────────────────────────────────────────────────
#  CELL 15 — Global Dashboard Objects
# ─────────────────────────────────────────────────────────────

spot_ba = {}
fut_ba = {}

fnd_map = {}
fut_month_yr = {}

spot_date = None

efp_data = {}

switch_data = {}

cash_carry_df = pd.DataFrame()

efp_rv_opportunity_df = pd.DataFrame()

efp_history = pd.DataFrame()


# ─────────────────────────────────────────────────────────────
#  CELL 16 — Main Calculation Engine
# ─────────────────────────────────────────────────────────────

def fetch_all_data():

    global spot_ba
    global fut_ba

    global fnd_map
    global fut_month_yr

    global current_sofr_curve
    global spot_date

    global efp_data
    global switch_data

    global efp_history

    global cash_carry_df
    global efp_rv_opportunity_df


    # ---------------------------------------------------------
    # Save forward inputs first
    # ---------------------------------------------------------

    save_forward_curves(
        verbose=False
    )


    # ---------------------------------------------------------
    # Build ticker universe
    # ---------------------------------------------------------

    all_spot_tickers = list(
        SPOT_TICKERS.values()
    )

    all_future_tickers = []

    for metal in METALS:

        all_future_tickers.extend(
            FUTURES_CHAIN[metal].values()
        )


    # ---------------------------------------------------------
    # Bloomberg fetch
    # ---------------------------------------------------------

    spot_ba = fetch_bid_ask(
        all_spot_tickers
    )

    fut_ba = fetch_bid_ask(
        all_future_tickers
    )

    fnd_map = fetch_fnd(
        all_future_tickers
    )

    fut_month_yr = (
        fetch_fut_month_yr(
            all_future_tickers
        )
    )


    # ---------------------------------------------------------
    # Fresh SOFR curve
    # ---------------------------------------------------------

    current_sofr_curve = (
        fetch_sofr_curve()
    )

    if not current_sofr_curve:

        current_sofr_curve = (
            fallback_sofr_curve()
        )


    # ---------------------------------------------------------
    # Fresh calc date
    #
    # Fixes the overnight notebook issue.
    # ---------------------------------------------------------

    calculation_date = (
        date.today()
    )

    spot_date = (
        get_spot_value_date(
            calculation_date
        )
    )


    # ---------------------------------------------------------
    # Build metal tables
    # ---------------------------------------------------------

    efp_data = {}


    for metal in METALS:

        spot_ticker = SPOT_TICKERS[
            metal
        ]

        spot_data = spot_ba[
            spot_ticker
        ]

        spot_mid = (
            spot_data['mid']
        )

        rows = []


        for contract, ticker in (
            FUTURES_CHAIN[metal]
            .items()
        ):

            future_data = fut_ba[
                ticker
            ]

            fnd = fnd_map.get(
                ticker
            )

            month_yr = (
                fut_month_yr.get(
                    ticker,
                    '—'
                )
            )


            days = (
                days_to_value(
                    fnd,
                    spot_date
                )
            )


            # -------------------------------------------------
            # EFP construction
            # -------------------------------------------------

            efp_bid = np.nan
            efp_ask = np.nan

            if (
                is_valid_number(
                    future_data['bid']
                )
                and
                is_valid_number(
                    spot_data['ask']
                )
            ):

                efp_bid = (
                    future_data['bid']
                    -
                    spot_data['ask']
                )


            if (
                is_valid_number(
                    future_data['ask']
                )
                and
                is_valid_number(
                    spot_data['bid']
                )
            ):

                efp_ask = (
                    future_data['ask']
                    -
                    spot_data['bid']
                )


            efp_mid = np.nan

            efp_spread = np.nan

            if (
                is_valid_number(
                    efp_bid
                )
                and
                is_valid_number(
                    efp_ask
                )
            ):

                efp_mid = (
                    efp_bid
                    +
                    efp_ask
                ) / 2.0

                efp_spread = (
                    efp_ask
                    -
                    efp_bid
                )


            # -------------------------------------------------
            # Annualised EFP
            # -------------------------------------------------

            ann_mid = np.nan

            if (
                is_valid_number(
                    efp_mid
                )
                and
                is_valid_number(
                    spot_mid
                )
                and
                is_valid_number(
                    days
                )
                and
                days > 0
            ):

                ann_mid = (
                    efp_mid
                    /
                    spot_mid
                    *
                    360.0
                    /
                    days
                    *
                    100.0
                )


            # -------------------------------------------------
            # OTC valuation
            # -------------------------------------------------

            metal_fwd_pct = (
                interpolate_metal_forward_rate(
                    metal,
                    days
                )
            )

            otc_value = (
                otc_value_dollar(
                    spot_mid,
                    metal_fwd_pct,
                    days
                )
            )


            efp_rv = np.nan

            if (
                is_valid_number(
                    efp_mid
                )
                and
                is_valid_number(
                    otc_value
                )
            ):

                efp_rv = (
                    efp_mid
                    -
                    otc_value
                )


            # -------------------------------------------------
            # Lease / bound
            # -------------------------------------------------

            lease_pct = (
                derived_lease_pct(
                    metal_fwd_pct,
                    days
                )
            )

            arb_bound = (
                arb_bound_dollar(
                    metal,
                    spot_mid,
                    lease_pct,
                    days
                )
            )


            noise_buffer = (
                noise_buffer_dollar(
                    spot_mid,
                    days
                )
            )


            sell_thr, buy_thr = (
                bound_thresholds(
                    metal,
                    arb_bound,
                    spot_mid,
                    days
                )
            )


            bound_x = (
                bound_multiple(
                    metal,
                    efp_rv,
                    arb_bound,
                    spot_mid,
                    days
                )
            )


            bound_signal = (
                classify_bound(
                    metal,
                    efp_rv,
                    arb_bound,
                    spot_mid,
                    days
                )
            )


            rows.append({

                'Contract':
                    contract,

                'Month/Year':
                    month_yr,

                'FND':
                    str(fnd)
                    if fnd
                    else '—',

                'Days':
                    days,

                'Spot Mid':
                    spot_mid,

                'Fut Mid':
                    future_data['mid'],

                'EFP Bid':
                    efp_bid,

                'EFP Ask':
                    efp_ask,

                'EFP Mid':
                    efp_mid,

                'EFP Spread':
                    efp_spread,

                'Ann Mid (%)':
                    ann_mid,

                'Metal Fwd (%)':
                    metal_fwd_pct,

                'Lease (%)':
                    lease_pct,

                'OTC ($/oz)':
                    otc_value,

                'EFP RV ($/oz)':
                    efp_rv,

                'SOFR (%)':
                    interpolate_sofr_rate(
                        days
                    ),

                'SOFR Spread (%)':
                    get_sofr_spread(),

                'Funding Cost (%)':
                    interpolate_funding_cost_rate(
                        days
                    ),

                'Arb Bound ($/oz)':
                    arb_bound,

                'Noise Buf ($/oz)':
                    noise_buffer,

                'Sell Thr ($/oz)':
                    sell_thr,

                'Buy Thr ($/oz)':
                    buy_thr,

                'Bound x':
                    bound_x,

                'Bound Signal':
                    bound_signal,
            })

        metal_df = (
            pd.DataFrame(rows)
            .set_index(
                'Contract'
            )
        )

        # FIXED:
        # Maturity-specific corroboration

        metal_df = (
            add_comex_corroboration(
                metal_df
            )
        )

        efp_data[metal] = (
            metal_df
        )


    # ---------------------------------------------------------
    # History calculations
    # ---------------------------------------------------------

    efp_history = (
        load_efp_history()
    )

    history_rows = []


    for metal, df in efp_data.items():

        for contract in df.index:

            stats = (
                get_history_stats(
                    history=efp_history,
                    metal=metal,
                    contract=contract,
                    current_value=df.loc[
                        contract,
                        'EFP RV ($/oz)'
                    ],
                    month_yr=df.loc[
                        contract,
                        'Month/Year'
                    ]
                )
            )

            for k, v in (
                stats.items()
            ):

                df.loc[
                    contract,
                    k
                ] = v


            row = df.loc[
                contract
            ]

            history_rows.append({

                'Timestamp':
                    datetime.now(),

                'Metal':
                    metal,

                'Contract':
                    contract,

                'Month/Year':
                    row['Month/Year'],

                'EFP RV ($/oz)':
                    row['EFP RV ($/oz)'],

                'COMEX Fwd (%)':
                    row['COMEX Fwd (%)'],

                'Corrob (x)':
                    row['Corrob (x)'],

                'Corrob Basis':
                    row['Corrob Basis'],

                'Arb Bound ($/oz)':
                    row['Arb Bound ($/oz)'],

                'Noise Buf ($/oz)':
                    row['Noise Buf ($/oz)'],

                'Sell Thr ($/oz)':
                    row['Sell Thr ($/oz)'],

                'Buy Thr ($/oz)':
                    row['Buy Thr ($/oz)'],

                'Bound x':
                    row['Bound x'],

                'Bound Signal':
                    row['Bound Signal'],

                # ---------------------------------------------------------
                # RESTORED. The v2 rewrite cut history from ~25 columns to 14
                # and dropped everything below. Two consequences:
                #
                #   1. EFP Bid/Ask/Spread is the DEALING COST that every
                #      backtest figure is struck gross of. 50bp costs 14% of
                #      P&L and it settles the capacity question. It is computed
                #      on every refresh and was being thrown away.
                #
                #   2. Metal Fwd (%) and SOFR (%) are what the regime
                #      classifier uses to rebuild lease history
                #      (lease = SOFR - forward). Without them it can never leave
                #      burn-in, so regime sizing would sit at 1.0x forever and
                #      look like it was working.
                #
                # None of this can be backfilled.
                # ---------------------------------------------------------
                'Spot Bid': row.get('Spot Bid'),
                'Spot Ask': row.get('Spot Ask'),
                'Spot Mid': row.get('Spot Mid'),
                'Fut Bid': row.get('Fut Bid'),
                'Fut Ask': row.get('Fut Ask'),
                'Fut Mid': row.get('Fut Mid'),
                'EFP Bid': row.get('EFP Bid'),
                'EFP Ask': row.get('EFP Ask'),
                'EFP Mid': row.get('EFP Mid'),
                'EFP Spread': row.get('EFP Spread'),
                'Ann Mid (%)': row.get('Ann Mid (%)'),
                'Metal Fwd (%)': row.get('Metal Fwd (%)'),
                'Lease (%)': row.get('Lease (%)'),
                'SOFR (%)': row.get('SOFR (%)'),
                'SOFR Spread (%)': row.get('SOFR Spread (%)'),
                'Funding Cost (%)': row.get('Funding Cost (%)'),
                'OTC ($/oz)': row.get('OTC ($/oz)'),
                'Days': row.get('Days'),
                'FND': row.get('FND'),
            })


    save_efp_history(
        history_rows
    )


    # ---------------------------------------------------------
    # Opportunity table
    # ---------------------------------------------------------

    efp_rv_opportunity_df = (
        build_efp_rv_opportunity_table(
            efp_data
        )
    )


    # ---------------------------------------------------------
    # Cash & Carry
    # ---------------------------------------------------------

    cash_carry_df = (
        build_cash_carry_table(
            efp_data
        )
    )


    # ---------------------------------------------------------
    # Switch tables
    # ---------------------------------------------------------

    switch_data = (
        build_switch_tables(
            efp_data
        )
    )


    print(
        f"✅ Refresh complete | "
        f"Spot date: {spot_date}"
    )


# ─────────────────────────────────────────────────────────────
#  CELL 17 — Dashboard Rendering
# ─────────────────────────────────────────────────────────────

def render_dashboard():

    clear_output(wait=True)

    display_forward_inputs()

    display(
        widgets.VBox([
            widgets.HBox([
                save_button,
                refresh_button
            ]),
            status_label
        ])
    )

    print(
        f"Spot Date: {spot_date}"
    )

    print(
        f"Refresh Time: "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    print(
        "\nTop opportunities ranked by "
        "absolute Bound x"
    )

    if not efp_rv_opportunity_df.empty:

        display(
            efp_rv_opportunity_df[
                EFP_OPPORTUNITY_DISPLAY_COLUMNS
            ].head(20).style.format({

                'EFP Mid ($/oz)':
                    '{:+.2f}',

                'EFP Spread ($/oz)':
                    '{:.2f}',

                'OTC ($/oz)':
                    '{:+.2f}',

                'EFP RV ($/oz)':
                    '{:+.2f}',

                'Arb Bound ($/oz)':
                    '{:.2f}',

                'Noise Buf ($/oz)':
                    '{:.2f}',

                'Sell Thr ($/oz)':
                    '{:+.2f}',

                'Buy Thr ($/oz)':
                    '{:+.2f}',

                'Bound x':
                    '{:+.2f}x',

                'Corrob (x)':
                    '{:.1f}x',

                'Metal Fwd (%)':
                    '{:+.2f}%',

                'COMEX Fwd (%)':
                    '{:+.2f}%',

                'Lease (%)':
                    '{:+.2f}%',

                'Z-Score':
                    '{:+.2f}'

            }, na_rep='—')
        )

    # ---------------------------------------------------------
    # Metal tables
    # ---------------------------------------------------------

    for metal in METALS:

        if metal not in efp_data:
            continue

        metal_df = efp_data[metal]

        print(
            f"\n{'='*80}"
        )

        print(
            f"{METAL_NAMES[metal]} "
            f"({metal})"
        )

        print(
            f"{'='*80}"
        )

        display(
            metal_df[
                EFP_METAL_DISPLAY_COLUMNS
            ].style.format({

                'EFP Bid':
                    '{:+.2f}',

                'EFP Ask':
                    '{:+.2f}',

                'EFP Mid':
                    '{:+.2f}',

                'OTC ($/oz)':
                    '{:+.2f}',

                'EFP RV ($/oz)':
                    '{:+.2f}',

                'Arb Bound ($/oz)':
                    '{:.2f}',

                'Noise Buf ($/oz)':
                    '{:.2f}',

                'Sell Thr ($/oz)':
                    '{:+.2f}',

                'Buy Thr ($/oz)':
                    '{:+.2f}',

                'Bound x':
                    '{:+.2f}x',

                'COMEX Fwd (%)':
                    '{:+.2f}%',

                'Metal Fwd (%)':
                    '{:+.2f}%',

                'Lease (%)':
                    '{:+.2f}%',

                'Corrob (x)':
                    '{:.1f}x',

                'Percentile':
                    '{:.0f}%',

                'Z-Score':
                    '{:+.2f}',

            }, na_rep='—')
        )

    # ---------------------------------------------------------
    # Cash & Carry
    # ---------------------------------------------------------

    if not cash_carry_df.empty:

        print(
            "\nCash & Carry Monitor"
        )

        display(
            cash_carry_df[
                CASH_CARRY_DISPLAY_COLUMNS
            ].style.format({

                'Front EFP ($/oz)':
                    '{:+.2f}',

                'Deferred EFP ($/oz)':
                    '{:+.2f}',

                'Hold SOFR (%)':
                    '{:+.2f}%',

                'SOFR Spread (%)':
                    '{:+.2f}%',

                'Hold Funding Cost (%)':
                    '{:+.2f}%',

                'Insurance (%)':
                    '{:.3f}%',

                'Gross Carry Spread ($/oz)':
                    '{:+.2f}',

                'Carry Cost ($/oz)':
                    '{:+.2f}',

                'Net Cash Carry ($/oz)':
                    '{:+.2f}',

                'Carry Per Day ($/oz/day)':
                    '{:+.4f}',

                'Capital Used ($/oz)':
                    '{:,.2f}',

                'ROC (%)':
                    '{:+.4f}%',

                'Annualised ROC (%)':
                    '{:+.2f}%'

            }, na_rep='—')
        )

    # ---------------------------------------------------------
    # Switch Markets
    # ---------------------------------------------------------

    for metal in METALS:

        if metal not in switch_data:
            continue

        switch_df = switch_data[
            metal
        ]

        if switch_df.empty:
            continue

        print(
            f"\n{METAL_NAMES[metal]} "
            f"Switch Market"
        )

        display(
            switch_df[
                SWITCH_DISPLAY_COLUMNS
            ].style.format({

                'Switch Bid':
                    '{:+.2f}',

                'Switch Ask':
                    '{:+.2f}',

                'Switch Mid':
                    '{:+.2f}',

                'Switch Spread':
                    '{:.2f}',

                'Ann Bid (%)':
                    '{:+.2f}%',

                'Ann Ask (%)':
                    '{:+.2f}%',

                'Ann Mid (%)':
                    '{:+.2f}%',

            }, na_rep='—')
        )

    render_charts()


# ─────────────────────────────────────────────────────────────
#  CELL 18 — Charts
# ─────────────────────────────────────────────────────────────

def render_charts():

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16,10)
    )

    axes = axes.flatten()

    for axis, metal in zip(
        axes,
        METALS
    ):

        if metal not in efp_data:
            continue

        df = efp_data[metal]

        x = np.arange(len(df))

        rv = pd.to_numeric(
            df['EFP RV ($/oz)'],
            errors='coerce'
        )

        axis.bar(
            x,
            rv,
            color=[
                C_RED if v > 0
                else C_GREEN
                for v in rv
            ]
        )

        axis.plot(
            x,
            df['Sell Thr ($/oz)'],
            marker='o',
            color=C_RED,
            label='Sell Thr'
        )

        axis.plot(
            x,
            df['Buy Thr ($/oz)'],
            marker='o',
            color=C_GREEN,
            label='Buy Thr'
        )

        axis.axhline(
            0,
            color=C_MUTED
        )

        axis.set_xticks(x)

        axis.set_xticklabels(
            df['Month/Year']
        )

        axis.set_title(
            f"{METAL_NAMES[metal]} "
            f"EFP RV"
        )

        axis.legend()

        axis.grid(True)

    plt.tight_layout()

    plt.show()


# ─────────────────────────────────────────────────────────────
#  CELL 19 — Button Handlers
# ─────────────────────────────────────────────────────────────

save_button = widgets.Button(
    description='Save Inputs',
    button_style='success'
)

refresh_button = widgets.Button(
    description='Refresh Data',
    button_style='info'
)

status_label = widgets.HTML()


def on_save_clicked(button):

    try:

        save_forward_curves(
            verbose=True
        )

        status_label.value = (
            f"✅ Saved "
            f"{datetime.now():%H:%M:%S}"
        )

    except Exception as error:

        status_label.value = (
            f"❌ Save Failed: "
            f"{error}"
        )


def on_refresh_clicked(button):

    status_label.value = (
        "Refreshing..."
    )

    try:

        fetch_all_data()

        render_dashboard()

        status_label.value = (
            f"✅ Refreshed "
            f"{datetime.now():%H:%M:%S}"
        )

    except Exception as error:

        import traceback

        traceback.print_exc()

        status_label.value = (
            f"❌ Refresh Failed: "
            f"{error}"
        )


save_button.on_click(
    on_save_clicked
)

refresh_button.on_click(
    on_refresh_clicked
)


# ─────────────────────────────────────────────────────────────
#  CELL 20 — Initial Startup
# ─────────────────────────────────────────────────────────────

print(
    "Launching dashboard..."
)

display_forward_inputs()

display(
    widgets.VBox([
        widgets.HBox([
            save_button,
            refresh_button
        ]),
        status_label
    ])
)

try:

    fetch_all_data()

    render_dashboard()

    status_label.value = (
        f"✅ Loaded "
        f"{datetime.now():%H:%M:%S}"
    )

except Exception as error:

    import traceback

    traceback.print_exc()

    status_label.value = (
        f"❌ Initial Load Failed: "
        f"{error}"
    )

print(
    "✅ EFP Dashboard Ready"
)
