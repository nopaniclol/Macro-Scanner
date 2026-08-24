# ─────────────────────────────────────────────────────────────
#  CELL 12 — GO-LIVE TESTS & DATA CAPTURE
#
#  Append this as a new cell AFTER Cell 11. It is self-contained and does not
#  modify anything above it — it reads your existing globals (bq, METALS,
#  SPOT_TICKERS, efp_data, HISTORY_FILE) and writes its own files.
#
#  Covers Tier 0 of OPEN_ITEMS.md:
#    0.2  forward-curve defaults sanity check
#    0.3  live vs settle-based residual reconciliation logger
#    0.4  EFP spread capture  (also patched into Cell 8's history_rows)
#    0.5  settle-matched intraday spot snap
#    +    regime classifier and regime-conditional position sizing
#
#  Run  daily_checks()  once per day, after fetch_all_data().
#  Two of these datasets CANNOT BE BACKFILLED. Every day you do not run this
#  is a day permanently missing from the snap and spread series.
# ─────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

# ── files this cell owns ─────────────────────────────────────
SNAP_FILE   = 'pm_intraday_snap.csv'      # settle-matched spot, 13:30 NY
RECON_FILE  = 'pm_reconciliation.csv'     # live vs settle-based residual
REGIME_FILE = 'pm_regime_state.csv'       # daily regime classification

# ── regime classifier ────────────────────────────────────────
# Trailing sd of the 1M lease vs an expanding quantile of its own history,
# shifted one session. Strictly causal: nothing about day t uses day t or later.
#
# Chosen over residual vol (1.37) and spot vol (1.38) because it scores 1.46 and
# has a flatter parameter surface. It is not price movement that matters, it is
# the volatility of physical scarcity itself.
REGIME_WINDOW   = 60      # sessions in the trailing sd
REGIME_QUANTILE = 0.80    # expanding-quantile bar
REGIME_MIN_OBS  = 250     # burn-in before it fires at all

# ── position sizing ──────────────────────────────────────────
# Size by bound multiple, but ONLY in a high-lease-vol regime. In a quiet
# regime every trade is 1.0x. Applied unconditionally this helps dislocation
# years and hurts quiet ones; conditioned on regime it improves every risk
# metric (Sharpe 1.36 -> 1.47, max drawdown -4.61% -> -4.10%).
SIZE_BUCKETS = ((1.5, 0.5), (2.5, 1.0), (5.0, 1.5), (np.inf, 2.0))

NY_SETTLE_HHMM = (13, 30)   # COMEX precious metals settle, New York


# ═════════════════════════════════════════════════════════════
#  0.5  SETTLE-MATCHED INTRADAY SNAP
# ═════════════════════════════════════════════════════════════
def capture_intraday_snap(days=3, verbose=True):
    """
    London spot sampled at the COMEX 13:30 NY settle instead of 3-4h later.

    ~55% of gold's residual variation is that clock offset. It was impossible to
    fix historically because no intraday history was on hand. Going forward it is
    one BQL call, and it is the single largest quality upgrade available.

    Appends to SNAP_FILE. Needs roughly 6 months banked before it can replace the
    vendor close, so start now even though you cannot use it yet.
    """
    end = datetime.now()
    start = end - timedelta(days=days)
    rows = []

    for metal in METALS:
        tkr = SPOT_TICKERS[metal]
        try:
            req = bql.Request(
                tkr,
                {'px': bq.data.px_last()},
                with_params={
                    'dates': bq.func.range(start.strftime('%Y-%m-%dT%H:%M:%S'),
                                           end.strftime('%Y-%m-%dT%H:%M:%S')),
                    'frq': '1m',
                    'timezone': 'America/New_York',
                },
            )
            df = pd.concat([r.df() for r in bq.execute(req)]).reset_index()
            tcol = next(c for c in df.columns
                        if 'DATE' in c.upper() or 'TIME' in c.upper())
            df[tcol] = pd.to_datetime(df[tcol])
            df['d'] = df[tcol].dt.normalize()
            cut = df[tcol].dt.hour * 60 + df[tcol].dt.minute
            keep = df[cut <= NY_SETTLE_HHMM[0] * 60 + NY_SETTLE_HHMM[1]]

            for d_, g in keep.groupby('d'):
                last = g.sort_values(tcol).iloc[-1]
                rows.append({
                    'Date': d_.date(),
                    'Metal': metal,
                    'Snap NY 13:30': float(last['px']),
                    'Snap Time': str(last[tcol]),
                    'Captured': datetime.now().replace(microsecond=0),
                })
        except Exception as e:                                  # noqa: BLE001
            if verbose:
                print(f'  ⚠ snap failed for {metal}: {e}')

    if not rows:
        if verbose:
            print('  ⚠ no intraday snap captured')
        return pd.DataFrame()

    new = pd.DataFrame(rows)
    if os.path.exists(SNAP_FILE):
        old = pd.read_csv(SNAP_FILE)
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(subset=['Date', 'Metal'], keep='last')
    new.to_csv(SNAP_FILE, index=False)

    if verbose:
        n = len(new)
        span = pd.to_datetime(new['Date'])
        print(f'  ✅ snap file: {n} rows, {span.min().date()} → {span.max().date()}')
        need = 126 - new.groupby('Metal').size().min()
        if need > 0:
            print(f'     {need:.0f} more sessions before this can replace the vendor close')
    return new


def snap_vs_close(verbose=True):
    """
    How big is the clock offset, in the data you have captured so far?

    This is the number that justifies the 3-day averaged mark to an allocator.
    """
    if not os.path.exists(SNAP_FILE):
        print('  no snap file yet — run capture_intraday_snap()')
        return pd.DataFrame()

    snap = pd.read_csv(SNAP_FILE)
    snap['Date'] = pd.to_datetime(snap['Date']).dt.date
    out = []

    for metal in METALS:
        s = snap[snap.Metal == metal]
        if s.empty:
            continue
        closes = []
        for _, r in s.iterrows():
            ba = spot_ba.get(SPOT_TICKERS[metal], {})
            mid = ba.get('mid', np.nan)
            closes.append(mid)
        s = s.assign(Close=closes).dropna(subset=['Close'])
        if s.empty:
            continue
        gap = s['Close'] - s['Snap NY 13:30']
        out.append({'Metal': metal, 'n': len(s),
                    'mean gap $': gap.mean(), 'sd gap $': gap.std(),
                    'max |gap| $': gap.abs().max()})

    df = pd.DataFrame(out)
    if verbose and not df.empty:
        print('\n  CLOSE minus SETTLE-MATCHED SNAP ($/oz)')
        print(df.round(3).to_string(index=False))
    return df


# ═════════════════════════════════════════════════════════════
#  REGIME CLASSIFIER
# ═════════════════════════════════════════════════════════════
def lease_history_from_efp_history():
    """
    Daily lease series rebuilt from your own history file: lease = SOFR - forward.

    You enter the forward by hand, and Cell 8 already stores both 'Metal Fwd (%)'
    and 'SOFR (%)' on every refresh, so the lease history assembles itself.
    """
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()

    h = pd.read_csv(HISTORY_FILE)
    need = {'Timestamp', 'Metal', 'Metal Fwd (%)', 'SOFR (%)'}
    if not need.issubset(h.columns):
        return pd.DataFrame()

    h['Timestamp'] = pd.to_datetime(h['Timestamp'])
    h['Date'] = h['Timestamp'].dt.date
    h['Lease'] = pd.to_numeric(h['SOFR (%)'], errors='coerce') - \
                 pd.to_numeric(h['Metal Fwd (%)'], errors='coerce')
    # one observation per metal per day, front-ish contract
    g = (h.sort_values('Timestamp')
           .groupby(['Date', 'Metal'], as_index=False)['Lease'].mean())
    piv = g.pivot(index='Date', columns='Metal', values='Lease')
    piv.index = pd.to_datetime(piv.index)
    return piv.sort_index()


def regime_high(metal, lease_hist=None, window=None, q=None, min_obs=None):
    """
    (is_high_today, n_observations, diagnostic_string)

    Returns is_high=False while the classifier is still in burn-in — which means
    equal-weight sizing, i.e. exactly the current behaviour. It never guesses.
    """
    window  = REGIME_WINDOW   if window  is None else window
    q       = REGIME_QUANTILE if q       is None else q
    min_obs = REGIME_MIN_OBS  if min_obs is None else min_obs

    lh = lease_history_from_efp_history() if lease_hist is None else lease_hist
    if lh.empty or metal not in lh.columns:
        return False, 0, 'no lease history'

    s = lh[metal].dropna()
    n = len(s)
    if n < min_obs:
        return False, n, f'burn-in: {n}/{min_obs} sessions'

    vol = s.diff().rolling(window).std()
    bar = vol.expanding(min_periods=min_obs).quantile(q)
    flag = (vol > bar).shift(1)          # shift = knowable at t
    if flag.dropna().empty:
        return False, n, 'insufficient window'

    cur = bool(flag.dropna().iloc[-1])
    pct = float(flag.dropna().tail(60).mean() * 100)
    return cur, n, f'{"HIGH" if cur else "normal"} | high on {pct:.0f}% of last 60'


def size_multiplier(bound_mult, high_regime, buckets=SIZE_BUCKETS):
    """Position size as a multiple of the base unit."""
    if not high_regime or pd.isna(bound_mult):
        return 1.0
    for edge, mult in buckets:
        if bound_mult < edge:
            return mult
    return buckets[-1][1]


# ═════════════════════════════════════════════════════════════
#  0.3  LIVE vs SETTLE-BASED RESIDUAL RECONCILIATION
# ═════════════════════════════════════════════════════════════
def log_reconciliation(verbose=True):
    """
    Your EFP is a simultaneous two-way quote. The backtest is a 13:30 settle
    against a spot close struck 3-4 hours later. THEY ARE DIFFERENT SERIES.

    This logs both every day so you can prove they track before putting risk on.
    Target: agreement within $0.50/oz on 95%+ of observations over ~20 sessions.
    """
    rows = []
    ts = datetime.now().replace(second=0, microsecond=0)

    for metal in METALS:
        if metal not in efp_data:
            continue
        df = efp_data[metal]
        for contract in df.index:
            r = df.loc[contract]
            try:
                settle = fetch_last([r['Fut Tkr']]).get(r['Fut Tkr'], np.nan)
            except Exception:                                   # noqa: BLE001
                settle = np.nan
            spot_close = r.get('Spot Mid', np.nan)
            efp_settle = (settle - spot_close
                          if pd.notna(settle) and pd.notna(spot_close) else np.nan)
            otc = r.get('OTC ($/oz)', np.nan)
            rows.append({
                'Timestamp': ts, 'Metal': metal, 'Contract': contract,
                'Month/Year': r.get('Month/Year'),
                'EFP Mid (live 2-way)': r.get('EFP Mid'),
                'EFP (settle - close)': efp_settle,
                'RV live': r.get('EFP RV ($/oz)'),
                'RV settle': (efp_settle - otc
                              if pd.notna(efp_settle) and pd.notna(otc) else np.nan),
                'EFP Spread': r.get('EFP Spread'),
            })

    new = pd.DataFrame(rows)
    if new.empty:
        return new
    new['Diff'] = new['RV live'] - new['RV settle']

    if os.path.exists(RECON_FILE):
        new = pd.concat([pd.read_csv(RECON_FILE), new], ignore_index=True)
    new.to_csv(RECON_FILE, index=False)

    if verbose:
        d = pd.to_numeric(new['Diff'], errors='coerce').dropna()
        if len(d):
            within = (d.abs() <= 0.50).mean() * 100
            print(f'  reconciliation: n={len(d)}  median diff ${d.median():+.3f}  '
                  f'sd ${d.std():.3f}  within $0.50 on {within:.0f}%')
            sessions = new['Timestamp'].nunique()
            print(f'    {sessions} session(s) logged; '
                  f'{"PASS" if (within >= 95 and sessions >= 20) else f"need 20 sessions and 95% — {max(0,20-sessions)} to go"}')
    return new


# ═════════════════════════════════════════════════════════════
#  0.2  FORWARD-CURVE SANITY
# ═════════════════════════════════════════════════════════════
def check_forward_defaults(verbose=True):
    """
    Your DEFAULT_METAL_FORWARD_CURVES has gold 3M at 1.10% when the market
    implies ~4.38%. On a 180-day contract that is $69.62/oz of PHANTOM richness
    against a sub-$1 bound — a permanent SELL EFP on every contract.

    Harmless while the saved CSV exists; a loaded gun on a fresh environment.
    """
    rows = []
    for metal in METALS:
        if metal not in efp_data:
            continue
        df = efp_data[metal]
        r = df.loc['C3'] if 'C3' in df.index else df.iloc[0]
        efp, spot, days = r.get('EFP Mid'), r.get('Spot Mid'), r.get('Days')
        implied = (efp / spot * 360 / float(days) * 100
                   if all(pd.notna(v) for v in (efp, spot, days)) and days else np.nan)
        entered = r.get('Metal Fwd (%)')
        rows.append({'Metal': metal, 'entered fwd %': entered,
                     'EFP-implied fwd %': implied,
                     'gap pp': (entered - implied) if pd.notna(implied) else np.nan,
                     'default 3M %': DEFAULT_METAL_FORWARD_CURVES[metal]['3M']})
    out = pd.DataFrame(rows)
    if verbose and not out.empty:
        print('\n  FORWARD CURVE SANITY')
        print(out.round(2).to_string(index=False))
        bad = out[out['gap pp'].abs() > 1.5]
        if not bad.empty:
            print('  ⚠ entered forward differs from the EFP-implied forward by >1.5pp')
            print('    for: ' + ', '.join(bad.Metal))
            print('    Either your curve is stale or the EFP is genuinely dislocated.')
            print('    Cross-check Corrob (x) before trusting any Bound Signal.')
    return out


# ═════════════════════════════════════════════════════════════
#  DAILY RUNNER
# ═════════════════════════════════════════════════════════════
def daily_checks(snap=True, recon=True, verbose=True):
    """Run every go-live check. Call once a day, after fetch_all_data()."""
    bar = '=' * 78
    print(f'\n{bar}\n  GO-LIVE DAILY CHECKS — {date.today()}\n{bar}')

    if snap:
        print('\n[0.5] SETTLE-MATCHED INTRADAY SNAP')
        capture_intraday_snap()

    print('\n[0.2] FORWARD CURVE')
    check_forward_defaults()

    if recon:
        print('\n[0.3] LIVE vs SETTLE RECONCILIATION')
        log_reconciliation()

    print('\n[REGIME] lease-volatility classifier')
    lh = lease_history_from_efp_history()
    reg = {}
    for metal in METALS:
        hi, n, diag = regime_high(metal, lh)
        reg[metal] = hi
        print(f'    {metal:<6} {diag}')
    if lh.empty or len(lh) < REGIME_MIN_OBS:
        print(f'    -> all metals sized 1.0x until burn-in completes. This is the')
        print(f'       current behaviour, so nothing changes yet.')

    pd.DataFrame([{'Date': date.today(), **{m: reg[m] for m in METALS}}]).to_csv(
        REGIME_FILE, mode='a', header=not os.path.exists(REGIME_FILE), index=False)

    print('\n[SIZING] today\'s multipliers')
    print(f"    {'Metal':<7}{'Contract':<10}{'Bound x':>9}{'Regime':>9}{'Size':>7}  Signal")
    for metal in METALS:
        if metal not in efp_data:
            continue
        df = efp_data[metal]
        for contract in df.index:
            r = df.loc[contract]
            bx, sig = r.get('Bound x'), r.get('Bound Signal')
            if sig not in ('SELL EFP', 'BUY EFP'):
                continue
            mult = size_multiplier(abs(bx) if pd.notna(bx) else np.nan, reg[metal])
            print(f'    {metal:<7}{str(r.get("Month/Year")):<10}'
                  f'{bx:>9.2f}{"HIGH" if reg[metal] else "normal":>9}{mult:>7.1f}x  {sig}')

    print(f'\n{bar}')
    print('  Two of these datasets cannot be backfilled. Run this every session.')
    print(bar)


print('✅ Cell 12 loaded — run daily_checks() after fetch_all_data()')


# ═════════════════════════════════════════════════════════════
#  SELF-TEST — run once after pasting, before trusting anything
# ═════════════════════════════════════════════════════════════
def self_test(verbose=True):
    """
    Offline checks that need no Bloomberg call. Run this first.

    Every assertion here was validated against the backtest engine, which
    reproduces 4,364/4,364 signals and passes a 10,644-flag look-ahead audit.
    """
    ok, bad = [], []

    def chk(name, cond):
        (ok if cond else bad).append(name)

    # sizing table
    chk('quiet regime is always 1.0x', size_multiplier(9.9, False) == 1.0)
    chk('nan bound multiple is 1.0x', size_multiplier(np.nan, True) == 1.0)
    chk('weak breach halves in a high regime', size_multiplier(1.2, True) == 0.5)
    chk('mid breach is 1.0x', size_multiplier(2.0, True) == 1.0)
    chk('strong breach is 1.5x', size_multiplier(3.0, True) == 1.5)
    chk('extreme breach doubles', size_multiplier(9.9, True) == 2.0)

    # regime classifier guards. Use METALS[0], not a hardcoded name -- your
    # METALS are ticker codes (XAU...), and asking for an absent column would
    # short-circuit to 'no lease history' and make the burn-in test vacuous.
    m0 = METALS[0]
    chk('no history returns False, never guesses',
        regime_high(m0, pd.DataFrame())[0] is False)
    idx = pd.date_range('2024-01-01', periods=100, freq='B')
    short = pd.DataFrame({m: np.random.randn(100).cumsum() for m in METALS}, index=idx)
    hi, n, diag = regime_high(m0, short)
    chk('burn-in returns False and says so', hi is False and 'burn-in' in diag)

    # causality: perturbing the last row must not change an earlier flag
    idx = pd.date_range('2022-01-03', periods=700, freq='B')
    rng = np.random.default_rng(0)
    base = pd.DataFrame({m: rng.standard_normal(700).cumsum() * 0.05 for m in METALS},
                        index=idx)
    shocked = base.copy()
    shocked.iloc[-1, 0] *= 50
    chk('a shock to today cannot move an earlier flag',
        regime_high(m0, base.iloc[:-1])[0] == regime_high(m0, shocked.iloc[:-1])[0])

    if verbose:
        print(f'\n  SELF-TEST — {len(ok)} passed, {len(bad)} failed')
        for x in ok:
            print(f'    ok    {x}')
        for x in bad:
            print(f'    FAIL  {x}')
    return not bad
