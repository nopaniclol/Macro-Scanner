"""
leverage_analysis.py
====================
$50M AUM leverage and return analysis for precious metals calendar spread strategies.
"""
import numpy as np
import pandas as pd

AUM = 50_000_000
RFR = 0.045   # T-bill rate

strategies = [
    dict(
        name              = "SI cal_12 norm",
        ann_ret_bps       = 75.63,
        ann_vol_bps       = 63.15,
        sharpe            = 1.198,
        oz_per_lot        = 5000,
        avg_spot          = 25.21,
        margin_per_spread = 750,
        adv_contracts     = 80_000,   # SI outright ADV
        max_pct_adv       = 0.05,
    ),
    dict(
        name              = "GC cal_12 norm",
        ann_ret_bps       = 35.17,
        ann_vol_bps       = 41.76,
        sharpe            = 0.842,
        oz_per_lot        = 100,
        avg_spot          = 2039.0,
        margin_per_spread = 1500,
        adv_contracts     = 400_000,
        max_pct_adv       = 0.05,
    ),
]

CORR = 0.4   # GC vs SI calendar spread correlation (same SOFR driver, different industrial)

sep = "-" * 120

def fmt(x):
    return f"{x:,.0f}"

rows_individual = []
rows_combined   = []

# ── Individual strategy analysis ─────────────────────────────────────────────
print("=" * 90)
print("  $50M AUM — INDIVIDUAL STRATEGY LEVERAGE & RETURN")
print("=" * 90)

for s in strategies:
    dol_per_bps     = s["avg_spot"] * s["oz_per_lot"] / 10_000
    ann_ret_per_lot = s["ann_ret_bps"] * dol_per_bps
    ann_vol_per_lot = s["ann_vol_bps"] * dol_per_bps

    # Calendar spread ADV ≈ 10–15% of outright; cap at 5% of that
    cal_adv = s["adv_contracts"] * 0.12
    liq_cap = int(cal_adv * s["max_pct_adv"])

    print(f"\n── {s['name']} ─────────────────────────────────────────────────────")
    print(f"   Per-lot: return=${ann_ret_per_lot:,.0f}/yr  vol=${ann_vol_per_lot:,.0f}/yr  "
          f"margin=${s['margin_per_spread']:,}  liquidity cap={liq_cap:,} lots")
    print()
    hdr = f"   {'Target Vol':>10}  {'Lots':>7}  {'Margin':>12}  {'Margin/AUM':>10}  "
    hdr += f"{'Gross Notl':>14}  {'Leverage':>9}  {'Strat Ret':>10}  {'+ T-bill':>9}  {'Total':>8}"
    print(hdr)
    print("   " + "-" * 105)

    for tv in [5, 10, 15, 20]:
        target_vol_usd = AUM * tv / 100
        lots_raw = int(target_vol_usd / ann_vol_per_lot)
        lots     = min(lots_raw, liq_cap)
        flag     = "  ← LIQ CAP" if lots_raw > liq_cap else ""

        margin      = lots * s["margin_per_spread"]
        margin_pct  = margin / AUM * 100
        gross_notl  = lots * s["avg_spot"] * s["oz_per_lot"] * 2
        leverage    = gross_notl / AUM
        actual_vol  = lots * ann_vol_per_lot / AUM * 100
        strat_ret   = lots * ann_ret_per_lot / AUM * 100
        tbill       = (AUM - margin) * RFR / AUM * 100
        total       = strat_ret + tbill

        print(f"   {tv:>9}%  {lots:>7,}  ${margin:>10,.0f}  {margin_pct:>9.1f}%  "
              f"${gross_notl:>12,.0f}  {leverage:>8.1f}x  {strat_ret:>9.1f}%  {tbill:>8.1f}%  "
              f"{total:>7.1f}%{flag}")

        rows_individual.append({
            "strategy": s["name"],
            "target_vol_pct": tv,
            "lots": lots,
            "margin_USD": margin,
            "margin_pct": round(margin_pct, 1),
            "gross_notional": gross_notl,
            "leverage_x": round(leverage, 1),
            "actual_vol_pct": round(actual_vol, 1),
            "strat_ret_pct": round(strat_ret, 1),
            "tbill_pct": round(tbill, 1),
            "total_ret_pct": round(total, 1),
            "liq_capped": lots_raw > liq_cap,
        })

    lots_cap = liq_cap
    actual_vol_at_cap = lots_cap * ann_vol_per_lot / AUM * 100
    ret_at_cap = lots_cap * ann_ret_per_lot / AUM * 100
    print(f"\n   → Max feasible ({liq_cap:,} lots): actual vol={actual_vol_at_cap:.1f}%  "
          f"strat return={ret_at_cap:.1f}%  total={ret_at_cap + (AUM - liq_cap*s['margin_per_spread'])*RFR/AUM*100:.1f}%")


# ── Combined portfolio ────────────────────────────────────────────────────────
print()
print("=" * 90)
print("  COMBINED PORTFOLIO  (SI + GC cal_12 norm, 50/50 vol split)")
print(f"  Inter-strategy correlation assumed: {CORR}")
print("=" * 90)

si = strategies[0]; gc = strategies[1]
si_dol = si["avg_spot"] * si["oz_per_lot"] / 10_000
gc_dol = gc["avg_spot"] * gc["oz_per_lot"] / 10_000
si_vol_lot = si["ann_vol_bps"] * si_dol
gc_vol_lot = gc["ann_vol_bps"] * gc_dol
si_ret_lot = si["ann_ret_bps"] * si_dol
gc_ret_lot = gc["ann_ret_bps"] * gc_dol
si_liq = int(si["adv_contracts"] * 0.12 * si["max_pct_adv"])
gc_liq = int(gc["adv_contracts"] * 0.12 * gc["max_pct_adv"])

print()
hdr2 = f"  {'Target Vol':>10}  {'SI lots':>8}  {'GC lots':>8}  {'Margin':>12}  "
hdr2 += f"{'Gross Notl':>14}  {'Leverage':>9}  {'Port Vol':>9}  {'Strat Ret':>10}  {'Total':>9}  {'Sharpe':>7}"
print(hdr2)
print("  " + "-" * 110)

for tv in [5, 10, 15, 20]:
    tvu = AUM * tv / 100
    si_lots = min(int((tvu / 2) / si_vol_lot), si_liq)
    gc_lots = min(int((tvu / 2) / gc_vol_lot), gc_liq)

    si_vol_tot = si_lots * si_vol_lot
    gc_vol_tot = gc_lots * gc_vol_lot
    port_vol   = np.sqrt(si_vol_tot**2 + gc_vol_tot**2 + 2 * CORR * si_vol_tot * gc_vol_tot)
    port_vol_pct = port_vol / AUM * 100

    total_margin = si_lots * si["margin_per_spread"] + gc_lots * gc["margin_per_spread"]
    margin_pct   = total_margin / AUM * 100
    gross_notl   = (si_lots * si["avg_spot"] * si["oz_per_lot"] * 2 +
                    gc_lots * gc["avg_spot"] * gc["oz_per_lot"] * 2)
    leverage     = gross_notl / AUM
    strat_ret    = (si_lots * si_ret_lot + gc_lots * gc_ret_lot) / AUM * 100
    tbill        = (AUM - total_margin) * RFR / AUM * 100
    total        = strat_ret + tbill
    eff_sharpe   = strat_ret / port_vol_pct

    print(f"  {tv:>9}%  {si_lots:>8,}  {gc_lots:>8,}  ${total_margin:>10,.0f}  "
          f"${gross_notl:>12,.0f}  {leverage:>8.1f}x  {port_vol_pct:>8.1f}%  "
          f"{strat_ret:>9.1f}%  {total:>8.1f}%  {eff_sharpe:>6.2f}")

    rows_combined.append({
        "target_vol_pct": tv,
        "si_lots": si_lots,
        "gc_lots": gc_lots,
        "margin_USD": total_margin,
        "margin_pct": round(margin_pct, 1),
        "gross_notional": gross_notl,
        "leverage_x": round(leverage, 1),
        "port_vol_pct": round(port_vol_pct, 1),
        "strat_ret_pct": round(strat_ret, 1),
        "tbill_pct": round(tbill, 1),
        "total_ret_pct": round(total, 1),
        "strategy_sharpe": round(eff_sharpe, 2),
    })

print()
print("  Notes:")
print(f"  - T-bill rate assumed: {RFR*100:.1f}% (uninvested capital earns T-bill yield)")
print(f"  - Liquidity cap: SI cal ≈ {si_liq:,} lots/day, GC cal ≈ {gc_liq:,} lots/day")
print(f"  - Gross notional = sum of both legs of each calendar spread")
print(f"  - 'Sharpe' column is strategy-only (excl T-bill), vs port vol")
print(f"  - Calendar spread ADV ≈ 12% of outright ADV, capped at 5% of that")

# Save CSVs
BASE = __import__("pathlib").Path(__file__).parent
pd.DataFrame(rows_individual).to_csv(BASE / "leverage_individual.csv", index=False)
pd.DataFrame(rows_combined).to_csv(BASE / "leverage_combined.csv", index=False)
print("\n  Saved: leverage_individual.csv, leverage_combined.csv")


if __name__ == "__main__":
    pass
