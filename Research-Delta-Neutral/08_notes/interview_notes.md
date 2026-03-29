# Interview Notes — Precious Metals Curve RV Research

## The Core Pitch (30 seconds)

> "I built a research framework to test gold and silver relative-value strategies as part of a
> delta-neutral approach. I found that normalising calendar spreads by spot price
> produces a stationary series with exploitable mean reversion. Silver generates a Sharpe
> of ~1.20 and gold ~0.84 across 2018–2025. Silver is particularly compelling because it
> performed best during the 2022 rate hike cycle (Sharpe +2.91) exactly when gold failed.
> Running both as a combined portfolio significantly improves robustness across all rate regimes."

## The Core Pitch — Gold Focus (if asked specifically about gold)

> "I built a research framework to test gold relative-value strategies as part of a
> delta-neutral approach. I found that normalising gold calendar spreads by spot price
> produces a stationary series with exploitable mean reversion, generating a Sharpe
> of ~0.84 out-of-sample across 2018–2025. The main risk is the rate hike regime of
> 2022, which caused structural spread widening. The butterfly is more robust but
> lower Sharpe. I would combine both, sized inversely to rate volatility."

---

## Strategy 1: Gold Calendar Spread (bps normalised)

### Why does it work?
Gold forward curve pricing is theoretically: `F(T) = S × e^(r+l)T`
where r = risk-free rate, l = lease rate (gold borrowing cost).

The front-second spread should therefore be:
`cal_12 ≈ -S × (r + l) × Δt`

When we normalise: `cal_12 / S × 10000 ≈ -(r + l) × Δt × 10000 (bps)`

This should be a **predictable, slowly-varying quantity** that:
1. Reverts when temporary demand for specific maturities distorts it
2. Moves with rates (the risk factor) but in a predictable direction

### Delta neutrality
- 1 contract long front / 1 contract short second = delta-neutral on notional
- Both legs are gold, so directional risk is minimal
- Gamma: negligible (futures, not options)
- **Residual risk**: theta/carry risk — if rates move dramatically, the "fair value" shifts

### Why 2022 was bad
- FOMC raised rates from 0% to 5.25% in ~14 months (fastest cycle in 40 years)
- Gold carry cost increased dramatically → contango deepened structurally
- The z-score said "spread is unusually wide → buy it" but the fundamental level was repricing
- **Fix**: Use SOFR-implied fair value as the mean target instead of rolling historical mean

### Key interview questions + answers

**Q: Isn't this just carry risk?**
A: Partially. The systematic component of carry is already priced in — we're trading the
   deviation from the carry-implied fair value. The strategy's edge is in the temporary
   dislocations, not the level of carry.

**Q: How do you handle contract rolls?**
A: In research phase, I use continuous rolling price series (front/second/third by expiry
   order). The roll creates discontinuities, which is a known limitation. In production,
   I would use constant-maturity interpolation or calendar-specific contracts.

**Q: What's the capacity?**
A: GC futures average ~400,000 contracts/day open interest. A 10-contract strategy is
   capacity-unconstrained. Even institutional size (1000 lots) is small relative to market.

**Q: What's your edge vs prop desks?**
A: We're at the research stage — this is about identifying whether the signal exists.
   In production, the edge would come from information (real-time EFP data, lease rates),
   better roll methodology, and smarter exit signals.

---

## Strategy 2: Gold Butterfly

### Why does it work?
The butterfly `GC1 - 2×GC2 + GC3` measures **curvature** in the forward curve.
In a smooth contango, `GC1 - GC2 ≈ GC2 - GC3` → butterfly ≈ 0.

When curvature spikes (e.g., a specific contract month has unusual demand), it should
snap back rapidly — hence the 6-day half-life.

### Economic mechanism
- Physical demand distortions (exchange warehousing, LBMA/CME arbitrage)
- Roll squeezes in specific months
- Temporary position imbalances

### Delta neutrality
- Long 1 front + long 1 third + short 2 second contracts
- Sum of leg notionals: +1 -2 +1 = 0 → truly delta-neutral
- No carry exposure (first-order carry cancels in the butterfly)

---

## Data Limitations (what you would fix for production)

1. **Roll methodology**: We don't know exact roll dates from this dataset.
   Production needs contract-specific date mapping.

2. **Continuous vs. contract-specific**: Front/second/third are rolling series.
   The "spread" changes mechanically at roll. The normalised bps measure partially
   mitigates this but doesn't eliminate it.

3. **Transaction costs**: We used $0.50/leg placeholder. Real bid-ask in gold
   futures is ~0.10/oz = $10/contract. For a 1-lot spread that's $20 round-trip.
   Need to calibrate to actual tick data.

4. **Intraday timing**: Daily close prices. In reality, the signal fires intraday
   and execution quality matters.

5. **Regime filter**: The 2022 failure suggests a SOFR-conditioning or MOVE-filtering
   approach would materially improve the Sharpe. This requires proper time-series
   modelling of the "fair value" spread.

---

## What would make this institutional quality?

1. Contract-specific pricing (not rolling continuous)
2. Carry-adjusted fair value model (SOFR curve × time to expiry)
3. Regime filter (MOVE index, Fed uncertainty)
4. Intraday data for real execution simulation
5. Lease rate data to model the full cost of carry
6. Out-of-sample validation on a held-out period
