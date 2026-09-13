import React, { useState, useMemo } from "react";

/* ------------------------------------------------------------------ */
/*  Holiday calendars                                                  */
/*  UK = England & Wales bank holidays                                 */
/*  US = Federal / Fedwire holidays (USD value-date calendar)          */
/* ------------------------------------------------------------------ */

const UK_HOLIDAYS = {
  "2024-01-01": "New Year's Day",
  "2024-03-29": "Good Friday",
  "2024-04-01": "Easter Monday",
  "2024-05-06": "Early May",
  "2024-05-27": "Spring",
  "2024-08-26": "Summer",
  "2024-12-25": "Christmas Day",
  "2024-12-26": "Boxing Day",

  "2025-01-01": "New Year's Day",
  "2025-04-18": "Good Friday",
  "2025-04-21": "Easter Monday",
  "2025-05-05": "Early May",
  "2025-05-26": "Spring",
  "2025-08-25": "Summer",
  "2025-12-25": "Christmas Day",
  "2025-12-26": "Boxing Day",

  "2026-01-01": "New Year's Day",
  "2026-04-03": "Good Friday",
  "2026-04-06": "Easter Monday",
  "2026-05-04": "Early May",
  "2026-05-25": "Spring",
  "2026-08-31": "Summer",
  "2026-12-25": "Christmas Day",
  "2026-12-28": "Boxing Day (substitute)",

  "2027-01-01": "New Year's Day",
  "2027-03-26": "Good Friday",
  "2027-03-29": "Easter Monday",
  "2027-05-03": "Early May",
  "2027-05-31": "Spring",
  "2027-08-30": "Summer",
  "2027-12-27": "Christmas Day (substitute)",
  "2027-12-28": "Boxing Day (substitute)",

  "2028-01-03": "New Year's Day (substitute)",
  "2028-04-14": "Good Friday",
  "2028-04-17": "Easter Monday",
  "2028-05-01": "Early May",
  "2028-05-29": "Spring",
  "2028-08-28": "Summer",
  "2028-12-25": "Christmas Day",
  "2028-12-26": "Boxing Day",
};

const US_HOLIDAYS = {
  "2024-01-01": "New Year's Day",
  "2024-01-15": "Martin Luther King Jr. Day",
  "2024-02-19": "Presidents' Day",
  "2024-05-27": "Memorial Day",
  "2024-06-19": "Juneteenth",
  "2024-07-04": "Independence Day",
  "2024-09-02": "Labor Day",
  "2024-10-14": "Columbus Day",
  "2024-11-11": "Veterans Day",
  "2024-11-28": "Thanksgiving",
  "2024-12-25": "Christmas Day",

  "2025-01-01": "New Year's Day",
  "2025-01-20": "Martin Luther King Jr. Day",
  "2025-02-17": "Presidents' Day",
  "2025-05-26": "Memorial Day",
  "2025-06-19": "Juneteenth",
  "2025-07-04": "Independence Day",
  "2025-09-01": "Labor Day",
  "2025-10-13": "Columbus Day",
  "2025-11-11": "Veterans Day",
  "2025-11-27": "Thanksgiving",
  "2025-12-25": "Christmas Day",

  "2026-01-01": "New Year's Day",
  "2026-01-19": "Martin Luther King Jr. Day",
  "2026-02-16": "Presidents' Day",
  "2026-05-25": "Memorial Day",
  "2026-06-19": "Juneteenth",
  "2026-07-03": "Independence Day (observed)",
  "2026-09-07": "Labor Day",
  "2026-10-12": "Columbus Day",
  "2026-11-11": "Veterans Day",
  "2026-11-26": "Thanksgiving",
  "2026-12-25": "Christmas Day",

  "2027-01-01": "New Year's Day",
  "2027-01-18": "Martin Luther King Jr. Day",
  "2027-02-15": "Presidents' Day",
  "2027-05-31": "Memorial Day",
  "2027-06-18": "Juneteenth (observed)",
  "2027-07-05": "Independence Day (observed)",
  "2027-09-06": "Labor Day",
  "2027-10-11": "Columbus Day",
  "2027-11-11": "Veterans Day",
  "2027-11-25": "Thanksgiving",
  "2027-12-24": "Christmas Day (observed)",
  "2027-12-31": "New Year's Day (observed)",

  "2028-01-17": "Martin Luther King Jr. Day",
  "2028-02-21": "Presidents' Day",
  "2028-05-29": "Memorial Day",
  "2028-06-19": "Juneteenth",
  "2028-07-04": "Independence Day",
  "2028-09-04": "Labor Day",
  "2028-10-09": "Columbus Day",
  "2028-11-10": "Veterans Day (observed)",
  "2028-11-23": "Thanksgiving",
  "2028-12-25": "Christmas Day",
};

const CALENDAR_MIN = "2024-01-01";
const CALENDAR_MAX = "2028-12-31";

/* ------------------------------------------------------------------ */
/*  Date helpers (all UTC to avoid timezone drift)                     */
/* ------------------------------------------------------------------ */

const MS_DAY = 86400000;
const parseISO = (s) => {
  const [y, m, d] = s.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d));
};
const toISO = (dt) => dt.toISOString().slice(0, 10);
const addDays = (dt, n) => new Date(dt.getTime() + n * MS_DAY);
const isWeekend = (dt) => dt.getUTCDay() === 0 || dt.getUTCDay() === 6;
const DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const dayLabel = (dt) => DOW[dt.getUTCDay()];
const prettyDate = (dt) =>
  `${String(dt.getUTCDate()).padStart(2, "0")} ${
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][dt.getUTCMonth()]
  } ${dt.getUTCFullYear()}`;

/* Returns a holiday name if the date is a holiday on the chosen basis. */
function holidayOn(dt, basis) {
  const k = toISO(dt);
  const uk = UK_HOLIDAYS[k];
  const us = US_HOLIDAYS[k];
  if (basis === "uk") return uk ? `London — ${uk}` : null;
  if (basis === "us") return us ? `New York — ${us}` : null;
  if (uk && us) return uk === us ? `London & New York — ${uk}` : `London — ${uk} / New York — ${us}`;
  if (uk) return `London — ${uk}`;
  if (us) return `New York — ${us}`;
  return null;
}

const isGoodBusinessDay = (dt, basis) => !isWeekend(dt) && !holidayOn(dt, basis);

/* Roll forward n good business days. n = 0 rolls only if the date is bad. */
function rollForward(dt, n, basis) {
  let d = new Date(dt.getTime());
  if (n <= 0) {
    while (!isGoodBusinessDay(d, basis)) d = addDays(d, 1);
    return d;
  }
  let count = 0;
  while (count < n) {
    d = addDays(d, 1);
    if (isGoodBusinessDay(d, basis)) count++;
  }
  return d;
}

/* ------------------------------------------------------------------ */
/*  Formatting                                                         */
/* ------------------------------------------------------------------ */

const fmt = (n, dp = 2) =>
  Number.isFinite(n)
    ? n.toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp })
    : "—";
const signed = (n, dp = 4) => (n >= 0 ? "+" : "−") + fmt(Math.abs(n), dp);

/* ------------------------------------------------------------------ */
/*  Palette                                                            */
/* ------------------------------------------------------------------ */

const C = {
  bg: "#13161B",
  panel: "#1A1E25",
  panelAlt: "#20252D",
  line: "#2C333D",
  lineSoft: "#242A32",
  text: "#E6E9ED",
  muted: "#8D96A3",
  faint: "#606A78",
  brass: "#D8A94A",
  brassDim: "#8A7134",
  pos: "#46B98C",
  neg: "#E0605A",
  warn: "#D8A94A",
};

const MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace";
const SANS =
  "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";

/* ------------------------------------------------------------------ */
/*  Small building blocks                                              */
/* ------------------------------------------------------------------ */

function Field({ label, hint, children }) {
  return (
    <label style={{ display: "block", marginBottom: 16 }}>
      <div
        style={{
          fontSize: 12.5,
          color: C.muted,
          marginBottom: 6,
          display: "flex",
          justifyContent: "space-between",
          gap: 8,
        }}
      >
        <span>{label}</span>
        {hint && <span style={{ color: C.faint, fontFamily: MONO, fontSize: 11 }}>{hint}</span>}
      </div>
      {children}
    </label>
  );
}

const inputStyle = {
  width: "100%",
  boxSizing: "border-box",
  background: C.bg,
  border: `1px solid ${C.line}`,
  borderRadius: 4,
  color: C.text,
  fontFamily: MONO,
  fontSize: 15,
  padding: "9px 11px",
  outline: "none",
};

function Seg({ options, value, onChange }) {
  return (
    <div style={{ display: "flex", border: `1px solid ${C.line}`, borderRadius: 4, overflow: "hidden" }}>
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            onClick={() => onChange(o.value)}
            style={{
              flex: 1,
              background: on ? C.brassDim : "transparent",
              color: on ? "#FFF6E2" : C.muted,
              border: "none",
              borderRight: `1px solid ${C.line}`,
              padding: "8px 4px",
              fontSize: 12.5,
              fontFamily: SANS,
              cursor: "pointer",
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

function Stat({ label, value, note, mono = true }) {
  return (
    <div style={{ padding: "11px 0", borderBottom: `1px solid ${C.lineSoft}` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
        <span style={{ fontSize: 13, color: C.muted }}>{label}</span>
        <span style={{ fontFamily: mono ? MONO : SANS, fontSize: 14.5, color: C.text }}>{value}</span>
      </div>
      {note && <div style={{ fontSize: 11.5, color: C.faint, marginTop: 4 }}>{note}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

export default function AsianSwapCalculator() {
  const [spot, setSpot] = useState("4000.00");
  const [rate, setRate] = useState("3.50");
  const [start, setStart] = useState("2026-10-01");
  const [end, setEnd] = useState("2026-10-31");
  const [tradeDate, setTradeDate] = useState("2026-09-14");

  // --- optional inputs (see note in the response) ---
  const [notional, setNotional] = useState("10000");
  const [basis, setBasis] = useState("both"); // uk | us | both
  const [dayCount, setDayCount] = useState(360); // 360 | 365
  const [spotLag, setSpotLag] = useState(2); // 0 | 1 | 2
  const [showSchedule, setShowSchedule] = useState(false);

  const calc = useMemo(() => {
    const S = parseFloat(spot);
    const r = parseFloat(rate) / 100;
    const oz = parseFloat(notional);

    if (!start || !end || !tradeDate) return { error: "Enter a trade date and an observation window." };
    if (!Number.isFinite(S) || !Number.isFinite(r)) return { error: "Enter a spot price and a swap rate." };
    const sDt = parseISO(start);
    const eDt = parseISO(end);
    const tDt = parseISO(tradeDate);
    if (eDt < sDt) return { error: "The observation end date falls before the start date." };
    if (toISO(sDt) < CALENDAR_MIN || toISO(eDt) > CALENDAR_MAX)
      return { error: `Holiday calendars cover ${CALENDAR_MIN} to ${CALENDAR_MAX}. Extend them to price outside that range.` };

    // Observation schedule: every good business day in the window.
    const fixings = [];
    const skipped = [];
    for (let d = new Date(sDt.getTime()); d <= eDt; d = addDays(d, 1)) {
      const hol = holidayOn(d, basis);
      if (isWeekend(d)) continue;
      if (hol) {
        skipped.push({ date: new Date(d.getTime()), reason: hol });
        continue;
      }
      fixings.push(new Date(d.getTime()));
    }
    if (fixings.length === 0) return { error: "No business days fall inside that observation window." };

    const spotVD = rollForward(tDt, spotLag, basis);

    const rows = fixings.map((f) => {
      const vd = rollForward(f, spotLag, basis);
      const days = Math.round((vd - spotVD) / MS_DAY);
      const fwd = S * (1 + r * (days / dayCount));
      return { fix: f, vd, days, fwd };
    });

    const n = rows.length;
    const avgDays = rows.reduce((a, x) => a + x.days, 0) / n;
    const swapPrice = S * (1 + r * (avgDays / dayCount));
    const adjPerOz = swapPrice - S;
    const totalAdj = Number.isFinite(oz) ? adjPerOz * oz : null;

    // Calendar-midpoint comparison: what a naive mid-date carry would give.
    const midMs = (sDt.getTime() + eDt.getTime()) / 2;
    const midDt = new Date(Math.round(midMs / MS_DAY) * MS_DAY);
    const midVD = rollForward(midDt, spotLag, basis);
    const midDays = Math.round((midVD - spotVD) / MS_DAY);
    const midAdj = S * r * (midDays / dayCount);

    const preTrade = rows.filter((x) => x.days < 0).length;

    return {
      rows, skipped, n, avgDays, swapPrice, adjPerOz, totalAdj,
      spotVD, midDt, midDays, midAdj, preTrade, S, oz,
      firstFix: rows[0].fix, lastFix: rows[n - 1].fix,
    };
  }, [spot, rate, start, end, tradeDate, notional, basis, dayCount, spotLag]);

  const err = calc.error;
  const adj = err ? 0 : calc.adjPerOz;
  const adjColor = err ? C.faint : adj >= 0 ? C.brass : C.neg;

  return (
    <div
      style={{
        background: C.bg,
        color: C.text,
        fontFamily: SANS,
        minHeight: "100%",
        padding: "28px 24px 40px",
        WebkitFontSmoothing: "antialiased",
      }}
    >
      <style>{`
        input:focus, select:focus, button:focus-visible {
          border-color: ${C.brass} !important;
          box-shadow: 0 0 0 2px rgba(216,169,74,0.22);
        }
        input[type="date"]::-webkit-calendar-picker-indicator { filter: invert(0.65); cursor: pointer; }
        .sched-row:hover { background: ${C.panelAlt}; }
        @media (max-width: 820px) { .cols { grid-template-columns: 1fr !important; } }
      `}</style>

      <div style={{ maxWidth: 1020, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 22 }}>
          <h1 style={{ fontSize: 19, fontWeight: 600, margin: 0, letterSpacing: -0.1 }}>
            Asian swap calculator
          </h1>
          <p style={{ margin: "6px 0 0", fontSize: 13.5, color: C.muted, maxWidth: 640, lineHeight: 1.55 }}>
            Fixed against the average of daily fixings over an observation window. Prices the carry
            adjustment to spot on business days common to both calendars.
          </p>
        </div>

        {/* Result strip */}
        <div
          style={{
            background: C.panel,
            border: `1px solid ${C.line}`,
            borderRadius: 6,
            padding: "22px 24px",
            marginBottom: 20,
          }}
        >
          {err ? (
            <div style={{ color: C.warn, fontSize: 14, padding: "12px 0" }}>{err}</div>
          ) : (
            <>
              <div
                style={{
                  display: "flex",
                  alignItems: "flex-end",
                  flexWrap: "wrap",
                  gap: 18,
                }}
              >
                <div>
                  <div style={{ fontSize: 12.5, color: C.muted, marginBottom: 5 }}>Spot</div>
                  <div style={{ fontFamily: MONO, fontSize: 24, color: C.muted }}>{fmt(calc.S, 2)}</div>
                </div>
                <div style={{ fontSize: 22, color: C.faint, paddingBottom: 3 }}>
                  {adj >= 0 ? "+" : "−"}
                </div>
                <div>
                  <div style={{ fontSize: 12.5, color: C.brass, marginBottom: 5 }}>Carry adjustment</div>
                  <div style={{ fontFamily: MONO, fontSize: 40, lineHeight: 1, color: adjColor }}>
                    {fmt(Math.abs(adj), 4)}
                  </div>
                </div>
                <div style={{ fontSize: 22, color: C.faint, paddingBottom: 3 }}>=</div>
                <div>
                  <div style={{ fontSize: 12.5, color: C.muted, marginBottom: 5 }}>Swap fixed price</div>
                  <div style={{ fontFamily: MONO, fontSize: 24 }}>{fmt(calc.swapPrice, 2)}</div>
                </div>
                <div style={{ fontSize: 12.5, color: C.faint, paddingBottom: 7 }}>$/oz</div>
              </div>

              <div
                style={{
                  marginTop: 20,
                  paddingTop: 16,
                  borderTop: `1px solid ${C.lineSoft}`,
                  display: "flex",
                  flexWrap: "wrap",
                  gap: "10px 34px",
                  fontSize: 13,
                  color: C.muted,
                }}
              >
                <span>
                  {calc.n} fixings · {prettyDate(calc.firstFix)} to {prettyDate(calc.lastFix)}
                </span>
                <span>
                  Average tenor{" "}
                  <span style={{ fontFamily: MONO, color: C.text }}>{fmt(calc.avgDays, 1)}</span> days
                </span>
                {calc.totalAdj !== null && (
                  <span>
                    On {fmt(calc.oz, 0)} oz:{" "}
                    <span style={{ fontFamily: MONO, color: adjColor }}>
                      {signed(calc.totalAdj, 2)}
                    </span>
                  </span>
                )}
              </div>
            </>
          )}
        </div>

        {/* Warnings */}
        {!err && calc.preTrade > 0 && (
          <div
            style={{
              background: "rgba(216,169,74,0.08)",
              border: `1px solid ${C.brassDim}`,
              borderRadius: 6,
              padding: "12px 16px",
              marginBottom: 20,
              fontSize: 13,
              color: C.warn,
              lineHeight: 1.55,
            }}
          >
            {calc.preTrade} of {calc.n} fixings fall before the spot value date. Every fixing is priced
            forward here — already-published fixings are not blended in. Split the window if part of it
            has fixed.
          </div>
        )}

        {/* Two columns */}
        <div
          className="cols"
          style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", gap: 20 }}
        >
          {/* Inputs */}
          <div
            style={{
              background: C.panel,
              border: `1px solid ${C.line}`,
              borderRadius: 6,
              padding: "20px 22px",
            }}
          >
            <div style={{ fontSize: 13.5, fontWeight: 600, marginBottom: 18 }}>Trade terms</div>

            <Field label="Spot price" hint="$/oz">
              <input style={inputStyle} value={spot} onChange={(e) => setSpot(e.target.value)} inputMode="decimal" />
            </Field>

            <Field label="Swap rate" hint={`% p.a. · ACT/${dayCount}`}>
              <input style={inputStyle} value={rate} onChange={(e) => setRate(e.target.value)} inputMode="decimal" />
            </Field>

            <Field label="Trade date" hint={err ? "" : `spot value ${toISO(calc.spotVD)}`}>
              <input type="date" style={inputStyle} value={tradeDate} onChange={(e) => setTradeDate(e.target.value)} />
            </Field>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <Field label="Observation start">
                <input type="date" style={inputStyle} value={start} onChange={(e) => setStart(e.target.value)} />
              </Field>
              <Field label="Observation end">
                <input type="date" style={inputStyle} value={end} onChange={(e) => setEnd(e.target.value)} />
              </Field>
            </div>

            <div style={{ height: 1, background: C.lineSoft, margin: "6px 0 20px" }} />

            <Field label="Notional" hint="oz">
              <input style={inputStyle} value={notional} onChange={(e) => setNotional(e.target.value)} inputMode="decimal" />
            </Field>

            <Field label="Fixing calendar">
              <Seg
                value={basis}
                onChange={setBasis}
                options={[
                  { value: "both", label: "London + NY" },
                  { value: "uk", label: "London" },
                  { value: "us", label: "New York" },
                ]}
              />
            </Field>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <Field label="Day count">
                <Seg
                  value={dayCount}
                  onChange={setDayCount}
                  options={[
                    { value: 360, label: "ACT/360" },
                    { value: 365, label: "ACT/365" },
                  ]}
                />
              </Field>
              <Field label="Value date lag">
                <Seg
                  value={spotLag}
                  onChange={setSpotLag}
                  options={[
                    { value: 0, label: "T+0" },
                    { value: 1, label: "T+1" },
                    { value: 2, label: "T+2" },
                  ]}
                />
              </Field>
            </div>
          </div>

          {/* Diagnostics */}
          <div
            style={{
              background: C.panel,
              border: `1px solid ${C.line}`,
              borderRadius: 6,
              padding: "20px 22px",
            }}
          >
            <div style={{ fontSize: 13.5, fontWeight: 600, marginBottom: 8 }}>Observation window</div>

            {err ? (
              <div style={{ color: C.faint, fontSize: 13, paddingTop: 10 }}>
                Results appear once the inputs above are complete.
              </div>
            ) : (
              <>
                <Stat label="Fixings in window" value={calc.n} />
                <Stat
                  label="Spot value date"
                  value={prettyDate(calc.spotVD)}
                  note={`T+${spotLag} from trade date, rolled on the fixing calendar`}
                />
                <Stat
                  label="Average days to fixing"
                  value={fmt(calc.avgDays, 2)}
                  note="Mean tenor from spot value date across all fixings"
                />
                <Stat
                  label="Calendar-midpoint tenor"
                  value={`${calc.midDays} days`}
                  note={`Shortcut estimate ${signed(calc.midAdj, 4)} — holidays move the true average by ${signed(
                    calc.adjPerOz - calc.midAdj,
                    4
                  )} $/oz`}
                />

                <div style={{ marginTop: 18 }}>
                  <div style={{ fontSize: 13, color: C.muted, marginBottom: 10 }}>
                    Holidays excluded ({calc.skipped.length})
                  </div>
                  {calc.skipped.length === 0 ? (
                    <div style={{ fontSize: 12.5, color: C.faint }}>
                      No bank holidays fall inside this window.
                    </div>
                  ) : (
                    calc.skipped.map((s) => (
                      <div
                        key={toISO(s.date)}
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          gap: 12,
                          fontSize: 12.5,
                          padding: "5px 0",
                          color: C.muted,
                        }}
                      >
                        <span style={{ fontFamily: MONO, color: C.text, whiteSpace: "nowrap" }}>
                          {prettyDate(s.date)}
                        </span>
                        <span style={{ textAlign: "right" }}>{s.reason}</span>
                      </div>
                    ))
                  )}
                </div>

                <button
                  type="button"
                  onClick={() => setShowSchedule((v) => !v)}
                  style={{
                    marginTop: 20,
                    width: "100%",
                    background: "transparent",
                    border: `1px solid ${C.line}`,
                    borderRadius: 4,
                    color: C.muted,
                    padding: "9px",
                    fontSize: 12.5,
                    fontFamily: SANS,
                    cursor: "pointer",
                  }}
                >
                  {showSchedule ? "Hide fixing schedule" : "Show fixing schedule"}
                </button>
              </>
            )}
          </div>
        </div>

        {/* Schedule table */}
        {!err && showSchedule && (
          <div
            style={{
              background: C.panel,
              border: `1px solid ${C.line}`,
              borderRadius: 6,
              marginTop: 20,
              overflow: "hidden",
            }}
          >
            <div style={{ maxHeight: 380, overflowY: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    {["#", "Fixing date", "Day", "Value date", "Days", "Forward $/oz"].map((h, i) => (
                      <th
                        key={h}
                        style={{
                          position: "sticky",
                          top: 0,
                          background: C.panelAlt,
                          borderBottom: `1px solid ${C.line}`,
                          padding: "10px 16px",
                          textAlign: i >= 4 ? "right" : "left",
                          fontWeight: 500,
                          fontSize: 12,
                          color: C.muted,
                        }}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {calc.rows.map((r, i) => (
                    <tr key={toISO(r.fix)} className="sched-row">
                      <td style={{ padding: "8px 16px", color: C.faint, fontFamily: MONO, borderBottom: `1px solid ${C.lineSoft}` }}>
                        {i + 1}
                      </td>
                      <td style={{ padding: "8px 16px", fontFamily: MONO, borderBottom: `1px solid ${C.lineSoft}` }}>
                        {toISO(r.fix)}
                      </td>
                      <td style={{ padding: "8px 16px", color: C.muted, borderBottom: `1px solid ${C.lineSoft}` }}>
                        {dayLabel(r.fix)}
                      </td>
                      <td style={{ padding: "8px 16px", fontFamily: MONO, color: C.muted, borderBottom: `1px solid ${C.lineSoft}` }}>
                        {toISO(r.vd)}
                      </td>
                      <td style={{ padding: "8px 16px", fontFamily: MONO, textAlign: "right", borderBottom: `1px solid ${C.lineSoft}` }}>
                        {r.days}
                      </td>
                      <td style={{ padding: "8px 16px", fontFamily: MONO, textAlign: "right", borderBottom: `1px solid ${C.lineSoft}` }}>
                        {fmt(r.fwd, 4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr>
                    <td colSpan={4} style={{ padding: "11px 16px", background: C.panelAlt, color: C.muted }}>
                      Average
                    </td>
                    <td style={{ padding: "11px 16px", background: C.panelAlt, fontFamily: MONO, textAlign: "right" }}>
                      {fmt(calc.avgDays, 1)}
                    </td>
                    <td style={{ padding: "11px 16px", background: C.panelAlt, fontFamily: MONO, textAlign: "right", color: C.brass }}>
                      {fmt(calc.swapPrice, 4)}
                    </td>
                  </tr>
                </tfoot>
              </table>
            </div>
          </div>
        )}

        <div style={{ marginTop: 22, fontSize: 12, color: C.faint, lineHeight: 1.65, maxWidth: 700 }}>
          Forward per fixing = Spot × (1 + rate × days / day count), simple interest, undiscounted.
          The swap level is the arithmetic mean of those forwards, so it equals the spot carried to the
          business-day-weighted average tenor. Holiday calendars run {CALENDAR_MIN} to {CALENDAR_MAX}.
        </div>
      </div>
    </div>
  );
}
