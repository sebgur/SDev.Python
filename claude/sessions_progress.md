# Trading sessions / "live" — progress notes

Last updated: 2026-11-07 (84 tests passing). Work is step-by-step: Claude proposes code, user types it in.

## Goal
Detect whether a date/datetime is "live", i.e. belongs to a trading center's current session,
with sessions that start/end at different local times per center.

## Decisions made
- **Inputs**
  - `dt.date` -> trade-date label.
  - aware `dt.datetime` -> moment (instant).
  - naive `dt.datetime` at midnight -> treated as a date (sdevpy passes valdates this way).
  - naive `dt.datetime` at any other time -> `ValueError`.
  - To mean an actual midnight moment, pass an aware datetime.
- **Sessions**: the session for trade date T runs from roll time (local) on the calendar day before T
  to roll time on T. Default roll 17:00 local (e.g. Mon session starts Sun 17:00 NY).
- **Gaps** (weekends, holidays): moments and dates map *backwards* to the last session (`BDC.P`).
  E.g. Fri after 17:00, Sat, Sun before 17:00 -> Fri.
- **Clock**: `is_live(x, now=None)`; `now` defaults to `dt.datetime.now(dt.UTC)`. Tests, backtests
  and batch reruns pass an explicit `now`. Naive `now` raises.
- Trading hours (open/close) do not enter `is_live`; they only matter for a future `is_open`.

## Done (in the library, 39 tests passing)
- `sdevpy/utilities/sessions.py`
  - `normalize(x)`: returns `date | datetime` (check `datetime` first, since it subclasses `date`).
  - `TradingCenter(name, tz: ZoneInfo, calendar: cdr.Calendar, roll=dt.time(17))`
    (frozen dataclass), with methods:
    - `session_date(t)`: aware moment -> trade date.
    - `to_session_date(x)`: any input -> trade date.
    - `is_live(x, now=None)`.
- `sdevpy/tests/test_sessions.py`: normalize, session_date (NY, DST, Tokyo input, Thanksgiving),
  is_live.

- **Step 6 (done, 52 tests passing)**: registry `make_center(name)` (cached) plus a `CENTER_SPECS`
  table of (tz, calendar name, roll). Centers: XNYS, XLON, XTKS, CME (Chicago, CME_Equity calendar),
  FX (NY, weekends only). All roll at 17:00 local.

## Key reframing (decided)
- "Live" is judged from the **user's branch** (desk location), not the instrument's market.
  Each branch = TradingCenter(tz, local holiday calendar, roll = company EOD time, fixed clock time).
- Branches: SYDNEY, TOKYO, SINGAPORE, AUCKLAND, MUMBAI, DUBAI, LONDON, NEWYORK. EOD times are
  placeholders (17:00 local) until the user gives real ones.

## Step 6b (done, 65 tests passing)
- Branch entries in `CENTER_SPECS`; `"AED"` added to `calendars.CCY_CALENDARS`.
- Default center: `set_default_center(name)` (code) overrides env var `SDEVPY_CENTER`;
  `default_center()`, `get_center(center)` (None/str/TradingCenter), module-level
  `is_live(x, center=None, now=None)`.

## Gaps and resolve_end_date (done, 84 tests passing)
- `ssn.resolve_end_date(end_date, center=None, now=None) -> (trade date, is_live)`; end_date None/'live'
  -> current session. Used at the top of `swap_series` (kwargs `center`, `now`), end date converted with
  `cdr.to_datetime`. Rest of `swap_series` (historical series, live quotes source) not designed yet:
  calibration repository has no live source nor date listing.
- Decided: **nothing is live in gaps**. After Fri EOD until Sun roll (and holiday gaps), is_live is False,
  but dates still map back (Sat -> Fri). New `TradingCenter.in_session(t)`; `is_live` and
  `resolve_end_date` require both `now` and `x` to be in a session (a Saturday date or moment is
  never live, whatever the clock says). Rest of `swap_series` postponed by the user.

## Next
- Use in `sdevpy/timeseries/swaptimeseries.py`: `swap_series(..., end_date=None|'live'|date)` replaces the
  last row with live quotes when end_date is live for the user's branch (optional `center` arg).

## Remaining plan
- **Step 4**: `Session` dataclass (trade_date, open_utc, close_utc), `center.session(d)`, `is_open(t)`,
  current/next/previous session. Local times + tz, UTC per date (DST-safe).
- **Step 5**: optionally take open/close (incl. early closes) from `pandas_market_calendars.schedule()`.
- **Step 7**: liveness for multi-center objects (reference center, or all/any). Wait for a use case.
- **Step 8**: integrate where sdevpy branches on live vs historical (fixings, market data, MC/pricer
  valuation date, backtesting loop passing simulated `now`).

## Open questions
- Per-center roll times for the registry (equities: 17:00 local, or a different time per exchange?).
