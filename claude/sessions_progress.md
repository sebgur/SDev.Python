# Trading sessions / "live" — progress notes

Last updated: 2026-11-09 (92 tests passing). Work is step-by-step: Claude proposes code, user types it in.

## Goal
Detect whether a date/datetime is "live", i.e. belongs to the current session of the user's branch,
with sessions that start/end at different local times per center.

## Decisions
- **Live is judged from the user's branch** (desk location), not the instrument's market.
  Branch = TradingCenter(tz, local holiday calendar, roll = branch's rates EOD snapshot time).
- **Inputs**
  - `dt.date` -> trade-date label (never localized).
  - aware `dt.datetime` -> moment.
  - naive `dt.datetime` at midnight -> date label (sdevpy passes valdates this way).
  - naive `dt.datetime` at any other time -> moment in the naive tz (machine-local by default,
    `set_naive_tz` overrides). So `dt.datetime.now()` is the current moment.
  - Live requests: None, 'live', 'now' (case-insensitive).
- **Sessions**: session T runs from roll (local) on the day before T to roll on T.
- **Gaps** (weekends, holidays): dates and moments map back to the last session (`BDC.P`), and nothing
  is live in a gap (both x and now must be in a session).
- **Clock**: `now` defaults to `dt.datetime.now(dt.UTC)`; tests, backtests and reruns pass `now`.
- **T-1 EOD**: business day before the trade date (Sat in a gap: session Fri, T-1 Thu).
- Trading hours (open/close) do not enter `is_live`; only for a future `is_open`.

## Done (`sdevpy/utilities/sessions.py`, tests in `sdevpy/tests/test_sessions.py`)
- `normalize`, `localize`, `set_naive_tz`.
- `TradingCenter` (frozen dataclass): `session_date`, `in_session`, `is_live`, `to_session_date`,
  `previous_eod`; private `_rolled_date`, `_unadjusted_date` (kept on purpose), `_session_of`.
- Registry: `CENTER_SPECS` (branches + exchanges XNYS, XLON, XTKS, CME, FX), `CENTER_ALIASES`
  (NY, SG, TK, ...; case/space-insensitive via `center_name`), `make_center` -> cached `_make_center`.
- Default center: `DEFAULT_CENTER = "NEWYORK"`, `set_default_center`, `default_center`, `get_center`.
- Module level: `is_live`, `resolve_date_input -> (trade date, is_live)`, `previous_eod`,
  `get_date_info -> {session, islive, tm1eod}`.
- `swap_series` (`sdevpy/timeseries/swaptimeseries.py`) calls `resolve_date_input` at the top
  (kwargs `center`, `now`); the rest of the function is a stub.
- Validated by eye: week simulation every 15 min (`__main__`, Singapore clock) for NY, SG, TK.
  Holidays seen: Veterans Day 11 Nov 2026 (USD calendar; NYSE open, XNYS differs),
  Deepavali observed Mon 9 Nov 2026 (SGD).

## Roll times (rates EOD)
- Branches: NEWYORK 17:00, LONDON 16:30, TOKYO 15:00, SYDNEY 16:30, AUCKLAND 17:00, SINGAPORE 17:00,
  MUMBAI 17:00, DUBAI 16:00. Exchanges: XLON 16:30, XTKS 15:00, others 17:00.
- Confirmed by the user (2026-11-09).

## Status: sessions work complete (2026-11-09)
- Next project (separate): implement `swap_series` live quotes, i.e. replace the last row with live
  quotes when the end date is live. Needs a live source and date listing in the calibration repository.

## Dropped / out of scope
- Step 4 (`Session` dataclass, session boundaries, current/last-closed/next session, `is_open`): no
  consumer without separate trading hours. Prototype kept in `claude/scratch/session_check.py`
  (roll-to-roll windows in UTC, DST-safe via `datetime.combine(d, roll, tzinfo=tz)`). Revisit if a
  scheduler/batch trigger needs session boundaries.
- Step 5 (open/close from pandas_market_calendars), Step 7 (multi-center liveness).
- Step 8 (integration across sdevpy: fixings, market data, pricer valuation date, backtests):
  to be done later, in another context.
- UAE Fri/Sat weekend before 2022: not a concern.
