import datetime as dt
from dataclasses import dataclass
from functools import cache
from zoneinfo import ZoneInfo
from sdevpy.utilities import calendars as cdr

#################################### ToDo #############################################################################
# ToDo: Can't we do the weekly test by simulating it in one go as per the below, just defining a spaced grid of
# datetime in a certain timezone?

DEFAULT_CENTER = "NEWYORK"
_default_center_name: str|None=None
_naive_tz: dt.tzinfo|None=None  # None: machine-local


def set_naive_tz(tz: str|dt.tzinfo|None) -> None:
    """ Timezone in which naive datetimes are read (None: machine-local) """
    global _naive_tz
    _naive_tz = ZoneInfo(tz) if isinstance(tz, str) else tz


def localize(t: dt.datetime) -> dt.datetime:
    """ Aware version of t: naive t is read in the naive timezone (machine-local by default) """
    if t.utcoffset() is not None:
        return t
    return t.astimezone() if _naive_tz is None else t.replace(tzinfo=_naive_tz)


def normalize(x: dt.date|dt.datetime) -> dt.date|dt.datetime:
    """ Classify a date/datetime input as either a trade-date label or an instant.
        - dt.date: trade-date label, returned unchanged
        - aware dt.datetime: instant, returned unchanged
        - naive dt.datetime at midnight: trade-date label, returned as dt.date
        - naive dt.datetime at any other time: instant in the naive timezone (machine-local
          by default, see set_naive_tz), returned as an aware datetime
        Callers distinguish the result with isinstance(r, dt.datetime). """
    if isinstance(x, dt.datetime):
        if x.utcoffset() is None and x.time() == dt.time(0):
            return x.date()
        return localize(x)

    if isinstance(x, dt.date):
        return x

    raise TypeError(f"Expected date or datetime, got {type(x).__name__}")


@dataclass(frozen=True)
class TradingCenter:
    """ A trading center: its timezone, business-day calendar and roll time.
        The session for trade date T runs from roll time on the day before T to roll time on T
        (local time). Moments outside any session (weekends, holidays) belong to the last session. """
    name: str
    tz: ZoneInfo
    calendar: cdr.Calendar
    roll: dt.time = dt.time(17)

    def session_date(self, t: dt.datetime) -> dt.date:
        """ Trade date of the session that moment t (naive: read in the naive timezone) belongs to """
        return self.calendar.adjust(self._rolled_date(t), cdr.BDC.P)

    def is_live(self, x: dt.date|dt.datetime, now: dt.datetime|None=None) -> bool:
        """ True if x belongs to the current session. Nothing is live in gaps (weekends, holidays):
            neither x nor now may fall in a gap.
            Args:
            - now: defaults to the system clock, only useful for testing/replicating results """
        x_date, x_in_session = self._session_of(x)
        if now is None:
            now = dt.datetime.now(dt.UTC)  # now expressed in UTC, == compares with tz
        return x_in_session and self.in_session(now) and x_date == self.session_date(now)

    def in_session(self, t: dt.datetime) -> bool:
        """ True if moment t (naive: read in the naive timezone) is inside a session,
            False in gaps (weekends, holidays) """
        return self.calendar.is_business_day(self._rolled_date(t))

    def to_session_date(self, x: dt.date | dt.datetime) -> dt.date:
        """ Trade date for x: a date (rolled back to a business day) or a moment (datetime, naive read
            in the naive timezone) """
        return self._session_of(x)[0]

    def previous_eod(self, x: dt.date | dt.datetime) -> dt.date:
        """ T-1 EOD: business day before the trade date of x """
        return self.calendar.adjust(self.to_session_date(x) - dt.timedelta(days=1), cdr.BDC.P)

    def _rolled_date(self, t: dt.datetime) -> dt.date:
        """ Local date of moment t, moved to the next day from roll time, before business-day adjustment """
        t = localize(t)
        local = t.astimezone(self.tz)
        d = local.date()
        if local.time() >= self.roll:
            d += dt.timedelta(days=1)
        return d

    def _unadjusted_date(self, x: dt.date | dt.datetime) -> dt.date:
        """ Date of x before business-day adjustment: x itself for a date, rolled date for a moment """
        x = normalize(x)
        return self._rolled_date(x) if isinstance(x, dt.datetime) else x

    def _session_of(self, x: dt.date | dt.datetime) -> tuple[dt.date, bool]:
        """ Trade date of x, and whether x is inside a session (False for dates and moments in gaps) """
        d = self._unadjusted_date(x)
        return self.calendar.adjust(d, cdr.BDC.P), self.calendar.is_business_day(d)


def center_name(name: str) -> str:
    """ Registry name for a center name or alias, case- and space-insensitive (e.g. 'ny', 'New York') """
    key = name.upper().replace(" ", "")
    key = CENTER_ALIASES.get(key, key)
    if key not in CENTER_SPECS:
        raise ValueError(f"Unknown trading center '{name}'. Available: {', '.join(CENTER_SPECS)} "
                         f"(aliases: {', '.join(CENTER_ALIASES)})")
    return key


def make_center(name: str) -> TradingCenter:
    """ Trading center from the CENTER_SPECS registry, by name or alias """
    return _make_center(center_name(name))


@cache
def _make_center(name: str) -> TradingCenter:
    """ Build a registry center (name must be a CENTER_SPECS key) """
    tz, calendar_name, roll = CENTER_SPECS[name]
    calendar = cdr.make_calendar(calendar_name) if calendar_name else cdr.Calendar(name)
    return TradingCenter(name, ZoneInfo(tz), calendar, roll)


def set_default_center(name: str|None) -> None:
    """ Set the process-wide default center (None: back to DEFAULT_CENTER) """
    global _default_center_name
    if name is not None:
        make_center(name)
    _default_center_name = name


def default_center() -> TradingCenter:
    """ Default center: set_default_center() if called, else DEFAULT_CENTER """
    return make_center(_default_center_name or DEFAULT_CENTER)


def get_center(center: str|TradingCenter|None=None) -> TradingCenter:
    """ Resolve a center given as a TradingCenter, a registry name, or None for the default center """
    if center is None:
        return default_center()
    if isinstance(center, str):
        return make_center(center)
    return center


def is_live(x: dt.date|dt.datetime, center: str|TradingCenter|None=None, now: dt.datetime|None=None) -> bool:
    """ True if x belongs to the current session of center (default center if None) """
    return get_center(center).is_live(x, now)


# name: (timezone, calendar name [None=weekends only], roll time)
CENTER_SPECS = {
    # Branches
    "SYDNEY": ("Australia/Sydney", "AUD", dt.time(16, 30)),
    "TOKYO": ("Asia/Tokyo", "JPY", dt.time(15)),
    "SINGAPORE": ("Asia/Singapore", "SGD", dt.time(17)),
    "AUCKLAND": ("Pacific/Auckland", "NZD", dt.time(17)),
    "MUMBAI": ("Asia/Kolkata", "INR", dt.time(17)),
    "DUBAI": ("Asia/Dubai", "AED", dt.time(16)),
    "LONDON": ("Europe/London", "GBP", dt.time(16, 30)),
    "NEWYORK": ("America/New_York", "USD", dt.time(17)),
    # Exchanges
    "XNYS": ("America/New_York", "XNYS", dt.time(17)),
    "XLON": ("Europe/London", "XLON", dt.time(16, 30)),
    "XTKS": ("Asia/Tokyo", "XTKS", dt.time(15)),
    "CME": ("America/Chicago", "CME_Equity", dt.time(17)),
    "FX": ("America/New_York", None, dt.time(17)),
}

CENTER_ALIASES = {
    "NY": "NEWYORK", "NYC": "NEWYORK",
    "LDN": "LONDON",
    "JP": "TOKYO", "TK": "TOKYO", "TKY": "TOKYO",
    "SG": "SINGAPORE", "SGP": "SINGAPORE",
    "AU": "SYDNEY", "SYD": "SYDNEY",
    "NZ": "AUCKLAND", "AKL": "AUCKLAND",
    "BOM": "MUMBAI",
    "DXB": "DUBAI",
}


def _is_live_request(date_input: dt.date | dt.datetime | str | None) -> bool:
    """ True for None or 'live'/'now' (case-insensitive) """
    return date_input is None or (isinstance(date_input, str) and date_input.lower() in ('live', 'now'))


def resolve_date_input(date_input: dt.date|dt.datetime|str|None, center: str|TradingCenter|None=None,
                       now: dt.datetime|None=None) -> tuple[dt.date, bool]:
    """ Resolve a date input given as None, 'live' or 'now' (current session), or a date/datetime.
        Args:
            - now: defaults to the system clock, only useful for testing/replicating results
        Returns its trade date for center (default center if None) and whether it is live. """
    c = get_center(center)
    if now is None:
        now = dt.datetime.now(dt.UTC)

    if _is_live_request(date_input):
        return c.session_date(now), c.in_session(now)

    if isinstance(date_input, str):
        raise ValueError(f"Invalid end date '{date_input}': expected None, 'live', 'now', a date or a datetime")

    return c.to_session_date(date_input), c.is_live(date_input, now)


def previous_eod(date_input: dt.date|dt.datetime|str|None=None, center: str|TradingCenter|None=None,
                 now: dt.datetime|None=None) -> dt.date:
    """ T-1 EOD for a date input: None, 'live' or 'now' (current moment), or a date/datetime """
    c = get_center(center)
    if _is_live_request(date_input):
        return c.previous_eod(dt.datetime.now(dt.UTC) if now is None else now)
    if isinstance(date_input, str):
        raise ValueError(f"Invalid date input '{date_input}': expected None, 'live', 'now', a date or a datetime")
    return c.previous_eod(date_input)


def get_date_info(date_input: dt.date|dt.datetime|str|None, center: str|TradingCenter|None=None,
                  now: dt.datetime|None=None) -> dict:
    """ Get live status, session, T-1 EOD etc. """
    session_date, is_live = resolve_date_input(date_input, center, now)
    tm1eod = previous_eod(session_date, center, now)
    return {'session': session_date, 'islive': is_live, 'tm1eod': tm1eod}



if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    from tzlocal import get_localzone_name, get_localzone
    tz_name = get_localzone_name()   # 'Europe/Paris'
    tz = get_localzone()        # ZoneInfo('Europe/Paris')
    print(tz_name)
    print(tz)

    # Week test: 'live' every 15 minutes over a week, on a simulated clock
    gen_center = "SG"
    gen_center_tz = make_center(gen_center).tz
    start = dt.datetime(2026, 11, 5, 0, 0, tzinfo=gen_center_tz)  # Sat before Thanksgiving week
    nows = pd.date_range(start, periods=10 * 24 * 4, freq="15min")

    test_center = "TK"
    rows = []
    for now in nows:
        info = get_date_info('live', test_center, now.to_pydatetime())
        rows.append({'Now': now, **info})

    df = pd.DataFrame(rows)
    df['Now'] = df['Now'].dt.tz_localize(None)  # Formattable in Excel
    file = Path.home() / "temp" / "sessions" / (test_center + ".tsv")
    df.to_csv(file, sep="\t")
    # pd.set_option('display.max_rows', None)
    # print(df)
