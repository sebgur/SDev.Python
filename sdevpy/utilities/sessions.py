import datetime as dt
import os
from dataclasses import dataclass
from functools import cache
from zoneinfo import ZoneInfo
from sdevpy.utilities import calendars as cdr


_default_center_name: str | None = None


def normalize(x: dt.date|dt.datetime) -> dt.date|dt.datetime:
    """ Classify a date/datetime input as either a trade-date label or an instant.
        - dt.date: trade-date label, returned unchanged
        - aware dt.datetime: instant, returned unchanged
        - naive dt.datetime at midnight: trade-date label, returned as dt.date
        - naive dt.datetime at any other time: ValueError (ambiguous instant)
        Callers distinguish the result with isinstance(r, dt.datetime). """
    if isinstance(x, dt.datetime):
        if x.utcoffset() is not None:
            return x
        if x.time() != dt.time(0):
            raise ValueError(f"Naive datetime with non-midnight time is ambiguous: {x}. "
                             "Pass an aware datetime or a date.")
        return x.date()
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
        """ Trade date of the session that moment t (aware datetime) belongs to """
        return self.calendar.adjust(self._rolled_date(t), cdr.BDC.P)

    def in_session(self, t: dt.datetime) -> bool:
        """ True if moment t (aware datetime) is inside a session, False in gaps (weekends, holidays) """
        return self.calendar.is_business_day(self._rolled_date(t))

    def _rolled_date(self, t: dt.datetime) -> dt.date:
        """ Local date of moment t, moved to the next day from roll time, before business-day adjustment """
        if t.utcoffset() is None:
            raise ValueError(f"Aware datetime required, got naive {t}")
        local = t.astimezone(self.tz)
        d = local.date()
        if local.time() >= self.roll:
            d += dt.timedelta(days=1)
        return d

    def to_session_date(self, x: dt.date | dt.datetime) -> dt.date:
        """ Trade date for x: a date (rolled back to a business day) or a moment (aware datetime) """
        return self._session_of(x)[0]

    def _session_of(self, x: dt.date | dt.datetime) -> tuple[dt.date, bool]:
        """ Trade date of x, and whether x is inside a session (False for dates and moments in gaps) """
        x = normalize(x)
        d = self._rolled_date(x) if isinstance(x, dt.datetime) else x
        return self.calendar.adjust(d, cdr.BDC.P), self.calendar.is_business_day(d)

    def is_live(self, x: dt.date | dt.datetime, now: dt.datetime | None = None) -> bool:
        """ True if x belongs to the current session. Nothing is live in gaps (weekends, holidays):
            neither x nor now may fall in a gap. now defaults to the system clock """
        x_date, x_in_session = self._session_of(x)
        if now is None:
            now = dt.datetime.now(dt.UTC)
        return x_in_session and self.in_session(now) and x_date == self.session_date(now)


@cache
def make_center(name: str) -> TradingCenter:
    """ Trading center from the CENTER_SPECS registry """
    if name not in CENTER_SPECS:
        raise ValueError(f"Unknown trading center '{name}'. Available: {', '.join(CENTER_SPECS)}")
    tz, calendar_name, roll = CENTER_SPECS[name]
    calendar = cdr.make_calendar(calendar_name) if calendar_name else cdr.Calendar(name)
    return TradingCenter(name, ZoneInfo(tz), calendar, roll)


def set_default_center(name: str | None) -> None:
    """ Set the process-wide default center (None: fall back to the SDEVPY_CENTER environment variable) """
    global _default_center_name
    if name is not None:
        make_center(name)  # Validate
    _default_center_name = name


def default_center() -> TradingCenter:
    """ Default center: set_default_center() if called, else the SDEVPY_CENTER environment variable """
    name = _default_center_name or os.environ.get("SDEVPY_CENTER")
    if name is None:
        raise ValueError("No default trading center: call set_default_center() or set SDEVPY_CENTER")
    return make_center(name)


def get_center(center: str | TradingCenter | None = None) -> TradingCenter:
    """ Resolve a center given as a TradingCenter, a registry name, or None for the default center """
    if center is None:
        return default_center()
    if isinstance(center, str):
        return make_center(center)
    return center


def is_live(x: dt.date | dt.datetime, center: str | TradingCenter | None = None,
            now: dt.datetime | None = None) -> bool:
    """ True if x belongs to the current session of center (default center if None) """
    return get_center(center).is_live(x, now)


# name: (timezone, calendar name for cdr.make_calendar or None for weekends only, roll time)
# Branch rolls are the company EOD times, local to each branch
CENTER_SPECS = {
    # Branches
    "SYDNEY": ("Australia/Sydney", "AUD", dt.time(17)),
    "TOKYO": ("Asia/Tokyo", "JPY", dt.time(17)),
    "SINGAPORE": ("Asia/Singapore", "SGD", dt.time(17)),
    "AUCKLAND": ("Pacific/Auckland", "NZD", dt.time(17)),
    "MUMBAI": ("Asia/Kolkata", "INR", dt.time(17)),
    "DUBAI": ("Asia/Dubai", "AED", dt.time(17)),
    "LONDON": ("Europe/London", "GBP", dt.time(17)),
    "NEWYORK": ("America/New_York", "USD", dt.time(17)),
    # Exchanges
    "XNYS": ("America/New_York", "XNYS", dt.time(17)),
    "XLON": ("Europe/London", "XLON", dt.time(17)),
    "XTKS": ("Asia/Tokyo", "XTKS", dt.time(17)),
    "CME": ("America/Chicago", "CME_Equity", dt.time(17)),
    "FX": ("America/New_York", None, dt.time(17)),
}


def resolve_end_date(end_date: dt.date | dt.datetime | str | None, center: str | TradingCenter | None = None,
                     now: dt.datetime | None = None) -> tuple[dt.date, bool]:
    """ Resolve an end date given as None or 'live' (current session), or a date/datetime.
        Returns its trade date for center (default center if None) and whether it is live. """
    c = get_center(center)
    if now is None:
        now = dt.datetime.now(dt.UTC)
    if end_date is None or end_date == 'live':
        return c.session_date(now), c.in_session(now)
    if isinstance(end_date, str):
        raise ValueError(f"Invalid end date '{end_date}': expected None, 'live', a date or a datetime")
    return c.to_session_date(end_date), c.is_live(end_date, now)
