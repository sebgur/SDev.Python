import datetime as dt
from dataclasses import dataclass
from functools import cache
from zoneinfo import ZoneInfo
from sdevpy.utilities import calendars as cdr


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
        if t.utcoffset() is None:
            raise ValueError(f"session_date requires an aware datetime, got naive {t}")
        local = t.astimezone(self.tz)
        d = local.date()
        if local.time() >= self.roll:
            d += dt.timedelta(days=1)
        return self.calendar.adjust(d, cdr.BDC.P)

    def to_session_date(self, x: dt.date | dt.datetime) -> dt.date:
        """ Trade date for x: a date (rolled back to a business day) or a moment (aware datetime) """
        x = normalize(x)
        if isinstance(x, dt.datetime):
            return self.session_date(x)
        return self.calendar.adjust(x, cdr.BDC.P)

    def is_live(self, x: dt.date | dt.datetime, now: dt.datetime | None = None) -> bool:
        """ True if x belongs to the current session. now defaults to the system clock """
        if now is None:
            now = dt.datetime.now(dt.UTC)
        return self.to_session_date(x) == self.session_date(now)

@cache
def make_center(name: str) -> TradingCenter:
    """ Trading center from the CENTER_SPECS registry """
    if name not in CENTER_SPECS:
        raise ValueError(f"Unknown trading center '{name}'. Available: {', '.join(CENTER_SPECS)}")
    tz, calendar_name, roll = CENTER_SPECS[name]
    calendar = cdr.make_calendar(calendar_name) if calendar_name else cdr.Calendar(name)
    return TradingCenter(name, ZoneInfo(tz), calendar, roll)


# name: (timezone, calendar name for cdr.make_calendar or None for weekends only, roll time)
CENTER_SPECS = {
    "XNYS": ("America/New_York", "XNYS", dt.time(17)),
    "XLON": ("Europe/London", "XLON", dt.time(17)),
    "XTKS": ("Asia/Tokyo", "XTKS", dt.time(17)),
    "CME": ("America/Chicago", "CME_Equity", dt.time(17)),
    "FX": ("America/New_York", None, dt.time(17)),
}
