import datetime as dt
import re
from dateutil.relativedelta import relativedelta
from openpyxl.utils.datetime import to_excel
from sdevpy.utilities.tools import isiterable


DATE_FORMAT = '%d-%b-%Y'
DATETIME_FORMAT = '%d-%b-%Y %H:%M:%S'
DATE_FILE_FORMAT = '%Y%m%d-%H%M%S'
_ANCHOR = dt.datetime(1960, 1, 1) # Arbitrary but fixed (only relative ordering matters)


def advance_int(base_date: dt.datetime, days: int=0, months: int=0, years: int=0) -> dt.datetime:
    """ Advance base_date by days, months and years, no calendars or conventions """
    return base_date + relativedelta(days=days, months=months, years=years)


def advance(base_date: dt.datetime, tenor_str: str) -> dt.datetime:
    """ Advance base_date by tenor, no calendars or conventions """
    return base_date + period(tenor_str)


def to_oadate(date: dt.datetime) -> int:
    """ Convert datetime to OA date (Excel) """
    if isiterable(date):
        oadate = [to_excel(d) for d in date]
    else:
        oadate = to_excel(date)
    return oadate


def period(tenor_str: str) -> relativedelta:
    """ Convert from string to Period (relativedelta) """
    pattern = r'([+-]?)(\d+Y)?(\d+M)?(\d+W)?(\d+D)?'
    match = re.fullmatch(pattern, tenor_str.upper())
    if not match or not any(match.groups()[1:]):
        raise ValueError(f"Invalid tenor string: '{tenor_str}'")

    sign   = -1 if match.group(1) == '-' else 1
    years  = sign * (int(match.group(2)[:-1]) if match.group(2) else 0)
    months = sign * (int(match.group(3)[:-1]) if match.group(3) else 0)
    weeks  = sign * (int(match.group(4)[:-1]) if match.group(4) else 0)
    days   = sign * (int(match.group(5)[:-1]) if match.group(5) else 0)

    return relativedelta(years=years, months=months, weeks=weeks, days=days)


def tenor_to_date(tenor_str: str, anchor: dt.datetime=_ANCHOR) -> dt.datetime:
    """ Map a tenor to a date under a fixed anchor, purely for ordering -- not a real
        settlement or valuation date. ON/TN/SN are shorter than any Y/M/W/D tenor by market
        convention (not by a fixed day count -- their real length depends on the trading
        calendar and weekends), so they map to the anchor itself rather than being parsed. """
    if tenor_str.upper() in ('ON', 'TN', 'SN'):
        return anchor
    return anchor + period(tenor_str)


def tenor_leq(t1: str, t2: str, anchor: dt.datetime=_ANCHOR) -> bool:
    """ True if t1 falls at or before t2, per a fixed, arbitrary anchor date. """
    return tenor_to_date(t1, anchor) <= tenor_to_date(t2, anchor)


if __name__ == "__main__":
    print("Hello")
    d = dt.datetime(2026, 2, 15, 10, 50, 7)
    print(d.strftime(DATE_FILE_FORMAT))

    d = [dt.datetime(2026, 3, 8), dt.datetime(2026, 3, 8)]
    print(to_oadate(d))
