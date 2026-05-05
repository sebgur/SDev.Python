import datetime as dt
import re
from dateutil.relativedelta import relativedelta
from openpyxl.utils.datetime import to_excel
from sdevpy.utilities.tools import isiterable


DATE_FORMAT = '%d-%b-%Y'
DATETIME_FORMAT = '%d-%b-%Y %H:%M:%S'
DATE_FILE_FORMAT = '%Y%m%d-%H%M%S'


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


if __name__ == "__main__":
    print("Hello")
    d = dt.datetime(2026, 2, 15, 10, 50, 7)
    print(d.strftime(DATE_FILE_FORMAT))

    d = [dt.datetime(2026, 3, 8), dt.datetime(2026, 3, 8)]
    print(to_oadate(d))
