import datetime as dt
import re
from dateutil.relativedelta import relativedelta
from openpyxl.utils.datetime import to_excel
from sdevpy.utilities.tools import isiterable


DATE_FORMAT = '%d-%b-%Y'
DATETIME_FORMAT = '%d-%b-%Y %H:%M:%S'
DATE_FILE_FORMAT = '%Y%m%d-%H%M%S'


def advance(base_date, days=0, months=0, years=0):
    """ Advancing by days, months and years with no calendar or convention considerations """
    return base_date + relativedelta(days=days, months=months, years=years)


def to_oadate(date):
    if isiterable(date):
        oadate = [to_excel(d) for d in date]
    else:
        oadate = to_excel(date)
    return oadate


def period(tenor_str: str) -> relativedelta:
    """ Conversion from string to Period (relativedelta) """
    pattern = r'(\d+Y)?(\d+M)?(\d+W)?(\d+D)?'
    match = re.fullmatch(pattern, tenor_str.upper())
    if not match or not any(match.groups()):
        raise ValueError(f"Invalid tenor string: '{tenor_str}'")

    years   = int(match.group(1)[:-1]) if match.group(1) else 0
    months  = int(match.group(2)[:-1]) if match.group(2) else 0
    weeks   = int(match.group(3)[:-1]) if match.group(3) else 0
    days    = int(match.group(4)[:-1]) if match.group(4) else 0

    return relativedelta(years=years, months=months, weeks=weeks, days=days)


if __name__ == "__main__":
    print("Hello")
    d = dt.datetime(2026, 2, 15, 10, 50, 7)
    print(d.strftime(DATE_FILE_FORMAT))

    d = [dt.datetime(2026, 3, 8), dt.datetime(2026, 3, 8)]
    print(to_oadate(d))
