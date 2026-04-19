import datetime as dt
from dateutil.relativedelta import relativedelta
from openpyxl.utils.datetime import to_excel
from sdevpy.utilities.utils import isiterable


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


if __name__ == "__main__":
    print("Hello")
    d = dt.datetime(2026, 2, 15, 10, 50, 7)
    print(d.strftime(DATE_FILE_FORMAT))

    d = [dt.datetime(2026, 3, 8), dt.datetime(2026, 3, 8)]
    print(to_oadate(d))
